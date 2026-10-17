"""Audit 2026-10-17: silent fallbacks must not report values indistinguishable
from real results. A public parameter changes the realized object or is
refused; an unknown is reported as unknown, never as 0 or as absent.

Each test below fails on the baseline (restored implementation file) and
passes with the fix.
"""
import dataclasses
import types

import pytest

import jaxfne as jtfne


def _small_model():
    cfg = jtfne.default_cortical_column_config(n=10, duration_ms=10, dt_ms=0.5)
    return jtfne.construct(cfg)


# --- site 1: connect() recurrent_backend override ---------------------------
def _fake_models(n=2):
    return tuple(
        types.SimpleNamespace(cfg=jtfne.Configuration()) for _ in range(n)
    )


def _merge(models):
    from jaxfne._construct_connectivity import _connect_merge_cfg

    return _connect_merge_cfg(
        models, None, "test", "vertical", [10] * len(models),
        10 * len(models), None, [0] * len(models),
    )


def test_recurrent_backend_override_applied():
    merged = _merge(_fake_models())
    assert merged.metadata["recurrent_backend_override"] == "applied"
    assert merged.metadata["recurrent_backend"] == "edge_list"


def test_recurrent_backend_override_unsupported_records_type(monkeypatch):
    def _boom(self, **kwargs):
        raise ValueError("forced: no consumer")

    monkeypatch.setattr(jtfne.Configuration, "runtime", _boom)
    merged = _merge(_fake_models())
    assert merged.metadata["recurrent_backend_override"] == "not_supported: ValueError"


def test_recurrent_backend_override_unexpected_propagates(monkeypatch):
    def _boom(self, **kwargs):
        raise RuntimeError("forced defect")

    monkeypatch.setattr(jtfne.Configuration, "runtime", _boom)
    with pytest.raises(RuntimeError, match="forced defect"):
        _merge(_fake_models())


# --- site 2a: history delay ---------------------------------------------------
def test_history_delay_failure_is_unknown_not_zero():
    model = _small_model()

    class _BoomArray:
        def __array__(self, dtype=None, copy=None):
            raise ValueError("forced delay readout failure")

    object.__setattr__(model.params["edge_list"], "delay_steps", _BoomArray())
    summ = jtfne.canonical_compact_summary(model)
    hs = summ["taxonomy"]["history_state"]
    assert hs["delay_max_steps"] is None
    assert hs["delay_buffer_Bt"] is None
    assert hs["total_buffer_elements"] is None
    assert hs["unknown_reason"] == "unknown: ValueError: forced delay readout failure"


# --- site 2b: trainable count -------------------------------------------------
def test_n_trainable_failure_is_unknown_not_zero():
    model = _small_model()

    class _BoomInt:
        def __bool__(self):
            return True

        def __int__(self):
            raise ValueError("forced trainable readout failure")

    md = dict(model.cfg.metadata)
    md["optimizer"] = {"n_trainable": _BoomInt()}
    object.__setattr__(model.cfg, "metadata", md)
    summ = jtfne.canonical_compact_summary(model)
    tp = summ["taxonomy"]["trainable_parameters"]
    assert tp["n_trainable"] is None
    assert tp["n_free"] is None
    assert tp["unknown_reason"] == "unknown: ValueError: forced trainable readout failure"


# --- site 3: executed delay ---------------------------------------------------
def test_executed_delay_failure_is_unknown_not_absent(monkeypatch):
    import jax.numpy as jnp

    model = _small_model()
    edge_list = model.params["edge_list"]
    steps = jnp.zeros(edge_list.n_edges, dtype=jnp.int32).at[0].set(2)
    object.__setattr__(edge_list, "delay_storage", "per_edge")
    object.__setattr__(edge_list, "delay_steps", steps)
    assert model.manifest()["executed_delay"]["max_steps"] == 2

    def _boom(edges):
        raise ValueError("forced executed-delay failure")

    monkeypatch.setattr("jaxfne.emitters.resolve_edge_delay_steps", _boom)
    assert model.manifest()["executed_delay"] == {
        "unknown": "ValueError: forced executed-delay failure"
    }


# --- site 4: static_events repr fallback --------------------------------------
def test_static_events_repr_fallback_flagged():
    from jaxfne.paradigm import general_sequential_oddball_paradigm

    class _BadStr:
        def __str__(self):
            raise ValueError("forced static-events failure")

        def __repr__(self):
            return "<BadStr-repr>"

    nested = {"nested": _BadStr()}
    paradigm = general_sequential_oddball_paradigm(
        event_windows={"fix": (0.0, 500.0)},
        sequence_event_labels=["fix"],
        conditions=[{"name": "c", "sequence": ["fix"]}],
        static_events={"weird": nested},
    )
    assert paradigm.metadata["static_events"] == {"weird": repr(nested)}
    assert paradigm.metadata["static_events_repr_fallback"] is True


# --- site 5: cell_types zero-column network -----------------------------------
def test_cell_types_zero_column_sum_refused():
    cfg = jtfne.Configuration().column(name="V1", layers=["L2/3"], n=4)
    zero_cols = [dict(c, n=0) for c in cfg.metadata["columns"]]
    bad = dataclasses.replace(
        cfg, networks=[], metadata={**cfg.metadata, "columns": zero_cols}
    )
    with pytest.raises(ValueError, match="summing to 0"):
        bad.cell_types({"E": 1.0})


def test_cell_types_no_columns_default_kept():
    cfg = jtfne.Configuration().cell_types({"E": 1.0})
    assert cfg.networks[0]["n"] == 100
