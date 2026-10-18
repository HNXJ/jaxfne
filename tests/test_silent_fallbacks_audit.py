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


# --- T-31 rest: narrowed silent fallbacks (unexpected errors propagate) ----------
# Each test forces an unexpected error type through a previously bare
# `except Exception` site. On the baseline the error is swallowed (the test
# fails); with the narrowed handler it propagates. Expected-type failures
# (AttributeError/TypeError/ValueError) still take the silent fallback.
import numpy as np


class _BoomContains:
    def __contains__(self, key):
        raise RuntimeError("forced metadata containment failure")

    def get(self, key, default=None):
        raise AssertionError("unreachable")


class _BoomGet:
    def __contains__(self, key):
        return False

    def get(self, key, default=None):
        raise RuntimeError("forced metadata get failure")


class _BoomArrayRt:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("forced readout failure")


def test_hdp_params_probe_unexpected_propagates():
    model = _small_model()
    object.__setattr__(model.cfg, "metadata", _BoomContains())
    with pytest.raises(RuntimeError, match="forced metadata containment"):
        jtfne.canonical_compact_summary(model)


def test_enable_hdp_lookup_unexpected_propagates():
    model = _small_model()
    object.__setattr__(model.cfg, "metadata", _BoomGet())
    with pytest.raises(RuntimeError, match="forced metadata get"):
        jtfne.canonical_compact_summary(model)


def test_hdp_initial_H_unexpected_propagates():
    model = _small_model()
    model.params["hdp_initial_H"] = _BoomArrayRt()
    with pytest.raises(RuntimeError, match="forced readout"):
        jtfne.canonical_compact_summary(model)


def test_tensor_H_schema_unexpected_propagates():
    model = _small_model()

    class _BoomAreas:
        def __iter__(self):
            raise RuntimeError("forced schema iteration failure")

    tensor = types.SimpleNamespace(areas=_BoomAreas(), area_connections=[])
    with pytest.raises(RuntimeError, match="forced schema iteration"):
        jtfne.canonical_compact_summary(model, tensor=tensor)


def test_signals_hdp_meta_unexpected_propagates():
    model = _small_model()

    class _BoomDict(dict):
        def get(self, key, default=None):
            raise RuntimeError("forced hdp-meta get failure")

    boom = _BoomDict(enabled=False)
    signals = types.SimpleNamespace(metadata={"hdp": boom}, sources=None, field=None)
    with pytest.raises(RuntimeError, match="forced hdp-meta get"):
        jtfne.canonical_compact_summary(model, signals=signals)


def test_last_hdp_diag_unexpected_propagates():
    model = _small_model()
    object.__setattr__(model, "_last_hdp_diag", {"H_trace": _BoomArrayRt()})
    with pytest.raises(RuntimeError, match="forced readout"):
        jtfne.canonical_compact_summary(model)


def test_ensemble_metadata_unexpected_propagates():
    from jaxfne._construct_connectivity import ensemble_edge_ownership

    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(metadata=_BoomGet()), params={}
    )
    with pytest.raises(RuntimeError, match="forced metadata get"):
        ensemble_edge_ownership(model)


# --- T-31 rest: agent diagnostics preserve input dtype ---------------------------
_AB = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"


def _tfne_spec(weight=0.5, delay=4.0):
    return (
        "O[k] := [direction = >; mechanism = AMPA; probability = 1.0; "
        f"weight = {weight}; delay = {delay}];\n{_AB}\nx : A O[k] B : y\n"
    )


def test_agent_diagnostics_preserve_float32():
    from jaxfne import agent as A

    run = A.simulate(A.realize(_tfne_spec(), seed=11), duration_ms=30.0, dt_ms=1.0, seed=0)
    sig_dtype = np.asarray(run.signals.V_m).dtype
    assert sig_dtype == np.dtype(np.float32)
    for q in ("V_m", "spikes"):
        assert A.observe(run, q).values.dtype == sig_dtype
    cmp = A.compare(run)
    assert {k: v["verdict"] for k, v in cmp.items()} == {
        "time": "EQUAL", "weight": "EQUAL", "mechanism": "EQUAL", "delay": "EQUAL"}
    assert A.verify(run, "finite")["verdict"] == "PASS"
