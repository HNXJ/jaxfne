"""An explicit backend request is realized or refused -- never reinterpreted.

W11 measured 16 cells that asked for ``recurrent_backend="dense"`` and silently
ran on ``edge_list``. The override is necessary -- the dense kernel runs on
``emitter.W``, which does not carry rule-materialized edges, so a dense run
would drop them -- but ``cfg.metadata`` distinguishes an explicit request from
an unset default, so a contradicted request is distinguishable from an absent
one and must not be answered by quietly substituting a different backend.

Contract, per surface (construct, connect, simulate / simulate_batch):

* backend unset            -> resolved automatically, no error
* backend explicit, compatible   -> honoured
* backend explicit, contradicted -> ValueError naming requested backend and cause
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import jaxfne as jtfne
from jaxfne.core import _SPARSE_DIRECT_N

N = 100


def _cfg(*, with_rule: bool, name: str = "c"):
    cfg = (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float32", duration_ms=10.0, dt_ms=0.5)
        .column(name=name, layers=["L4"], n=N)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
    )
    if with_rule:
        cfg = cfg.connectivity(p_connect=0.0)
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
        )
    )
    if with_rule:
        cfg = cfg.mechanisms(
            name="ampa",
            kind="exponential",
            params={"tau_ms": 2.0, "receptor": "AMPA"},
        ).connections(
            name="rec",
            source={},
            target={},
            mechanism="ampa",
            weight=0.03,
            max_in_degree=25,
            spatial_sigma=0.1,
        )
    return cfg


def _sim():
    return jtfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=1)


def _realized_backend(model):
    return model.cfg.metadata.get("recurrent_backend")


def _force_backend_metadata(model, backend: str):
    """Inject contradictory metadata on a constructed model for simulate guards."""
    md = dict(model.cfg.metadata)
    md["recurrent_backend"] = backend
    object.__setattr__(model, "cfg", replace(model.cfg, metadata=md))


def _sparse_direct_cfg():
    return (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float32", duration_ms=10.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=_SPARSE_DIRECT_N)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(p_connect=0.02)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
        )
    )


def _sparse_direct_model(backend: str | None = None):
    cfg = _with_backend(_sparse_direct_cfg(), backend)
    model = jtfne.construct(cfg)
    assert model.params["emitter"].W.shape == (0, 0)
    return model


def _with_backend(cfg, backend: str | None):
    if backend is None:
        return cfg
    return cfg.runtime(recurrent_backend=backend)


def _assert_refusal(excinfo, *, requested: str, cause_fragment: str):
    message = str(excinfo.value)
    assert f"recurrent_backend='{requested}'" in message or (
        f"recurrent_backend={requested!r}" in message
    )
    assert cause_fragment in message
    assert "drop the explicit recurrent_backend" in message.lower()
    assert "edge_list" in message


# --- construct() -------------------------------------------------------------


class TestConstructBackendRealization:
    def test_unset_resolves_to_edge_list_when_rules_materialize(self):
        model = jtfne.construct(_cfg(with_rule=True))
        assert _realized_backend(model) == "edge_list"

    def test_unset_stays_unset_without_rules(self):
        model = jtfne.construct(_cfg(with_rule=False))
        assert _realized_backend(model) is None

    def test_explicit_edge_list_honoured_when_rules_materialize(self):
        model = jtfne.construct(
            _with_backend(_cfg(with_rule=True), "edge_list")
        )
        assert _realized_backend(model) == "edge_list"

    def test_explicit_dense_honoured_without_contradiction(self):
        model = jtfne.construct(
            _with_backend(_cfg(with_rule=False), "dense")
        )
        assert _realized_backend(model) == "dense"

    def test_explicit_dense_refused_when_rules_materialize(self):
        with pytest.raises(ValueError) as excinfo:
            jtfne.construct(_with_backend(_cfg(with_rule=True), "dense"))
        _assert_refusal(
            excinfo,
            requested="dense",
            cause_fragment="edge(s) were materialized",
        )

    def test_explicit_dense_refused_on_sparse_direct_path(self):
        with pytest.raises(ValueError) as excinfo:
            jtfne.construct(_with_backend(_sparse_direct_cfg(), "dense"))
        _assert_refusal(
            excinfo,
            requested="dense",
            cause_fragment="sparse-direct path",
        )


# --- connect() ---------------------------------------------------------------


class TestConnectBackendRealization:
    def _pair(self, backend: str | None):
        a = jtfne.construct(
            _with_backend(_cfg(with_rule=False, name="a"), backend)
        )
        b = jtfne.construct(
            _with_backend(_cfg(with_rule=False, name="b"), backend)
        )
        return a, b

    def test_unset_resolves_to_edge_list(self):
        merged = jtfne.connect(*self._pair(None))
        assert _realized_backend(merged) == "edge_list"

    def test_explicit_edge_list_honoured(self):
        merged = jtfne.connect(*self._pair("edge_list"))
        assert _realized_backend(merged) == "edge_list"

    def test_explicit_dense_refused(self):
        a, b = self._pair("dense")
        with pytest.raises(ValueError) as excinfo:
            jtfne.connect(a, b)
        _assert_refusal(
            excinfo,
            requested="dense",
            cause_fragment="cannot be realized by connect()",
        )

# --- simulate() / simulate_batch() -------------------------------------------


class TestSimulateBackendRealization:
    def _rule_model(self, backend: str | None):
        return jtfne.construct(_with_backend(_cfg(with_rule=True), backend))

    def _dense_model(self):
        return jtfne.construct(_with_backend(_cfg(with_rule=False), "dense"))

    def test_unset_sparse_direct_simulates_on_edge_list(self):
        model = _sparse_direct_model(None)
        signals = model.simulate(_sim())
        assert signals.spikes.shape[1] == _SPARSE_DIRECT_N

    def test_explicit_edge_list_sparse_direct_simulates(self):
        model = _sparse_direct_model("edge_list")
        signals = model.simulate(_sim())
        assert signals.spikes.shape[1] == _SPARSE_DIRECT_N

    def test_explicit_dense_without_contradiction_simulates(self):
        model = self._dense_model()
        signals = model.simulate(_sim())
        assert signals.spikes.shape[1] == N

    def test_simulate_refuses_contradicted_dense_metadata(self):
        """Defense in depth: simulate guards placeholder-W models explicitly."""
        model = _sparse_direct_model("edge_list")
        _force_backend_metadata(model, "dense")
        with pytest.raises(ValueError) as excinfo:
            model.simulate(_sim())
        _assert_refusal(
            excinfo,
            requested="dense",
            cause_fragment="placeholder dense W",
        )

    def test_simulate_batch_refuses_contradicted_dense_metadata(self):
        model = _sparse_direct_model("edge_list")
        _force_backend_metadata(model, "dense")
        with pytest.raises(ValueError) as excinfo:
            model.simulate_batch(_sim(), n_seeds=2, seed=1)
        _assert_refusal(
            excinfo,
            requested="dense",
            cause_fragment="placeholder dense W",
        )

    def test_simulate_batch_honours_compatible_dense(self):
        model = self._dense_model()
        result = model.simulate_batch(_sim(), n_seeds=2, seed=1)
        assert result["metadata"]["n_seeds"] == 2
