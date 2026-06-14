"""v0.3.40: RuntimeConfig.backend is honored for device placement, honestly.

simulate() pins execution to the requested backend device via jax.default_device
when that device exists; runtime_report() reflects enforcement by device
availability (no false placement claim on absent devices). On a single-device
host these are no-ops, so behavior is unchanged.
"""

import jax
import numpy as np

import jaxfne as jtfne
from jaxfne import RuntimeConfig
from jaxfne.core import _resolve_backend_device


def test_auto_backend_is_enforced_and_no_device_pin():
    assert _resolve_backend_device("auto") is None
    assert _resolve_backend_device(None) is None
    rep = RuntimeConfig(backend="auto").runtime_report()
    assert rep["backend_enforced"] is True


def test_cpu_backend_resolves_and_is_enforced():
    # A CPU device always exists, so cpu must resolve and report enforced.
    assert _resolve_backend_device("cpu") is not None
    rep = RuntimeConfig(backend="cpu").runtime_report()
    assert rep["actual_backend"] == "cpu"
    assert rep["backend_enforced"] is True
    assert rep["backend_warning"] is None


def test_unavailable_backend_is_honest_downgrade_not_false_claim():
    # On a CPU-only host, gpu must NOT be falsely reported as enforced.
    if _resolve_backend_device("gpu") is None:
        rep = RuntimeConfig(backend="gpu").runtime_report()
        assert rep["backend_enforced"] is False
        assert rep["actual_backend"] != "gpu"
        assert "requested_backend_unavailable" in (rep["backend_warning"] or "")


def test_simulate_runs_under_explicit_cpu_backend():
    cfg = jtfne.configuration().network(n=8).emitter().field().probe(n_contacts=4)
    model = jtfne.construct(cfg)
    rc = RuntimeConfig(backend="cpu", jit=True, dtype="float32")
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=rc)
    vm = np.asarray(model.simulate(sim).V_m)
    assert vm.dtype == np.float32
    assert bool(np.all(np.isfinite(vm)))
