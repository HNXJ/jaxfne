"""T-33: gpu/tpu with no such JAX device fail fast with ``RuntimeError``.

``simulate()`` and ``construct()`` refuse an absent gpu/tpu backend, naming
the backend and the available platforms, instead of silently downgrading.
``runtime_report()`` still reports without raising, and auto/cpu are unchanged.
"""

from __future__ import annotations

import pytest

import jaxfne as jtfne
from jaxfne import RuntimeConfig
from jaxfne._runtime_config import _resolve_backend_device
from jaxfne.neuronal_tensor import RuntimeConfiguration


def _absent(backend: str) -> bool:
    return _resolve_backend_device(backend) is None


needs_no_gpu = pytest.mark.skipif(not _absent("gpu"), reason="a gpu JAX device is present")
needs_no_tpu = pytest.mark.skipif(not _absent("tpu"), reason="a tpu JAX device is present")


def _cfg():
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=6, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


@needs_no_gpu
def test_simulate_refuses_absent_gpu():
    model = jtfne.construct(_cfg())
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0,
                           runtime=RuntimeConfig(backend="gpu"))
    with pytest.raises(RuntimeError, match="platforms="):
        model.simulate(sim)


@needs_no_tpu
def test_simulate_refuses_absent_tpu():
    model = jtfne.construct(_cfg())
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0,
                           runtime=RuntimeConfig(backend="tpu"))
    with pytest.raises(RuntimeError, match="backend='tpu'"):
        model.simulate(sim)


@needs_no_gpu
def test_construct_configuration_refuses_absent_gpu():
    cfg = _cfg().runtime(backend="gpu")
    with pytest.raises(RuntimeError, match="no such JAX device"):
        jtfne.construct(cfg)


@needs_no_gpu
def test_construct_tensor_refuses_absent_gpu():
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    runtime = RuntimeConfiguration(seed=1, duration_ms=20.0, dt_ms=1.0, device="gpu")
    with pytest.raises(RuntimeError, match="no such JAX device"):
        jtfne.construct(tensor, runtime)


@needs_no_gpu
def test_simulate_batch_refuses_absent_gpu():
    model = jtfne.construct(_cfg())
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0,
                           runtime=RuntimeConfig(backend="gpu"))
    with pytest.raises(RuntimeError, match="no such JAX device"):
        model.simulate_batch(sim, n_seeds=2)


def test_runtime_report_still_reports_without_raising():
    report = RuntimeConfig(backend="gpu").runtime_report()
    assert report["requested_backend"] == "gpu"
    if _absent("gpu"):
        assert report["backend_enforced"] is False
        assert report["backend_warning"] is not None


def test_auto_and_cpu_backends_are_unchanged():
    model = jtfne.construct(_cfg())
    for backend in ("auto", "cpu"):
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0,
                               runtime=RuntimeConfig(backend=backend))
        assert model.simulate(sim).V_m.shape[1] == 6
