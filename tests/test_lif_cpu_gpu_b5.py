"""B5: the LIF route on the first GPU matches CPU under the T-32c seed-batch contract.

Contract (Hamm ruling 2026-11-23, B5 option a): spikes bitwise equal at every
timestep; V_m allclose rtol=0, atol=1e-4 mV. Both the eager path
(``model.simulate``) and the seed-batch path (``simulate_lif_seed_batch``) run
seeds 11, 12, 13 for 200 ms on cpu and on gpu. Placement is
``RuntimeConfig(backend=...)``, which pins each run through ``_device_scope``.
The test skips when JAX exposes no GPU device.
"""

import numpy as np
import pytest

import jax

import jaxfne as jtfne
from jaxfne._model_simulate import simulate_lif_seed_batch
from test_lif_tensor_route import DT_MS, _spiking_lif_model

DUR_MS = 200.0
SEEDS = [11, 12, 13]
POISSON = {"rate_hz": 50.0, "amplitude": 100.0}
V_RTOL = 0.0
V_ATOL_MV = 1e-4


def _require_gpu():
    try:
        devices = jax.devices("gpu")
    except (RuntimeError, ValueError):
        devices = []
    if not devices:
        pytest.skip("no GPU device visible to jax")


def _sim(backend, seed=None):
    kwargs = dict(
        duration_ms=DUR_MS, dt_ms=DT_MS, poisson_drive=POISSON,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", backend=backend))
    if seed is not None:
        kwargs["seed"] = seed
    return jtfne.Simulation(**kwargs)


def _assert_contract(cpu_spk, gpu_spk, cpu_v, gpu_v, label):
    cpu_spk = np.asarray(cpu_spk)
    gpu_spk = np.asarray(gpu_spk)
    assert int(cpu_spk.sum()) > 0, f"{label}: fixture must spike on cpu"
    assert gpu_spk.shape == cpu_spk.shape, f"{label}: spike shape differs"
    steps = np.flatnonzero((gpu_spk != cpu_spk).reshape(cpu_spk.shape[0], -1).any(axis=1))
    assert steps.size == 0, f"{label}: spikes differ at steps {steps[:10].tolist()}"
    np.testing.assert_allclose(
        np.asarray(gpu_v, dtype=float), np.asarray(cpu_v, dtype=float),
        rtol=V_RTOL, atol=V_ATOL_MV, err_msg=label)


def test_lif_eager_gpu_matches_cpu_at_200ms():
    _require_gpu()
    model = _spiking_lif_model()
    cpu = {s: model.simulate(_sim("cpu", seed=s)) for s in SEEDS}
    gpu = {s: model.simulate(_sim("gpu", seed=s)) for s in SEEDS}
    for s in SEEDS:
        assert gpu[s].metadata["runtime"]["actual_backend"] == "gpu"
        assert gpu[s].metadata["runtime"]["backend_enforced"] is True
        _assert_contract(cpu[s].spikes, gpu[s].spikes, cpu[s].V_m, gpu[s].V_m,
                         f"eager seed {s}")


def test_lif_seed_batch_gpu_matches_cpu_at_200ms():
    _require_gpu()
    model = _spiking_lif_model()
    cpu = simulate_lif_seed_batch(model, _sim("cpu"), SEEDS)
    gpu = simulate_lif_seed_batch(model, _sim("gpu"), SEEDS)
    assert gpu["metadata"]["runtime"]["actual_backend"] == "gpu"
    assert gpu["metadata"]["runtime"]["backend_enforced"] is True
    for k, s in enumerate(SEEDS):
        _assert_contract(cpu["spikes"][k], gpu["spikes"][k], cpu["V_m"][k], gpu["V_m"][k],
                         f"seed-batch seed {s}")
