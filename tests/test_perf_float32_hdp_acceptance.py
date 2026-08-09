"""P-07 float32 HDP acceptance gate (CPU, deterministic).

Execution-record acceptance for the supported HDP path, exactly as the
roadmap's P-07 row specifies: three explicit fixed seeds (finite outputs,
invariant shapes, float32 arrays at the public surface, and
``record_weight_trace=False``), one same-seed deterministic repeat, one
closest-supported matched non-HDP control, and one real disconnected null
control through the existing ``ablation="disconnected_null"`` mechanism.

What this test establishes (no more):
- finite float32 execution on the tested CPU path, with shape/dtype
  invariants and same-seed repeat behavior;
- the disconnected null executes and its weights stay at the kernel's
  documented floor scale (no growth without recurrent input);
- truth metadata stays ``physical_amplitude_calibrated=False`` /
  ``field_claim_level="proxy_readout"``.

What it explicitly does not claim:
- no CUDA (or GPU) evidence of any kind;
- no claim that HDP improves stability, convergence, rate variance, or speed;
- no trajectory-equivalence between HDP and non-HDP simulations (the two
  kernels legitimately differ in floating-point detail; see the null-gains
  note in ``tests/test_hdp_dispatch.py``).

Source anchors:
- HDP kernel: ``jaxfne/emitters.py:1040``
  ``simulate_edge_recurrent_izhikevich_hdp``; alpha=beta=gamma=delta=
  C_spike=0 (the default) pins H_i at its 1.0 initial value and makes the
  K_HDP-scaled weight term identically zero -- the documented null control
  (:1052-1055); synaptic current is ``I_syn_i = sum_j w_ji * x_j`` (:1092),
  so a zeroed edge set carries no recurrent contribution.
- Kernel stochastic noise: ``noise_scale=None`` -> ``noise_coef`` 0.5 with
  a seeded ``bulk_noise`` array (``emitters.py:422-435``); a disconnected
  network at default settings therefore still has noise-driven per-neuron
  spikes, and the null control must NOT be asserted to have zero spike
  output.
- Ablation dispatch: ``jaxfne/_model_simulate.py:221-223`` zeroes every
  edge weight for ``ablation="disconnected_null"`` before HDP kernel entry.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxfne as jtfne

D, DT = 200.0, 0.5  # 400 steps; short CPU workload (not the P-06 large preset)
N = 45
HDP_RUNTIME = dict(
    enable_hdp=True,
    dtype="float32",
    hdp_params=dict(
        K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0,
        rho_passive=0.0, barrier_c=0.01, barrier_d=0.01,
        record_weight_trace=False,
    ),
)
# Measured on the P-07 baseline probe (same workload): the null-control
# |w_final| max is 1.0038e-3, exactly the kernel's w_floor=1e-3 magnitude
# (floor, not weight growth). The H equilibrium box is deliberately ~20x
# wider than the measured H_trace range [1.000000, 1.002415] so it cannot
# be mistaken for a stability claim.
W_FLOOR = 1.0e-3
H_EQUILIBRIUM_BOX = (0.98, 1.02)


def _build_model(n: int = N):
    cfg = (
        jtfne.build_laminar_column(n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return jtfne.construct(cfg)


def _run(model, *, seed: int, hdp: bool, ablation: str | None = None):
    if hdp:
        rt = jtfne.RuntimeConfig(**HDP_RUNTIME)
    else:
        rt = jtfne.RuntimeConfig(enable_hdp=False, dtype="float32")
    sim = jtfne.simulation(duration_ms=D, dt_ms=DT, seed=seed,
                           ablation=ablation, runtime=rt)
    return model.simulate(sim)


@pytest.fixture(scope="module")
def hdp_model():
    return _build_model()


def _n_steps() -> int:
    return int(D / DT)


def _assert_float32_contract(sig):
    assert sig.V_m.dtype == jnp.float32
    assert sig.spikes.dtype == jnp.float32
    if sig.sources is not None:
        assert sig.sources.dtype == jnp.float32
    assert np.isfinite(np.asarray(sig.V_m)).all()
    assert np.isfinite(np.asarray(sig.spikes)).all()


def _assert_metadata_truth(sig):
    meta = sig.metadata
    assert meta["runtime"]["actual_dtype"] == "float32"
    assert meta["source_bookkeeping"]["physical_amplitude_calibrated"] is False
    assert meta["field_claim_level"] == "proxy_readout"
    assert isinstance(meta["runtime"]["jit"], bool)


# ---------------------------------------------------------------------------
# 1. Three-seed float32 acceptance for the supported HDP path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_hdp_float32_three_seed_acceptance(hdp_model, seed):
    """Supported HDP kernel with fixed seed: finite, invariant shapes,
    float32 at the public surface, ``record_weight_trace=False`` honored."""
    sig = _run(hdp_model, seed=seed, hdp=True)
    assert sig.metadata["hdp"]["enabled"] is True
    assert list(sig.V_m.shape) == [_n_steps(), N]
    assert list(sig.spikes.shape) == [_n_steps(), N]
    if sig.sources is not None:
        assert list(sig.sources.shape) == [_n_steps(), N]

    diag = hdp_model.last_hdp_diagnostics()
    assert diag is not None
    # record_weight_trace=False -> no O(T,E) per-step trace anywhere.
    assert diag["w_trace"] is None
    assert diag["H_trace"] is not None
    assert diag["w_final"] is not None

    _assert_float32_contract(sig)
    _assert_metadata_truth(sig)
    # No cross-seed trajectory equality is asserted (seeds differ by design).


# ---------------------------------------------------------------------------
# 2. Same-seed deterministic repeat
# ---------------------------------------------------------------------------

def test_hdp_float32_repeat_is_deterministic(hdp_model):
    """Identical HDP configuration, identical seed: bit-identical results.

    Exact equality is asserted because the eager CPU path is fully seeded
    (key = PRNGKey(seed); emitters.py:428-435 splits noise and drive from
    the simulation key) and the HDP dispatch is deterministic under a fixed
    seed. This mirrors the exact-result equality already pinned in
    tests/test_phaseC_H_carry_resume.py for chunk/continuous HDP carry.
    """
    a = _run(hdp_model, seed=7, hdp=True)
    da = hdp_model.last_hdp_diagnostics()
    b = _run(hdp_model, seed=7, hdp=True)
    db = hdp_model.last_hdp_diagnostics()
    assert np.array_equal(np.asarray(da["H_trace"]), np.asarray(db["H_trace"]))
    assert np.array_equal(np.asarray(da["w_final"]), np.asarray(db["w_final"]))


# ---------------------------------------------------------------------------
# 3. Matched non-HDP control
# ---------------------------------------------------------------------------

def test_hdp_float32_matched_non_hdp_control(hdp_model):
    """Closest-supported non-HDP arm: same population/duration/dt/dtype/
    seed/recording; HDP disabled only via the existing non-HDP path."""
    sig = _run(hdp_model, seed=7, hdp=False)
    assert "hdp" not in sig.metadata or sig.metadata["hdp"] is None
    assert sig.V_m.shape == (_n_steps(), N)
    assert sig.spikes.shape == (_n_steps(), N)
    _assert_float32_contract(sig)
    _assert_metadata_truth(sig)
    # No speedup/stability/superiority claim, and no trajectory-equivalence
    # assertion: the HDP and non-HDP kernels are different implementations
    # whose outputs are not bit-comparable (see test_hdp_dispatch.py).


# ---------------------------------------------------------------------------
# 4. Disconnected null control
# ---------------------------------------------------------------------------

def test_hdp_float32_disconnected_null_control(hdp_model):
    """Null control via ``ablation="disconnected_null"`` (zeroes every edge
    weight before kernel entry: _model_simulate.py:221-223), on the HDP path.

    Structural expectations derived from source (not invented):
    - no recurrent contribution: with all w_ji = 0, I_syn_i = sum_j w_ji*x_j
      is identically zero (emitters.py:1092);
    - weight magnitudes therefore stay at the floor scale ~w_floor=1e-3
      (measured |w_final| max 1.0038e-3) -- they never grow;
    - H stays inside the H-equilibrium box [0.98, 1.02] (measured H_trace
      [1.000000, 1.002415]); the box is deliberately much wider than
      measured so it is not a stability claim;
    - spikes are NOT asserted to be zero: the kernel injects documented
      noise_scale=0.5 bulk noise (emitters.py:422-435) even without
      recurrent input (measured 106 spikes in the probe).
    """
    sig = _run(hdp_model, seed=7, hdp=True, ablation="disconnected_null")
    assert sig.metadata["ablation"] == "disconnected_null"
    diag = hdp_model.last_hdp_diagnostics()
    wf = np.asarray(diag["w_final"])
    H = np.asarray(diag["H_trace"])
    assert np.max(np.abs(wf)) <= 2.0 * W_FLOOR  # floor scale, no growth
    lo, hi = H_EQUILIBRIUM_BOX
    assert float(np.min(H)) >= lo and float(np.max(H)) <= hi
    assert sig.V_m.shape == (_n_steps(), N)
    assert sig.spikes.shape == (_n_steps(), N)
    _assert_float32_contract(sig)
    _assert_metadata_truth(sig)