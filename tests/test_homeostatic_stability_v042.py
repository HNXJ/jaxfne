"""v0.4.2 homeostatic numerical stability: hard-bounded, float32-safe emitters.

The homeostatic mode keeps each emitter in a stable operating regime AND hard-bounds
its state, so a single neuron / simple component can never overflow or underflow in
float32 — even under absurd or non-finite drive. These tests drive deliberately
extreme inputs and assert finite, in-range output.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne

# These tests assert the float32 guarantee; they must run in default (float32) precision.
_F32 = jnp.float32(0.0).dtype


def _bounds_ok(V, lo=-150.0, hi=100.0, tol=1e-2):
    V = np.asarray(V)
    return bool(np.all(np.isfinite(V))) and float(V.min()) >= lo - tol and float(V.max()) <= hi + tol


@pytest.mark.parametrize("drive", [1.0e6, -1.0e6, np.inf, -np.inf])
def test_native_izhikevich_homeostasis_hard_bounded_float32(drive):
    """Native Izhikevich homeostatic emitter stays finite + bounded under extreme drive."""
    cfg = (
        jtfne.build_laminar_column(n=24, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
    )
    model = jtfne.construct(cfg)
    n = int(model.params["emitter"].v0.shape[0])
    model = jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.full(n, drive, dtype=jnp.float32))
    rt = jtfne.RuntimeConfig(enable_homeostasis=True, homeostasis_params={"k_gain": 1.0})
    sig = jtfne.simulate(model, sim=jtfne.Simulation(duration_ms=150.0, dt_ms=0.5, seed=0, runtime=rt))

    V = np.asarray(sig.V_m)
    assert sig.V_m.dtype == _F32, f"expected float32, got {sig.V_m.dtype}"
    assert _bounds_ok(V), f"V_m not finite/bounded under drive={drive}: range {V.min()}..{V.max()}"


def test_native_homeostasis_does_not_perturb_normal_regime():
    """The hard bounds are a pure safety net: normal-regime dynamics are physiological."""
    cfg = (
        jtfne.build_laminar_column(n=24, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
    )
    model = jtfne.construct(cfg)
    rt = jtfne.RuntimeConfig(enable_homeostasis=True, homeostasis_params={"k_gain": 1.0})
    sig = jtfne.simulate(model, sim=jtfne.Simulation(duration_ms=200.0, dt_ms=0.5, seed=0, runtime=rt))
    V = np.asarray(sig.V_m)
    # rest near -65, spike-reset peak well below the +100 ceiling; never hits clamps
    assert np.isfinite(V).all()
    assert -90.0 < V.min() and V.max() < 60.0


def test_jaxley_homeostatic_single_neuron_float32_no_overflow():
    """Single HH neuron under absurd drive stays finite: current is hard-bounded."""
    jaxley = pytest.importorskip("jaxley")
    from jaxley.channels import HH

    cell = jaxley.Cell(jaxley.Branch(jaxley.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    cell.record("v")
    bridge = jtfne.JaxleyBridge(model=cell)
    # absurd base current -> clamped to current_clip_nA -> solver stays stable
    sig = bridge.simulate_homeostatic(
        duration_ms=150.0, base_current_nA=1.0e6, target_rate_hz=40.0,
        k_gain=0.3, current_clip_nA=0.02, strict_finite=True,
    )
    V = np.asarray(sig.V_m)
    assert np.isfinite(V).all()
    assert sig.metadata["homeostasis"]["all_finite"] is True
    assert sig.metadata["homeostasis"]["state_hard_bounded"] is True
    # single neuron -> one column
    assert V.shape[1] == 1
