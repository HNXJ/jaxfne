"""End-to-end Jaxley emitter bridge: clip-compat shim, simulate(), and converters.

These exercise the real Jaxley path (channels actually integrate). They skip
cleanly when Jaxley is not installed, so the suite stays green without the
optional dependency.
"""
import numpy as np
import pytest

jaxley = pytest.importorskip("jaxley")
import jaxfne as jtfne
from jaxfne import bridges


def _hh_cell():
    from jaxley.channels import HH
    cell = jaxley.Cell(jaxley.Branch(jaxley.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    return cell


def test_clip_compat_shim_lets_hh_integrate():
    # The shim is what makes Jaxley channels run on current JAX.
    assert bridges._install_jax_clip_compat() in (True, False)
    import jax.numpy as jnp
    # after require_jaxley, a_min/a_max kwargs must be accepted (shim or native)
    bridges.require_jaxley()
    out = jnp.clip(jnp.asarray([-5.0, 0.0, 5.0]), a_min=-1.0, a_max=1.0)
    assert float(out.min()) == -1.0 and float(out.max()) == 1.0


def test_hh_jaxley_reference_trace_is_physiological():
    t, V, I_inj = bridges.hh_jaxley_reference_trace(duration_ms=80.0, dt_ms=0.025)
    assert t.shape == V.shape == I_inj.shape
    assert np.all(np.isfinite(V))
    # reduced HH single compartment: rest near -70, spike peak well above 0
    assert -90.0 < float(V.min()) < -50.0
    assert float(V.max()) > 0.0


def test_jaxley_bridge_simulate_end_to_end():
    cell = _hh_cell()
    cell.record("v")
    # realistic usage: the user wires stimulus on the Jaxley model, bridge runs it
    i = jaxley.step_current(i_delay=8.0, i_dur=40.0, i_amp=0.1, delta_t=0.025, t_max=80.0)
    cell.stimulate(i)
    bridge = jtfne.JaxleyBridge(model=cell)
    sig = bridge.simulate(duration_ms=80.0, dt_ms=0.025)
    # returns a real Signals with plausible Vm and conservative proxy gates
    assert sig.V_m.ndim == 2  # [T, N]
    assert bool(np.all(np.isfinite(np.asarray(sig.V_m))))
    assert float(np.asarray(sig.V_m).max()) > 0.0  # stimulated HH spikes above 0 mV
    assert sig.field is None
    assert sig.metadata["physical_amplitude_calibrated"] is False
    assert sig.metadata["claim_level"] == "computational_scaffold"
    assert sig.metadata["bridge"] == "jaxley_module_to_signals"


def test_jaxley_to_signals_carries_positions_and_layout():
    # 2-compartment cell, record both -> recordings (2, T) maps to V_m [T, 2]
    cell = jaxley.Cell(jaxley.Branch(jaxley.Compartment(), ncomp=2), parents=[-1])
    from jaxley.channels import HH
    cell.insert(HH())
    cell.branch(0).comp(0).record("v")
    cell.branch(0).comp(1).record("v")
    rec = jaxley.integrate(cell, delta_t=0.025, t_max=40.0, solver="bwd_euler")
    assert rec.shape[0] == 2  # (n_recordings, n_time)
    sig = jtfne.jaxley_to_signals(cell, rec, dt_ms=0.025)
    assert sig.V_m.shape[1] == 2  # [T, N=2]
    assert sig.metadata["jaxley_n_recordings"] == 2
    # geometry pulled from module.nodes for the recorded compartments
    pos = sig.metadata.get("recorded_positions_xyz")
    assert pos is not None and len(pos) == 2 and len(pos[0]) == 3
