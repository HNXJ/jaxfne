"""Config #2 of the small-network smart-test matrix (see plans.json:
smart-test-matrix-configs-2-5). Proves jaxley can be borrowed as an emitter
backend for the SAME neuron model (Izhikevich) as jaxfne's native kernel,
smoothly integrating into the jaxfne Signals/vis pipeline via JaxleyBridge.

Not a numerical parity test (two independent implementations won't produce
bit-identical dynamics) -- a smoke/plausibility test: does a jaxley-built
2-neuron E-PV Izhikevich network run end-to-end through JaxleyBridge and
produce Signals as sane as the native pipeline's (tests/test_epv_2neuron_pipeline_smoke.py)?

Uses jaxley.connect() (real API, jaxley/connect.py) with IonotropicSynapse to
wire a single E->PV synapse -- jaxley.channels.Izhikevich is the real jaxley
Izhikevich channel (confirmed 2026-07-01: default params
Izhikevich_a=0.02/b=0.2/c=-65.0/d=8, matching jaxfne's own native defaults).
JaxleyBridge.simulate() is channel-agnostic (works with HH or Izhikevich);
only JaxleyBridge.simulate_laminar_field() requires HH specifically (see
config #2b, tests/test_ei_jaxley_hh_field_readout.py, not yet built).
"""
import jax.numpy as jnp
import pytest

jaxley = pytest.importorskip("jaxley")
import jaxley as jx
from jaxley.channels import Izhikevich
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import connect

import jaxfne as jtfne

DT_MS = 0.1
DURATION_MS = 100.0


def _jaxley_epv_network():
    cells = [jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1]) for _ in range(2)]
    net = jx.Network(cells)
    net.insert(Izhikevich())
    # cell 0 = E (drives cell 1 = PV) via a single excitatory-style ionotropic synapse
    connect(net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse())
    net.cell(0).branch(0).comp(0).record("v")
    net.cell(1).branch(0).comp(0).record("v")
    drive = jx.step_current(i_delay=10.0, i_dur=80.0, i_amp=0.02, delta_t=DT_MS, t_max=DURATION_MS)
    net.cell(0).branch(0).comp(0).stimulate(drive)
    return net


def test_jaxley_izhikevich_channel_has_native_matching_defaults():
    # Confirms the borrowed emitter uses the same neuron-model constants
    # jaxfne's own native Izhikevich kernel does (a=0.02, b=0.2, c=-65.0, d=8) --
    # a genuine "same model, different backend" comparison, not apples-to-oranges.
    params = Izhikevich().channel_params
    assert params["Izhikevich_a"] == pytest.approx(0.02)
    assert params["Izhikevich_b"] == pytest.approx(0.2)
    assert params["Izhikevich_c"] == pytest.approx(-65.0)
    assert params["Izhikevich_d"] == pytest.approx(8)


def test_jaxley_bridge_simulate_produces_finite_two_neuron_signals():
    net = _jaxley_epv_network()
    bridge = jtfne.JaxleyBridge(model=net)
    sig = bridge.simulate(duration_ms=DURATION_MS, dt_ms=DT_MS)

    expected_steps = round(DURATION_MS / DT_MS) + 1  # jaxley integrate includes t=0
    assert sig.V_m.shape == (expected_steps, 2)
    assert bool(jnp.all(jnp.isfinite(sig.V_m)))
    # driven E cell should spike (Izhikevich reset ~ -65mV, peak well above 0mV)
    assert float(sig.V_m.max()) > 0.0
    assert float(sig.V_m.min()) < -50.0


def test_jaxley_bridge_carries_conservative_proxy_gates():
    # Same claim-level discipline as the native pipeline: never escalated.
    net = _jaxley_epv_network()
    sig = jtfne.JaxleyBridge(model=net).simulate(duration_ms=DURATION_MS, dt_ms=DT_MS)
    assert sig.field is None
    assert sig.metadata["physical_amplitude_calibrated"] is False
    assert sig.metadata["claim_level"] == "computational_scaffold"
    assert sig.metadata["bridge"] == "jaxley_module_to_signals"


def test_jaxley_bridge_signals_support_vis():
    pytest.importorskip("matplotlib")
    net = _jaxley_epv_network()
    sig = jtfne.JaxleyBridge(model=net).simulate(duration_ms=DURATION_MS, dt_ms=DT_MS)

    vm_fig = jtfne.vis.vm(sig)
    assert vm_fig is not None
    raster_fig = jtfne.vis.raster(sig)
    assert raster_fig is not None
