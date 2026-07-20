"""Config #2b of the small-network smart-test matrix (see plans.json:
smart-test-matrix-configs-2-5). Companion to config #2a
(tests/test_ei_jaxley_izhikevich_parity.py, JaxleyBridge.simulate() -- voltage
only, no field).

Same E-PV connectivity shape as config #1/#2a (one excitatory-style synapse,
E->PV), but with a jaxley HH bridge emitter instead of native Izhikevich, and
via JaxleyBridge.simulate_laminar_field() specifically -- this is the
cross-*mechanism* comparison angle: HH is required here because Izhikevich is
non-capacitive (zero reconstructible ionic current) and cannot generate a
field (see tests/test_jaxley_emitter_bridge_e2e.py::test_simulate_laminar_field_requires_hh).
Intrinsic params are left native to jaxley's own HH defaults -- no forced
parameter parity with jaxfne's native kernel (this angle is about the
readout path, not cross-implementation parity).
"""
import numpy as np
import pytest

jaxley = pytest.importorskip("jaxley")
import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import connect

import jaxfne as jtfne

DT_MS = 0.025
DURATION_MS = 60.0
N_CONTACTS = 8


def _jaxley_epv_hh_network():
    cells = [jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1]) for _ in range(2)]
    net = jx.Network(cells)
    net.insert(HH())
    # E (cell 0, superficial) drives PV (cell 1, deeper) via one excitatory synapse.
    net.cell(0).branch(0).comp(0).set("z", 0.0)
    net.cell(1).branch(0).comp(0).set("z", 0.1)
    connect(net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse())
    drive = jx.step_current(i_delay=5.0, i_dur=40.0, i_amp=0.1, delta_t=DT_MS, t_max=DURATION_MS)
    net.cell(0).branch(0).comp(0).stimulate(drive)
    return net


def test_simulate_laminar_field_produces_finite_lfp_csd():
    net = _jaxley_epv_hh_network()
    sig = jtfne.JaxleyBridge(model=net).simulate_laminar_field(
        duration_ms=DURATION_MS, dt_ms=DT_MS, n_contacts=N_CONTACTS
    )
    assert sig.field is not None  # real FieldOutput, unlike voltage-only simulate()
    lfp = np.asarray(sig.get("lfp_proxy"))
    csd = np.asarray(sig.get("csd_proxy"))
    assert lfp.shape[1] == N_CONTACTS
    assert csd.shape[1] == N_CONTACTS
    assert bool(np.all(np.isfinite(lfp)))
    assert bool(np.all(np.isfinite(csd)))
    assert float(np.abs(lfp).max()) > 0.0  # driven E cell must generate a nonzero field


def test_simulate_laminar_field_carries_conservative_proxy_gates():
    net = _jaxley_epv_hh_network()
    sig = jtfne.JaxleyBridge(model=net).simulate_laminar_field(
        duration_ms=DURATION_MS, dt_ms=DT_MS, n_contacts=N_CONTACTS
    )
    assert sig.metadata["source_mode"] == "hh_ionic_current_reconstructed"
    assert sig.metadata["projection_mode"] == "density_preserving"
    assert sig.metadata["physical_amplitude_calibrated"] is False
    assert sig.metadata["claim_level"] == "computational_scaffold"


def test_izhikevich_cannot_generate_a_field_for_this_same_epv_shape():
    """The reason this config needs HH specifically, demonstrated on the SAME
    E-PV connectivity shape as config #2a: Izhikevich is non-capacitive."""
    from jaxley.channels import Izhikevich

    cells = [jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1]) for _ in range(2)]
    net = jx.Network(cells)
    net.insert(Izhikevich())
    connect(net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse())
    with pytest.raises(ValueError, match="HH"):
        jtfne.JaxleyBridge(model=net).simulate_laminar_field(duration_ms=DURATION_MS, dt_ms=DT_MS)
