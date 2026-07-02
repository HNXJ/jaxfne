"""Config #1 of the jaxfne-modular-grammar smoke-test matrix (plans.json:
vis-smoke-test-coverage-gap follow-on; see AGENTS.md jaxfne-modular-grammar rule 4).

Smallest possible end-to-end pipeline pin: 2 neurons (1 E, 1 PV), native
Izhikevich emitter, dt=0.1ms / 100ms, full chain
Configuration -> construct -> simulate -> Signals -> vis, with a 5-contact
laminar LFP probe and a manually-derived EEG proxy. Deliberately a smoke test
(finite, right shape, no exception), not a value-accuracy test -- accuracy of
the Izhikevich/HDP mechanisms themselves is covered elsewhere
(tests/test_hdp_kernel_standalone.py, tests/test_homeostatic_stability_v042.py).

Uses the ``.network(n=..., cell_types=...)`` construction path (same as
tests/test_api_smoke.py), NOT ``.population()``/``.geometry()``/``.cell_types()``
-- that alternate chain was found to mis-assign cell types at N=2 (produces
PV+VIP instead of the declared E+PV split); see progress.json entry for
jaxfne/core.py::Configuration.population for the tracked bug.
"""
import jax.numpy as jnp
import pytest

import jaxfne as jtfne

DT_MS = 0.1
DURATION_MS = 100.0
N_CONTACTS = 5


def _two_neuron_epv_config() -> "jtfne.Configuration":
    return (
        jtfne.configuration()
        .network(name="EI2", kind="cortical_column", n=2, cell_types={"E": 0.5, "PV": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"], n_contacts=N_CONTACTS)
    )


def _simulate():
    cfg = _two_neuron_epv_config()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0)
    signals = model.simulate(sim)
    return model, signals


def test_config_roundtrips_json():
    cfg = _two_neuron_epv_config()
    payload = jtfne.io.json_safe(cfg.metadata)
    assert isinstance(payload, dict)


def test_construct_produces_exactly_one_e_and_one_pv():
    cfg = _two_neuron_epv_config()
    model = jtfne.construct(cfg)
    rows = model.neuron_table()
    assert len(rows) == 2
    cell_types = sorted(r["cell_type"] for r in rows)
    assert cell_types == ["E", "PV"]


def test_lfp_proxy_has_declared_contact_count():
    _, sig = _simulate()

    lfp = sig.get("lfp")
    assert lfp is not None
    assert lfp.shape[-1] == N_CONTACTS
    assert bool(jnp.all(jnp.isfinite(lfp)))


def test_eeg_proxy_derivable_from_lfp():
    # eeg_proxy is NOT auto-produced by simulate()/model.probe() -- it must be
    # derived manually from an lfp array via jtfne.fields.eeg_proxy_probe().
    # signals.get("eeg")/get("eeg_proxy") raises KeyError; do not expect it to work.
    _, sig = _simulate()
    lfp = sig.get("lfp")

    with pytest.raises(KeyError):
        sig.get("eeg")

    eeg_readout = jtfne.fields.eeg_proxy_probe(lfp)
    assert eeg_readout.data is not None
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(eeg_readout.data))))
