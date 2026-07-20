"""Config #5 of the jaxfne-modular-grammar smoke-test matrix
(plans.json:smart-test-matrix-configs-2-5).

Population scale-up sanity: same E-PV motif and probe config as config #1
(tests/test_epv_2neuron_pipeline_smoke.py) -- native Izhikevich emitter,
dt=0.1ms/100ms, full Configuration -> construct -> simulate -> Signals chain,
5-contact laminar LFP probe, manually-derived EEG proxy -- just a larger N=20
population instead of N=2. Deliberately a smoke test (finite, right shape, no
exception), not a value-accuracy test.
"""
import jax.numpy as jnp
import pytest

import jaxfne as jtfne

DT_MS = 0.1
DURATION_MS = 100.0
N_CONTACTS = 5
N_NEURONS = 20


def _twenty_neuron_epv_config() -> "jtfne.Configuration":
    return (
        jtfne.configuration()
        .network(name="EI20", kind="cortical_column", n=N_NEURONS, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"], n_contacts=N_CONTACTS)
    )


def _simulate():
    cfg = _twenty_neuron_epv_config()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0)
    signals = model.simulate(sim)
    return model, signals


def test_config_roundtrips_json():
    cfg = _twenty_neuron_epv_config()
    payload = jtfne.io.json_safe(cfg.metadata)
    assert isinstance(payload, dict)


def test_construct_produces_15_e_and_5_pv():
    cfg = _twenty_neuron_epv_config()
    model = jtfne.construct(cfg)
    rows = model.neuron_table()
    assert len(rows) == N_NEURONS
    cell_types = [r["cell_type"] for r in rows]
    assert cell_types.count("E") == 15
    assert cell_types.count("PV") == 5


def test_signals_finite_and_sane():
    _, sig = _simulate()
    V_m = jnp.asarray(sig.V_m)
    spikes = jnp.asarray(sig.spikes)
    assert V_m.shape == (1000, N_NEURONS)  # 100ms / 0.1ms
    assert bool(jnp.all(jnp.isfinite(V_m)))
    assert bool(jnp.all(jnp.abs(V_m) < 150.0))
    assert bool(jnp.all(jnp.isfinite(spikes)))


def test_lfp_proxy_has_declared_contact_count():
    _, sig = _simulate()

    lfp = sig.get("lfp")
    assert lfp is not None
    assert lfp.shape[-1] == N_CONTACTS
    assert bool(jnp.all(jnp.isfinite(lfp)))


def test_eeg_proxy_derivable_from_lfp():
    # eeg_proxy is NOT auto-produced by simulate()/model.probe() -- it must be
    # derived manually from an lfp array via jtfne.fields.eeg_proxy_probe().
    _, sig = _simulate()
    lfp = sig.get("lfp")

    with pytest.raises(KeyError):
        sig.get("eeg")

    eeg_readout = jtfne.fields.eeg_proxy_probe(lfp)
    assert eeg_readout.data is not None
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(eeg_readout.data))))
