import jax
import jax.numpy as jnp
import pytest
import jaxfne as jtfne


def _cfg(n=20):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_ablation_e_silence():
    cfg = _cfg(20)
    model = jtfne.construct(cfg)
    
    # E_silence: silence Excitatory neurons
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, ablation="E_silence")
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m"])
    
    # Excitatory neurons are 0.8 * 20 = 16 neurons
    # Let's verify that silenced excitatory neurons emit exactly 0 spikes
    spikes_out = readout["spikes"]  # shape (n_steps, n_neurons)
    
    for idx, row in enumerate(model.neuron_table()):
        if row["cell_type"].startswith("E"):
            # Excitatory neuron spikes must be exactly 0
            assert jnp.sum(spikes_out[:, row["neuron_id"]]) == 0
        else:
            # Inhibitory/other neurons can spike normally
            pass

    # Verify metadata
    assert signals.metadata["ablation"] == "E_silence"


def test_ablation_i_silence():
    cfg = _cfg(20)
    model = jtfne.construct(cfg)
    
    # I_silence: silence inhibitory neurons
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, ablation="I_silence")
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m"])
    
    spikes_out = readout["spikes"]
    
    for idx, row in enumerate(model.neuron_table()):
        if not row["cell_type"].startswith("E"):
            # Inhibitory neuron spikes must be exactly 0
            assert jnp.sum(spikes_out[:, row["neuron_id"]]) == 0

    assert signals.metadata["ablation"] == "I_silence"


def test_ablation_disconnected_null():
    cfg = _cfg(10)
    model = jtfne.construct(cfg)
    
    # disconnected_null: all weights set to 0
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, ablation="disconnected_null")
    signals = model.simulate(sim)
    
    # Verify the metadata and that the simulation executes without failure
    assert signals.metadata["ablation"] == "disconnected_null"
    assert jnp.all(jnp.isfinite(signals.V_m))


def test_ablation_shuffled_timing():
    cfg = _cfg(10)
    model = jtfne.construct(cfg)
    
    # Provide a drive schedule
    sched = jtfne.stimulus_schedule(
        events=[
            jtfne.core.ParadigmEvent(label="stim", onset_ms=2.0, duration_ms=5.0, stimulus="excitatory_pulse")
        ],
        n_neurons=10
    )
    
    # Verify shuffled timing run
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, ablation="shuffled_timing")
    signals = model.simulate(sim, paradigm=sched)
    
    assert signals.metadata["ablation"] == "shuffled_timing"
    assert jnp.all(jnp.isfinite(signals.V_m))
