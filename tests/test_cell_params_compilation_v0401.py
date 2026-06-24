"""Runtime verification of cell parameter compilation (cell_params)."""

import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne

def test_cell_params_compilation():
    # 1. Setup Configuration with overrides
    cfg = (jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=10.0, dt_ms=0.1)
        .column(name="v1_col", layers=["L4"], n=100)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity()
        .cell_params({"cell_type": "SST"}, {"a": 0.01})
        .cell_params({"cell_type": "VIP"}, {"a": 0.01})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"]))
        
    model = jtfne.construct(cfg)
    emitter = model.params["emitter"]
    
    # 2. Extract types and parameters
    labels = emitter.labels
    a = np.asarray(emitter.a)
    
    # Check SST overrides compiled
    sst_indices = [i for i, lbl in enumerate(labels) if lbl == "SST"]
    assert len(sst_indices) > 0, "No SST neurons found in column."
    for idx in sst_indices:
        assert np.isclose(a[idx], 0.01), f"SST neuron at index {idx} had a={a[idx]}, expected 0.01"
        
    # Check VIP overrides compiled
    vip_indices = [i for i, lbl in enumerate(labels) if lbl == "VIP"]
    assert len(vip_indices) > 0, "No VIP neurons found in column."
    for idx in vip_indices:
        assert np.isclose(a[idx], 0.01), f"VIP neuron at index {idx} had a={a[idx]}, expected 0.01"
        
    # Check E defaults preserved (a=0.02)
    e_indices = [i for i, lbl in enumerate(labels) if lbl == "E"]
    for idx in e_indices:
        assert np.isclose(a[idx], 0.02), f"E neuron at index {idx} had a={a[idx]}, expected 0.02"
        
    # Check PV defaults preserved (a=0.10)
    pv_indices = [i for i, lbl in enumerate(labels) if lbl == "PV"]
    for idx in pv_indices:
        assert np.isclose(a[idx], 0.10), f"PV neuron at index {idx} had a={a[idx]}, expected 0.10"

def test_cell_params_simulation_smoke():
    # 3. Verify simulate() executes successfully with these parameters
    cfg = (jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=10.0, dt_ms=0.1)
        .column(name="v1_col", layers=["L4"], n=100)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity()
        .cell_params({"cell_type": "SST"}, {"a": 0.01})
        .cell_params({"cell_type": "VIP"}, {"a": 0.01})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"]))
        
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1)
    
    assert isinstance(signals, jtfne.Signals)
    assert signals.spikes.shape == (100, 100)
    assert bool(jnp.all(jnp.isfinite(signals.spikes)))
