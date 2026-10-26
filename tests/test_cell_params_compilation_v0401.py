"""Runtime verification of cell parameter compilation (cell_params)."""

import numpy as np
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


def _het_cfg(c_values=None, drive=None):
    cfg = (jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=50.0, dt_ms=0.5)
        .column(name="v1_col", layers=["L4"], n=100)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"]))
    params = {}
    if c_values is not None:
        params["c"] = c_values
    if drive is not None:
        params["drive"] = drive
    return cfg.cell_params({"cell_type": "E"}, params) if params else cfg


def test_cell_params_accepts_one_value_per_matched_cell():
    n_e = sum(lbl == "E" for lbl in jtfne.construct(_het_cfg()).params["emitter"].labels)
    c_values = [-65.0 + 0.5 * i for i in range(n_e)]
    emitter = jtfne.construct(_het_cfg(c_values)).params["emitter"]
    e = [i for i, lbl in enumerate(emitter.labels) if lbl == "E"]
    assert np.allclose(np.asarray(emitter.c)[e], c_values)
    assert len(set(np.asarray(emitter.c)[e].tolist())) == n_e, "values must differ per cell"


def test_cell_params_refuses_a_value_list_of_the_wrong_length():
    import pytest

    with pytest.raises(ValueError, match="2 values.*matches"):
        jtfne.construct(_het_cfg([-65.0, -60.0]))


def test_per_cell_drive_changes_the_simulated_spikes():
    n_e = sum(lbl == "E" for lbl in jtfne.construct(_het_cfg()).params["emitter"].labels)
    uniform = jtfne.simulate(jtfne.construct(_het_cfg(drive=[6.0] * n_e)))
    graded = jtfne.simulate(jtfne.construct(_het_cfg(drive=[2.0 + 8.0 * i / n_e for i in range(n_e)])))
    assert float(np.asarray(uniform.get("spk")).sum()) > 0, "fixture must spike"
    assert not np.array_equal(np.asarray(uniform.get("spk")), np.asarray(graded.get("spk")))
