"""Runtime verification of TCM V1 6-Population column construction and connection rules."""

import json
import numpy as np
from pathlib import Path
import pytest
import jaxfne as jtfne

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "tutorials" / "etudes" / "jaxfne_etude_tcm_v1_6pop.ipynb"
OUTPUT_DIR = REPO_ROOT / "local" / "tcm_v1"

def test_notebook_file_exists():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

def test_tcm_v1_6pop_connectivity_mask():
    # 1. Setup Configuration with tcm_v1_6pop=True
    cfg = (jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=10.0, dt_ms=0.1)
        .column(name="tcm_v1_col", layers=["L2/3", "L4", "L5", "L6"], n=100)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(tcm_v1_6pop=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"]))
        
    model = jtfne.construct(cfg)
    W = np.asarray(model.params["emitter"].W)
    n = W.shape[0]
    
    # 2. Extract neuron labels and layers
    labels = model.params["emitter"].labels
    layer_labels = model.params["emitter"].layer_labels
    
    # Map to 6 populations
    pop = []
    for idx in range(n):
        lyr = layer_labels[idx]
        cell = labels[idx]
        is_e = (cell == "E")
        if lyr in {"L2/3", "L2", "L3"}:
            pop.append("SP" if is_e else "SI")
        elif lyr == "L4":
            pop.append("SS" if is_e else "other")
        elif lyr == "L5":
            pop.append("DP" if is_e else "DI")
        elif lyr == "L6":
            pop.append("TP" if is_e else "other")
        else:
            pop.append("other")
            
    allowed_pairs = {
        ("SS", "SP"),
        ("SS", "SI"),
        ("SP", "SP"),
        ("SP", "SI"),
        ("SI", "SP"),
        ("DP", "DP"),
        ("DP", "DI"),
        ("DI", "DP"),
        ("SP", "DP"),
        ("DP", "SP"),
        ("SP", "TP"),
        ("TP", "DP")
    }
    
    # 3. Verify mask: non-allowed pairs must be strictly 0
    for i in range(n):
        for j in range(n):
            val = W[i, j]
            pre_pop = pop[j]
            post_pop = pop[i]
            if (pre_pop, post_pop) not in allowed_pairs:
                assert np.isclose(val, 0.0), f"Disallowed connection {pre_pop}->{post_pop} at ({i},{j}) has weight {val}"
            else:
                # Diagonal connections (self-connections) must still be 0.0
                if i == j:
                    assert np.isclose(val, 0.0), "Self connection at diagonal should be zero."
                    
    # Confirm that we have non-zero connections for allowed paths
    assert np.any(W != 0.0), "Weight matrix is entirely zero."

def test_tcm_v1_6pop_simulation_smoke():
    # 4. Verify simulation executes successfully
    cfg = (jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=10.0, dt_ms=0.1)
        .column(name="tcm_v1_col", layers=["L2/3", "L4", "L5", "L6"], n=100)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(tcm_v1_6pop=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"]))
        
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1)
    assert signals is not None
    assert signals.spikes is not None
