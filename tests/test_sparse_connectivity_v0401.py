"""Runtime verification of sparse connectivity (p_connect) compilation."""

import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne

def _build_cfg(seed, p_connect=None):
    cfg = (jtfne.Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=10.0, dt_ms=0.1)
        .column(name="test_col", layers=["L4"], n=100)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}))
    if p_connect is not None:
        cfg = cfg.connectivity(p_connect=p_connect)
    else:
        cfg = cfg.connectivity()
    cfg = cfg.set_emitter("izhikevich", "cortical_eig")
    cfg = cfg.probes(["spikes"])
    return cfg

def test_sparse_connectivity_density():
    # 1. Verify density with p_connect=0.1
    cfg = _build_cfg(seed=42, p_connect=0.1)
    model = jtfne.construct(cfg)
    W = np.asarray(model.params["emitter"].W)
    n = W.shape[0]
    
    # Calculate actual density (excluding diagonal)
    off_diag_elements = n * (n - 1)
    non_zeros = np.sum(W != 0)
    density = non_zeros / off_diag_elements
    
    # 0.1 connectivity on 100x100 matrix should be around 10%
    assert 0.06 <= density <= 0.14, f"Density was {density:.4f}, expected ~0.10"

def test_sparse_connectivity_determinism():
    # 2. Verify identical seed yields same W
    cfg1 = _build_cfg(seed=100, p_connect=0.2)
    cfg2 = _build_cfg(seed=100, p_connect=0.2)
    
    model1 = jtfne.construct(cfg1)
    model2 = jtfne.construct(cfg2)
    
    W1 = np.asarray(model1.params["emitter"].W)
    W2 = np.asarray(model2.params["emitter"].W)
    
    assert np.allclose(W1, W2), "Reconstruction with same seed was not deterministic."

def test_sparse_connectivity_stochasticity():
    # 3. Verify different seeds yield different W
    cfg1 = _build_cfg(seed=100, p_connect=0.2)
    cfg2 = _build_cfg(seed=200, p_connect=0.2)
    
    model1 = jtfne.construct(cfg1)
    model2 = jtfne.construct(cfg2)
    
    W1 = np.asarray(model1.params["emitter"].W)
    W2 = np.asarray(model2.params["emitter"].W)
    
    assert not np.allclose(W1, W2), "Different seeds produced identical connectivity matrices."

def test_sparse_connectivity_backward_compatibility():
    # 4. Verify omitting p_connect yields dense weights
    cfg = _build_cfg(seed=42, p_connect=None)
    model = jtfne.construct(cfg)
    W = np.asarray(model.params["emitter"].W)
    n = W.shape[0]
    
    # Verify off-diagonal is fully non-zero
    off_diag_mask = ~np.eye(n, dtype=bool)
    assert np.all(W[off_diag_mask] != 0.0), "Default connectivity should be fully dense off-diagonal."
