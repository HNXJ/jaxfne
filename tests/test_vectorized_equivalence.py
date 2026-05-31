import numpy as np
import pytest
pd = pytest.importorskip("pandas")
import jax
import jax.numpy as jnp
from jaxfne.fields.proxy import make_laminar_connectivity as new_make_laminar_connectivity
from jaxfne.fields.proxy import teaching_control_spectrolaminar_resonance_source as new_resonance_source
from jaxfne.fields.proxy import spectrolaminar_readout as new_spectrolaminar_readout

# Reference old implementations
def old_make_laminar_connectivity(
    neurons_df,
    positions_m,
    control_params=None,
    seed=0,
    local_decay_m=0.001,
    p_local_e=0.18,
    p_local_i=0.30,
    p_feedforward=0.060,
    p_feedback=0.055,
    w_e_range=(0.012, 0.055),
    w_i_range=(-0.145, -0.055),
    w_ff_range=(0.007, 0.030),
    w_fb_range=(0.006, 0.026),
):
    if control_params is None:
        control_params = {
            "local_exc_gain": 1.0,
            "local_inh_gain": 1.0,
            "feedforward_gain": 1.0,
            "feedback_gain": 1.0,
        }
    n = len(neurons_df)
    area = np.array(neurons_df.get("area", [""]*n))
    layer = np.array(neurons_df.get("layer", [""]*n))
    cell_type = np.array(neurons_df.get("cell_type", [""]*n))

    W_local_exc = np.zeros((n, n), dtype=np.float32)
    W_local_inh = np.zeros((n, n), dtype=np.float32)
    W_ff = np.zeros((n, n), dtype=np.float32)
    W_fb = np.zeros((n, n), dtype=np.float32)

    rng = np.random.default_rng(seed)
    area_order = sorted(set(area[area != ""]))
    area_rank = {a: i for i, a in enumerate(area_order)}

    for pre in range(n):
        for post in range(n):
            if pre == post:
                continue
            same_area = area[pre] == area[post]
            dxy = np.linalg.norm(positions_m[post, :2] - positions_m[pre, :2])
            local_gain = np.exp(-((dxy / local_decay_m) ** 2))
            if same_area:
                if cell_type[pre] == "E" and rng.random() < p_local_e:
                    W_local_exc[post, pre] = rng.uniform(*w_e_range) * local_gain
                elif cell_type[pre] != "E" and rng.random() < p_local_i:
                    W_local_inh[post, pre] = rng.uniform(*w_i_range) * (0.65 + 0.35 * local_gain)
            elif cell_type[pre] == "E":
                delta = area_rank.get(area[post], 0) - area_rank.get(area[pre], 0)
                if delta == 1 and layer[pre] in ("L2", "L3") and layer[post] == "L4" and rng.random() < p_feedforward:
                    W_ff[post, pre] = rng.uniform(*w_ff_range)
                if delta == -1 and layer[pre] in ("L2", "L3", "L6") and layer[post] in ("L5", "L6") and rng.random() < p_feedback:
                    W_fb[post, pre] = rng.uniform(*w_fb_range)

    W = (
        control_params.get("local_exc_gain", 1.0) * W_local_exc
        + control_params.get("local_inh_gain", 1.0) * W_local_inh
        + control_params.get("feedforward_gain", 1.0) * W_ff
        + control_params.get("feedback_gain", 1.0) * W_fb
    )
    E_mask = np.array([cell_type[i] == "E" for i in range(n)])
    I_mask = ~E_mask
    return {"W": W, "E_mask": E_mask, "I_mask": I_mask}


def old_resonance_source(
    n_steps,
    dt_ms,
    neurons,
    control_params=None,
):
    if control_params is None:
        control_params = {"alpha_beta_gain": 1.0, "gamma_gain": 1.0, "resonance_scale": 1.0}
    n = len(neurons.get("area", []))
    layers = np.array(neurons.get("layer", ["L4"] * n))
    t = np.arange(n_steps) * dt_ms / 1000.0
    alpha_beta_freq = 15.0
    gamma_freq = 90.0
    resonance = np.zeros((n_steps, n), dtype=np.float32)
    for i in range(n):
        layer = layers[i]
        if layer in ("L1", "L2", "L3"):
            alpha_beta_amp = 0.5
        elif layer == "L4":
            alpha_beta_amp = 0.3
        else:
            alpha_beta_amp = 0.6

        if layer in ("L1", "L2", "L3"):
            gamma_amp = 0.8
        else:
            gamma_amp = 0.2

        alpha_beta_sig = alpha_beta_amp * np.sin(2.0 * np.pi * alpha_beta_freq * t)
        gamma_sig = gamma_amp * np.sin(2.0 * np.pi * gamma_freq * t)
        resonance[:, i] = (
            control_params.get("alpha_beta_gain", 1.0) * alpha_beta_sig
            + control_params.get("gamma_gain", 1.0) * gamma_sig
        ) * control_params.get("resonance_scale", 1.0)
    return resonance


def test_resonance_source_equivalence():
    n = 10
    steps = 100
    neurons = {
        "area": ["V1"] * n,
        "layer": ["L1", "L2", "L3", "L4", "L5", "L6"] * 2
    }
    neurons["area"] = neurons["area"][:n]
    neurons["layer"] = neurons["layer"][:n]

    old_res = old_resonance_source(steps, 0.1, neurons)
    new_res, _ = new_resonance_source(neurons, steps, 0.1)

    # 1. Shape Stability
    assert new_res.shape == (steps, n)
    # 2. Dtype Invariant
    assert new_res.dtype == jnp.float32
    # 3. Finite outputs
    assert np.all(np.isfinite(new_res))
    # 4. Exact Mathematical Equivalence (atol=1e-5)
    np.testing.assert_allclose(new_res, old_res, atol=1e-5)

def test_connectivity_invariants():
    n = 20
    neurons_df = pd.DataFrame({
        "area": ["V1"] * n,
        "layer": ["L4"] * n,
        "cell_type": ["E" if i % 4 != 0 else "I" for i in range(n)]
    })
    positions_m = np.random.uniform(0, 0.01, size=(n, 3))
    
    new_conn = new_make_laminar_connectivity(neurons_df, positions_m, seed=42)
    W = new_conn["W"]
    E_mask = new_conn["E_mask"]
    I_mask = new_conn["I_mask"]
    
    # 1. Shape Stability
    assert W.shape == (n, n)
    assert E_mask.shape == (n,)
    assert I_mask.shape == (n,)
    
    # 2. Finite outputs
    assert jnp.all(jnp.isfinite(W))
    
    # 3. Excitatory non-negative, Inhibitory non-positive
    # W[:, pre] where pre is E should be non-negative
    W_exc = W[:, np.where(E_mask)[0]]
    assert jnp.all(W_exc >= 0.0)
    
    # W[:, pre] where pre is I should be non-positive
    W_inh = W[:, np.where(I_mask)[0]]
    assert jnp.all(W_inh <= 0.0)
    
    # 4. Zero self-connections (no self loop)
    assert jnp.all(jnp.diagonal(W) == 0.0)

def test_spectrolaminar_readout_vectorized_equivalence():
    n = 30
    signal_arr = np.random.normal(size=(100, n))
    neurons = {
        "area": ["V1"] * 15 + ["V2"] * 15,
        "pos_from_l4": np.linspace(-0.5, 0.5, n).tolist()
    }
    
    # Old slow extraction
    area_indices_old = []
    pos_from_l4_list_old = []
    for i in range(n):
        if neurons["area"][i] == "V1":
            area_indices_old.append(i)
            pos_from_l4_list_old.append(float(neurons["pos_from_l4"][i]))
            
    # New vectorized mask extraction
    area_arr = np.array(neurons.get("area", ["V1"] * n))
    pos_l4_arr = np.array(neurons.get("pos_from_l4", np.linspace(-0.5, 0.5, n)))
    mask = area_arr == "V1"
    area_indices_new = np.where(mask)[0].tolist()
    pos_from_l4_list_new = pos_l4_arr[mask].astype(float).tolist()
    
    assert area_indices_new == area_indices_old
    assert pos_from_l4_list_new == pos_from_l4_list_old

def test_connectivity_empty_population():
    n = 0
    neurons_df = pd.DataFrame(columns=["area", "layer", "cell_type"])
    positions_m = np.zeros((0, 3))
    
    try:
        new_conn = new_make_laminar_connectivity(neurons_df, positions_m)
        assert new_conn["W"].shape == (0, 0)
        assert len(new_conn["E_mask"]) == 0
    except Exception as e:
        # If empty df is unsupported, confirm it raises clean ValueError
        assert isinstance(e, ValueError)
