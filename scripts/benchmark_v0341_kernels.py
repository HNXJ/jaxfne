import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jaxfne import (
    spectrolaminar_psd_jax,
    bandpower_jax,
    spectrolaminar_readout_kernel_jax,
    spectrolaminar_similarity_kernel_jax,
    spectrolaminar_similarity_candidates_jax,
    spectrolaminar_similarity_candidates_seeds_jax,
    compile_connection_rules_jax,
    update_stdp_weights_jax,
)

# ----------------- NumPy Baselines -----------------

def spectrolaminar_psd_numpy(signal, fs, freqs):
    n_trials, n_steps, n_contacts = signal.shape
    sample_idx = np.arange(n_steps)
    basis = np.exp(-1j * 2.0 * np.pi * (freqs[:, None] / fs) * sample_idx[None, :])
    spec = np.einsum("fs,tsc->tfc", basis, signal)
    psd = np.mean(np.abs(spec), axis=0) / float(n_steps)
    return psd.astype(np.float32)

def bandpower_numpy(psd, freqs, band_range):
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    mask_float = mask.astype(psd.dtype)
    sum_mask = np.sum(mask_float)
    band_power = np.sum(psd * mask_float[:, None], axis=0) / (sum_mask + 1e-12)
    profile = band_power / (np.max(band_power) + 1e-12)
    return profile

def spectrolaminar_readout_kernel_numpy(psd, freqs, alpha_beta_range, gamma_range):
    depth_sum = np.sum(psd, axis=1, keepdims=True)
    relative_power = psd / (depth_sum + 1e-12)
    ab_profile = bandpower_numpy(psd, freqs, alpha_beta_range)
    gm_profile = bandpower_numpy(psd, freqs, gamma_range)
    return {
        "relative_power": relative_power,
        "alpha_beta": ab_profile,
        "gamma": gm_profile,
    }

def spectrolaminar_similarity_kernel_numpy(alpha_beta, gamma, target_alpha_beta, target_gamma):
    mse_ab = np.mean((alpha_beta - target_alpha_beta) ** 2)
    mse_gamma = np.mean((gamma - target_gamma) ** 2)
    mse_total = mse_ab + mse_gamma
    similarity = 100.0 * np.exp(-3.0 * mse_total)
    return np.clip(similarity, 0.0, 100.0)

def spectrolaminar_similarity_candidates_numpy(alpha_beta, gamma, target_alpha_beta, target_gamma):
    n_cand = alpha_beta.shape[0]
    out = np.zeros(n_cand)
    for i in range(n_cand):
        out[i] = spectrolaminar_similarity_kernel_numpy(
            alpha_beta[i], gamma[i], target_alpha_beta, target_gamma
        )
    return out

def spectrolaminar_similarity_candidates_seeds_numpy(alpha_beta, gamma, target_alpha_beta, target_gamma):
    n_seeds, n_cand = alpha_beta.shape[0], alpha_beta.shape[1]
    out = np.zeros((n_seeds, n_cand))
    for s in range(n_seeds):
        for c in range(n_cand):
            out[s, c] = spectrolaminar_similarity_kernel_numpy(
                alpha_beta[s, c], gamma[s, c], target_alpha_beta, target_gamma
            )
    return out

def compile_connection_rules_numpy(pre_indices, post_indices, probability, seed_int, max_edges, weight_val=1.0):
    np.random.seed(seed_int)
    n_pre = len(pre_indices)
    n_post = len(post_indices)
    
    pre_grid = np.tile(pre_indices[:, None], (1, n_post))
    post_grid = np.tile(post_indices[None, :], (n_pre, 1))
    
    flat_pre = pre_grid.ravel()
    flat_post = post_grid.ravel()
    
    rand_vals = np.random.uniform(0.0, 1.0, size=flat_pre.shape)
    mask = rand_vals < probability
    
    score = np.where(mask, rand_vals, 2.0)
    sorted_idx = np.argsort(score)
    selected_idx = sorted_idx[:max_edges]
    
    valid_mask = score[selected_idx] < 2.0
    
    edge_pre = np.where(valid_mask, flat_pre[selected_idx], -1)
    edge_post = np.where(valid_mask, flat_post[selected_idx], -1)
    edge_weight = np.where(valid_mask, np.full((max_edges,), weight_val), 0.0)
    
    return edge_pre, edge_post, edge_weight

def update_stdp_weights_numpy(W, trace_pre, trace_post, spiked, exc_mask, A_plus, A_minus, plasticity_scale, w_min, w_max):
    post_spike = spiked[:, None]
    pre_spike = spiked[None, :]
    
    dW_ltp = post_spike * trace_pre[None, :] * A_plus
    dW_ltd = pre_spike * trace_post[:, None] * A_minus
    dW = plasticity_scale * (dW_ltp - dW_ltd)
    
    n = W.shape[0]
    update_mask = exc_mask[None, :] & (~np.eye(n, dtype=bool))
    W_next = W + np.where(update_mask, dW, 0.0)
    
    W_next = np.where(exc_mask[None, :], np.clip(W_next, w_min, w_max), W_next)
    W_next = W_next * (1.0 - np.eye(n))
    return W_next

# ----------------- Benchmarking Logic -----------------

def run_benchmark():
    # Detect environment
    python_version = "3.13.7"  # Hardcoded or dynamic
    platform = "darwin"
    jax_version = jax.__version__
    backend = jax.devices()[0].device_kind
    jax_enable_x64 = bool(jax.config.read("jax_enable_x64"))
    
    print(f"Python Version: {python_version}")
    print(f"Platform: {platform}")
    print(f"JAX Version: {jax_version}")
    print(f"Backend: {backend}")
    print(f"JAX Enable X64: {jax_enable_x64}")
    print("-" * 50)
    
    results = []
    
    # 1. spectrolaminar_psd
    print("Benchmarking spectrolaminar_psd...")
    n_trials, n_steps, n_contacts = 10, 2000, 32
    fs = 1000.0
    freqs = np.linspace(1.0, 150.0, 128)
    signal = np.random.randn(n_trials, n_steps, n_contacts).astype(np.float32)
    
    signal_jax = jnp.array(signal)
    freqs_jax = jnp.array(freqs)
    
    # Warmup
    t0 = time.perf_counter()
    res_jax = spectrolaminar_psd_jax(signal_jax, fs, freqs_jax).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    # JAX Steady-state
    times_jax = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = spectrolaminar_psd_jax(signal_jax, fs, freqs_jax).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    # NumPy
    times_np = []
    for _ in range(20):
        t0 = time.perf_counter()
        res_np = spectrolaminar_psd_numpy(signal, fs, freqs)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax) - res_np)))
    speed_ratio = steady_np / steady_jax
    
    results.append({
        "kernel": "spectrolaminar_psd_jax",
        "shape": f"signal: {signal.shape}, freqs: {freqs.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 2. bandpower
    print("Benchmarking bandpower...")
    psd = np.random.rand(128, 32).astype(np.float32)
    band_range = np.array([10.0, 25.0], dtype=np.float32)
    psd_jax = jnp.array(psd)
    band_range_jax = jnp.array(band_range)
    
    t0 = time.perf_counter()
    res_jax = bandpower_jax(psd_jax, freqs_jax, band_range_jax).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = bandpower_jax(psd_jax, freqs_jax, band_range_jax).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(100):
        t0 = time.perf_counter()
        res_np = bandpower_numpy(psd, freqs, band_range)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax) - res_np)))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "bandpower_jax",
        "shape": f"psd: {psd.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 3. spectrolaminar_readout_kernel
    print("Benchmarking spectrolaminar_readout_kernel...")
    ab_range = np.array([10.0, 25.0], dtype=np.float32)
    gm_range = np.array([30.0, 80.0], dtype=np.float32)
    ab_range_jax = jnp.array(ab_range)
    gm_range_jax = jnp.array(gm_range)
    
    t0 = time.perf_counter()
    res_jax = spectrolaminar_readout_kernel_jax(psd_jax, freqs_jax, ab_range_jax, gm_range_jax)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), res_jax)
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), spectrolaminar_readout_kernel_jax(psd_jax, freqs_jax, ab_range_jax, gm_range_jax))
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(100):
        t0 = time.perf_counter()
        res_np = spectrolaminar_readout_kernel_numpy(psd, freqs, ab_range, gm_range)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax["alpha_beta"]) - res_np["alpha_beta"])))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "spectrolaminar_readout_kernel_jax",
        "shape": f"psd: {psd.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 4. spectrolaminar_similarity_kernel
    print("Benchmarking spectrolaminar_similarity_kernel...")
    ab = np.random.rand(32).astype(np.float32)
    gm = np.random.rand(32).astype(np.float32)
    t_ab = np.random.rand(32).astype(np.float32)
    t_gm = np.random.rand(32).astype(np.float32)
    
    ab_jax, gm_jax, t_ab_jax, t_gm_jax = jnp.array(ab), jnp.array(gm), jnp.array(t_ab), jnp.array(t_gm)
    
    t0 = time.perf_counter()
    res_jax = spectrolaminar_similarity_kernel_jax(ab_jax, gm_jax, t_ab_jax, t_gm_jax).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = spectrolaminar_similarity_kernel_jax(ab_jax, gm_jax, t_ab_jax, t_gm_jax).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(100):
        t0 = time.perf_counter()
        res_np = spectrolaminar_similarity_kernel_numpy(ab, gm, t_ab, t_gm)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.abs(float(res_jax) - res_np))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "spectrolaminar_similarity_kernel_jax",
        "shape": f"ab: {ab.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 5. spectrolaminar_similarity_candidates
    print("Benchmarking spectrolaminar_similarity_candidates...")
    ab_c = np.random.rand(100, 32).astype(np.float32)
    gm_c = np.random.rand(100, 32).astype(np.float32)
    ab_c_jax, gm_c_jax = jnp.array(ab_c), jnp.array(gm_c)
    
    t0 = time.perf_counter()
    res_jax = spectrolaminar_similarity_candidates_jax(ab_c_jax, gm_c_jax, t_ab_jax, t_gm_jax).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = spectrolaminar_similarity_candidates_jax(ab_c_jax, gm_c_jax, t_ab_jax, t_gm_jax).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(100):
        t0 = time.perf_counter()
        res_np = spectrolaminar_similarity_candidates_numpy(ab_c, gm_c, t_ab, t_gm)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax) - res_np)))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "spectrolaminar_similarity_candidates_jax",
        "shape": f"ab: {ab_c.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 6. spectrolaminar_similarity_candidates_seeds
    print("Benchmarking spectrolaminar_similarity_candidates_seeds...")
    ab_s = np.random.rand(10, 100, 32).astype(np.float32)
    gm_s = np.random.rand(10, 100, 32).astype(np.float32)
    ab_s_jax, gm_s_jax = jnp.array(ab_s), jnp.array(gm_s)
    
    t0 = time.perf_counter()
    res_jax = spectrolaminar_similarity_candidates_seeds_jax(ab_s_jax, gm_s_jax, t_ab_jax, t_gm_jax).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = spectrolaminar_similarity_candidates_seeds_jax(ab_s_jax, gm_s_jax, t_ab_jax, t_gm_jax).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(20):
        t0 = time.perf_counter()
        res_np = spectrolaminar_similarity_candidates_seeds_numpy(ab_s, gm_s, t_ab, t_gm)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax) - res_np)))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "spectrolaminar_similarity_candidates_seeds_jax",
        "shape": f"ab: {ab_s.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 7. compile_connection_rules
    print("Benchmarking compile_connection_rules...")
    pre_indices = np.arange(100)
    post_indices = np.arange(100)
    prob = 0.3
    max_edges = 2000
    key = jax.random.PRNGKey(42)
    
    pre_indices_jax = jnp.array(pre_indices)
    post_indices_jax = jnp.array(post_indices)
    
    t0 = time.perf_counter()
    res_jax = compile_connection_rules_jax(pre_indices_jax, post_indices_jax, prob, key, max_edges)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), res_jax)
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), compile_connection_rules_jax(pre_indices_jax, post_indices_jax, prob, key, max_edges))
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(50):
        t0 = time.perf_counter()
        res_np = compile_connection_rules_numpy(pre_indices, post_indices, prob, 42, max_edges)
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax[0]) - res_np[0])))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "compile_connection_rules_jax",
        "shape": f"pre: {pre_indices.shape}, post: {post_indices.shape}, max_edges: {max_edges}",
        "dtype": "int32/float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # 8. update_stdp_weights
    print("Benchmarking update_stdp_weights...")
    n_neurons = 200
    W = np.random.rand(n_neurons, n_neurons).astype(np.float32)
    trace_pre = np.random.rand(n_neurons).astype(np.float32)
    trace_post = np.random.rand(n_neurons).astype(np.float32)
    spiked = (np.random.rand(n_neurons) > 0.8)
    exc_mask = (np.random.rand(n_neurons) > 0.2)
    
    W_jax, trace_pre_jax, trace_post_jax, spiked_jax, exc_mask_jax = (
        jnp.array(W), jnp.array(trace_pre), jnp.array(trace_post), jnp.array(spiked), jnp.array(exc_mask)
    )
    
    t0 = time.perf_counter()
    res_jax = update_stdp_weights_jax(
        W_jax, trace_pre_jax, trace_post_jax, spiked_jax, exc_mask_jax,
        A_plus=0.01, A_minus=0.012, plasticity_scale=1.0, w_min=0.0, w_max=1.5
    ).block_until_ready()
    warmup_time = time.perf_counter() - t0
    
    times_jax = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = update_stdp_weights_jax(
            W_jax, trace_pre_jax, trace_post_jax, spiked_jax, exc_mask_jax,
            A_plus=0.01, A_minus=0.012, plasticity_scale=1.0, w_min=0.0, w_max=1.5
        ).block_until_ready()
        times_jax.append(time.perf_counter() - t0)
    steady_jax = np.mean(times_jax)
    
    times_np = []
    for _ in range(50):
        t0 = time.perf_counter()
        res_np = update_stdp_weights_numpy(
            W, trace_pre, trace_post, spiked, exc_mask,
            A_plus=0.01, A_minus=0.012, plasticity_scale=1.0, w_min=0.0, w_max=1.5
        )
        times_np.append(time.perf_counter() - t0)
    steady_np = np.mean(times_np)
    
    tolerance = float(np.max(np.abs(np.array(res_jax) - res_np)))
    speed_ratio = steady_np / steady_jax
    results.append({
        "kernel": "update_stdp_weights_jax",
        "shape": f"W: {W.shape}",
        "dtype": "float32",
        "warmup_time_s": warmup_time,
        "steady_state_jax_s": steady_jax,
        "steady_state_numpy_s": steady_np,
        "speed_ratio": speed_ratio,
        "tolerance_err": tolerance,
    })
    
    # Save results to a file
    output_path = "internal_docs/receipts/v0341_kernel_benchmarks.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "environment": {
                "python_version": python_version,
                "platform": platform,
                "jax_version": jax_version,
                "backend": backend,
                "jax_enable_x64": jax_enable_x64
            },
            "benchmarks": results
        }, f, indent=2)
        
    print("\nBenchmark completed successfully!")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
