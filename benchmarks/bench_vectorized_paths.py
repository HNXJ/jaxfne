import sys
import time
import platform
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

# Import the new vectorized versions from the current codebase
from jaxfne.fields.proxy import make_laminar_connectivity as new_make_laminar_connectivity
from jaxfne.fields.proxy import teaching_control_spectrolaminar_resonance_source as new_resonance_source

# Define the baseline functions manually to be 100% self-contained
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


def run_connectivity_benchmarks():
    print("\n=== BENCHMARKING: make_laminar_connectivity ===")
    for n in [50, 200, 500]:
        print(f"\nPopulation Size (N): {n}")
        neurons_df = pd.DataFrame({
            "area": ["V1"] * n,
            "layer": ["L4"] * n,
            "cell_type": ["E" if i % 4 != 0 else "I" for i in range(n)]
        })
        positions_m = np.random.uniform(0, 0.01, size=(n, 3))
        
        # Warmup
        _ = old_make_laminar_connectivity(neurons_df, positions_m)
        _ = new_make_laminar_connectivity(neurons_df, positions_m)
        
        # Benchmark Old
        old_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = old_make_laminar_connectivity(neurons_df, positions_m)
            old_times.append(time.perf_counter() - t0)
        old_median = np.median(old_times) * 1000.0
        
        # Benchmark New
        new_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = new_make_laminar_connectivity(neurons_df, positions_m)
            new_times.append(time.perf_counter() - t0)
        new_median = np.median(new_times) * 1000.0
        
        speedup = old_median / new_median
        print(f"  Old (O(N^2) Loop): {old_median:7.2f} ms")
        print(f"  New (Vectorized):  {new_median:7.2f} ms")
        print(f"  Speedup:           {speedup:7.1f}x")

def run_resonance_benchmarks():
    print("\n=== BENCHMARKING: teaching_control_spectrolaminar_resonance_source ===")
    for n, steps in [(100, 500)]:
        print(f"\nPopulation Size (N): {n}, n_steps: {steps}")
        neurons = {
            "area": ["V1"] * n,
            "layer": ["L1", "L2", "L3", "L4", "L5", "L6"] * (n // 6 + 1)
        }
        neurons["area"] = neurons["area"][:n]
        neurons["layer"] = neurons["layer"][:n]
        
        # Warmup
        _ = old_resonance_source(steps, 0.1, neurons)
        _ = new_resonance_source(neurons, steps, 0.1)
        
        # Benchmark Old
        old_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = old_resonance_source(steps, 0.1, neurons)
            old_times.append(time.perf_counter() - t0)
        old_median = np.median(old_times) * 1000.0
        
        # Benchmark New
        new_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = new_resonance_source(neurons, steps, 0.1)
            new_times.append(time.perf_counter() - t0)
        new_median = np.median(new_times) * 1000.0
        
        speedup = old_median / new_median
        print(f"  Old (Loop):        {old_median:7.2f} ms")
        print(f"  New (Vectorized):  {new_median:7.2f} ms")
        print(f"  Speedup:           {speedup:7.1f}x")

def run_jit_adam_benchmarks():
    print("\n=== BENCHMARKING: Inner Adam JIT-per-candidate ===")
    
    W_shape = (5, 5)
    W_flat_size = 25
    
    def toy_simulate(W_flat):
        W = W_flat.reshape(W_shape)
        x = jnp.arange(5, dtype=jnp.float32)
        def step(carry, _):
            carry = jnp.tanh(jnp.dot(W, carry) + 0.1)
            return carry, carry
        _, y = jax.lax.scan(step, x, None, length=5)
        return jnp.sum(y**2)
    
    # OLD PATH: eager value_and_grad call inside step loop (un-jitted)
    def run_old_inner_loop(W_init, steps):
        current_W = W_init
        opt = optax.adam(0.01)
        opt_state = opt.init(W_init)
        
        for step in range(steps):
            def loss_fn(W):
                return toy_simulate(W)
            loss_val, grads = jax.value_and_grad(loss_fn)(current_W)
            updates, opt_state = opt.update(grads, opt_state)
            current_W = optax.apply_updates(current_W, updates)
        return current_W
        
    # NEW PATH: JIT compilation once before loop, then JIT execution
    def run_new_inner_loop(W_init, steps):
        current_W = W_init
        opt = optax.adam(0.01)
        opt_state = opt.init(W_init)
        
        def loss_fn(W):
            return toy_simulate(W)
            
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        
        for step in range(steps):
            loss_val, grads = grad_fn(current_W)
            updates, opt_state = opt.update(grads, opt_state)
            current_W = optax.apply_updates(current_W, updates)
        return current_W

    W_init = jax.random.normal(jax.random.PRNGKey(0), (W_flat_size,))
    
    # 1. Warmup and compile-time detection
    # OLD
    t0 = time.perf_counter()
    _ = run_old_inner_loop(W_init, 1)
    old_warmup_and_compile = (time.perf_counter() - t0) * 1000.0
    
    # NEW
    t0 = time.perf_counter()
    _ = run_new_inner_loop(W_init, 1)
    new_warmup_and_compile = (time.perf_counter() - t0) * 1000.0
    
    print(f"Compilation / Warmup Time (1 step):")
    print(f"  Old Path Compilation (Eager trace): {old_warmup_and_compile:7.2f} ms")
    print(f"  New Path Compilation (JIT compile): {new_warmup_and_compile:7.2f} ms")

    # 2. Execution time over repeats (warmup fully excluded)
    for steps in [5]:
        print(f"\nInner Steps per Candidate: {steps}")
        
        old_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = run_old_inner_loop(W_init, steps)
            old_times.append(time.perf_counter() - t0)
        old_median = np.median(old_times) * 1000.0
        
        new_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = run_new_inner_loop(W_init, steps)
            new_times.append(time.perf_counter() - t0)
        new_median = np.median(new_times) * 1000.0
        
        speedup = old_median / new_median
        print(f"  Old (Eager tracing): {old_median:7.2f} ms")
        print(f"  New (JIT-refinement): {new_median:7.2f} ms")
        print(f"  Speedup:            {speedup:7.1f}x")

if __name__ == "__main__":
    print(f"Baseline SHA:  c6ff021bdbbe9b0e6fac064241361c0daae876b7 (origin/main)")
    print(f"Optimized SHA: 3ff0e173e86380996cc024cc06886495ac615965 (agy)")
    print(f"JAX Version:   {jax.__version__}")
    print(f"Platform:      {platform.platform()}")
    run_connectivity_benchmarks()
    run_resonance_benchmarks()
    run_jit_adam_benchmarks()
