"""One-shot probes for simplification review R-01 (not a public API).

Measures:
  F — per-edge tau decay: resolve+exp vs class-table gather+exp
  G — scan output arity with/without record_* flags (allocation proxy)
  H — sequential run_trials vs batched vmap simulate_batch
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import EdgeList, resolve_edge_tau_ms


def _bench(fn, *, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def probe_tau_decay(E: int, *, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    pre = jax.random.randint(key, (E,), 0, 500, dtype=jnp.int32)
    post = jax.random.randint(jax.random.split(key)[0], (E,), 0, 500, dtype=jnp.int32)
    weight = jax.random.uniform(jax.random.split(key)[1], (E,), dtype=jnp.float32)
    ri = jax.random.randint(jax.random.split(key)[0], (E,), 0, 2, dtype=jnp.int32)
    tau = jax.random.uniform(jax.random.split(key)[0], (E,), minval=1.0, maxval=20.0)
    edges = EdgeList(
        pre=pre,
        post=post,
        weight=weight,
        receptor_index=ri,
        tau_ms=tau,
        tau_storage="per_edge",
    )
    dt = jnp.asarray(0.5, dtype=jnp.float32)
    jdtype = jnp.float32

    def per_edge():
        tau_ms = jnp.maximum(resolve_edge_tau_ms(edges, jdtype), jnp.asarray(1e-6, dtype=jdtype))
        return jnp.exp(-dt / tau_ms)

    # Compact: unique tau classes (simulate small mechanism table)
    tau_host = np.asarray(tau)
    classes, inv = np.unique(tau_host, return_inverse=True)
    table = jnp.asarray(classes, dtype=jdtype)
    inv_j = jnp.asarray(inv, dtype=jnp.int32)

    def class_gather():
        tau_ms = jnp.maximum(table[inv_j], jnp.asarray(1e-6, dtype=jdtype))
        return jnp.exp(-dt / tau_ms)

    t_per, decay_per = _bench(per_edge)
    t_cls, decay_cls = _bench(class_gather)
    max_abs = float(jnp.max(jnp.abs(decay_per - decay_cls)))
    return {
        "E": E,
        "n_tau_classes": int(classes.size),
        "max_abs_diff": max_abs,
        "t_per_edge_s": t_per,
        "t_class_gather_s": t_cls,
        "ratio_per_over_class": t_per / t_cls if t_cls else None,
    }


def probe_scan_recording():
    from jaxfne.emitters import IzhikevichParams, simulate_edge_recurrent_izhikevich

    n, e = 64, 256
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="x",
    )
    key = jax.random.PRNGKey(0)
    pre = jax.random.randint(key, (e,), 0, n, dtype=jnp.int32)
    post = jax.random.randint(jax.random.split(key)[0], (e,), 0, n, dtype=jnp.int32)
    edges = EdgeList(
        pre=pre,
        post=post,
        weight=jnp.ones((e,), dtype=jdtype) * 0.1,
        receptor_index=jnp.zeros((e,), dtype=jnp.int32),
        tau_ms=jnp.full((e,), 2.0, dtype=jdtype),
    )
    n_steps = 200

    def run(**flags):
        tracemalloc.start()
        _, _, _, _ = simulate_edge_recurrent_izhikevich(
            params,
            edges,
            n_steps,
            0.5,
            jax.random.PRNGKey(1),
            record_edge_current=flags.get("record_edge_current", False),
            record_current_trace=flags.get("record_current_trace", False),
            record_u_trace=flags.get("record_u_trace", False),
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    minimal_peak = run()
    full_peak = run(
        record_edge_current=True,
        record_current_trace=True,
        record_u_trace=True,
    )
    return {
        "n_neurons": n,
        "n_edges": e,
        "n_steps": n_steps,
        "peak_bytes_minimal_scan_outputs": minimal_peak,
        "peak_bytes_full_record_scan_outputs": full_peak,
        "peak_ratio_full_over_minimal": full_peak / minimal_peak if minimal_peak else None,
        "note": "tracemalloc peak during Python call; JAX may defer allocation",
    }


def probe_trials_vmap(n_trials: int = 8):
    from dataclasses import replace

    from jaxfne._signals import TrialBatch, TrialSpec

    cfg = jtfne.suite2_net1_config(seed=1, n=8, duration_ms=20.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=0.5, seed=0)

    trials = tuple(TrialSpec(trial_id=f"t{i}", seed=100 + i) for i in range(n_trials))
    batch = TrialBatch(batch_id="probe", trials=trials)

    def sequential():
        return model.run_trials(batch, sim)

    def python_loop():
        outs = []
        for t in trials:
            outs.append(model.simulate(replace(sim, seed=t.seed)))
        return outs

    def batched():
        return model.simulate_batch(sim, n_seeds=n_trials, seed=100)

    t_seq, _ = _bench(sequential, warmup=1, repeats=3)
    t_loop, _ = _bench(python_loop, warmup=1, repeats=3)
    t_bat, _ = _bench(batched, warmup=1, repeats=3)
    return {
        "n_trials": n_trials,
        "t_run_trials_s": t_seq,
        "t_python_simulate_loop_s": t_loop,
        "t_simulate_batch_s": t_bat,
        "ratio_run_trials_over_batch": t_seq / t_bat if t_bat else None,
        "note": "simulate_batch uses vmap over PRNG keys; run_trials is sequential; APIs differ",
    }


def main():
    out = {
        "probe_id": "simplification_r01",
        "F_tau_decay": [probe_tau_decay(e) for e in (1000, 10000, 100000)],
        "G_scan_recording": probe_scan_recording(),
        "H_trials": probe_trials_vmap(8),
    }
    path = Path(__file__).resolve().parents[2] / "artifacts/audit/simplification_r01_probe_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
