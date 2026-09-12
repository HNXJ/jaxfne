"""JAX execution profile programme baseline (24-JAX-01).

Measures, per execution path: first-call (compile-inclusive) wall time,
steady-state wall time, output bytes, and numerical identity across
jit/eager and vmap/loop variants. Run: python scripts/perf/jax_profile_v0424.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jaxfne as jtfne


def _med(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out.V_m)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _cell(model, sim_kwargs, runtime_kwargs, label):
    rt = jtfne.RuntimeConfig(**runtime_kwargs)
    sim = jtfne.simulation(record_sources=True, record_fields=False, runtime=rt, **sim_kwargs)
    t0 = time.perf_counter()
    sig = jtfne.simulate(model, sim)
    jax.block_until_ready(sig.V_m)
    t_first = time.perf_counter() - t0
    t_steady = _med(lambda: jtfne.simulate(model, sim))
    nbytes = int(np.asarray(sig.V_m).nbytes + np.asarray(sig.spikes).nbytes)
    return {
        "label": label,
        "first_s": round(t_first, 4),
        "steady_s": round(t_steady, 4),
        "output_MB": round(nbytes / 1e6, 4),
        "spike_count": int(np.asarray(sig.spikes).sum()),
    }


def main() -> None:
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=64, cell_types={"E": 0.8, "PV": 0.2})
        .cell_type_drives({"E": 10.0, "PV": 10.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    sim_kw = dict(duration_ms=500.0, dt_ms=1.0, seed=3)
    rows = []
    base = dict(recurrent_backend="edge_list", hdp_params={"noise_scale": 0.0})
    model = jtfne.construct(cfg)
    rows.append(_cell(model, sim_kw, {**base, "jit": False}, "baseline/eager"))
    rows.append(_cell(model, sim_kw, {**base, "jit": True}, "baseline/jit"))
    hdp = dict(recurrent_backend="edge_list", enable_hdp=True,
               hdp_params={"K_HDP": 0.01, "K_ctrl": 0.15, "noise_scale": 0.0})
    rows.append(_cell(model, sim_kw, {**hdp, "jit": False}, "hdp/eager"))
    rows.append(_cell(model, sim_kw, {**hdp, "jit": True}, "hdp/jit"))
    # batch variants
    rt_vmap = jtfne.RuntimeConfig(recurrent_backend="edge_list", vmap=True,
                                  hdp_params={"noise_scale": 0.0})
    rt_loop = jtfne.RuntimeConfig(recurrent_backend="edge_list", vmap=False,
                                  hdp_params={"noise_scale": 0.0})
    sim_b = jtfne.simulation(duration_ms=200.0, dt_ms=1.0, seed=3, runtime=rt_vmap)
    t0 = time.perf_counter()
    out_v = model.simulate_batch(sim_b, n_seeds=3)
    jax.block_until_ready(out_v["V_m"])
    t_vmap = time.perf_counter() - t0
    sim_b2 = jtfne.simulation(duration_ms=200.0, dt_ms=1.0, seed=3, runtime=rt_loop)
    t0 = time.perf_counter()
    out_l = model.simulate_batch(sim_b2, n_seeds=3)
    jax.block_until_ready(out_l["V_m"])
    t_loop = time.perf_counter() - t0
    rows.append({"label": "batch/vmap", "first_s": round(t_vmap, 4),
                 "steady_s": None,
                 "output_MB": round(float(np.asarray(out_v["V_m"]).nbytes) / 1e6, 4),
                 "spike_count": int(np.asarray(out_v["spikes"]).sum()),
                 "vmap_loop_identical": bool(np.array_equal(np.asarray(out_v["V_m"]), np.asarray(out_l["V_m"])))})
    rows.append({"label": "batch/loop", "first_s": round(t_loop, 4), "steady_s": None,
                 "output_MB": None, "spike_count": int(np.asarray(out_l["spikes"]).sum())})
    print(json.dumps({"cells": rows}, indent=1))


if __name__ == "__main__":
    main()
