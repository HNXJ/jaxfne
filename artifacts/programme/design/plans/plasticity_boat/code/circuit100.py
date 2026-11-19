"""100-neuron LIF circuit controls, plasticity-boat lane B (EX-25).

Feasibility and negative controls only: no plasticity, no competition. Development starts only;
no held-out starts are used here. Gates: build (B0), zero drive (B1), stationary open loop (B2),
runtime (B3), closed-loop smoke (B4, refused unless B1 passed).

Choices fixed before the run (the packet does not state them):
  - Motor groups: the first 20 E neurons by ascending index, cut into consecutive blocks of 5
    for +x, -x, +y, -y.
  - Non-motor E neurons: fixture distance drive g * e / (e + d0), g = 24, e = |p|. This follows the
    encoder paragraph; it overrides the Step 2 wording "zero for the rest".
  - Motor E neurons: directional current only (encoder paragraph). PV neurons: no external drive. b = 0.
  - B1 "no external input": a zero-amplitude stimulus schedule on every neuron (same code path).
  - B2: the scalar distance drive held at e = 5 on every E neuron (motor included), PV zero, boat
    fixed at p0 = (5, 0), K = 600 (Plant episode length; the packet gives no K for B2).
  - B4 distance at K = |p_K|, after K intervals.

Usage (PYTHONPATH=<repo root>, PYTHONDONTWRITEBYTECODE=1), in this order:
  python3 -u circuit100.py build --out DIR
  python3 -u circuit100.py b1 --out DIR
  python3 -u circuit100.py b2 --out DIR
  python3 -u circuit100.py b3 --out DIR
  python3 -u circuit100.py b4 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne import paradigm as sp
from jaxfne._construct_lif import lif_edge_arrays

N = 100
GROUP = 5
CFG = dict(n_dim=2, dt=0.02, damping=0.5)
SEED = 0
DT_MS = 1.0
CONTROL_MS = 20.0
F0 = 5.0
G = 24.0
D0 = 1.0
R_STAR = 5.0
P0 = (5.0, 0.0)
K_EPISODE = 600
K_B1 = 100
K_B2 = K_EPISODE
K_B3 = 60
K_B4 = 60
P0_B4 = (-3.54, 3.54)
B1_TOL = 1e-6
RATE_MAX_HZ = 100.0
WALL_LIMIT_S = 30 * 60
N_STRATEGIES = 8
N_STARTS = 20
HERE = os.path.dirname(os.path.abspath(__file__))


def build_model():
    """The 100-neuron LIF circuit: same construction as the explore harness, N = 100."""

    def conn(src, dst, mech):
        return nt.InterConnection(
            source_layer="L", source_neuron_type=src, target_layer="L",
            target_neuron_type=dst, mechanism=mech,
            static=nt.StaticParams(dT_ms=5.0))

    tensor = nt.NeuronalTensor(
        name="crazy-sailor-circuit100",
        areas=[nt.Area(name="A", layers=[nt.Layer(
            name="L", n_neurons=N,
            neuron_types=[nt.NeuronType.make("E", fraction=0.8),
                          nt.NeuronType.make("PV", fraction=0.2)])],
            inter_connections=[
                conn("E", "E", "AMPA"), conn("E", "PV", "AMPA"),
                conn("PV", "E", "GABA_A"), conn("PV", "PV", "GABA_A")])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=SEED, dt_ms=1.0, emitter="lif", duration_ms=20.0))


def git_sha():
    out = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def layout_of(model):
    md = model.static["neuron_metadata"]
    if len(md) != N:
        raise ValueError(f"model has {len(md)} neurons, expected {N}")
    ctype = [str(row["cell_type"]) for row in md]
    e_idx = [i for i, c in enumerate(ctype) if c == "E"]
    pv_idx = [i for i, c in enumerate(ctype) if c == "PV"]
    if len(e_idx) + len(pv_idx) != N or len(e_idx) < 4 * GROUP:
        raise ValueError(f"need {N} E/PV neurons with >= {4 * GROUP} E, got {len(e_idx)} E, {len(pv_idx)} PV")
    groups = [e_idx[k * GROUP:(k + 1) * GROUP] for k in range(4)]
    motor = [i for g in groups for i in g]
    motor_set = set(motor)
    g_vec = np.zeros(N, np.float32)
    g_vec[e_idx] = G
    return {
        "ctype": ctype, "E": e_idx, "PV": pv_idx, "groups": groups, "motor": motor,
        "nonmotor_E": [i for i in e_idx if i not in motor_set], "g_vec": g_vec,
    }


def _u(z):
    zp = max(float(z), 0.0)
    return zp / (zp + D0)


def current_vector(mode, p, e, lay):
    """Per-neuron external current (N,) float32. Modes: zero | fixture_e5 | directional."""
    if mode == "zero":
        return np.zeros(N, np.float32)
    enc = sp.SailorTonicEncoder(b=np.zeros(N, np.float32), g=lay["g_vec"], d0=D0)
    if mode == "fixture_e5":
        return np.asarray(sp.sailor_tonic_current(enc, 5.0), np.float32)
    if mode != "directional":
        raise ValueError(f"unknown mode {mode!r}")
    cur = np.array(sp.sailor_tonic_current(enc, e), dtype=np.float32)  # writable copy of a JAX array
    gpx, gmx, gpy, gmy = lay["groups"]
    cur[gpx] = G * _u(-p[0])
    cur[gmx] = G * _u(p[0])
    cur[gpy] = G * _u(-p[1])
    cur[gmy] = G * _u(p[1])
    return cur


def run_loop(model, lay, mode, *, p0, n_intervals, seed=SEED, hold_boat=False):
    """Closed loop as jaxfne sailor_neural_rollout: the force decoded from interval k acts in k+1.

    hold_boat=True keeps the boat at p0 (no force is applied to it).
    """
    cfg = sp.SailorConfig(**CFG)
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    state = sp.sailor_reset(cfg, jax.random.PRNGKey(seed), p0=np.asarray(p0, np.float32))
    cont = None
    pending = jnp.zeros((2,), sp._SAILOR_DTYPE)
    rec = {"positions": [np.asarray(state.p, np.float64).tolist()], "errors": [], "counts": [],
           "actions": [], "motor_current": [], "wall_s": []}
    spike_total = np.zeros(N, np.int64)
    spike_max = 0.0
    for _ in range(n_intervals):
        t0 = time.perf_counter()
        p_np = np.asarray(state.p, np.float64)
        e = float(jnp.linalg.norm(state.p - cfg.target))
        cur = current_vector(mode, p_np, e, lay)
        sched = jtfne.stimulus_schedule(
            [{"onset_ms": 0.0, "duration_ms": CONTROL_MS, "amplitude": float(c), "target_indices": [i]}
             for i, c in enumerate(cur)],
            N,
        )
        sim = jtfne.Simulation(duration_ms=CONTROL_MS, dt_ms=DT_MS, seed=seed, runtime=rt)
        if cont is None:
            sig, cont = model.simulate(sim, paradigm=sched, return_state=True)
        else:
            sig, cont = model.simulate(sim, paradigm=sched, continuation=cont, return_state=True)
        spk = np.asarray(sig.spikes)
        c4 = np.array([spk[:, g].sum() for g in lay["groups"]], np.int64)
        spike_total += spk.sum(axis=0).astype(np.int64)
        spike_max = max(spike_max, float(spk.max()))
        if not hold_boat:
            state, _ = sp.sailor_step(cfg, state, pending)
        rec["actions"].append(np.asarray(pending, np.float64).tolist())
        pending = sp.sailor_decode(c4, F0)
        rec["errors"].append(e)
        rec["counts"].append(c4.tolist())
        rec["motor_current"].append([cur[g].tolist() for g in lay["groups"]])
        rec["positions"].append(np.asarray(state.p, np.float64).tolist())
        rec["wall_s"].append(time.perf_counter() - t0)
    rec["spike_total_per_neuron"] = spike_total.tolist()
    rec["spike_max"] = spike_max
    return rec


def header(gate, lay):
    return {
        "gate": gate, "git_sha": git_sha(), "timestamp_utc": utc_now(),
        "config": {
            "n_neurons": N, "sailor": CFG, "seed": SEED, "neural_dt_ms": DT_MS,
            "control_ms": CONTROL_MS, "f0": F0, "g": G, "d0": D0, "p_star": [0.0, 0.0],
            "motor_order": ["+x", "-x", "+y", "-y"], "motor_groups": lay["groups"],
            "encoder": "directional Option A (provisional); non-motor E fixture distance drive; PV none; b = 0",
        },
    }


def write_json(path, payload):
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def require_b1(out):
    path = os.path.join(out, "B1_zero_drive.json")
    if not os.path.exists(path):
        raise SystemExit("B1 JSON missing: run b1 first; B2 to B4 are not run")
    with open(path) as fh:
        b1 = json.load(fh)
    if b1.get("pass") is not True:
        raise SystemExit("B1 did not pass (boat moved): B2 to B4 are not run")


def cmd_build(out):
    model = build_model()
    lay = layout_of(model)
    edges = lif_edge_arrays(model)
    pre = np.asarray(edges["pre"]).astype(np.int64).ravel()
    post = np.asarray(edges["post"]).astype(np.int64).ravel()
    ct = lay["ctype"]
    pairs = {}
    for p_, q_ in zip(pre, post):
        key = f"{ct[p_]}->{ct[q_]}"
        pairs[key] = pairs.get(key, 0) + 1
    payload = header("B0_build", lay)
    payload.update({
        "n_neurons": len(ct), "n_E": len(lay["E"]), "n_PV": len(lay["PV"]),
        "E_indices": lay["E"], "PV_indices": lay["PV"],
        "motor_groups": {"+x": lay["groups"][0], "-x": lay["groups"][1],
                         "+y": lay["groups"][2], "-y": lay["groups"][3]},
        "n_edges": int(pre.size), "edge_pair_counts": pairs,
        "n_floored_to_one_step": edges.get("n_floored_to_one_step"),
    })
    write_json(os.path.join(out, "B0_build.json"), payload)
    print(f"B0 n_E={payload['n_E']} n_PV={payload['n_PV']} n_edges={payload['n_edges']} pairs={pairs}")
    print(f"B0 motor_groups={payload['motor_groups']}")
    return 0


def cmd_b1(out):
    model = build_model()
    lay = layout_of(model)
    rec = run_loop(model, lay, "zero", p0=P0, n_intervals=K_B1)
    errs = np.asarray(rec["errors"], np.float64)
    max_dev = float(np.max(np.abs(errs - R_STAR)))
    pos = np.asarray(rec["positions"], np.float64)
    passed = max_dev < B1_TOL
    payload = header("B1_zero_drive", lay)
    payload.update({
        "pass": bool(passed), "criterion": "max|error - 5| < 1e-6, zero external input, p0 = (5, 0), K = 100",
        "K": K_B1, "p0": list(P0), "max_abs_error_dev": max_dev,
        "max_abs_position_dev": float(np.max(np.abs(pos - np.asarray(P0)))), **rec,
    })
    write_json(os.path.join(out, "B1_zero_drive.json"), payload)
    print(f"B1 pass={passed} max_abs_error_dev={max_dev:.3e} motor_counts_total={int(np.sum(rec['counts']))}")
    return 0 if passed else 1


def cmd_b2(out):
    require_b1(out)
    model = build_model()
    lay = layout_of(model)
    rec = run_loop(model, lay, "fixture_e5", p0=P0, n_intervals=K_B2, hold_boat=True)
    T_s = K_B2 * CONTROL_MS / 1000.0
    rates = np.asarray(rec["spike_total_per_neuron"], np.float64) / T_s
    rE = rates[lay["E"]]
    rPV = rates[lay["PV"]]
    finite = bool(np.all(np.isfinite(rates)) and np.all(np.isfinite(np.asarray(rec["errors"], np.float64)))
                  and np.all(np.isfinite(np.asarray(rec["positions"], np.float64))))
    in_range = bool(np.all((rE >= 0.0) & (rE <= RATE_MAX_HZ)))
    passed = finite and in_range
    payload = header("B2_stationary", lay)
    payload.update({
        "pass": bool(passed),
        "criterion": "all E mean rates in [0, 100] Hz and no NaN; boat fixed at p0 = (5, 0), e = 5 held",
        "K": K_B2, "T_s": T_s, "p0": list(P0), "finite": finite, "E_rate_in_range": in_range,
        "E_rate_Hz": rE.tolist(), "PV_rate_Hz": rPV.tolist(),
        "E_rate_mean_Hz": float(rE.mean()), "E_rate_min_Hz": float(rE.min()), "E_rate_max_Hz": float(rE.max()),
        "PV_rate_mean_Hz": float(rPV.mean()), "PV_rate_min_Hz": float(rPV.min()), "PV_rate_max_Hz": float(rPV.max()),
        "wall_total_s": float(np.sum(rec["wall_s"])), **rec,
    })
    write_json(os.path.join(out, "B2_stationary.json"), payload)
    print(f"B2 pass={passed} finite={finite} E_rate mean={rE.mean():.4f} min={rE.min():.4f} "
          f"max={rE.max():.4f} Hz PV_rate mean={rPV.mean():.4f} Hz wall_total_s={np.sum(rec['wall_s']):.2f}")
    return 0 if passed else 1


def cmd_b3(out):
    require_b1(out)
    model = build_model()
    lay = layout_of(model)
    rec = run_loop(model, lay, "directional", p0=P0, n_intervals=K_B3)
    w = np.asarray(rec["wall_s"], np.float64)
    cold, second = float(w[0]), float(w[1])
    steady_median = float(np.median(w[2:]))
    steady_mean = float(np.mean(w[2:]))
    per_run_steady = K_EPISODE * steady_median
    per_run_incl_compile = cold + second + (K_EPISODE - 2) * steady_median
    grid_s = N_STRATEGIES * N_STARTS * per_run_incl_compile
    passed = per_run_incl_compile < WALL_LIMIT_S
    payload = header("B3_runtime", lay)
    payload.update({
        "pass": bool(passed),
        "criterion": "one 600-interval run projected under 30 min (incl. first-interval compile)",
        "timed_intervals": K_B3, "first_interval_s": cold, "second_interval_s": second,
        "steady_median_s": steady_median, "steady_mean_s": steady_mean,
        "projected_600_steady_s": per_run_steady,
        "projected_600_incl_compile_s": per_run_incl_compile,
        "projected_grid_8x20_s": grid_s, "grid_runs": N_STRATEGIES * N_STARTS,
        "limit_s": WALL_LIMIT_S, **{k: rec[k] for k in ("wall_s", "errors", "counts", "positions")},
    })
    write_json(os.path.join(out, "B3_runtime.json"), payload)
    print(f"B3 pass={passed} first={cold:.3f}s second={second:.3f}s steady_median={steady_median:.4f}s "
          f"proj_600_incl_compile={per_run_incl_compile:.1f}s proj_grid={grid_s / 3600:.2f}h")
    return 0 if passed else 1


def cmd_b4(out):
    require_b1(out)
    model = build_model()
    lay = layout_of(model)
    rec = run_loop(model, lay, "directional", p0=P0_B4, n_intervals=K_B4)
    pos = np.asarray(rec["positions"], np.float64)
    dist_K = float(np.linalg.norm(pos[-1]))
    payload = header("B4_smoke", lay)
    payload.update({
        "smoke_not_gate": True, "K": K_B4, "p0": list(P0_B4),
        "first10_positions": pos[:10].tolist(), "distance_at_K": dist_K,
        "last_pre_step_error": float(rec["errors"][-1]), **rec,
    })
    write_json(os.path.join(out, "B4_smoke.json"), payload)
    print(f"B4 K={K_B4} distance_at_K={dist_K:.6f} first10={np.round(pos[:10], 4).tolist()}")
    return 0


GATES = {"build": cmd_build, "b1": cmd_b1, "b2": cmd_b2, "b3": cmd_b3, "b4": cmd_b4}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=sorted(GATES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    return int(GATES[args.gate](args.out) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
