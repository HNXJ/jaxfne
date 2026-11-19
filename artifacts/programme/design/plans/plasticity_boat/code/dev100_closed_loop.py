"""Plasticity boat, dev-100 closed loop: P1 gate, N1 negative control, T05 sensitivity.

Development starts only (numpy default_rng(2025)). The loop copies paradigm.sailor_neural_rollout
(contract m0_contract.md section 12.3.4) with two changes:

- the per-interval drive is ``drive_fn(p, v)``, computed from the boat state (p, v);
- the motor groups are four groups of five E neurons: +x 0-4, -x 5-9, +y 10-14, -y 15-19. A group
  count is the spike count summed over its five neurons; the decoder is unchanged.

Frozen criteria: artifacts/programme/design/plans/plasticity_boat/criteria_v1_frozen.md.
Conditions:

  P1   tau = 0.1 s, position plus velocity   (primary gate)
  N1   tau = 0,     position only           (negative control, not a gate)
  T05  tau = 0.5 s, position plus velocity   (sensitivity, reported, not a gate)

Gate P1: PASS iff all 20 starts have fixation occupancy >= 0.9, where occupancy is the fraction
of errors[480:600] with error <= 1.0. Evidence goes only to artifacts/_temp/plasticity_boat/dev100/.

Usage (PYTHONPATH = repository root; run by path with /usr/local/bin/python3):
  dev100_closed_loop.py smoke
  dev100_closed_loop.py run --condition P1 --start 0
  dev100_closed_loop.py summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import paradigm as sp

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
PLAN_DIR = HERE.parent
RESULTS = REPO / "artifacts" / "_temp" / "plasticity_boat" / "dev100"
CRITERIA = PLAN_DIR / "criteria_v1_frozen.md"
MODEL_FILE = HERE / "dev100_model.py"
CRITERIA_SHA256 = "d57fd9810732945e8f592e88e65fccf1b65b894e737ac1ed0f745e46a44037ee"

N = 100
E_IDX = np.arange(0, 80)
PV_IDX = np.arange(80, 100)
MOTOR_IDX = np.arange(0, 20)
NON_MOTOR_E_IDX = np.arange(20, 80)
GROUP_IDX = (np.arange(0, 5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20))
GROUP_NAMES = ("+x", "-x", "+y", "-y")
GROUP_SIZE = 5
CFG = dict(n_dim=2, dt=0.02, damping=0.5)  # plant, criteria "Plant"
G = 24.0  # distance and directional drive gain, criteria "Model (dev-100)"
D0 = 1.0  # e / (e + d0) and u(z) saturation constant, criteria "Model (dev-100)"
F0 = 5.0  # decoder gain, criteria "Model (dev-100)"
DT_MS = 1.0  # neural step
CONTROL_MS = 20.0  # control interval
K = 600  # episode length in control intervals
SEED = 0
R = 5.0  # start radius
RADIUS = 1.0  # fixation radius
OCC_MIN = 0.9  # fixation occupancy gate
DEV_RNG_SEED = 2025
N_DEV = 20
CHECK_INTERVALS = (0, 30, 60, 120, 240, 480, 599)
DRIVE_TOL = 1e-4  # float32 tolerance for analytic drive values
DT = jnp.float32
CONDITIONS = {
    "P1": {"tau": 0.1, "dir": "P1_pos_vel", "role": "primary gate"},
    "N1": {"tau": 0.0, "dir": "N1_pos_only", "role": "negative control, not a gate"},
    "T05": {"tau": 0.5, "dir": "P1_tau05", "role": "sensitivity, reported, not a gate"},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_criteria() -> str:
    got = sha256_file(CRITERIA)
    if got != CRITERIA_SHA256:
        raise SystemExit(f"criteria sha256 {got} != frozen {CRITERIA_SHA256}")
    return got


def provenance() -> dict[str, Any]:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return {
        "git_head": head,
        "code": str(Path(__file__).resolve().relative_to(REPO)),
        "code_sha256": sha256_file(Path(__file__)),
        "model": str(MODEL_FILE.relative_to(REPO)),
        "model_sha256": sha256_file(MODEL_FILE),
        "criteria": str(CRITERIA.relative_to(REPO)),
        "criteria_sha256": check_criteria(),
        "criteria_sha256_expected": CRITERIA_SHA256,
    }


def write_json(path: Path, obj: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
    tmp.replace(path)


def build_model():
    """dev-100 network from dev100_model.py (declared drive 0.0, E 0-79, PV 80-99)."""
    sys.path.insert(0, str(HERE))
    import dev100_model

    return dev100_model.build_dev100()


def check_layout(model) -> None:
    types = [row["cell_type"] for row in model.static["neuron_metadata"]]
    expected = ["E"] * 80 + ["PV"] * 20
    if types != expected:
        raise RuntimeError(f"dev-100 layout {types} != {expected}")
    if len(types) != N:
        raise RuntimeError(f"dev-100 has {len(types)} neurons, expected {N}")


def dev_starts() -> tuple[np.ndarray, np.ndarray]:
    theta = np.random.default_rng(DEV_RNG_SEED).uniform(0.0, 2.0 * np.pi, N_DEV)
    p0 = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return theta, p0


def make_drive(tau: float, g: float = G, d0: float = D0) -> Callable[[Any, Any], jax.Array]:
    """Drive function (p, v) -> current (N,). tau = 0 is position only; signal s = p + tau * v.

    Non-motor E [20-79]: g * e / (e + d0), e = |p|. Motor E: each group of five gets g * u(z),
    u(z) = max(0, z) / (max(0, z) + d0), z = (-s_x, +s_x, -s_y, +s_y) for (+x, -x, +y, -y).
    PV [80-99]: 0. Built-in tonic drive is zero in the model (declared drive 0.0).
    """
    tau = float(tau)

    def drive(p, v):
        p = jnp.asarray(p, DT)
        v = jnp.asarray(v, DT)
        s = p if tau == 0.0 else p + tau * v
        e = jnp.linalg.norm(p)
        z = jnp.stack([-s[0], s[0], -s[1], s[1]])
        zp = jnp.maximum(z, 0.0)
        u = zp / (zp + d0)
        cur = jnp.zeros((N,), DT)
        cur = cur.at[NON_MOTOR_E_IDX].set(g * (e / (e + d0)))
        cur = cur.at[MOTOR_IDX].set(jnp.repeat(g * u, GROUP_SIZE))
        return cur

    return drive


def drive_checks() -> dict[str, Any]:
    """Analytic drive values at fixed (p, v), tau = 0.1 and 0.5, and position only (N1)."""
    cases = [
        ("A tau0.1 p=(-2,0) v=0: +x=16, others motor 0, nonmotor=16",
         0.1, (-2.0, 0.0), (0.0, 0.0), {"+x": 16.0, "-x": 0.0, "+y": 0.0, "-y": 0.0, "nonmotor": 16.0}),
        ("B tau0.1 p=(0,0) v=(1,0): s=(0.1,0), -x=24*0.1/1.1",
         0.1, (0.0, 0.0), (1.0, 0.0), {"+x": 0.0, "-x": 24.0 * 0.1 / 1.1, "+y": 0.0, "-y": 0.0, "nonmotor": 0.0}),
        ("C tau0 (N1) p=(0,0) v=(1,0): velocity ignored, all motor 0",
         0.0, (0.0, 0.0), (1.0, 0.0), {"+x": 0.0, "-x": 0.0, "+y": 0.0, "-y": 0.0, "nonmotor": 0.0}),
        ("D tau0.1 p=(2,0) v=0: -x=16, +x=0",
         0.1, (2.0, 0.0), (0.0, 0.0), {"+x": 0.0, "-x": 16.0, "+y": 0.0, "-y": 0.0, "nonmotor": 16.0}),
        ("E tau0.5 p=(0,0) v=(1,0): s=(0.5,0), -x=24*0.5/1.5",
         0.5, (0.0, 0.0), (1.0, 0.0), {"+x": 0.0, "-x": 24.0 * 0.5 / 1.5, "+y": 0.0, "-y": 0.0, "nonmotor": 0.0}),
    ]
    rows = []
    for label, tau, p, v, expected in cases:
        cur = np.asarray(make_drive(tau)(np.array(p, np.float32), np.array(v, np.float32)), float)
        got = {name: float(cur[idx].mean()) for name, idx in zip(GROUP_NAMES, GROUP_IDX)}
        spread = {name: float(cur[idx].max() - cur[idx].min()) for name, idx in zip(GROUP_NAMES, GROUP_IDX)}
        got["nonmotor"] = float(cur[NON_MOTOR_E_IDX].mean())
        pv_sum = float(cur[PV_IDX].sum())
        dev = max(abs(got[k] - expected[k]) for k in expected)
        ok = dev < DRIVE_TOL and max(spread.values()) < DRIVE_TOL and pv_sum == 0.0
        rows.append({"case": label, "tau_s": tau, "p": list(p), "v": list(v), "expected": expected,
                     "got": got, "max_group_spread": max(spread.values()), "pv_sum": pv_sum,
                     "max_abs_dev": dev, "pass": bool(ok)})
    return {"tolerance": DRIVE_TOL, "all_pass": all(r["pass"] for r in rows), "cases": rows}


def rollout(model, drive_fn, *, n_intervals: int, p0, seed: int = SEED, f0: float = F0,
            dt_ms: float = DT_MS, control_ms: float = CONTROL_MS) -> dict[str, Any]:
    """Closed loop on the LIF model, copied from paradigm.sailor_neural_rollout.

    Same continuation, pending-force timing (counts of interval k applied in interval k+1),
    stimulus schedule and Simulation settings. The drive is drive_fn(state.p, state.v).
    Returns host arrays: positions (K+1, 2), velocities (K+1, 2), errors (K,), actions (K, 2),
    counts (K, 4), group_current (K, 4), spikes_E (K,), spikes_PV (K,).
    """
    steps = int(round(control_ms / dt_ms))
    if steps < 1 or abs(steps * dt_ms - control_ms) > 1e-9:
        raise ValueError(f"control_ms ({control_ms}) must be a whole number of dt_ms ({dt_ms}) steps")
    cfg = sp.SailorConfig(**CFG)
    n = len(model.static["neuron_metadata"])
    if n != N:
        raise ValueError(f"model has {n} neurons, expected {N}")
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    state = sp.sailor_reset(cfg, jax.random.PRNGKey(seed), p0=np.asarray(p0, np.float32), v0=None)
    cont = None
    # Contract section 12.3.4: counts of interval k are decoded and applied during interval k+1.
    pending = jnp.zeros((2,), sp._SAILOR_DTYPE)
    positions = [np.asarray(state.p, float)]
    velocities = [np.asarray(state.v, float)]
    errors, actions, counts, group_current, spikes_e, spikes_pv = [], [], [], [], [], []
    for _ in range(n_intervals):
        e = float(jnp.linalg.norm(state.p - cfg.target))
        cur = np.asarray(drive_fn(state.p, state.v), float)
        sched = jtfne.stimulus_schedule(
            [{"onset_ms": 0.0, "duration_ms": control_ms, "amplitude": float(c),
              "target_indices": [i]} for i, c in enumerate(cur)],
            n,
        )
        sim = jtfne.Simulation(duration_ms=control_ms, dt_ms=dt_ms, seed=seed, runtime=rt)
        if cont is None:
            sig, cont = model.simulate(sim, paradigm=sched, return_state=True)
        else:
            sig, cont = model.simulate(sim, paradigm=sched, continuation=cont, return_state=True)
        sp_arr = np.asarray(sig.spikes)
        c4 = np.array([sp_arr[:, idx].sum() for idx in GROUP_IDX], float)
        # The boat moves through interval k under the force decoded from interval k-1.
        state, _ = sp.sailor_step(cfg, state, pending)
        actions.append(np.asarray(pending, float))
        pending = sp.sailor_decode(c4, f0)
        errors.append(e)
        counts.append(c4)
        group_current.append(np.array([cur[idx].mean() for idx in GROUP_IDX], float))
        spikes_e.append(int(sp_arr[:, E_IDX].sum()))
        spikes_pv.append(int(sp_arr[:, PV_IDX].sum()))
        positions.append(np.asarray(state.p, float))
        velocities.append(np.asarray(state.v, float))
    return {
        "positions": np.stack(positions),
        "velocities": np.stack(velocities),
        "errors": np.asarray(errors, float),
        "actions": np.stack(actions),
        "counts": np.stack(counts),
        "group_current": np.stack(group_current),
        "spikes_E": np.asarray(spikes_e),
        "spikes_PV": np.asarray(spikes_pv),
    }


def episode_metrics(errors, positions) -> dict[str, Any]:
    """Pinned window = errors[n - n//5 : n]; occupancy = fraction of window with error <= RADIUS.

    axis_sign_changes_window (this file's operationalization, not in the criteria): number of
    consecutive boundary positions in the window where p_x or p_y changes sign, i.e. the trace
    crosses a coordinate axis through the origin.
    """
    errors = np.asarray(errors, float)
    pos = np.asarray(positions, float)
    n = errors.size
    lo = n - n // 5
    window = errors[lo:n]
    seg = pos[lo:n + 1]
    flips_x = np.sum(np.sign(seg[:-1, 0]) * np.sign(seg[1:, 0]) < 0)
    flips_y = np.sum(np.sign(seg[:-1, 1]) * np.sign(seg[1:, 1]) < 0)
    return {
        "final_distance": float(np.linalg.norm(pos[-1])),
        "min_distance": float(np.min(np.linalg.norm(pos, axis=1))),
        "pinned_window": [int(lo), int(n)],
        "pinned_mean_error": float(np.mean(window)),
        "fixation_occupancy": float(np.mean(window <= RADIUS)),
        "axis_sign_changes_window": int(flips_x + flips_y),
    }


def first_interval_sign(positions, actions, n: int = 20) -> dict[str, Any]:
    """Sign of force . (0 - p_k) for the force applied in intervals 0..n-1 (actions[k] acts at positions[k]).

    actions[0] is zero by construction, so it is counted as zero.
    """
    pos = np.asarray(positions, float)[:n]
    force = np.asarray(actions, float)[:n]
    dot = np.sum(force * (-pos), axis=1)
    return {
        "toward_origin": int(np.sum(dot > 0)),
        "away_from_origin": int(np.sum(dot < 0)),
        "zero": int(np.sum(dot == 0)),
        "dot_force_to_origin": [float(x) for x in dot],
    }


def run_one(condition: str, start: int, *, n_intervals: int = K, out_path: Path | None = None) -> dict[str, Any]:
    spec = CONDITIONS[condition]
    theta, p0s = dev_starts()
    model = build_model()
    check_layout(model)
    prov = provenance()
    drive = make_drive(spec["tau"])
    t0 = time.time()
    res = rollout(model, drive, n_intervals=n_intervals, p0=p0s[start])
    wall = time.time() - t0
    m = episode_metrics(res["errors"], res["positions"])
    m["pass_occupancy_ge_0_9"] = bool(m["fixation_occupancy"] >= OCC_MIN)
    rec = {
        "condition": condition, "role": spec["role"], "tau_s": spec["tau"],
        "start_index": start, "theta": float(theta[start]), "p0": [float(x) for x in p0s[start]],
        "n_intervals": n_intervals, "K": K, "seed": SEED, "f0": F0, "g": G, "d0": D0,
        "neural_dt_ms": DT_MS, "control_ms": CONTROL_MS, "sailor": CFG, "target": [0.0, 0.0],
        "motor_groups": {name: [int(i) for i in idx] for name, idx in zip(GROUP_NAMES, GROUP_IDX)},
        "non_motor_E": [int(NON_MOTOR_E_IDX[0]), int(NON_MOTOR_E_IDX[-1])],
        "PV": [int(PV_IDX[0]), int(PV_IDX[-1])],
        "provenance": prov, "wall_s": wall, "metrics": m,
        "first20_force_sign": first_interval_sign(res["positions"], res["actions"], 20),
        "errors": res["errors"].tolist(),
        "positions": res["positions"].tolist(),
        "velocities": res["velocities"].tolist(),
        "actions": res["actions"].tolist(),
        "counts": res["counts"].tolist(),
        "group_current": res["group_current"].tolist(),
        "spikes_E": res["spikes_E"].tolist(),
        "spikes_PV": res["spikes_PV"].tolist(),
    }
    if out_path is None:
        out_path = RESULTS / spec["dir"] / f"start_{start:02d}.json"
    write_json(out_path, rec)
    return rec


def smoke() -> int:
    checks = drive_checks()
    write_json(RESULTS / "smoke" / "drive_checks.json", checks, indent=2)
    print(f"drive_checks all_pass={checks['all_pass']} tolerance={checks['tolerance']}", flush=True)
    for row in checks["cases"]:
        print(f"  {row['pass']} {row['case']} max_abs_dev={row['max_abs_dev']:.3e}", flush=True)
    out = RESULTS / "smoke" / "N1_start00_K20.json"
    rec = run_one("N1", 0, n_intervals=20, out_path=out)
    fs = rec["first20_force_sign"]
    err = rec["errors"]
    print(f"N1 start 0, 20 intervals: wall_s={rec['wall_s']:.1f}", flush=True)
    print(f"  errors[0,5,10,15,19]={[round(err[i], 4) for i in (0, 5, 10, 15, 19)]}", flush=True)
    print(f"  first20 force sign toward={fs['toward_origin']} away={fs['away_from_origin']} zero={fs['zero']}",
          flush=True)
    print(f"  spikes_E total={sum(rec['spikes_E'])} spikes_PV total={sum(rec['spikes_PV'])} "
          f"finite={bool(np.all(np.isfinite(rec['positions'])))}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if checks["all_pass"] else 1


def run_cmd(condition: str, start: int) -> int:
    if not 0 <= start < N_DEV:
        raise SystemExit(f"start must be in 0..{N_DEV - 1}, got {start}")
    rec = run_one(condition, start)
    m = rec["metrics"]
    print(f"{condition} start {start:02d} theta={rec['theta']:.4f} final={m['final_distance']:.4f} "
          f"min={m['min_distance']:.4f} pinned_mean={m['pinned_mean_error']:.4f} "
          f"occ={m['fixation_occupancy']:.4f} pass={m['pass_occupancy_ge_0_9']} "
          f"crossings={m['axis_sign_changes_window']} wall_s={rec['wall_s']:.1f}", flush=True)
    return 0


def load_condition(condition: str) -> tuple[list[dict[str, Any]], list[int]]:
    folder = RESULTS / CONDITIONS[condition]["dir"]
    recs, missing = [], []
    for i in range(N_DEV):
        f = folder / f"start_{i:02d}.json"
        if f.exists():
            recs.append(json.loads(f.read_text()))
        else:
            missing.append(i)
    return recs, missing


def condition_block(condition: str) -> dict[str, Any]:
    recs, missing = load_condition(condition)
    rows = []
    for r in sorted(recs, key=lambda x: x["start_index"]):
        m = r["metrics"]
        rows.append({
            "start_index": r["start_index"], "theta": r["theta"],
            "final_distance": m["final_distance"], "min_distance": m["min_distance"],
            "pinned_mean_error": m["pinned_mean_error"], "fixation_occupancy": m["fixation_occupancy"],
            "axis_sign_changes_window": m["axis_sign_changes_window"],
            "first20_toward_origin": r["first20_force_sign"]["toward_origin"],
            "first20_away_from_origin": r["first20_force_sign"]["away_from_origin"],
            "pass": bool(m["fixation_occupancy"] >= OCC_MIN),
        })
    n_pass = sum(1 for row in rows if row["pass"])
    complete = not missing
    occ = [row["fixation_occupancy"] for row in rows]
    cross = [row["axis_sign_changes_window"] for row in rows]
    block = {
        "condition": condition, "dir": CONDITIONS[condition]["dir"], "role": CONDITIONS[condition]["role"],
        "tau_s": CONDITIONS[condition]["tau"], "n_starts_expected": N_DEV, "n_starts_present": len(rows),
        "missing_starts": missing, "n_pass": n_pass, "rows": rows,
        "occupancy_min": min(occ) if occ else None, "occupancy_max": max(occ) if occ else None,
        "occupancy_mean": float(np.mean(occ)) if occ else None,
        "axis_sign_changes_min": min(cross) if cross else None,
        "axis_sign_changes_max": max(cross) if cross else None,
        "starts_with_any_axis_sign_change": int(sum(1 for c in cross if c >= 1)) if cross else None,
    }
    if condition == "P1":
        if not complete:
            block["verdict"] = "INCOMPLETE"
        else:
            block["verdict"] = "PASS" if n_pass == N_DEV else "FAIL"
        if rows:
            worst = min(rows, key=lambda row: (row["fixation_occupancy"], row["start_index"]))
            block["worst_start_index"] = worst["start_index"]
            worst_rec = next(r for r in recs if r["start_index"] == worst["start_index"])
            block["worst_start_trace"] = {
                "intervals": list(CHECK_INTERVALS),
                "positions_at_interval_start": [worst_rec["positions"][k] for k in CHECK_INTERVALS],
                "errors_at_interval_start": [worst_rec["errors"][k] for k in CHECK_INTERVALS],
                "velocities_at_interval_start": [worst_rec["velocities"][k] for k in CHECK_INTERVALS],
            }
    else:
        block["verdict"] = "NOT A GATE" if condition == "N1" else "NOT A GATE, NOT USED TO SELECT"
    return block


def summary() -> int:
    blocks = {c: condition_block(c) for c in ("P1", "N1", "T05")}
    out = {
        "name": "plasticity_boat dev-100 P1 gate, N1 negative control, T05 sensitivity",
        "scope": "development starts only: numpy.random.default_rng(2025).uniform(0, 2*pi, 20), R = 5, target 0",
        "gate_criterion": f"every start fixation_occupancy >= {OCC_MIN}; window errors[{K - K // 5}:{K}], radius {RADIUS}",
        "provenance": provenance(),
        "P1_verdict": blocks["P1"]["verdict"],
        "P1": blocks["P1"], "N1": blocks["N1"], "T05": blocks["T05"],
        "drive_checks_file": str((RESULTS / "smoke" / "drive_checks.json").relative_to(REPO)),
    }
    write_json(RESULTS / "P1_N1_summary.json", out, indent=2)
    p1 = blocks["P1"]
    print(f"P1 verdict={p1['verdict']} n_pass={p1['n_pass']}/{N_DEV} present={p1['n_starts_present']}", flush=True)
    for c in ("N1", "T05"):
        b = blocks[c]
        print(f"{c} present={b['n_starts_present']} n_pass={b['n_pass']} occ_min={b['occupancy_min']} "
              f"occ_max={b['occupancy_max']} starts_with_crossing={b['starts_with_any_axis_sign_change']}",
              flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="drive checks and N1 start 0, 20 intervals")
    r = sub.add_parser("run", help="one condition, one development start, K = 600")
    r.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    r.add_argument("--start", type=int, required=True)
    sub.add_parser("summary", help="write P1_N1_summary.json from the per-start JSON")
    args = ap.parse_args(argv)
    if not (REPO / "AGENTS.md").exists():
        raise SystemExit(f"REPO resolved to {REPO}, which has no AGENTS.md")
    if args.cmd == "smoke":
        return smoke()
    if args.cmd == "run":
        return run_cmd(args.condition, args.start)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
