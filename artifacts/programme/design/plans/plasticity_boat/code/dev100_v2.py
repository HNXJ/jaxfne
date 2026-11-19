"""Plasticity boat, dev-100 criteria v2: sensory-gain sweep, mean decoding, calibration, validation.

Criteria: artifacts/programme/design/plans/plasticity_boat/criteria_v2.md (sha256 checked at run time).
The v1 modules dev100_closed_loop.py and dev100_resting.py are imported as libraries and are not
modified. v2 changes, and only these, relative to v1:

- non-motor E distance drive g_s * e / (e + d0) (g_s from the sweep grid); motor drive g_d = 24 unchanged;
- decoder: population mean of spike counts per five-neuron group, F_g = kappa * mean(n_i);
  v1 used the group sum with kappa = 5, so kappa = 25 under mean decoding gives the same force.

Subcommands (PYTHONPATH = repository root; run by path with /usr/local/bin/python3):
  regression --out DIR                     V0: v2 rollout (g_s 24, sum, kappa 5) reproduces v1 P1 start 00
  resting --sensory-gain G                 R1 to R4 fixed-boat sweep at one g_s (output in dev100_v2/resting/)
  closed --set {calib,dev} --condition {P1,N1,T05} --start I --sensory-gain G --kappa K
                                           one closed-loop run; calib = 10 starts, dev = 20 validation starts
  select                                   apply the C3 rule to the files present; prints the decision
  summary --sensory-gain G --kappa K       per-configuration summary (calibration and validation)
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
sys.path.insert(0, str(HERE))
import dev100_closed_loop as v1  # noqa: E402  (library only, unchanged)
import dev100_resting as rest  # noqa: E402  (library only, unchanged)

REPO = Path(__file__).resolve().parents[6]
PLAN_DIR = HERE.parent
CRITERIA = PLAN_DIR / "criteria_v2.md"
CRITERIA_SHA256 = "2b610a9bfceda99f783f5256219338a26b7efb752bb7e0297c26d7bd86fbd401"
RESULTS = REPO / "artifacts" / "_temp" / "plasticity_boat" / "dev100_v2"
V1_P1_START00 = REPO / "artifacts" / "_temp" / "plasticity_boat" / "dev100" / "P1_pos_vel" / "start_00.json"
V1_RESTING = REPO / "artifacts" / "_temp" / "plasticity_boat" / "dev100" / "R1_R4_resting_responsiveness.json"

N = v1.N
E_IDX = v1.E_IDX
PV_IDX = v1.PV_IDX
MOTOR_IDX = v1.MOTOR_IDX
NON_MOTOR_E_IDX = v1.NON_MOTOR_E_IDX
GROUP_IDX = v1.GROUP_IDX
GROUP_NAMES = v1.GROUP_NAMES
GROUP_SIZE = v1.GROUP_SIZE
DT = v1.DT
CFG = v1.CFG
G_D = 24.0  # directional motor drive gain, unchanged from v1
D0 = v1.D0
F_V1 = v1.F0  # v1 decoder gain (sum decoding)
DT_MS = v1.DT_MS
CONTROL_MS = v1.CONTROL_MS
K = v1.K
SEED = v1.SEED
R = v1.R
OCC_MIN = v1.OCC_MIN
RADIUS = v1.RADIUS
TAUS = {"P1": 0.1, "N1": 0.0, "T05": 0.5}
G_GRID = (24.0, 18.0, 12.0, 9.0, 6.0, 4.5, 3.0)  # C1, descending
KAPPA_GRID = (5.0, 10.0, 15.0, 20.0, 25.0)  # C2, ascending
CALIB_RNG_SEED = 2027
N_CALIB = 10
DEV_RNG_SEED = 2025
N_DEV = 20
SPIKE_SECONDS = K * CONTROL_MS / 1000.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_criteria() -> str:
    got = sha256_file(CRITERIA)
    if got != CRITERIA_SHA256:
        raise SystemExit(f"criteria_v2 sha256 {got} != frozen {CRITERIA_SHA256}")
    return got


def provenance() -> dict[str, Any]:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return {
        "git_head": head,
        "code": str(Path(__file__).resolve().relative_to(REPO)),
        "code_sha256": sha256_file(Path(__file__)),
        "library_closed_loop_sha256": sha256_file(HERE / "dev100_closed_loop.py"),
        "library_resting_sha256": sha256_file(HERE / "dev100_resting.py"),
        "model": str((HERE / "dev100_model.py").relative_to(REPO)),
        "model_sha256": sha256_file(HERE / "dev100_model.py"),
        "criteria": str(CRITERIA.relative_to(REPO)),
        "criteria_sha256": check_criteria(),
    }


def write_json(path: Path, obj: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
    tmp.replace(path)


def label(g_s: float, kappa: float) -> str:
    return f"g{str(g_s).replace('.', 'p')}_k{int(kappa)}"


def starts(set_name: str) -> tuple[np.ndarray, np.ndarray]:
    if set_name == "calib":
        theta = np.random.default_rng(CALIB_RNG_SEED).uniform(0.0, 2.0 * np.pi, N_CALIB)
    elif set_name == "dev":
        theta = np.random.default_rng(DEV_RNG_SEED).uniform(0.0, 2.0 * np.pi, N_DEV)
    else:
        raise SystemExit(f"unknown start set {set_name!r}")
    return theta, R * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def make_drive_v2(tau: float, g_s: float) -> Callable[[Any, Any], jax.Array]:
    """(p, v) -> current (N,). Non-motor E: g_s * e / (e + d0). Motor E: G_D * u(z), as in v1."""
    tau = float(tau)

    def drive(p, v):
        p = jnp.asarray(p, DT)
        v = jnp.asarray(v, DT)
        s = p if tau == 0.0 else p + tau * v
        e = jnp.linalg.norm(p)
        z = jnp.stack([-s[0], s[0], -s[1], s[1]])
        zp = jnp.maximum(z, 0.0)
        u = zp / (zp + D0)
        cur = jnp.zeros((N,), DT)
        cur = cur.at[NON_MOTOR_E_IDX].set(g_s * (e / (e + D0)))
        cur = cur.at[MOTOR_IDX].set(jnp.repeat(G_D * u, GROUP_SIZE))
        return cur

    return drive


def rollout_v2(model, drive_fn, *, n_intervals: int, p0, kappa: float, decoder: str,
               seed: int = SEED) -> dict[str, Any]:
    """v1 rollout (dev100_closed_loop.rollout) with the decoder selectable: 'sum' (v1) or 'mean' (v2).

    Pending-force timing is unchanged: counts of interval k are decoded and applied in interval k+1.
    Also records the per-neuron spike totals over the episode (neuron_spikes).
    """
    if decoder not in ("sum", "mean"):
        raise ValueError(f"decoder must be 'sum' or 'mean', got {decoder!r}")
    cfg = sp.SailorConfig(**CFG)
    n = len(model.static["neuron_metadata"])
    if n != N:
        raise ValueError(f"model has {n} neurons, expected {N}")
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    state = sp.sailor_reset(cfg, jax.random.PRNGKey(seed), p0=np.asarray(p0, np.float32), v0=None)
    cont = None
    pending = jnp.zeros((2,), sp._SAILOR_DTYPE)
    positions = [np.asarray(state.p, float)]
    velocities = [np.asarray(state.v, float)]
    errors, actions, counts, group_current, spikes_e, spikes_pv = [], [], [], [], [], []
    neuron_spikes = np.zeros(N, np.int64)
    for _ in range(n_intervals):
        e = float(jnp.linalg.norm(state.p - cfg.target))
        cur = np.asarray(drive_fn(state.p, state.v), float)
        sched = jtfne.stimulus_schedule(
            [{"onset_ms": 0.0, "duration_ms": CONTROL_MS, "amplitude": float(c),
              "target_indices": [i]} for i, c in enumerate(cur)],
            n,
        )
        sim = jtfne.Simulation(duration_ms=CONTROL_MS, dt_ms=DT_MS, seed=seed, runtime=rt)
        if cont is None:
            sig, cont = model.simulate(sim, paradigm=sched, return_state=True)
        else:
            sig, cont = model.simulate(sim, paradigm=sched, continuation=cont, return_state=True)
        sp_arr = np.asarray(sig.spikes)
        if sp_arr.shape != (int(CONTROL_MS / DT_MS), N):
            raise RuntimeError(f"spike array shape {sp_arr.shape}")
        c4 = np.array([sp_arr[:, idx].sum() for idx in GROUP_IDX], float)
        neuron_spikes += sp_arr.sum(axis=0).astype(np.int64)
        state, _ = sp.sailor_step(cfg, state, pending)
        actions.append(np.asarray(pending, float))
        if decoder == "sum":
            dec_counts = c4
        else:
            dec_counts = c4 / GROUP_SIZE  # population mean of spike counts per group, C2
        pending = sp.sailor_decode(dec_counts, kappa)
        errors.append(e)
        counts.append(c4)
        group_current.append(np.array([cur[idx].mean() for idx in GROUP_IDX], float))
        spikes_e.append(int(sp_arr[:, E_IDX].sum()))
        spikes_pv.append(int(sp_arr[:, PV_IDX].sum()))
        positions.append(np.asarray(state.p, float))
        velocities.append(np.asarray(state.v, float))
    return {
        "positions": np.stack(positions), "velocities": np.stack(velocities),
        "errors": np.asarray(errors, float), "actions": np.stack(actions),
        "counts": np.stack(counts), "group_current": np.stack(group_current),
        "spikes_E": np.asarray(spikes_e), "spikes_PV": np.asarray(spikes_pv),
        "neuron_spikes": neuron_spikes,
    }


def closed_loop_rates(res: dict[str, Any]) -> dict[str, Any]:
    """Moving-input rates over the episode (K intervals of CONTROL_MS)."""
    ns = np.asarray(res["neuron_spikes"], float)
    return {
        "duration_s": SPIKE_SECONDS,
        "E_mean_Hz": float(ns[E_IDX].sum() / (len(E_IDX) * SPIKE_SECONDS)),
        "PV_mean_Hz": float(ns[PV_IDX].sum() / (len(PV_IDX) * SPIKE_SECONDS)),
        "non_motor_E_mean_Hz": float(ns[NON_MOTOR_E_IDX].sum() / (len(NON_MOTOR_E_IDX) * SPIKE_SECONDS)),
        "max_neuron_Hz": float(ns.max() / SPIKE_SECONDS),
        "max_neuron_index": int(np.argmax(ns)),
        "per_neuron_Hz": (ns / SPIKE_SECONDS).tolist(),
    }


def run_closed(condition: str, set_name: str, start: int, g_s: float, kappa: float,
               out_path: Path | None = None, *, decoder: str = "mean") -> dict[str, Any]:
    theta, p0s = starts(set_name)
    model = v1.build_model()
    v1.check_layout(model)
    prov = provenance()
    drive = make_drive_v2(TAUS[condition], g_s)
    t0 = time.time()
    res = rollout_v2(model, drive, n_intervals=K, p0=p0s[start], kappa=kappa, decoder=decoder)
    wall = time.time() - t0
    m = v1.episode_metrics(res["errors"], res["positions"])
    m["pass_occupancy_ge_0_9"] = bool(m["fixation_occupancy"] >= OCC_MIN)
    rec = {
        "criteria": "v2", "condition": condition, "set": set_name, "start_index": start,
        "theta": float(theta[start]), "p0": [float(x) for x in p0s[start]],
        "tau_s": TAUS[condition], "g_sensory": float(g_s), "g_directional": G_D, "d0": D0,
        "decoder": decoder, "kappa": float(kappa),
        "decoder_definition": "F_g = kappa * (1/5) * sum_{i in g} n_i(interval); applied next interval",
        "n_intervals": K, "K": K, "seed": SEED, "neural_dt_ms": DT_MS, "control_ms": CONTROL_MS,
        "sailor": CFG, "target": [0.0, 0.0],
        "motor_groups": {name: [int(i) for i in idx] for name, idx in zip(GROUP_NAMES, GROUP_IDX)},
        "provenance": prov, "wall_s": wall, "metrics": m,
        "rates_closed_loop": closed_loop_rates(res),
        "first20_force_sign": v1.first_interval_sign(res["positions"], res["actions"], 20),
        "errors": res["errors"].tolist(), "positions": res["positions"].tolist(),
        "velocities": res["velocities"].tolist(), "actions": res["actions"].tolist(),
        "counts": res["counts"].tolist(), "group_current": res["group_current"].tolist(),
        "spikes_E": res["spikes_E"].tolist(), "spikes_PV": res["spikes_PV"].tolist(),
    }
    if out_path is None:
        if set_name == "calib":
            out_path = RESULTS / "calib" / label(g_s, kappa) / f"start_{start:02d}.json"
        else:
            out_path = RESULTS / "validate" / label(g_s, kappa) / condition / f"start_{start:02d}.json"
    write_json(out_path, rec)
    return rec


def cur_r2_v2(g_s: float) -> np.ndarray:
    cur = np.zeros(N, np.float64)
    cur[NON_MOTOR_E_IDX] = g_s * 5.0 / (5.0 + D0)
    return cur


def u_of(z: float) -> float:
    zp = max(float(z), 0.0)
    return zp / (zp + D0)


def cur_r3_v2(p, g_s: float, g_d: float = G_D) -> np.ndarray:
    px, py = float(p[0]), float(p[1])
    e = float(np.hypot(px, py))
    cur = np.zeros(N, np.float64)
    cur[NON_MOTOR_E_IDX] = g_s * e / (e + D0)
    cur[rest.GROUPS["+x"]] = g_d * u_of(-px)
    cur[rest.GROUPS["-x"]] = g_d * u_of(px)
    cur[rest.GROUPS["+y"]] = g_d * u_of(-py)
    cur[rest.GROUPS["-y"]] = g_d * u_of(py)
    return cur


def resting(g_s: float, out_dir: Path, g_d: float = G_D) -> dict[str, Any]:
    """R1 to R4 at one sensory gain g_s and directional gain g_d.

    g_d = 24 (v1) is the criteria v2 setting. Any other g_d is a diagnostic: its file is named with
    _gd and is not read by select(), so it cannot change the selection.
    """
    criteria_sha = check_criteria()
    model = rest.build_dev100()
    build = rest.check_build(model)
    cur1 = rest.cur_zero()
    s1 = rest.summarize("R1", "intrinsic: no external input", cur1, rest.run_block(model, cur1, rest.N_R1), rest.N_R1)
    cur2 = cur_r2_v2(g_s)
    assert np.count_nonzero(cur2) == 60 and np.all(cur2[rest.MOTOR_E] == 0.0) and np.all(cur2[PV_IDX] == 0.0)
    s2 = rest.summarize("R2", f"stationary: non-motor E distance drive at e=5, g_s={g_s}; motor and PV none",
                        cur2, rest.run_block(model, cur2, rest.N_R2), rest.N_R2)
    s3 = []
    for label_, p, toward in rest.POSITIONS:
        cur = cur_r3_v2(p, g_s, g_d)
        opp = rest.OPPOSITE[toward]
        assert np.count_nonzero(cur) == 65, f"{label_}: expected 60 non-motor + 5 toward-group nonzero"
        assert np.allclose(cur[rest.GROUPS[toward]], g_d * 5.0 / 6.0), f"{label_}: toward group g_d*u(5)"
        assert np.all(cur[rest.GROUPS[toward]] > 0.0) and np.all(cur[rest.GROUPS[opp]] == 0.0)
        blk = rest.run_block(model, cur, rest.N_R3)
        s = rest.summarize(f"R3 {label_}", f"directional, g_s={g_s}, g_d={g_d}", cur, blk, rest.N_R3)
        s["toward_group"], s["opposite_group"] = toward, opp
        s3.append(s)
    gates = rest.evaluate(s1, s2, s3)
    payload = {
        "lane": "dev100 v2 resting sweep, one (g_s, g_d) pair",
        "diagnostic": bool(g_d != G_D),
        "criteria": {"path": str(CRITERIA.relative_to(REPO)), "sha256": criteria_sha, "expected_sha256": CRITERIA_SHA256},
        "provenance": provenance(), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build": build, "sensory_gain_g_s": g_s, "directional_gain_g_d": g_d,
        "blocks": {"R1": s1, "R2": s2, "R3": s3}, "gates": gates,
        "pass_R2_R3_R4": bool(gates["R2"]["pass"] and gates["R3"]["pass"] and gates["R4"]["pass"]),
        "all_pass_R1_to_R4": bool(all(g["pass"] for g in gates.values())),
    }
    stem = f"resting_g{str(g_s).replace('.', 'p')}"
    if g_d != G_D:
        stem += f"_gd{str(g_d).replace('.', 'p')}"
    write_json(out_dir / f"{stem}.json", payload)
    return payload


def regression(out_dir: Path) -> dict[str, Any]:
    """V0: v2 rollout with g_s 24, sum decoding and kappa 5 must reproduce the stored v1 P1 start 00."""
    ref = json.loads(V1_P1_START00.read_text())
    rec = run_closed("P1", "dev", 0, G_D, F_V1, out_path=out_dir / "V0_start00_sum_k5.json", decoder="sum")
    e_new = np.asarray(rec["errors"], float)
    e_ref = np.asarray(ref["errors"], float)
    p_new = np.asarray(rec["positions"], float)
    p_ref = np.asarray(ref["positions"], float)
    out = {
        "check": "V0 regression: v2 rollout (g_s 24, sum, kappa 5, P1, dev start 0) vs stored v1 P1 start_00",
        "max_abs_diff_errors": float(np.max(np.abs(e_new - e_ref))),
        "max_abs_diff_positions": float(np.max(np.abs(p_new - p_ref))),
        "exact": bool(np.array_equal(e_new, e_ref) and np.array_equal(p_new, p_ref)),
        "reference": str(V1_P1_START00.relative_to(REPO)),
    }
    out["pass"] = out["exact"]
    write_json(out_dir / "V0_regression.json", out, indent=2)
    return out


def select() -> dict[str, Any]:
    """Apply C3 to the files present. Prints the next action or the selected pair."""
    passing = []
    for g in G_GRID:
        f = RESULTS / "resting" / f"resting_g{str(g).replace('.', 'p')}.json"
        if not f.exists():
            return {"status": "MISSING_RESTING", "missing": str(f.relative_to(REPO))}
        d = json.loads(f.read_text())
        if d["pass_R2_R3_R4"]:
            passing.append(g)
    result: dict[str, Any] = {"resting_pass_set": passing, "C3_step": 1}
    for g in passing:  # descending by construction of G_GRID
        for k in KAPPA_GRID:
            f = RESULTS / "summary" / f"{label(g, k)}.json"
            if not f.exists():
                result.update({"status": "NEXT_CALIBRATION", "g_sensory": g, "kappa": k})
                return result
            s = json.loads(f.read_text())
            if s["calibration_pass"]:
                result.update({"status": "SELECTED", "g_sensory": g, "kappa": k, "C3_step": 2})
                return result
    result["status"] = "NO_PAIR_PASSES"
    return result


def summary(g_s: float, kappa: float) -> dict[str, Any]:
    lab = label(g_s, kappa)
    out: dict[str, Any] = {"label": lab, "g_sensory": g_s, "kappa": kappa, "decoder": "mean"}
    cal_dir = RESULTS / "calib" / lab
    recs = []
    for i in range(N_CALIB):
        f = cal_dir / f"start_{i:02d}.json"
        if f.exists():
            recs.append(json.loads(f.read_text()))
    passes = [r["metrics"]["fixation_occupancy"] >= OCC_MIN for r in recs]
    out["calibration"] = {
        "n_present": len(recs), "n_expected": N_CALIB,
        "n_pass": int(sum(passes)),
        "occupancy_min": min((r["metrics"]["fixation_occupancy"] for r in recs), default=None),
        "final_distance_max": max((r["metrics"]["final_distance"] for r in recs), default=None),
    }
    out["calibration_pass"] = bool(len(recs) == N_CALIB and all(passes))
    val = {}
    for cond in ("P1", "N1", "T05"):
        d = RESULTS / "validate" / lab / cond
        rs = []
        for i in range(N_DEV):
            f = d / f"start_{i:02d}.json"
            if f.exists():
                rs.append(json.loads(f.read_text()))
        occ = [r["metrics"]["fixation_occupancy"] for r in rs]
        val[cond] = {
            "n_present": len(rs), "n_expected": N_DEV,
            "n_pass": int(sum(o >= OCC_MIN for o in occ)),
            "occupancy_min": min(occ, default=None), "occupancy_max": max(occ, default=None),
            "final_distance_min": min((r["metrics"]["final_distance"] for r in rs), default=None),
            "final_distance_max": max((r["metrics"]["final_distance"] for r in rs), default=None),
            "E_mean_Hz_moving": [r["rates_closed_loop"]["E_mean_Hz"] for r in rs],
            "PV_mean_Hz_moving": [r["rates_closed_loop"]["PV_mean_Hz"] for r in rs],
            "max_neuron_Hz_moving": [r["rates_closed_loop"]["max_neuron_Hz"] for r in rs],
        }
    out["validation"] = val
    out["P1_verdict"] = "PASS" if (val["P1"]["n_present"] == N_DEV and val["P1"]["n_pass"] == N_DEV) else (
        "INCOMPLETE" if val["P1"]["n_present"] < N_DEV else "FAIL")
    write_json(RESULTS / "summary" / f"{lab}.json", out, indent=2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rg = sub.add_parser("regression")
    rg.add_argument("--out", required=True)
    rs = sub.add_parser("resting")
    rs.add_argument("--sensory-gain", type=float, required=True)
    rs.add_argument("--directional-gain", type=float, default=G_D,
                    help="default 24 (criteria v2). Other values are diagnostics only.")
    cl = sub.add_parser("closed")
    cl.add_argument("--set", choices=("calib", "dev"), required=True)
    cl.add_argument("--condition", choices=sorted(TAUS), required=True)
    cl.add_argument("--start", type=int, required=True)
    cl.add_argument("--sensory-gain", type=float, required=True)
    cl.add_argument("--kappa", type=float, required=True)
    sub.add_parser("select")
    sm = sub.add_parser("summary")
    sm.add_argument("--sensory-gain", type=float, required=True)
    sm.add_argument("--kappa", type=float, required=True)
    args = ap.parse_args(argv)
    if not (REPO / "AGENTS.md").exists():
        raise SystemExit(f"REPO resolved to {REPO}, which has no AGENTS.md")
    if args.cmd == "regression":
        out = regression(Path(args.out))
        print(f"V0 exact={out['exact']} max_abs_diff_errors={out['max_abs_diff_errors']:.3e} pass={out['pass']}")
        return 0 if out["pass"] else 1
    if args.cmd == "resting":
        p = resting(args.sensory_gain, RESULTS / "resting", args.directional_gain)
        g = p["gates"]
        print(f"g_s={args.sensory_gain} g_d={args.directional_gain} R2 E={g['R2']['E_rate_Hz']:.3f} pass={g['R2']['pass']} "
              f"R3 pass={g['R3']['pass']} R4 max={g['R4']['max_neuron_rate_R2_R3_Hz']:.3f} pass={g['R4']['pass']} "
              f"R1 pass={g['R1']['pass']}")
        return 0
    if args.cmd == "closed":
        rec = run_closed(args.condition, args.set, args.start, args.sensory_gain, args.kappa)
        m = rec["metrics"]
        rc = rec["rates_closed_loop"]
        print(f"{args.set} {args.condition} start {args.start:02d} g_s={args.sensory_gain} kappa={args.kappa} "
              f"final={m['final_distance']:.4f} occ={m['fixation_occupancy']:.4f} "
              f"pass={m['pass_occupancy_ge_0_9']} E_Hz={rc['E_mean_Hz']:.2f} max_Hz={rc['max_neuron_Hz']:.2f} "
              f"wall_s={rec['wall_s']:.1f}", flush=True)
        return 0
    if args.cmd == "select":
        print(json.dumps(select(), indent=2))
        return 0
    s = summary(args.sensory_gain, args.kappa)
    print(json.dumps({"label": s["label"], "calibration_pass": s["calibration_pass"],
                      "P1_verdict": s["P1_verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
