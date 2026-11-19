"""Resting and responsiveness gates R1 to R4 on the dev-100 model.

Criteria: artifacts/programme/design/plans/plasticity_boat/criteria_v1_frozen.md (frozen 2026-11-19,
sha256 d57fd981...). The thresholds below are copied from that file and none is tuned. The file hash
is checked at run time and the run refuses on a mismatch.

Fixed-boat runs: the boat is not moved and the decoder is not applied. Each control interval
(20 ms = 20 neural steps of dt 1 ms) holds one constant external current per neuron, delivered with
stimulus_schedule, Simulation and full-state continuation, as in jaxfne sailor_neural_rollout.

Blocks. Each block starts from the reset network state with seed 0.
  R1  intrinsic: no external input at all, 100 intervals.
  R2  input-driven, stationary: non-motor E (20 to 79) distance drive g*e/(e+d0) at e = 5.
      Motor E (0 to 19) get no drive (packet choice). PV none. 600 intervals.
  R3  input-driven, directional: boat fixed at (-5,0), (5,0), (0,-5), (0,5), 100 intervals each.
      Non-motor E keep the distance drive at e = |p| = 5 (model definition). Motor groups get the
      directional drive g*u(z), z from position only. PV none.

Population rate = spikes / (n_pop * duration_s). Group rate = mean over the group's 5 neurons.

Usage (PYTHONPATH=<repo root>, PYTHONDONTWRITEBYTECODE=1):
  python3 -u dev100_resting.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np

import jaxfne as jtfne
from jaxfne._construct_lif import lif_edge_arrays

from dev100_model import build_dev100

CRITERIA_SHA256 = "d57fd9810732945e8f592e88e65fccf1b65b894e737ac1ed0f745e46a44037ee"
HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA_REL = "artifacts/programme/design/plans/plasticity_boat/criteria_v1_frozen.md"
OUT_NAME = "R1_R4_resting_responsiveness.json"

N = 100
N_E = 80
N_PV = 20
E_IDX = np.arange(0, 80)
PV_IDX = np.arange(80, 100)
MOTOR_E = np.arange(0, 20)
NONMOTOR_E = np.arange(20, 80)
GROUPS = {"+x": np.arange(0, 5), "-x": np.arange(5, 10), "+y": np.arange(10, 15), "-y": np.arange(15, 20)}
OPPOSITE = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}
# (label, boat position, group pointing toward the target at that position)
POSITIONS = [
    ("p(-5,0)", (-5.0, 0.0), "+x"),
    ("p(5,0)", (5.0, 0.0), "-x"),
    ("p(0,-5)", (0.0, -5.0), "+y"),
    ("p(0,5)", (0.0, 5.0), "-y"),
]
TARGET = (0.0, 0.0)

G = 24.0
D0 = 1.0
DT_MS = 1.0
CONTROL_MS = 20.0
SEED = 0
E_DISTANCE_R2 = 5.0
N_R1 = 100
N_R2 = 600
N_R3 = 100

R1_MAX_HZ = 1.0
R2_E_LO_HZ, R2_E_HI_HZ = 5.0, 100.0
R2_PV_LO_HZ, R2_PV_HI_HZ = 0.0, 100.0
R2_MIN_DELTA_HZ = 5.0
R3_MIN_DIFF_HZ = 5.0
R4_MAX_HZ = 100.0


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_head():
    out = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def check_criteria():
    path = os.path.normpath(os.path.join(HERE, "..", "criteria_v1_frozen.md"))
    digest = sha256_file(path)
    if digest != CRITERIA_SHA256:
        raise SystemExit(f"criteria hash mismatch: {digest} != {CRITERIA_SHA256}; refusing to run")
    return {"path": CRITERIA_REL, "sha256": digest, "expected_sha256": CRITERIA_SHA256, "match": True}


def check_build(model):
    md = model.static["neuron_metadata"]
    ct = [str(row["cell_type"]) for row in md]
    e_idx = [i for i, c in enumerate(ct) if c == "E"]
    pv_idx = [i for i, c in enumerate(ct) if c == "PV"]
    n_edges = int(np.asarray(lif_edge_arrays(model)["pre"]).size)
    ok = (len(ct) == N and e_idx == list(range(N_E)) and pv_idx == list(range(N_E, N))
          and n_edges == 9900)
    if not ok:
        raise SystemExit(f"build check failed: n={len(ct)} E={len(e_idx)} PV={len(pv_idx)} edges={n_edges}")
    for name, idx in GROUPS.items():
        assert all(i in e_idx for i in idx), f"motor group {name} must lie in E"
    return {"n_neurons": len(ct), "n_E": len(e_idx), "n_PV": len(pv_idx),
            "E_range": [e_idx[0], e_idx[-1]], "PV_range": [pv_idx[0], pv_idx[-1]], "n_edges": n_edges}


def u_of(z):
    zp = max(float(z), 0.0)
    return zp / (zp + D0)


def distance_drive(e):
    e = float(e)
    return G * e / (e + D0)


def cur_zero():
    return np.zeros(N, np.float64)


def cur_r2():
    """Non-motor E at e = 5; motor E and PV get no drive."""
    cur = np.zeros(N, np.float64)
    cur[NONMOTOR_E] = distance_drive(E_DISTANCE_R2)
    return cur


def cur_r3(p):
    """Non-motor E distance drive at e = |p - target|; motor groups directional from position z."""
    p = (float(p[0]), float(p[1]))
    e = float(np.hypot(p[0] - TARGET[0], p[1] - TARGET[1]))
    cur = np.zeros(N, np.float64)
    cur[NONMOTOR_E] = distance_drive(e)
    cur[GROUPS["+x"]] = G * u_of(-p[0])
    cur[GROUPS["-x"]] = G * u_of(p[0])
    cur[GROUPS["+y"]] = G * u_of(-p[1])
    cur[GROUPS["-y"]] = G * u_of(p[1])
    return cur


def run_block(model, cur, n_intervals):
    """Hold ``cur`` for n_intervals control intervals, boat fixed, from the reset network state."""
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    cont = None
    total = np.zeros(N, np.int64)
    e_counts, pv_counts, walls = [], [], []
    spk_shape = None
    for _ in range(n_intervals):
        t0 = time.perf_counter()
        sched = jtfne.stimulus_schedule(
            [{"onset_ms": 0.0, "duration_ms": CONTROL_MS, "amplitude": float(c), "target_indices": [i]}
             for i, c in enumerate(cur)],
            N,
        )
        sim = jtfne.Simulation(duration_ms=CONTROL_MS, dt_ms=DT_MS, seed=SEED, runtime=rt)
        if cont is None:
            sig, cont = model.simulate(sim, paradigm=sched, return_state=True)
        else:
            sig, cont = model.simulate(sim, paradigm=sched, continuation=cont, return_state=True)
        spk = np.asarray(sig.spikes)
        if spk_shape is None:
            spk_shape = [int(x) for x in spk.shape]
        cnt = spk.sum(axis=0).astype(np.int64)
        total += cnt
        e_counts.append(int(cnt[E_IDX].sum()))
        pv_counts.append(int(cnt[PV_IDX].sum()))
        walls.append(time.perf_counter() - t0)
    return {"total": total, "E_counts": e_counts, "PV_counts": pv_counts,
            "spk_shape": spk_shape, "wall_s": walls}


def summarize(label, regime, cur, blk, n_intervals):
    assert blk["spk_shape"] == [int(CONTROL_MS), N], f"unexpected spike shape {blk['spk_shape']}"
    T = n_intervals * CONTROL_MS / 1000.0
    total = blk["total"]
    rate = total / T
    e_rate = float(total[E_IDX].sum() / (N_E * T))
    pv_rate = float(total[PV_IDX].sum() / (N_PV * T))
    last = min(100, n_intervals)
    e_last = float(np.sum(blk["E_counts"][-last:]) / (N_E * last * CONTROL_MS / 1000.0))
    return {
        "label": label, "regime": regime, "n_intervals": n_intervals, "duration_s": T,
        "current_nonzero_count": int(np.count_nonzero(cur)), "current_per_neuron": cur.tolist(),
        "E_rate_Hz": e_rate, "PV_rate_Hz": pv_rate, "E_rate_last100_intervals_Hz": e_last,
        "group_rate_Hz": {g: float(rate[idx].mean()) for g, idx in GROUPS.items()},
        "per_neuron_rate_Hz": rate.tolist(),
        "max_neuron_rate_Hz": float(rate.max()),
        "spikes_total_E": int(total[E_IDX].sum()), "spikes_total_PV": int(total[PV_IDX].sum()),
        "per_interval_E_spikes": blk["E_counts"], "per_interval_PV_spikes": blk["PV_counts"],
        "spike_array_shape_per_interval": blk["spk_shape"], "wall_s_total": float(np.sum(blk["wall_s"])),
    }


def evaluate(s1, s2, s3):
    r1 = {
        "criterion": "R1 intrinsic: no external input, 100 intervals; mean E <= 1 Hz and mean PV <= 1 Hz",
        "E_rate_Hz": s1["E_rate_Hz"], "PV_rate_Hz": s1["PV_rate_Hz"],
        "pass": bool(s1["E_rate_Hz"] <= R1_MAX_HZ and s1["PV_rate_Hz"] <= R1_MAX_HZ),
    }
    delta = s2["E_rate_Hz"] - s1["E_rate_Hz"]
    r2 = {
        "criterion": ("R2 stationary input: mean E in [5,100] Hz; mean PV in [0,100] Hz; "
                      "E(R2) - E(R1) >= 5 Hz; 600 intervals; e = 5 on non-motor E"),
        "E_rate_Hz": s2["E_rate_Hz"], "PV_rate_Hz": s2["PV_rate_Hz"], "E_minus_R1_Hz": delta,
        "pass": bool(R2_E_LO_HZ <= s2["E_rate_Hz"] <= R2_E_HI_HZ
                     and R2_PV_LO_HZ <= s2["PV_rate_Hz"] <= R2_PV_HI_HZ
                     and delta >= R2_MIN_DELTA_HZ),
    }
    positions = []
    for s in s3:
        toward, opp = s["toward_group"], s["opposite_group"]
        diff = s["group_rate_Hz"][toward] - s["group_rate_Hz"][opp]
        positions.append({
            "position": s["label"], "toward_group": toward, "opposite_group": opp,
            "toward_Hz": s["group_rate_Hz"][toward], "opposite_Hz": s["group_rate_Hz"][opp],
            "diff_Hz": diff, "pass": bool(diff >= R3_MIN_DIFF_HZ),
        })
    r3 = {
        "criterion": "R3 directional: at each position, toward-group mean minus opposite-group mean >= 5 Hz",
        "positions": positions, "pass": all(p["pass"] for p in positions),
    }
    r3_max = max(s["max_neuron_rate_Hz"] for s in s3)
    r4_max = max(s2["max_neuron_rate_Hz"], r3_max)
    r4 = {
        "criterion": "R4 no saturation: every neuron mean rate <= 100 Hz in R2 and in each R3 position block",
        "max_neuron_rate_R2_Hz": s2["max_neuron_rate_Hz"], "max_neuron_rate_R3_Hz": r3_max,
        "max_neuron_rate_R2_R3_Hz": r4_max, "pass": bool(r4_max <= R4_MAX_HZ),
    }
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4}


def write_json(path, payload):
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    criteria = check_criteria()
    model = build_dev100()
    build = check_build(model)

    cur1 = cur_zero()
    assert np.count_nonzero(cur1) == 0, "R1 must carry no external input"
    s1 = summarize("R1", "intrinsic: no external input", cur1, run_block(model, cur1, N_R1), N_R1)

    cur2 = cur_r2()
    assert np.count_nonzero(cur2) == 60 and np.all(cur2[MOTOR_E] == 0.0) and np.all(cur2[PV_IDX] == 0.0)
    assert np.allclose(cur2[NONMOTOR_E], G * 5.0 / 6.0), "non-motor E current must be g*5/6 = 20"
    s2 = summarize("R2", "input-driven, stationary: non-motor E distance drive at e=5; motor and PV none",
                   cur2, run_block(model, cur2, N_R2), N_R2)

    s3 = []
    for label, p, toward in POSITIONS:
        cur = cur_r3(p)
        opp = OPPOSITE[toward]
        assert np.count_nonzero(cur) == 65, f"{label}: expected 60 non-motor + 5 toward-group nonzero"
        assert np.allclose(cur[GROUPS[toward]], G * 5.0 / 6.0), f"{label}: toward group must get g*u(5) = 20"
        assert np.all(cur[GROUPS[toward]] > 0.0) and np.all(cur[GROUPS[opp]] == 0.0), f"{label}: direction sign"
        blk = run_block(model, cur, N_R3)
        s = summarize(f"R3 {label}", "input-driven, directional: non-motor distance drive at e=5 plus motor "
                      "directional drive from position z; PV none", cur, blk, N_R3)
        s["toward_group"], s["opposite_group"] = toward, opp
        s3.append(s)

    gates = evaluate(s1, s2, s3)
    all_pass = all(g["pass"] for g in gates.values())
    concat = np.mean(np.asarray([s["per_neuron_rate_Hz"] for s in s3]), axis=0)
    payload = {
        "lane": "dev100 resting and responsiveness R1 to R4",
        "git_head": git_head(),
        "timestamp_utc": utc_now(),
        "criteria": criteria,
        "code": {
            "dev100_model_sha256": sha256_file(os.path.join(HERE, "dev100_model.py")),
            "dev100_resting_sha256": sha256_file(os.path.abspath(__file__)),
        },
        "build": build,
        "config": {
            "fixed_boat": True, "decoder_applied": False, "starts": "none (fixed positions only)",
            "control_ms": CONTROL_MS, "dt_ms": DT_MS, "seed": SEED, "G": G, "d0": D0,
            "target": list(TARGET), "R_star": 5.0, "E_distance_R2": E_DISTANCE_R2,
            "groups": {k: v.tolist() for k, v in GROUPS.items()},
            "E_indices": "0-79", "PV_indices": "80-99",
        },
        "interpretation": [
            "R1 is intrinsic: zero external current on all 100 neurons, so any activity is recurrent.",
            "R2 and R3 are input-driven. R2: non-motor E distance drive at e = 5; motor E get no drive "
            "(packet choice, stated); PV none.",
            "R3: non-motor E keep the distance drive at e = |p| = 5 (model definition); motor groups get "
            "directional drive g*u(z), z position only; PV none.",
            "The boat is fixed, so velocity is zero and position-plus-velocity z equals position-only z.",
            "Each R3 position is its own block from the reset network state (seed 0), 100 intervals.",
            "R4 is checked per block (each R3 position separately), which is stricter than a 4-block mean.",
            "Thresholds are copied from criteria_v1_frozen.md; nothing was tuned on outcomes.",
        ],
        "blocks": {"R1": s1, "R2": s2, "R3": s3},
        "R3_concat_per_neuron_mean_max_Hz": float(concat.max()),
        "gates": gates,
        "all_pass": bool(all_pass),
    }
    path = os.path.join(args.out, OUT_NAME)
    write_json(path, payload)

    print(f"R1 E={s1['E_rate_Hz']:.6f} PV={s1['PV_rate_Hz']:.6f} pass={gates['R1']['pass']}")
    print(f"R2 E={s2['E_rate_Hz']:.6f} PV={s2['PV_rate_Hz']:.6f} E-R1={gates['R2']['E_minus_R1_Hz']:.6f} "
          f"pass={gates['R2']['pass']}")
    for p in gates["R3"]["positions"]:
        print(f"R3 {p['position']} toward={p['toward_group']} {p['toward_Hz']:.6f} opp={p['opposite_group']} "
              f"{p['opposite_Hz']:.6f} diff={p['diff_Hz']:.6f} pass={p['pass']}")
    print(f"R3 all pass={gates['R3']['pass']}")
    print(f"R4 max_R2={gates['R4']['max_neuron_rate_R2_Hz']:.6f} max_R3={gates['R4']['max_neuron_rate_R3_Hz']:.6f} "
          f"pass={gates['R4']['pass']}")
    print(f"ALL PASS={all_pass} wrote {path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
