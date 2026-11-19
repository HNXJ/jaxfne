"""Directional positive control on the 10-neuron fixture (plasticity_boat, EX-24, lane A).

Closed loop: boat position -> signed-position population encoder on the four motor E neurons
(+x, -x, +y, -y) -> LIF spikes -> four-channel decoder -> boat force -> boat position.
Non-motor E neurons keep the M2 fixture distance drive g*e/(e+d0); PV neurons get no external drive.
Encoder: Option A of protocol_v0.md section 2 (provisional; owner trade-off recorded there).

Gates, fixed before any run:
  A0  fixture replay: rollout_with_drive with the fixture distance drive reproduces the F1.json seed-0
      errors (first 600 intervals) to max abs difference <= 1e-6.
  A1  sign: p=(-2,0) gives +x>0, -x=0; p=(2,0) gives -x>0, +x=0; p=(0,-2) gives +y>0, -y=0.
  A2  zero motor drive (g_dir = 0 on the four motor neurons): boat stays at R=5, max |error-5| < 1e-6
      over 100 intervals.
  A3  EX-24 positive control: on all 20 development starts, fixation occupancy (fraction of
      errors[480:600] with error <= 1.0) >= 0.9. PASS only if every start passes.

Usage (PYTHONPATH = repository root; run by path):
  /usr/local/bin/python3 -u <this file> gates   # A0, A1, A2 -> artifacts/_temp/plasticity_boat/gates/
  /usr/local/bin/python3 -u <this file> ex24    # A3, only if gates_summary.json says all pass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne import paradigm as sp

# code/ sits at <repo>/artifacts/programme/design/plans/plasticity_boat/code/
REPO = Path(__file__).resolve().parents[6]
RESULTS = REPO / "artifacts" / "_temp" / "plasticity_boat"
GATE_DIR = RESULTS / "gates"
EX24_DIR = RESULTS / "ex24"
F1_REF = REPO / "artifacts" / "_temp" / "crazy_sailor_explore" / "lane_C" / "F1.json"

N = 10
MOTOR = (0, 1, 2, 3)  # E neurons in (+x, -x, +y, -y) order
NON_MOTOR_E = (4, 5, 6, 7)
PV = (8, 9)
CFG = dict(n_dim=2, dt=0.02, damping=0.5)
F0_GAIN = 5.0
DT_MS = 1.0
CONTROL_MS = 20.0
K = 600
SEED = 0
R = 5.0
G_FIXTURE = 24.0
D0 = 1.0
G_DIR = 24.0
FIXATION_RADIUS = 1.0
OCCUPANCY_MIN = 0.9
DEV_RNG_SEED = 2025
N_DEV = 20
A0_TOL = 1e-6
A2_TOL = 1e-6
A2_INTERVALS = 100


def build_model():
    """The M2 fixture network (copied from the lane C explore harness, build_model)."""

    def conn(src, dst, mech):
        return nt.InterConnection(
            source_layer="L", source_neuron_type=src, target_layer="L",
            target_neuron_type=dst, mechanism=mech,
            static=nt.StaticParams(dT_ms=5.0))

    tensor = nt.NeuronalTensor(
        name="crazy-sailor-m2",
        areas=[nt.Area(name="A", layers=[nt.Layer(
            name="L", n_neurons=N,
            neuron_types=[nt.NeuronType.make("E", fraction=0.8),
                          nt.NeuronType.make("PV", fraction=0.2)])],
            inter_connections=[
                conn("E", "E", "AMPA"), conn("E", "PV", "AMPA"),
                conn("PV", "E", "GABA_A"), conn("PV", "PV", "GABA_A")])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, dt_ms=1.0, emitter="lif", duration_ms=20.0))


def check_layout(model):
    """Fixture must build the case it is named after: E at 0-7, PV at 8-9."""
    types = [row["cell_type"] for row in model.static["neuron_metadata"]]
    expected = ["E"] * 8 + ["PV"] * 2
    if types != expected:
        raise RuntimeError(f"fixture neuron layout {types} != {expected}")


def enc_fixture():
    """M2 fixture encoder: E g = 24 except -x (g = 0); b = 0; d0 = 1; PV no tonic input."""
    g = np.zeros(N, np.float32)
    g[:8] = G_FIXTURE
    g[1] = 0.0
    b = np.zeros(N, np.float32)
    return sp.SailorTonicEncoder(b=b, g=g, d0=D0)


def enc_zero():
    """All-zero encoder (diagnostic only)."""
    return sp.SailorTonicEncoder(b=np.zeros(N, np.float32), g=np.zeros(N, np.float32), d0=D0)


def fixture_drive_fn():
    """Fixture distance drive: sailor_tonic_current(enc_fixture, |p - target|)."""
    enc = enc_fixture()

    def drive(p):
        e = jnp.linalg.norm(jnp.asarray(p, jnp.float32) - jnp.zeros((2,), jnp.float32))
        return sp.sailor_tonic_current(enc, e)

    return drive


def rect_unit(z, d0=D0):
    """u(z) = max(0, z) / (max(0, z) + d0), elementwise."""
    zp = jnp.maximum(z, 0.0)
    return zp / (zp + d0)


def directional_drive_fn(g_dir=G_DIR, enc=None):
    """Option A signed-position encoder.

    Motor entries (+x, -x, +y, -y): g_dir * u(-p_x), g_dir * u(+p_x), g_dir * u(-p_y), g_dir * u(+p_y).
    Other entries keep the distance drive of ``enc`` (default: the fixture; E 4-7 keep g*e/(e+d0), PV 0).
    """
    enc = enc_fixture() if enc is None else enc
    motor_idx = jnp.asarray(MOTOR)

    def drive(p):
        p = jnp.asarray(p, jnp.float32)
        e = jnp.linalg.norm(p - jnp.zeros((2,), jnp.float32))
        base = sp.sailor_tonic_current(enc, e)
        signed = jnp.stack([
            g_dir * rect_unit(-p[0]),
            g_dir * rect_unit(p[0]),
            g_dir * rect_unit(-p[1]),
            g_dir * rect_unit(p[1]),
        ])
        return base.at[motor_idx].set(signed)

    return drive


def rollout_with_drive(model, cfg, enc_or_none, drive_fn, motor_idx, f0, n_intervals, seed, p0,
                       dt_ms=DT_MS, control_ms=CONTROL_MS):
    """Closed loop; body copied from paradigm.sailor_neural_rollout, with cur = drive_fn(state.p).

    Same continuation, pending-force timing (counts of interval k applied in interval k+1),
    stimulus_schedule construction and Simulation settings. Returns the same dict keys.
    """
    steps = int(round(control_ms / dt_ms))
    if steps < 1 or abs(steps * dt_ms - control_ms) > 1e-9:
        raise ValueError(f"control_ms ({control_ms}) must be a whole number of dt_ms ({dt_ms}) steps")
    if cfg.n_dim != 2 or cfg.n_act != 2:
        raise ValueError("the four-channel decoder drives a 2-D boat: n_dim and n_act must be 2")
    md = model.static["neuron_metadata"]
    n = len(md)
    motor = [int(i) for i in motor_idx]
    if len(motor) != 4 or len(set(motor)) != 4:
        raise ValueError(f"motor_idx must hold four distinct neurons, got {motor_idx!r}")
    for i in motor:
        if not (0 <= i < n) or md[i].get("cell_type") != "E":
            raise ValueError(f"motor neuron {i} must be an excitatory (E) neuron of the model")
    if enc_or_none is not None and np.shape(enc_or_none.b) != (n,):
        raise ValueError(f"encoder must have one value per neuron ({n}), got {np.shape(enc_or_none.b)}")

    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    state = sp.sailor_reset(
        cfg, jax.random.PRNGKey(seed),
        p0=None if p0 is None else np.asarray(p0, np.float32),
        v0=None,
    )
    cont = None
    # Contract section 12.3.4: counts of interval k are decoded and applied during interval k+1.
    pending = jnp.zeros((2,), sp._SAILOR_DTYPE)
    positions, errors, drives, spikes, volts, counts, actions = (
        [np.asarray(state.p)], [], [], [], [], [], [])
    for _ in range(n_intervals):
        e = float(jnp.linalg.norm(state.p - cfg.target))
        cur = np.asarray(drive_fn(state.p))
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
        c4 = sp_arr[:, motor].sum(axis=0)
        # The boat moves through interval k under the force decoded from interval k-1.
        state, _ = sp.sailor_step(cfg, state, pending)
        actions.append(np.asarray(pending))
        pending = sp.sailor_decode(c4, f0)
        errors.append(e)
        drives.append(cur)
        spikes.append(sp_arr)
        volts.append(np.asarray(sig.V_m))
        counts.append(c4)
        positions.append(np.asarray(state.p))
    return {
        "positions": np.stack(positions),
        "errors": np.asarray(errors),
        "drive": np.stack(drives),
        "spikes": np.concatenate(spikes, axis=0),
        "V_m": np.concatenate(volts, axis=0),
        "counts": np.stack(counts),
        "actions": np.stack(actions),
        "metadata": {
            "neural_dt_ms": float(dt_ms), "control_ms": float(control_ms),
            "physics_dt": float(cfg.dt), "motor_idx": motor, "f0": float(f0),
            "seed": int(seed), "n_intervals": int(n_intervals), "emitter": "lif",
        },
    }


def episode_metrics(errors, positions):
    """Pinned window = errors[K-K//5 : K]; occupancy = fraction of window with error <= 1.0."""
    errors = np.asarray(errors, float)
    n = errors.size
    lo = n - n // 5
    window = errors[lo:n]
    return {
        "final_distance": float(np.linalg.norm(positions[-1])),
        "min_distance": float(np.min(np.linalg.norm(positions, axis=1))),
        "pinned_window": [int(lo), int(n)],
        "pinned_mean_error": float(np.mean(window)),
        "fixation_occupancy": float(np.mean(window <= FIXATION_RADIUS)),
    }


def first_interval_sign(positions, actions, n=20):
    """Sign of force . (0 - p) for the force applied in intervals 0..n-1 (actions[k] acts at positions[k])."""
    pos = np.asarray(positions)[:n]
    force = np.asarray(actions)[:n]
    dot = np.sum(force * (-pos), axis=1)
    return {
        "toward_origin": int(np.sum(dot > 0)),
        "away_from_origin": int(np.sum(dot < 0)),
        "zero": int(np.sum(dot == 0)),
        "dot_force_to_origin": [float(x) for x in dot],
    }


def git_head():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def code_sha256():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def write_json(path, obj, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
    tmp.replace(path)


def base_meta():
    return {"git_head": git_head(), "code": str(Path(__file__).resolve().relative_to(REPO)),
            "code_sha256": code_sha256(), "sailor": CFG, "f0": F0_GAIN, "motor_idx": list(MOTOR),
            "non_motor_E": list(NON_MOTOR_E), "PV": list(PV), "neural_dt_ms": DT_MS,
            "control_ms": CONTROL_MS, "seed": SEED}


def gate_a0(model):
    cfg = sp.SailorConfig(**CFG)
    res = rollout_with_drive(model, cfg, enc_fixture(), fixture_drive_fn(), MOTOR, F0_GAIN,
                             K, SEED, p0=(R, 0.0))
    ref_rec = json.loads(F1_REF.read_text())["records"][0]
    if ref_rec["seed"] != SEED:
        raise RuntimeError(f"F1.json records[0] is seed {ref_rec['seed']}, expected {SEED}")
    ref = np.asarray(ref_rec["errors"][:K], float)
    got = np.asarray(res["errors"], float)
    if ref.size != K or got.size != K:
        raise RuntimeError(f"length mismatch: ref {ref.size}, got {got.size}")
    diff = float(np.max(np.abs(got - ref)))
    passed = diff <= A0_TOL
    out = {"gate": "A0", "name": "fixture replay vs F1.json records[0] (seed 0, first 600)",
           "criterion": f"max abs diff <= {A0_TOL}", "max_abs_diff": diff, "n": K,
           "ref_path": str(F1_REF.relative_to(REPO)), "p0": [R, 0.0], "theta": 0.0,
           "pass": bool(passed), "errors_first_last": [float(got[0]), float(got[-1])], **base_meta()}
    write_json(GATE_DIR / "gate_A0_fixture_replay.json", out)
    return out


def gate_a1():
    drive = directional_drive_fn(G_DIR)
    points = {"(-2,0)": (-2.0, 0.0), "(2,0)": (2.0, 0.0), "(0,-2)": (0.0, -2.0)}
    rows = {}
    for name, p in points.items():
        cur = np.asarray(drive(jnp.asarray(p, jnp.float32)))
        rows[name] = {"+x": float(cur[0]), "-x": float(cur[1]), "+y": float(cur[2]), "-y": float(cur[3])}
    checks = {
        "(-2,0): +x>0 and -x==0": rows["(-2,0)"]["+x"] > 0 and rows["(-2,0)"]["-x"] == 0,
        "(2,0): -x>0 and +x==0": rows["(2,0)"]["-x"] > 0 and rows["(2,0)"]["+x"] == 0,
        "(0,-2): +y>0 and -y==0": rows["(0,-2)"]["+y"] > 0 and rows["(0,-2)"]["-y"] == 0,
    }
    passed = all(checks.values())
    out = {"gate": "A1", "name": "sign of directional motor currents",
           "criterion": "signs as listed; zero entries exactly 0", "values": rows,
           "checks": checks, "pass": bool(passed), **base_meta()}
    write_json(GATE_DIR / "gate_A1_sign.json", out)
    return out


def gate_a2(model):
    cfg = sp.SailorConfig(**CFG)
    res = rollout_with_drive(model, cfg, enc_fixture(), directional_drive_fn(g_dir=0.0), MOTOR, F0_GAIN,
                             A2_INTERVALS, SEED, p0=(R, 0.0))
    dev = float(np.max(np.abs(np.asarray(res["errors"], float) - R)))
    diag = rollout_with_drive(model, cfg, enc_zero(), directional_drive_fn(g_dir=0.0, enc=enc_zero()),
                              MOTOR, F0_GAIN, A2_INTERVALS, SEED, p0=(R, 0.0))
    dev_diag = float(np.max(np.abs(np.asarray(diag["errors"], float) - R)))
    passed = dev < A2_TOL
    out = {"gate": "A2", "name": "negative control, g=0 on motor neurons (non-motor E keep distance drive)",
           "criterion": f"max |error - {R}| < {A2_TOL} over {A2_INTERVALS} intervals",
           "max_abs_deviation": dev, "n": A2_INTERVALS, "motor_spike_counts_total": int(res["counts"].sum()),
           "pass": bool(passed),
           "diagnostic_full_zero_encoder": {
               "note": "diagnostic only, not the gate: all E and PV current zero",
               "max_abs_deviation": dev_diag, "motor_spike_counts_total": int(diag["counts"].sum())},
           **base_meta()}
    write_json(GATE_DIR / "gate_A2_zero_motor_drive.json", out)
    return out


def run_gates(model):
    a0 = gate_a0(model)
    print(f"A0 max_abs_diff={a0['max_abs_diff']:.3e} pass={a0['pass']}", flush=True)
    results = {"A0": a0["pass"]}
    if not a0["pass"]:
        write_json(GATE_DIR / "gates_summary.json", {"gates": results, "all_pass": False, "stopped_at": "A0"})
        return False
    a1 = gate_a1()
    print(f"A1 values={a1['values']} pass={a1['pass']}", flush=True)
    results["A1"] = a1["pass"]
    a2 = gate_a2(model)
    print(f"A2 max_abs_deviation={a2['max_abs_deviation']:.3e} pass={a2['pass']} "
          f"motor_spikes={a2['motor_spike_counts_total']} diag_dev={a2['diagnostic_full_zero_encoder']['max_abs_deviation']:.3e}",
          flush=True)
    results["A2"] = a2["pass"]
    all_pass = all(results.values())
    write_json(GATE_DIR / "gates_summary.json", {"gates": results, "all_pass": all_pass, **base_meta()})
    return all_pass


def dev_starts():
    theta = np.random.default_rng(DEV_RNG_SEED).uniform(0.0, 2.0 * np.pi, N_DEV)
    p0 = 5.0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return theta, p0


def run_ex24(model):
    gs_path = GATE_DIR / "gates_summary.json"
    if not gs_path.exists() or not json.loads(gs_path.read_text()).get("all_pass"):
        raise SystemExit("refusing A3: gates_summary.json does not show A0-A2 all pass")
    cfg = sp.SailorConfig(**CFG)
    drive = directional_drive_fn(G_DIR)
    theta, p0s = dev_starts()
    rows = []
    for i in range(N_DEV):
        t0 = time.time()
        res = rollout_with_drive(model, cfg, enc_fixture(), drive, MOTOR, F0_GAIN, K, SEED,
                                 p0=p0s[i])
        m = episode_metrics(res["errors"], res["positions"])
        m["pass_occupancy_ge_0_9"] = bool(m["fixation_occupancy"] >= OCCUPANCY_MIN)
        rec = {"start_index": i, "theta": float(theta[i]), "p0": [float(x) for x in p0s[i]],
               "controller": "directional_option_A", "g_dir": G_DIR, "d0": D0,
               "K": K, "seed": SEED, **base_meta(), "metrics": m,
               "first20_force_sign": first_interval_sign(res["positions"], res["actions"], 20),
               "spikes_total_all_neurons": int(res["spikes"].sum()),
               "motor_counts_total": int(res["counts"].sum()),
               "errors": [float(x) for x in res["errors"]],
               "positions": [[float(x) for x in q] for q in res["positions"]],
               "counts": [[int(x) for x in c] for c in res["counts"]],
               "actions": [[float(x) for x in a] for a in res["actions"]]}
        write_json(EX24_DIR / f"start_{i:02d}.json", rec, indent=None)
        rows.append({"start_index": i, "theta": rec["theta"], "final_distance": m["final_distance"],
                     "min_distance": m["min_distance"], "pinned_mean_error": m["pinned_mean_error"],
                     "fixation_occupancy": m["fixation_occupancy"], "pass": m["pass_occupancy_ge_0_9"]})
        print(f"start {i:02d} theta={theta[i]:.4f} final={m['final_distance']:.4f} "
              f"min={m['min_distance']:.4f} pinned_mean={m['pinned_mean_error']:.4f} "
              f"occ={m['fixation_occupancy']:.4f} pass={m['pass_occupancy_ge_0_9']} "
              f"wall_s={time.time() - t0:.1f}", flush=True)
    n_pass = sum(r["pass"] for r in rows)
    verdict = "PASS" if n_pass == N_DEV else "FAIL"
    summary = {"gate": "A3", "name": "EX-24 directional positive control, 20 development starts",
               "criterion": f"every start fixation_occupancy >= {OCCUPANCY_MIN}; window errors[{K - K // 5}:{K}], radius {FIXATION_RADIUS}",
               "starts": "development only: numpy.random.default_rng(2025).uniform(0, 2*pi, 20), R=5",
               "n_pass": n_pass, "n_starts": N_DEV, "verdict": verdict, "rows": rows, **base_meta()}
    write_json(EX24_DIR / "summary.json", summary)
    print(f"A3 n_pass={n_pass}/{N_DEV} verdict={verdict}", flush=True)
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gates", "ex24"])
    args = ap.parse_args(argv)
    if not (REPO / "AGENTS.md").exists():
        raise SystemExit(f"REPO resolved to {REPO}, which has no AGENTS.md")
    model = build_model()
    check_layout(model)
    if args.command == "gates":
        run_gates(model)
    else:
        run_ex24(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
