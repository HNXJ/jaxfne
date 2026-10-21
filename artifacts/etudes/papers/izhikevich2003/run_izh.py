"""Runner and scorer for the signed Izhikevich 2003 P3 preregistration.

Spec: ``artifacts/etudes/izhikevich2003/PREREG_P3.md`` (signed v7); network
per ``PORT_SPEC.md``. Subcommands ``fig2``, ``fig3``, ``score``, ``sens``
(S2/S3/S4) and ``sens-report``. No scored run: an ``--out`` containing
"scored" is refused unless ``--signed`` is passed. Sensitivity S1 runs
through ``net_reference.py``; S2-S4 run here with manifest sensitivity
labels. A directory whose manifest carries a sensitivity label never
enters ``score``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
_ANALYSIS = HERE / "izh_analysis.py"
_spec = importlib.util.spec_from_file_location("izh_analysis_for_run", _ANALYSIS)
izh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(izh)

CALIBRATION_FILE = HERE / "calibration" / "calibration_v7_15f36087.json"
GAMMA_BAND = (30.0, 50.0)
DISCARD_MS = 200.0

FIG2_DT = 0.1
FIG2_DTYPE = "float32"
FIG2_TYPES = {
    "RS": (0.02, 0.2, -65.0, 8.0),
    "IB": (0.02, 0.2, -55.0, 4.0),
    "CH": (0.02, 0.2, -50.0, 2.0),
    "FS": (0.1, 0.2, -65.0, 2.0),
    "LTS": (0.02, 0.25, -65.0, 2.0),
    "TC": (0.02, 0.25, -65.0, 0.05),
    "RZ": (0.1, 0.26, -65.0, 2.0),
}
FIG2_RUNS = {
    # run_id: (type, segments in (ms, current), event row, steps, kind)
    "C1": ("RS", ((10.0, 0.0), (310.0, 10.0)), 100, 3200, "step"),
    "C2": ("IB", ((10.0, 0.0), (310.0, 10.0)), 100, 3200, "step"),
    "C3": ("CH", ((10.0, 0.0), (310.0, 10.0)), 100, 3200, "step"),
    "C4": ("FS", ((10.0, 0.0), (310.0, 10.0)), 100, 3200, "step"),
    "C5": ("LTS", ((10.0, 0.0), (310.0, 10.0)), 100, 3200, "step"),
    "C7": ("TC", ((10.0, 0.0), (100.0, -36.5), (410.0, 0.0)), 1100, 5200, "release"),
    "C8": ("RZ", ((10.0, 0.0), (1.0, 5.0), (311.0, 0.0)), 110, 3220, "pulse"),
}
C6_AMPLITUDES = list(range(1, 11))
FIG2_RUNS_C6_SEGS = {i: ((10.0, 0.0), (310.0, float(i))) for i in C6_AMPLITUDES}
FIG3_DT = 0.5
FIG3_DTYPE = "float32"
FIG3_SEEDS = [1, 2, 3, 4, 5]
FIG3_T = 10.2

SENS_IDS = ("S2", "S3", "S4")
SENS_SEED = 1  # S2/S3 run the Fig. 3 network at seed 1 (PREREG_P3.md Sensitivity)
S3_DT = 0.25
S3_W_MULTIPLE = 4.0  # W = 4S, the paper's S (scored: 2S)
# S3 realizes 2 x the paper's sigma (10 E, 4 I) at dt 0.25 (PREREG_P3.md S3). noise_scale is the
# coefficient at DT_REF_MS = 0.5 and the kernel scales it by sqrt(0.5 / dt) (jaxfne/emitters.py
# _noise_coef, D1), so the parameter is sigma x sqrt(2), as in the scored run.
S3_NOISE_E = 5.0 * np.sqrt(2.0)
S3_NOISE_I = 2.0 * np.sqrt(2.0)
S3_REALIZED_NOISE = (10.0, 4.0)
S4_I_STEP = -10.0  # C7 step current (scored: -36.5; fixed point -76.0 mV)


def fixed_point_vstar(b):
    """Resting fixed point at I = 0: the lower root of 0.04 v^2 + (5-b) v + 140.

    Computed in float64 (Python float) and cast by the caller.
    """
    disc = (5.0 - b) ** 2 - 4.0 * 0.04 * 140.0
    return (-(5.0 - b) - np.sqrt(disc)) / (2.0 * 0.04)


def check_out(out, signed):
    """Refuse an --out containing "scored" unless --signed is passed."""
    if "scored" in str(out).lower() and not signed:
        raise SystemExit("refusing --out containing 'scored' without --signed")


def _repo_root():
    p = Path(__file__).resolve()
    for q in (p,) + tuple(p.parents):
        if (q / ".git").exists():
            return q
    return HERE.parents[3]


def _git_sha():
    try:
        out = subprocess.run(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        return out or "unknown"
    except OSError:
        return "unknown"


def _jax_backend():
    import jax

    return jax.devices()[0].platform


def load_calibration():
    receipt = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return float(receipt["theta"]), (float(receipt["alpha_band"][0]),
                                     float(receipt["alpha_band"][1]))


def fig3_keys(seed):
    """k_r, k_w, k_noise for the parameter draw, the weights, the kernel noise."""
    import jax

    k_r, k_w, k_noise = jax.random.split(jax.random.PRNGKey(seed), 3)
    return k_r, k_w, k_noise


def _segments_to_schedule(segments, n_steps, dt_ms):
    sched = np.zeros(n_steps)
    row = 0
    for ms, cur in segments:
        n = int(round(ms / dt_ms))
        sched[row:row + n] = cur
        row += n
    assert row == n_steps
    return sched


def run_single_cell(type_name, segments, n_steps, dt_ms=FIG2_DT,
                    dtype=FIG2_DTYPE, key_seed=0):
    """One Fig. 2 kernel run from the resting fixed point; returns (v, spike_rows)."""
    import jax
    import jax.numpy as jnp

    from jaxfne.emitters import IzhikevichParams, simulate_eig_izhikevich

    a, b, c, d = FIG2_TYPES[type_name]
    vstar = fixed_point_vstar(b)
    jdtype = jnp.float32 if dtype == "float32" else jnp.float64
    params = IzhikevichParams(
        a=jnp.asarray([a], dtype=jdtype), b=jnp.asarray([b], dtype=jdtype),
        c=jnp.asarray([c], dtype=jdtype), d=jnp.asarray([d], dtype=jdtype),
        drive=jnp.zeros((1,), dtype=jdtype), sign=jnp.ones((1,), dtype=jdtype),
        W=jnp.zeros((1, 1), dtype=jdtype),
        v0=jnp.asarray([vstar], dtype=jdtype),
        u0=jnp.asarray([b * vstar], dtype=jdtype),
        source_scale=jnp.ones((1,), dtype=jdtype), labels=(type_name,))
    sched = _segments_to_schedule(segments, n_steps, dt_ms)
    voltages, spikes, _ = simulate_eig_izhikevich(
        params, n_steps, dt_ms, jax.random.PRNGKey(key_seed), dtype=dtype,
        drive_schedule=jnp.asarray(sched, dtype=jdtype).reshape(n_steps, 1),
        noise_scale=0.0)
    v = np.asarray(voltages[:, 0], dtype=float)
    rows = [int(r) for r in np.flatnonzero(np.asarray(spikes[:, 0]))]
    return v, rows


def expected_fig2_settings():
    """Prereg settings per run id; the scorer compares the manifest against these."""
    out = {}
    for run_id, (t, segs, event, n, kind) in FIG2_RUNS.items():
        a, b, c, d = FIG2_TYPES[t]
        out[run_id] = {"type": t, "a": a, "b": b, "c": c, "d": d,
                       "vstar": fixed_point_vstar(b),
                       "segments": [[ms, cur] for ms, cur in segs],
                       "event_row": event, "n_steps": n, "dt_ms": FIG2_DT,
                       "dtype": FIG2_DTYPE, "key_seed": 0,
                       "noise_scale": 0.0, "kind": kind}
    for i in C6_AMPLITUDES:
        a, b, c, d = FIG2_TYPES["TC"]
        out[f"C6_I{i}"] = {"type": "TC", "a": a, "b": b, "c": c, "d": d,
                           "vstar": fixed_point_vstar(b),
                           "segments": [[10.0, 0.0], [310.0, float(i)]],
                           "event_row": 100, "n_steps": 3200, "dt_ms": FIG2_DT,
                           "dtype": FIG2_DTYPE, "key_seed": 0,
                           "noise_scale": 0.0, "kind": "tonic", "I_plus": float(i)}
    return out


def cmd_fig2(out_dir, sensitivity=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = expected_fig2_settings()
    manifest_runs = {}
    c6_passed = False
    for run_id in list(FIG2_RUNS) + [f"C6_I{i}" for i in C6_AMPLITUDES]:
        if run_id.startswith("C6_"):
            if c6_passed:
                break  # ascending search stops at the first passing amplitude
            i = int(run_id.split("C6_I")[1])
            settings = expected[run_id]
            v, rows = run_single_cell(settings["type"], FIG2_RUNS_C6_SEGS[i],
                                      settings["n_steps"])
        else:
            settings = expected[run_id]
            t, segs, event, n, kind = FIG2_RUNS[run_id]
            v, rows = run_single_cell(t, segs, n)
        np.savez_compressed(out / f"{run_id}.npz",
                            spike_rows=np.asarray(rows, dtype=np.int64), v=v)
        manifest_runs[run_id] = dict(settings, file=f"{run_id}.npz",
                                     n_spikes=len(rows))
        if run_id.startswith("C6_"):
            win = izh.window_rows(rows, settings["event_row"], FIG2_DT)
            c6_passed = _c6_run_ok(win, FIG2_DT)
    manifest = {"prereg": "PREREG_P3.md signed v7, Fig. 2",
                "code_sha": _git_sha(), "backend": _jax_backend(),
                "kernel": "jaxfne.emitters.simulate_eig_izhikevich",
                "dtype": FIG2_DTYPE, "dt_ms": FIG2_DT,
                "sensitivity": sensitivity, "runs": manifest_runs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest_runs)} Fig. 2 runs to {out}")
    return manifest


def build_fig3_network(seed):
    """1000-cell network per PORT_SPEC: W = 2S, noise_scale = paper sigma x sqrt(2)."""
    import jax
    import jax.numpy as jnp

    from jaxfne.emitters import IzhikevichParams

    k_r, k_w, k_noise = fig3_keys(seed)
    r = jax.random.uniform(k_r, (1000,))
    r_e, r_i = r[:800], r[800:]
    a = jnp.concatenate([jnp.full((800,), 0.02), 0.02 + 0.08 * r_i])
    b = jnp.concatenate([jnp.full((800,), 0.2), 0.25 - 0.05 * r_i])
    c = jnp.concatenate([-65.0 + 15.0 * r_e ** 2, jnp.full((200,), -65.0)])
    d = jnp.concatenate([8.0 - 6.0 * r_e ** 2, jnp.full((200,), 2.0)])
    u = jax.random.uniform(k_w, (1000, 1000))
    col_e = jnp.arange(1000)[None, :] < 800
    s = jnp.where(col_e, 0.5 * u, -u)
    w = 2.0 * s
    noise = jnp.concatenate([jnp.full((800,), 5.0 * np.sqrt(2.0)),
                             jnp.full((200,), 2.0 * np.sqrt(2.0))])
    params = IzhikevichParams(
        a=a, b=b, c=c, d=d, drive=jnp.zeros((1000,)),
        sign=jnp.concatenate([jnp.ones((800,)), -jnp.ones((200,))]), W=w,
        v0=jnp.full((1000,), -65.0), u0=b * -65.0,
        source_scale=jnp.ones((1000,)), labels=("E",) * 800 + ("I",) * 200)
    return params, noise, (k_r, k_w, k_noise)


def run_fig3_seed(seed, t_s):
    """One Fig. 3 kernel run; returns (rows, cols) spike indices."""
    from jaxfne.emitters import simulate_eig_izhikevich

    params, noise, keys = build_fig3_network(seed)
    n_steps = int(round(t_s * 1000.0 / FIG3_DT))
    _, spikes, _ = simulate_eig_izhikevich(
        params, n_steps, FIG3_DT, keys[2], dtype=FIG3_DTYPE, noise_scale=noise)
    idx = np.flatnonzero(np.asarray(spikes))
    rows = (idx // 1000).astype(np.int64)
    cols = (idx % 1000).astype(np.int64)
    key_hex = ["-".join(str(int(x)) for x in np.asarray(k)) for k in keys]
    return rows, cols, n_steps, key_hex


def cmd_fig3(out_dir, seeds, t_s, sensitivity=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for seed in seeds:
        rows, cols, n_steps, key_hex = run_fig3_seed(seed, t_s)
        np.savez_compressed(out / f"{seed}.npz",
                            spike_rows=rows, spike_cols=cols,
                            n_steps=np.int64(n_steps),
                            n_cells=np.int64(1000), dt_ms=np.float64(FIG3_DT))
        entries[str(seed)] = {"seed": seed, "file": f"{seed}.npz",
                              "n_steps": n_steps, "dt_ms": FIG3_DT,
                              "dtype": FIG3_DTYPE, "T_s": t_s,
                              "W": "2S", "noise": "paper_sigma*sqrt2",
                              "key_order": ["k_r", "k_w", "k_noise"],
                              "key_hex": key_hex, "n_spikes": int(rows.size)}
    manifest = {"prereg": "PREREG_P3.md signed v7, Fig. 3",
                "code_sha": _git_sha(), "backend": _jax_backend(),
                "kernel": "jaxfne.emitters.simulate_eig_izhikevich",
                "dtype": FIG3_DTYPE, "dt_ms": FIG3_DT, "T_s": t_s,
                "seeds": list(seeds), "sensitivity": sensitivity, "runs": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(entries)} Fig. 3 seeds to {out}")
    return manifest


# --- sensitivity S2/S3/S4 (reported only) ---------------------------------------------------------
#
# Spec: PREREG_P3.md "Sensitivity" (S2: float64 seed 1 with jax_enable_x64 on;
# S3: dt 0.25, W = 4S, noise 2 x paper sigma, seed 1; S4: C7 with I = -10).
# The scored builders above are untouched; the sens builder below repeats
# their expressions so that at scored settings it reproduces them exactly.


def _ensure_x64():
    """Enable jax_enable_x64 now, before the S2 run builds any array."""
    import jax

    if not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)


def build_sens_fig3_network(seed, dtype=FIG3_DTYPE, w_multiple=2.0,
                            noise_e=5.0 * np.sqrt(2.0), noise_i=2.0 * np.sqrt(2.0)):
    """Fig. 3 network with sensitivity settings; scored defaults reproduce build_fig3_network.

    ``w_multiple`` scales the paper's S (scored: 2.0); ``noise_e``/``noise_i``
    are the noise_scale parameters (scored: paper sigma x sqrt(2)). At
    ``dtype="float64"`` x64 is enabled first and the arrays are cast (exact).
    """
    import jax
    import jax.numpy as jnp

    from jaxfne.emitters import IzhikevichParams

    if dtype == "float64":
        _ensure_x64()
    k_r, k_w, k_noise = fig3_keys(seed)
    r = jax.random.uniform(k_r, (1000,))
    r_e, r_i = r[:800], r[800:]
    a = jnp.concatenate([jnp.full((800,), 0.02), 0.02 + 0.08 * r_i])
    b = jnp.concatenate([jnp.full((800,), 0.2), 0.25 - 0.05 * r_i])
    c = jnp.concatenate([-65.0 + 15.0 * r_e ** 2, jnp.full((200,), -65.0)])
    d = jnp.concatenate([8.0 - 6.0 * r_e ** 2, jnp.full((200,), 2.0)])
    u = jax.random.uniform(k_w, (1000, 1000))
    col_e = jnp.arange(1000)[None, :] < 800
    s = jnp.where(col_e, 0.5 * u, -u)
    w = float(w_multiple) * s
    noise = jnp.concatenate([jnp.full((800,), float(noise_e)),
                             jnp.full((200,), float(noise_i))])
    if dtype == "float64":
        a, b, c, d, w, noise = (x.astype(jnp.float64) for x in (a, b, c, d, w, noise))
        v0 = jnp.full((1000,), -65.0, dtype=jnp.float64)
        u0 = (b * -65.0).astype(jnp.float64)
        drive = jnp.zeros((1000,), dtype=jnp.float64)
        sign = jnp.concatenate([jnp.ones((800,), dtype=jnp.float64),
                                -jnp.ones((200,), dtype=jnp.float64)])
        source_scale = jnp.ones((1000,), dtype=jnp.float64)
    else:
        v0 = jnp.full((1000,), -65.0)
        u0 = b * -65.0
        drive = jnp.zeros((1000,))
        sign = jnp.concatenate([jnp.ones((800,)), -jnp.ones((200,))])
        source_scale = jnp.ones((1000,))
    params = IzhikevichParams(
        a=a, b=b, c=c, d=d, drive=drive, sign=sign, W=w,
        v0=v0, u0=u0, source_scale=source_scale,
        labels=("E",) * 800 + ("I",) * 200)
    return params, noise, (k_r, k_w, k_noise)


def run_sens_fig3_seed(seed, t_s, dtype, dt_ms, w_multiple, noise_e, noise_i):
    """One sensitivity Fig. 3 run; asserts the kernel output dtype. Returns rows, cols, n_steps, keys, dtype."""
    import jax.numpy as jnp

    from jaxfne.emitters import simulate_eig_izhikevich

    params, noise, keys = build_sens_fig3_network(seed, dtype, w_multiple, noise_e, noise_i)
    n_steps = int(round(t_s * 1000.0 / dt_ms))
    voltages, spikes, _ = simulate_eig_izhikevich(
        params, n_steps, dt_ms, keys[2], dtype=dtype, noise_scale=noise)
    want = jnp.float64 if dtype == "float64" else jnp.float32
    if voltages.dtype != want:
        raise SystemExit(f"S2/sens run built {voltages.dtype}, expected {want}")
    idx = np.flatnonzero(np.asarray(spikes))
    rows = (idx // 1000).astype(np.int64)
    cols = (idx % 1000).astype(np.int64)
    key_hex = ["-".join(str(int(x)) for x in np.asarray(k)) for k in keys]
    return rows, cols, n_steps, key_hex, str(voltages.dtype)


def _write_sens_fig3(out_dir, sens_id, dtype, dt_ms, t_s, w_label, w_multiple,
                     noise_label, noise_e, noise_i):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, cols, n_steps, key_hex, out_dtype = run_sens_fig3_seed(
        SENS_SEED, t_s, dtype, dt_ms, w_multiple, noise_e, noise_i)
    np.savez_compressed(out / f"{SENS_SEED}.npz",
                        spike_rows=rows, spike_cols=cols,
                        n_steps=np.int64(n_steps),
                        n_cells=np.int64(1000), dt_ms=np.float64(dt_ms))
    entries = {str(SENS_SEED): {"seed": SENS_SEED, "file": f"{SENS_SEED}.npz",
                                "n_steps": n_steps, "dt_ms": dt_ms,
                                "dtype": dtype, "output_dtype": out_dtype,
                                "T_s": t_s, "W": w_label,
                                "noise": noise_label,
                                "noise_ref_E": float(noise_e),
                                "noise_ref_I": float(noise_i),
                                "noise_realized_E": float(noise_e * np.sqrt(0.5 / dt_ms)),
                                "noise_realized_I": float(noise_i * np.sqrt(0.5 / dt_ms)),
                                "key_order": ["k_r", "k_w", "k_noise"],
                                "key_hex": key_hex, "n_spikes": int(rows.size)}}
    manifest = {"prereg": "PREREG_P3.md signed v7, Fig. 3",
                "code_sha": _git_sha(), "backend": _jax_backend(),
                "kernel": "jaxfne.emitters.simulate_eig_izhikevich",
                "dtype": dtype, "output_dtype": out_dtype,
                "dt_ms": dt_ms, "T_s": t_s,
                "seeds": [SENS_SEED], "sensitivity": sens_id, "runs": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote sensitivity {sens_id} to {out}")
    return manifest


def _write_sens_s4(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = expected_fig2_settings()["C7"]
    segments = ((10.0, 0.0), (100.0, S4_I_STEP), (410.0, 0.0))
    v, rows = run_single_cell("TC", segments, scored["n_steps"])
    np.savez_compressed(out / "C7.npz",
                        spike_rows=np.asarray(rows, dtype=np.int64), v=v)
    settings = dict(scored, segments=[[ms, cur] for ms, cur in segments],
                    file="C7.npz", n_spikes=len(rows))
    manifest = {"prereg": "PREREG_P3.md signed v7, Fig. 2",
                "code_sha": _git_sha(), "backend": _jax_backend(),
                "kernel": "jaxfne.emitters.simulate_eig_izhikevich",
                "dtype": FIG2_DTYPE, "dt_ms": FIG2_DT,
                "sensitivity": "S4", "runs": {"C7": settings}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote sensitivity S4 to {out}")
    return manifest


def cmd_sens(sens_id, out_dir, t_s=FIG3_T):
    """Sensitivity run S2/S3/S4; the manifest label keeps it out of ``score``."""
    if sens_id == "S2":
        return _write_sens_fig3(out_dir, "S2", "float64", FIG3_DT, t_s, "2S", 2.0,
                                "paper_sigma*sqrt2", 5.0 * np.sqrt(2.0), 2.0 * np.sqrt(2.0))
    if sens_id == "S3":
        return _write_sens_fig3(out_dir, "S3", FIG3_DTYPE, S3_DT, t_s, "4S",
                                S3_W_MULTIPLE, "2x_paper_sigma_realized", S3_NOISE_E, S3_NOISE_I)
    if sens_id == "S4":
        return _write_sens_s4(out_dir)
    raise SystemExit(f"unknown sensitivity id {sens_id!r}; want one of {SENS_IDS}")


# --- scorer -------------------------------------------------------------------------------------


def _c6_run_ok(win, dt_ms):
    return (len(win) >= 3 and not izh.bursts(win, dt_ms)
            and (izh.isi_cv(win) is not None) and izh.isi_cv(win) <= 0.3)


def _numbers(win, dt_ms):
    return {"n_spikes": len(win), "rate_hz": izh.rate_hz(win),
            "isi_ratio": izh.isi_ratio(win), "isi_cv": izh.isi_cv(win)}


def eval_c1(rows, event_row, dt_ms):
    win = izh.window_rows(rows, event_row, dt_ms)
    ratio = izh.isi_ratio(win)
    ok = len(win) >= 4 and ratio is not None and ratio >= 1.5
    return {"verdict": "pass" if ok else "fail", "numbers": _numbers(win, dt_ms)}


def eval_c2(rows, event_row, dt_ms):
    win = izh.window_rows(rows, event_row, dt_ms)
    ok = izh.c2_burst_then_tail(win, dt_ms)
    found = izh.bursts(win, dt_ms)
    tail_n = len(win) - len(found[0]) if (win and found and found[0][0] == win[0]) else 0
    numbers = _numbers(win, dt_ms)
    numbers.update({"initial_burst": bool(win and found and found[0][0] == win[0]),
                    "tail_n_spikes": tail_n})
    return {"verdict": "pass" if ok else "fail", "numbers": numbers}


def eval_c3(rows, event_row, dt_ms):
    win = izh.window_rows(rows, event_row, dt_ms)
    n_b = len(izh.bursts(win, dt_ms))
    freq = izh.burst_onset_frequency_hz(win, dt_ms)
    ok = n_b >= 3 and freq is not None and 15.0 <= freq <= 50.0
    numbers = _numbers(win, dt_ms)
    numbers.update({"n_bursts": n_b, "onset_freq_hz": freq})
    return {"verdict": "pass" if ok else "fail", "numbers": numbers}


def eval_c4(rows, event_row, dt_ms):
    win = izh.window_rows(rows, event_row, dt_ms)
    ratio = izh.isi_ratio(win)
    ok = izh.rate_hz(win) >= 50.0 and ratio is not None and ratio <= 1.2
    return {"verdict": "pass" if ok else "fail", "numbers": _numbers(win, dt_ms)}


def eval_c5(rows, event_row, dt_ms, rs_rate_hz):
    win = izh.window_rows(rows, event_row, dt_ms)
    ratio = izh.isi_ratio(win)
    rate = izh.rate_hz(win)
    ok = (len(win) >= 4 and rate > rs_rate_hz
          and ratio is not None and ratio >= 1.2)
    numbers = _numbers(win, dt_ms)
    numbers.update({"rs_rate_hz": rs_rate_hz})
    return {"verdict": "pass" if ok else "fail", "numbers": numbers}


def eval_c6_search(runs_in_order, dt_ms):
    """runs_in_order: list of (I_plus, rows, event_row). Ascending search rule."""
    outcomes = []
    for i_plus, rows, event_row in runs_in_order:
        win = izh.window_rows(rows, event_row, dt_ms)
        cv = izh.isi_cv(win)
        ok = _c6_run_ok(win, dt_ms)
        outcomes.append({"I_plus": i_plus, "n_spikes": len(win),
                         "has_burst": bool(izh.bursts(win, dt_ms)),
                         "isi_cv": cv, "ok": ok})
    selected = next((o["I_plus"] for o in outcomes if o["ok"]), None)
    return {"verdict": "pass" if selected is not None else "fail",
            "numbers": {"outcomes": outcomes, "selected_I_plus": selected}}


def eval_c7(rows, release_row, dt_ms):
    win = izh.window_rows(rows, release_row, dt_ms)
    lat = None
    for burst in izh.bursts(win, dt_ms):
        gap = burst[0] - release_row
        if 0 <= gap < round(50.0 / dt_ms):
            lat = gap * dt_ms
            break
    ok = lat is not None
    numbers = _numbers(win, dt_ms)
    numbers.update({"burst_latency_ms": lat})
    return {"verdict": "pass" if ok else "fail", "numbers": numbers}


def eval_c8(v, start_row, dt_ms, rows):
    win_spikes = [int(r) for r in rows]  # "no spike" (PREREG_P3.md C8): the whole run
    maxima = izh.c8_maxima(np.asarray(v, dtype=float), start_row)
    freq = izh.c8_frequency_hz(maxima, dt_ms)
    decreasing = bool(len(maxima) >= 2 and v[maxima[1]] < v[maxima[0]])
    ok = (not win_spikes) and len(maxima) >= 2 and decreasing
    return {"verdict": "pass" if ok else "fail",
            "numbers": {"n_spikes": len(win_spikes), "n_maxima": len(maxima),
                        "decreasing": decreasing, "freq_hz": freq,
                        "maxima_rows": [int(m) for m in maxima]}}


def _safe_peak_excess(signal, fs, band):
    if len(signal) < izh.WELCH_NPERSEG:
        return {"f_pk": None, "pe": None, "local_max": False, "edge_ok": False}
    try:
        return izh.peak_excess(np.asarray(signal, dtype=float), fs, band)
    except (ValueError, RuntimeError):
        return {"f_pk": None, "pe": None, "local_max": False, "edge_ok": False}


def _kept_window_regime(cells_rows, n_steps, dt_ms, populations=None):
    """Per-population activity regimes over the kept window (T-23 part 2).

    Builds the (kept_steps, n_cells) spike matrix once and classifies each
    named population with ``jaxfne.analysis.activity_regime``. ``None``
    reports the whole network as ``"all"``. N1-N4 are untouched.
    """
    from jaxfne.analysis import activity_regime as _activity_regime

    n_cells = len(cells_rows)
    discard = int(round(DISCARD_MS / dt_ms))
    kept_steps = n_steps - discard
    mat = np.zeros((max(kept_steps, 0), n_cells), dtype=bool)
    for j, rows in enumerate(cells_rows):
        for row in rows:
            k = int(row) - discard
            if 0 <= k < kept_steps:
                mat[k, j] = True
    pops = dict(populations) if populations else {"all": np.arange(n_cells)}
    regimes = {name: _activity_regime(mat[:, np.asarray(idx)], float(dt_ms))
               for name, idx in pops.items()}
    first = next(iter(regimes.values()))
    return {"populations": regimes,
            "labels": {name: reg["label"] for name, reg in regimes.items()},
            "window_steps": [discard, n_steps],
            "window_ms": [discard * float(dt_ms), n_steps * float(dt_ms)],
            "bin_ms": first["thresholds"]["bin_ms"],
            "dt_ms": float(dt_ms)}


def score_network_seed(cells_rows, n_steps, dt_ms, theta, alpha_band, gamma_band=GAMMA_BAND,
                       populations=None):
    """N1-N4 for one seed. cells_rows: per-cell absolute spike rows.

    ``populations`` maps a population name to its cell columns; the default
    (None) reports the whole network as ``"all"``. T-23 part 2 (Hamm
    2026-10-14): the returned dict also carries ``activity_regime`` with the
    per-population regime labels and measured values (rate, active fraction,
    kappa, CV_ISI) over the kept window, plus the window and bin. N1-N4 are
    untouched.
    """
    n_cells = len(cells_rows)
    discard = int(round(DISCARD_MS / dt_ms))
    kept = [[int(r) for r in rows if int(r) >= discard] for rows in cells_rows]
    kept_s = (n_steps - discard) * dt_ms / 1000.0
    n1 = sum(len(w) for w in kept) / (n_cells * kept_s) if kept_s > 0 else 0.0
    cvs, n_qual = [], 0
    for w in kept:
        if len(w) >= 10:
            n_qual += 1
            cvs.append(izh.isi_cv(w))
    n2 = float(np.median(cvs)) if cvs else None
    steps_per_bin = int(round(1.0 / dt_ms))
    bins = {}
    for w in kept:
        for r in w:
            b = (r - discard) // steps_per_bin
            bins[b] = bins.get(b, 0) + 1
    n_bins = (n_steps - discard) // steps_per_bin
    signal = np.array([bins.get(b, 0) for b in range(n_bins)], dtype=float)
    alpha = _safe_peak_excess(signal, 1000.0, alpha_band)
    gamma = _safe_peak_excess(signal, 1000.0, gamma_band)
    return {
        "N1": {"verdict": "pass" if 4.0 <= n1 <= 16.0 else "fail",
               "numbers": {"mean_rate_hz": n1}},
        "N2": {"verdict": "pass" if (n_qual >= 500 and n2 is not None
                                     and 0.6 <= n2 <= 1.5) else "fail",
               "numbers": {"median_cv": n2, "n_qualifying_cells": n_qual}},
        "N3": {"verdict": "pass" if izh.holds(alpha, theta) else "fail",
               "numbers": {"f_pk": alpha["f_pk"], "pe": alpha["pe"],
                           "local_max": alpha["local_max"], "edge_ok": alpha["edge_ok"]}},
        "N4": {"verdict": "pass" if izh.holds(gamma, theta) else "fail",
               "numbers": {"f_pk": gamma["f_pk"], "pe": gamma["pe"],
                           "local_max": gamma["local_max"], "edge_ok": gamma["edge_ok"]}},
        "activity_regime": _kept_window_regime(cells_rows, n_steps, dt_ms, populations),
    }


def _load_run_npz(fig_dir, run_id):
    z = np.load(Path(fig_dir) / f"{run_id}.npz")
    return [int(r) for r in np.asarray(z["spike_rows"]).tolist()], \
        (np.asarray(z["v"], dtype=float).tolist() if "v" in z else None)


def score_fig2_dir(fig_dir, theta, alpha_band):
    manifest = json.loads((Path(fig_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sensitivity"):
        raise SystemExit("sensitivity runs never enter score")
    expected = expected_fig2_settings()
    out = {}
    data = {}

    def _get(run_id):
        if run_id not in manifest.get("runs", {}):
            return "incomplete", None
        got = {k: v for k, v in manifest["runs"][run_id].items()
               if k in expected.get(run_id, {})}
        if got != expected.get(run_id):
            return "incomplete", None
        try:
            rows, v = _load_run_npz(fig_dir, run_id)
        except OSError:
            return "incomplete", None
        return "ok", (rows, v)

    for run_id in ("C1", "C2", "C3", "C4", "C5", "C7", "C8"):
        status, payload = _get(run_id)
        if status == "incomplete":
            out[run_id] = {"verdict": "incomplete", "numbers": {}}
        else:
            data[run_id] = payload
    # C6: the ascending search must stop at the first passing amplitude.
    c6_ids = [f"C6_I{i}" for i in C6_AMPLITUDES if f"C6_I{i}" in manifest.get("runs", {})]
    c6_ids_sorted = sorted(c6_ids, key=lambda s: int(s.split("C6_I")[1]))
    c6_amps = [int(s.split("C6_I")[1]) for s in c6_ids_sorted]
    c6_complete = bool(c6_amps) and c6_amps == list(range(1, len(c6_amps) + 1))
    c6_payloads = []
    for rid in c6_ids_sorted:
        got = {k: v for k, v in manifest["runs"][rid].items() if k in expected[rid]}
        if got != expected[rid]:
            c6_complete = False
            break
        try:
            rows, _ = _load_run_npz(fig_dir, rid)
        except OSError:
            c6_complete = False
            break
        c6_payloads.append((float(manifest["runs"][rid]["I_plus"]), rows,
                            expected[rid]["event_row"]))
    c6_runs = eval_c6_search(c6_payloads, FIG2_DT) if (c6_complete and c6_payloads) else None
    if c6_runs is not None:
        n_run = len(c6_payloads)
        oks = [o["ok"] for o in c6_runs["numbers"]["outcomes"]]
        stop_ok = ((not any(oks)) and n_run == 10
                   or (any(oks) and oks.index(True) == n_run - 1))
        c6_complete = c6_complete and stop_ok
    if not c6_complete or c6_runs is None:
        out["C6"] = {"verdict": "incomplete", "numbers": {}}
    else:
        out["C6"] = c6_runs
    if "C1" in data:
        rows, _ = data["C1"]
        out["C1"] = eval_c1(rows, expected["C1"]["event_row"], FIG2_DT)
    if "C2" in data:
        rows, _ = data["C2"]
        out["C2"] = eval_c2(rows, expected["C2"]["event_row"], FIG2_DT)
    if "C3" in data:
        rows, _ = data["C3"]
        out["C3"] = eval_c3(rows, expected["C3"]["event_row"], FIG2_DT)
    if "C4" in data:
        rows, _ = data["C4"]
        out["C4"] = eval_c4(rows, expected["C4"]["event_row"], FIG2_DT)
    if "C5" in data:
        rows, _ = data["C5"]
        if out["C1"].get("verdict") == "incomplete":
            out["C5"] = {"verdict": "incomplete", "numbers": {}}
        else:
            out["C5"] = eval_c5(rows, expected["C5"]["event_row"], FIG2_DT,
                                out["C1"]["numbers"]["rate_hz"])
    if "C7" in data:
        rows, _ = data["C7"]
        out["C7"] = eval_c7(rows, expected["C7"]["event_row"], FIG2_DT)
    if "C8" in data:
        rows, v = data["C8"]
        out["C8"] = eval_c8(v, expected["C8"]["event_row"], FIG2_DT, rows)
    verdicts = {k: out[k]["verdict"] for k in
                ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}
    overall = ("pass" if all(v == "pass" for v in verdicts.values())
               else "incomplete" if any(v == "incomplete" for v in verdicts.values())
               else "fail")
    return {"criteria": out, "verdicts": verdicts, "overall": overall}


def score_fig3_dir(fig_dir, theta, alpha_band):
    manifest = json.loads((Path(fig_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sensitivity"):
        raise SystemExit("sensitivity runs never enter score")
    complete = (manifest.get("seeds") == FIG3_SEEDS
                and abs(float(manifest.get("T_s", -1)) - FIG3_T) < 1e-12
                and abs(float(manifest.get("dt_ms", -1)) - FIG3_DT) < 1e-12
                and manifest.get("dtype") == FIG3_DTYPE)
    seeds_out = {}
    for seed in FIG3_SEEDS:
        entry = manifest.get("runs", {}).get(str(seed))
        if entry is None or not complete:
            seeds_out[str(seed)] = {"verdict": "incomplete", "numbers": {}}
            continue
        try:
            z = np.load(Path(fig_dir) / f"{seed}.npz")
            n_steps = int(z["n_steps"])
            n_cells = int(z["n_cells"])
            srows = np.asarray(z["spike_rows"]).tolist()
            scols = np.asarray(z["spike_cols"]).tolist()
        except OSError:
            seeds_out[str(seed)] = {"verdict": "incomplete", "numbers": {}}
            continue
        if (n_steps != entry.get("n_steps") or n_steps != round(FIG3_T * 1000.0 / FIG3_DT)
                or n_cells != 1000 or len(srows) != entry.get("n_spikes")):
            seeds_out[str(seed)] = {"verdict": "incomplete", "numbers": {}}
            continue
        cells = [[] for _ in range(n_cells)]
        for r, c_ in zip(srows, scols):
            cells[int(c_)].append(int(r))
        # T-23 part 2: E = cols 0-799, I = cols 800-999 per build_fig3_network.
        res = score_network_seed(cells, n_steps, FIG3_DT, theta, alpha_band,
                                 populations={"E": np.arange(800),
                                              "I": np.arange(800, 1000)})
        seeds_out[str(seed)] = res
    overall = {}
    for nid in ("N1", "N2", "N3", "N4"):
        vs = [seeds_out[str(s)].get(nid, {}).get("verdict") for s in FIG3_SEEDS]
        if not complete or any(v == "incomplete" for v in vs):
            overall[nid] = "incomplete"
        else:
            overall[nid] = "pass" if all(v == "pass" for v in vs) else "fail"
    arm = ("pass" if all(v == "pass" for v in overall.values())
           else "incomplete" if any(v == "incomplete" for v in overall.values())
           else "fail")
    regime_labels = {}
    for seed in FIG3_SEEDS:
        regime = seeds_out[str(seed)].get("activity_regime")
        regime_labels[str(seed)] = (regime["labels"] if isinstance(regime, dict)
                                    else None)
    return {"seeds": seeds_out, "overall": overall, "arm": arm,
            "complete": complete, "activity_regime_labels": regime_labels}


def cmd_score(fig2_dir, fig3_dir, out_path):
    theta, alpha_band = load_calibration()
    fig2 = score_fig2_dir(fig2_dir, theta, alpha_band)
    fig3 = score_fig3_dir(fig3_dir, theta, alpha_band)
    # Completeness: every scored run uses one code SHA (PREREG_P3.md).
    shas = {json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8")).get("code_sha")
            for d in (fig2_dir, fig3_dir)}
    if len(shas) != 1 or None in shas:
        fig2["overall"] = fig3["arm"] = "incomplete"
    report = {"prereg": "PREREG_P3.md signed v7", "code_sha": _git_sha(),
              "run_code_shas": sorted(str(s) for s in shas),
              "theta": theta, "alpha_band": list(alpha_band),
              "gamma_band": list(GAMMA_BAND),
              "fig2": fig2, "fig3": fig3}
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"fig2={fig2['overall']} fig3={fig3['arm']} -> {out_path}")
    return report


def _seed_cells(fig_dir, seed):
    z = np.load(Path(fig_dir) / f"{seed}.npz")
    cells = [[] for _ in range(int(z["n_cells"]))]
    for r, c_ in zip(np.asarray(z["spike_rows"]).tolist(), np.asarray(z["spike_cols"]).tolist()):
        cells[int(c_)].append(int(r))
    return cells, int(z["n_steps"])


def cmd_s1(ref_dir, kernel_dir, out_path):
    """Sensitivity S1, reported only: N1-N4 on the reference arm and the kernel arm, per seed."""
    theta, alpha_band = load_calibration()
    ref_manifest = json.loads((Path(ref_dir) / "manifest.json").read_text(encoding="utf-8"))
    if ref_manifest.get("sensitivity") != "S1":
        raise SystemExit("cmd_s1 reads a manifest labelled sensitivity S1")
    seeds = {}
    for entry in ref_manifest["runs"]:
        seed = entry["seed"]
        arms = {}
        for arm, d in (("reference", ref_dir), ("kernel", kernel_dir)):
            cells, n_steps = _seed_cells(d, seed)
            # Each arm's own step: the reference rows are 1 ms, the kernel rows 0.5 ms.
            dt_ms = float(np.load(Path(d) / f"{seed}.npz")["dt_ms"])
            res = score_network_seed(cells, n_steps, dt_ms, theta, alpha_band)
            arms[arm] = {n: res[n] for n in ("N1", "N2", "N3", "N4")}
        seeds[str(seed)] = dict(arms, mean_extra_u_per_spike=entry["mean_extra_u_per_spike"])
    report = {"sensitivity": "S1", "reported_only": True, "code_sha": _git_sha(),
              "reference_code_sha": ref_manifest.get("code_sha"), "seeds": seeds}
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"S1 report -> {out_path}")
    return report


def _sens_fig3_arm(fig_dir, seed, theta, alpha_band):
    z = np.load(Path(fig_dir) / f"{seed}.npz")
    n_steps, n_cells = int(z["n_steps"]), int(z["n_cells"])
    dt_ms = float(z["dt_ms"])
    cells = [[] for _ in range(n_cells)]
    for r_, c_ in zip(np.asarray(z["spike_rows"]).tolist(), np.asarray(z["spike_cols"]).tolist()):
        cells[int(c_)].append(int(r_))
    res = score_network_seed(cells, n_steps, dt_ms, theta, alpha_band)
    manifest = json.loads((Path(fig_dir) / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest.get("runs", {}).get(str(seed), {})
    settings = {k: entry.get(k) for k in
                ("dt_ms", "dtype", "output_dtype", "W", "noise", "noise_ref_E",
                 "noise_ref_I", "n_spikes")}
    return {n: res[n] for n in ("N1", "N2", "N3", "N4")}, settings


def cmd_sens_report(sens_id, sens_dir, kernel_dir, out_path):
    """Sensitivity S2/S3/S4, reported only: criteria on the sens run and the scored run side by side."""
    theta, alpha_band = load_calibration()
    sens_manifest = json.loads((Path(sens_dir) / "manifest.json").read_text(encoding="utf-8"))
    if sens_manifest.get("sensitivity") != sens_id:
        raise SystemExit(f"cmd_sens_report reads a manifest labelled sensitivity {sens_id}")
    if sens_id in ("S2", "S3"):
        arms, arm_settings = {}, {}
        for arm, d in (("sensitivity", sens_dir), ("scored", kernel_dir)):
            res, settings = _sens_fig3_arm(d, SENS_SEED, theta, alpha_band)
            arms[arm], arm_settings[arm] = res, settings
        report = {"sensitivity": sens_id, "reported_only": True, "code_sha": _git_sha(),
                  "seed": SENS_SEED, "theta": theta, "alpha_band": list(alpha_band),
                  "gamma_band": list(GAMMA_BAND),
                  "arms": arms, "arm_settings": arm_settings}
    elif sens_id == "S4":
        event_row = expected_fig2_settings()["C7"]["event_row"]
        s_rows, _ = _load_run_npz(sens_dir, "C7")
        k_rows, _ = _load_run_npz(kernel_dir, "C7")
        report = {"sensitivity": "S4", "reported_only": True, "code_sha": _git_sha(),
                  "C7": {"sensitivity": eval_c7(s_rows, event_row, FIG2_DT),
                         "scored": eval_c7(k_rows, event_row, FIG2_DT)},
                  "C7_settings": {
                      "sensitivity": sens_manifest["runs"]["C7"]["segments"],
                      "event_row": event_row, "dt_ms": FIG2_DT}}
    else:
        raise SystemExit(f"unknown sensitivity id {sens_id!r}; want one of {SENS_IDS}")
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{sens_id} report -> {out_path}")
    return report


# --- amendment v8.2 (PREREG_P3_v8.md, signed v8.2) ---------------------------------------------------
#
# The v7 entries above (FIG2_RUNS, cmd_fig2, cmd_score) are unchanged; v8 adds
# new names only. v8 scores one new run (C8 at pulse amplitude 2.0) and a
# relative C4 rule against the v7 C1/C4/C5 runs.

V8_AMENDMENT = "v8.2"
V8_PREREG = "PREREG_P3_v8.md signed v8.2"
V8_C8_AMPLITUDE = 2.0
FIG2_V8_RUNS = {
    # run_id: (type, segments in (ms, current), event row, steps, kind)
    "C8": ("RZ", ((10.0, 0.0), (1.0, V8_C8_AMPLITUDE), (311.0, 0.0)), 110, 3220, "pulse"),
}


def expected_fig2_v8_settings():
    """Prereg settings of the v8 C8 run; the v7 C8 entry keeps amplitude 5.0."""
    t, segs, event, n, kind = FIG2_V8_RUNS["C8"]
    a, b, c, d = FIG2_TYPES[t]
    return {"C8": {"type": t, "a": a, "b": b, "c": c, "d": d,
                   "vstar": fixed_point_vstar(b),
                   "segments": [[ms, cur] for ms, cur in segs],
                   "event_row": event, "n_steps": n, "dt_ms": FIG2_DT,
                   "dtype": FIG2_DTYPE, "key_seed": 0,
                   "noise_scale": 0.0, "kind": kind}}


def _git_status_porcelain():
    """Porcelain status; a failed git call reads as a tracked change, so the run is refused."""
    try:
        res = subprocess.run(["git", "-C", str(_repo_root()), "status", "--porcelain"],
                             capture_output=True, text=True)
    except OSError:
        return " M <git unavailable>"
    return res.stdout if res.returncode == 0 else " M <git status failed>"


def _v8_tree_dirty(status=None):
    """True when `git status --porcelain` lists a tracked change (`??` rows are untracked)."""
    if status is None:
        status = _git_status_porcelain()
    for line in str(status).splitlines():
        if line.strip() and not line.startswith("??"):
            return True
    return False


def cmd_fig2_v8(out_dir):
    """v8 Fig. 2 C8 run at pulse amplitude 2.0; refuses on a dirty tree (exit 2)."""
    if _v8_tree_dirty():
        raise SystemExit(2)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = expected_fig2_v8_settings()["C8"]
    t, segs, event, n, _kind = FIG2_V8_RUNS["C8"]
    v, rows = run_single_cell(t, segs, n)
    np.savez_compressed(out / "C8.npz",
                        spike_rows=np.asarray(rows, dtype=np.int64), v=v)
    settings = dict(expected, file="C8.npz", n_spikes=len(rows))
    manifest = {"prereg": V8_PREREG + ", Fig. 2 C8",
                "amendment": V8_AMENDMENT,
                "code_sha": _git_sha(), "backend": _jax_backend(),
                "kernel": "jaxfne.emitters.simulate_eig_izhikevich",
                "dtype": FIG2_DTYPE, "dt_ms": FIG2_DT,
                "sensitivity": None, "runs": {"C8": settings}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote v8 Fig. 2 C8 run to {out}")
    return manifest


def eval_c4_v8(c4_rows, c1_rows, c5_rows, dt_ms=FIG2_DT,
               c4_event=100, c1_event=100, c5_event=100):
    """v8 C4: rate >= 50 Hz and the C4 ISI ratio strictly below the C1 and C5 ratios.

    Ratios use v7's formula (last/first). Any None ratio fails; a missing run is
    reported as incomplete by cmd_v8, not here.
    """
    c4_win = izh.window_rows([int(r) for r in c4_rows], c4_event, dt_ms)
    c1_win = izh.window_rows([int(r) for r in c1_rows], c1_event, dt_ms)
    c5_win = izh.window_rows([int(r) for r in c5_rows], c5_event, dt_ms)
    r4, r1, r5 = (izh.isi_ratio(c4_win), izh.isi_ratio(c1_win), izh.isi_ratio(c5_win))
    rate = izh.rate_hz(c4_win)
    numbers = {"n_spikes": len(c4_win), "rate_hz": rate,
               "isi_ratio": r4, "c1_isi_ratio": r1, "c5_isi_ratio": r5}
    ok = (rate >= 50.0 and r4 is not None and r1 is not None and r5 is not None
          and r4 < r1 and r4 < r5)
    return {"verdict": "pass" if ok else "fail", "numbers": numbers}


def _dir_shas(fig_dir):
    """code_sha values of every manifest.json under a run dir."""
    shas = set()
    for m in sorted(Path(fig_dir).rglob("manifest.json")):
        try:
            shas.add(json.loads(m.read_text(encoding="utf-8")).get("code_sha"))
        except (OSError, ValueError):
            shas.add(None)
    return shas


def cmd_v8(v7_score_path, v7_fig2_dir, v8_fig2_dir, out_path):
    """v8 scorer: C1-C3/C5-C7 and Fig. 3 copied from the v7 score, C8 on the v8 run, C4 relative."""
    v7_report = json.loads(Path(v7_score_path).read_text(encoding="utf-8"))
    v7_shas = _dir_shas(v7_fig2_dir) | set(v7_report.get("run_code_shas", []))
    v8_shas = _dir_shas(v8_fig2_dir)
    if len(v7_shas) != 1 or None in v7_shas or len(v8_shas) != 1 or None in v8_shas:
        raise SystemExit(f"v8 needs one SHA per side: "
                         f"v7={sorted(map(str, v7_shas))} v8={sorted(map(str, v8_shas))}")
    v7_sha = next(iter(v7_shas))
    v8_sha = next(iter(v8_shas))
    v7_fig2 = v7_report.get("fig2", {}).get("criteria", {})
    criteria = {}
    for rid in ("C1", "C2", "C3", "C5", "C6", "C7"):
        entry = v7_fig2.get(rid)
        if entry is None:
            criteria[rid] = {"verdict": "incomplete", "numbers": {}}
        else:
            criteria[rid] = json.loads(json.dumps(entry))
    expected = expected_fig2_settings()
    expected_v8 = expected_fig2_v8_settings()

    def _v7_run(run_id):
        try:
            man = json.loads((Path(v7_fig2_dir) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        exp = expected.get(run_id)
        got = man.get("runs", {}).get(run_id)
        if exp is None or got is None:
            return None
        if {k: v for k, v in got.items() if k in exp} != exp:
            return None
        try:
            rows, _ = _load_run_npz(v7_fig2_dir, run_id)
        except OSError:
            return None
        return rows

    c1_rows = _v7_run("C1")
    c4_rows = _v7_run("C4")
    c5_rows = _v7_run("C5")
    if c1_rows is None or c4_rows is None or c5_rows is None:
        criteria["C4"] = {"verdict": "incomplete", "numbers": {}}
    else:
        criteria["C4"] = eval_c4_v8(c4_rows, c1_rows, c5_rows, FIG2_DT,
                                   expected["C4"]["event_row"],
                                   expected["C1"]["event_row"],
                                   expected["C5"]["event_row"])
    exp8 = expected_v8["C8"]
    try:
        man8 = json.loads((Path(v8_fig2_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        man8 = {}
    got8 = man8.get("runs", {}).get("C8")
    if got8 is None or {k: v for k, v in got8.items() if k in exp8} != exp8:
        criteria["C8"] = {"verdict": "incomplete", "numbers": {}}
    else:
        try:
            rows8, v8 = _load_run_npz(v8_fig2_dir, "C8")
        except OSError:
            rows8, v8 = None, None
        if rows8 is None or v8 is None:
            criteria["C8"] = {"verdict": "incomplete", "numbers": {}}
        else:
            criteria["C8"] = eval_c8(v8, exp8["event_row"], FIG2_DT, rows8)
    verdicts = {k: criteria[k]["verdict"] for k in
                ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}
    overall = ("pass" if all(v == "pass" for v in verdicts.values())
               else "incomplete" if any(v == "incomplete" for v in verdicts.values())
               else "fail")
    report = {"prereg": V8_PREREG, "amendment": V8_AMENDMENT,
              "code_sha": _git_sha(), "v7_sha": v7_sha, "v8_sha": v8_sha,
              "v7_score": str(v7_score_path),
              "fig2": {"criteria": criteria, "verdicts": verdicts, "overall": overall},
              "fig3": json.loads(json.dumps(v7_report.get("fig3", {})))}
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"fig2={overall} fig3=copied (v7) -> {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("fig2")
    p2.add_argument("--out", required=True)
    p2.add_argument("--signed", action="store_true")
    p3 = sub.add_parser("fig3")
    p3.add_argument("--seeds", type=int, nargs="+", required=True)
    p3.add_argument("--T", type=float, required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--signed", action="store_true")
    ps = sub.add_parser("score")
    ps.add_argument("--fig2", required=True)
    ps.add_argument("--fig3", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--signed", action="store_true")
    p1 = sub.add_parser("s1")
    p1.add_argument("--reference", required=True)
    p1.add_argument("--kernel", required=True)
    p1.add_argument("--out", required=True)
    psens = sub.add_parser("sens")
    psens.add_argument("--id", required=True, choices=list(SENS_IDS))
    psens.add_argument("--out", required=True)
    psens.add_argument("--signed", action="store_true")
    prep = sub.add_parser("sens-report")
    prep.add_argument("--id", required=True, choices=list(SENS_IDS))
    prep.add_argument("--dir", required=True)
    prep.add_argument("--kernel", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--signed", action="store_true")
    p8 = sub.add_parser("fig2-v8")
    p8.add_argument("--out", required=True)
    p8.add_argument("--signed", action="store_true")
    pv = sub.add_parser("v8")
    pv.add_argument("--v7-score", required=True)
    pv.add_argument("--v7-fig2", required=True)
    pv.add_argument("--v8-fig2", required=True)
    pv.add_argument("--out", required=True)
    pv.add_argument("--signed", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fig2-v8":
        check_out(a.out, a.signed)
        cmd_fig2_v8(a.out)
        return
    if a.cmd == "v8":
        check_out(a.out, a.signed)
        cmd_v8(a.v7_score, a.v7_fig2, a.v8_fig2, a.out)
        return
    if a.cmd == "s1":
        cmd_s1(a.reference, a.kernel, a.out)
        return
    if a.cmd == "sens":
        check_out(a.out, a.signed)
        cmd_sens(a.id, a.out)
        return
    if a.cmd == "sens-report":
        check_out(a.out, a.signed)
        cmd_sens_report(a.id, a.dir, a.kernel, a.out)
        return
    if a.cmd == "fig2":
        check_out(a.out, a.signed)
        cmd_fig2(a.out)
    elif a.cmd == "fig3":
        check_out(a.out, a.signed)
        cmd_fig3(a.out, a.seeds, a.T)
    else:
        check_out(a.out, a.signed)
        cmd_score(a.fig2, a.fig3, a.out)


if __name__ == "__main__":
    main()
