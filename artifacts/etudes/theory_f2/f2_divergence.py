"""F2: replication under divergence in the PCL column (declaration: ``theory_f2/README.md``).

Phase 1 only, reusing ``pcl_column_hdp`` (runner, weight init, calibration) by import.
The arms run in lockstep, one sequence at a time. ``draw`` copies the per-sequence keys,
orientations and directions that ``pcl_column_hdp.block`` draws; ``pcl_column_hdp`` itself
is not edited. ``analyse`` and ``verdicts`` use only the per-sequence series, so the
verdicts can be recomputed from the JSON files alone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import artifacts.etudes.pcl.pcl_column_hdp as P  # noqa: E402

import jaxfne  # noqa: E402,F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

PLASTIC = (0, 1, 4, 5)  # excitatory plastic types, as in pcl_column_hdp phase 1
PLASTIC_EDGE = np.isin(P.TYP, PLASTIC)
WIN_LO = 1e-6  # README: window lower bound on D_w
N_SAT = 50  # README: D_sat averages the last 50 sequences
PAIRS = ("base_nudge", "base_ref", "frozen")
N_RUNS = 5  # runner calls per sequence: base, nudge, ref, frozen w0, frozen nudge(w0)


def draw(key, n_seq):
    """Keys, orientations and directions exactly as ``pcl_column_hdp.block`` draws them (labels=None)."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    oris = rng.integers(0, P.C.N_ORI, n_seq)
    dirs = rng.choice([-1.0, 1.0], len(oris))
    return jax.random.split(key, len(oris)), oris, dirs


def setup(seed):
    """main()'s setup up to phase 1: same calls, same rng and key splits."""
    P.K.DT = P.DT
    amp, w_cancel = P.K.calibrate()
    s = w_cancel / 5.3
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    k_p1 = jax.random.split(key, 4)[0]
    w0 = P.nudge_ulp(P.init_weights(rng, s, PLASTIC), 0)
    return amp, s, w_cancel, k_p1, w0


def d_w(a, b):
    """README D_w: relative L1 distance of the plastic-edge weights."""
    a = np.asarray(a, np.float64)[PLASTIC_EDGE]
    b = np.asarray(b, np.float64)[PLASTIC_EDGE]
    return float(np.abs(a - b).sum() / np.abs(a).sum())


def d_s(ca, cb):
    """README D_s: relative L1 distance of per-cell spike counts, simple and complex cells (S0 onward)."""
    ca = np.asarray(ca, np.float64)[P.S0:]
    cb = np.asarray(cb, np.float64)[P.S0:]
    den = (ca + cb).sum()
    return float(np.abs(ca - cb).sum() / den) if den > 0 else float("nan")


def run_lockstep(seed, n_seq):
    """Base, nudge, ref and the frozen pair, stepped sequence by sequence; returns series and base's final weights."""
    t0 = time.time()
    amp, s, w_cancel, k_p1, w0 = setup(seed)
    w0_nudge = P.nudge_ulp(w0, 1)
    plastic = P.make_runner(s, amp, PLASTIC)
    frozen = P.make_runner(s, amp, ())
    keys, oris, dirs = draw(k_p1, n_seq)
    rkeys, roris, rdirs = draw(jax.random.fold_in(k_p1, 1000), n_seq)

    def f32(w):
        return jnp.asarray(w, jnp.float32)

    wb, wn, wr = f32(w0), f32(w0_nudge), f32(w0)  # plastic arms, carried through training
    fa, fb = f32(w0), f32(w0_nudge)  # frozen arms, fixed weights
    pairs = {name: {"D_w": [], "D_s": []} for name in PAIRS}
    t_seq = []
    loop0 = time.time()
    for n in range(n_seq):
        ts = time.time()
        k, o, d = keys[n], int(oris[n]), float(dirs[n])
        wb, cnt_b = plastic(wb, k, o, d)
        wn, cnt_n = plastic(wn, k, o, d)
        wr, cnt_r = plastic(wr, rkeys[n], int(roris[n]), float(rdirs[n]))
        _, cnt_fa = frozen(fa, k, o, d)
        _, cnt_fb = frozen(fb, k, o, d)
        for name, (xa, xb, ca, cb) in {
            "base_nudge": (wb, wn, cnt_b, cnt_n),
            "base_ref": (wb, wr, cnt_b, cnt_r),
            "frozen": (fa, fb, cnt_fa, cnt_fb),
        }.items():
            pairs[name]["D_w"].append(d_w(xa, xb))
            pairs[name]["D_s"].append(d_s(ca, cb))
        t_seq.append(time.time() - ts)
    loop_s = time.time() - loop0
    steady = t_seq[1:] if len(t_seq) > 1 else t_seq
    timing = dict(wall_s=round(time.time() - t0, 1), loop_s=round(loop_s, 1), first_seq_s=round(t_seq[0], 2),
                  s_per_arm_seq_steady=round(float(np.mean(steady)) / N_RUNS, 3),
                  s_per_arm_seq_mean=round(loop_s / (N_RUNS * n_seq), 3))
    return dict(seed=seed, n_seq=n_seq, amp=amp, w_cancel=w_cancel, scale=s, pairs=pairs, timing=timing,
                w_base_final=np.asarray(wb))


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def _jsonable(x):
    return [float(v) if math.isfinite(v) else None for v in x]


def _t975(df):
    try:
        from scipy import stats
    except ImportError:
        return 1.96, "least squares; normal 1.96 (scipy not importable)"
    return float(stats.t.ppf(0.975, df)), "least squares; scipy.stats.t.ppf(0.975, n-2)"


def _slope(x, y):
    """Least-squares slope and its 95% interval (t with n-2 df)."""
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = (y - ym) - b * (x - xm)
    se = math.sqrt((resid ** 2).sum() / (len(x) - 2) / sxx)
    q, method = _t975(len(x) - 2)
    return float(b), [float(b - q * se), float(b + q * se)], method


def growth(dw, d_sat):
    """H2.1 fit: slope of ln D_w on n over the window WIN_LO <= D_w <= 0.1 D_sat."""
    n = np.arange(1, len(dw) + 1)
    win = (dw >= WIN_LO) & (dw <= 0.1 * d_sat)
    idx = n[win]
    out = dict(window=[int(idx[0]), int(idx[-1])] if idx.size else None, n_window=int(idx.size),
               window_contiguous=bool(idx.size and idx[-1] - idx[0] + 1 == idx.size),
               log="natural", lambda_per_seq=None, ci95=None, ci_method=None)
    if idx.size >= 3:
        lam, ci, method = _slope(idx.astype(float), np.log(dw[win]))
        out.update(lambda_per_seq=lam, ci95=ci, ci_method=method)
    return out


def horizon(dw, d_sat):
    """H2.2: first n (1-based) with D_w(base, nudge) >= 0.5 D_sat, or None."""
    hit = np.nonzero(dw >= 0.5 * d_sat)[0]
    return int(hit[0]) + 1 if hit.size else None


def frozen_slope(ds):
    """H2.3: least-squares slope of D_s on n in the frozen arm, over all sequences."""
    n = np.arange(1, len(ds) + 1, dtype=float)
    ok = np.isfinite(ds)
    lam, ci, method = _slope(n[ok], ds[ok])
    return dict(slope_per_seq=lam, ci95=ci, ci_method=method, n_nan=int((~ok).sum()))


def _yn(flag):
    return "supported" if flag else "rejected"


def analyse(pairs):
    """Analysis of one seed from its per-sequence series (the only input)."""
    dw_nudge = _arr(pairs["base_nudge"]["D_w"])
    dw_ref = _arr(pairs["base_ref"]["D_w"])
    k = min(N_SAT, len(dw_ref))
    d_sat = float(dw_ref[-k:].mean())
    g = growth(dw_nudge, d_sat)
    n_star = horizon(dw_nudge, d_sat)
    fz = frozen_slope(_arr(pairs["frozen"]["D_s"]))
    h21 = g["n_window"] >= 10 and g["ci95"] is not None and g["ci95"][0] > 0
    h22 = n_star is not None
    h23 = fz["ci95"][0] <= 0
    return dict(n_seq=int(len(dw_nudge)), d_sat=d_sat, d_sat_last_k=k, growth=g, n_star=n_star,
                horizon_threshold=0.5 * d_sat, frozen_slope=fz,
                verdicts={"H2.1": _yn(h21), "H2.2": _yn(h22), "H2.3": _yn(h23)})


def verdicts(paths):
    """H2.1-H2.3 per README, computed from the JSON files only.

    Per seed as analysed. Aggregate: H2.2 is supported only if it holds in every seed (README rule);
    H2.1 and H2.3 give their common verdict, or "seeds disagree" (README states no seed rule for them).
    """
    per_seed = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        per_seed[doc["seed"]] = analyse(doc["pairs"])
    agg = {}
    for h in ("H2.1", "H2.2", "H2.3"):
        vals = {v["verdicts"][h] for v in per_seed.values()}
        if h == "H2.2":
            agg[h] = _yn(vals == {"supported"})
        else:
            agg[h] = vals.pop() if len(vals) == 1 else "seeds disagree"
    return dict(per_seed=per_seed, aggregate=agg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-seq", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = run_lockstep(a.seed, a.n_seq)
    pairs = {name: {"D_w": _jsonable(v["D_w"]), "D_s": _jsonable(v["D_s"])} for name, v in res["pairs"].items()}
    doc = dict(seed=a.seed, n_seq=a.n_seq, rule=P.RULE, dt_ms=P.DT, pulse_amp=res["amp"],
               w_cancel=res["w_cancel"], scale=res["scale"], n_edges=int(len(P.PRE)),
               n_plastic_edges=int(PLASTIC_EDGE.sum()), index="entry i is measured after sequence i+1",
               pairs=pairs, analysis=analyse(pairs), timing=res["timing"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))
    an, g = doc["analysis"], doc["analysis"]["growth"]
    print(f"seed {a.seed} n_seq {a.n_seq} D_sat {an['d_sat']:.4g} lambda {g['lambda_per_seq']} "
          f"ci95 {g['ci95']} window {g['window']} n_window {g['n_window']} n_star {an['n_star']}", flush=True)
    print(f"timing {res['timing']}", flush=True)
    print(f"verdicts {an['verdicts']}", flush=True)
    return res


if __name__ == "__main__":
    main()
