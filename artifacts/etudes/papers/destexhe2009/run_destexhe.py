"""Destexhe 2009 replication runner (PORT_SPEC.md, PREREG_P3.md).

Subcommands: ``fig1`` (single-cell patterns), ``fig6b`` (S2 two-phase run),
``score`` (CV_ISI, CC, rates off a written raster). Analysis only reads
Signals-equivalent rasters the core kernel produced; nothing is simulated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from artifacts.etudes.papers.destexhe2009 import destexhe_api as D  # noqa: E402


def cmd_fig1(args):
    import numpy as np

    out = {"dt_ms": D.DT_MS, "cells": {}}
    for dtype in ("RS", "RSweak", "FS", "LTS", "TC", "RE"):
        for protocol in ("depol", "hyper"):
            spikes = D.single_cell(dtype, protocol)
            t = np.flatnonzero(spikes).astype(float) * D.DT_MS
            isis = np.diff(t).tolist()
            out["cells"][f"{dtype}/{protocol}"] = {
                "n_spikes": int(t.size),
                "isis_ms": [float(x) for x in isis],
            }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print("wrote %s" % dest)


def cmd_fig6b(args):
    result = D.run(
        args.seed,
        args.tag,
        args.out,
        free_t_s=args.free_t_s,
        n_rs=args.n_rs,
        n_fs=args.n_fs,
        signed=args.signed,
    )
    print(
        "seed %d tag %s steps %d wall_s %.1f"
        % (args.seed, args.tag, result["steps"], result["wall_s"])
    )


def cmd_fig6b_cont(args):
    result = D.run_continuous(
        args.seed,
        args.tag,
        args.out,
        free_t_s=args.free_t_s,
        n_rs=args.n_rs,
        n_fs=args.n_fs,
        signed=args.signed,
        kick_fraction=args.kick_fraction,
        kick_rate_hz=args.kick_rate_hz,
        v0_seed=args.v0_seed,
        ongoing=args.ongoing,
        probability=args.probability,
    )
    print(
        "seed %d tag %s steps %d wall_s %.1f"
        % (args.seed, args.tag, result["steps"], result["wall_s"])
    )


def cmd_score(args):
    import numpy as np

    with open(Path(args.out) / ("spikes_seed%d_%s.npz" % (args.seed, args.tag)), "rb") as f:
        pack = np.load(f)
        spikes = np.asarray(pack["spikes"], dtype=bool)
        dt_ms = float(pack["dt_ms"])
    cv = D.cv_isi(spikes, dt_ms)
    cc = D.pairwise_cc(spikes, dt_ms, seed=args.seed)
    rate = D.mean_rate_hz(spikes, dt_ms)
    alive = bool(spikes[-int(1000.0 / dt_ms) :, :].sum() > 0)
    # T-23 part 2 (Hamm 2026-10-14): report the activity regime of the scored
    # raster. The raster stores no RS/FS split, so the regime is reported for
    # the whole network only; no subpopulation is inferred. Scored numbers
    # above are untouched.
    from jaxfne.analysis import ACTIVITY_REGIME_DEFAULTS
    from jaxfne.analysis import activity_regime as _activity_regime

    network_regime = _activity_regime(np.asarray(spikes), float(dt_ms))
    activity_regime_report = {
        "populations": {"all": network_regime},
        "labels": {"all": network_regime["label"]},
        "window_steps": [0, int(spikes.shape[0])],
        "window_ms": [0.0, float(spikes.shape[0]) * float(dt_ms)],
        "bin_ms": ACTIVITY_REGIME_DEFAULTS["bin_ms"],
        "dt_ms": float(dt_ms),
        "note": "whole-network population: the raster stores no RS/FS split",
    }
    report = {
        "seed": args.seed,
        "tag": args.tag,
        "cv_isi": cv,
        "cc": cc,
        "mean_rate_hz": rate,
        "alive_last_s": alive,
        "n": int(spikes.shape[1]),
        "steps": int(spikes.shape[0]),
        "activity_regime": activity_regime_report,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))


def build_parser():
    ap = argparse.ArgumentParser(description="Destexhe 2009 replication runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("fig1", help="single-cell patterns to JSON")
    p1.add_argument("--out", required=True)
    p1.set_defaults(fn=cmd_fig1)
    p6 = sub.add_parser("fig6b", help="S2 two-phase cortical run")
    p6.add_argument("--seed", type=int, required=True)
    p6.add_argument("--tag", default="fig6b")
    p6.add_argument("--out", required=True)
    p6.add_argument("--free-t-s", type=float, default=D.FREE_T_S)
    p6.add_argument("--n-rs", type=int, default=D.N_RS)
    p6.add_argument("--n-fs", type=int, default=D.N_FS)
    p6.add_argument("--signed", action="store_true")
    p6.set_defaults(fn=cmd_fig6b)
    p6c = sub.add_parser("fig6b_cont", help="S2 continuous run (gated kick, no restart)")
    p6c.add_argument("--seed", type=int, required=True)
    p6c.add_argument("--tag", default="fig6b_cont")
    p6c.add_argument("--out", required=True)
    p6c.add_argument("--free-t-s", type=float, default=D.FREE_T_S)
    p6c.add_argument("--n-rs", type=int, default=D.N_RS)
    p6c.add_argument("--n-fs", type=int, default=D.N_FS)
    p6c.add_argument("--signed", action="store_true")
    p6c.add_argument("--kick-fraction", type=float, default=1.0)
    p6c.add_argument("--kick-rate-hz", type=float, default=D.KICK_RATE_HZ)
    p6c.add_argument("--v0-seed", type=int, default=None)
    p6c.add_argument("--ongoing", action="store_true")
    p6c.add_argument("--probability", type=float, default=D.P)
    p6c.set_defaults(fn=cmd_fig6b_cont)
    ps = sub.add_parser("score", help="score a written raster")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--tag", default="fig6b")
    ps.add_argument("--out", required=True)
    ps.add_argument("--report", default=None)
    ps.set_defaults(fn=cmd_score)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
