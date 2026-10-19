#!/usr/bin/env python3
"""Scaffold a new paper replication lane under ``artifacts/etudes/papers/<lane>/``.

Dry run by default (prints the files it would create); ``--write`` creates
them. Refuses an existing lane folder. At most 7 files are created, so a new
lane passes the R6 tree-degree gate. Only the standard library is used.

Usage:
    python scripts/new_lane.py <lane> --paper "<citation>" [--write]
    python scripts/new_lane.py <lane> --paper "<citation>" --write --root DIR  # tests
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS_REL = Path("artifacts") / "etudes" / "papers"

LANE_RE = re.compile(r"[A-Za-z0-9_]+")


def lane_files(lane: str, paper: str) -> dict[str, str]:
    """The scaffold contents (at most 7 files), keyed by file name."""
    api = f"{lane}_api.py"
    test = f"test_{lane}_api.py"
    return {
        "README.md": f"""# {lane} replication

Paper: {paper} (checklist S3: the reference lives in `docs/reference/references.md`).

Tier: paper lane; stage P1 (source record only, no scored run).
Figures targeted: (fill: figure ids and the estimand per figure).

Checklist rows this record serves: R1 (the lane folder exists with its P1
record), R3 (dashboard row points here).
""",
        "SOURCE_P1.md": f"""# {lane}: P1 source record

Target: {paper}. Read the paper (file:page pointers below, checklist S4:
every item observed or inferred; unknowns stay unknown, never filled).

## 1. Model equations (paper section, file:page)

(Quote no passage over 15 words; equations and parameter symbols here.)

## 2. Parameters (paper tables, file:page)

| Parameter | Paper value | Port value | Status |
|---|---|---|---|
| (fill) | | | observed / inferred / unknown |

## 3. Claims per figure the lane targets (paper Results, file:page)

- (fill: one row per estimand y with its uncertainty where stated)

## 4. Data, code and licence (checklist S4)

- Paper licence: (fill)
- Authors' code location and licence: (fill; read as text only, never import:
  checklist S1)
- Data location and licence: (fill)

## 5. Gaps (checklist S4: the gap list has no unknowns at P1 close)

- G1: (fill: each item observed or inferred)
""",
        "PREREG_P3.md": f"""# {lane} preregistration P3 (draft, UNSIGNED — no scored run may use it)

Lane: {lane}. Paper: {paper}. No scored run starts from this draft;
Hamm signs the frozen protocol before scoring (checklist P1).

## Estimands and criteria (checklist P1)

| Id | y (paper) | z (jaxfne) | e | criterion |
|---|---|---|---|---|
| (fill) | | | | close: every e <= 0.05; off: any e > 0.05 |

## Seeds (checklist P2)

Seed 0 is for pilots only. Fit (if needed): seeds 1-3. Score: held-out
seeds 4-8. The seed shown to change the output is named.

## Scored / unscored split

- Unscored (pilots, G0 checks): seed 0, timing runs; outputs never under a
  `scored/` path.
- Scored (needs `--signed`): seeds 1-8 per above; one code SHA per output
  directory (enforced by `lane_kit.refuse_existing`).

## Sensitivity set (declared before scoring)

- (fill: S1..Sn, one declared variation each; sensitivity runs never enter
  score, checklist P6)

## Pilot (B8)

- The full pipeline runs on unscored pilot seeds and on a known positive
  before this prereg is signed (project rule B8, Hamm 2026-10-18).
- The sensitivity set is declared in this prereg (see above); after
  signature at most one amendment, any further change is a new protocol
  version scored on fresh seeds.
""",
        "CALIBRATION.md": f"""# {lane} calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P1 (no runs yet). Every number below is reopened from its JSON or
receipt before commit (checklist R2); verdicts use the status tags
(checklist R3).

## Provenance

- Code SHA: (fill; one SHA per output directory, enforced by
  `{lane}_api.write_run` via `lane_kit`).
- Backend, dtype, seeds: (fill per run; float32 default, one float64 check per
  scored result, checklist P3).

## 1. Estimand table

e = |z - y| / |y| (absolute error for estimands near 0, with the scale named).

| Id | y (paper) | z at theta0 | z at theta* | e | Band |
|---|---|---|---|---|---|
| (fill) | | | | | |

## 2. Simplifications, fidelity fixes, X forms (checklist F1-F4)

- (fill: each step with its file and seed; a fidelity fix carries a test
  that fails on the code before the fix)

## 3. D (degrees of freedom, declared before any fit)

- (fill: each parameter with its range and reason; never a parameter the
  paper states exactly, unless declared as a deviation with a reason)

## 4. Held-out result and claim (checklist P6)

- Fit on seeds 1-3, score on held-out seeds 4-8. Claim: "reproduced after
  calibration of [theta in D] (post hoc), held-out seeds 4-8" or "needs X:
  [form], |a| = ...".
""",
        api: f'''"""{lane} replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``Configuration -> construct -> Model -> simulate -> Signals`` (checklist
E1). Estimators come from ``jaxfne.analysis`` (checklist E3). No scored run:
outputs land in the caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype and seeds (checklist P4).
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jaxfne as J
from scripts import lane_kit


def build_spec(n=20, seed=0):
    """The lane network built only from the public API (fill: port rows)."""
    cfg = (
        J.Configuration()
        .network(name="{lane}", n=int(n), cell_types={{"E": 0.8, "PV": 0.2}})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
    )
    return cfg.connectivity(edge_seed=int(seed))


def build_sim(T_s=1.0, dt_ms=0.1):
    """The ``Simulation`` for a run (fill: duration, dt, drive)."""
    return J.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=float(dt_ms))


def run(seed, tag, out_dir, *, T_s=1.0, signed=False):
    """Build, run and write the raster plus the ``manifest.json`` the scorer reads."""
    import numpy as np

    lane_kit.refuse_scored_dir(out_dir, signed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = "spikes_seed%d_%s.npz" % (int(seed), tag)
    sha = lane_kit.git_sha(_repo_root)
    prior = lane_kit.refuse_existing(out, fname, sha, signed=signed)
    entries = {{e["file"]: e for e in prior["runs"]}} if prior else {{}}
    model = J.construct(build_spec(seed=seed))
    sim = build_sim(T_s=T_s)
    spikes = np.zeros((sim.n_steps, model.n), dtype=bool)  # fill: simulate
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {{"seed": int(seed), "tag": tag, "file": fname,
              "steps": int(spikes.shape[0]), "n": int(spikes.shape[1])}}
    entries[fname] = entry
    manifest = {{"git_sha": sha, "runs": list(entries.values())}}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {{"spikes": spikes, "steps": sim.n_steps}}


def score(spikes):
    """Score a raster with ``jaxfne.analysis`` estimators (fill: estimands)."""
    from jaxfne.analysis import activity_regime as _activity_regime

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    return _activity_regime(arr, 0.1)
''',
        test: f'''"""Smoke test for the {lane} lane API (one test: build the spec)."""

from artifacts.etudes.papers.{lane} import {lane}_api as L


def test_build_spec_smoke():
    """The scaffold spec compiles edges without simulating (fixture must build)."""
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"
''',
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scaffold a new paper replication lane.")
    ap.add_argument("lane", help="lane folder name under artifacts/etudes/papers/")
    ap.add_argument("--paper", required=True, help="paper citation for the scaffold")
    ap.add_argument("--write", action="store_true", help="create the lane folder")
    ap.add_argument("--root", type=Path, default=None, help="repo root (tests only)")
    args = ap.parse_args(argv)
    if not LANE_RE.fullmatch(args.lane):
        print(f"error: refusing lane name {args.lane!r} (want [A-Za-z0-9_]+)")
        return 1
    root = args.root if args.root is not None else ROOT
    folder = root / PAPERS_REL / args.lane
    files = lane_files(args.lane, args.paper)
    if len(files) > 7:
        print(f"error: scaffold holds {len(files)} files (at most 7)")
        return 1
    if folder.exists():
        print(f"error: refusing existing lane folder {folder}")
        return 1
    for name in sorted(files):
        print(f"{'write' if args.write else 'create'} {PAPERS_REL.as_posix()}/{args.lane}/{name}")
    if not args.write:
        return 0
    for name, text in files.items():
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
