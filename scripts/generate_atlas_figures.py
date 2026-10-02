#!/usr/bin/env python3
"""Generate the Atlas manuscript figures from per-AT in-memory bundles (0.5.5 ENGINE item 2).

For each canonical id in ``artifacts/atlas/at_manifest.py:REGISTRY``: run the
runner once with ``keep_bundle=True`` (``artifacts/atlas/at_bundle.py``), then
draw every bundle arm with the view-only ``jaxfne.vis.render_atlas``, stamped
with the manifest's run seed and duration. Output:
``artifacts/publication/atlas/<AT-id>/<arm>/`` (7 panels + index + manifest)
plus ``<AT-id>/atlas_run.json`` (spec digest, environment, per-arm hdp source
and panel digest).

Write-once (human decision 2026-09-23): an existing ``<AT-id>/`` directory is
refused, never overwritten.

HDP routing: an arm's own ``hdp`` diagnostics are passed explicitly. A model's
``last_*_diagnostics`` describe only its latest run, so they are used only when
the arm owns its model; a fixed-W arm or an arm sharing its model with another
arm gets ``hdp={}``, which omits the H/HDP panels instead of drawing another
arm's state.

Usage:
    python scripts/generate_atlas_figures.py --list
    python scripts/generate_atlas_figures.py --at AT-03,AT-10
    python scripts/generate_atlas_figures.py            # every REGISTRY id
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "artifacts" / "publication" / "atlas"


def hdp_for_arm(arm: dict[str, Any], shared: bool) -> tuple[dict[str, Any] | None, str]:
    """Return (hdp argument for render_atlas, provenance label) for one bundle arm."""
    own = arm.get("hdp")
    if own is not None:
        return own, "bundle"
    if "hdp" in arm:
        return {}, "none: fixed-W arm"
    if shared:
        return {}, "none: model shared across arms, no per-arm diagnostics"
    return None, "model: sole owner of its latest run"


def generate(at_id: str, out_root: Path = OUT_ROOT, bundle_fn: Any = None,
             manifest_fn: Any = None) -> dict[str, Any]:
    """Render every arm of one AT id; return the run record written to atlas_run.json."""
    from artifacts.atlas.at_bundle import bundle as _bundle
    from artifacts.atlas.at_manifest import manifest as _manifest
    from jaxfne.vis import render_atlas

    at_dir = out_root / at_id
    if at_dir.exists():
        raise FileExistsError(f"{at_dir} exists; Atlas figures are write-once")
    man = (manifest_fn or _manifest)(at_id)
    run = man["spec"]["run"]
    seed = man["spec"]["seeds"]["run"]
    arms = (bundle_fn or _bundle)(at_id)
    owners = Counter(id(a["model"]) for a in arms.values())

    record: dict[str, Any] = {
        "at_id": at_id,
        "spec_digest": man["spec_digest"],
        "lineage": man["lineage"],
        "environment": man["environment"],
        "seed": seed,
        "duration_ms": run["duration_ms"],
        "arms": {},
    }
    for name, arm in arms.items():
        hdp, source = hdp_for_arm(arm, owners[id(arm["model"])] > 1)
        panels = render_atlas(
            arm["model"], arm["signals"], out_dir=str(at_dir / name),
            title=f"{at_id} {name}", seed=seed, duration_ms=run["duration_ms"],
            dt_ms=run["dt_ms"], hdp=hdp,
            provenance={"at_id": at_id, "arm": name, "spec_digest": man["spec_digest"]})
        record["arms"][name] = {
            "hdp_source": source,
            "panels_sha256": panels["sha256"],
            "status": {p["file"]: p["status"] for p in panels["panels"]},
        }
    with open(at_dir / "atlas_run.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    return record


def main(argv: list[str] | None = None) -> int:
    from artifacts.atlas.at_manifest import REGISTRY

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list AT ids and exit")
    ap.add_argument("--at", default="", help="comma-separated AT ids (default: all)")
    args = ap.parse_args(argv)
    if args.list:
        for k, v in REGISTRY.items():
            print(f"{k}\t{v['stage']}\t{v['status']}\t{v['runner']}")
        return 0
    ids = [s.strip() for s in args.at.split(",") if s.strip()] or list(REGISTRY)
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        print(f"unknown AT ids: {unknown}", file=sys.stderr)
        return 2
    for at_id in ids:
        rec = generate(at_id)
        for name, a in rec["arms"].items():
            bad = {f: s for f, s in a["status"].items() if s != "AVAILABLE"}
            print(f"[{at_id}/{name}] sha={a['panels_sha256']} hdp={a['hdp_source']} "
                  f"not_available={bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
