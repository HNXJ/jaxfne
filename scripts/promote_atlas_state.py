#!/usr/bin/env python3
"""Promote an atlas figure state: GENERATED -> VALIDATED -> CANONICAL (Batch B4).

Promotion is a gated batch operation, never a hand-edit. Requirements:
every panel AVAILABLE, manifest sha matches recomputed panels hash, and an
explicit reason. The promoter, git HEAD, and UTC time are recorded.

Usage:
    python scripts/promote_atlas_state.py --dir docs/_static/atlas/foo \
        --to validated --reason "Batch C6: gates green on <sha>"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

VALID = ("validated", "canonical")


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True).strip()
    except Exception:
        return "unknown"


def promote(d: Path, to: str, reason: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = {"generated": 0, "validated": 1, "canonical": 2}
    cur = str(manifest.get("figure_state", "generated"))
    if to not in VALID:
        raise SystemExit(f"target must be one of {VALID}")
    if order.get(cur, 0) >= order[to]:
        raise SystemExit(f"cannot promote {cur} -> {to} (no downgrades, no repeats)")
    if to == "canonical" and cur != "validated":
        raise SystemExit("canonical requires validated first")
    panels = manifest.get("panels", [])
    if not panels or any(p.get("status") != "AVAILABLE" for p in panels):
        raise SystemExit("all panels must be AVAILABLE")
    recomputed = hashlib.sha256(
        json.dumps(panels, sort_keys=True, default=str).encode()).hexdigest()[:16]
    if recomputed != manifest.get("sha256"):
        raise SystemExit("manifest sha256 does not match panels (hand-edit suspected)")
    if not reason.strip():
        raise SystemExit("a reason is required")
    manifest["figure_state"] = to
    manifest.setdefault("promotion", []).append({
        "to": to,
        "reason": reason.strip(),
        "promoter": "promote_atlas_state.py",
        "git_head": _git_head(root),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"dir": str(d), "state": to, "panels": len(panels)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--to", required=True, choices=list(VALID))
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    result = promote(Path(args.dir), args.to, args.reason)
    print(f"promoted {result['dir']} -> {result['state']} "
          f"({result['panels']} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
