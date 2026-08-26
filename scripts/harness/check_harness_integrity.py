#!/usr/bin/env python3
"""Harness v2.1 integrity verification.

Verifies every file listed in the project and global HARNESS_MANIFEST.json
against its recorded sha256, plus the frozen-path manifest
(.opencode/frozen_paths.json) for write-once publication artifacts.

Exit codes: 0 PASS; 1 CANONICAL_CHANGED/other content mismatch;
2 MIRROR_DRIFT (manual edit of a generated mirror); 3 SOURCE_MISSING;
4 FROZEN_MISMATCH (a frozen publication artifact changed).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_MANIFEST = ROOT / "scripts" / "harness" / "HARNESS_MANIFEST.json"
GLOBAL_MANIFEST = Path.home() / ".config" / "opencode" / "harness" / "HARNESS_MANIFEST.json"
FROZEN_PATHS = ROOT / "artifacts" / "publication" / "frozen_manifest.json"
# DEPRECATED: legacy .opencode/frozen_paths.json fallback kept for migration from the
# pre-artifacts layout; do not add new references to it.
_FROZEN_PATHS_OLD = ROOT / ".opencode" / "frozen_paths.json"
MIRROR_PREFIXES = (".opencode/skills/", ".cursor/skills/")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_manifest(manifest: Path, base: Path):
    if not manifest.exists():
        print(f"INFO: manifest not present (expected for {manifest})")
        return [], [], []
    m = json.loads(manifest.read_text())
    components = m.get("components", {})
    mismatched, missing, drift = [], [], []
    for comp, entry in components.items():
        if isinstance(entry, dict) and "file" in entry:
            f = Path(entry["file"].replace("~", str(Path.home())))
            if not f.is_absolute():
                f = base / f
            want = entry["sha256"]
            got = None if not f.exists() else sha(f)
            if got is None:
                missing.append(f"{comp} ({f})")
                continue
            if got != want:
                if any(p in str(f) for p in MIRROR_PREFIXES):
                    drift.append(f"{comp} ({f})")
                else:
                    mismatched.append(f"{comp} ({f})")
        elif isinstance(entry, dict):
            for rel, item in entry.items():
                if isinstance(item, dict) and "file" in item:
                    f = Path(item["file"].replace("~", str(Path.home())))
                    if not f.is_absolute():
                        f = base / f
                    want = item["sha256"]
                else:
                    want = item
                    if comp == "canonical_skills":
                        f = base / "artifacts" / "skills" / rel / "SKILL.md"
                    else:
                        f = base / rel
                got = None if not f.exists() else sha(f)
                if got is None:
                    missing.append(f"{comp}/{rel} ({f})")
                    continue
                if got != want:
                    if comp == "mirrors" or any(p in str(f) for p in MIRROR_PREFIXES):
                        drift.append(f"{comp}/{rel} ({f})")
                    else:
                        mismatched.append(f"{comp}/{rel} ({f})")
    return mismatched, missing, drift


def verify_frozen() -> list[str]:
    # Migration: prefer new neutral owner, fallback to old .opencode path
    path = FROZEN_PATHS if FROZEN_PATHS.exists() else _FROZEN_PATHS_OLD
    if not path.exists():
        return ["frozen_paths.json missing (checked %s and %s)" % (FROZEN_PATHS, _FROZEN_PATHS_OLD)]
    m = json.loads(path.read_text())
    bad = []
    for rel, want in m.get("files", {}).items():
        f = ROOT / rel
        if not f.exists() or sha(f) != want:
            bad.append(rel)
    return bad


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-manifest", type=Path, default=PROJECT_MANIFEST,
                    help="override project manifest path (adversarial replay)")
    a = ap.parse_args()
    rc = 0
    pm, pmiss, pdrift = verify_manifest(a.project_manifest, ROOT)
    gm, gmiss, gdrift = verify_manifest(GLOBAL_MANIFEST, Path.home() / ".config" / "opencode")
    frozen_bad = verify_frozen()
    for label, items in [
        ("PROJECT-CONTENT", pm), ("GLOBAL-CONTENT", gm),
        ("PROJECT-DRIFT", pdrift), ("GLOBAL-DRIFT", gdrift),
        ("MISSING", pmiss + gmiss), ("FROZEN", frozen_bad),
    ]:
        for it in items:
            print(f"FAIL[{label}]: {it}")
    if pm or gm:
        rc = 1
    if pdrift or gdrift:
        rc = 2
    if pmiss or gmiss:
        rc = 3
    if frozen_bad:
        rc = 4
    if rc == 0:
        print("integrity PASS (project + global manifests, frozen paths)")
    else:
        print(f"integrity FAIL (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())