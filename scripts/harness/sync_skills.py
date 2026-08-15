#!/usr/bin/env python3
"""Harness v2.1 skill synchronization: canonical skills/ -> generated client mirrors.

--check             verify every mirror file is byte-identical to canonical (and manifest, if present)
--update            regenerate mirrors from canonical skills/
--update --manifest also refresh mirror/canonical hashes in the project HARNESS_MANIFEST.json
Exit codes: 0 ok, 1 drift/missing.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "skills"
MIRRORS = [ROOT / ".opencode/skills", ROOT / ".cursor/skills"]
MANIFEST = ROOT / ".opencode" / "HARNESS_MANIFEST.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical_skills() -> list[Path]:
    return sorted(CANONICAL.glob("*/SKILL.md"))


def check() -> int:
    canon = {p.parent.name: p for p in canonical_skills()}
    if not canon:
        print("ERROR: no canonical skills under", CANONICAL)
        return 1
    manifest_hashes = {}
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        manifest_hashes = m.get("components", {}).get("mirrors", {})
    failures = []
    for mirror in MIRRORS:
        for name, cp in canon.items():
            mp = mirror / name / "SKILL.md"
            if not mp.exists():
                failures.append(f"MISSING {mp.relative_to(ROOT)}")
                continue
            if sha(cp) != sha(mp):
                failures.append(f"DRIFT {mp.relative_to(ROOT)} != canonical/{name}")
            rel = str((mirror / name / "SKILL.md").relative_to(ROOT))
            if rel in manifest_hashes and manifest_hashes[rel] != sha(mp):
                failures.append(f"MANIFEST-MISMATCH {rel}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print(f"sync OK: {len(canon)} skills x {len(MIRRORS)} mirrors identical")
    return 0


def update(refresh_manifest: bool) -> int:
    canon = {p.parent.name: p for p in canonical_skills()}
    if not canon:
        print("ERROR: no canonical skills under", CANONICAL)
        return 1
    for mirror in MIRRORS:
        for name, cp in canon.items():
            mp = mirror / name / "SKILL.md"
            mp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cp, mp)
        print("mirror updated:", mirror.relative_to(ROOT))
    if refresh_manifest:
        if not MANIFEST.exists():
            print("ERROR: manifest missing; create it before --manifest", MANIFEST)
            return 1
        m = json.loads(MANIFEST.read_text())
        m["components"]["canonical_skills"] = {n: sha(p) for n, p in canon.items()}
        m["components"]["mirrors"] = {
            str((mirror / n / "SKILL.md").relative_to(ROOT)): sha(mirror / n / "SKILL.md")
            for mirror in MIRRORS
            for n in canon
        }
        comp = m["components"]
        if "project_agents" in comp:
            comp["project_agents"]["sha256"] = sha(ROOT / "AGENTS.md")
        if "harness_scripts" in comp:
            comp["harness_scripts"] = {
                name: {"file": f"scripts/harness/{name}.py", "sha256": sha(ROOT / "scripts" / "harness" / f"{name}.py")}
                for name in comp["harness_scripts"]
            }
        m["generated_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
        print("manifest refreshed:", MANIFEST)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    if a.update:
        return update(a.manifest)
    return check()


if __name__ == "__main__":
    sys.exit(main())