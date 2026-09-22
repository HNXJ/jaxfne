#!/usr/bin/env python3
"""Harness v2.1 skill synchronization: canonical artifacts/skills/ -> generated client mirrors.

--check             verify canonical skills match the manifest (and any
                    configured mirrors are byte-identical)
--update            regenerate mirrors from canonical artifacts/skills/
--update --manifest also refresh mirror/canonical hashes in the project HARNESS_MANIFEST.json
Exit codes: 0 ok, 1 drift/missing.

NOTE: mirrors are tool-local outside the repository, so MIRRORS is empty
and no mirror bytes are checked here. --check is non-vacuous via the
canonical-vs-manifest comparison below; it must fail on deliberate drift.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "artifacts" / "skills"
# Mirrors are now tool-local outside the repository (e.g., ~/.config/opencode/skills).
# Keeping the list empty satisfies the "project authority not in tool-specific hidden directory" invariant.
MIRRORS: list[Path] = []
MANIFEST = ROOT / "scripts/harness/HARNESS_MANIFEST.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical_skills() -> list[Path]:
    return sorted(CANONICAL.glob("*/SKILL.md"))


def check() -> int:
    canon = {p.parent.name: p for p in canonical_skills()}
    if not canon:
        print("ERROR: no canonical skills under", CANONICAL)
        return 1
    # Mirrors in .opencode/ and .cursor/ are now tool-local, gitignored, and not required for harness integrity.
    # If the mirror parent directory is gitignored, skip the check (tool will regenerate outside repo if needed).
    import subprocess

    def is_ignored(p: Path) -> bool:
        try:
            subprocess.check_output(["git", "check-ignore", "-q", str(p)], cwd=ROOT)
            return True
        except subprocess.CalledProcessError:
            return False

    manifest_hashes = {}
    canonical_hashes = {}
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        manifest_hashes = m.get("components", {}).get("mirrors", {})
        canonical_hashes = m.get("components", {}).get("canonical_skills", {})
    failures = []
    # Non-vacuous core: every canonical skill must match the manifest record.
    # (Mirror bytes are tool-local and unmanaged; the loop below checks zero
    # mirrors by design and must never be the sole basis for success.)
    if MANIFEST.exists():
        for name, cp in canon.items():
            want = canonical_hashes.get(name)
            if want is None:
                failures.append(f"MANIFEST-MISSING canonical_skills/{name}")
            elif sha(cp) != want:
                failures.append(f"CANONICAL-MISMATCH canonical_skills/{name}")
    for mirror in MIRRORS:
        if is_ignored(mirror):
            continue
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
    print(
        f"sync OK: {len(canon)} canonical skills match manifest; "
        f"{len(MIRRORS)} mirrors (tool-local, unmanaged, unchecked)"
    )
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
            comp["project_agents"]["sha256"] = sha(ROOT / "artifacts" / "AGENTS.md")
        if "harness_scripts" in comp:
            comp["harness_scripts"] = {
                name: {
                    "file": f"scripts/harness/{name}.py",
                    "sha256": sha(ROOT / "scripts" / "harness" / f"{name}.py"),
                }
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
