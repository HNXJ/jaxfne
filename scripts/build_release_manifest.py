#!/usr/bin/env python3
"""Bind built distributions to the source they were built from.

The manifest is *generated*, never committed. A manifest that names the commit
containing it cannot exist: writing it changes the tree, which changes the SHA
it claims to certify. So it is produced next to the artifacts it describes and
carried as a CI artifact, and ``--verify`` re-derives the hashes at consumption
time instead of trusting the recorded ones.

Usage
-----
Generate (release-build, after ``python -m build``)::

    python scripts/build_release_manifest.py --dist dist \
        --source-sha "$GITHUB_SHA" --ci-run-url "$URL" --out dist/RELEASE_MANIFEST.json

Verify (publish, before upload)::

    python scripts/build_release_manifest.py --verify dist/RELEASE_MANIFEST.json --dist dist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(dist: Path, pattern: str) -> Path:
    matches = sorted(dist.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} in {dist}, found {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _describe(path: Path) -> dict:
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _wheel_generator(wheel: Path) -> str | None:
    """Read ``Generator:`` out of the wheel's own WHEEL metadata.

    This is the build backend that actually produced these bytes, not the one a
    version range happens to resolve to now.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
        if len(names) != 1:
            return None
        for line in archive.read(names[0]).decode("utf-8").splitlines():
            if line.startswith("Generator:"):
                return line.split(":", 1)[1].strip()
    return None


def _require_backend_matches_pin(declared: list[str], observed: str | None) -> None:
    """The pin is only worth having if it is the version that actually built.

    Pinning ``hatchling==1.29.0`` while the wheel reports a different Generator
    means the build ran against something else -- a stale lock, a cached
    environment, a vendored backend -- and the pin is decorative.
    """
    if observed is None:
        raise SystemExit("wheel carries no Generator field; build backend unverifiable")
    name, _, version = observed.partition(" ")
    for spec in declared:
        pinned_name, sep, pinned_version = spec.partition("==")
        if not sep or pinned_name.strip() != name:
            continue
        if pinned_version.strip() != version.strip():
            raise SystemExit(
                f"build backend pin {spec!r} does not match the wheel's "
                f"Generator {observed!r}; the pinned backend is not what built these bytes"
            )
        return
    raise SystemExit(
        f"wheel Generator {observed!r} matches no exact pin in build-system.requires "
        f"{declared}; the build backend is unpinned or mismatched"
    )


def build(dist: Path, source_sha: str, ci_run_url: str | None) -> dict:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    wheel = _one(dist, "*.whl")
    sdist = _one(dist, "*.tar.gz")

    version = project["version"]
    for artifact in (wheel, sdist):
        if f"-{version}" not in artifact.name and f"-{version}." not in artifact.name:
            raise SystemExit(
                f"{artifact.name} does not carry pyproject version {version}; "
                "the dist directory holds artifacts from another version"
            )

    declared = pyproject["build-system"]["requires"]
    observed = _wheel_generator(wheel)
    _require_backend_matches_pin(declared, observed)

    return {
        "schema": "jaxfne.release_manifest/1",
        "version": version,
        "source_sha": source_sha,
        "source_tree_sha": _git("rev-parse", f"{source_sha}^{{tree}}"),
        "ci_run_url": ci_run_url,
        "python_support": {
            "requires_python": project["requires-python"],
            "classifiers": [
                c for c in project.get("classifiers", []) if "Programming Language :: Python ::" in c
            ],
        },
        "build_backend": {
            "declared": declared,
            "observed_generator": observed,
        },
        "wheel": _describe(wheel),
        "sdist": _describe(sdist),
    }


def verify(manifest_path: Path, dist: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for kind in ("wheel", "sdist"):
        recorded = manifest[kind]
        candidate = dist / recorded["filename"]
        if not candidate.is_file():
            failures.append(f"{kind}: {recorded['filename']} missing from {dist}")
            continue
        observed_size = candidate.stat().st_size
        observed_sha = _sha256(candidate)
        if observed_size != recorded["size_bytes"]:
            failures.append(
                f"{kind}: size {observed_size} != manifest {recorded['size_bytes']}"
            )
        if observed_sha != recorded["sha256"]:
            failures.append(f"{kind}: sha256 {observed_sha} != manifest {recorded['sha256']}")
        else:
            print(f"OK  {kind:5s} {recorded['filename']}  sha256={observed_sha}")

    # Anything else in dist/ would be publishable-by-accident.
    expected = {manifest["wheel"]["filename"], manifest["sdist"]["filename"], manifest_path.name}
    extra = sorted(p.name for p in dist.iterdir() if p.is_file() and p.name not in expected)
    if extra:
        failures.append(f"unattested files present in {dist}: {extra}")

    if failures:
        print("RELEASE MANIFEST VERIFY: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"RELEASE MANIFEST VERIFY: PASS ({manifest['version']} @ {manifest['source_sha']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--verify", type=Path, help="verify an existing manifest against --dist")
    parser.add_argument("--source-sha")
    parser.add_argument("--ci-run-url")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify, args.dist)

    if not args.source_sha:
        parser.error("--source-sha is required when generating a manifest")
    manifest = build(args.dist, args.source_sha, args.ci_run_url)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
