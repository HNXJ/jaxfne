#!/usr/bin/env python3
"""
Release artifact validator for jaxfne.

Validates built wheel and sdist in dist/ before PyPI upload:
- sdist (.tar.gz) contains pyproject.toml and version matches expected
- wheel (.whl) contains METADATA and version matches expected
- Missing dist/ or empty dist/ fails in normal mode

Usage:
    # Normal release mode — fails if dist/ is missing or artifacts missing
    python scripts/release/validate_release_artifacts.py --version 0.3.14

    # Non-release readiness check — skips if dist/ absent
    python scripts/release/validate_release_artifacts.py --version 0.3.14 --allow-missing-dist

Exit codes:
    0 — all required artifacts found and valid
    1 — validation failed (missing, wrong version, or corrupt artifact)
"""

import argparse
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path


def check_sdist(path: Path, expected_version: str) -> list[str]:
    """Validate sdist tarball. Returns list of error strings."""
    errors = []
    try:
        with tarfile.open(path, "r:gz") as tar:
            members = tar.getnames()
            has_pyproject = any("pyproject.toml" in m for m in members)
            if not has_pyproject:
                errors.append(f"{path.name}: missing pyproject.toml in sdist")
                return errors

            # Extract pyproject.toml content and check version
            pyproject_member = next(
                (m for m in members if m.endswith("pyproject.toml")), None
            )
            if pyproject_member:
                f = tar.extractfile(pyproject_member)
                if f:
                    content = f.read().decode("utf-8", errors="replace")
                    m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                    if m:
                        found_version = m.group(1)
                        if found_version != expected_version:
                            errors.append(
                                f"{path.name}: pyproject version {found_version!r} "
                                f"!= expected {expected_version!r}"
                            )
                    else:
                        errors.append(f"{path.name}: could not find version in pyproject.toml")
    except Exception as e:
        errors.append(f"{path.name}: failed to open sdist: {e}")
    return errors


def check_wheel(path: Path, expected_version: str) -> list[str]:
    """Validate wheel zip. Returns list of error strings."""
    errors = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = zf.namelist()
            metadata_member = next(
                (m for m in members if re.search(r"[\\/]METADATA$", m)), None
            )
            if not metadata_member:
                errors.append(f"{path.name}: missing METADATA file in wheel")
                return errors

            content = zf.read(metadata_member).decode("utf-8", errors="replace")
            m = re.search(r"^Version:\s*(.+)$", content, re.MULTILINE)
            if m:
                found_version = m.group(1).strip()
                if found_version != expected_version:
                    errors.append(
                        f"{path.name}: wheel METADATA version {found_version!r} "
                        f"!= expected {expected_version!r}"
                    )
            else:
                errors.append(f"{path.name}: could not find Version in METADATA")
    except Exception as e:
        errors.append(f"{path.name}: failed to open wheel: {e}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate release artifacts in dist/.")
    parser.add_argument("--version", required=True, help="Expected version string (e.g. 0.3.14)")
    parser.add_argument(
        "--dist",
        default="dist",
        help="Path to dist directory (default: dist/)",
    )
    parser.add_argument(
        "--allow-missing-dist",
        action="store_true",
        help="Skip gracefully if dist/ is absent (non-release readiness check mode)",
    )
    args = parser.parse_args()

    dist_path = Path(args.dist)
    all_errors = []

    if not dist_path.exists() or not dist_path.is_dir():
        if args.allow_missing_dist:
            print(f"dist/ not found at {dist_path}. --allow-missing-dist set. Skipping.")
            sys.exit(0)
        else:
            print(f"ERROR: dist/ directory not found at {dist_path}. Build artifacts first.")
            sys.exit(1)

    artifacts = list(dist_path.iterdir())
    if not artifacts:
        if args.allow_missing_dist:
            print("dist/ is empty. --allow-missing-dist set. Skipping.")
            sys.exit(0)
        else:
            print(f"ERROR: dist/ is empty. Build artifacts first.")
            sys.exit(1)

    sdists = [f for f in artifacts if f.name.endswith(".tar.gz")]
    wheels = [f for f in artifacts if f.name.endswith(".whl")]
    other = [f for f in artifacts if not f.name.endswith((".tar.gz", ".whl"))]

    if not sdists:
        all_errors.append(f"No sdist (.tar.gz) found in {dist_path}")
    if not wheels:
        all_errors.append(f"No wheel (.whl) found in {dist_path}")

    for sdist in sdists:
        print(f"Checking sdist: {sdist.name}")
        all_errors.extend(check_sdist(sdist, args.version))

    for wheel in wheels:
        print(f"Checking wheel: {wheel.name}")
        all_errors.extend(check_wheel(wheel, args.version))

    if other:
        print(f"Note: {len(other)} unrecognised file(s) in dist/ ignored.")

    if all_errors:
        print("\nValidation FAILED:")
        for e in all_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(sdists)} sdist(s) and {len(wheels)} wheel(s) valid for v{args.version}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
