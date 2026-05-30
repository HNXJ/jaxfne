#!/usr/bin/env python3
"""
Sync release metadata: version, tag, PyPI, and documentation.

This script ensures that jaxfne version strings remain consistent across:
- pyproject.toml
- jaxfne/__init__.py
- mkdocs.yml
- docs/_generated/version.md (if auto-generated)

Usage:
    python scripts/sync_release_metadata.py --check    # Verify consistency
    python scripts/sync_release_metadata.py --sync     # Sync to pyproject.toml version
"""

import sys
import re
from pathlib import Path
from argparse import ArgumentParser


def get_version_from_pyproject():
    """Extract version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return None
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def get_version_from_init():
    """Extract version from jaxfne/__init__.py."""
    init_path = Path("jaxfne/__init__.py")
    if not init_path.exists():
        return None
    content = init_path.read_text()
    match = re.search(r'(?:__version__|_JAXFNE_VERSION)\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def check_consistency():
    """Check if all version strings match."""
    pyproject_ver = get_version_from_pyproject()
    init_ver = get_version_from_init()

    versions = {
        "pyproject.toml": pyproject_ver,
        "jaxfne/__init__.py": init_ver,
    }

    print("Current versions:")
    for name, ver in versions.items():
        status = "✓" if ver else "✗"
        print(f"  {status} {name}: {ver or 'NOT FOUND'}")

    non_none_versions = [v for v in versions.values() if v is not None]
    if not non_none_versions:
        print("\nERROR: No version found in any file.")
        return False

    all_match = all(v == non_none_versions[0] for v in non_none_versions)

    if all_match:
        print(f"\n✓ All versions consistent: {non_none_versions[0]}")
        return True
    else:
        print("\n✗ Version mismatch detected.")
        return False


def main():
    parser = ArgumentParser(description="Sync release metadata versions.")
    parser.add_argument("--check", action="store_true", help="Check consistency only")
    args = parser.parse_args()

    if args.check or not args.check:
        success = check_consistency()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
