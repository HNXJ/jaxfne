#!/usr/bin/env python3
import sys
import re
from pathlib import Path
from argparse import ArgumentParser

# Ensure repository root is on sys.path to allow jaxfne import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def get_version_from_pyproject():
    """Extract version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return None
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None

def get_version_from_import():
    """Import jaxfne and read version."""
    try:
        import jaxfne
        return getattr(jaxfne, "__version__", None)
    except Exception as e:
        print(f"Error importing jaxfne: {e}")
        return None

def get_version_from_mkdocs():
    """Extract version from mkdocs.yml."""
    mkdocs_path = Path("mkdocs.yml")
    if not mkdocs_path.exists():
        return None
    content = mkdocs_path.read_text()
    match = re.search(r'jaxfne_version:\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None

def get_version_from_doc_version():
    """Extract version from docs/_generated/version.md."""
    doc_path = Path("docs/_generated/version.md")
    if not doc_path.exists():
        return None
    content = doc_path.read_text()
    match = re.search(r'Current source version:\s*([^\s\n]+)', content)
    return match.group(1) if match else None

def check_consistency():
    """Check if all required version strings exist and match exactly."""
    pyproject_ver = get_version_from_pyproject()
    import_ver = get_version_from_import()
    mkdocs_ver = get_version_from_mkdocs()
    doc_ver = get_version_from_doc_version()

    versions = {
        "pyproject.toml": pyproject_ver,
        "jaxfne.__version__ (import)": import_ver,
        "mkdocs.yml": mkdocs_ver,
        "docs/_generated/version.md": doc_ver,
    }

    print("Current versions:")
    all_present = True
    for name, ver in versions.items():
        status = "✓" if ver else "✗"
        print(f"  {status} {name}: {ver or 'NOT FOUND'}")
        if not ver:
            all_present = False

    if not all_present:
        print("\n✗ ERROR: Missing required version metadata source(s). Check failed.")
        return False

    ver_list = list(versions.values())
    all_match = all(v == ver_list[0] for v in ver_list)

    if all_match:
        print(f"\n✓ All versions consistent: {ver_list[0]}")
        return True
    else:
        print("\n✗ ERROR: Version mismatch detected across metadata sources.")
        return False

def main():
    parser = ArgumentParser(description="Check consistency of release metadata versions.")
    parser.add_argument("--check", action="store_true", help="Check consistency")
    args = parser.parse_args()

    # --check is the primary verification gate
    success = check_consistency()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
