#!/usr/bin/env python3
"""Sync jaxfne version from pyproject.toml to mkdocs.yml and docs/_generated/version.md."""

import os
import re
from pathlib import Path

def get_pyproject_version() -> str:
    root_dir = Path(__file__).resolve().parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    
    # We want to match version in the [project] section
    project_match = re.search(r"^\[project\].*?^version\s*=\s*\"([^\"]+)\"", content, re.MULTILINE | re.DOTALL)
    if project_match:
        return project_match.group(1)
        
    # Fallback to general search
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", content, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError("Could not find project version in pyproject.toml")

def sync_mkdocs_version(version: str):
    root_dir = Path(__file__).resolve().parent.parent
    mkdocs_path = root_dir / "mkdocs.yml"
    if not mkdocs_path.exists():
        print("Warning: mkdocs.yml not found, skipping sync.")
        return
        
    content = mkdocs_path.read_text(encoding="utf-8")
    
    # Check if 'extra:' block exists
    extra_match = re.search(r"^extra:", content, re.MULTILINE)
    if extra_match:
        # Check if 'jaxfne_version:' exists under 'extra:'
        # We look for jaxfne_version under extra, indented by spaces
        version_pattern = r"(?<=extra:\n)((\s+.*\n)*?)(\s+jaxfne_version:\s*\"[^\"]*\")(.*)"
        # But simpler: let's match any jaxfne_version under extra
        # Let's check if jaxfne_version exists anywhere
        if re.search(r"^\s+jaxfne_version:", content, re.MULTILINE):
            content = re.sub(
                r"^(\s+jaxfne_version:\s*)\"[^\"]*\"",
                rf'\g<1>"{version}"',
                content,
                flags=re.MULTILINE
            )
        else:
            # extra: exists, but no jaxfne_version. Let's insert it right after extra:
            content = re.sub(
                r"^(extra:)",
                rf'extra:\n  jaxfne_version: "{version}"',
                content,
                flags=re.MULTILINE
            )
    else:
        # extra: does not exist. Append it at the end of the file
        if not content.endswith("\n"):
            content += "\n"
        content += f'\nextra:\n  jaxfne_version: "{version}"\n'
        
    mkdocs_path.write_text(content, encoding="utf-8")
    print(f"Updated mkdocs.yml with version: {version}")

def write_generated_version_md(version: str):
    root_dir = Path(__file__).resolve().parent.parent
    gen_dir = root_dir / "docs" / "_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    version_md_path = gen_dir / "version.md"
    version_md_content = f"Current source version: {version}\n"
    version_md_path.write_text(version_md_content, encoding="utf-8")
    print(f"Wrote docs/_generated/version.md with version: {version}")

def _sub_file(path: Path, pattern: str, replacement: str, *, count: int = 0) -> bool:
    """Apply one regex substitution to a file; return True if it changed anything."""
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return False
    content = path.read_text(encoding="utf-8")
    new_content, n = re.subn(pattern, replacement, content, count=count, flags=re.MULTILINE)
    if n == 0:
        print(f"Warning: pattern not found in {path}, skipping.")
        return False
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
    return new_content != content


def sync_current_version_claims(version: str):
    """Sync the small set of docs pages that assert 'this is the latest PyPI
    release' as a live fact (not a historical 'verified as of vX.Y.Z' stamp,
    which is intentionally left alone -- it describes when something was last
    checked, not what today's version is).
    """
    root_dir = Path(__file__).resolve().parent.parent
    changed = []

    if _sub_file(root_dir / "docs" / "install.md",
                 r'The latest \*\*PyPI\*\* release is \*\*`jaxfne==[^`]+`\*\* \(tag `v[^`]+`\)',
                 rf'The latest **PyPI** release is **`jaxfne=={version}`** (tag `v{version}`)'):
        changed.append("docs/install.md (latest release line)")
    if _sub_file(root_dir / "docs" / "install.md",
                 r'pip install "jaxfne==[^"]+"',
                 rf'pip install "jaxfne=={version}"'):
        changed.append("docs/install.md (pin example)")

    if _sub_file(root_dir / "docs" / "colab.md",
                 r'\*\*Version:\*\* latest PyPI release `jaxfne==[^`]+` \(tag `v[^`]+`\)',
                 rf'**Version:** latest PyPI release `jaxfne=={version}` (tag `v{version}`)'):
        changed.append("docs/colab.md")

    if _sub_file(root_dir / "docs" / "quickstart.md",
                 r'Verified against `jaxfne==[^`]+`',
                 rf'Verified against `jaxfne=={version}`'):
        changed.append("docs/quickstart.md")

    if _sub_file(root_dir / "docs" / "api" / "index.md",
                 r'Latest PyPI release `jaxfne==[^`]+` \(tag `v[^`]+`\)',
                 rf'Latest PyPI release `jaxfne=={version}` (tag `v{version}`)'):
        changed.append("docs/api/index.md")

    if _sub_file(root_dir / "docs" / "citation.md",
                 r'version = \{[^}]+\}',
                 rf'version = {{{version}}}'):
        changed.append("docs/citation.md (version fields)")

    if _sub_file(root_dir / "CITATION.cff",
                 r'^version: .+$',
                 rf'version: {version}'):
        changed.append("CITATION.cff")

    if changed:
        print("Updated current-version claims in: " + ", ".join(changed))


def main():
    try:
        version = get_pyproject_version()
        print(f"Parsed version from pyproject.toml: {version}")
        sync_mkdocs_version(version)
        write_generated_version_md(version)
        sync_current_version_claims(version)
        print("Docs version sync completed successfully.")
    except Exception as e:
        print(f"Error during docs version sync: {e}")
        raise

if __name__ == "__main__":
    main()
