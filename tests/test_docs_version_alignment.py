import re
from pathlib import Path
import jaxfne

def get_pyproject_version() -> str:
    root_dir = Path(__file__).resolve().parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    
    project_match = re.search(r"^\[project\].*?^version\s*=\s*\"([^\"]+)\"", content, re.MULTILINE | re.DOTALL)
    if project_match:
        return project_match.group(1)
    
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", content, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError("Could not find project version in pyproject.toml")

def test_pyproject_version_format():
    version = get_pyproject_version()
    assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version format in pyproject.toml: {version}"

def test_jaxfne_version_alignment():
    pyproject_version = get_pyproject_version()
    assert jaxfne.__version__ == pyproject_version, (
        f"jaxfne.__version__ ({jaxfne.__version__}) does not match pyproject.toml version ({pyproject_version})"
    )

def test_mkdocs_version_alignment():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    mkdocs_path = root_dir / "mkdocs.yml"
    
    assert mkdocs_path.exists(), "mkdocs.yml does not exist"
    content = mkdocs_path.read_text(encoding="utf-8")
    
    match = re.search(r"jaxfne_version:\s*\"([^\"]+)\"", content)
    assert match, "Could not find jaxfne_version in mkdocs.yml"
    
    mkdocs_version = match.group(1)
    assert mkdocs_version == pyproject_version, (
        f"mkdocs.yml jaxfne_version ({mkdocs_version}) does not match pyproject.toml version ({pyproject_version})"
    )

def test_generated_version_md_alignment():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    version_md_path = root_dir / "docs" / "_generated" / "version.md"
    
    assert version_md_path.exists(), "docs/_generated/version.md does not exist"
    content = version_md_path.read_text(encoding="utf-8").strip()
    
    expected_content = f"Current source version: {pyproject_version}"
    assert content == expected_content, (
        f"docs/_generated/version.md content ({repr(content)}) does not match expected ({repr(expected_content)})"
    )

def test_no_stale_active_versions_in_public_docs():
    """Verify that active public docs do not present stale active-baseline versions (like stating current is 0.3.4)."""
    root_dir = Path(__file__).resolve().parent.parent
    docs_dir = root_dir / "docs"
    
    # We want to ignore archived/historical release pages or legacy internal material
    ignored_paths = [
        docs_dir / "releases",
        docs_dir / "_generated",
    ]
    
    # Stale pattern: e.g. claiming "current version is 0.3.4" or "active version: 0.3.4"
    stale_patterns = [
        re.compile(r"current version is 0\.3\.4", re.IGNORECASE),
        re.compile(r"active version:\s*0\.3\.4", re.IGNORECASE),
        re.compile(r"active version\s*0\.3\.4", re.IGNORECASE),
    ]
    
    for p in docs_dir.rglob("*.md"):
        if any(ignored in p.parents or ignored == p for ignored in ignored_paths):
            continue
            
        content = p.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            assert not pattern.search(content), (
                f"Found stale version indicator in public doc {p.relative_to(root_dir)} matching: {pattern.pattern}"
            )
