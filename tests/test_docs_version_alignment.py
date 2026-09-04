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
    assert re.match(r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$", version), f"Invalid version format in pyproject.toml: {version}"

def test_jaxfne_version_comparison():
    pyproject_version = get_pyproject_version()
    assert jaxfne.__version__ == pyproject_version, (
        f"jaxfne.__version__ ({jaxfne.__version__}) does not match pyproject.toml version ({pyproject_version})"
    )

def test_mkdocs_version_comparison():
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

def test_generated_version_md_comparison():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    version_md_path = root_dir / "docs" / "_generated" / "version.md"
    
    assert version_md_path.exists(), "docs/_generated/version.md does not exist"
    content = version_md_path.read_text(encoding="utf-8").strip()
    
    expected_content = f"Current source version: {pyproject_version}"
    assert content == expected_content, (
        f"docs/_generated/version.md content ({repr(content)}) does not match expected ({repr(expected_content)})"
    )

def test_install_md_latest_pypi_version():
    """docs/install.md must truthfully distinguish published PyPI from candidate version."""
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "docs" / "install.md").read_text(encoding="utf-8")

    match_rc = re.search(r"release candidate is \*\*`([^`]+)`\*\*", content)
    assert match_rc, "Could not find the 'release candidate is' claim in docs/install.md"
    assert match_rc.group(1) == pyproject_version, (
        f"docs/install.md claims candidate {match_rc.group(1)!r}, "
        f"pyproject.toml says {pyproject_version!r}"
    )
    # Published PyPI release must remain truthful (0.4.18 until post-publication)
    match_pub = re.search(r"published \*\*PyPI\*\* release is \*\*`jaxfne==([^`]+)`\*\*", content)
    assert match_pub, "Could not find published PyPI release in docs/install.md"
    assert match_pub.group(1) == "0.4.18"


def test_colab_md_version():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "docs" / "colab.md").read_text(encoding="utf-8")

    match = re.search(r"release candidate `([^`]+)`", content)
    assert match, "Could not find the release candidate version in docs/colab.md"
    assert match.group(1) == pyproject_version, (
        f"docs/colab.md claims candidate {match.group(1)!r}, pyproject.toml says {pyproject_version!r}"
    )
    assert "published PyPI release `jaxfne==0.4.18`" in content


def test_quickstart_md_documents_dev_contract():
    """quickstart.md tracks the development public contract (not a pinned PyPI stamp)."""
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "docs" / "quickstart.md").read_text(encoding="utf-8")

    assert "dev" in content.lower() or "NeuronalTensor" in content
    assert "Verified against `jaxfne==" not in content


def test_api_index_md_structural_contract_reference():
    """api/index.md references the frozen public surface contract, not a symbol inventory."""
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "docs" / "api" / "index.md").read_text(encoding="utf-8")

    assert "190" in content
    assert "public_surface_contract" in content
    assert "Latest PyPI release" not in content


def test_citation_md_version_fields():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "docs" / "citation.md").read_text(encoding="utf-8")

    matches = re.findall(r"version = \{([^}]+)\}", content)
    assert matches, "Could not find any 'version = {...}' BibTeX fields in docs/citation.md"
    for v in matches:
        assert v == pyproject_version, (
            f"docs/citation.md has a BibTeX version field {v!r}, pyproject.toml says {pyproject_version!r}"
        )


def test_citation_cff_version():
    pyproject_version = get_pyproject_version()
    root_dir = Path(__file__).resolve().parent.parent
    content = (root_dir / "CITATION.cff").read_text(encoding="utf-8")

    match = re.search(r"^version:\s*(.+)$", content, re.MULTILINE)
    assert match, "Could not find version: in CITATION.cff"
    assert match.group(1).strip() == pyproject_version, (
        f"CITATION.cff version ({match.group(1).strip()!r}) does not match pyproject.toml version ({pyproject_version!r})"
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
