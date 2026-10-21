"""v0.3.30a docs atlas and link validation tests.

Validate that docs build cleanly and links resolve.
"""

import shutil
import subprocess
import pathlib

import pytest


class TestDocsLinksV0330:
    """Validate docs links and mkdocs build for v0.3.30a."""

    def test_mkdocs_strict_build_passes(self):
        """MkDocs must build with --strict flag."""
        if not shutil.which("mkdocs"):
            pytest.skip("mkdocs not installed; skipping docs build test")

        result = subprocess.run(
            ["mkdocs", "build", "--strict"],
            capture_output=True,
            text=True,
        encoding="utf-8",
            cwd=pathlib.Path.cwd(),
            # 16 s alone; over 120 s under the xdist broad gate on a loaded machine (rc, 2026-10-21).
            # The test checks the strict build, so the limit only bounds a hang.
            timeout=600,
        )

        assert result.returncode == 0, (
            f"mkdocs build --strict failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_docs_api_markdown_files_exist(self):
        """All expected API documentation files must exist."""
        docs_api_dir = pathlib.Path("docs/reference/api")
        expected_files = [
            "core/core.md",
            "model/emitters.md",
            "readout/fields.md",
            "readout/probes.md",
            "readout/objectives.md",
            "core/runtime.md",
            "core/validation.md",
            "index.md",
        ]

        for filename in expected_files:
            filepath = docs_api_dir / filename
            assert (
                filepath.exists()
            ), f"Expected API doc missing: {filepath}"

    def test_docs_version_markdown_consistency(self):
        """docs/_generated/version.md must match package version."""
        import jaxfne as jtfne

        version_md = pathlib.Path("docs/_generated/version.md")
        assert version_md.exists(), f"Version markdown not found: {version_md}"

        with open(version_md) as f:
            content = f.read()

        assert jtfne.__version__ in content, (
            f"Package version {jtfne.__version__} not found in {version_md}. "
            f"Content: {content}"
        )
