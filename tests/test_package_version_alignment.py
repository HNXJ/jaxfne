"""Ensure pyproject.toml version matches jaxfne.__version__.

This guards against the v0.0.8 incident where the GitHub tag/release shipped
with __version__ = "0.0.8" but pyproject.toml still declared "0.0.4". Wheel
and sdist metadata must agree with the runtime package version before any
TestPyPI/PyPI publication.
"""

from pathlib import Path

import jaxfne


def _read_pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    in_project = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = (line == "[project]")
            continue
        if in_project and line.startswith("version"):
            _, _, rhs = line.partition("=")
            return rhs.strip().strip('"').strip("'")
    raise AssertionError("project.version not found in pyproject.toml")


def test_pyproject_version_matches_runtime_version():
    assert _read_pyproject_version() == jaxfne.__version__
