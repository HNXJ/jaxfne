"""Provenance guard: published artifacts must be the artifacts the gates validated."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publish_consumes_validated_artifacts_and_never_builds():
    """Both publish jobs must ship retained bytes, not bytes made at publish time.

    Supersedes the earlier isolated-outdir rule. Building into a clean outdir
    stopped publish from shipping stale local ``dist/``, but the bytes it did
    ship had still passed no gate: twine check, the wheel smoke and the
    clean-room verification all ran against a different build. The invariant is
    now the stronger one -- publish builds nothing, and uploads only after
    SHA256 equality with the manifest generated beside those artifacts.
    """
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    executable = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("#") and "python -m build" in line
    ]
    assert executable == [], f"publish.yml still builds distributions: {executable}"

    jobs = text.count("pypa/gh-action-pypi-publish")
    assert jobs >= 2, "expected a TestPyPI job and a PyPI job"
    assert text.count("fetch_release_artifacts.sh") == jobs, (
        "every publish job must fetch the artifacts validated for its exact commit"
    )
    assert text.count("build_release_manifest.py --verify") == jobs, (
        "every publish job must require SHA256 equality before uploading"
    )


def test_dist_is_gitignored_and_untracked():
    """dist/ must stay gitignored so stale local bytes can never be the candidate."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dist/" in gitignore
    res = subprocess.run(
        ["git", "ls-files", "dist/"], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert res.stdout.strip() == "", "dist/ files are tracked; release must not depend on them"
