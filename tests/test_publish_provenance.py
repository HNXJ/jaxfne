"""Provenance guard: release artifacts must be fresh builds, never stale dist/."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publish_builds_into_isolated_outdir():
    """Both publish jobs must build to and publish from an isolated outdir."""
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert text.count("--outdir /tmp/jaxfne-publish-dist") >= 2, (
        "publish.yml must build into an isolated outdir in both jobs"
    )
    assert text.count("packages-dir: /tmp/jaxfne-publish-dist") >= 2, (
        "publish.yml must publish from the isolated outdir in both jobs"
    )
    assert "python -m build\n" not in text, (
        "bare `python -m build` (dist/ default) must not remain in publish.yml"
    )


def test_dist_is_gitignored_and_untracked():
    """dist/ must stay gitignored so stale local bytes can never be the candidate."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dist/" in gitignore
    res = subprocess.run(
        ["git", "ls-files", "dist/"], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert res.stdout.strip() == "", "dist/ files are tracked; release must not depend on them"
