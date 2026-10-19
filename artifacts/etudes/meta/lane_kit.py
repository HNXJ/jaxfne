"""Shared helpers for paper replication lanes (R1).

One copy of the guards every lane re-implements: the git SHA stamp, the
scored-directory refusal and the one-SHA-per-directory refusal, plus the
manifest writer. New lanes import this module; migrated lanes delete their
local copies. Semantics are the strictest of the copies found in the tree
(the ``caminos_api.py`` messages and conditions).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_sha(repo_root):
    """Git SHA via ``git rev-parse HEAD`` at ``repo_root`` (``"unknown"`` when git fails)."""
    out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()


def _is_scored_part(part):
    """A path part naming a scored directory: ``scored`` or ``scored_*``/``scored-``."""
    low = str(part).lower()
    return low == "scored" or low.startswith("scored_") or low.startswith("scored-")


def refuse_scored_dir(out_dir, signed=False):
    """Refuse a directory under a ``scored`` path unless ``signed`` is passed.

    Only a path part equal to ``scored`` (case-insensitive) or starting
    with ``scored_``/``scored-`` counts; ``unscored`` pilot folders pass.
    """
    if signed:
        return
    parts = Path(out_dir).resolve().parts
    if any(_is_scored_part(part) for part in parts):
        raise SystemExit("error: refusing to write under a directory containing "
                         "'scored': %s" % out_dir)


def refuse_existing(out_dir, fname, sha, *, signed=False):
    """Refuse a directory holding another code SHA or this run's file; return its manifest.

    With ``signed=True`` an ``"unknown"`` SHA is refused: a scored run
    needs provenance.
    """
    if signed and sha == "unknown":
        raise SystemExit("error: refusing scored run without code provenance "
                         "(git SHA is unknown)")
    out = Path(out_dir)
    prior = None
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if prior.get("git_sha") != sha:
            raise SystemExit("error: %s holds runs from code %s; this run is %s (one code SHA "
                             "per directory)" % (out, prior.get("git_sha"), sha))
        if any(e["file"] == fname for e in prior["runs"]):
            raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    if (out / fname).exists():
        raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    return prior


def write_manifest(out_dir, fname, payload, *, sha):
    """Write the manifest JSON ``fname`` under ``out_dir`` (sorted keys, atomic replace).

    ``payload`` is the lane's manifest content (its ``runs`` entries and any
    lane keys). The writer records the code ``sha``, the UTC timestamp, the
    jaxfne version and the platform/device, and returns the manifest path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from jaxfne import __version__ as jaxfne_version
    except Exception:
        jaxfne_version = "unknown"
    doc = dict(payload)
    doc["git_sha"] = sha
    doc.setdefault("start_time", datetime.now(timezone.utc).isoformat())
    doc["jaxfne_version"] = jaxfne_version
    doc["platform"] = sys.platform
    try:
        import jax

        doc["jax_version"] = jax.__version__
        doc["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    tmp = out / (fname + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    os.replace(tmp, out / fname)
    return out / fname
