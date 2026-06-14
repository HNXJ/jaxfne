#!/usr/bin/env python3
"""Generate Extended Data ED10: release/archive receipt panel.

Records branch, commit SHA, evidence inventory, checklist/output hashes,
and release/archive field status. Does not tag, release, publish, or archive.

Outputs:
  figures/evidence/ed10_release_archive_receipt.png
  outputs/evidence/ed10_release_archive_receipt_manifest.json
  outputs/evidence/ed10_release_archive_receipt_receipt.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _figure_common import (
    ensure_evidence_dirs,
    jaxfne_version,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)

_REPO_ROOT = repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SOURCE_FILES = [
    "scripts/evidence_figures/ed10_release_archive_receipt.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_inventory.py",
    "docs/evidence/evidence_checklist.json",
    "pyproject.toml",
]

TITLE = "Extended Data 10 - release/archive receipt (evidence panel)"

SCOPE_STATUS = (
    "Release/archive evidence receipt only; no tag/release/publish/archive action; "
    "not empirical validation; physical_amplitude_claim_allowed: false"
)

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed10_release_archive_receipt.png"
RECEIPT_PATH = _dirs["outputs"] / "ed10_release_archive_receipt_receipt.json"


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_branch() -> str:
    return _run_git(["branch", "--show-current"]) or "unknown"


def _git_status_short() -> list[str]:
    out = _run_git(["status", "--short"]) or ""
    return [line for line in out.splitlines() if line.strip()]


def _runtime_receipt() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _load_inventory() -> dict[str, Any]:
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from evidence_inventory import build_inventory

    return build_inventory()


def _checklist_hash() -> dict[str, Any]:
    path = _REPO_ROOT / "docs" / "evidence" / "evidence_checklist.json"
    return {
        "path": str(path.relative_to(_REPO_ROOT)),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _output_inventory_hash(inventory: dict[str, Any]) -> dict[str, Any]:
    path = _REPO_ROOT / "outputs" / "evidence" / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(_REPO_ROOT)),
        "exists": True,
        "sha256": sha256_file(path),
        "summary": inventory.get("summary", {}),
    }


def _wheel_sdist_status() -> dict[str, Any]:
    dist = _REPO_ROOT / "dist"
    if not dist.is_dir():
        return {
            "status": "not_applicable_without_release_approval",
            "build_performed": False,
            "publish_performed": False,
            "reason": "wheel/sdist build not approved in this run",
            "artifacts": [],
        }
    artifacts: list[dict[str, Any]] = []
    for path in sorted(dist.glob("*")):
        if path.is_file() and path.suffix in {".whl", ".tar.gz"}:
            artifacts.append(
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not artifacts:
        return {
            "status": "pending_approval",
            "build_performed": False,
            "publish_performed": False,
            "reason": "dist/ exists but no wheel/sdist artifacts; build not approved",
            "artifacts": [],
        }
    return {
        "status": "pending_approval",
        "build_performed": False,
        "publish_performed": False,
        "reason": "pre-existing dist artifacts recorded; publish not approved",
        "artifacts": artifacts,
    }


def _tag_status() -> dict[str, Any]:
    version = jaxfne_version()
    tag_name = f"v{version}" if not version.startswith("v") else version
    tag_sha = _run_git(["rev-list", "-n", "1", tag_name])
    head = repo_sha()
    if tag_sha and tag_sha == head:
        return {
            "status": "existing_at_head",
            "tag_name": tag_name,
            "tag_commit_sha": tag_sha,
            "matches_head": True,
            "action_taken": False,
        }
    if tag_sha:
        return {
            "status": "pending_approval",
            "tag_name": tag_name,
            "tag_commit_sha": tag_sha,
            "matches_head": False,
            "action_taken": False,
            "reason": "version tag exists but not at current HEAD; retag not approved",
        }
    return {
        "status": "not_created",
        "tag_name": tag_name,
        "tag_commit_sha": None,
        "matches_head": False,
        "action_taken": False,
        "reason": "no matching version tag; tag creation not approved",
    }


def _archive_doi_status() -> dict[str, Any]:
    return {
        "status": "not_applicable_without_release_approval",
        "archive_url": None,
        "doi": None,
        "action_taken": False,
        "reason": "archive/DOI creation not approved in this run",
    }


def _root_import_laziness_smoke() -> dict[str, Any]:
    code = """
import sys
mods_before = set(sys.modules)
import jaxfne
mods_after = set(sys.modules) - mods_before
optional_hits = sorted(
    m for m in mods_after if m.split('.')[0] in {'pandas','plotly','jaxley','pynwb','h5py'}
)
print(jaxfne.__version__)
print(','.join(optional_hits))
"""
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
    version = lines[0] if lines else "unknown"
    optional = lines[1].split(",") if len(lines) > 1 and lines[1] else []
    optional = [x for x in optional if x]
    return {
        "gate_outcome": "pass" if result.returncode == 0 and not optional else "fail",
        "jaxfne_version": version,
        "optional_loaded": optional,
        "exit_code": result.returncode,
        "validator": "isolated_subprocess_root_import",
    }


def _clean_install_smoke() -> dict[str, Any]:
    return {
        "status": "pending_approval",
        "pip_install_performed": False,
        "reason": "full clean-room pip install not approved; isolated import smoke recorded",
        "import_smoke": _root_import_laziness_smoke(),
    }


def _mkdocs_status() -> dict[str, Any]:
    candidates = [
        _REPO_ROOT / ".venv-docs" / "bin" / "python",
        Path(sys.executable),
    ]
    for py in candidates:
        if not py.is_file():
            continue
        result = subprocess.run(
            [str(py), "-m", "mkdocs", "build", "--strict"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return {
                "gate_outcome": "pass",
                "python": str(py),
                "validator": "mkdocs build --strict",
                "exit_code": 0,
            }
    return {
        "gate_outcome": "fail",
        "python": str(sys.executable),
        "validator": "mkdocs build --strict",
        "exit_code": 1,
        "reason": "mkdocs strict build did not pass in available interpreters",
    }


def _remaining_limitations(inventory: dict[str, Any], tag: dict[str, Any]) -> list[str]:
    summary = inventory.get("summary", {})
    limits = [
        "computational_scaffold_only; not empirical EEG/MEG validation",
        "laminar_proxy_no_pde; no PDE solver claim",
        "physical_amplitude_claim_allowed: false",
        "wheel/sdist build and PyPI publish require explicit approval",
        "archive/DOI assignment requires explicit approval",
    ]
    if summary.get("extended_data_present", 0) < summary.get("extended_data_total", 10):
        limits.append("ED10 panel completes inventory; prior ED count was 9/10 before this run")
    if tag.get("status") == "existing" and not tag.get("matches_head"):
        limits.append(f"version tag {tag.get('tag_name')} does not point at current HEAD")
    return limits


def build_receipt_body() -> dict[str, Any]:
    status_lines = _git_status_short()
    inventory = _load_inventory()
    checklist = _checklist_hash()
    output_inventory = _output_inventory_hash(inventory)
    tag = _tag_status()
    archive = _archive_doi_status()
    wheel = _wheel_sdist_status()
    clean_install = _clean_install_smoke()
    mkdocs = _mkdocs_status()
    summary = inventory.get("summary", {})

    body = {
        "schema_version": "jaxfne.evidence_ed10_release_archive_receipt.v0.1.0",
        "artifact": "ed10_release_archive_receipt",
        "generated_at_utc": utc_now_iso(),
        "runtime_receipt": _runtime_receipt(),
        "scope_status": SCOPE_STATUS,
        "release_action_taken": False,
        "tag_action_taken": False,
        "archive_action_taken": False,
        "publish_action_taken": False,
        "local_receipt_only": True,
        "empirical_validation_claim_allowed": False,
        "mechanism_claim_allowed": False,
        "physical_amplitude_claim_allowed": False,
        "repo_state": {
            "branch": _git_branch(),
            "commit_sha": repo_sha(),
            "working_tree_clean": len(status_lines) == 0,
            "status_short": status_lines,
        },
        "package_version": jaxfne_version(),
        "evidence_inventory": {
            "main_figures": (
                f"{summary.get('main_figures_present', 0)}/"
                f"{summary.get('main_figures_total', 8)}"
            ),
            "extended_data": (
                f"{summary.get('extended_data_present', 0)}/"
                f"{summary.get('extended_data_total', 10)}"
            ),
            "summary": summary,
        },
        "checklist_hash": checklist,
        "output_inventory_hash": output_inventory,
        "wheel_sdist_status": wheel,
        "clean_install_smoke": clean_install,
        "tag_status": tag,
        "archive_doi_status": archive,
        "mkdocs_strict_build": mkdocs,
        "root_import_laziness": clean_install["import_smoke"],
        "remaining_limitations": _remaining_limitations(inventory, tag),
        "truth_gates": truth_gates(),
        "claim_boundary": "release_archive_evidence_receipt_only",
    }
    return body


def _validate_receipt(body: dict[str, Any]) -> None:
    """Hard-fail only on reproducibility/docs gates, not release-pending fields."""
    failures: list[str] = []
    if body["root_import_laziness"].get("gate_outcome") != "pass":
        failures.append("root import laziness")
    if body["mkdocs_strict_build"].get("gate_outcome") != "pass":
        failures.append("mkdocs strict build")
    if body.get("release_action_taken"):
        failures.append("release action taken without approval")
    if body.get("tag_action_taken"):
        failures.append("tag action taken without approval")
    if body.get("archive_action_taken"):
        failures.append("archive action taken without approval")
    if failures:
        raise RuntimeError(f"ED10 gate failures: {', '.join(failures)}")


def draw_figure(body: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.0, 8.55, TITLE, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(
        7.0,
        8.05,
        "evidence panel only; no release/tag/archive action taken",
        ha="center",
        fontsize=8,
        family="monospace",
        color="#C62828",
    )

    rows = [
        ("branch", body["repo_state"]["branch"]),
        ("commit_sha", body["repo_state"]["commit_sha"][:12]),
        ("tree_clean", str(body["repo_state"]["working_tree_clean"])),
        ("package_version", body["package_version"]),
        ("main_figures", body["evidence_inventory"]["main_figures"]),
        ("extended_data", body["evidence_inventory"]["extended_data"]),
        ("checklist_sha256", (body["checklist_hash"].get("sha256") or "")[:16]),
        ("inventory_sha256", (body["output_inventory_hash"].get("sha256") or "")[:16]),
        ("tag_status", body["tag_status"]["status"]),
        ("tag_matches_head", str(body["tag_status"].get("matches_head"))),
        ("wheel_sdist", body["wheel_sdist_status"]["status"]),
        ("archive_doi", body["archive_doi_status"]["status"]),
        ("import_smoke", body["root_import_laziness"]["gate_outcome"]),
        ("mkdocs_strict", body["mkdocs_strict_build"]["gate_outcome"]),
        ("release_action", str(body["release_action_taken"])),
    ]

    y = 7.35
    for key, val in rows:
        ax.text(0.4, y, f"{key}:", fontsize=7.5, fontweight="bold", family="monospace")
        ok_vals = {
            "pass",
            "True",
            "existing_at_head",
            "not_applicable_without_release_approval",
            "pending_approval",
            "not_created",
        }
        color = "#2E7D32" if val in ok_vals else "#333333"
        if val == "fail":
            color = "#C62828"
        ax.text(4.0, y, str(val), fontsize=7.5, family="monospace", color=color)
        y -= 0.42

    ax.text(0.4, 1.35, SCOPE_STATUS, fontsize=6.5, family="monospace", color="#444444")
    limits = body.get("remaining_limitations", [])[:3]
    ax.text(
        0.4,
        0.75,
        "limits: " + "; ".join(limits),
        fontsize=6,
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    body = build_receipt_body()
    _validate_receipt(body)
    draw_figure(body)

    # Refresh inventory now that ED10 PNG exists (10/10 Extended Data).
    body = build_receipt_body()
    write_json_strict(RECEIPT_PATH, body)

    root = _REPO_ROOT
    manifest = save_figure_manifest(
        figure_id="ed10",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed10_release_archive_receipt",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed10_release_archive_receipt.py",
            "claim_boundary": "release_archive_evidence_receipt_only",
            "release_action_taken": False,
            "tag_action_taken": False,
            "archive_action_taken": False,
            "publish_action_taken": False,
            "local_receipt_only": True,
            "repo_state": body["repo_state"],
            "package_version": body["package_version"],
            "evidence_inventory": body["evidence_inventory"],
            "checklist_hash": body["checklist_hash"],
            "output_inventory_hash": body["output_inventory_hash"],
            "wheel_sdist_status": body["wheel_sdist_status"],
            "tag_status": body["tag_status"],
            "archive_doi_status": body["archive_doi_status"],
            "mkdocs_strict_build": body["mkdocs_strict_build"],
            "remaining_limitations": body["remaining_limitations"],
            "receipt_json_path": str(RECEIPT_PATH.relative_to(root)),
            "receipt_json_sha256": sha256_file(RECEIPT_PATH),
            "truth_gates": truth_gates(),
            "physical_amplitude_claim_allowed": False,
        },
    )

    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: outputs/evidence/{FIGURE_PATH.stem}_manifest.json")
    print(f"wrote: {RECEIPT_PATH.relative_to(root)}")
    print(f"sha256: {sha256_file(FIGURE_PATH)}")
    print(f"figure_id: {manifest['figure_id']}")
    print(
        "inventory: "
        f"{body['evidence_inventory']['main_figures']} main, "
        f"{body['evidence_inventory']['extended_data']} ED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
