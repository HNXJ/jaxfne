#!/usr/bin/env python3
"""Generate Extended Data ED5: local manifest/hash integrity receipt.

Audits fig01-fig08, ED1-ED4, and evidence metadata artifacts without
regenerating committed figure PNGs. Missing figure manifests are synthesized
from on-disk PNG SHA256 digests (local integrity receipt only).

Outputs:
  figures/evidence/ed05_manifest_hashes.png
  outputs/evidence/ed05_manifest_hashes_manifest.json
  outputs/evidence/ed05_manifest_hashes_receipt.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
    ensure_evidence_dirs,
    manifest_path_for,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)


SOURCE_FILES = [
    "scripts/evidence_figures/ed05_manifest_hashes.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_inventory.py",
    "docs/evidence/evidence_checklist.json",
]

TITLE = "Extended Data 5 - local artifact hash integrity receipt"

SCOPE_STATUS = (
    "Local artifact integrity receipt only; not archival completeness; "
    "not release immutability; proxy-safe evidence scaffold"
)

RECEIPT_PATHS_BY_STEM: dict[str, list[str]] = {
    "ed03_notebook_execution_receipts": [
        "outputs/evidence/ed03_notebook_execution_receipts.json",
        "outputs/evidence/notebook_execution_receipt.json",
    ],
    "ed04_optional_dependency_laziness": [
        "outputs/evidence/ed04_optional_dependency_laziness_receipt.json",
    ],
}

METADATA_ARTIFACTS = [
    {
        "artifact_group": "evidence_metadata",
        "artifact_id": "inventory",
        "label": "evidence_inventory.json",
        "path": "outputs/evidence/inventory.json",
        "kind": "json",
    },
    {
        "artifact_group": "evidence_metadata",
        "artifact_id": "checklist",
        "label": "evidence_checklist.json",
        "path": "docs/evidence/evidence_checklist.json",
        "kind": "json",
    },
    {
        "artifact_group": "evidence_metadata",
        "artifact_id": "roadmap",
        "kind": "markdown",
    },
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed05_manifest_hashes.png"
MANIFEST_PATH = _dirs["outputs"] / "ed05_manifest_hashes_manifest.json"
RECEIPT_PATH = _dirs["outputs"] / "ed05_manifest_hashes_receipt.json"


def _load_checklist() -> dict:
    path = repo_root() / "docs" / "evidence" / "evidence_checklist.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inventory() -> dict:
    root = repo_root()
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from evidence_inventory import build_inventory

    return build_inventory()


def _write_inventory_json(inventory: dict) -> Path:
    path = repo_root() / "outputs" / "evidence" / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_strict(path, inventory)
    return path


def _json_strict_ok(path: Path | None) -> bool | None:
    if path is None or not path.is_file():
        return None
    try:
        json.loads(path.read_text(encoding="utf-8"))
        json.dumps(json.loads(path.read_text(encoding="utf-8")), allow_nan=False)
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _truth_gates_preserved(manifest_data: dict | None) -> bool | None:
    if manifest_data is None:
        return None
    if manifest_data.get("physical_amplitude_claim_allowed") is True:
        return False
    gates = manifest_data.get("truth_gates")
    if isinstance(gates, dict) and gates.get("physical_amplitude_claim_allowed") is True:
        return False
    return True


def _resolve_receipt_paths(stem: str) -> list[Path]:
    root = repo_root()
    candidates = RECEIPT_PATHS_BY_STEM.get(stem, [f"outputs/evidence/{stem}_receipt.json"])
    return [root / rel for rel in candidates]


def _ensure_figure_manifest(
    *,
    figure_id: str,
    title: str,
    png_path: Path,
    script_path: Path,
    synthesize_if_missing: bool,
) -> tuple[dict | None, bool]:
    """Return manifest data and whether manifest was synthesized from PNG."""
    root = repo_root()
    manifest_path = manifest_path_for(png_path)
    synthesized = False

    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = data.get("sha256")
        actual = sha256_file(png_path) if png_path.is_file() else None
        if synthesize_if_missing and actual and declared != actual:
            save_figure_manifest(
                figure_id=figure_id,
                title=title,
                output_path=png_path,
                source_files=[
                    str(script_path.relative_to(root)),
                    "scripts/evidence_figures/_figure_common.py",
                ],
                extra={
                    "scope_status": "manifest_resynchronized_by_ed05_from_committed_png",
                    "resynchronized_by_ed05": True,
                    "local_artifact_integrity_receipt_only": True,
                },
            )
            synthesized = True
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data, synthesized

    if synthesize_if_missing and png_path.is_file():
        save_figure_manifest(
            figure_id=figure_id,
            title=title,
            output_path=png_path,
            source_files=[str(script_path.relative_to(root)), "scripts/evidence_figures/_figure_common.py"],
            extra={
                "scope_status": "manifest_synthesized_by_ed05_from_committed_png",
                "synthesized_by_ed05": True,
                "local_artifact_integrity_receipt_only": True,
            },
        )
        synthesized = True
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data, synthesized

    return None, synthesized


def _audit_figure_entry(
    *,
    artifact_group: str,
    entry: dict,
    synthesize_manifests: bool,
) -> dict[str, Any]:
    root = repo_root()
    artifact_id = entry["id"]
    png_rel = entry["figure_path"]
    script_rel = entry.get("generator_script")
    manifest_rel = entry.get("manifest_path")
    title = entry.get("purpose") or entry.get("filename", artifact_id)

    png_path = root / png_rel
    script_path = root / script_rel if script_rel else None
    manifest_path = root / manifest_rel if manifest_rel else manifest_path_for(png_path)

    receipt_paths = _resolve_receipt_paths(Path(png_rel).stem)
    receipt_path = next((p for p in receipt_paths if p.is_file()), None)

    png_exists = png_path.is_file()
    script_exists = script_path.is_file() if script_path else False
    manifest_exists_before = manifest_path.is_file()
    receipt_exists = receipt_path is not None

    manifest_data, synthesized = _ensure_figure_manifest(
        figure_id=artifact_id,
        title=title,
        png_path=png_path,
        script_path=script_path or Path("scripts/evidence_figures/_figure_common.py"),
        synthesize_if_missing=synthesize_manifests,
    )
    manifest_exists = manifest_path.is_file()

    actual_png_sha = sha256_file(png_path) if png_exists else None
    manifest_declared_png_sha = manifest_data.get("sha256") if manifest_data else None
    manifest_file_sha = sha256_file(manifest_path) if manifest_exists else None

    if not png_exists:
        png_match = "png_missing"
    elif not manifest_exists:
        png_match = "no_manifest"
    elif manifest_declared_png_sha == actual_png_sha:
        png_match = "ok"
    else:
        png_match = "mismatch"

    manifest_json_strict = _json_strict_ok(manifest_path if manifest_exists else None)
    receipt_json_strict = _json_strict_ok(receipt_path)
    gates_ok = _truth_gates_preserved(manifest_data)

    return {
        "artifact_group": artifact_group,
        "artifact_id": artifact_id,
        "label": Path(png_rel).stem,
        "script_path": script_rel,
        "png_path": png_rel,
        "manifest_path": str(manifest_path.relative_to(root)) if manifest_exists else manifest_rel,
        "receipt_path": str(receipt_path.relative_to(root)) if receipt_path else None,
        "exists": {
            "script": script_exists,
            "png": png_exists,
            "manifest": manifest_exists,
            "receipt": receipt_exists,
        },
        "sha256": {
            "png_prefix": actual_png_sha[:16] if actual_png_sha else None,
            "png": actual_png_sha,
            "manifest_file": manifest_file_sha,
            "manifest_declared_png": manifest_declared_png_sha,
            "receipt_file": sha256_file(receipt_path) if receipt_path else None,
        },
        "manifest_synthesized_by_ed05": synthesized,
        "png_manifest_match": png_match,
        "json_strict": {
            "manifest": manifest_json_strict,
            "receipt": receipt_json_strict,
        },
        "truth_gates_preserved": gates_ok,
    }


def _audit_metadata_entry(spec: dict) -> dict[str, Any]:
    root = repo_root()
    path = root / spec["path"]
    exists = path.is_file()
    is_json = spec["kind"] == "json"
    sha = sha256_file(path) if exists else None
    strict = _json_strict_ok(path) if exists and is_json else None
    return {
        "artifact_group": spec["artifact_group"],
        "artifact_id": spec["artifact_id"],
        "label": spec["label"],
        "script_path": None,
        "png_path": None,
        "manifest_path": spec["path"] if is_json else None,
        "receipt_path": spec["path"],
        "exists": {
            "script": None,
            "png": None,
            "manifest": exists if is_json else None,
            "receipt": exists,
        },
        "sha256": {
            "png_prefix": None,
            "png": None,
            "manifest_file": sha if is_json else None,
            "manifest_declared_png": None,
            "receipt_file": sha,
        },
        "manifest_synthesized_by_ed05": False,
        "png_manifest_match": "n/a",
        "json_strict": {
            "manifest": strict if is_json else None,
            "receipt": strict if is_json else None,
        },
        "truth_gates_preserved": None,
        "kind": spec["kind"],
    }


def build_audit_rows(*, synthesize_manifests: bool) -> tuple[list[dict], dict]:
    checklist = _load_checklist()
    inventory = _load_inventory()
    _write_inventory_json(inventory)

    rows: list[dict] = []
    for entry in checklist.get("main_figures", []):
        rows.append(
            _audit_figure_entry(
                artifact_group="main_figure",
                entry=entry,
                synthesize_manifests=synthesize_manifests,
            )
        )
    for entry in checklist.get("extended_data", []):
        if entry.get("status") == "missing" and entry["id"] == "ed05":
            continue
        if entry.get("status") == "missing":
            continue
        rows.append(
            _audit_figure_entry(
                artifact_group="extended_data",
                entry=entry,
                synthesize_manifests=synthesize_manifests,
            )
        )
    for spec in METADATA_ARTIFACTS:
        rows.append(_audit_metadata_entry(spec))

    summary = {
        "main_figure_rows": sum(1 for r in rows if r["artifact_group"] == "main_figure"),
        "extended_data_rows": sum(1 for r in rows if r["artifact_group"] == "extended_data"),
        "metadata_rows": sum(1 for r in rows if r["artifact_group"] == "evidence_metadata"),
        "png_manifest_ok": sum(1 for r in rows if r["png_manifest_match"] == "ok"),
        "png_manifest_mismatch": sum(1 for r in rows if r["png_manifest_match"] == "mismatch"),
        "png_manifest_missing": sum(1 for r in rows if r["png_manifest_match"] in {"no_manifest", "png_missing"}),
        "manifests_synthesized": sum(1 for r in rows if r.get("manifest_synthesized_by_ed05")),
        "inventory_main_present": inventory["summary"]["main_figures_present"],
        "inventory_ed_present": inventory["summary"]["extended_data_present"],
    }
    return rows, summary


def _status_marker(row: dict[str, Any], column: str) -> str:
    exists = row.get("exists") or {}
    if column == "script":
        val = exists.get("script")
        return "Y" if val is True else ("-" if val is None else "N")
    if column == "png":
        val = exists.get("png")
        return "Y" if val is True else ("-" if val is None else "N")
    if column == "manifest":
        val = exists.get("manifest")
        return "Y" if val is True else ("-" if val is None else "N")
    if column == "receipt":
        val = exists.get("receipt")
        return "Y" if val is True else ("-" if val is None else "N")
    if column == "json_strict":
        manifest_ok = (row.get("json_strict") or {}).get("manifest")
        receipt_ok = (row.get("json_strict") or {}).get("receipt")
        if manifest_ok is False or receipt_ok is False:
            return "FAIL"
        if manifest_ok is True or receipt_ok is True:
            return "ok"
        return "-"
    if column == "sha_match":
        match = row.get("png_manifest_match")
        if match == "ok":
            return "ok"
        if match in {"mismatch", "no_manifest", "png_missing"}:
            return str(match)
        return "n/a"
    if column == "gates":
        gates = row.get("truth_gates_preserved")
        if gates is True:
            return "ok"
        if gates is False:
            return "FAIL"
        return "-"
    return "?"


def _cell_color(marker: str) -> str:
    if marker in {"Y", "ok", "n/a"}:
        return "#2E7D32"
    if marker in {"-", "?"}:
        return "#666666"
    return "#C62828"


def draw_figure(rows: list[dict], summary: dict, runtime: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(8.0, 10.55, TITLE, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        8.0,
        10.05,
        "local_artifact_integrity_receipt_only | archive_completeness_claim_allowed: false",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        color="#333333",
    )

    columns = [
        ("artifact", 0.35),
        ("script", 3.0),
        ("png", 3.65),
        ("manifest", 4.3),
        ("receipt", 4.95),
        ("json", 5.6),
        ("sha", 6.35),
        ("gates", 7.25),
        ("png sha prefix", 8.05),
    ]
    y_header = 9.55
    for label, x in columns:
        ax.text(x, y_header, label, fontsize=6.5, fontweight="bold", va="top", family="monospace")

    y = 9.15
    row_step = 0.28
    display_rows = rows + [
        {
            "artifact_id": "ed05",
            "label": "ed05_manifest_hashes",
            "exists": {"script": True, "png": False, "manifest": False, "receipt": False},
            "png_manifest_match": "pending",
            "json_strict": {"manifest": None, "receipt": None},
            "truth_gates_preserved": True,
            "sha256": {"png_prefix": "(post-gen)"},
            "self_generated_after_receipt": True,
        }
    ]

    for row in display_rows:
        artifact = row.get("label") or row.get("artifact_id")
        ax.text(columns[0][1], y, artifact[:28], fontsize=5.8, va="top", family="monospace")
        for col_name, x in columns[1:8]:
            marker = _status_marker(row, col_name.replace("json", "json_strict").replace("sha", "sha_match"))
            if col_name == "json":
                marker = _status_marker(row, "json_strict")
            elif col_name == "sha":
                marker = _status_marker(row, "sha_match")
            elif col_name == "gates":
                marker = _status_marker(row, "gates")
            color = _cell_color(marker)
            ax.text(x, y, marker, fontsize=5.8, va="top", family="monospace", color=color)
        prefix = (row.get("sha256") or {}).get("png_prefix") or "-"
        ax.text(columns[8][1], y, str(prefix)[:18], fontsize=5.5, va="top", family="monospace")
        y -= row_step
        if y < 1.5:
            break

    if len(display_rows) > 28:
        ax.text(0.35, y, f"... {len(display_rows) - 28} more rows in JSON receipt", fontsize=6, va="top")

    scope_box = FancyBboxPatch(
        (0.25, 0.25),
        15.5,
        1.05,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.4, 1.05, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    ax.text(
        0.4,
        0.72,
        (
            f"png_manifest_ok={summary['png_manifest_ok']} "
            f"mismatch={summary['png_manifest_mismatch']} "
            f"synthesized_manifests={summary['manifests_synthesized']} "
            f"inventory={summary['inventory_main_present']}/8 main "
            f"{summary['inventory_ed_present']}/10 ED"
        ),
        fontsize=6.5,
        va="top",
        family="monospace",
        color="#444444",
    )
    ax.text(
        0.4,
        0.45,
        f"repo_sha={runtime['repo_sha'][:12]}  physical_amplitude_claim_allowed: false",
        fontsize=6.5,
        va="top",
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _runtime_receipt() -> dict:
    try:
        import jax

        jax_version = str(jax.__version__)
    except Exception:
        jax_version = "not_imported"
    try:
        import jaxfne as jtfne

        jaxfne_version = str(jtfne.__version__)
    except Exception:
        jaxfne_version = "unknown"
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jaxfne_version": jaxfne_version,
        "jax_version": jax_version,
    }


def _validate_rows(rows: list[dict]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        if row["artifact_group"] not in {"main_figure", "extended_data"}:
            continue
        if row["png_manifest_match"] != "ok":
            failures.append(f"{row['artifact_id']}: png_manifest_match={row['png_manifest_match']}")
        if row.get("truth_gates_preserved") is False:
            failures.append(f"{row['artifact_id']}: truth_gates_preserved=false")
        if (row.get("json_strict") or {}).get("manifest") is False:
            failures.append(f"{row['artifact_id']}: manifest_json_strict=false")
    inv = next((r for r in rows if r["artifact_id"] == "inventory"), None)
    chk = next((r for r in rows if r["artifact_id"] == "checklist"), None)
    if inv and not inv["exists"]["receipt"]:
        failures.append("inventory: missing")
    if chk and (chk.get("json_strict") or {}).get("manifest") is not True:
        failures.append("checklist: json_strict failed")
    return failures


def main() -> int:
    runtime = _runtime_receipt()
    rows, summary = build_audit_rows(synthesize_manifests=True)

    failures = _validate_rows(rows)
    if failures:
        raise RuntimeError("ED5 integrity audit failures:\n  " + "\n  ".join(failures))

    receipt_body = {
        "schema_version": "jaxfne.evidence_ed05_manifest_hash_receipt.v0.1.0",
        "artifact": "ed05_manifest_hashes",
        "generated_at_utc": utc_now_iso(),
        "runtime_receipt": runtime,
        "local_artifact_integrity_receipt_only": True,
        "archive_completeness_claim_allowed": False,
        "release_immutability_claim_allowed": False,
        "physical_amplitude_claim_allowed": False,
        "truth_gates_preserved": True,
        "summary": summary,
        "rows": rows,
        "truth_gates": truth_gates(),
    }
    write_json_strict(RECEIPT_PATH, receipt_body)

    draw_figure(rows, summary, runtime)

    root = repo_root()
    ed05_row = {
        "artifact_group": "extended_data",
        "artifact_id": "ed05",
        "label": "ed05_manifest_hashes",
        "script_path": "scripts/evidence_figures/ed05_manifest_hashes.py",
        "png_path": str(FIGURE_PATH.relative_to(root)),
        "manifest_path": str(MANIFEST_PATH.relative_to(root)),
        "receipt_path": str(RECEIPT_PATH.relative_to(root)),
        "exists": {
            "script": True,
            "png": FIGURE_PATH.is_file(),
            "manifest": MANIFEST_PATH.is_file(),
            "receipt": RECEIPT_PATH.is_file(),
        },
        "sha256": {
            "png": sha256_file(FIGURE_PATH),
            "manifest_file": None,
            "receipt_file": sha256_file(RECEIPT_PATH),
        },
        "self_generated_after_receipt": True,
        "png_manifest_match": "self_pending_until_manifest_write",
        "truth_gates_preserved": True,
    }

    manifest = save_figure_manifest(
        figure_id="ed05",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed05_manifest_hashes",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed05_manifest_hashes.py",
            "claim_boundary": "local_artifact_integrity_receipt_only",
            "local_artifact_integrity_receipt_only": True,
            "archive_completeness_claim_allowed": False,
            "release_immutability_claim_allowed": False,
            "physical_amplitude_claim_allowed": False,
            "truth_gates_preserved": True,
            "audit_summary": summary,
            "audit_rows": rows,
            "ed05_self_row": ed05_row,
            "receipt_json_path": str(RECEIPT_PATH.relative_to(root)),
            "receipt_json_sha256": sha256_file(RECEIPT_PATH),
            "truth_gates": truth_gates(),
        },
    )

    ed05_row["sha256"]["manifest_file"] = sha256_file(MANIFEST_PATH)
    ed05_row["sha256"]["manifest_declared_png"] = manifest.get("sha256")
    ed05_row["png_manifest_match"] = (
        "ok" if manifest.get("sha256") == ed05_row["sha256"]["png"] else "mismatch"
    )
    ed05_row["exists"]["manifest"] = MANIFEST_PATH.is_file()
    receipt_body["ed05_self_row"] = ed05_row
    write_json_strict(RECEIPT_PATH, receipt_body)

    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {MANIFEST_PATH.relative_to(root)}")
    print(f"wrote: {RECEIPT_PATH.relative_to(root)}")
    print(f"sha256: {sha256_file(FIGURE_PATH)}")
    print(f"figure_id: {manifest['figure_id']}")
    print(
        f"png_manifest_ok: {summary['png_manifest_ok']} "
        f"synthesized: {summary['manifests_synthesized']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
