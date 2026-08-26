#!/usr/bin/env python3
"""Generate Extended Data ED3: notebook execution receipt panel.

Receipt-driven atlas survey of tutorial notebooks. Does not execute the full
notebook stack unless ``--run-smoke`` is passed (controlled small subset).

Outputs:
  figures/evidence/ed03_notebook_execution_receipts.png
  outputs/evidence/ed03_notebook_execution_receipts_manifest.json
  outputs/evidence/ed03_notebook_execution_receipts.json
  outputs/evidence/notebook_execution_receipt.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
    save_matplotlib_figure,
    ensure_evidence_dirs,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)


SOURCE_FILES = [
    "scripts/evidence_figures/ed03_notebook_execution_receipts.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/audit_notebooks_and_assets.py",
    "tests/test_suite_no1_notebook_execution.py",
    "tests/test_suite_no4_notebook_execution.py",
]

TITLE = "Notebook execution receipts (receipt-driven structural scan)"

SCOPE_STATUS = (
    "Receipt-driven structural scan of tutorial notebooks; structural metadata from .ipynb files; "
    "optional smoke execution via --run-smoke only; not a full atlas execution claim; "
    "notebook_execution_completeness_claim_allowed remains false"
)

# Controlled smoke subset (short runtime notebooks only).
SMOKE_NOTEBOOKS = [
    "artifacts/tutorials/jaxfne_suite_no_2_evoked_l4_drive.ipynb",
    "artifacts/tutorials/jaxfne_v031_single_neuron.ipynb",
]

# Notebooks with dedicated slow pytest execution gates in tests/.
CI_EXECUTION_TESTS: dict[str, str] = {
    "artifacts/tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb": (
        "tests/test_suite_no1_notebook_execution.py"
    ),
    "artifacts/tutorials/jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb": (
        "tests/test_suite_no4_notebook_execution.py"
    ),
}

# Known external receipt locations (relative to repo root).
KNOWN_RECEIPT_PATHS: dict[str, list[str]] = {
    "artifacts/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb": [
        "artifacts/tutorials/suite_no3_v0310_execution_receipt.json",
        "outputs/suite_no3_v0310_execution_receipt.json",
    ],
    "artifacts/tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb": [
        "artifacts/tutorials/suite_no4_v0310_execution_receipt.json",
        "outputs/suite_no4_v0310_execution_receipt.json",
    ],
    "artifacts/tutorials/etudes/jaxfne_etude_no_1_base.ipynb": [
        "outputs/etude_no_1/cell_execution_receipt.json",
    ],
}

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed03_notebook_execution_receipts.png"
RECEIPT_JSON_PATH = _dirs["outputs"] / "ed03_notebook_execution_receipts.json"
AGGREGATE_RECEIPT_PATH = _dirs["outputs"] / "notebook_execution_receipt.json"


def _discover_notebooks(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in sorted(root.rglob("*.ipynb")):
        rel = str(path.relative_to(root))
        if "archive" in rel or "test_execution" in rel:
            continue
        notebooks.append(path)
    return notebooks


def _load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _structural_scan(nb_path: Path) -> dict[str, Any]:
    nb = _load_nb(nb_path)
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    error_count = 0
    has_jaxfne = False
    for cell in code_cells:
        source = "".join(cell.get("source", []))
        if "jaxfne" in source or "jtfne" in source:
            has_jaxfne = True
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_count += 1
    return {
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": sum(1 for c in cells if c.get("cell_type") == "markdown"),
        "committed_error_count": error_count,
        "has_jaxfne_import": has_jaxfne,
    }


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_prior_ed03_receipt(root: Path) -> dict[str, dict]:
    """Index prior ED3 notebook rows by path."""
    path = root / "outputs/evidence/ed03_notebook_execution_receipts.json"
    data = _read_json_if_exists(path)
    if not data:
        return {}
    rows = data.get("notebooks", [])
    return {row.get("notebook_path", ""): row for row in rows if row.get("notebook_path")}


def _load_aggregate_receipt(root: Path) -> dict[str, dict]:
    path = root / "outputs/evidence/notebook_execution_receipt.json"
    data = _read_json_if_exists(path)
    if not data:
        return {}
    rows = data.get("notebooks", [])
    return {row.get("notebook_path", ""): row for row in rows if row.get("notebook_path")}


def _find_external_receipt(root: Path, rel_notebook: str) -> tuple[str | None, dict | None]:
    candidates = list(KNOWN_RECEIPT_PATHS.get(rel_notebook, []))
    for rel in candidates:
        data = _read_json_if_exists(root / rel)
        if data is not None:
            return rel, data
    return None, None


def _status_from_receipt(data: dict) -> str | None:
    for key in ("execution_status", "status", "notebook_execution"):
        value = data.get(key)
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"pass", "nbclient_pass", "ok", "success"}:
                return "pass"
            if lowered in {"fail", "error", "failed"}:
                return "fail"
    if data.get("finite_outputs") is True and data.get("strict_json_pass") is True:
        return "pass"
    return None


def _mode_from_receipt(data: dict) -> str | None:
    mode = data.get("mode") or data.get("execution_mode")
    if isinstance(mode, str) and mode in {"smoke", "full", "not-run"}:
        return mode
    if data.get("smoke") is True:
        return "smoke"
    if data.get("notebook_execution") == "nbclient_pass":
        return "full"
    return None


@contextmanager
def _portable_kernel_context():
    import shutil

    import nbformat as _nbf
    from jupyter_client.kernelspec import KernelSpecManager

    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "JAXFNE ED3 Smoke Kernel",
        "language": "python",
    }
    temp_dir = tempfile.mkdtemp()
    kernel_name = "jaxfne_ed3_smoke_kernel"
    try:
        spec_path = Path(temp_dir) / "kernel.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        ksm = KernelSpecManager()
        ksm.install_kernel_spec(temp_dir, kernel_name, user=True, replace=True)
        yield kernel_name
    finally:
        try:
            KernelSpecManager().remove_kernel_spec(kernel_name)
        except Exception:
            pass
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def _execute_notebook_smoke(nb_path: Path, *, timeout_s: int = 300) -> dict[str, Any]:
    """Execute notebook with nbclient; return execution receipt fields."""
    nbformat = __import__("nbformat")
    nbclient = __import__("nbclient")
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    repo = repo_root()
    inject_source = (
        f"import sys\nsys.path.insert(0, {str(repo)!r})\n"
        f"import os\nos.environ.setdefault('TFNE_SMOKE', '1')\n"
    )
    inject_cell = nbformat.v4.new_code_cell(source=inject_source)
    inject_cell.metadata["tags"] = ["injected-by-ed03-smoke"]
    nb.cells.insert(1, inject_cell)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        with _portable_kernel_context() as kernel_name:
            client = NotebookClient(
                nb,
                timeout=timeout_s,
                kernel_name=kernel_name,
                resources={"metadata": {"path": str(tmp)}},
            )
            client.execute()

    error_count = 0
    error_samples: list[str] = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_count += 1
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                error_samples.append(f"cell {i}: {ename}: {evalue[:80]}")

    return {
        "execution_status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "error_samples": error_samples[:3],
        "executed_at_utc": utc_now_iso(),
        "verified_repo_sha": repo_sha(),
    }


def _build_notebook_row(
    nb_path: Path,
    *,
    prior_ed03: dict[str, dict],
    prior_aggregate: dict[str, dict],
    run_smoke: bool,
) -> dict[str, Any]:
    root = repo_root()
    rel = str(nb_path.relative_to(root))
    structural = _structural_scan(nb_path)

    row: dict[str, Any] = {
        "notebook_path": rel,
        "mode": "not-run",
        "execution_status": "not-run",
        "cell_count": structural["cell_count"],
        "code_cell_count": structural["code_cell_count"],
        "error_count": structural["committed_error_count"],
        "artifact_receipt_path": None,
        "last_verified_sha": None,
        "last_verified_at_utc": None,
        "ci_execution_test": CI_EXECUTION_TESTS.get(rel),
        "has_jaxfne_import": structural["has_jaxfne_import"],
        "source": "structural_scan",
    }

    receipt_rel, receipt_data = _find_external_receipt(root, rel)
    if receipt_data is not None and receipt_rel:
        row["artifact_receipt_path"] = receipt_rel
        resolved_mode = _mode_from_receipt(receipt_data) or "full"
        row["source"] = "full_executed" if resolved_mode == "full" else "prior_receipt"
        row["mode"] = resolved_mode
        row["execution_status"] = _status_from_receipt(receipt_data) or "receipt-on-file"
        row["last_verified_sha"] = receipt_data.get("repo_sha") or receipt_data.get("verified_repo_sha")
        row["last_verified_at_utc"] = receipt_data.get("generated_at_utc") or receipt_data.get(
            "executed_at_utc"
        )

    for cache in (prior_ed03, prior_aggregate):
        if rel in cache:
            cached = cache[rel]
            cached_source = cached.get("source")
            has_explicit_receipt = bool(cached.get("artifact_receipt_path"))
            if not run_smoke and (
                cached_source == "smoke_executed"
                or cached.get("mode") == "smoke"
                or not has_explicit_receipt
            ):
                continue
            if row["execution_status"] == "not-run" and cached.get("execution_status"):
                row["execution_status"] = cached["execution_status"]
            if row["mode"] == "not-run" and cached.get("mode"):
                row["mode"] = cached["mode"]
            if not row["artifact_receipt_path"] and cached.get("artifact_receipt_path"):
                row["artifact_receipt_path"] = cached["artifact_receipt_path"]
            if not row["last_verified_sha"] and cached.get("last_verified_sha"):
                row["last_verified_sha"] = cached["last_verified_sha"]
            if not row["last_verified_at_utc"] and cached.get("last_verified_at_utc"):
                row["last_verified_at_utc"] = cached["last_verified_at_utc"]
            if row["source"] == "structural_scan" and cached_source in {
                "prior_receipt",
                "full_executed",
            }:
                row["source"] = cached_source

    if run_smoke and rel in SMOKE_NOTEBOOKS:
        try:
            exec_result = _execute_notebook_smoke(nb_path)
            row["mode"] = "smoke"
            row["execution_status"] = exec_result["execution_status"]
            row["error_count"] = exec_result["error_count"]
            row["error_samples"] = exec_result.get("error_samples", [])
            row["last_verified_sha"] = exec_result["verified_repo_sha"]
            row["last_verified_at_utc"] = exec_result["executed_at_utc"]
            row["source"] = "smoke_executed"
            smoke_receipt_rel = f"outputs/evidence/smoke_receipts/{nb_path.stem}.json"
            row["artifact_receipt_path"] = smoke_receipt_rel
        except Exception as exc:
            row["mode"] = "smoke"
            row["execution_status"] = "fail"
            row["error_count"] = row.get("error_count", 0) + 1
            row["error_samples"] = [str(exc)[:120]]
            row["last_verified_sha"] = repo_sha()
            row["last_verified_at_utc"] = utc_now_iso()
            row["source"] = "smoke_executed"

    return row


def _runtime_receipt() -> dict:
    try:
        import jax

        jax_version = str(jax.__version__)
    except Exception:
        jax_version = "unknown"
    try:
        import jaxlib

        jaxlib_version = str(jaxlib.__version__)
    except Exception:
        jaxlib_version = "unknown"
    nbclient_available = True
    try:
        __import__("nbclient")
        __import__("nbformat")
    except ImportError:
        nbclient_available = False
    return {
        "jaxfne_version": __import__("jaxfne").__version__,
        "repo_sha": repo_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "nbclient_available": nbclient_available,
        "smoke_subset": list(SMOKE_NOTEBOOKS),
    }


def _summary_counts(rows: list[dict]) -> dict[str, int]:
    return {
        "notebook_total": len(rows),
        "mode_smoke": sum(1 for r in rows if r["mode"] == "smoke"),
        "mode_full": sum(1 for r in rows if r["mode"] == "full"),
        "mode_not_run": sum(1 for r in rows if r["mode"] == "not-run"),
        "status_pass": sum(1 for r in rows if r["execution_status"] == "pass"),
        "status_fail": sum(1 for r in rows if r["execution_status"] == "fail"),
        "status_not_run": sum(1 for r in rows if r["execution_status"] == "not-run"),
        "status_receipt_on_file": sum(1 for r in rows if r["execution_status"] == "receipt-on-file"),
        "with_artifact_receipt": sum(1 for r in rows if r.get("artifact_receipt_path")),
        "ci_execution_tests_available": sum(1 for r in rows if r.get("ci_execution_test")),
    }


def draw_figure(rows: list[dict], summary: dict, receipt: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 10.5), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(7.0, 10.05, TITLE, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(
        7.0,
        9.55,
        "import jaxfne as jtfne",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        color="#333333",
    )

    table_box = FancyBboxPatch(
        (0.45, 1.55),
        13.1,
        7.65,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#1A4A8A",
        facecolor="#F8FAFF",
    )
    ax.add_patch(table_box)
    ax.text(0.65, 8.95, "Panel A - receipt-driven structural scan", fontsize=9, fontweight="bold", va="top")

    headers = ["notebook", "mode", "status", "cells", "errs", "receipt"]
    col_x = [0.65, 5.55, 6.55, 7.55, 8.25, 8.85]
    y = 8.55
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    y = 8.15
    row_height = 0.34
    display_rows = rows[:20]
    for row in display_rows:
        name = Path(row["notebook_path"]).name
        if len(name) > 34:
            name = name[:31] + "..."
        status = row["execution_status"]
        color = "#2E7D32" if status == "pass" else "#C62828" if status == "fail" else "#555555"
        receipt_flag = "yes" if row.get("artifact_receipt_path") else "no"
        ax.text(col_x[0], y, name, fontsize=5.8, va="top", family="monospace")
        ax.text(col_x[1], y, row["mode"], fontsize=5.8, va="top", family="monospace")
        ax.text(col_x[2], y, status, fontsize=5.8, va="top", family="monospace", color=color)
        ax.text(col_x[3], y, str(row["cell_count"]), fontsize=5.8, va="top", family="monospace")
        ax.text(col_x[4], y, str(row["error_count"]), fontsize=5.8, va="top", family="monospace")
        ax.text(col_x[5], y, receipt_flag, fontsize=5.8, va="top", family="monospace")
        y -= row_height

    if len(rows) > len(display_rows):
        ax.text(0.65, y - 0.05, f"... +{len(rows) - len(display_rows)} more in JSON receipt", fontsize=6, va="top")

    receipt_box = FancyBboxPatch(
        (0.45, 0.35),
        13.1,
        1.0,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(receipt_box)
    ax.text(0.65, 1.15, "Panel B - scope / summary", fontsize=9, fontweight="bold", va="top")
    ax.text(0.75, 0.82, SCOPE_STATUS, fontsize=6.5, va="top", color="#333333")
    lines = [
        f"notebooks: {summary['notebook_total']} | pass: {summary['status_pass']} | fail: {summary['status_fail']} | not-run: {summary['status_not_run']}",
        f"mode smoke/full/not-run: {summary['mode_smoke']}/{summary['mode_full']}/{summary['mode_not_run']} | receipts on file: {summary['with_artifact_receipt']}",
        f"jaxfne: {receipt['jaxfne_version']}  repo_sha: {receipt['repo_sha'][:12]}  nbclient: {receipt['nbclient_available']}",
        "notebook_execution_completeness_claim_allowed: false",
    ]
    for i, line in enumerate(lines):
        ax.text(0.75, 0.55 - i * 0.18, line, fontsize=6.2, va="top", family="monospace", color="#444444")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def _write_smoke_receipts(rows: list[dict]) -> None:
    root = repo_root()
    smoke_dir = root / "outputs" / "evidence" / "smoke_receipts"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("source") != "smoke_executed":
            continue
        rel = row.get("artifact_receipt_path")
        if not rel:
            continue
        path = root / rel
        write_json_strict(
            path,
            {
                "schema_version": "jaxfne.evidence_notebook_smoke_receipt.v0.1.0",
                "notebook_path": row["notebook_path"],
                "mode": row["mode"],
                "execution_status": row["execution_status"],
                "error_count": row["error_count"],
                "error_samples": row.get("error_samples", []),
                "verified_repo_sha": row.get("last_verified_sha"),
                "executed_at_utc": row.get("last_verified_at_utc"),
                "truth_gates": truth_gates(),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ED3 notebook execution receipts.")
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Execute controlled smoke subset via nbclient (optional; off by default).",
    )
    args = parser.parse_args()

    if args.run_smoke and os.environ.get("SKIP_NOTEBOOK_EXECUTION", "0") == "1":
        raise RuntimeError("SKIP_NOTEBOOK_EXECUTION=1 blocks --run-smoke")

    root = repo_root()
    tutorials = root / "artifacts" / "tutorials"
    notebooks = _discover_notebooks(tutorials)
    prior_ed03 = _load_prior_ed03_receipt(root)
    prior_aggregate = _load_aggregate_receipt(root)

    rows = [
        _build_notebook_row(
            nb_path,
            prior_ed03=prior_ed03,
            prior_aggregate=prior_aggregate,
            run_smoke=args.run_smoke,
        )
        for nb_path in notebooks
    ]

    if args.run_smoke:
        _write_smoke_receipts(rows)

    receipt = _runtime_receipt()
    summary = _summary_counts(rows)
    # Full atlas execution is out of scope for the default ED3 generator path.
    completeness_allowed = False

    ed03_payload = {
        "schema_version": "jaxfne.evidence_ed03_notebook_receipt.v0.1.0",
        "generated_at_utc": utc_now_iso(),
        "runtime_receipt": receipt,
        "summary": summary,
        "notebooks": rows,
        "truth_gates": truth_gates(),
        "notebook_execution_completeness_claim_allowed": completeness_allowed,
        "smoke_executed_this_run": args.run_smoke,
    }
    write_json_strict(RECEIPT_JSON_PATH, ed03_payload)

    aggregate_payload = {
        "schema_version": "jaxfne.evidence_notebook_execution_receipt.v0.1.0",
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "summary": summary,
        "notebooks": [
            {
                "notebook_path": r["notebook_path"],
                "mode": r["mode"],
                "execution_status": r["execution_status"],
                "cell_count": r["cell_count"],
                "error_count": r["error_count"],
                "artifact_receipt_path": r.get("artifact_receipt_path"),
                "last_verified_sha": r.get("last_verified_sha"),
                "last_verified_at_utc": r.get("last_verified_at_utc"),
            }
            for r in rows
        ],
        "truth_gates": truth_gates(),
        "notebook_execution_completeness_claim_allowed": completeness_allowed,
    }
    write_json_strict(AGGREGATE_RECEIPT_PATH, aggregate_payload)

    draw_figure(rows, summary, receipt)

    manifest = save_figure_manifest(
        figure_id="ed03",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed03_notebook_execution_receipts.py",
            "claim_boundary": "receipt_driven_structural_scan",
            "summary": summary,
            "runtime_receipt": receipt,
            "receipt_json_path": str(RECEIPT_JSON_PATH.relative_to(root)),
            "receipt_json_sha256": sha256_file(RECEIPT_JSON_PATH),
            "aggregate_receipt_path": str(AGGREGATE_RECEIPT_PATH.relative_to(root)),
            "truth_gates": truth_gates(),
            "notebook_execution_completeness_claim_allowed": completeness_allowed,
            "smoke_executed_this_run": args.run_smoke,
        },
    )

    sha = sha256_file(FIGURE_PATH)
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {RECEIPT_JSON_PATH.relative_to(root)}")
    print(f"wrote: {AGGREGATE_RECEIPT_PATH.relative_to(root)}")
    print(f"wrote: outputs/evidence/{FIGURE_PATH.stem}_manifest.json")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"notebooks: {summary['notebook_total']}")
    print(f"pass/fail/not-run: {summary['status_pass']}/{summary['status_fail']}/{summary['status_not_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
