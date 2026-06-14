#!/usr/bin/env python3
"""Generate Extended Data ED8: tutorial atlas coverage matrix.

Receipt-driven atlas survey of tutorial notebooks and docs/tutorial pages.
Does not claim full notebook execution unless artifact receipts exist.

Outputs:
  figures/evidence/ed08_tutorial_atlas_coverage.png
  outputs/evidence/ed08_tutorial_atlas_coverage_manifest.json
  outputs/evidence/ed08_tutorial_atlas_coverage_receipt.json
"""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
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
    "scripts/evidence_figures/ed08_tutorial_atlas_coverage.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_figures/ed03_notebook_execution_receipts.py",
    "scripts/audit_notebooks_and_assets.py",
    "docs/tutorials/tutorial_outputs.md",
]

TITLE = "Extended Data 8 - tutorial atlas coverage matrix"

SCOPE_STATUS = (
    "Tutorial atlas coverage receipt; structural scan plus on-file execution receipts; "
    "notebook_execution_completeness_claim_allowed: false; "
    "does not claim all tutorials executed unless receipts exist"
)

KNOWN_RECEIPT_PATHS: dict[str, list[str]] = {
    "tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb": [
        "tutorials/suite_no3_v0310_execution_receipt.json",
        "outputs/suite_no3_v0310_execution_receipt.json",
    ],
    "tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb": [
        "tutorials/suite_no4_v0310_execution_receipt.json",
        "outputs/suite_no4_v0310_execution_receipt.json",
    ],
    "tutorials/etudes/jaxfne_etude_no_1_base.ipynb": [
        "outputs/etude_no_1/cell_execution_receipt.json",
    ],
}

CI_EXECUTION_TESTS: dict[str, str] = {
    "tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb": (
        "tests/test_suite_no1_notebook_execution.py"
    ),
    "tutorials/jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb": (
        "tests/test_suite_no4_notebook_execution.py"
    ),
}

MATRIX_COLUMNS = ["import", "receipt", "ci_test", "artifacts", "gates"]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed08_tutorial_atlas_coverage.png"
RECEIPT_PATH = _dirs["outputs"] / "ed08_tutorial_atlas_coverage_receipt.json"


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _discover_notebooks(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in sorted((root / "tutorials").rglob("*.ipynb")):
        rel = str(path.relative_to(root))
        if any(skip in rel for skip in ("archive", "test_execution", ".legacy")):
            continue
        notebooks.append(path)
    return notebooks


def _discover_tutorial_docs(root: Path) -> list[Path]:
    docs_dir = root / "docs" / "tutorials"
    if not docs_dir.is_dir():
        return []
    return sorted(docs_dir.glob("*.md"))


def _scenario_id_from_path(rel_path: str) -> str:
    stem = Path(rel_path).stem
    return re.sub(r"^jaxfne_", "", stem)


def _import_style_notebook(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    has_jtfne = False
    has_jaxfne = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if re.search(r"import\s+jaxfne\s+as\s+jtfne", source):
            has_jtfne = True
        if re.search(r"import\s+jaxfne\b", source) and "as jtfne" not in source:
            has_jaxfne = True
    if has_jtfne and has_jaxfne:
        return "mixed"
    if has_jtfne:
        return "jtfne"
    if has_jaxfne:
        return "jaxfne"
    return "none"


def _import_style_markdown(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    has_jtfne = "import jaxfne as jtfne" in text
    has_jaxfne = bool(re.search(r"import\s+jaxfne\b", text)) and "as jtfne" not in text
    if has_jtfne:
        return "jtfne"
    if has_jaxfne:
        return "jaxfne"
    return "none"


def _notebook_artifact_coverage(root: Path, rel: str, nb_path: Path) -> dict[str, bool]:
    stem = nb_path.stem
    outputs_dir = root / "outputs"
    receipt_candidates = list(KNOWN_RECEIPT_PATHS.get(rel, []))
    receipt_candidates.append(f"outputs/evidence/smoke_receipts/{stem}.json")
    has_receipt = any((root / p).is_file() for p in receipt_candidates)
    has_output_bundle = any(outputs_dir.rglob(f"*{stem}*manifest.json")) if outputs_dir.is_dir() else False
    has_png = any((root / "tutorials").rglob(f"{stem}*.png")) or any(
        (root / "figures").rglob(f"{stem}*.png")
    )
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    has_colab = any(
        "colab.research.google.com" in "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "markdown"
    )
    return {
        "has_execution_receipt": has_receipt,
        "has_output_manifest_candidate": has_output_bundle,
        "has_figure_artifact": has_png,
        "has_colab_link": has_colab,
    }


def _doc_artifact_coverage(root: Path, md_path: Path) -> dict[str, bool]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    return {
        "has_execution_receipt": False,
        "has_output_manifest_candidate": "manifest.json" in text,
        "has_figure_artifact": ".png" in text or "figures/" in text,
        "has_colab_link": "colab.research.google.com" in text,
    }


def _execution_receipt_status(root: Path, rel: str) -> str:
    for candidate in KNOWN_RECEIPT_PATHS.get(rel, []):
        data = _read_json_if_exists(root / candidate)
        if data is not None:
            status = data.get("execution_status") or data.get("status")
            if isinstance(status, str):
                return status.lower()
            return "receipt-on-file"
    ed03 = _read_json_if_exists(root / "outputs/evidence/ed03_notebook_execution_receipts.json")
    if ed03:
        for row in ed03.get("notebooks", []):
            if row.get("notebook_path") == rel:
                return str(row.get("execution_status", "not-run"))
    ci_test = CI_EXECUTION_TESTS.get(rel)
    if ci_test and (root / ci_test).is_file():
        return "ci-test-available"
    return "not-run"


def _truth_gates_row() -> dict[str, Any]:
    gates = truth_gates()
    return {
        **gates,
        "physical_amplitude_claim_allowed": False,
        "notebook_execution_completeness_claim_allowed": False,
    }


def _build_notebook_row(root: Path, nb_path: Path) -> dict[str, Any]:
    rel = str(nb_path.relative_to(root))
    artifacts = _notebook_artifact_coverage(root, rel, nb_path)
    return {
        "atlas_kind": "notebook",
        "path": rel,
        "scenario_id": _scenario_id_from_path(rel),
        "import_style": _import_style_notebook(nb_path),
        "artifact_coverage": artifacts,
        "execution_receipt_status": _execution_receipt_status(root, rel),
        "ci_execution_test": CI_EXECUTION_TESTS.get(rel),
        "truth_gates": _truth_gates_row(),
    }


def _build_doc_row(root: Path, md_path: Path) -> dict[str, Any]:
    rel = str(md_path.relative_to(root))
    artifacts = _doc_artifact_coverage(root, md_path)
    return {
        "atlas_kind": "tutorial_doc",
        "path": rel,
        "scenario_id": _scenario_id_from_path(rel),
        "import_style": _import_style_markdown(md_path),
        "artifact_coverage": artifacts,
        "execution_receipt_status": "doc-only",
        "ci_execution_test": None,
        "truth_gates": _truth_gates_row(),
    }


def build_atlas_rows() -> list[dict[str, Any]]:
    root = repo_root()
    rows = [_build_notebook_row(root, p) for p in _discover_notebooks(root)]
    rows.extend(_build_doc_row(root, p) for p in _discover_tutorial_docs(root))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    notebooks = [r for r in rows if r["atlas_kind"] == "notebook"]
    docs = [r for r in rows if r["atlas_kind"] == "tutorial_doc"]
    return {
        "atlas_total": len(rows),
        "notebook_total": len(notebooks),
        "tutorial_doc_total": len(docs),
        "import_jtfne": sum(1 for r in rows if r["import_style"] == "jtfne"),
        "with_execution_receipt": sum(
            1 for r in notebooks if r["artifact_coverage"]["has_execution_receipt"]
        ),
        "with_ci_test": sum(1 for r in notebooks if r.get("ci_execution_test")),
        "receipt_status_not_run": sum(
            1 for r in notebooks if r["execution_receipt_status"] in {"not-run", "doc-only"}
        ),
        "with_colab_link": sum(1 for r in rows if r["artifact_coverage"]["has_colab_link"]),
    }


def _matrix_marker(row: dict[str, Any], column: str) -> str:
    if column == "import":
        return row["import_style"][:6]
    if column == "receipt":
        status = row["execution_receipt_status"]
        if row["artifact_coverage"]["has_execution_receipt"]:
            return "yes"
        return status[:8]
    if column == "ci_test":
        return "Y" if row.get("ci_execution_test") else "-"
    if column == "artifacts":
        art = row["artifact_coverage"]
        score = sum(
            1
            for k in ("has_output_manifest_candidate", "has_figure_artifact", "has_colab_link")
            if art.get(k)
        )
        return f"{score}/3"
    if column == "gates":
        return "ok" if not row["truth_gates"].get("physical_amplitude_claim_allowed") else "FAIL"
    return "?"


def draw_figure(rows: list[dict], summary: dict, runtime: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=150)

    # Panel A - summary bars
    ax = axes[0]
    labels = [
        "notebooks",
        "docs",
        "jtfne import",
        "receipts",
        "ci tests",
        "colab",
    ]
    values = [
        summary["notebook_total"],
        summary["tutorial_doc_total"],
        summary["import_jtfne"],
        summary["with_execution_receipt"],
        summary["with_ci_test"],
        summary["with_colab_link"],
    ]
    ax.barh(labels, values, color="#1565C0")
    ax.set_title("Panel A - atlas coverage summary", fontsize=10, fontweight="bold")
    ax.set_xlabel("count")
    ax.grid(True, axis="x", alpha=0.25)

    # Panel B - notebook table (compact)
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Panel B - notebook atlas rows (sample)", fontsize=10, fontweight="bold", pad=10)
    headers = ["scenario", *MATRIX_COLUMNS]
    col_x = [0.02, 0.28, 0.40, 0.52, 0.64, 0.74, 0.84]
    y = 0.92
    for x, h in zip(col_x, headers):
        ax.text(x, y, h, fontsize=7, fontweight="bold", family="monospace", transform=ax.transAxes)
    y = 0.86
    notebook_rows = [r for r in rows if r["atlas_kind"] == "notebook"][:18]
    for row in notebook_rows:
        ax.text(
            col_x[0],
            y,
            row["scenario_id"][:22],
            fontsize=6,
            family="monospace",
            transform=ax.transAxes,
        )
        for idx, col in enumerate(MATRIX_COLUMNS, start=1):
            ax.text(
                col_x[idx],
                y,
                _matrix_marker(row, col)[:8],
                fontsize=6,
                family="monospace",
                transform=ax.transAxes,
            )
        y -= 0.038
    if summary["notebook_total"] > len(notebook_rows):
        ax.text(
            0.02,
            y,
            f"... +{summary['notebook_total'] - len(notebook_rows)} notebooks in JSON receipt",
            fontsize=6,
            transform=ax.transAxes,
        )

    fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "atlas coverage receipt; not a full execution claim",
        ha="center",
        fontsize=9,
        family="monospace",
        color="#C62828",
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.01,
        SCOPE_STATUS,
        fontsize=7,
        family="monospace",
        color="#444444",
    )
    fig.text(
        0.02,
        -0.04,
        f"repo_sha={runtime['repo_sha'][:12]}  atlas_total={summary['atlas_total']}",
        fontsize=6.5,
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _runtime_receipt() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def main() -> int:
    runtime = _runtime_receipt()
    rows = build_atlas_rows()
    if not rows:
        raise RuntimeError("No tutorial atlas rows discovered")

    summary = _summary(rows)
    if summary["notebook_total"] == 0:
        raise RuntimeError("No tutorial notebooks discovered")

    receipt_body = {
        "schema_version": "jaxfne.evidence_ed08_tutorial_atlas_receipt.v0.1.0",
        "artifact": "ed08_tutorial_atlas_coverage",
        "generated_at_utc": utc_now_iso(),
        "runtime_receipt": runtime,
        "scope_status": SCOPE_STATUS,
        "atlas_rows": rows,
        "summary": summary,
        "notebook_execution_completeness_claim_allowed": False,
        "physical_amplitude_claim_allowed": False,
        "performance_claims_allowed": False,
        "truth_gates": truth_gates(),
        "claim_boundary": "tutorial_atlas_coverage_receipt_only",
    }
    write_json_strict(RECEIPT_PATH, receipt_body)

    draw_figure(rows, summary, runtime)

    root = repo_root()
    manifest = save_figure_manifest(
        figure_id="ed08",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed08_tutorial_atlas_coverage",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed08_tutorial_atlas_coverage.py",
            "claim_boundary": "tutorial_atlas_coverage_receipt_only",
            "notebook_execution_completeness_claim_allowed": False,
            "physical_amplitude_claim_allowed": False,
            "performance_claims_allowed": False,
            "atlas_summary": summary,
            "atlas_rows": rows,
            "receipt_json_path": str(RECEIPT_PATH.relative_to(root)),
            "receipt_json_sha256": sha256_file(RECEIPT_PATH),
            "truth_gates": truth_gates(),
        },
    )

    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: outputs/evidence/{FIGURE_PATH.stem}_manifest.json")
    print(f"wrote: {RECEIPT_PATH.relative_to(root)}")
    print(f"sha256: {sha256_file(FIGURE_PATH)}")
    print(f"figure_id: {manifest['figure_id']}")
    print(
        f"atlas: notebooks={summary['notebook_total']} docs={summary['tutorial_doc_total']} "
        f"receipts={summary['with_execution_receipt']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
