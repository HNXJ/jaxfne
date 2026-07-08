#!/usr/bin/env python3
"""Generate Extended Data ED9: failure modes and null controls panel.

Local receipt for objective null controls, JSON/gate failure modes, and
overinterpretation guards. Not empirical validation.

Outputs:
  figures/evidence/ed09_failure_modes_and_nulls.png
  outputs/evidence/ed09_failure_modes_and_nulls_manifest.json
  outputs/evidence/ed09_failure_modes_and_nulls_receipt.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

_REPO_ROOT = repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SOURCE_FILES = [
    "scripts/evidence_figures/ed09_failure_modes_and_nulls.py",
    "scripts/evidence_figures/_figure_common.py",
    "tests/test_objective_null_reproducibility_v0330.py",
    "tests/test_artifact_json_safety_v0330.py",
    "tests/test_public_docs_hygiene.py",
    "jaxfne/objectives.py",
    "jaxfne/validation.py",
]

TITLE = "Extended Data 9 - failure modes and null controls (local receipt)"

SCOPE_STATUS = (
    "Local failure/null control receipt; not empirical validation; "
    "mechanism_claim_allowed: false; physical_amplitude_calibrated: false"
)

NULL_CONTROLS = [
    "layer_shuffle_null",
    "band_label_shuffle_null",
    "uniform_gain_null",
    "no_field_projection_null",
    "phase_randomized_null",
    "source_polarity_flip_null",
]

FAILURE_MODES = [
    ("json_nan_rejection", "json.dumps allow_nan=False rejects NaN"),
    ("json_inf_rejection", "json.dumps allow_nan=False rejects Inf"),
    ("truth_gate_physical_amplitude", "default physical_amplitude_calibrated false"),
    ("proxy_field_status", "field_solver_status linear_solver"),
    ("mechanism_not_from_objective", "objective alone not mechanism proof (status)"),
]

PYTEST_TARGETS = [
    "tests/test_objective_null_reproducibility_v0330.py",
    "tests/test_artifact_json_safety_v0330.py",
    "tests/test_public_docs_hygiene.py",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed09_failure_modes_and_nulls.png"
RECEIPT_PATH = _dirs["outputs"] / "ed09_failure_modes_and_nulls_receipt.json"


def _runtime_receipt() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _readout() -> dict[str, np.ndarray]:
    return {
        "alpha_beta": np.array([0.2, 0.3, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]),
        "gamma": np.array([0.05, 0.03, 0.08, 0.15, 0.2, 0.18, 0.12, 0.08]),
    }


def _evaluate_null_controls() -> list[dict[str, Any]]:
    from jaxfne.objectives import (
        band_label_shuffle_null,
        layer_shuffle_null,
        no_field_projection_null,
        phase_randomized_null,
        source_polarity_flip_null,
        uniform_gain_null,
    )

    fns = {
        "layer_shuffle_null": layer_shuffle_null,
        "band_label_shuffle_null": band_label_shuffle_null,
        "uniform_gain_null": uniform_gain_null,
        "no_field_projection_null": no_field_projection_null,
        "phase_randomized_null": phase_randomized_null,
        "source_polarity_flip_null": source_polarity_flip_null,
    }
    rows: list[dict[str, Any]] = []
    readout = _readout()
    for name in NULL_CONTROLS:
        fn = fns[name]
        exc: Exception | None = None
        finite = False
        try:
            if name in {"band_label_shuffle_null", "source_polarity_flip_null"}:
                out_a = fn(readout)
                out_b = fn(readout)
                same_seed_ok = np.array_equal(
                    out_a["alpha_beta_shuffled"], out_b["alpha_beta_shuffled"]
                )
                status = "pass" if same_seed_ok else "fail"
            else:
                out_a = fn(readout, rng=np.random.default_rng(7))
                out_b = fn(readout, rng=np.random.default_rng(7))
                same_seed_ok = np.array_equal(
                    out_a["alpha_beta_shuffled"], out_b["alpha_beta_shuffled"]
                )
                out_c = fn(readout, rng=np.random.default_rng(99))
                diff_ok = not np.array_equal(
                    out_a["alpha_beta_shuffled"], out_c["alpha_beta_shuffled"]
                )
                status = "pass" if same_seed_ok and diff_ok else "fail"
            finite = bool(
                np.all(np.isfinite(out_a["alpha_beta_shuffled"]))
                and np.all(np.isfinite(out_a["gamma_shuffled"]))
            )
            if not finite:
                status = "fail"
        except Exception as caught:
            exc = caught
            status = "fail"
        rows.append(
            {
                "control_id": name,
                "category": "objective_null",
                "gate_outcome": status,
                "finite_outputs": finite,
                "validator": "ed09_inline_null_smoke",
                "error": str(exc)[:120] if exc else None,
            }
        )
    return rows


def _evaluate_failure_modes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nan_rejects = False
    inf_rejects = False
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
    except (ValueError, TypeError):
        nan_rejects = True
    try:
        json.dumps({"x": float("inf")}, allow_nan=False)
    except (ValueError, TypeError):
        inf_rejects = True

    gates = truth_gates()
    rows.append(
        {
            "control_id": "json_nan_rejection",
            "category": "failure_mode",
            "gate_outcome": "pass" if nan_rejects else "fail",
            "validator": "json.dumps allow_nan=False",
        }
    )
    rows.append(
        {
            "control_id": "json_inf_rejection",
            "category": "failure_mode",
            "gate_outcome": "pass" if inf_rejects else "fail",
            "validator": "json.dumps allow_nan=False",
        }
    )
    rows.append(
        {
            "control_id": "truth_gate_physical_amplitude",
            "category": "failure_mode",
            "gate_outcome": "pass" if not gates.get("physical_amplitude_calibrated") else "fail",
            "validator": "truth_gates()",
        }
    )
    rows.append(
        {
            "control_id": "proxy_field_status",
            "category": "failure_mode",
            "gate_outcome": "pass"
            if gates.get("field_solver_status") == "linear_solver"
            else "fail",
            "validator": "truth_gates()",
        }
    )
    rows.append(
        {
            "control_id": "mechanism_not_from_objective",
            "category": "interpretation_guard",
            "gate_outcome": "pass",
            "validator": "ed09_status_only",
            "note": "objective success is computational criterion only",
        }
    )
    return rows


def _run_pytest_file(rel_path: str) -> dict[str, Any]:
    env = {
        **dict(os.environ),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(repo_root()),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", rel_path, "-q", "--tb=line"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return {
        "control_id": rel_path,
        "category": "pytest_receipt",
        "gate_outcome": "pass" if result.returncode == 0 else "fail",
        "validator": rel_path,
        "exit_code": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-3:],
    }


def build_rows() -> list[dict[str, Any]]:
    rows = _evaluate_null_controls()
    rows.extend(_evaluate_failure_modes())
    for target in PYTEST_TARGETS:
        rows.append(_run_pytest_file(target))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows_total": len(rows),
        "pass_count": sum(1 for r in rows if r.get("gate_outcome") == "pass"),
        "fail_count": sum(1 for r in rows if r.get("gate_outcome") == "fail"),
        "null_controls": sum(1 for r in rows if r.get("category") == "objective_null"),
        "failure_modes": sum(1 for r in rows if r.get("category") == "failure_mode"),
        "pytest_receipts": sum(1 for r in rows if r.get("category") == "pytest_receipt"),
    }


def draw_figure(rows: list[dict], summary: dict, runtime: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.5, 9.55, TITLE, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(
        7.5,
        9.05,
        "local receipt; not mechanism proof; not empirical validation",
        ha="center",
        fontsize=8,
        family="monospace",
        color="#C62828",
    )

    headers = ["id", "category", "outcome", "validator"]
    col_x = [0.3, 5.0, 8.0, 9.2]
    y = 8.55
    for x, h in zip(col_x, headers):
        ax.text(x, y, h, fontsize=7, fontweight="bold", family="monospace")

    y = 8.15
    for row in rows:
        color = "#2E7D32" if row.get("gate_outcome") == "pass" else "#C62828"
        cid = str(row.get("control_id", ""))[:42]
        ax.text(col_x[0], y, cid, fontsize=5.8, family="monospace")
        ax.text(col_x[1], y, str(row.get("category", ""))[:18], fontsize=5.8, family="monospace")
        ax.text(
            col_x[2],
            y,
            str(row.get("gate_outcome", "")),
            fontsize=5.8,
            family="monospace",
            color=color,
        )
        ax.text(
            col_x[3],
            y,
            str(row.get("validator", ""))[:36],
            fontsize=5.5,
            family="monospace",
        )
        y -= 0.34
        if y < 1.2:
            break

    if len(rows) > 20:
        ax.text(0.3, y, f"... +{len(rows) - 20} rows in JSON receipt", fontsize=6)

    ax.text(
        0.3,
        0.55,
        SCOPE_STATUS,
        fontsize=7,
        family="monospace",
        color="#444444",
    )
    ax.text(
        0.3,
        0.25,
        (
            f"pass={summary['pass_count']}/{summary['rows_total']} "
            f"repo_sha={runtime['repo_sha'][:12]} "
            "physical_amplitude_calibrated: false"
        ),
        fontsize=6.5,
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def main() -> int:
    runtime = _runtime_receipt()
    rows = build_rows()
    summary = _summary(rows)

    failures = [r for r in rows if r.get("gate_outcome") == "fail"]
    if failures:
        names = ", ".join(str(r.get("control_id")) for r in failures[:6])
        raise RuntimeError(f"ED9 gate failures: {names}")

    receipt_body = {
        "schema_version": "jaxfne.evidence_ed09_failure_null_receipt.v0.1.0",
        "artifact": "ed09_failure_modes_and_nulls",
        "generated_at_utc": utc_now_iso(),
        "runtime_receipt": runtime,
        "scope_status": SCOPE_STATUS,
        "local_receipt_only": True,
        "empirical_validation_claim_allowed": False,
        "mechanism_claim_allowed": False,
        "physical_amplitude_calibrated": False,
        "summary": summary,
        "rows": rows,
        "truth_gates": truth_gates(),
        "claim_boundary": "failure_null_local_receipt_only",
    }
    write_json_strict(RECEIPT_PATH, receipt_body)

    draw_figure(rows, summary, runtime)

    root = repo_root()
    manifest = save_figure_manifest(
        figure_id="ed09",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed09_failure_modes_and_nulls",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed09_failure_modes_and_nulls.py",
            "claim_boundary": "failure_null_local_receipt_only",
            "local_receipt_only": True,
            "empirical_validation_claim_allowed": False,
            "mechanism_claim_allowed": False,
            "physical_amplitude_calibrated": False,
            "control_summary": summary,
            "control_rows": rows,
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
    print(f"pass: {summary['pass_count']}/{summary['rows_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
