#!/usr/bin/env python3
"""Generate Extended Data ED7: probe/readout operator contract matrix.

Evidence-local contract matrix for eight proxy readout families. Separates
operator status, field solver posture, calibration, JSON strictness, and truth
gates without claiming validated EEG/MEG or physical amplitude.

Outputs:
  figures/evidence/ed07_probe_operator_contracts.png
  outputs/evidence/ed07_probe_operator_contracts_manifest.json
  outputs/evidence/ed07_probe_operator_contracts_receipt.json
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import jaxfne as jtfne
from jaxfne.fields import (
    csd_proxy_probe,
    eeg_proxy_probe,
    emm_proxy_probe,
    lfp_proxy_probe,
    meg_proxy_probe,
    source_probe,
    spk_probe,
    vm_probe,
)
from jaxfne.io import json_safe

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
    "scripts/evidence_figures/ed07_probe_operator_contracts.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_figures/fig06_readout_family_panel.py",
    "jaxfne/fields/probes.py",
    "tests/test_probe_operators_v021.py",
]

TITLE = "Extended Data 7 - probe/readout operator contract matrix"

SCOPE_STATUS = (
    "Simulated/proxy readout operator contracts only; proxy_readout; "
    "physical_amplitude_calibrated: false; no real EEG/MEG or mechanism claims"
)

READOUT_ORDER = [
    "spk",
    "vm",
    "source",
    "lfp_proxy",
    "csd_proxy",
    "eeg_proxy",
    "meg_proxy",
    "emm_proxy",
]

MATRIX_COLUMNS = [
    "operator",
    "field",
    "calibration",
    "json",
    "finite",
    "gates",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed07_probe_operator_contracts.png"
RECEIPT_PATH = _dirs["outputs"] / "ed07_probe_operator_contracts_receipt.json"


def _two_neuron_config() -> jtfne.Configuration:
    return (
        jtfne.configuration()
        .network(
            name="two_neuron_ei",
            kind="coupled_neurons",
            n=2,
            cell_types={"E": 0.5, "I": 0.5},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(
            name="multimodal_two_neuron_ei",
            modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"],
        )
    )


def collect_readouts(signals) -> dict[str, object]:
    readouts: dict[str, object] = {
        "spk": spk_probe(signals.spikes),
        "vm": vm_probe(signals.V_m),
    }
    if signals.sources is not None:
        readouts["source"] = source_probe(signals.sources[0])
    if signals.field is not None:
        readouts["lfp_proxy"] = lfp_proxy_probe(signals.field.lfp_proxy)
        readouts["csd_proxy"] = csd_proxy_probe(signals.field.csd_proxy)
        field_signal = signals.field.lfp_proxy
    else:
        field_signal = signals.V_m
    readouts["eeg_proxy"] = eeg_proxy_probe(field_signal)
    readouts["meg_proxy"] = meg_proxy_probe(field_signal)
    readouts["emm_proxy"] = emm_proxy_probe(field_signal)
    return readouts


def _finite_array(data) -> bool:
    if data is None or not hasattr(data, "shape"):
        return True
    return bool(jnp.all(jnp.isfinite(jnp.asarray(data))))


def _json_strict_ok(report: dict) -> bool:
    try:
        json.dumps(json_safe(report), allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _contract_row(name: str, readout) -> dict[str, Any]:
    report = dict(readout.report)
    return {
        "family": name,
        "method": report.get("method"),
        "operator_status": report.get("operator_status"),
        "field_solver_status": report.get("field_solver_status"),
        "field_claim_level": report.get("field_claim_level"),
        "calibration_status": report.get("calibration_status"),
        "source_calibration_status": report.get("source_calibration_status"),
        "physical_amplitude_calibrated": report.get("physical_amplitude_calibrated"),
        "shape": tuple(readout.data.shape) if hasattr(readout.data, "shape") else None,
        "finite_outputs": _finite_array(readout.data),
        "json_strict": _json_strict_ok(report),
        "truth_gates_preserved": report.get("physical_amplitude_calibrated") is not True,
        "report": report,
    }


def _matrix_cell(row: dict[str, Any], column: str) -> str:
    if column == "operator":
        return str(row.get("operator_status") or "?")[:14]
    if column == "field":
        return str(row.get("field_solver_status") or "?")[:14]
    if column == "calibration":
        return str(row.get("calibration_status") or "?")[:14]
    if column == "json":
        return "ok" if row.get("json_strict") else "FAIL"
    if column == "finite":
        return "ok" if row.get("finite_outputs") else "FAIL"
    if column == "gates":
        return "ok" if row.get("truth_gates_preserved") else "FAIL"
    return "?"


def _cell_color(marker: str) -> str:
    if marker in {"ok", "simulated_proxy", "laminar_proxy_", "uncalibrated_p"}:
        return "#2E7D32"
    if marker.startswith("?"):
        return "#666666"
    if marker == "FAIL":
        return "#C62828"
    return "#333333"


def draw_figure(rows: list[dict[str, Any]], runtime: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.5, 8.55, TITLE, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        7.5,
        8.05,
        "import jaxfne as jtfne  |  two-neuron E/I contract matrix",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        color="#333333",
    )

    col_x = [0.4, 2.2, 4.0, 5.8, 7.2, 8.6, 10.0]
    headers = ["family"] + MATRIX_COLUMNS
    y = 7.55
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    y = 7.15
    row_step = 0.62
    for row in rows:
        ax.text(col_x[0], y, row["family"], fontsize=6.5, va="top", family="monospace")
        for idx, column in enumerate(MATRIX_COLUMNS, start=1):
            marker = _matrix_cell(row, column)
            color = "#C62828" if marker == "FAIL" else "#2E7D32" if marker == "ok" else "#333333"
            ax.text(
                col_x[idx],
                y,
                marker[:16],
                fontsize=6,
                va="top",
                family="monospace",
                color=color,
            )
        y -= row_step

    scope_box = FancyBboxPatch(
        (0.25, 0.25),
        14.5,
        1.15,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.4, 1.15, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    ok = sum(1 for r in rows if r["finite_outputs"] and r["json_strict"] and r["truth_gates_preserved"])
    ax.text(
        0.4,
        0.78,
        f"contracts_ok={ok}/{len(rows)}  repo_sha={runtime['repo_sha'][:12]}  jaxfne={runtime['jaxfne_version']}",
        fontsize=6.5,
        va="top",
        family="monospace",
        color="#444444",
    )
    ax.text(
        0.4,
        0.48,
        "physical_amplitude_calibrated: false  |  proxy_readout",
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
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_sha": repo_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jaxfne_version": str(jtfne.__version__),
        "jax_version": jax_version,
    }


def _validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        if not row["finite_outputs"]:
            failures.append(f"{row['family']}: non-finite outputs")
        if not row["json_strict"]:
            failures.append(f"{row['family']}: json_strict failed")
        if not row["truth_gates_preserved"]:
            failures.append(f"{row['family']}: truth_gates_preserved=false")
        if row.get("physical_amplitude_calibrated"):
            failures.append(f"{row['family']}: physical_amplitude_calibrated=true")
    if len(rows) != len(READOUT_ORDER):
        failures.append(f"expected {len(READOUT_ORDER)} families, got {len(rows)}")
    return failures


def main() -> int:
    runtime = _runtime_receipt()
    cfg = _two_neuron_config()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    signals = model.simulate(sim)
    readouts = collect_readouts(signals)
    rows = [_contract_row(name, readouts[name]) for name in READOUT_ORDER if name in readouts]

    failures = _validate_rows(rows)
    if failures:
        raise RuntimeError("ED7 probe contract failures:\n  " + "\n  ".join(failures))

    summary = {
        "families": len(rows),
        "finite_ok": sum(1 for r in rows if r["finite_outputs"]),
        "json_strict_ok": sum(1 for r in rows if r["json_strict"]),
        "gates_ok": sum(1 for r in rows if r["truth_gates_preserved"]),
    }

    write_json_strict(
        RECEIPT_PATH,
        {
            "schema_version": "jaxfne.evidence_ed07_probe_contract_receipt.v0.1.0",
            "artifact": "ed07_probe_operator_contracts",
            "generated_at_utc": utc_now_iso(),
            "runtime_receipt": runtime,
            "scope_status": SCOPE_STATUS,
            "physical_amplitude_calibrated": False,
            "performance_claims_allowed": False,
            "truth_gates_preserved": True,
            "simulation": {"duration_ms": 100.0, "dt_ms": 0.1, "seed": 42, "n_neurons": 2},
            "summary": summary,
            "contract_rows": [
                {k: v for k, v in row.items() if k != "report"} for row in rows
            ],
            "truth_gates": truth_gates(),
        },
    )

    draw_figure(rows, runtime)

    root = repo_root()
    manifest = save_figure_manifest(
        figure_id="ed07",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed07_probe_operator_contracts",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed07_probe_operator_contracts.py",
            "claim_boundary": "proxy_readout_operator_contracts_only",
            "physical_amplitude_calibrated": False,
            "performance_claims_allowed": False,
            "truth_gates_preserved": True,
            "simulation": {"duration_ms": 100.0, "dt_ms": 0.1, "seed": 42},
            "contract_summary": summary,
            "contract_rows": rows,
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
    print(f"contracts_ok: {summary['gates_ok']}/{summary['families']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
