#!/usr/bin/env python3
"""Generate Extended Data ED2: JSON schema and config validation panel.

Outputs:
  figures/evidence/ed02_json_schema_validation.png
  outputs/evidence/ed02_json_schema_validation_manifest.json

REDESIGNED 2026-06-30: previously validated the legacy .jcfg.json/JaxFNEConfig
format (load_config/validate_config), which was deleted (lived only in tests,
never a real asset outside this package). Repurposed to validate the current
NeuronalTensor JSON schema instead, exercising jtfne.load()'s real
raise-on-invalid / warn-on-version-drift behavior plus a JSON-safe round-trip
check on NeuronalTensor.to_dict() -- the same evidence-receipt spirit
(schema validation receipts), on the format that's actually current.
"""

from __future__ import annotations

import json
import platform
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import jaxfne as jtfne
from jaxfne.neuronal_tensor import Area, Layer, NeuronalTensor, NeuronType

from _figure_common import (
    save_matplotlib_figure,
    ensure_evidence_dirs,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    write_json_strict,
)


SOURCE_FILES = [
    "scripts/evidence_figures/ed02_json_schema_validation.py",
    "scripts/evidence_figures/_figure_common.py",
    "tests/test_neuronal_tensor.py",
    "jaxfne/neuronal_tensor.py",
]

TITLE = "JSON schema and NeuronalTensor validation receipts"

SCOPE_STATUS = (
    "Evidence-local validation panel exercising jtfne.load()'s real schema checks "
    "(missing top-level 'areas' key raises ValueError, schema_version drift warns but "
    "still loads -- forward-readability, not strict lockstep) plus a JSON-safe "
    "NeuronalTensor.to_dict() round-trip check with allow_nan=False"
)

_MINIMAL_TENSOR_DICT = {
    "schema_version": jtfne.NEURONAL_TENSOR_SCHEMA_VERSION,
    "areas": [
        {
            "name": "ED2TestArea",
            "layers": [
                {
                    "name": "L1",
                    "n_neurons": 10,
                    "neuron_types": [{"name": "E", "fraction": 1.0}],
                }
            ],
            "inter_connections": [],
        }
    ],
    "area_connections": [],
}

VALIDATION_FIXTURES: list[tuple[str, dict, bool, str]] = [
    ("valid_minimal", _MINIMAL_TENSOR_DICT, True, "baseline NeuronalTensor JSON"),
    (
        "missing_areas_key",
        {"schema_version": jtfne.NEURONAL_TENSOR_SCHEMA_VERSION},
        False,
        "does not look like a NeuronalTensor JSON config",
    ),
    (
        "schema_version_drift_warns_not_errors",
        {**_MINIMAL_TENSOR_DICT, "schema_version": "future_version_not_yet_released"},
        True,
        "declares schema_version=",
    ),
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed02_json_schema_validation.png"
RECEIPT_JSON_PATH = _dirs["outputs"] / "ed02_json_schema_validation_receipt.json"


def _write_config(config_dict: dict, path: Path) -> None:
    path.write_text(json.dumps(config_dict, allow_nan=False) + "\n", encoding="utf-8")


def run_validation_fixtures() -> list[dict]:
    """Execute jtfne.load() schema fixtures and return receipt rows."""
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for fixture_id, config_dict, expected_loadable, expected_message_hint in VALIDATION_FIXTURES:
            path = tmp / f"{fixture_id}.json"
            _write_config(config_dict, path)
            observed_loadable = True
            issue_text = ""
            warning_text = ""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    tensor = jtfne.load(str(path))
                    observed_loadable = tensor is not None
                except ValueError as exc:
                    observed_loadable = False
                    issue_text = str(exc)
                if caught:
                    warning_text = "; ".join(str(w.message) for w in caught[:2])
            message_text = issue_text or warning_text
            message_ok = expected_message_hint in message_text if message_text else expected_loadable
            passed = (observed_loadable is expected_loadable) and message_ok
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "expected_loadable": expected_loadable,
                    "observed_loadable": observed_loadable,
                    "issue_sample": issue_text[:120],
                    "warning_sample": warning_text[:120],
                    "expected_message_hint": expected_message_hint,
                    "fixture_pass": passed,
                }
            )
    return rows


def json_safe_roundtrip_check() -> dict:
    """Verify NeuronalTensor.to_dict() is strict JSON-safe (allow_nan=False)."""
    tensor = NeuronalTensor(
        areas=[
            Area(
                name="ED2RoundtripArea",
                layers=[Layer(name="L1", n_neurons=10, neuron_types=[NeuronType(name="E", fraction=1.0)])],
            )
        ]
    )
    tensor_dict = tensor.to_dict()
    json.dumps(tensor_dict, allow_nan=False)
    return {
        "tensor_to_dict_keys": sorted(tensor_dict.keys()),
        "json_strict_allow_nan_false": True,
    }


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
    return {
        "jaxfne_version": str(getattr(jtfne, "__version__", "unknown")),
        "repo_sha": repo_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "neuronal_tensor_schema_version": jtfne.NEURONAL_TENSOR_SCHEMA_VERSION,
    }


def draw_figure(rows: list[dict], json_check: dict, receipt: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.0, 8.55, TITLE, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        7.0,
        8.05,
        "import jaxfne as jtfne",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        color="#333333",
    )

    table_box = FancyBboxPatch(
        (0.55, 2.85),
        12.9,
        4.75,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#1A4A8A",
        facecolor="#F8FAFF",
    )
    ax.add_patch(table_box)
    ax.text(
        0.75,
        7.35,
        "Panel A - jtfne.load() NeuronalTensor schema fixture receipts",
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    headers = ["fixture", "expected", "observed", "pass", "message sample"]
    col_x = [0.75, 3.0, 4.3, 5.5, 6.3]
    y = 6.95
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    y = 6.55
    for row in rows:
        color = "#2E7D32" if row["fixture_pass"] else "#C62828"
        ax.text(col_x[0], y, row["fixture_id"], fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[1], y, str(row["expected_loadable"]), fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[2], y, str(row["observed_loadable"]), fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[3], y, str(row["fixture_pass"]), fontsize=6.5, va="top", family="monospace", color=color)
        sample = row["issue_sample"] or row["warning_sample"] or "(none)"
        ax.text(col_x[4], y, sample[:58], fontsize=6, va="top", family="monospace")
        y -= 0.55

    receipt_box = FancyBboxPatch(
        (0.55, 0.45),
        12.9,
        2.15,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(receipt_box)
    ax.text(0.75, 2.35, "Panel B - JSON/schema receipt / scope", fontsize=9, fontweight="bold", va="top")
    ax.text(0.85, 1.95, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    lines = [
        f"neuronal_tensor_schema_version: {receipt['neuronal_tensor_schema_version']}",
        f"json_strict_roundtrip: {json_check['json_strict_allow_nan_false']}",
        f"tensor_to_dict_keys: {', '.join(json_check['tensor_to_dict_keys'])}",
        f"jaxfne: {receipt['jaxfne_version']}  repo_sha: {receipt['repo_sha'][:12]}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.85, 1.55 - i * 0.22, line, fontsize=6.5, va="top", family="monospace", color="#444444")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def main() -> int:
    receipt = _runtime_receipt()
    rows = run_validation_fixtures()
    json_check = json_safe_roundtrip_check()

    if not all(row["fixture_pass"] for row in rows):
        failed = [r["fixture_id"] for r in rows if not r["fixture_pass"]]
        raise RuntimeError(f"jtfne.load() schema fixture failures: {failed}")
    if not json_check["json_strict_allow_nan_false"]:
        raise RuntimeError("JSON strict round-trip check failed")

    write_json_strict(
        RECEIPT_JSON_PATH,
        {
            "schema_version": "jaxfne.evidence_ed02_validation_receipt.v0.1.0",
            "validation_fixtures": rows,
            "json_check": json_check,
            "runtime_receipt": receipt,
            "truth_gates": truth_gates(),
        },
    )

    draw_figure(rows, json_check, receipt)

    receipt_rel = str(RECEIPT_JSON_PATH.relative_to(repo_root()))
    manifest = save_figure_manifest(
        figure_id="ed02",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed02_json_schema_validation.py",
            "claim_boundary": "neuronal_tensor_schema_validation_receipt",
            "neuronal_tensor_schema_version": jtfne.NEURONAL_TENSOR_SCHEMA_VERSION,
            "validation_fixtures": rows,
            "json_check": json_check,
            "runtime_receipt": receipt,
            "receipt_json_path": receipt_rel,
            "receipt_json_sha256": sha256_file(RECEIPT_JSON_PATH),
            "truth_gates": truth_gates(),
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"wrote: {receipt_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"fixtures_passed: {sum(1 for r in rows if r['fixture_pass'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
