#!/usr/bin/env python3
"""Generate Extended Data ED2: JSON schema and config validation panel.

Outputs:
  figures/evidence/ed02_json_schema_validation.png
  outputs/evidence/ed02_json_schema_validation_manifest.json

Uses validate_config fixtures aligned with tests/test_config_schema_v015.py.
"""

from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import jaxfne as jtfne
from jaxfne.core import _JAXFNE_CONFIG_SCHEMA_VERSION

from _figure_common import (
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
    "tests/test_config_schema_v015.py",
    "jaxfne/core.py",
]

TITLE = "JSON schema and config validation receipts (JaxFNEConfig)"

SCOPE_STATUS = (
    "Evidence-local validation panel from validate_config fixtures; "
    "blocking issues for missing sections, unsupported schema_version, and truth escalation; "
    "JSON-safe dict emission checked with allow_nan=False"
)

_VALID_TRUTH = {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "laminar_proxy_no_pde",
    "physical_amplitude_claim_allowed": False,
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}

_MINIMAL_CONFIG = {
    "schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
    "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 42},
    "truth": dict(_VALID_TRUTH),
    "network": {"n": 10},
    "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
    "field": {
        "domain": "laminar_column",
        "conductivity": "proxy",
        "boundary": "mean_zero_neumann",
        "gauge": "mean_zero",
    },
    "probes": [{"name": "laminar_probe", "n_contacts": 16}],
}

VALIDATION_FIXTURES: list[tuple[str, dict, bool, str]] = [
    ("valid_minimal", _MINIMAL_CONFIG, True, "baseline .jcfg.json"),
    (
        "missing_network",
        {k: v for k, v in _MINIMAL_CONFIG.items() if k != "network"},
        False,
        "required_section_missing:network",
    ),
    (
        "truth_escalation_blocked",
        {
            **_MINIMAL_CONFIG,
            "truth": {**_VALID_TRUTH, "physical_amplitude_claim_allowed": True},
        },
        False,
        "physical_amplitude_claim_allowed",
    ),
    (
        "unsupported_schema_version",
        {**_MINIMAL_CONFIG, "schema_version": "unsupported.version"},
        False,
        "schema_version_unsupported",
    ),
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed02_json_schema_validation.png"
RECEIPT_JSON_PATH = _dirs["outputs"] / "ed02_json_schema_validation_receipt.json"


def _write_config(config_dict: dict, path: Path) -> None:
    path.write_text(json.dumps(config_dict, allow_nan=False) + "\n", encoding="utf-8")


def run_validation_fixtures() -> list[dict]:
    """Execute validate_config fixtures and return receipt rows."""
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for fixture_id, config_dict, expected_valid, expected_issue_hint in VALIDATION_FIXTURES:
            path = tmp / f"{fixture_id}.jcfg.json"
            _write_config(config_dict, path)
            cfg = jtfne.load_config(path)
            result = jtfne.validate_config(cfg)
            issue_text = "; ".join(result.issues[:3]) if result.issues else ""
            warning_text = "; ".join(result.warnings[:2]) if result.warnings else ""
            passed = result.valid is expected_valid
            if expected_valid:
                issue_ok = len(result.issues) == 0
            else:
                issue_ok = expected_issue_hint in issue_text or any(
                    expected_issue_hint in issue for issue in result.issues
                )
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "expected_valid": expected_valid,
                    "observed_valid": result.valid,
                    "issues": list(result.issues),
                    "warnings": list(result.warnings),
                    "issue_sample": issue_text,
                    "warning_sample": warning_text,
                    "expected_issue_hint": expected_issue_hint,
                    "fixture_pass": passed and issue_ok,
                    "schema_version": result.schema_version,
                    "truth_boundary_physical_amplitude": result.truth_boundary.get(
                        "physical_amplitude_claim_allowed"
                    ),
                }
            )
    return rows


def json_safe_roundtrip_check() -> dict:
    """Verify ConfigValidationResult and JaxFNEConfig dicts are strict JSON-safe."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "roundtrip.jcfg.json"
        _write_config(_MINIMAL_CONFIG, path)
        cfg = jtfne.load_config(path)
        result = jtfne.validate_config(cfg)
        cfg_dict = cfg.to_dict()
        result_dict = result.to_dict()
        json.dumps(cfg_dict, allow_nan=False)
        json.dumps(result_dict, allow_nan=False)
        return {
            "cfg_to_dict_keys": sorted(cfg_dict.keys()),
            "validation_valid": result.valid,
            "json_strict_allow_nan_false": True,
            "config_hash": cfg.config_hash,
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
        "config_schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
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
        "Panel A - validate_config fixture receipts",
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    headers = ["fixture", "expected", "observed", "pass", "issue sample"]
    col_x = [0.75, 3.0, 4.3, 5.5, 6.3]
    y = 6.95
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    y = 6.55
    for row in rows:
        color = "#2E7D32" if row["fixture_pass"] else "#C62828"
        ax.text(col_x[0], y, row["fixture_id"], fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[1], y, str(row["expected_valid"]), fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[2], y, str(row["observed_valid"]), fontsize=6.5, va="top", family="monospace")
        ax.text(col_x[3], y, str(row["fixture_pass"]), fontsize=6.5, va="top", family="monospace", color=color)
        sample = row["issue_sample"] or "(none)"
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
        f"schema_version: {receipt['config_schema_version']}",
        f"json_strict_roundtrip: {json_check['json_strict_allow_nan_false']}",
        f"validation_valid_baseline: {json_check['validation_valid']}",
        f"config_hash_prefix: {str(json_check['config_hash'])[:16]}",
        f"jaxfne: {receipt['jaxfne_version']}  repo_sha: {receipt['repo_sha'][:12]}",
        "physical_amplitude_claim_allowed: false  |  blocking truth escalation enforced",
    ]
    for i, line in enumerate(lines):
        ax.text(0.85, 1.55 - i * 0.22, line, fontsize=6.5, va="top", family="monospace", color="#444444")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    receipt = _runtime_receipt()
    rows = run_validation_fixtures()
    json_check = json_safe_roundtrip_check()

    if not all(row["fixture_pass"] for row in rows):
        failed = [r["fixture_id"] for r in rows if not r["fixture_pass"]]
        raise RuntimeError(f"validate_config fixture failures: {failed}")
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
            "claim_boundary": "config_schema_validation_receipt",
            "config_schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
            "validation_fixtures": rows,
            "json_check": json_check,
            "runtime_receipt": receipt,
            "receipt_json_path": receipt_rel,
            "receipt_json_sha256": sha256_file(RECEIPT_JSON_PATH),
            "truth_gates": truth_gates(),
            "physical_amplitude_claim_allowed": False,
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
