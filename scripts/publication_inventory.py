#!/usr/bin/env python3
"""Inventory publication artifacts for the Nature Methods evidence stack.

Lightweight: no matplotlib/plotly/JAX required. Writes JSON inventory with
SHA256 hashes for files that exist.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

MAIN_FIGURES = [
    "fig01_tfne_architecture.png",
    "fig02_source_field_contracts.png",
    "fig03_jaxfne_backend.png",
    "fig04_minimal_install_run.png",
    "fig05_runtime_scaling.png",
    "fig06_readout_family_panel.png",
    "fig07_reproducibility_artifacts.png",
    "fig08_adjacent_tools_comparison.png",
]

EXTENDED_DATA = [
    "ed01_api_stability_snapshot.png",
    "ed02_json_schema_validation.png",
    "ed03_notebook_execution_receipts.png",
    "ed04_optional_dependency_laziness.png",
    "ed05_manifest_hashes.png",
    "ed06_benchmark_scaling_tables.png",
    "ed07_probe_operator_contracts.png",
    "ed08_tutorial_atlas_coverage.png",
    "ed09_failure_modes_and_nulls.png",
    "ed10_release_archive_receipt.png",
]

INSPECT_DIRS = [
    "docs/publication",
    "tutorials",
    "examples",
    "outputs/publication",
    "figures/publication",
]

MANIFEST_PATHS = [
    "outputs/publication/inventory.json",
    "outputs/publication/fig01_tfne_architecture_manifest.json",
    "outputs/publication/fig02_source_field_contracts_manifest.json",
    "outputs/publication/fig03_jaxfne_backend_manifest.json",
    "outputs/publication/fig04_minimal_install_run_manifest.json",
    "outputs/publication/fig05_runtime_scaling_manifest.json",
    "outputs/publication/fig06_readout_family_panel_manifest.json",
    "outputs/publication/fig07_reproducibility_artifacts_manifest.json",
    "outputs/publication/fig08_adjacent_tools_comparison_manifest.json",
    "outputs/publication/ed01_api_stability_snapshot_manifest.json",
    "outputs/publication/ed02_json_schema_validation_manifest.json",
    "outputs/publication/ed03_notebook_execution_receipts_manifest.json",
    "outputs/publication/ed04_optional_dependency_laziness_manifest.json",
    "outputs/publication/ed05_manifest_hashes_manifest.json",
    "outputs/publication/ed06_benchmark_scaling_tables_manifest.json",
    "outputs/publication/ed07_probe_operator_contracts_manifest.json",
    "outputs/publication/ed08_tutorial_atlas_coverage_manifest.json",
    "outputs/publication/ed09_failure_modes_and_nulls_manifest.json",
    "docs/publication/publication_checklist.json",
    "outputs/v029_two_neuron_ei_multimodal/manifest.json",
    "outputs/v0210_network_100_ei_multimodal/manifest.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def inventory_asset(filename: str, figures_dir: Path) -> dict:
    path = figures_dir / filename
    entry = {
        "filename": filename,
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.is_file(),
        "sha256": None,
        "size_bytes": None,
    }
    if entry["exists"]:
        entry["sha256"] = sha256_file(path)
        entry["size_bytes"] = path.stat().st_size
    return entry


def inventory_path(rel_path: str) -> dict:
    path = REPO_ROOT / rel_path
    entry = {
        "path": rel_path,
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "sha256": None,
        "size_bytes": None,
    }
    if path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["size_bytes"] = path.stat().st_size
    return entry


def count_glob(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob(pattern))


def build_inventory() -> dict:
    figures_dir = REPO_ROOT / "figures" / "publication"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    main_assets = [inventory_asset(name, figures_dir) for name in MAIN_FIGURES]
    ed_assets = [inventory_asset(name, figures_dir) for name in EXTENDED_DATA]

    inspect_summary = {}
    for rel in INSPECT_DIRS:
        path = REPO_ROOT / rel
        inspect_summary[rel] = {
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "file_count": count_glob(path, "*") if path.is_dir() else 0,
            "notebook_count": count_glob(path, "*.ipynb") if path.is_dir() else 0,
            "py_count": count_glob(path, "*.py") if path.is_dir() else 0,
        }

    manifests = [inventory_path(rel) for rel in MANIFEST_PATHS]

    main_present = sum(1 for a in main_assets if a["exists"])
    ed_present = sum(1 for a in ed_assets if a["exists"])

    return {
        "schema_version": "jaxfne.publication_inventory.v0.1.0",
        "generated_at_utc": generated_at,
        "repo_sha": git_head_sha(),
        "truth_gates": {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
        },
        "inspect_dirs": inspect_summary,
        "main_figures": main_assets,
        "extended_data": ed_assets,
        "manifests_and_reports": manifests,
        "summary": {
            "main_figures_present": main_present,
            "main_figures_total": len(MAIN_FIGURES),
            "extended_data_present": ed_present,
            "extended_data_total": len(EXTENDED_DATA),
            "all_figure_assets_present": (
                main_present == len(MAIN_FIGURES) and ed_present == len(EXTENDED_DATA)
            ),
        },
    }


def print_summary(inventory: dict) -> None:
    summary = inventory["summary"]
    print("=== jaxfne publication inventory ===")
    print(f"repo_sha: {inventory.get('repo_sha')}")
    print(f"generated_at_utc: {inventory['generated_at_utc']}")
    print(
        "main figures: "
        f"{summary['main_figures_present']}/{summary['main_figures_total']} present"
    )
    print(
        "extended data: "
        f"{summary['extended_data_present']}/{summary['extended_data_total']} present"
    )
    for rel, info in inventory["inspect_dirs"].items():
        status = "ok" if info["exists"] else "missing"
        print(f"  {rel}: {status} (files={info['file_count']})")
    missing_main = [
        a["filename"] for a in inventory["main_figures"] if not a["exists"]
    ]
    if missing_main:
        print(f"missing main figures ({len(missing_main)}): {', '.join(missing_main[:4])}...")


def main() -> int:
    inventory = build_inventory()
    out_dir = REPO_ROOT / "outputs" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inventory.json"
    out_path.write_text(json.dumps(inventory, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print_summary(inventory)
    print(f"wrote: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
