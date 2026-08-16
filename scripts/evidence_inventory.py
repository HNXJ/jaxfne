#!/usr/bin/env python3
"""Inventory release artifacts for the evidence stack.

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

# The v0.3.42-era evidence scheme (8 main + 10 extended, figures/evidence)
# was superseded by the publication figure set committed under
# figures/publication/ (fig01..fig07 manuscript figures). Extended-data
# figures are assigned during final Supplement assembly; none committed yet.
MAIN_FIGURES = [
    "fig01_tfne_grammar.png",
    "fig02_emitter_source.png",
    "fig03_local_observation.png",
    "fig04_multiscale_boundary.png",
    "fig05_traveling_wave_no_wave.png",
    "fig06_rbs_hdp_ladder.png",
    "fig07_e_integration.png",
]

EXTENDED_DATA: list[str] = []

EXTENDED_DATA: list[str] = []

INSPECT_DIRS = [
    "docs/evidence",
    "docs/evidence_artifacts",
    "tutorials",
    "examples",
    "outputs/evidence",
    "outputs/publication",
    "figures/evidence",
    "figures/publication",
]

FIGURE_DIR_CANDIDATES = [
    "figures/evidence",
    "figures/publication",
]

# Publication-figure receipts under artifacts/publication/ are the live
# manifests for the committed figures (legacy outputs/evidence manifests no
# longer exist in-tree).
MANIFEST_PATHS = [
    "artifacts/publication/fig01_generation_receipt.json",
    "artifacts/publication/fig02_generation_receipt.json",
    "artifacts/publication/fig03_generation_receipt.json",
    "artifacts/publication/fig04_generation_receipt.json",
    "artifacts/publication/fig05_generation_receipt.json",
    "artifacts/publication/fig06_generation_receipt.json",
    "artifacts/publication/fig07_generation_receipt.json",
    "artifacts/publication/figures_1_7_cross_figure_audit.json",
    "artifacts/publication/publication_evidence_index.json",
    "artifacts/publication/publication_claim_ledger.json",
    "docs/evidence_artifacts/evidence_checklist.json",
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


def inventory_asset(filename: str, figure_dirs: list[Path]) -> dict:
    """Inventory one PNG from canonical or compatibility figure directories.

    The v0.3.42 evidence scripts write to ``figures/evidence``. Some release
    bundles still contain the previously tracked ``figures/publication`` assets.
    Treat the latter as a compatibility source so inventory reports do not false
    fail before the evidence figures are regenerated.
    """
    preferred = figure_dirs[0] / filename
    path = preferred
    for candidate_dir in figure_dirs:
        candidate = candidate_dir / filename
        if candidate.is_file():
            path = candidate
            break
    entry = {
        "filename": filename,
        "path": str(path.relative_to(REPO_ROOT)),
        "canonical_path": str(preferred.relative_to(REPO_ROOT)),
        "exists": path.is_file(),
        "compatibility_path_used": path != preferred,
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
    figure_dirs = [REPO_ROOT / rel for rel in FIGURE_DIR_CANDIDATES]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    main_assets = [inventory_asset(name, figure_dirs) for name in MAIN_FIGURES]
    ed_assets = [inventory_asset(name, figure_dirs) for name in EXTENDED_DATA]

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
        "schema_version": "jaxfne.evidence_inventory.v0.1.0",
        "generated_at_utc": generated_at,
        "repo_sha": git_head_sha(),
        "truth_gates": {
            "claim_level": "computational_scaffold",
            "field_solver_status": "linear_solver",
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
        },
        "inspect_dirs": inspect_summary,
        "main_figures": main_assets,
        "extended_data": ed_assets,
        "manifests_and_reports": manifests,
        "summary": {
            "figure_dir_candidates": FIGURE_DIR_CANDIDATES,
            "compatibility_assets_used": sum(
                1
                for asset in [*main_assets, *ed_assets]
                if asset.get("compatibility_path_used")
            ),
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
    print("=== jaxfne evidence inventory ===")
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
    out_dir = REPO_ROOT / "outputs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inventory.json"
    out_path.write_text(json.dumps(inventory, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print_summary(inventory)
    print(f"wrote: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
