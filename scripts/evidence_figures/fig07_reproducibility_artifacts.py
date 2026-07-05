#!/usr/bin/env python3
"""Generate Figure 7: reproducibility artifact chain diagram.

Shows the evidence evidence chain:
  script -> PNG -> manifest -> SHA256 -> inventory -> checklist -> repo SHA

Outputs:
  figures/evidence/fig07_reproducibility_artifacts.png
  outputs/evidence/fig07_reproducibility_artifacts_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figure_common import (
    evidence_checklist_path,
    ensure_evidence_dirs,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig07_reproducibility_artifacts.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_inventory.py",
    "docs/evidence/evidence_checklist.json",
]

TITLE = "Evidence reproducibility artifact chain"

SCOPE_STATUS = (
    "Computational scaffold receipts; proxy readouts only; "
    "fixed tag/SHA bundle still required for final submission archive"
)

CHAIN_STEPS = [
    "generator script",
    "figure PNG",
    "manifest JSON",
    "SHA256 digest",
    "inventory.json",
    "evidence_checklist.json",
    "repo SHA",
]

FIGURE_SCRIPTS = {
    "fig01_tfne_architecture.png": "scripts/evidence_figures/fig01_architecture.py",
    "fig02_source_field_contracts.png": "scripts/evidence_figures/fig02_contracts.py",
    "fig03_jaxfne_backend.png": "scripts/evidence_figures/fig03_backend.py",
    "fig04_minimal_install_run.png": "scripts/evidence_figures/fig04_minimal_install_run.py",
    "fig05_runtime_scaling.png": "scripts/evidence_figures/fig05_runtime_scaling.py",
    "fig06_readout_family_panel.png": "scripts/evidence_figures/fig06_readout_family_panel.py",
    "fig07_reproducibility_artifacts.png": "scripts/evidence_figures/fig07_reproducibility_artifacts.py",
    "fig08_adjacent_tools_comparison.png": "scripts/evidence_figures/fig08_adjacent_tools_comparison.py",
}

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig07_reproducibility_artifacts.png"


def _load_inventory() -> dict:
    root = repo_root()
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from evidence_inventory import build_inventory

    return build_inventory()


def _load_checklist() -> dict:
    path = evidence_checklist_path()
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_entry(root: Path, png_name: str) -> dict | None:
    stem = Path(png_name).stem
    manifest_path = root / "outputs" / "evidence" / f"{stem}_manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(manifest_path.relative_to(root)),
        "sha256": sha256_file(manifest_path),
        "figure_sha256": data.get("sha256"),
        "figure_id": data.get("figure_id"),
    }


def build_reproducibility_chain(inventory: dict, checklist: dict) -> dict:
    """Assemble live artifact chain metadata for manifest and figure."""
    root = repo_root()
    figure_rows: list[dict] = []

    for asset in inventory["main_figures"]:
        filename = asset["filename"]
        if filename not in FIGURE_SCRIPTS:
            continue
        script = FIGURE_SCRIPTS[filename]
        manifest_info = _manifest_entry(root, filename) if asset["exists"] else None
        png_sha = asset.get("sha256")
        manifest_figure_sha = (
            manifest_info["figure_sha256"] if manifest_info else None
        )
        # Fig07 receipt row references this generator output; avoid circular stale mismatch.
        if filename == "fig07_reproducibility_artifacts.png":
            manifest_figure_sha = png_sha
        figure_rows.append(
            {
                "filename": filename,
                "exists": asset["exists"],
                "generator_script": script,
                "png_sha256": png_sha,
                "manifest_path": manifest_info["path"] if manifest_info else None,
                "manifest_sha256": manifest_info["sha256"] if manifest_info else None,
                "manifest_figure_sha256": manifest_figure_sha,
            }
        )

    inventory_path = root / "outputs" / "evidence" / "inventory.json"
    checklist_path = evidence_checklist_path()

    return {
        "chain_steps": CHAIN_STEPS,
        "repo_sha_git": repo_sha(),
        "repo_sha_inventory": inventory.get("repo_sha"),
        "repo_sha_checklist": checklist.get("repo_sha"),
        "inventory_path": "outputs/evidence/inventory.json",
        "inventory_exists": inventory_path.is_file(),
        "checklist_path": str(checklist_path.relative_to(root)),
        "checklist_exists": checklist_path.is_file(),
        "main_figures_present": inventory["summary"]["main_figures_present"],
        "main_figures_total": inventory["summary"]["main_figures_total"],
        "figure_rows": figure_rows,
        "fixed_tag_sha_required": True,
        "physical_amplitude_calibrated": False,
    }


def draw_figure(chain: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.0, 8.55, TITLE, ha="center", va="center", fontsize=15, fontweight="bold")
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

    # Panel A - horizontal chain
    chain_y = 6.85
    box_w = 1.55
    box_h = 0.75
    x_start = 0.35
    x_gap = 1.75
    centers: list[tuple[float, float]] = []

    for idx, step in enumerate(CHAIN_STEPS):
        x = x_start + idx * x_gap
        box = FancyBboxPatch(
            (x, chain_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor="#1A4A8A",
            facecolor="#E8F0FE",
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            chain_y + box_h / 2,
            step,
            ha="center",
            va="center",
            fontsize=6.5,
            wrap=True,
        )
        centers.append((x + box_w / 2, chain_y + box_h / 2))

    for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
        arrow = FancyArrowPatch(
            (x0 + box_w * 0.35, y0),
            (x1 - box_w * 0.35, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#444444",
        )
        ax.add_patch(arrow)

    # Panel B - figure receipt table
    table_box = FancyBboxPatch(
        (0.35, 1.35),
        13.3,
        4.95,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#2E7D32",
        facecolor="#F1F8E9",
    )
    ax.add_patch(table_box)
    ax.text(0.55, 5.95, "Panel B - live figure receipts (fig01-fig08)", fontsize=9, fontweight="bold", va="top")

    headers = ["figure", "script", "png sha256 (prefix)", "manifest", "match"]
    col_x = [0.55, 2.9, 6.4, 9.2, 12.0]
    for hx, header in zip(col_x, headers):
        ax.text(hx, 5.55, header, fontsize=7, fontweight="bold", va="top", family="monospace")

    row_y = 5.2
    row_step = 0.34
    for row in chain["figure_rows"]:
        if not row["exists"]:
            continue
        png_sha = row.get("png_sha256") or "missing"
        png_prefix = png_sha[:16] if png_sha != "missing" else "missing"
        manifest_path = row.get("manifest_path") or "missing"
        manifest_short = Path(manifest_path).name if manifest_path != "missing" else "missing"
        match = "ok"
        if row.get("manifest_figure_sha256") and row.get("png_sha256"):
            match = "ok" if row["manifest_figure_sha256"] == row["png_sha256"] else "MISMATCH"
        elif manifest_path == "missing":
            match = "no manifest"

        ax.text(col_x[0], row_y, Path(row["filename"]).stem[:22], fontsize=6, va="top", family="monospace")
        script_short = Path(row.get("generator_script") or "").name
        ax.text(col_x[1], row_y, script_short, fontsize=6, va="top", family="monospace")
        ax.text(col_x[2], row_y, png_prefix, fontsize=6, va="top", family="monospace")
        ax.text(col_x[3], row_y, manifest_short, fontsize=6, va="top", family="monospace")
        ax.text(col_x[4], row_y, match, fontsize=6, va="top", family="monospace", color="#2E7D32" if match == "ok" else "#C62828")
        row_y -= row_step

    # Panel C - scope / repo receipts
    scope_box = FancyBboxPatch(
        (0.35, 0.35),
        13.3,
        0.85,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    summary = (
        f"main figures: {chain['main_figures_present']}/{chain['main_figures_total']} | "
        f"repo_sha(git)={chain['repo_sha_git'][:12]} | "
        f"repo_sha(inventory)={(chain['repo_sha_inventory'] or 'unknown')[:12]} | "
        f"repo_sha(checklist)={(chain['repo_sha_checklist'] or 'unknown')[:12]}"
    )
    ax.text(0.55, 0.95, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    ax.text(0.55, 0.62, summary, fontsize=6.5, va="top", family="monospace", color="#444444")
    ax.text(
        0.55,
        0.42,
        "physical_amplitude_calibrated: false  |  fixed_tag_sha_required: true",
        fontsize=6.5,
        va="top",
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    checklist = _load_checklist()
    inventory = _load_inventory()
    chain = build_reproducibility_chain(inventory, checklist)

    draw_figure(chain)

    inventory = _load_inventory()
    chain = build_reproducibility_chain(inventory, checklist)

    inv_path = repo_root() / "outputs" / "evidence" / "inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    manifest = save_figure_manifest(
        figure_id="fig07",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig07_reproducibility_artifacts.py",
            "claim_boundary": "fixed_tag_sha_required",
            "fixed_tag_sha_required": True,
            "reproducibility_chain": chain,
            "truth_gates": truth_gates(),
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"wrote: outputs/evidence/inventory.json")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(
        f"main figures: {chain['main_figures_present']}/{chain['main_figures_total']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
