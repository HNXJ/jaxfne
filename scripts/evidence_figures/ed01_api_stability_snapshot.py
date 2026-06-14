#!/usr/bin/env python3
"""Generate Extended Data ED1: public API stability snapshot panel.

Outputs:
  figures/evidence/ed01_api_stability_snapshot.png
  outputs/evidence/ed01_api_stability_snapshot_manifest.json

Local snapshot only; does not claim completeness beyond tested import surface.
"""

from __future__ import annotations

import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import jaxfne as jtfne

from _figure_common import (
    ensure_evidence_dirs,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
)


SOURCE_FILES = [
    "scripts/evidence_figures/ed01_api_stability_snapshot.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/snapshot_public_api.py",
    "jaxfne/__init__.py",
]

TITLE = "jaxfne public API stability snapshot (local import surface)"

SCOPE_STATUS = (
    "Local snapshot of import jaxfne as jtfne public names; "
    "not a formal completeness audit; optional visualization extras not eagerly imported"
)

API_GROUPS = {
    "config": {
        "label": "config",
        "members": (
            "configuration",
            "Configuration",
            "Config",
            "validate_config",
            "load_config",
            "runtime",
            "RuntimeConfig",
            "config_truth_boundary",
        ),
    },
    "construct": {
        "label": "construct",
        "members": ("construct", "Model", "config_to_configuration"),
    },
    "simulate": {
        "label": "simulate",
        "members": (
            "simulate",
            "simulation",
            "Simulation",
            "run_trials",
            "Paradigm",
            "paradigm",
        ),
    },
    "signals": {
        "label": "signals",
        "members": ("Signals", "Signal", "get_signal", "FieldOutput"),
    },
    "probes": {
        "label": "probes",
        "members": (
            "ReadoutSpec",
            "readout_spec",
            "probe_laminar_modes",
            "project_laminar_sources",
            "LinearReadout",
        ),
    },
    "vis": {
        "label": "vis",
        "members": ("vis",),
    },
    "optim": {
        "label": "optim",
        "members": (
            "agsdr",
            "gsdr",
            "objective",
            "Objective",
            "optax_adam",
            "optax_sgd",
            "random_search",
            "TuneResult",
        ),
    },
    "io": {
        "label": "io",
        "members": (
            "manifest",
            "save_json",
            "json_safe",
            "sha256_file",
            "sha256_text",
            "save_receipt",
            "RunReceipt",
            "run_receipt",
        ),
    },
}

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed01_api_stability_snapshot.png"


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
        "snapshot_scope": "local_import_surface_only",
    }


def collect_api_snapshot() -> dict:
    """Build grouped public API snapshot from jtfne.__all__ without optional eager imports."""
    public_names = set(getattr(jtfne, "__all__", []))
    groups: dict[str, dict] = {}
    covered: set[str] = set()

    for key, spec in API_GROUPS.items():
        present = [name for name in spec["members"] if name in public_names]
        missing = [name for name in spec["members"] if name not in public_names]
        groups[key] = {
            "label": spec["label"],
            "present": present,
            "missing": missing,
            "present_count": len(present),
            "tracked_count": len(spec["members"]),
        }
        covered.update(spec["members"])

    other_public = sorted(public_names - covered)
    return {
        "public_name_count": len(public_names),
        "groups": groups,
        "other_public_sample": other_public[:12],
        "other_public_count": len(other_public),
        "has_vis_module": hasattr(jtfne, "vis"),
        "matplotlib_imported_for_figure_only": True,
    }


def draw_figure(snapshot: dict, receipt: dict) -> None:
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
        (0.55, 2.65),
        12.9,
        5.0,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#1A4A8A",
        facecolor="#F8FAFF",
    )
    ax.add_patch(table_box)
    ax.text(0.75, 7.35, "Panel A - public API groups (__all__ membership)", fontsize=9, fontweight="bold", va="top")

    headers = ["group", "present/tracked", "sample present names"]
    col_x = [0.75, 2.4, 4.0]
    y = 6.95
    for x, header in zip(col_x, headers):
        ax.text(x, y, header, fontsize=7.5, fontweight="bold", va="top", family="monospace")

    y = 6.55
    for group in snapshot["groups"].values():
        present = group["present"]
        sample = ", ".join(present[:4]) if present else "(none)"
        if len(present) > 4:
            sample += ", ..."
        ax.text(col_x[0], y, group["label"], fontsize=7, va="top", family="monospace")
        ax.text(
            col_x[1],
            y,
            f"{group['present_count']}/{group['tracked_count']}",
            fontsize=7,
            va="top",
            family="monospace",
        )
        ax.text(col_x[2], y, sample, fontsize=6.5, va="top", family="monospace")
        y -= 0.42

    ax.text(
        0.75,
        y - 0.1,
        f"__all__ count: {snapshot['public_name_count']} | other public names: {snapshot['other_public_count']}",
        fontsize=7,
        va="top",
        family="monospace",
    )
    ax.text(
        0.75,
        y - 0.45,
        f"other sample: {', '.join(snapshot['other_public_sample'][:8])}",
        fontsize=6.5,
        va="top",
        family="monospace",
        color="#444444",
    )

    receipt_box = FancyBboxPatch(
        (0.55, 0.45),
        12.9,
        1.95,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(receipt_box)
    ax.text(0.75, 2.15, "Panel B - snapshot receipt / scope", fontsize=9, fontweight="bold", va="top")
    ax.text(0.85, 1.75, SCOPE_STATUS, fontsize=7, va="top", color="#333333")
    receipt_lines = [
        f"jaxfne_version: {receipt['jaxfne_version']}",
        f"repo_sha: {receipt['repo_sha']}",
        f"python: {receipt['python_version']}  jax: {receipt['jax_version']}  jaxlib: {receipt['jaxlib_version']}",
        f"has_vis_module: {snapshot['has_vis_module']}  matplotlib_for_figure_only: {snapshot['matplotlib_imported_for_figure_only']}",
        "physical_amplitude_claim_allowed: false  |  snapshot_scope: local_import_surface_only",
    ]
    for i, line in enumerate(receipt_lines):
        ax.text(0.85, 1.35 - i * 0.22, line, fontsize=6.5, va="top", family="monospace", color="#444444")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    receipt = _runtime_receipt()
    snapshot = collect_api_snapshot()

    draw_figure(snapshot, receipt)

    manifest = save_figure_manifest(
        figure_id="ed01",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed01_api_stability_snapshot.py",
            "claim_boundary": "local_api_snapshot_only",
            "api_snapshot": snapshot,
            "runtime_receipt": receipt,
            "truth_gates": truth_gates(),
            "completeness_claim_allowed": False,
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"public_names: {snapshot['public_name_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
