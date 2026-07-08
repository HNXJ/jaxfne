#!/usr/bin/env python3
"""Generate Figure 4: minimal install, 10 s smoke run, multimodal proxy readout receipt.

Outputs:
  figures/evidence/fig04_minimal_install_run.png
  outputs/evidence/fig04_minimal_install_run_manifest.json
  outputs/evidence/fig04_smoke_stdout.log
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import jaxfne as jtfne

from _figure_common import (
    save_matplotlib_figure,
    ensure_evidence_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
    write_json_strict,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig04_minimal_install_run.py",
    "scripts/evidence_figures/_figure_common.py",
    "docs/quickstart.md",
    "tests/test_api_smoke.py",
]

TITLE = "Minimal install, 10 s smoke run, and multimodal proxy readout receipt"

SCOPE_STATUS = (
    "CPU-safe tutorial smoke; proxy readouts only; no GPU requirement; "
    "no calibrated EEG/MEG or physical amplitude"
)

INSTALL_LINES = [
    "pip install jaxfne",
    'python -c "import jaxfne as jtfne; print(jtfne.__version__)"',
    "# editable dev install: pip install -e .",
]

CODE_LINES = [
    "import jaxfne as jtfne",
    "cfg = (",
    '    jtfne.configuration()',
    '    .network(name="V1", kind="cortical_column", n=12,',
    '             cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})',
    '    .emitter(family="izhikevich", preset="cortical_eig")',
    '    .field(domain="laminar_column", conductivity="proxy",',
    '           boundary="mean_zero_neumann", gauge="mean_zero")',
    '    .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])',
    ")",
    "model = jtfne.construct(cfg)",
    "signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0))",
    'readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])',
]

READOUT_PANELS = [
    ("spikes", "Spikes (MUA-proxy path)", "#C62828"),
    ("V_m", "V_m (membrane)", "#1565C0"),
    ("csd_proxy", "CSD proxy (laminar)", "#2E7D32"),
    ("lfp_proxy", "LFP proxy (laminar)", "#6A1B9A"),
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig04_minimal_install_run.png"
STDOUT_LOG_PATH = _dirs["outputs"] / "fig04_smoke_stdout.log"


def _smoke_config(n: int = 12) -> jtfne.Configuration:
    return (
        jtfne.configuration()
        .network(
            name="V1",
            kind="cortical_column",
            n=n,
            cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def run_smoke() -> tuple[dict, dict, dict, list[str]]:
    """Execute 10 s CPU smoke and return signals, readout, manifest, receipt lines."""
    cfg = _smoke_config(12)
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    manifest = model.manifest(signals, readout)

    receipt: list[str] = []
    receipt.append(f"jaxfne_version: {jtfne.__version__}")
    receipt.append(f"V_m.shape: {tuple(signals.V_m.shape)} dtype={signals.V_m.dtype}")
    receipt.append(f"spikes.shape: {tuple(signals.spikes.shape)} sum={float(signals.spikes.sum())}")
    for key in ("spikes", "V_m", "CSD", "csd_proxy", "LFP", "lfp_proxy"):
        if key not in readout:
            continue
        arr = readout[key]
        if hasattr(arr, "shape"):
            finite = bool(jnp.all(jnp.isfinite(arr)))
            receipt.append(f"{key}.shape: {tuple(arr.shape)} finite={finite}")
    receipt.append(
        "physical_amplitude_calibrated: "
        f"{manifest['source_field_status']['physical_amplitude_calibrated']}"
    )
    receipt.append("field_solver_status: linear_solver")
    receipt.append("claim_level: computational_scaffold")

    return signals, readout, manifest, receipt


def write_stdout_log(receipt: list[str]) -> None:
    """Persist smoke stdout receipt for evidence evidence."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("=== fig04 smoke run receipt ===")
        for line in receipt:
            print(line)
    STDOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    STDOUT_LOG_PATH.write_text(buffer.getvalue(), encoding="utf-8")


def _text_panel(ax, title: str, lines: list[str], *, facecolor: str, edgecolor: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(0.06, 0.92, title, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(
            0.06,
            0.84 - i * 0.055,
            line,
            fontsize=6.5,
            va="top",
            family="monospace",
            transform=ax.transAxes,
        )


def _plot_readout_trace(ax, time_ms, values, label: str, color: str) -> None:
    if values.ndim == 2:
        trace = jnp.mean(values, axis=1)
    else:
        trace = values
    ax.plot(time_ms, trace, color=color, linewidth=1.0)
    ax.set_title(label, fontsize=8, pad=2)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("time (ms)", fontsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def draw_figure(signals, readout: dict, receipt: list[str]) -> None:
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, 1.35, 0.85], hspace=0.38, wspace=0.32)

    fig.suptitle(TITLE, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.955, "import jaxfne as jtfne", ha="center", fontsize=9, family="monospace", color="#333333")

    ax_install = fig.add_subplot(gs[0, 0:2])
    _text_panel(
        ax_install,
        "Panel A — minimal install (CPU-safe)",
        INSTALL_LINES,
        facecolor="#E8F5E9",
        edgecolor="#2E7D32",
    )

    ax_code = fig.add_subplot(gs[0, 2:4])
    _text_panel(
        ax_code,
        "Panel B — 10 s smoke run (package-native API)",
        CODE_LINES,
        facecolor="#E8F0FE",
        edgecolor="#1A4A8A",
    )

    time_ms = signals.time_ms
    for col, (key, label, color) in enumerate(READOUT_PANELS):
        ax = fig.add_subplot(gs[1, col])
        values = readout.get(key)
        if values is None:
            ax.text(0.5, 0.5, f"{key}\n(missing)", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue
        _plot_readout_trace(ax, time_ms, values, label, color)

    ax_receipt = fig.add_subplot(gs[2, :])
    ax_receipt.set_xlim(0, 1)
    ax_receipt.set_ylim(0, 1)
    ax_receipt.axis("off")
    box = FancyBboxPatch(
        (0.01, 0.05),
        0.98,
        0.9,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
        transform=ax_receipt.transAxes,
    )
    ax_receipt.add_patch(box)
    ax_receipt.text(0.02, 0.88, "Panel C — smoke receipt / scope", fontsize=9, fontweight="bold", va="top", transform=ax_receipt.transAxes)
    ax_receipt.text(0.02, 0.72, SCOPE_STATUS, fontsize=7.5, va="top", transform=ax_receipt.transAxes, color="#333333")
    for i, line in enumerate(receipt[:6]):
        ax_receipt.text(
            0.02,
            0.58 - i * 0.09,
            line,
            fontsize=6.5,
            va="top",
            family="monospace",
            transform=ax_receipt.transAxes,
        )
    for i, line in enumerate(receipt[6:]):
        ax_receipt.text(
            0.52,
            0.58 - i * 0.09,
            line,
            fontsize=6.5,
            va="top",
            family="monospace",
            transform=ax_receipt.transAxes,
        )
    ax_receipt.text(
        0.02,
        0.08,
        "proxy_readout  |  physical_amplitude_calibrated: false  |  no GPU required",
        fontsize=6.5,
        va="bottom",
        family="monospace",
        color="#555555",
        transform=ax_receipt.transAxes,
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def main() -> int:
    signals, readout, manifest, receipt = run_smoke()
    write_stdout_log(receipt)
    draw_figure(signals, readout, receipt)

    smoke_receipt_path = str(STDOUT_LOG_PATH.relative_to(repo_root()))
    manifest = save_figure_manifest(
        figure_id="fig04",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig04_minimal_install_run.py",
            "smoke_duration_ms": 10.0,
            "smoke_dt_ms": 0.1,
            "smoke_seed": 0,
            "smoke_stdout_log": smoke_receipt_path,
            "readout_keys": sorted(k for k in readout if hasattr(readout.get(k), "shape")),
            "claim_boundary": "cpu_safe_tutorial",
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"wrote: {smoke_receipt_path}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
