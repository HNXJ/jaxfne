#!/usr/bin/env python3
"""Generate Figure 6: eight proxy readout family panel with operator receipts.

Outputs:
  figures/evidence/fig06_readout_family_panel.png
  outputs/evidence/fig06_readout_family_panel_manifest.json
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

from _figure_common import (
    save_matplotlib_figure,
    ensure_evidence_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig06_readout_family_panel.py",
    "scripts/evidence_figures/_figure_common.py",
    "examples/04_two_neuron_ei_multimodal.py",
    "jaxfne/fields/probes.py",
]

TITLE = "Eight proxy readout families (operator contracts)"

SCOPE_STATUS = (
    "All readouts are simulated proxies; proxy_readout; "
    "physical_amplitude_calibrated: false; no empirical EEG/MEG validation"
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

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig06_readout_family_panel.png"


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
    """Apply all eight probe operators and return ProbeReadout map."""
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


def _probe_receipt(name: str, readout) -> dict:
    report = dict(readout.report)
    data = readout.data
    return {
        "family": name,
        "shape": tuple(data.shape) if hasattr(data, "shape") else None,
        "method": report.get("method"),
        "operator_status": report.get("operator_status"),
        "calibration_status": report.get("calibration_status"),
        "source_calibration_status": report.get("source_calibration_status"),
        "field_solver_status": report.get("field_solver_status"),
        "field_claim_level": report.get("field_claim_level"),
        "physical_amplitude_calibrated": report.get("physical_amplitude_calibrated"),
        "finite_outputs": _finite_array(data),
    }


def _plot_trace(ax, name: str, data, time_ms) -> None:
    arr = np.asarray(data)
    if arr.ndim == 2:
        trace = arr.mean(axis=1)
        x = time_ms
    elif arr.ndim == 1 and arr.shape[0] == time_ms.shape[0]:
        trace = arr
        x = time_ms
    else:
        x = np.arange(arr.size)
        trace = arr
    if name == "spk":
        ax.plot(x, trace, drawstyle="steps-post", linewidth=0.9, color="#C62828")
    else:
        ax.plot(x, trace, linewidth=0.9, color="#1565C0")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.2, linewidth=0.4)


def draw_figure(readouts: dict, receipts: list[dict], time_ms) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), dpi=150)
    axes_flat = axes.flatten()

    for idx, name in enumerate(READOUT_ORDER):
        ax = axes_flat[idx]
        readout = readouts.get(name)
        receipt = next(r for r in receipts if r["family"] == name)
        if readout is None:
            ax.text(0.5, 0.5, f"{name}\n(unavailable)", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue

        _plot_trace(ax, name, readout.data, time_ms)
        ax.set_title(name, fontsize=8, fontweight="bold", pad=2)
        meta = (
            f"shape={receipt['shape']}\n"
            f"method={receipt['method']}\n"
            f"operator={receipt['operator_status']}\n"
            f"field={receipt['field_solver_status']}\n"
            f"finite={receipt['finite_outputs']}"
        )
        ax.text(
            0.02,
            0.98,
            meta,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0.3),
        )

    fig.suptitle(TITLE, fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.985,
        "import jaxfne as jtfne  |  two-neuron E/I multimodal proxy panel",
        ha="center",
        fontsize=8,
        family="monospace",
        color="#333333",
    )
    fig.text(
        0.02,
        0.01,
        SCOPE_STATUS,
        fontsize=7,
        family="monospace",
        color="#444444",
    )
    fig.text(
        0.02,
        -0.01,
        "physical_amplitude_calibrated: false  |  proxy_readout",
        fontsize=6.5,
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def main() -> int:
    cfg = _two_neuron_config()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    signals = model.simulate(sim)
    readouts = collect_readouts(signals)
    receipts = [_probe_receipt(name, readouts[name]) for name in READOUT_ORDER if name in readouts]

    if not all(r["finite_outputs"] for r in receipts):
        raise RuntimeError("Non-finite readout detected in Figure 6 panel")
    if any(r["physical_amplitude_calibrated"] for r in receipts):
        raise RuntimeError("physical_amplitude_calibrated must remain false")

    draw_figure(readouts, receipts, np.asarray(signals.time_ms))

    manifest = save_figure_manifest(
        figure_id="fig06",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig06_readout_family_panel.py",
            "claim_boundary": "proxy_readouts_only",
            "simulation": {"duration_ms": 100.0, "dt_ms": 0.1, "seed": 42, "n_neurons": 2},
            "readout_families": READOUT_ORDER,
            "probe_receipts": receipts,
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"readouts: {len(receipts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
