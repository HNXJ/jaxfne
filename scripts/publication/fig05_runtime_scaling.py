#!/usr/bin/env python3
"""Generate Figure 5: CPU-safe runtime scaling receipt over N, contacts, and seeds.

Outputs:
  figures/publication/fig05_runtime_scaling.png
  outputs/publication/fig05_runtime_scaling_manifest.json
  outputs/publication/fig05_runtime_scaling.csv

No universal performance claims. Local environment receipt only.
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jaxfne as jtfne

from _figure_common import (
    ensure_publication_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
    truth_gates,
)


SOURCE_FILES = [
    "scripts/publication/fig05_runtime_scaling.py",
    "scripts/publication/_figure_common.py",
    "scripts/benchmark_jaxfne.py",
]

TITLE = "jaxfne runtime scaling receipt (CPU-safe local benchmark)"

SCOPE_STATUS = (
    "Local environment receipt only; no GPU requirement; no speedup or superiority claims; "
    "proxy readouts; durations kept small for CI/local smoke"
)

# Small CPU-safe grid for publication smoke.
N_NEURONS_GRID = (8, 12, 24)
N_CONTACTS_GRID = (4, 16)
SEEDS_GRID = (0, 1)
DURATION_MS = 5.0
DT_MS = 0.1
PROBE_MODES = ["spikes", "V_m", "CSD", "LFP"]

_dirs = ensure_publication_dirs()
FIGURE_PATH = _dirs["figures"] / "fig05_runtime_scaling.png"
CSV_PATH = _dirs["outputs"] / "fig05_runtime_scaling.csv"


def hardware_receipt() -> dict:
    """Collect runtime/hardware metadata for benchmark receipt."""
    devices = jax.devices()
    device_rows = [
        {"device_kind": str(dev.device_kind), "device_id": int(dev.id)}
        for dev in devices
    ]
    try:
        import jaxlib

        jaxlib_version = str(jaxlib.__version__)
    except Exception:
        jaxlib_version = "unknown"

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "jax_version": str(jax.__version__),
        "jaxlib_version": jaxlib_version,
        "default_backend": str(jax.default_backend()),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "device_count": len(devices),
        "devices": device_rows,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "local_environment_receipt_only": True,
        "performance_claims_allowed": False,
    }


def _build_config(n_neurons: int, n_contacts: int) -> jtfne.Configuration:
    return (
        jtfne.configuration()
        .network(
            name="V1",
            kind="cortical_column",
            n=n_neurons,
            cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(
            name="laminar_probe",
            modes=list(PROBE_MODES),
            n_contacts=n_contacts,
        )
    )


def _finite_status(signals, readout: dict) -> bool:
    checks = [jnp.all(jnp.isfinite(signals.V_m)), jnp.all(jnp.isfinite(signals.spikes))]
    for key in ("CSD", "LFP", "csd_proxy", "lfp_proxy"):
        if key in readout and hasattr(readout[key], "shape"):
            checks.append(jnp.all(jnp.isfinite(readout[key])))
    return bool(all(checks))


def run_scaling_grid() -> list[dict]:
    """Execute the publication scaling grid and return per-run receipts."""
    hw = hardware_receipt()
    rows: list[dict] = []
    run_idx = 0
    for n_neurons in N_NEURONS_GRID:
        for n_contacts in N_CONTACTS_GRID:
            for seed in SEEDS_GRID:
                run_idx += 1
                cfg = _build_config(n_neurons, n_contacts)
                model = jtfne.construct(cfg)
                # Discarded warmup per (N, contacts) before timed seed runs.
                if seed == SEEDS_GRID[0]:
                    warmup = jtfne.simulation(
                        duration_ms=DURATION_MS,
                        dt_ms=DT_MS,
                        seed=seed,
                    )
                    _ = model.simulate(warmup)
                sim = jtfne.simulation(
                    duration_ms=DURATION_MS,
                    dt_ms=DT_MS,
                    seed=seed,
                )
                t0 = time.perf_counter()
                signals = model.simulate(sim)
                jax.block_until_ready(signals.V_m)
                jax.block_until_ready(signals.spikes)
                elapsed_simulate_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                readout = model.probe(signals, modes=list(PROBE_MODES))
                for key in ("CSD", "LFP", "csd_proxy", "lfp_proxy", "V_m", "spikes"):
                    if key in readout and hasattr(readout[key], "shape"):
                        jax.block_until_ready(readout[key])
                elapsed_probe_s = time.perf_counter() - t1

                csd = readout.get("CSD")
                n_contacts_observed = int(csd.shape[1]) if csd is not None else n_contacts

                rows.append(
                    {
                        "run_id": run_idx,
                        "n_neurons": n_neurons,
                        "n_contacts_requested": n_contacts,
                        "n_contacts_observed": n_contacts_observed,
                        "seed": seed,
                        "duration_ms": DURATION_MS,
                        "dt_ms": DT_MS,
                        "dtype": str(signals.V_m.dtype),
                        "elapsed_simulate_s": round(elapsed_simulate_s, 6),
                        "elapsed_probe_s": round(elapsed_probe_s, 6),
                        "elapsed_s": round(elapsed_simulate_s + elapsed_probe_s, 6),
                        "finite_outputs": _finite_status(signals, readout),
                        "n_timesteps": int(signals.V_m.shape[0]),
                        "n_readout_modes": len(PROBE_MODES),
                        "backend": hw["default_backend"],
                        "platform": hw["platform"],
                        "python_version": hw["python_version"],
                        "jax_version": hw["jax_version"],
                        "jaxlib_version": hw["jaxlib_version"],
                    }
                )
    return rows


def write_csv(rows: list[dict]) -> None:
    """Write strict CSV receipt."""
    if not rows:
        raise RuntimeError("No benchmark rows to write")
    fieldnames = list(rows[0].keys())
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _contacts_series_label(rows: list[dict], n_contacts_requested: int) -> str:
    observed = {
        r["n_contacts_observed"]
        for r in rows
        if r["n_contacts_requested"] == n_contacts_requested
    }
    if len(observed) == 1 and next(iter(observed)) == n_contacts_requested:
        return f"contacts={n_contacts_requested}"
    obs_str = ",".join(str(v) for v in sorted(observed))
    return f"contacts_req={n_contacts_requested}_obs={obs_str}"


def _mean_elapsed(rows: list[dict], n_neurons: int, n_contacts: int) -> float:
    subset = [
        r["elapsed_s"]
        for r in rows
        if r["n_neurons"] == n_neurons and r["n_contacts_requested"] == n_contacts
    ]
    return float(np.mean(subset))


def draw_figure(rows: list[dict], hw: dict) -> None:
    """Render publication scaling figure from benchmark rows."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Panel A - mean elapsed vs N, grouped by contacts (seed-averaged).
    ax = axes[0]
    for n_contacts in N_CONTACTS_GRID:
        ys = [_mean_elapsed(rows, n, n_contacts) for n in N_NEURONS_GRID]
        ax.plot(
            list(N_NEURONS_GRID),
            ys,
            marker="o",
            linewidth=1.5,
            label=_contacts_series_label(rows, n_contacts),
        )
    ax.set_title("Panel A - simulate+probe elapsed (seed mean)", fontsize=10, fontweight="bold")
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("elapsed (s)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)

    # Panel B - seed spread at mid configuration.
    ax = axes[1]
    mid_n = 12 if 12 in N_NEURONS_GRID else N_NEURONS_GRID[1]
    for n_contacts in N_CONTACTS_GRID:
        subset = [
            r
            for r in rows
            if r["n_neurons"] == mid_n and r["n_contacts_requested"] == n_contacts
        ]
        xs = [r["seed"] for r in subset]
        ys = [r["elapsed_s"] for r in subset]
        ax.plot(xs, ys, marker="s", linewidth=1.2, label=_contacts_series_label(rows, n_contacts))
    ax.set_title(f"Panel B - seed spread at n={mid_n}", fontsize=10, fontweight="bold")
    ax.set_xlabel("seed")
    ax.set_ylabel("elapsed (s)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)

    fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "import jaxfne as jtfne  |  local_environment_receipt_only",
        ha="center",
        fontsize=8,
        family="monospace",
        color="#333333",
    )

    receipt_lines = [
        SCOPE_STATUS,
        f"platform: {hw['platform']}",
        f"python: {hw['python_version']}  jax: {hw['jax_version']}  backend: {hw['default_backend']}",
        f"grid: N={list(N_NEURONS_GRID)} contacts={list(N_CONTACTS_GRID)} seeds={list(SEEDS_GRID)}",
        f"duration_ms={DURATION_MS} dt_ms={DT_MS} runs={len(rows)}",
        "physical_amplitude_claim_allowed: false  |  performance_claims_allowed: false",
    ]
    fig.text(
        0.02,
        -0.08,
        "\n".join(receipt_lines),
        ha="left",
        va="top",
        fontsize=7,
        family="monospace",
        color="#444444",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    hw = hardware_receipt()
    rows = run_scaling_grid()
    if not all(r["finite_outputs"] for r in rows):
        raise RuntimeError("Non-finite outputs detected in scaling grid")

    write_csv(rows)
    draw_figure(rows, hw)

    csv_rel = str(CSV_PATH.relative_to(repo_root()))
    manifest = save_figure_manifest(
        figure_id="fig05",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/publication/fig05_runtime_scaling.py",
            "claim_boundary": "benchmark_receipt_required",
            "local_environment_receipt_only": True,
            "performance_claims_allowed": False,
            "warmup_policy": "one discarded simulate per (n_neurons, n_contacts) before seed=0 timed run",
            "timed_phases": ["simulate", "probe"],
            "timing_excludes": [
                "model_construct",
                "warmup_simulate",
                "csv_write",
                "figure_render",
                "manifest_write",
            ],
            "jax_timing_policy": "jax.block_until_ready on simulate and probe arrays before stopping timers",
            "n_contacts_observed_values": sorted(
                {r["n_contacts_observed"] for r in rows}
            ),
            "benchmark_grid": {
                "n_neurons": list(N_NEURONS_GRID),
                "n_contacts": list(N_CONTACTS_GRID),
                "seeds": list(SEEDS_GRID),
                "duration_ms": DURATION_MS,
                "dt_ms": DT_MS,
                "probe_modes": list(PROBE_MODES),
            },
            "hardware_receipt": hw,
            "csv_path": csv_rel,
            "csv_sha256": sha256_file(CSV_PATH),
            "truth_gates": truth_gates(),
        },
    )

    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/publication/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"wrote: {csv_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    print(f"runs: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
