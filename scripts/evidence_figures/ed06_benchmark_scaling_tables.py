#!/usr/bin/env python3
"""Generate Extended Data ED6: benchmark scaling tables (local CPU receipt).

CPU-safe evidence benchmark grid aligned with Fig5 axes. Local environment
receipt only — no speedup or superiority claims.

Outputs:
  figures/evidence/ed06_benchmark_scaling_tables.png
  outputs/evidence/ed06_benchmark_scaling_tables_manifest.json
  outputs/evidence/ed06_benchmark_scaling_tables_receipt.json
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jaxfne as jtfne

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
    "scripts/evidence_figures/ed06_benchmark_scaling_tables.py",
    "scripts/evidence_figures/_figure_common.py",
    "scripts/evidence_figures/fig05_runtime_scaling.py",
    "scripts/benchmark_jaxfne.py",
]

TITLE = "Extended Data 6 - benchmark scaling tables (local CPU/runtime receipt)"

SCOPE_STATUS = (
    "local CPU/runtime receipt, not a speedup claim; "
    "local_environment_receipt_only; no superiority claims; proxy readouts only"
)

N_NEURONS_GRID = (8, 12, 24)
N_CONTACTS_GRID = (4, 16)
SEEDS_GRID = (0, 1)
DURATION_MS = 5.0
DT_MS = 0.1
PROBE_MODES = ["spikes", "V_m", "CSD", "LFP"]

TIMING_POLICY = {
    "warmup": "one discarded simulate per (n_neurons, n_contacts) before seed=0 timed run",
    "block_until_ready": "jax.block_until_ready on simulate and probe arrays before timer stop",
    "phases_timed": ["construct_s", "simulate_s", "probe_s", "total_timed_s"],
}

TIMING_EXCLUSIONS = [
    "warmup_simulate",
    "csv_write",
    "json_write",
    "figure_render",
    "manifest_write",
    "mkdocs",
    "pytest",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "ed06_benchmark_scaling_tables.png"
RECEIPT_PATH = _dirs["outputs"] / "ed06_benchmark_scaling_tables_receipt.json"


def hardware_receipt() -> dict[str, Any]:
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
        "repo_sha": repo_sha(),
        "generated_at_utc": utc_now_iso(),
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


def run_benchmark_grid() -> list[dict[str, Any]]:
    hw = hardware_receipt()
    rows: list[dict[str, Any]] = []
    run_idx = 0
    warmed: set[tuple[int, int]] = set()

    for n_neurons in N_NEURONS_GRID:
        for n_contacts in N_CONTACTS_GRID:
            for seed in SEEDS_GRID:
                run_idx += 1
                key = (n_neurons, n_contacts)

                t_construct0 = time.perf_counter()
                cfg = _build_config(n_neurons, n_contacts)
                model = jtfne.construct(cfg)
                jax.block_until_ready(model) if hasattr(model, "block_until_ready") else None
                construct_s = time.perf_counter() - t_construct0

                if key not in warmed:
                    warmup = jtfne.simulation(
                        duration_ms=DURATION_MS,
                        dt_ms=DT_MS,
                        seed=SEEDS_GRID[0],
                    )
                    warmup_signals = model.simulate(warmup)
                    jax.block_until_ready(warmup_signals.V_m)
                    warmed.add(key)

                sim = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=seed)

                t_sim0 = time.perf_counter()
                signals = model.simulate(sim)
                jax.block_until_ready(signals.V_m)
                jax.block_until_ready(signals.spikes)
                simulate_s = time.perf_counter() - t_sim0

                t_probe0 = time.perf_counter()
                readout = model.probe(signals, modes=list(PROBE_MODES))
                for probe_key in ("CSD", "LFP", "csd_proxy", "lfp_proxy", "V_m", "spikes"):
                    if probe_key in readout and hasattr(readout[probe_key], "shape"):
                        jax.block_until_ready(readout[probe_key])
                probe_s = time.perf_counter() - t_probe0

                total_timed_s = construct_s + simulate_s + probe_s
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
                        "construct_s": round(construct_s, 6),
                        "simulate_s": round(simulate_s, 6),
                        "probe_s": round(probe_s, 6),
                        "total_timed_s": round(total_timed_s, 6),
                        "finite_outputs": _finite_status(signals, readout),
                        "n_timesteps": int(signals.V_m.shape[0]),
                        "backend": hw["default_backend"],
                        "jax_enable_x64": hw["jax_enable_x64"],
                    }
                )
    return rows


def _median_total(rows: list[dict], n_neurons: int, n_contacts: int) -> float:
    subset = [
        r["total_timed_s"]
        for r in rows
        if r["n_neurons"] == n_neurons and r["n_contacts_requested"] == n_contacts
    ]
    return float(np.median(subset))


def draw_figure(rows: list[dict], hw: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150)

    # Panel A - median total_timed_s heatmap (n_neurons x n_contacts).
    ax = axes[0]
    heat = np.zeros((len(N_NEURONS_GRID), len(N_CONTACTS_GRID)), dtype=np.float64)
    for i, n in enumerate(N_NEURONS_GRID):
        for j, c in enumerate(N_CONTACTS_GRID):
            heat[i, j] = _median_total(rows, n, c)
    im = ax.imshow(heat, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(N_CONTACTS_GRID)))
    ax.set_xticklabels([str(c) for c in N_CONTACTS_GRID])
    ax.set_yticks(range(len(N_NEURONS_GRID)))
    ax.set_yticklabels([str(n) for n in N_NEURONS_GRID])
    ax.set_xlabel("n_contacts")
    ax.set_ylabel("n_neurons")
    ax.set_title("Panel A - median total_timed_s (s)", fontsize=10, fontweight="bold")
    for i in range(len(N_NEURONS_GRID)):
        for j in range(len(N_CONTACTS_GRID)):
            ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B - table of per-run receipts (compact).
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Panel B - per-run timing receipt (s)", fontsize=10, fontweight="bold", pad=12)
    headers = ["N", "Z", "seed", "construct", "simulate", "probe", "total", "finite"]
    col_x = [0.02, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.78]
    y = 0.92
    for x, h in zip(col_x, headers):
        ax.text(x, y, h, fontsize=7, fontweight="bold", family="monospace", transform=ax.transAxes)
    y = 0.86
    for row in rows:
        vals = [
            str(row["n_neurons"]),
            str(row["n_contacts_requested"]),
            str(row["seed"]),
            f"{row['construct_s']:.4f}",
            f"{row['simulate_s']:.4f}",
            f"{row['probe_s']:.4f}",
            f"{row['total_timed_s']:.4f}",
            "Y" if row["finite_outputs"] else "N",
        ]
        for x, val in zip(col_x, vals):
            ax.text(x, y, val, fontsize=6.5, family="monospace", transform=ax.transAxes)
        y -= 0.055
        if y < 0.05:
            break

    fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "local CPU/runtime receipt, not a speedup claim",
        ha="center",
        fontsize=9,
        family="monospace",
        color="#C62828",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.94,
        f"backend={hw['default_backend']}  jax={hw['jax_version']}  x64={hw['jax_enable_x64']}",
        ha="center",
        fontsize=7,
        family="monospace",
        color="#333333",
    )
    fig.text(
        0.02,
        -0.02,
        SCOPE_STATUS,
        ha="left",
        fontsize=7,
        family="monospace",
        color="#444444",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    hw = hardware_receipt()
    rows = run_benchmark_grid()

    if not all(r["finite_outputs"] for r in rows):
        raise RuntimeError("Non-finite benchmark outputs detected")

    benchmark_grid = {
        "n_neurons": list(N_NEURONS_GRID),
        "n_contacts": list(N_CONTACTS_GRID),
        "seeds": list(SEEDS_GRID),
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "dtype": "float32",
        "probe_modes": list(PROBE_MODES),
    }

    receipt_body = {
        "schema_version": "jaxfne.evidence_ed06_benchmark_receipt.v0.1.0",
        "artifact": "ed06_benchmark_scaling_tables",
        "generated_at_utc": utc_now_iso(),
        "benchmark_grid": benchmark_grid,
        "rows": rows,
        "hardware_receipt": hw,
        "timing_policy": TIMING_POLICY,
        "timing_exclusions": TIMING_EXCLUSIONS,
        "finite_outputs_all": all(r["finite_outputs"] for r in rows),
        "local_environment_receipt_only": True,
        "performance_claims_allowed": False,
        "superiority_claims_allowed": False,
        "physical_amplitude_calibrated": False,
        "truth_gates": truth_gates(),
        "claim_boundary": "local_cpu_runtime_receipt_only",
    }
    write_json_strict(RECEIPT_PATH, receipt_body)

    draw_figure(rows, hw)

    root = repo_root()
    manifest = save_figure_manifest(
        figure_id="ed06",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "artifact": "ed06_benchmark_scaling_tables",
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/ed06_benchmark_scaling_tables.py",
            "claim_boundary": "local_cpu_runtime_receipt_only",
            "local_environment_receipt_only": True,
            "performance_claims_allowed": False,
            "superiority_claims_allowed": False,
            "physical_amplitude_calibrated": False,
            "benchmark_grid": benchmark_grid,
            "hardware_receipt": hw,
            "timing_policy": TIMING_POLICY,
            "timing_exclusions": TIMING_EXCLUSIONS,
            "finite_outputs_all": receipt_body["finite_outputs_all"],
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
    print(f"runs: {len(rows)} finite_all={receipt_body['finite_outputs_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
