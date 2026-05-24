#!/usr/bin/env python3
"""v0.3.2 single-neuron Izhikevich parameter sweep tutorial.

This script sweeps reduced Izhikevich a/b/c/d/drive settings over a small grid
and writes JSON/PNG evidence under outputs/v030_02_single_neuron_parameter_sweep/.

Uses the public jtfne.with_emitter_parameters() helper for clean parameter
substitution without internal dataclass access.

Claim gates: computational_scaffold, laminar_proxy_no_pde, no physical amplitude
claim, no biological mechanism claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import jaxfne as jtfne

OUT = Path("outputs/v030_02_single_neuron_parameter_sweep")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, allow_nan=False, indent=2, sort_keys=True)


def main() -> None:
    seed = 0
    duration_ms = 1000.0
    dt_ms = 0.1
    run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=seed)

    cfg = (
        jtfne.configuration()
        .network(name="v030_02_parameter_sweep", kind="isolated_neuron", n=1, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="single_neuron_sweep_probe", modes=["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
        .update_metadata(
            dx_mm=0.010,
            dy_mm=0.010,
            dz_mm=0.010,
            geometry_mode="declared_tutorial_metadata_not_solved_3d_grid",
            tutorial_id="v0.3.2",
        )
    )
    base = jtfne.construct(cfg)

    # Sweep conditions using the public with_emitter_parameters helper.
    # Each condition overrides only the named parameters; all others are
    # copied from the base model.  Claim gates are preserved unchanged.
    conditions = [
        {"name": "baseline", "a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0, "drive_scale": 1.0},
        {"name": "faster_recovery", "a": 0.05, "b": 0.20, "c": -65.0, "d": 8.0, "drive_scale": 1.0},
        {"name": "lower_reset", "a": 0.02, "b": 0.20, "c": -70.0, "d": 8.0, "drive_scale": 1.0},
        {"name": "higher_adaptation", "a": 0.02, "b": 0.20, "c": -65.0, "d": 12.0, "drive_scale": 1.0},
        {"name": "stronger_drive", "a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0, "drive_scale": 1.2},
        {"name": "weaker_drive", "a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0, "drive_scale": 0.8},
    ]

    rows = []
    traces = {}
    for cond in conditions:
        model = jtfne.with_emitter_parameters(
            base,
            a=cond["a"],
            b=cond["b"],
            c=cond["c"],
            d=cond["d"],
            drive_scale=cond["drive_scale"],
        )
        signals = model.simulate(jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=run))
        rate = float(jnp.mean(signals.spikes) * (1000.0 / dt_ms))
        row = {
            **{k: v for k, v in cond.items()},
            "mean_firing_rate_hz": rate,
            "spike_count_total": float(jnp.sum(signals.spikes)),
            "Vm_mean": float(jnp.mean(signals.V_m)),
            "Vm_min": float(jnp.min(signals.V_m)),
            "Vm_max": float(jnp.max(signals.V_m)),
            "finite": bool(
                jnp.all(jnp.isfinite(signals.V_m)) and jnp.all(jnp.isfinite(signals.spikes))
            ),
            "gate_2_25_hz": 2.0 <= rate <= 25.0,
        }
        rows.append(row)
        traces[cond["name"]] = [float(x) for x in signals.V_m[:, 0][::10]]

    manifest = base.manifest()
    validation_report = {
        "tutorial_id": "v0.3.2",
        "truth_mode_gate": manifest.get("truth_mode") == "truth_safe_unverified",
        "claim_level_gate": manifest.get("claim_level") == "computational_scaffold",
        "duration_gate": duration_ms >= 1000.0,
        "dt_gate": dt_ms == 0.1,
        "dtype_gate": run.actual_dtype == "float32",
        "all_conditions_finite": all(r["finite"] for r in rows),
        "all_conditions_rate_2_25_hz": all(r["gate_2_25_hz"] for r in rows),
        "json_safe_gate": True,
        "with_emitter_parameters_api": "public_helper_used",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "metrics.json", {"conditions": rows})
    write_json(OUT / "voltage_traces_decimated.json", traces)
    write_json(OUT / "validation_report.json", validation_report)

    fig_paths: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = OUT / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([r["name"] for r in rows], [r["mean_firing_rate_hz"] for r in rows])
        ax.axhline(2.0, linestyle="--", linewidth=1, color="gray", label="2 Hz gate")
        ax.axhline(25.0, linestyle="--", linewidth=1, color="gray", label="25 Hz gate")
        ax.set_ylabel("mean firing rate (Hz)")
        ax.set_title("v0.3.2 Izhikevich parameter sweep — proxy readout only")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        p = fig_dir / "firing_rate_sweep.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        for name, trace in traces.items():
            ax.plot(range(len(trace)), trace, label=name, linewidth=0.8)
        ax.set_title("v0.3.2 voltage-like traces, decimated (proxy readout)")
        ax.set_xlabel("decimated time index")
        ax.set_ylabel("V_m / native voltage-like state")
        ax.legend(fontsize=7)
        p = fig_dir / "voltage_sweep_decimated.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(p)
    except Exception as exc:  # pragma: no cover
        write_json(OUT / "figure_generation_warning.json", {"plotting_error": repr(exc)})

    json_files = [
        OUT / "manifest.json",
        OUT / "metrics.json",
        OUT / "voltage_traces_decimated.json",
        OUT / "validation_report.json",
    ]
    hashes = {p.name: sha256_file(p) for p in json_files}
    for p in fig_paths:
        hashes[str(p.relative_to(OUT))] = sha256_file(p)
    write_json(OUT / "asset_hashes.json", hashes)

    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "validation_report": validation_report,
                "conditions": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
