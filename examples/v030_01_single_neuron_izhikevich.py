#!/usr/bin/env python3
"""v0.3.1 single-neuron Izhikevich tutorial smoke.

This script is a Colab-compatible tutorial evidence generator built on the
stable jaxfne==0.2.30 public toolbox. It writes strict JSON outputs and PNG
figures under outputs/v030_01_single_neuron_izhikevich/.

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

OUT = Path("outputs/v030_01_single_neuron_izhikevich")


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


def finite_all(*arrays: Any) -> bool:
    return bool(all(jnp.all(jnp.isfinite(jnp.asarray(a))) for a in arrays if a is not None))


def main() -> None:
    seed = 0
    duration_ms = 1000.0
    dt_ms = 0.1
    run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=seed)

    cfg = (
        jtfne.configuration()
        .network(name="v030_01_single_neuron", kind="isolated_neuron", n=1, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="single_neuron_laminar_proxy", modes=["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
        .update_metadata(
            dx_mm=0.010,
            dy_mm=0.010,
            dz_mm=0.010,
            geometry_mode="declared_tutorial_metadata_not_solved_3d_grid",
            tutorial_id="v0.3.1",
        )
    )

    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=run)
    signals = model.simulate(sim)

    field = signals.field
    if field is None:
        raise RuntimeError("Expected laminar proxy field output because record_fields=True")

    readouts = {
        "spk": spk_probe(signals.spikes),
        "vm": vm_probe(signals.V_m),
        "source": source_probe(signals.sources),
        "lfp_proxy": lfp_proxy_probe(field.lfp_proxy),
        "csd_proxy": csd_proxy_probe(field.csd_proxy),
        "eeg_proxy": eeg_proxy_probe(field.lfp_proxy),
        "meg_proxy": meg_proxy_probe(field.lfp_proxy),
        "emm_proxy": emm_proxy_probe(jnp.mean(jnp.abs(field.lfp_proxy), axis=1)),
    }

    spike_rate_hz = float(jnp.mean(signals.spikes) * (1000.0 / dt_ms))
    metrics = {
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
        "dtype": str(signals.V_m.dtype),
        "n_steps": int(signals.time_ms.shape[0]),
        "n_neurons": int(signals.V_m.shape[1]),
        "mean_firing_rate_hz": spike_rate_hz,
        "spike_count_total": float(jnp.sum(signals.spikes)),
        "Vm_mean": float(jnp.mean(signals.V_m)),
        "Vm_min": float(jnp.min(signals.V_m)),
        "Vm_max": float(jnp.max(signals.V_m)),
        "finite_all_core_arrays": finite_all(signals.V_m, signals.spikes, signals.sources, field.lfp_proxy, field.csd_proxy),
    }

    manifest = model.manifest(signals=signals)
    validation_report = {
        "tutorial_id": "v0.3.1",
        "truth_mode_gate": manifest.get("truth_mode") == "truth_safe_unverified",
        "claim_level_gate": manifest.get("claim_level") == "computational_scaffold",
        "field_solver_gate": manifest.get("field_solver_status") == "laminar_proxy_no_pde",
        "physical_amplitude_gate": manifest.get("physical_amplitude_claim_allowed") is False,
        "duration_gate": duration_ms >= 1000.0,
        "dt_gate": dt_ms == 0.1,
        "dtype_gate": str(signals.V_m.dtype) == "float32",
        "firing_rate_gate_2_25_hz": 2.0 <= spike_rate_hz <= 25.0,
        "finite_gate": metrics["finite_all_core_arrays"],
        "json_safe_gate": True,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "probe_report.json", {k: v.report for k, v in readouts.items()})
    write_json(OUT / "metrics.json", metrics)
    write_json(OUT / "validation_report.json", validation_report)

    figure_paths: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = OUT / "figures"
        fig_dir.mkdir(exist_ok=True)
        t = jnp.asarray(signals.time_ms)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, signals.V_m[:, 0])
        ax.set_title("v0.3.1 single-neuron voltage-like trace")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("V_m / native voltage-like state")
        p = fig_dir / "voltage_trace.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 3))
        spike_t = t[signals.spikes[:, 0] > 0.5]
        ax.vlines(spike_t, 0, 1)
        ax.set_title("v0.3.1 spike raster proxy")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("single unit")
        p = fig_dir / "spike_raster.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(field.csd_proxy.T, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, label="proxy units")
        ax.set_title("v0.3.1 CSD-proxy heatmap")
        ax.set_xlabel("time index")
        ax.set_ylabel("laminar contact")
        p = fig_dir / "csd_proxy_heatmap.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(p)
    except Exception as exc:  # pragma: no cover - tutorial fallback path
        write_json(OUT / "figure_generation_warning.json", {"plotting_error": repr(exc)})

    hashes = {p.name: sha256_file(p) for p in [OUT / "manifest.json", OUT / "probe_report.json", OUT / "metrics.json", OUT / "validation_report.json"]}
    for p in figure_paths:
        hashes[str(p.relative_to(OUT))] = sha256_file(p)
    write_json(OUT / "asset_hashes.json", hashes)

    print(json.dumps({"output_dir": str(OUT), "metrics": metrics, "validation_report": validation_report}, indent=2))


if __name__ == "__main__":
    main()
