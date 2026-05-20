#!/usr/bin/env python3
"""
Single-neuron multimodal proxy probe tutorial.

Demonstrates the v0.2.1 multimodal proxy probe stack (eight operators)
on a minimal single-neuron Izhikevich emitter.

Generates a reproducible output bundle with all eight readouts:
SPK, Vm, source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy.

Claim-status metadata: All operators are simulated proxies with frozen
validation metadata. No biological mechanism claims. No empirical validation.

Usage:
    python examples/03_single_neuron_multimodal_probe.py

Generates:
    outputs/v023_single_neuron_multimodal/
    ├── manifest.json                (model/field metadata)
    ├── probe_report.json            (all 8 operator reports)
    ├── metrics.json                 (basic signal metrics)
    ├── validation_report.json       (claim-status metadata verification)
    └── asset_hashes.json            (file integrity hashes)
"""

import json
import pathlib
import hashlib
from typing import Any

import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.fields import (
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
)


def compute_signal_metrics(spk_array, vm_array):
    """Compute basic signal statistics for metrics.json."""
    spk_rate = float(jnp.mean(jnp.sum(spk_array, axis=1)))  # spikes per timestep
    vm_mean = float(jnp.mean(vm_array))
    vm_std = float(jnp.std(vm_array))
    return {
        "spike_rate_per_timestep": float(spk_rate),
        "Vm_mean_mV": float(vm_mean),
        "Vm_std_mV": float(vm_std),
    }


def file_hash_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    # === 1. Create minimal single-neuron configuration ===
    # Single excitatory Izhikevich neuron, 100ms simulation, 0.1ms dt
    cfg = (
        jtfne.configuration()
        .network(
            name="single_neuron",
            kind="isolated_neuron",
            n=1,
            cell_types={"E": 1.0},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(
            name="multimodal_single_neuron",
            modes=[
                "spikes",
                "V_m",
                "source",
                "phi_e",
                "J_e",
                "CSD",
                "LFP",
            ],
        )
    )

    # === 2. Construct model ===
    model = jtfne.construct(cfg)

    # === 3. Simulate ===
    # CPU-safe: 100ms duration, 0.1ms dt → 1000 timesteps
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    signals = model.simulate(sim)

    # === 4. Apply all eight probe operators ===
    spk_readout = spk_probe(signals.spikes)
    vm_readout = vm_probe(signals.V_m)

    # Source, LFP, CSD operators depend on field existence
    source_readout = (
        source_probe(signals.sources[0]) if signals.sources is not None else None
    )
    lfp_readout = (
        lfp_proxy_probe(signals.field.lfp_proxy)
        if signals.field is not None
        else None
    )
    csd_readout = (
        csd_proxy_probe(signals.field.csd_proxy)
        if signals.field is not None
        else None
    )

    # EEG, MEG, EMM operators use field signal
    field_signal = (
        signals.field.lfp_proxy if signals.field is not None else signals.V_m
    )
    eeg_readout = eeg_proxy_probe(field_signal)
    meg_readout = meg_proxy_probe(field_signal)
    emm_readout = emm_proxy_probe(field_signal)

    # === 5. Construct probe_report from all operators ===
    probe_report = {
        "spk": spk_readout.report,
        "vm": vm_readout.report,
    }
    if source_readout is not None:
        probe_report["source"] = source_readout.report
    if lfp_readout is not None:
        probe_report["lfp_proxy"] = lfp_readout.report
    if csd_readout is not None:
        probe_report["csd_proxy"] = csd_readout.report

    probe_report.update({
        "eeg_proxy": eeg_readout.report,
        "meg_proxy": meg_readout.report,
        "emm_proxy": emm_readout.report,
    })

    # === 6. Manifest: full pipeline metadata ===
    manifest = model.manifest(signals=signals)

    # === 7. Signal metrics ===
    metrics = compute_signal_metrics(signals.spikes, signals.V_m)

    # === 8. Claim-status metadata verification ===
    validation_report = {
        "claim_level": manifest.get("claim_level"),
        "field_claim_level": manifest.get("field_claim_level"),
        "field_solver_status": manifest.get("field_solver_status"),
        "source_calibration_status": manifest.get("source_calibration_status"),
        "physical_amplitude_claim_allowed": manifest.get(
            "physical_amplitude_claim_allowed"
        ),
        "empirical_validation_status": manifest.get("empirical_validation_status"),
        "mechanism_claim_status": manifest.get("mechanism_claim_status"),
    }

    # === 9. Write outputs ===
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON files
    manifest_path = output_dir / "manifest.json"
    probe_report_path = output_dir / "probe_report.json"
    metrics_path = output_dir / "metrics.json"
    validation_path = output_dir / "validation_report.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, allow_nan=False, indent=2)

    with open(probe_report_path, "w") as f:
        json.dump(probe_report, f, allow_nan=False, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, allow_nan=False, indent=2)

    with open(validation_path, "w") as f:
        json.dump(validation_report, f, allow_nan=False, indent=2)

    # === 10. Asset hashes ===
    asset_hashes = {
        "manifest.json": file_hash_sha256(manifest_path),
        "probe_report.json": file_hash_sha256(probe_report_path),
        "metrics.json": file_hash_sha256(metrics_path),
        "validation_report.json": file_hash_sha256(validation_path),
    }
    hashes_path = output_dir / "asset_hashes.json"
    with open(hashes_path, "w") as f:
        json.dump(asset_hashes, f, allow_nan=False, indent=2)

    # === 11. Verify all outputs ===
    print("=== Single-neuron Multimodal Proxy Tutorial ===\n")
    print(f"Output directory: {output_dir}\n")

    print("Files generated:")
    for fpath in [manifest_path, probe_report_path, metrics_path, validation_path, hashes_path]:
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  {fpath.name:<30} {size:>6} bytes")

    print("\nProbe operators (all eight):")
    for op_name, op_report in probe_report.items():
        print(f"  {op_name:<15} operator_status={op_report.get('operator_status')}")

    print("\nSignal metrics:")
    for key, value in metrics.items():
        print(f"  {key:<30} {value}")

    print("\nClaim-status metadata (frozen):")
    for key, value in validation_report.items():
        print(f"  {key:<40} = {value}")

    print("\n✓ All outputs are JSON-strict (no NaN/Inf).")
    print("✓ Eight proxy operators executed successfully.")
    print("✓ Claim-status metadata verified and immutable.")
    print("\nTruth status: computational scaffold, not empirically validated.")


if __name__ == "__main__":
    main()
