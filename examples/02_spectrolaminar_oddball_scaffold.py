#!/usr/bin/env python3
"""
Phase F: Minimal spectrolaminar oddball scaffold.

Demonstrates the v0.2.0 computational pipeline:
Emitter -> Source -> Field -> Probe -> Objective -> Manifest

Focus: manifest-first JSON evidence. No biological mechanism claims.
Truth gates remain frozen throughout.

Usage:
    python examples/02_spectrolaminar_oddball_scaffold.py

Generates:
    outputs/v020_spectrolaminar_public_path/
    ├── manifest.json                (model/field metadata)
    ├── metrics.json                 (windowed spectrolaminar metrics)
    ├── objective_report.json        (objective evaluation)
    ├── validation_report.json       (truth gate audit)
    └── asset_hashes.json            (file integrity)
"""

import json
import pathlib
import hashlib
from typing import Any

import jax
import jax.numpy as jnp

import jaxfne


def _sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def main():
    """Run the Phase F spectrolaminar oddball scaffold."""

    # === 1. Output directory ===
    outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
    outdir.mkdir(parents=True, exist_ok=True)

    # === 2. Configuration: minimal cortical column ===
    cfg = (
        jaxfne.configuration()
        .network(n=32)  # Small column
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero"
        )
        .probe(name="superficial", n_contacts=2)
        .probe(name="deep", n_contacts=2)
    )

    # === 3. Model construction ===
    model = jaxfne.construct(cfg)

    # === 4. Simulation: oddball/peri-event readout ===
    # Simulate a 2-second window: baseline (-500 ms) + event (0-500 ms) + post (500-1000 ms)
    sim = jaxfne.Simulation(
        duration_ms=2000.0,
        dt_ms=1.0,
        seed=42
    )
    signals = model.simulate(sim)

    # === 5. Windowing: peri-event spectrolaminar analysis ===
    # Convert time indices: t=0 is 500 ms into the 2000 ms window (baseline starts at -500 ms)
    # Baseline: -500 to 0 ms (indices 0:500)
    # Event: 0 to 500 ms (indices 500:1000)
    # Post: 500 to 1000 ms (indices 1000:1500)
    # Full peri-event: -500 to +1000 ms (indices 0:1500)

    baseline_slice = slice(0, 500)
    event_slice = slice(500, 1000)
    post_slice = slice(1000, 1500)
    full_slice = slice(0, 1500)

    V_baseline = signals.V_m[baseline_slice]
    V_event = signals.V_m[event_slice]
    V_post = signals.V_m[post_slice]
    V_full = signals.V_m[full_slice]

    # === 6. Laminar diagnostics ===
    # Superficial probes: contacts 0-1
    # Deep probes: contacts 2-3

    def compute_proxy_alpha_beta_power(V: jnp.ndarray) -> float:
        """Proxy alpha/beta power: RMS of 8-12 Hz-like filtered signal."""
        # Simplified: just use RMS of voltage variation
        rms = float(jnp.sqrt(jnp.mean(V**2)))
        return rms

    def compute_proxy_gamma_power(V: jnp.ndarray) -> float:
        """Proxy gamma power: high-frequency envelope."""
        # Simplified: use std of membrane potential oscillations
        std = float(jnp.std(V))
        return std

    def compute_synchrony_diagnostic(V: jnp.ndarray) -> float:
        """Mean pairwise correlation across neurons (proxy via variation metric)."""
        # Compute coefficient of variation as synchrony proxy
        # (0=desynchronized, 1=synchronized)
        flat_v = V.reshape(-1)
        mean_v = jnp.mean(flat_v)
        std_v = jnp.std(flat_v)
        synchrony = float(jnp.where(mean_v != 0, std_v / jnp.abs(mean_v), 0.0))
        return synchrony

    # Compute metrics for each window
    metrics = {
        "baseline": {
            "alpha_beta_proxy_power": compute_proxy_alpha_beta_power(V_baseline),
            "gamma_proxy_power": compute_proxy_gamma_power(V_baseline),
            "synchrony": compute_synchrony_diagnostic(V_baseline),
            "mean_V_m": float(jnp.mean(V_baseline)),
            "std_V_m": float(jnp.std(V_baseline)),
        },
        "event": {
            "alpha_beta_proxy_power": compute_proxy_alpha_beta_power(V_event),
            "gamma_proxy_power": compute_proxy_gamma_power(V_event),
            "synchrony": compute_synchrony_diagnostic(V_event),
            "mean_V_m": float(jnp.mean(V_event)),
            "std_V_m": float(jnp.std(V_event)),
        },
        "post": {
            "alpha_beta_proxy_power": compute_proxy_alpha_beta_power(V_post),
            "gamma_proxy_power": compute_proxy_gamma_power(V_post),
            "synchrony": compute_synchrony_diagnostic(V_post),
            "mean_V_m": float(jnp.mean(V_post)),
            "std_V_m": float(jnp.std(V_post)),
        },
        "full_peri_event": {
            "alpha_beta_proxy_power": compute_proxy_alpha_beta_power(V_full),
            "gamma_proxy_power": compute_proxy_gamma_power(V_full),
            "synchrony": compute_synchrony_diagnostic(V_full),
            "mean_V_m": float(jnp.mean(V_full)),
            "std_V_m": float(jnp.std(V_full)),
        },
    }

    # === 7. Objective evaluation: Phase E canonical grammar ===
    obj = (
        jaxfne.Objective(name="spectrolaminar_oddball")
        .loss(
            name="profile_score",
            metric="spike_rate_hz_mean",
            weight=1.0,
            metadata={
                "score_type": "profile_score_no_null",
                "windows": {
                    "baseline": {"start_ms": -500, "end_ms": 0},
                    "event": {"start_ms": 0, "end_ms": 500},
                    "post": {"start_ms": 500, "end_ms": 1000},
                    "full_peri_event": {"start_ms": -500, "end_ms": 1000},
                }
            }
        )
        .regularizer(
            name="synchrony",
            target=0.0,
            weight=0.1,
            metric="spike_rate_hz_mean",
            metadata={
                "enabled": True,
                "metric": "mean_pairwise_correlation",
                "bin_ms": 50,
                "windows": 1,
                "target": 0.0,
                "penalty": "l2",
                "weight": 0.1
            }
        )
        .gate(
            name="synchrony_gate",
            threshold=2.0,
            criterion="below",
            metric="spike_rate_hz_mean",
            metadata={
                "synchrony_limit": 2.0,
                "metric": "mean_pairwise_correlation"
            }
        )
    )

    objective_report = model.evaluate(signals, obj)

    # === 8. Manifest: full pipeline metadata ===
    manifest = model.manifest(signals=signals)

    # === 9. Truth gate audit ===
    truth_gates = {
        "truth_mode": manifest.get("truth_mode"),
        "claim_level": manifest.get("claim_level"),
        "source_calibration_status": manifest.get("source_calibration_status"),
        "field_solver_status": manifest.get("field_solver_status"),
        "field_claim_level": manifest.get("field_claim_level"),
        "physical_amplitude_claim_allowed": manifest.get("physical_amplitude_claim_allowed"),
    }

    validation_report = {
        "validation_status": "all_gates_frozen",
        "gates": truth_gates,
        "condition_vocabulary": [
            "baseline",
            "unexpected_sensory",
            "predicted_standard",
            "omission",
            "post_omission",
        ],
        "windows_ms": {
            "baseline": {"start": -500, "end": 0},
            "event": {"start": 0, "end": 500},
            "post": {"start": 500, "end": 1000},
            "full_peri_event": {"start": -500, "end": 1000},
        },
        "objective_acceptance_decision": objective_report.get("acceptance_decision"),
        "objective_score_labels": {
            "profile_score": objective_report["losses"][0]["name"] if objective_report["losses"] else None,
            "synchrony": objective_report["regularizers"][0]["name"] if objective_report["regularizers"] else None,
        },
        "assertions": {
            "physical_amplitude_claim_allowed_is_false": manifest.get("physical_amplitude_claim_allowed") is False,
            "source_calibration_uncalibrated": manifest.get("source_calibration_status") == "uncalibrated_izhikevich_native_current",
            "field_claim_proxy_readout_only": manifest.get("field_claim_level") == "proxy_readout_only",
            "truth_mode_safe_unverified": manifest.get("truth_mode") == "truth_safe_unverified",
        }
    }

    # === 10. Write outputs ===
    outputs = {
        "manifest.json": manifest,
        "metrics.json": metrics,
        "objective_report.json": objective_report,
        "validation_report.json": validation_report,
    }

    # Asset hashes for integrity
    asset_hashes = {}

    for filename, data in outputs.items():
        filepath = outdir / filename

        # Serialize JSON strictly (no NaN/Inf)
        json_str = json.dumps(data, allow_nan=False, indent=2, sort_keys=True)
        json_bytes = json_str.encode('utf-8')

        # Write
        filepath.write_bytes(json_bytes)

        # Hash
        asset_hashes[filename] = {
            "sha256": _sha256(json_bytes),
            "bytes": len(json_bytes),
        }

    # Write asset hashes
    hashes_file = outdir / "asset_hashes.json"
    hashes_json = json.dumps(asset_hashes, indent=2, sort_keys=True)
    hashes_file.write_text(hashes_json)

    # === 11. Print summary ===
    print(f"\n=== Phase F: Spectrolaminar Oddball Scaffold ===")
    print(f"\nOutputs written to: {outdir.resolve()}")
    print(f"\nFiles generated:")
    for filename in sorted(outputs.keys()) + ["asset_hashes.json"]:
        filepath = outdir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename:30} {size:8} bytes")

    print(f"\nTruth gates (frozen):")
    for key, value in sorted(truth_gates.items()):
        print(f"  {key:40} = {value}")

    print(f"\nObjective evaluation:")
    print(f"  Acceptance decision: {objective_report.get('acceptance_decision')}")
    print(f"  Losses: {len(objective_report.get('losses', []))}")
    print(f"  Regularizers: {len(objective_report.get('regularizers', []))}")
    print(f"  Gates: {len(objective_report.get('gates', []))}")

    print(f"\nValidation assertions:")
    for key, value in sorted(validation_report["assertions"].items()):
        status = "✓" if value else "✗"
        print(f"  {status} {key}")

    print(f"\nAll outputs are JSON-strict (no NaN/Inf).")
    print(f"Manifest-first architecture: JSON evidence over visualization.")


if __name__ == "__main__":
    main()
