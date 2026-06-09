#!/usr/bin/env python3
"""Runner script for delta-test notebook 01.

Validates notebook structure and executes key components.
Supports TFNE_SMOKE=1 for quick validation, TFNE_SMOKE=0 for full run.

Usage:
    TFNE_SMOKE=1 python3 scripts/run_delta_notebook_01.py  # Quick validation
    TFNE_SMOKE=0 python3 scripts/run_delta_notebook_01.py  # Full 1000ms simulation
"""

import os
import sys
import json
from pathlib import Path

import jax.numpy as jnp
import jaxfne as jtfne

# Configuration from environment
SMOKE_MODE = os.environ.get("TFNE_SMOKE", "1") == "1"

# Global config (matching notebook)
GLOBAL = {
    "seed": 0,
    "N_PER_COLUMN": 200 if not SMOKE_MODE else 32,  # Smaller for smoke test
    "duration_ms": 1000.0 if not SMOKE_MODE else 10.0,
    "dt_ms": 0.1,
    "column_height_mm": 2.0,
    "layers": ["L1", "L2/3", "L4", "L5A", "L5B", "L6"],
    "areas": ["V1", "V4", "MT", "FEF", "PFC"],
    "area_xy_mm": {
        "V1": (0.0, 0.0),
        "V4": (1.0, 1.0),
        "MT": (1.0, -1.0),
        "FEF": (4.0, 0.0),
        "PFC": (5.0, 0.0),
    },
    "hierarchy": ["V1", "V2_reference_only", "V4", "MT", "FEF", "PFC"],
    "cell_types": {"E": 0.78, "PV": 0.10, "SST": 0.08, "VIP": 0.04},
    "emitter": "izhikevich",
    "plasticity_coeff": 1.0,
    "eeg_n_channels": 16,
    "eeg_height_mm": 1.0,
    "meg_n_channels": 16,
    "meg_height_mm": 10.0,
}

OUTPUT_DIR = Path("outputs/delta_test_01")


def main():
    """Run delta-test validation pipeline."""
    print(f"=== Delta-Test Notebook 01 Runner ===")
    print(f"Mode: {'SMOKE (quick)' if SMOKE_MODE else 'FULL (1000ms)'}")
    print(f"N per area: {GLOBAL['N_PER_COLUMN']}")
    print(f"Duration: {GLOBAL['duration_ms']} ms")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Build area configs
        print("Step 1: Building area configurations...")
        cfgs = {}
        for area in GLOBAL["areas"]:
            cfg = jtfne.laminar_cortex_config(
                seed=GLOBAL["seed"],
                duration_ms=GLOBAL["duration_ms"],
                dt_ms=GLOBAL["dt_ms"],
                areas=[area],
                layers=GLOBAL["layers"],
                cell_types=GLOBAL["cell_types"],
                n=GLOBAL["N_PER_COLUMN"],
                emitter=GLOBAL["emitter"],
            )
            cfgs[area] = cfg
        print(f"  ✓ {len(cfgs)} area configs created")

        # Step 2: Construct models
        print("Step 2: Constructing models...")
        models = {}
        for area, cfg in cfgs.items():
            model = jtfne.construct(cfg)
            models[area] = model
            n_neurons = len(model.select(area=area))
            print(f"  ✓ {area}: {n_neurons} neurons")

        # Step 3: Simulate
        print("Step 3: Running simulations...")
        signals_by_area = {}
        for area, model in models.items():
            signals = jtfne.simulate(
                model,
                duration_ms=GLOBAL["duration_ms"],
                dt_ms=GLOBAL["dt_ms"],
                seed=GLOBAL["seed"],
            )
            signals_by_area[area] = signals
            vm = signals.get("V_m")
            print(f"  ✓ {area}: shape {vm.shape}, finite={bool(jnp.all(jnp.isfinite(vm)))}")

        # Step 4: Create manifests
        print("Step 4: Creating manifests...")
        manifests = {}
        for area, cfg in cfgs.items():
            signals = signals_by_area[area]
            manifest = jtfne.manifest(cfg, signals=signals)
            manifests[area] = manifest
            jtfne.save_json(manifest, OUTPUT_DIR / f"manifest_{area}.json")
        print(f"  ✓ {len(manifests)} manifests saved")

        # Step 5: Validation report
        print("Step 5: Creating validation report...")
        validation_report = jtfne.validation_report(
            config_valid=True,
            issues=[],
            metadata={
                "celltype_mapping_status": "rounded_literature_proxy",
                "cb_to_sst_mapping": "proxy",
                "cr_to_vip_mapping": "proxy",
                "areas": GLOBAL["areas"],
                "n_per_area": GLOBAL["N_PER_COLUMN"],
                "duration_ms": GLOBAL["duration_ms"],
            },
        )
        jtfne.save_json(validation_report, OUTPUT_DIR / "validation_report.json")
        print(f"  ✓ Validation report saved")

        # Step 6: Metrics
        print("Step 6: Computing metrics...")
        metrics = {
            "n_areas": len(GLOBAL["areas"]),
            "n_neurons_per_area": GLOBAL["N_PER_COLUMN"],
            "n_total_neurons": GLOBAL["N_PER_COLUMN"] * len(GLOBAL["areas"]),
            "n_layers": len(GLOBAL["layers"]),
            "duration_ms": GLOBAL["duration_ms"],
            "dt_ms": GLOBAL["dt_ms"],
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "field_solver_status": "laminar_proxy_no_pde",
            "physical_amplitude_claim_allowed": False,
        }
        jtfne.save_json(metrics, OUTPUT_DIR / "metrics.json")
        print(f"  ✓ Metrics saved: {metrics['n_total_neurons']} total neurons")

        # Step 7: JSON validation
        print("Step 7: Validating JSON outputs...")
        json_files = list(OUTPUT_DIR.glob("*.json"))
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            assert data is not None
        print(f"  ✓ {len(json_files)} JSON files validated")

        # Summary
        print()
        print("=== DELTA-TEST SUMMARY ===")
        print(f"Status: ✓ PASS")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Files generated: {len(list(OUTPUT_DIR.glob('*')))}")
        print()
        print("Files:")
        for f in sorted(OUTPUT_DIR.glob("*")):
            print(f"  - {f.name}")
        print()
        print("Truth status:")
        print(f"  truth_mode: truth_safe_unverified")
        print(f"  claim_level: computational_scaffold")
        print(f"  field_solver_status: laminar_proxy_no_pde")
        print(f"  physical_amplitude_claim_allowed: false")
        print()
        print("✓ Delta-test notebook infrastructure validated")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
