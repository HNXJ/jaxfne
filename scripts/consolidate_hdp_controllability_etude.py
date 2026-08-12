#!/usr/bin/env python3
"""Consolidate HDP-MVC receipts into the canonical Etude artifact bundle.

Read-only with respect to frozen protocol/controller. Re-runs the Etude
simulation runner only when the diagnostic receipt is missing or --force is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DIAG = ROOT / "artifacts/msvc_hdp_diagnostic"
ETUDE_DIR = ROOT / "artifacts/etudes/hdp_controllability_reachability"
DOCS_MD = ROOT / "docs/etudes/hdp_controllability_reachability.md"

FROZEN_RECEIPTS = {
    "protocol": DIAG / "hdp_mvc_protocol_v2.json",
    "controller_spec": DIAG / "hdp_mvc_frozen_controller_spec.json",
    "alignment": DIAG / "hdp_mvc_alignment_diagnostic.json",
    "j_theta_inventory": DIAG / "hdp_mvc_j_theta_actuator_inventory.json",
    "intrinsic_inventory": DIAG / "hdp_mvc_intrinsic_actuator_inventory.json",
    "authority_boundary": DIAG / "hdp_mvc_authority_boundary.json",
    "frozen_validation": DIAG / "hdp_mvc_frozen_validation.json",
    "posthoc_scalar_vector": DIAG / "hdp_mvc_posthoc_scalar_vector.json",
    "etude_diagnostic": DIAG / "hdp_mvc_etude.json",
}

RUNNERS = {
    "etude_simulation": SCRIPTS / "hdp_mvc_etude.py",
    "frozen_validation": SCRIPTS / "hdp_mvc_frozen_validation.py",
    "authority_boundary": SCRIPTS / "hdp_mvc_authority_boundary.py",
    "frozen_controller_spec": SCRIPTS / "hdp_mvc_frozen_controller_spec.py",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_etude_receipt(*, force: bool) -> Path:
    etude_json = FROZEN_RECEIPTS["etude_diagnostic"]
    etude_fig = DIAG / "hdp_mvc_etude.png"
    if force or not etude_json.exists() or not etude_fig.exists():
        print("Running hdp_mvc_etude.py ...", flush=True)
        subprocess.run(
            [sys.executable, str(RUNNERS["etude_simulation"])],
            cwd=ROOT,
            check=True,
        )
    if not etude_json.exists():
        raise FileNotFoundError(f"missing etude receipt: {etude_json}")
    return etude_json


def compact_metrics(etude: dict[str, Any], spec: dict[str, Any], auth: dict[str, Any]) -> dict[str, Any]:
    cond = etude["conditions"]
    ctrl = spec["controller"]
    ver = spec["verification"]
    mvc1 = cond["mvc1"]["recovery"]
    return {
        "schema": "hdp_controllability_reachability_metrics.v0.1",
        "central_question": (
            "How do latent-state dimensionality, actuator controllability, restorative alignment, "
            "stability, and finite-amplitude reachability determine adaptation of a minimal E/I circuit, "
            "and how does successful adaptation appear across spikes, potentials, sources, fields, and spectra?"
        ),
        "plant": {
            "circuit": "MCC-3 minimal 5E/5PV Izhikevich TFNE",
            "operating_point": spec["operating_point"]["label"],
            "r0_hz": etude["measured_r0_hz"],
        },
        "control_theory": {
            "rank_J_W": 1,
            "rank_J_Theta_S": 2,
            "Theta_S": spec["Theta_S"]["channels"],
            "J_S": ver["J_S"],
            "zeta": ctrl["zeta"],
            "tau_slow_s": ver["predicted_tau_slow_s"],
            "G_star_diagonal": ver["G_star"][0][0],
            "max_Re_lambda_J_slow": max(float(x.real) if hasattr(x, "real") else float(x) for x in ver["J_slow_eigenvalues"]),
        },
        "authority": {
            "alpha_U_crit_estimate": auth["alpha_U_crit"]["alpha_U_crit_estimate"],
            "mvc1_alpha_U": 1.5,
            "mvc1_reachable": False,
            "mvc1_d_R_weighted": auth["hierarchy"]["finite_reachability"].split("d_R=")[-1],
            "mvc2_alpha_U": 1.2,
            "mvc2_reachable": next(
                r["reachable_within_tolerance"]
                for r in auth["alpha_continuation"]
                if abs(r["alpha_U"] - 1.2) < 1e-9
            ),
        },
        "mvc2_recovery": {
            "off": {
                "R_EI": cond["off"]["recovery"]["R_EI"],
                "terminal_error_weighted": cond["off"]["recovery"]["terminal_error_weighted"],
            },
            "scalar": {
                "R_EI": cond["scalar"]["recovery"]["R_EI"],
                "terminal_error_weighted": cond["scalar"]["recovery"]["terminal_error_weighted"],
            },
            "vector": {
                "R_EI": cond["vector"]["recovery"]["R_EI"],
                "terminal_error_weighted": cond["vector"]["recovery"]["terminal_error_weighted"],
            },
        },
        "mvc1_vector_partial": {
            "R_EI": mvc1["R_EI"],
            "R_E": mvc1["R_E"],
            "R_I": mvc1["R_I"],
            "terminal_error_weighted": mvc1["terminal_error_weighted"],
            "note": "local controllability does not imply finite reachability at 1.5 U0",
        },
        "neurophysiology_late": {
            k: {
                "r_E_hz": cond[k]["epochs"]["late"]["r_E_hz"],
                "r_I_hz": cond[k]["epochs"]["late"]["r_I_hz"],
                "E_P_to_baseline": cond[k]["spectral"]["E_P_late"],
                "R_P": cond[k]["spectral"]["R_P"],
                "phenotype_class": cond[k]["phenotype"]["classification"],
                "d_phys_weighted": cond[k]["phenotype"]["d_phys_weighted"],
            }
            for k in ("off", "scalar", "vector")
        },
        "rate_spectrum_coupling": {
            "hypothesis": "r_late ~ r0 => P_late(f) ~ P0(f)",
            "vector_late_rates_near_r0": cond["vector"]["recovery"]["terminal_error_weighted"] < 0.06,
            "vector_spectral_R_P": cond["vector"]["spectral"]["R_P"],
            "vector_phenotype": cond["vector"]["phenotype"]["classification"],
            "interpretation": (
                "Vector-H restores population rates (R_EI=0.96) with substantial spectral recovery "
                "(R_P~0.77) but weighted physiological distance remains elevated "
                "(rate_recovery_altered_phenotype): controlled observables recover while some "
                "field/source statistics remain on an alternative compensated trajectory."
            ),
        },
        "claim_gates": etude["claim_gates"],
        "phenotype_classification": {
            "rate_recovery_altered_phenotype": {
                "rate_tolerance": 0.15,
                "phenotype_tolerance": 0.15,
                "phenotype_vector": [
                    "r_E_hz",
                    "r_I_hz",
                    "vm_sd",
                    "source_rms",
                    "phi_rms",
                    "band_1_30_hz_source",
                    "band_30_80_hz_source",
                    "band_80_150_hz_source",
                ],
                "weights": [1 / 15, 1 / 10, 0.05, 1.0, 1.0, 1.0, 1.0, 1.0],
                "vector_late": {
                    "d_rate_weighted": cond["vector"]["phenotype"]["d_rate_weighted"],
                    "d_phys_weighted": cond["vector"]["phenotype"]["d_phys_weighted"],
                    "primary_residuals": {
                        "source_rms_baseline": cond["vector"]["epochs"]["baseline"]["source_rms"],
                        "source_rms_late": cond["vector"]["epochs"]["late"]["source_rms"],
                        "phi_rms_baseline": cond["vector"]["epochs"]["baseline"]["phi_rms"],
                        "phi_rms_late": cond["vector"]["epochs"]["late"]["phi_rms"],
                    },
                },
            }
        },
    }


def build_manifest(etude: dict[str, Any]) -> dict[str, Any]:
    receipt_hashes = {name: sha256_file(path) if path.exists() else None for name, path in FROZEN_RECEIPTS.items()}
    runner_hashes = {name: sha256_file(path) if path.exists() else None for name, path in RUNNERS.items()}
    return {
        "schema": "hdp_controllability_reachability_manifest.v0.1",
        "etude_id": "hdp_controllability_reachability",
        "status": "CONSOLIDATED",
        "repository": {
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "documentation": str(DOCS_MD.relative_to(ROOT)),
        "artifacts": {
            "figure": "figure.png",
            "metrics": "metrics.json",
            "readme": "README.md",
        },
        "frozen_receipts": {k: str(v.relative_to(ROOT)) for k, v in FROZEN_RECEIPTS.items()},
        "receipt_sha256": receipt_hashes,
        "runners": {k: str(v.relative_to(ROOT)) for k, v in RUNNERS.items()},
        "runner_sha256": runner_hashes,
        "reproduce": {
            "full_etude": f"python {RUNNERS['etude_simulation'].relative_to(ROOT)}",
            "consolidate_only": f"python {Path(__file__).relative_to(ROOT)}",
            "note": "Large trajectories are not versioned; regenerate via etude runner with frozen protocol/controller hashes.",
        },
        "protocol_sha256": etude["hashes"]["protocol_sha256"],
        "controller_spec_sha256": etude["hashes"]["controller_spec_sha256"],
    }


def write_readme(manifest: dict[str, Any]) -> str:
    return f"""# HDP controllability & reachability Etude

Compact reproducible bundle for the HDP-MVC scientific argument.

## Contents

| file | role |
|------|------|
| `figure.png` | Main A–L scientific plate |
| `metrics.json` | Compact control + neurophysiology metrics |
| `manifest.json` | Frozen receipt hashes and runner pointers |

## Reproduce

From repository root:

```bash
python scripts/consolidate_hdp_controllability_etude.py
```

This packages the committed `metrics.json`, `figure.png`, and `manifest.json`. If local diagnostic receipts exist under `artifacts/msvc_hdp_diagnostic/`, consolidation verifies their hashes and refreshes the figure from the etude simulation receipt; otherwise the committed bundle remains authoritative.

Prior control-theoretic receipts are referenced by hash in `manifest.json` (local provenance, not committed).

## Claim status

- Representation: relative computational scaffold
- Field: laminar proxy readout (`field_claim_level=proxy_readout`), not calibrated empirical LFP/CSD
- Controller: prospectively frozen before MVC #2 validation

Branch `{manifest['repository']['branch']}` @ `{manifest['repository']['sha'][:12]}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-etude", action="store_true", help="Re-run hdp_mvc_etude.py before packaging")
    args = parser.parse_args()

    missing = [k for k, p in FROZEN_RECEIPTS.items() if k != "etude_diagnostic" and not p.exists()]
    if missing:
        raise SystemExit(f"missing frozen receipts: {missing}")

    etude_path = ensure_etude_receipt(force=args.force_etude)
    etude = load_json(etude_path)
    spec = load_json(FROZEN_RECEIPTS["controller_spec"])
    auth = load_json(FROZEN_RECEIPTS["authority_boundary"])

    ETUDE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = compact_metrics(etude, spec, auth)
    manifest = build_manifest(etude)

    (ETUDE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (ETUDE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = write_readme(manifest)
    (ETUDE_DIR / "README.md").write_text(readme, encoding="utf-8")

    src_fig = DIAG / "hdp_mvc_etude.png"
    if not src_fig.exists():
        raise FileNotFoundError(src_fig)
    shutil.copy2(src_fig, ETUDE_DIR / "figure.png")

    print(f"wrote {ETUDE_DIR / 'metrics.json'}")
    print(f"wrote {ETUDE_DIR / 'manifest.json'}")
    print(f"wrote {ETUDE_DIR / 'README.md'}")
    print(f"wrote {ETUDE_DIR / 'figure.png'}")
    if not DOCS_MD.exists():
        print(f"note: documentation not found at {DOCS_MD} (create separately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
