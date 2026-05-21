#!/usr/bin/env python3
"""
v0.2.20: Independent tutorial output validator.

Validates an existing tutorial output tree against v0.2.19 contract.
Can be run independently of tutorial execution (no re-running required).

Usage:
    python scripts/validate_tutorial_outputs.py outputs/tutorials_v020

Exit codes:
    0: All tutorials validated successfully
    1: Any contract violation detected
"""

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Dict, List


# Tutorial output directories and expected figures
TUTORIAL_OUTPUTS = {
    "v023_single_neuron_multimodal": {
        "name": "03_single_neuron_multimodal_probe",
        "expected_figures": ["raster.png"],
    },
    "v029_two_neuron_ei_multimodal": {
        "name": "04_two_neuron_ei_multimodal",
        "expected_figures": ["raster.png"],
    },
    "v0210_network_100_ei_multimodal": {
        "name": "05_network_100_ei_multimodal",
        "expected_figures": ["raster.png"],
    },
    "v020_spectrolaminar_public_path": {
        "name": "02_spectrolaminar_oddball_scaffold",
        "expected_figures": ["spectrolaminar_profile.png"],
    },
}

# Contract files required in each output
CONTRACT_FILES = [
    "manifest.json",
    "probe_report.json",
    "metrics.json",
    "validation_report.json",
    "asset_hashes.json",
]


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_tutorial_output(
    output_dir: pathlib.Path, tutorial_key: str, tutorial_info: dict, require_interactive: bool = False
) -> dict:
    """
    Validate a single tutorial output directory.

    Returns validation result dict.
    Raises ValueError on contract violation.
    """
    result = {
        "tutorial_key": tutorial_key,
        "tutorial_name": tutorial_info["name"],
        "path": str(output_dir),
        "contract_files": {},
        "source_data": {},
        "figures": {},
        "interactive": {},
        "claim_gates": {},
        "metrics": {},
        "probe_report": {},
        "status": "valid",
        "errors": [],
    }

    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    # === Contract file validation ===
    for contract_file in CONTRACT_FILES:
        file_path = output_dir / contract_file
        result["contract_files"][contract_file] = {"exists": False, "valid": False}

        if not file_path.exists():
            result["errors"].append(f"Missing contract file: {contract_file}")
            raise ValueError(f"Missing contract file: {contract_file}")

        result["contract_files"][contract_file]["exists"] = True

        # Validate JSON structure
        try:
            json_str = file_path.read_text()

            # Check for non-JSON-safe values
            if "NaN" in json_str or "Infinity" in json_str or "-Infinity" in json_str:
                raise ValueError(f"JSON contains non-safe values (NaN/Inf): {contract_file}")

            data = json.loads(json_str)
            result["contract_files"][contract_file]["valid"] = True

            # Extract claim gates from manifest and validation_report
            if contract_file == "manifest.json":
                if "claim_level" in data:
                    result["claim_gates"]["claim_level"] = data["claim_level"]
                    if data["claim_level"] != "computational_scaffold":
                        raise ValueError(
                            f"Invalid claim_level in manifest.json: {data['claim_level']} "
                            f"(expected: computational_scaffold)"
                        )

                if "physical_amplitude_claim_allowed" in data:
                    result["claim_gates"]["physical_amplitude_claim_allowed"] = data[
                        "physical_amplitude_claim_allowed"
                    ]
                    if data["physical_amplitude_claim_allowed"] is not False:
                        raise ValueError(
                            f"physical_amplitude_claim_allowed in manifest must be False, "
                            f"got {data['physical_amplitude_claim_allowed']}"
                        )

                if "field_claim_level" in data:
                    result["claim_gates"]["field_claim_level"] = data["field_claim_level"]
                    if data["field_claim_level"] != "proxy_readout_only":
                        raise ValueError(
                            f"Invalid field_claim_level in manifest: {data['field_claim_level']} "
                            f"(expected: proxy_readout_only)"
                        )

            if contract_file == "validation_report.json":
                if "claim_level" in data:
                    if data["claim_level"] != "computational_scaffold":
                        raise ValueError(
                            f"Invalid claim_level in validation_report.json: {data['claim_level']}"
                        )
                if "physical_amplitude_claim_allowed" in data:
                    if data["physical_amplitude_claim_allowed"] is not False:
                        raise ValueError(
                            f"physical_amplitude_claim_allowed in validation_report must be False"
                        )

            # Extract metrics
            if contract_file == "metrics.json":
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        if value == 0 and "spike_rate" in key:
                            result["metrics"][key] = value
                            # Note: spike_rate of 0 is allowed in some conditions (e.g., single neuron)
                        elif isinstance(value, float) and not (-1e10 < value < 1e10):
                            raise ValueError(f"Metric value out of expected range: {key}={value}")
                        result["metrics"][key] = value

            # Extract probe report operators
            if contract_file == "probe_report.json":
                expected_operators = ["spk", "vm", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy"]
                for op_name in expected_operators:
                    if op_name in data:
                        result["probe_report"][op_name] = {
                            "present": True,
                            "has_operator_status": "operator_status" in data[op_name],
                        }

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {contract_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing {contract_file}: {e}")

    # === Source data validation ===
    source_data_path = output_dir / "figures" / "source_data.json"
    if not source_data_path.exists():
        raise ValueError(f"Missing source_data.json in figures/ directory: {output_dir}")

    try:
        source_data_str = source_data_path.read_text()
        if "NaN" in source_data_str or "Infinity" in source_data_str:
            raise ValueError("source_data.json contains non-safe values (NaN/Inf)")
        source_data = json.loads(source_data_str)

        # Validate claim gates in source data
        if source_data.get("claim_level") != "computational_scaffold":
            raise ValueError(f"source_data.json claim_level must be computational_scaffold, got {source_data.get('claim_level')}")
        if source_data.get("physical_amplitude_claim_allowed") is not False:
            raise ValueError(f"source_data.json physical_amplitude_claim_allowed must be False")

        # Validate source data kind and arrays
        source_data_kind = source_data.get("source_data_kind")
        if source_data_kind == "spike_events":
            time_ms = source_data.get("time_ms", [])
            unit_id = source_data.get("unit_id", [])
            if not isinstance(time_ms, list) or not isinstance(unit_id, list):
                raise ValueError("source_data.json spike_events: time_ms and unit_id must be lists")
            if len(time_ms) == 0 or len(unit_id) == 0:
                raise ValueError("source_data.json spike_events: arrays must be non-empty")
            if len(time_ms) != len(unit_id):
                raise ValueError(f"source_data.json spike_events: time_ms and unit_id length mismatch ({len(time_ms)} vs {len(unit_id)})")
            # Verify finite values
            if not all(isinstance(t, (int, float)) and -1e10 < float(t) < 1e10 for t in time_ms):
                raise ValueError("source_data.json spike_events: time_ms contains non-finite values")
            if not all(isinstance(u, (int, float)) for u in unit_id):
                raise ValueError("source_data.json spike_events: unit_id contains non-numeric values")
            result["source_data"] = {
                "kind": "spike_events",
                "event_count": len(time_ms),
                "unique_units": len(set(int(u) for u in unit_id)),
            }
        elif source_data_kind == "spectrolaminar_profile":
            alpha_beta = source_data.get("alpha_beta_profile", [])
            gamma = source_data.get("gamma_profile", [])
            layers = source_data.get("layers_or_depths", [])
            if not all(isinstance(p, list) for p in [alpha_beta, gamma, layers]):
                raise ValueError("source_data.json spectrolaminar_profile: profiles must be lists")
            if len(alpha_beta) == 0 or len(gamma) == 0 or len(layers) == 0:
                raise ValueError("source_data.json spectrolaminar_profile: arrays must be non-empty")
            if not (len(alpha_beta) == len(gamma) == len(layers)):
                raise ValueError(f"source_data.json spectrolaminar_profile: profile length mismatch")
            # Verify finite values
            if not all(isinstance(v, (int, float)) and -1e10 < float(v) < 1e10 for v in alpha_beta + gamma):
                raise ValueError("source_data.json spectrolaminar_profile: non-finite values")
            result["source_data"] = {
                "kind": "spectrolaminar_profile",
                "profile_length": len(alpha_beta),
                "alpha_dynamic_range": float(max(alpha_beta)) - float(min(alpha_beta)) if alpha_beta else 0,
                "gamma_dynamic_range": float(max(gamma)) - float(min(gamma)) if gamma else 0,
            }
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in source_data.json: {e}")

    # === Figure validation ===
    figures_dir = output_dir / "figures"
    if not figures_dir.exists():
        raise ValueError(f"Missing figures/ directory: {output_dir}")

    # Load asset_hashes for comparison
    asset_hashes_path = output_dir / "asset_hashes.json"
    try:
        asset_hashes_str = asset_hashes_path.read_text()
        if "NaN" in asset_hashes_str or "Infinity" in asset_hashes_str:
            raise ValueError("asset_hashes.json contains non-safe values")
        asset_hashes = json.loads(asset_hashes_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in asset_hashes.json: {e}")

    # Validate each expected figure
    for figure_name in tutorial_info["expected_figures"]:
        figure_path = figures_dir / figure_name
        result["figures"][figure_name] = {
            "exists": False,
            "size": 0,
            "hash_recorded": None,
            "hash_computed": None,
            "hash_match": False,
        }

        if not figure_path.exists():
            raise ValueError(f"Missing figure: {figure_path}")

        result["figures"][figure_name]["exists"] = True

        # Check file size
        file_size = figure_path.stat().st_size
        result["figures"][figure_name]["size"] = file_size

        if file_size == 0:
            raise ValueError(f"Figure is zero-size: {figure_path}")

        # Compute and compare hash
        computed_hash = compute_sha256(figure_path)
        result["figures"][figure_name]["hash_computed"] = computed_hash

        # Find recorded hash in asset_hashes.json
        # Handle both string hash and dict {sha256, bytes} formats
        recorded_hash = None
        for hash_key in [figure_name, f"figures/{figure_name}"]:
            if hash_key in asset_hashes:
                hash_entry = asset_hashes[hash_key]
                if isinstance(hash_entry, str):
                    recorded_hash = hash_entry
                elif isinstance(hash_entry, dict) and "sha256" in hash_entry:
                    recorded_hash = hash_entry["sha256"]
                break

        if recorded_hash is None:
            raise ValueError(f"Figure hash missing from asset_hashes.json: {figure_name}")

        result["figures"][figure_name]["hash_recorded"] = recorded_hash

        # Compare hashes
        if computed_hash != recorded_hash:
            raise ValueError(
                f"Figure hash mismatch for {figure_name}: "
                f"computed={computed_hash} vs recorded={recorded_hash}"
            )

        result["figures"][figure_name]["hash_match"] = True

    # === Interactive artifact validation (if required) ===
    if require_interactive:
        # Determine expected HTML filename based on tutorial
        html_filename = None
        if "spike_events" in str(result):  # Spike raster tutorials
            html_filename = "raster.html"
        else:
            # Could be spectrolaminar profile or other
            potential_html_files = list(figures_dir.glob("*.html")) if figures_dir.exists() else []
            if potential_html_files:
                html_filename = potential_html_files[0].name
            else:
                # Try to infer from tutorial name
                if "spectrolaminar" in tutorial_info.get("name", "").lower():
                    html_filename = "spectrolaminar_profile.html"
                else:
                    html_filename = "raster.html"

        if html_filename:
            html_path = figures_dir / html_filename
            result["interactive"]["expected_html"] = html_filename

            if not html_path.exists():
                raise ValueError(f"Interactive HTML missing: {html_filename}")

            result["interactive"]["exists"] = True

            # Check file size
            html_size = html_path.stat().st_size
            result["interactive"]["size"] = html_size

            if html_size == 0:
                raise ValueError(f"Interactive HTML is zero-size: {html_filename}")

            # Compute and compare hash
            computed_hash = compute_sha256(html_path)
            result["interactive"]["hash_computed"] = computed_hash

            # Find recorded hash in asset_hashes.json
            recorded_hash = None
            for hash_key in [html_filename, f"figures/{html_filename}"]:
                if hash_key in asset_hashes:
                    hash_entry = asset_hashes[hash_key]
                    if isinstance(hash_entry, str):
                        recorded_hash = hash_entry
                    elif isinstance(hash_entry, dict) and "sha256" in hash_entry:
                        recorded_hash = hash_entry["sha256"]
                    break

            if recorded_hash is None:
                raise ValueError(f"Interactive HTML hash missing from asset_hashes.json: {html_filename}")

            result["interactive"]["hash_recorded"] = recorded_hash

            # Compare hashes
            if computed_hash != recorded_hash:
                raise ValueError(
                    f"Interactive HTML hash mismatch for {html_filename}: "
                    f"computed={computed_hash} vs recorded={recorded_hash}"
                )

            result["interactive"]["hash_match"] = True

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate an existing tutorial output tree against v0.2.19 contract."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Root directory containing tutorial outputs (e.g., outputs/tutorials_v020)",
    )
    parser.add_argument(
        "--require-interactive",
        action="store_true",
        default=False,
        help="Require and validate interactive HTML artifacts (default: False)",
    )

    args = parser.parse_args()
    output_root = pathlib.Path(args.output_root)

    # Validate all tutorial outputs
    all_results = {
        "status": "valid",
        "output_root": str(output_root),
        "tutorials": [],
        "errors": [],
    }

    for tutorial_key, tutorial_info in TUTORIAL_OUTPUTS.items():
        tutorial_output_dir = output_root / tutorial_key
        try:
            result = validate_tutorial_output(
                tutorial_output_dir,
                tutorial_key,
                tutorial_info,
                require_interactive=args.require_interactive,
            )
            all_results["tutorials"].append(result)
        except ValueError as e:
            all_results["errors"].append(str(e))
            all_results["status"] = "invalid"
            print(f"❌ {tutorial_info['name']}: {e}", file=sys.stderr)

    # Print summary
    print(f"\n=== Tutorial Output Validation Summary ===")
    print(f"Output root: {output_root}")
    print(f"Status: {all_results['status']}")
    print(f"Tutorials validated: {len([t for t in all_results['tutorials']])}/{len(TUTORIAL_OUTPUTS)}")

    for result in all_results["tutorials"]:
        status_str = "✓" if result["status"] == "valid" else "✗"
        print(
            f"{status_str} {result['tutorial_name']}: "
            f"files={len(result['contract_files'])}, "
            f"figures={len(result['figures'])}"
        )

    if all_results["errors"]:
        print(f"\nValidation Errors:")
        for error in all_results["errors"]:
            print(f"  - {error}")

    # Print JSON report
    print(f"\n=== JSON Report ===")
    print(json.dumps(all_results, indent=2))

    # Return appropriate exit code
    if all_results["status"] == "invalid":
        sys.exit(1)
    else:
        print(f"\n✓ All tutorial outputs validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
