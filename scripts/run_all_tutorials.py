#!/usr/bin/env python3
"""
v0.2.20: Automated tutorial figure regeneration runner.

Executes all 4 core example scripts, validates outputs against contract,
and reports status. Supports explicit output root, smoke mode, and figure
generation control.

Usage:
    python scripts/run_all_tutorials.py --smoke --write-figures --out-root outputs/tutorials_v020
    python scripts/run_all_tutorials.py --out-root /tmp/test_tutorials

Exit codes:
    0: All tutorials completed successfully, all contracts validated
    1: Any tutorial failed, or contract validation failed
"""

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Optional


# Tutorial execution order and configurations
TUTORIALS = [
    {
        "name": "03_single_neuron_multimodal_probe",
        "script": "examples/03_single_neuron_multimodal_probe.py",
        "expected_output": "outputs/v023_single_neuron_multimodal",
        "expected_figures": ["figures/raster.png"],
    },
    {
        "name": "04_two_neuron_ei_multimodal",
        "script": "examples/04_two_neuron_ei_multimodal.py",
        "expected_output": "outputs/v029_two_neuron_ei_multimodal",
        "expected_figures": ["figures/raster.png"],
    },
    {
        "name": "05_network_100_ei_multimodal",
        "script": "examples/05_network_100_ei_multimodal.py",
        "expected_output": "outputs/v0210_network_100_ei_multimodal",
        "expected_figures": ["figures/raster.png"],
    },
    {
        "name": "02_spectrolaminar_oddball_scaffold",
        "script": "examples/02_spectrolaminar_oddball_scaffold.py",
        "expected_output": "outputs/v020_spectrolaminar_public_path",
        "expected_figures": ["figures/spectrolaminar_profile.png"],
    },
]

# Contract files required in each tutorial output
CONTRACT_FILES = [
    "manifest.json",
    "probe_report.json",
    "metrics.json",
    "validation_report.json",
    "asset_hashes.json",
]


def validate_tutorial_output(tutorial_dir: pathlib.Path, tutorial_name: str) -> dict:
    """
    Validate tutorial output directory against contract.

    Returns dict with validation results.
    Raises ValueError if contract is violated.
    """
    results = {
        "name": tutorial_name,
        "path": str(tutorial_dir),
        "status": "valid",
        "files_found": [],
        "files_missing": [],
        "figures_found": [],
        "figures_missing": [],
        "claim_gates": {},
        "errors": [],
    }

    if not tutorial_dir.exists():
        raise ValueError(f"Tutorial output directory does not exist: {tutorial_dir}")

    # Check contract files
    for contract_file in CONTRACT_FILES:
        file_path = tutorial_dir / contract_file
        if file_path.exists():
            results["files_found"].append(contract_file)
        else:
            results["files_missing"].append(contract_file)
            results["errors"].append(f"Missing contract file: {contract_file}")

    if results["files_missing"]:
        raise ValueError(f"Contract violation in {tutorial_name}: missing files {results['files_missing']}")

    # Check figures directory and figures
    figures_dir = tutorial_dir / "figures"
    if not figures_dir.exists():
        raise ValueError(f"Tutorial output missing figures/ directory: {tutorial_dir}")

    expected_figures = next(
        t["expected_figures"] for t in TUTORIALS if t["name"] == tutorial_name
    )

    for figure in expected_figures:
        fig_path = figures_dir / pathlib.Path(figure).name
        if fig_path.exists():
            size = fig_path.stat().st_size
            if size == 0:
                raise ValueError(f"Figure exists but is zero-size: {fig_path}")
            results["figures_found"].append(figure)
        else:
            results["figures_missing"].append(figure)
            results["errors"].append(f"Missing figure: {figure}")

    if results["figures_missing"]:
        raise ValueError(f"Contract violation in {tutorial_name}: missing figures {results['figures_missing']}")

    # Parse and validate JSON contract files
    for contract_file in ["manifest.json", "validation_report.json"]:
        try:
            json_path = tutorial_dir / contract_file
            json_str = json_path.read_text()

            # Check for NaN/Inf
            if "NaN" in json_str or "Infinity" in json_str or "-Infinity" in json_str:
                raise ValueError(f"JSON contains non-safe values (NaN/Inf): {contract_file}")

            data = json.loads(json_str)

            # Validate claim gates
            if "claim_level" in data:
                results["claim_gates"]["claim_level"] = data["claim_level"]
                if data["claim_level"] != "computational_scaffold":
                    raise ValueError(
                        f"Invalid claim_level: {data['claim_level']} "
                        f"(expected: computational_scaffold)"
                    )

            if "physical_amplitude_claim_allowed" in data:
                results["claim_gates"]["physical_amplitude_claim_allowed"] = data[
                    "physical_amplitude_claim_allowed"
                ]
                if data["physical_amplitude_claim_allowed"] is not False:
                    raise ValueError(
                        f"physical_amplitude_claim_allowed must be False, got {data['physical_amplitude_claim_allowed']}"
                    )

            if "field_claim_level" in data:
                results["claim_gates"]["field_claim_level"] = data["field_claim_level"]
                if data["field_claim_level"] != "proxy_readout_only":
                    raise ValueError(
                        f"Invalid field_claim_level: {data['field_claim_level']} "
                        f"(expected: proxy_readout_only)"
                    )

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {contract_file}: {e}")

    # Validate asset hashes structure
    try:
        asset_hashes_path = tutorial_dir / "asset_hashes.json"
        asset_hashes_str = asset_hashes_path.read_text()
        if "NaN" in asset_hashes_str or "Infinity" in asset_hashes_str:
            raise ValueError("asset_hashes.json contains non-safe values")
        asset_hashes = json.loads(asset_hashes_str)

        # Verify figure hashes are present
        for figure in expected_figures:
            fig_name = pathlib.Path(figure).name
            # Handle both string hash and dict {sha256, bytes} formats
            if fig_name not in asset_hashes and f"figures/{fig_name}" not in asset_hashes:
                raise ValueError(f"Figure hash missing from asset_hashes.json: {figure}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in asset_hashes.json: {e}")

    return results


def run_tutorial(
    tutorial: dict,
    out_root: pathlib.Path,
    write_figures: bool,
) -> dict:
    """
    Execute a single tutorial script and validate output.

    Returns dict with execution and validation results.
    Raises ValueError if execution or validation fails.
    """
    import os

    result = {
        "name": tutorial["name"],
        "script": tutorial["script"],
        "exit_code": None,
        "validation": {},
        "error": None,
    }

    # Execute the tutorial script
    cmd = [sys.executable, tutorial["script"]]

    # Set up environment with PYTHONPATH for jaxfne import
    env = os.environ.copy()
    jaxfne_root = "/Users/hamednejat/workspace/main/jaxfne"
    env["PYTHONPATH"] = f"{jaxfne_root}:{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=jaxfne_root,
            timeout=120,  # 2 minute timeout per tutorial
            env=env,
        )
        result["exit_code"] = proc.returncode

        if proc.returncode != 0:
            raise ValueError(
                f"Tutorial script exited with code {proc.returncode}: {tutorial['script']}\n"
                f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
            )

    except subprocess.TimeoutExpired:
        raise ValueError(f"Tutorial script timed out: {tutorial['script']}")
    except Exception as e:
        raise ValueError(f"Failed to run tutorial script {tutorial['script']}: {e}")

    # Validate output
    tutorial_output_dir = pathlib.Path(
        "/Users/hamednejat/workspace/main/jaxfne"
    ) / tutorial["expected_output"]

    try:
        validation = validate_tutorial_output(tutorial_output_dir, tutorial["name"])
        result["validation"] = validation
    except ValueError as e:
        raise ValueError(f"Tutorial validation failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run all core tutorial examples and validate outputs."
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="outputs/",
        help="Output root directory (currently unused, tutorial scripts use hardcoded paths)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: reduced runtime (not yet implemented in tutorial scripts)",
    )
    parser.add_argument(
        "--write-figures",
        action="store_true",
        default=True,
        help="Generate figures (default: True)",
    )

    args = parser.parse_args()

    out_root = pathlib.Path(args.out_root)

    # Run all tutorials and collect results
    all_results = {
        "status": "ok",
        "tutorials": [],
        "errors": [],
    }

    for tutorial in TUTORIALS:
        try:
            result = run_tutorial(tutorial, out_root, args.write_figures)
            all_results["tutorials"].append(result)
        except ValueError as e:
            all_results["errors"].append(str(e))
            all_results["status"] = "failed"
            print(f"❌ {tutorial['name']}: {e}", file=sys.stderr)

    # Print summary
    print(f"\n=== Tutorial Run Summary ===")
    print(f"Status: {all_results['status']}")
    print(f"Tutorials completed: {len([t for t in all_results['tutorials'] if t['exit_code'] == 0])}/{len(TUTORIALS)}")

    for result in all_results["tutorials"]:
        status_str = "✓" if result["exit_code"] == 0 else "✗"
        print(f"{status_str} {result['name']}: exit_code={result['exit_code']}")

    if all_results["errors"]:
        print(f"\nErrors:")
        for error in all_results["errors"]:
            print(f"  - {error}")

    # Print JSON report
    print(f"\n=== JSON Report ===")
    print(json.dumps(all_results, indent=2))

    # Return appropriate exit code
    if all_results["status"] == "failed":
        sys.exit(1)
    else:
        print(f"\n✓ All tutorials completed and validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
