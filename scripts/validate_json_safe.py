#!/usr/bin/env python
"""JSON safety validator for jaxfne v0.2.30+.

Walks JSON files under outputs/ and docs/_static/tutorial_figures/ to detect
serialization violations: NaN, Infinity, non-JSON types.

Claim status: validation artifact, no scientific claim.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional


def is_json_safe(obj: Any) -> bool:
    """Check if an object is JSON-safe (no NaN/Inf)."""
    if obj is None or isinstance(obj, bool) or isinstance(obj, (int, str)):
        return True
    if isinstance(obj, float):
        # NaN or Inf
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return False
        return True
    if isinstance(obj, dict):
        return all(is_json_safe(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(is_json_safe(v) for v in obj)
    return True


def validate_json_file(filepath: Path) -> tuple[bool, Optional[str]]:
    """Validate a single JSON file.

    Returns (is_safe, error_message).
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read error: {e}"

    # Check for NaN/Inf in parsed data
    if not is_json_safe(data):
        return False, "Contains NaN/Infinity values after parsing"

    # Try to re-serialize with strict NaN check
    try:
        json.dumps(data, allow_nan=False)
    except ValueError as e:
        return False, f"Re-serialization failed: {e}"

    return True, None


def scan_directory(dirpath: Path) -> dict[str, Any]:
    """Scan a directory tree for JSON files and validate."""
    results = {
        "directory": str(dirpath),
        "files_checked": 0,
        "files_valid": 0,
        "files_invalid": 0,
        "failures": [],
    }

    json_files = list(dirpath.glob("**/*.json"))
    for filepath in json_files:
        results["files_checked"] += 1
        is_safe, error_msg = validate_json_file(filepath)

        if is_safe:
            results["files_valid"] += 1
        else:
            results["files_invalid"] += 1
            results["failures"].append({
                "file": str(filepath.relative_to(dirpath)),
                "error": error_msg,
            })

    return results


def main():
    """Validate JSON in outputs/ and docs/_static/tutorial_figures/."""
    output_base = Path("outputs")
    figures_dir = Path("docs/_static/tutorial_figures")

    all_results = {
        "validation_target": "jaxfne_json_safety_v0.2.30",
        "scans": [],
        "total_valid": 0,
        "total_invalid": 0,
    }

    # Scan outputs/
    if output_base.exists():
        print(f"Scanning {output_base}...", flush=True)
        result = scan_directory(output_base)
        all_results["scans"].append(result)
        all_results["total_valid"] += result["files_valid"]
        all_results["total_invalid"] += result["files_invalid"]
        print(f"  {result['files_checked']} files checked, {result['files_valid']} valid", flush=True)
        if result["failures"]:
            print(f"  {result['files_invalid']} INVALID:", flush=True)
            for failure in result["failures"]:
                print(f"    {failure['file']}: {failure['error']}", flush=True)

    # Scan docs/_static/tutorial_figures/
    if figures_dir.exists():
        print(f"Scanning {figures_dir}...", flush=True)
        result = scan_directory(figures_dir)
        all_results["scans"].append(result)
        all_results["total_valid"] += result["files_valid"]
        all_results["total_invalid"] += result["files_invalid"]
        print(f"  {result['files_checked']} files checked, {result['files_valid']} valid", flush=True)
        if result["failures"]:
            print(f"  {result['files_invalid']} INVALID:", flush=True)
            for failure in result["failures"]:
                print(f"    {failure['file']}: {failure['error']}", flush=True)

    # Write summary report
    report_path = Path("outputs/json_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, allow_nan=False)

    print(f"\nValidation report: {report_path}")
    print(f"Summary: {all_results['total_valid']} valid, {all_results['total_invalid']} invalid")

    if all_results["total_invalid"] > 0:
        print("\nStatus: VALIDATION FAILED (JSON safety violations detected)")
        sys.exit(1)
    else:
        print("\nStatus: VALIDATION PASSED (all JSON files are safe)")
        sys.exit(0)


if __name__ == "__main__":
    main()
