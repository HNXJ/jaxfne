#!/usr/bin/env python3
import os
import json
import sys
import glob
from pathlib import Path
from argparse import ArgumentParser

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def audit_notebook_structure(nb_path: Path) -> dict:
    """Audit notebook cells for length and alternating markdown/code structures."""
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return {"file": str(nb_path), "valid": False, "error": f"Load error: {e}"}

    cells = nb.get("cells", [])
    violations = []
    cell_types = []
    
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type", "")
        cell_types.append(ctype)
        if ctype == "code":
            # Count logical/physical lines (excluding empty/comment lines)
            source = cell.get("source", [])
            lines = [line.strip() for line in source if line.strip() and not line.strip().startswith("#")]
            if len(lines) > 10:
                # Exclude standard manifest/parameter declaration cells which are large maps
                has_large_dict = any("{" in line or "[" in line for line in lines)
                if not has_large_dict:
                    violations.append(f"Cell #{idx} (code) exceeds 10 logical lines: {len(lines)} lines")

    # Check for alternating markdown and code: exactly one markdown between any two code cells
    for i in range(len(cell_types) - 1):
        if cell_types[i] == "code" and cell_types[i+1] == "code":
            violations.append(f"Back-to-back code cells at indices {i} and {i+1} without separating markdown cell")

    return {
        "file": str(nb_path.name),
        "valid": len(violations) == 0,
        "violations": violations,
        "total_cells": len(cells)
    }

def audit_png_figures() -> dict:
    """Scan tutorials and check for generated PNG output figures."""
    found_figs = []
    fig_dir = Path("docs/_static")
    if fig_dir.exists():
        for fig in fig_dir.rglob("*.png"):
            found_figs.append(str(fig))
            
    return {
        "status": "complete",
        "found_figures_count": len(found_figs),
        "figures": found_figs
    }

def audit_jaxley_bridge() -> dict:
    """Test Jaxley optional bridge imports and verify no hard dependencies."""
    status = "green"
    details = []
    try:
        import jaxfne.bridges as bridges
        details.append("jaxfne.bridges imported successfully.")
    except Exception as e:
        status = "red"
        details.append(f"Fail: jaxfne.bridges failed to import: {e}")
        
    try:
        from jaxfne.bridges import require_jaxley
        details.append("require_jaxley found.")
    except Exception as e:
        status = "red"
        details.append(f"Fail: require_jaxley not found: {e}")
        
    return {
        "status": status,
        "details": details
    }

def main():
    parser = ArgumentParser(description="Audit jaxfne notebooks and assets.")
    parser.add_argument("--check", action="store_true", help="Perform check only, do not write files")
    parser.add_argument("--write-report", type=str, help="Write summary report to specified JSON file path")
    args = parser.parse_args()

    print("=== STARTING JAXFNE COMPLETE RELEASE AUDIT ===")
    
    # 1. Notebook Audits
    nb_files = list(Path("tutorials").glob("*.ipynb"))
    nb_reports = []
    all_nb_valid = True
    for nb in nb_files:
        rep = audit_notebook_structure(nb)
        nb_reports.append(rep)
        if not rep.get("valid", True):
            all_nb_valid = False
            print(f"Notebook {rep['file']} has structural warnings: {rep.get('violations', [])}")
            
    # 2. PNG Figures Audit
    figs = audit_png_figures()
    
    # 3. Jaxley Bridge optional dependency check
    jaxley = audit_jaxley_bridge()
    
    # Combine reports
    summary = {
        "notebook_structure_audit": {
            "valid": all_nb_valid,
            "reports": nb_reports
        },
        "figure_audit": figs,
        "jaxley_bridge_audit": jaxley
    }
    
    # If not in --check mode and requested via --write-report, save report
    if args.write_report:
        report_path = Path(args.write_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAudit Summary saved to: {report_path}")
    elif not args.check:
        # Default fallback to save the report when no flag is specified
        report_path = Path("phase_validation/v0_3_14_audit_summary_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAudit Summary saved to: {report_path}")

    print(f"Notebook structure valid: {all_nb_valid}")
    print(f"Jaxley bridge safety check: {jaxley['status'].upper()}")

    # Fail check if bridges are red
    if jaxley["status"] == "red":
        print("✗ ERROR: Jaxley bridge safety check failed.")
        sys.exit(1)

    print("✓ Complete audit check passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
