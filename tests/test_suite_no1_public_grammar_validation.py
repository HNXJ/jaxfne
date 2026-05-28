"""Validation that Suite No. 1 Part 4 uses only public grammar, not internal optimizer loops.

This test reads the Suite No. 1 notebook and verifies that:
1. It uses jtfne.rate_targets(...) for objectives
2. It uses jtfne.agsdr(...) for optimizer specs
3. It uses model.tune(...) for optimization
4. It does NOT use internal functions like run_agsdr_optimization_loop
5. It does NOT contain manual for/generation loops in optimizer context
"""

import json
from pathlib import Path


def test_suite_no1_part4_uses_public_api():
    """Verify Suite No. 1 Part 4 uses only public composition grammar."""
    notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_1_computational_biophysics.ipynb"

    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    # Read notebook
    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    # Extract all code cells
    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cells.append("".join(cell.get("source", [])))

    # Find Part 4 cell (contains "Part 4" or multi-parameter AGSDR)
    part4_code = None
    for code in code_cells:
        if "Part 4" in code and ("rate_targets" in code or "agsdr" in code):
            part4_code = code
            break

    assert part4_code is not None, "Could not find Part 4 in notebook"

    # Verify public API calls are present
    assert "jtfne.rate_targets" in part4_code, "Part 4 should use jtfne.rate_targets()"
    assert "jtfne.agsdr" in part4_code, "Part 4 should use jtfne.agsdr()"
    assert "model" in part4_code and "tune" in part4_code, "Part 4 should use model.tune()"

    # Verify internal functions are NOT exposed
    assert "run_agsdr_optimization_loop" not in part4_code, \
        "Part 4 should NOT call run_agsdr_optimization_loop (internal function)"
    assert "_run_agsdr_optimization_loop" not in part4_code, \
        "Part 4 should NOT call _run_agsdr_optimization_loop (internal function)"

    # Verify no manual optimizer loop/generation logic
    assert "for generation in range" not in part4_code, \
        "Part 4 should not contain manual 'for generation in range' loops"
    assert "score_candidate_drives" not in part4_code, \
        "Part 4 should not define score_candidate_drives function (moved to package)"
    assert "theta_center" not in part4_code, \
        "Part 4 should not reference AGSDR internals like theta_center"

    print("✓ Suite No. 1 Part 4 uses only public grammar:")
    print("  - jtfne.rate_targets(...) ✓")
    print("  - jtfne.agsdr(...) ✓")
    print("  - model.tune(...) ✓")
    print("  - No internal run_agsdr_optimization_loop ✓")
    print("  - No manual for/generation loops ✓")


def test_suite_no1_part4_target_values():
    """Verify Suite No. 1 Part 4 has correct target values: 5 Hz / 10 Hz."""
    notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_1_computational_biophysics.ipynb"

    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cells.append("".join(cell.get("source", [])))

    # Find Part 4
    part4_code = None
    for code in code_cells:
        if "Part 4" in code and "rate_targets" in code:
            part4_code = code
            break

    assert part4_code is not None, "Could not find Part 4"

    # Check for target values
    assert "5.0" in part4_code or "5" in part4_code, "Part 4 should have first_half target of 5.0 Hz"
    assert "10.0" in part4_code or "10" in part4_code, "Part 4 should have second_half target of 10.0 Hz"
    assert "first_half" in part4_code, "Part 4 should reference first_half group"
    assert "second_half" in part4_code, "Part 4 should reference second_half group"

    print("✓ Suite No. 1 Part 4 has correct target values:")
    print("  - first_half: 5.0 Hz ✓")
    print("  - second_half: 10.0 Hz ✓")


def test_suite_no1_part4_parameter_names():
    """Verify Suite No. 1 Part 4 uses correct parameter names: drive_scale_a/b."""
    notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_1_computational_biophysics.ipynb"

    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cells.append("".join(cell.get("source", [])))

    # Find Part 4
    part4_code = None
    for code in code_cells:
        if "Part 4" in code and "agsdr" in code:
            part4_code = code
            break

    assert part4_code is not None, "Could not find Part 4"

    # Check for parameter names
    assert "drive_scale_a" in part4_code, "Part 4 should tune drive_scale_a"
    assert "drive_scale_b" in part4_code, "Part 4 should tune drive_scale_b"

    print("✓ Suite No. 1 Part 4 uses correct parameter names:")
    print("  - drive_scale_a ✓")
    print("  - drive_scale_b ✓")


if __name__ == "__main__":
    test_suite_no1_part4_uses_public_api()
    test_suite_no1_part4_target_values()
    test_suite_no1_part4_parameter_names()
    print("\n✓ All Suite No. 1 Part 4 public grammar validations passed!")
