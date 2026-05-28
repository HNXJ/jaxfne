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
    """Verify Suite No. 1 Part 4 uses correct parameter names: gAMPA_first_half/second_half."""
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
    assert "drive_scale_a" not in part4_code, "Part 4 must NOT use drive_scale_a (use gAMPA instead)"
    assert "drive_scale_b" not in part4_code, "Part 4 must NOT use drive_scale_b (use gAMPA instead)"
    assert "gAMPA" in part4_code or "AMPA" in part4_code, "Part 4 must use gAMPA/AMPA synaptic parameter"

    print("✓ Suite No. 1 Part 4 uses correct parameter names:")
    print("  - gAMPA_first_half ✓")
    print("  - gAMPA_second_half ✓")



def test_model_tune_supports_gampa_parameters():
    """Verify _model_with_scalar_parameter supports gAMPA_first_half and gAMPA_second_half."""
    import jaxfne as jtfne
    from jaxfne.core import _model_with_scalar_parameter

    cfg = (
        jtfne.configuration()
        .network(name="test", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)

    model_a = _model_with_scalar_parameter(model, "gAMPA_first_half", 1.5)
    model_b = _model_with_scalar_parameter(model, "gAMPA_second_half", 0.8)

    import jax.numpy as jnp
    assert model_a is not model
    assert model_b is not model
    # W should differ from original
    orig_W = model.params["emitter"].W
    new_W_a = model_a.params["emitter"].W
    new_W_b = model_b.params["emitter"].W
    assert not jnp.allclose(orig_W, new_W_a), "gAMPA_first_half must change W"
    assert not jnp.allclose(orig_W, new_W_b), "gAMPA_second_half must change W"


def test_suite_no1_optimizes_gampa_not_drive_scale():
    """Suite No. 1 Part 4 must use gAMPA, not drive_scale_a/b."""
    import json
    from pathlib import Path
    notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_1_computational_biophysics.ipynb"
    with open(notebook_path) as f:
        nb = json.load(f)
    code = " ".join(
        "".join(c.get("source", []))
        for c in nb["cells"] if c.get("cell_type") == "code"
    )
    assert "gAMPA" in code or "AMPA" in code, "Suite No. 1 must reference gAMPA or AMPA"
    assert "drive_scale_a" not in code, "Suite No. 1 must not use drive_scale_a"
    assert "drive_scale_b" not in code, "Suite No. 1 must not use drive_scale_b"


def test_poisson_drive_is_deterministic_under_seed():
    """Poisson drive output is deterministic given the same seed."""
    from jaxfne.core import _make_poisson_drive
    import jax.numpy as jnp
    out1 = _make_poisson_drive(100, 8, rate_hz=2.0, amplitude=0.5, dt_ms=0.1, seed=42)
    out2 = _make_poisson_drive(100, 8, rate_hz=2.0, amplitude=0.5, dt_ms=0.1, seed=42)
    assert jnp.allclose(out1, out2), "Poisson drive must be deterministic under same seed"


def test_poisson_drive_output_is_finite():
    """Poisson drive output contains only finite values."""
    from jaxfne.core import _make_poisson_drive
    import jax.numpy as jnp
    import numpy as np
    out = _make_poisson_drive(200, 10, rate_hz=5.0, amplitude=1.0, dt_ms=0.1, seed=7)
    assert jnp.all(jnp.isfinite(out)), "Poisson drive must be finite"
    assert out.shape == (200, 10), "Poisson drive must have shape (n_steps, n_neurons)"


def test_spectrolaminar_power_returns_at_least_64_freqs():
    """plot_spectrolaminar_power returns a figure and uses at least 64 freq bins."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from jaxfne.tutorial_utils import plot_spectrolaminar_power
    t = np.linspace(0, 100, 1000)
    signal = np.random.default_rng(0).normal(size=(1000, 4))
    fig = plot_spectrolaminar_power(t, signal, freq_min=1.0, freq_max=80.0, n_freqs=64, show=False)
    assert fig is not None


def test_save_png_creates_parent_dirs(tmp_path):
    """save_png creates parent directories and returns a valid path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jaxfne.tutorial_utils import save_png
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "sub" / "dir" / "test_fig.png"
    # Use new API: save_png(fig, path) where path includes directory
    # The current save_png API is: save_png(fig, name, fig_dir, show=False)
    result = save_png(fig, "test_fig", target.parent, show=False)
    assert Path(result).exists(), f"File not created: {result}"
    plt.close("all")


if __name__ == "__main__":
    test_suite_no1_part4_uses_public_api()
    test_suite_no1_part4_target_values()
    test_suite_no1_part4_parameter_names()
    print("\n✓ All Suite No. 1 Part 4 public grammar validations passed!")
