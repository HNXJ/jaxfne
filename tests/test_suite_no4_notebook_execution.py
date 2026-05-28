"""
Suite No. 4 notebook execution test.

Validates that the Suite No. 4 notebook runs without errors.
This test is marked as 'slow' because it executes the full notebook.
"""
import json
import pytest
import jaxfne as jtfne
import optax
import jax.numpy as jnp


@pytest.mark.slow
def test_suite_no4_notebook_execution():
    """Execute Suite No. 4 notebook cells and verify correctness."""
    # Cell 1: imports
    # import jaxfne as jtfne
    # import optax
    # import jax.numpy as jnp
    # import json

    # Cell 3: Build the Model
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m", "CSD"])
    )
    model = jtfne.construct(cfg)
    assert model.cfg.metadata["truth_mode"] is not None

    # Cell 5: Define Objectives and Simulation
    n_E = 16  # 80% of 20
    n_I = 4   # 20% of 20
    objective = jtfne.rate_targets(
        groups={"E": list(range(n_E)), "I": list(range(n_E, n_E + n_I))},
        targets_hz={"E": 12.0, "I": 8.0},
    )
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0)
    assert objective is not None
    assert sim is not None

    # Cell 7: Define Matrix Parameter Specifications
    gAMPA_w_spec = jtfne.matrix_parameter(
        mask="excitatory_to_all",
        bounds=(0.3, 3.0),
    )
    gGABA_w_spec = jtfne.matrix_parameter(
        mask="E_to_I",
        bounds=(0.1, 5.0),
    )
    assert gAMPA_w_spec.mask == "excitatory_to_all"
    assert gGABA_w_spec.mask == "E_to_I"

    # Cell 9: Two-Level AGSDR + Adam Optimizer
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": gAMPA_w_spec, "gGABA_w": gGABA_w_spec},
        generations=4,
        population_size=4,
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=5,
        seed=42,
    )
    assert optimizer.to_dict()["optimizer_class"] == "multiparameter_blackbox"
    assert set(optimizer.parameters.keys()) == {"gAMPA_w", "gGABA_w"}

    # Cell 11: Run Matrix Optimization
    result = model.tune(
        objectives=objective,
        optimizer=optimizer,
        simulation=sim,
    )
    assert result is not None
    assert hasattr(result, "best_parameters")
    assert hasattr(result, "best_score")
    assert hasattr(result, "summary")

    # Cell 13: Inspect and Validate Result
    # Verify JSON safety of summary
    json_str = json.dumps(result.summary, allow_nan=False)
    assert isinstance(json_str, str)

    # Verify gAMPA_w and gGABA_w are in best parameters
    assert "gAMPA_w" in result.best_parameters
    assert "gGABA_w" in result.best_parameters

    # Verify matrices are 2D
    gAMPA = jnp.asarray(result.best_parameters["gAMPA_w"])
    gGABA = jnp.asarray(result.best_parameters["gGABA_w"])
    assert gAMPA.ndim == 2, f"gAMPA_w ndim={gAMPA.ndim}, expected 2"
    assert gGABA.ndim == 2, f"gGABA_w ndim={gGABA.ndim}, expected 2"

    # Verify W changed
    orig_W = model.params["emitter"].W
    tuned_W = result.model.params["emitter"].W
    assert not jnp.allclose(orig_W, tuned_W), "W should have changed after tuning"

    print("✓ Suite No. 4 notebook execution test PASSED")
