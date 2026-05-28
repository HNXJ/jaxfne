"""
Suite No. 4 public grammar tests.

Validates the matrix parameter optimization API for AMPA/GABA tuning.
"""
import json
import pytest
import jaxfne as jtfne
import optax


def test_suite_no4_matrix_parameter_specs_are_created():
    """Verify gAMPA_w and gGABA_w specs are created correctly."""
    gAMPA_spec = jtfne.matrix_parameter(
        mask="excitatory_to_all",
        bounds=(0.0, 3.0),
    )
    gGABA_spec = jtfne.matrix_parameter(
        mask="E_to_I",
        bounds=(0.0, 5.0),
    )
    assert gAMPA_spec.mask == "excitatory_to_all"
    assert gGABA_spec.mask == "E_to_I"
    assert gAMPA_spec.bounds == (0.0, 3.0)
    assert gGABA_spec.bounds == (0.0, 5.0)


def test_suite_no4_agsdr_accepts_optax_adam_inner_optimizer():
    """Verify AGSDR accepts Optax Adam as inner optimizer."""
    gAMPA_spec = jtfne.matrix_parameter(
        mask="excitatory_to_all",
        bounds=(0.0, 3.0),
    )
    optimizer_spec = jtfne.agsdr(
        parameters={"gAMPA_w": gAMPA_spec},
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=5,
        generations=2,
        population_size=4,
        seed=42,
    )
    assert optimizer_spec.inner_optimizer is not None
    assert optimizer_spec.inner_steps == 5


def test_suite_no4_tune_with_both_matrix_parameters():
    """Verify model.tune() works with both gAMPA_w and gGABA_w."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)

    n_E = 16
    n_I = 4
    objective = jtfne.rate_targets(
        groups={"E": list(range(n_E)), "I": list(range(n_E, n_E + n_I))},
        targets_hz={"E": 12.0, "I": 8.0},
    )
    sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)

    gAMPA_spec = jtfne.matrix_parameter(
        mask="excitatory_to_all",
        bounds=(0.3, 3.0),
    )
    gGABA_spec = jtfne.matrix_parameter(
        mask="E_to_I",
        bounds=(0.1, 5.0),
    )

    optimizer = jtfne.agsdr(
        parameters={
            "gAMPA_w": gAMPA_spec,
            "gGABA_w": gGABA_spec,
        },
        generations=2,
        population_size=3,
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=3,
        seed=42,
    )

    result = model.tune(
        objectives=objective,
        optimizer=optimizer,
        simulation=sim,
    )

    assert "gAMPA_w" in result.best_parameters
    assert "gGABA_w" in result.best_parameters


def test_suite_no4_best_parameters_are_2d_matrices():
    """Verify gAMPA_w and gGABA_w are returned as 2D matrices."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)

    n_E = 16
    n_I = 4
    objective = jtfne.rate_targets(
        groups={"E": list(range(n_E)), "I": list(range(n_E, n_E + n_I))},
        targets_hz={"E": 12.0, "I": 8.0},
    )
    sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)

    optimizer = jtfne.agsdr(
        parameters={
            "gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.3, 3.0)),
            "gGABA_w": jtfne.matrix_parameter(mask="E_to_I", bounds=(0.1, 5.0)),
        },
        generations=2,
        population_size=3,
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=3,
        seed=42,
    )

    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)

    import numpy as np
    gAMPA = np.asarray(result.best_parameters["gAMPA_w"])
    gGABA = np.asarray(result.best_parameters["gGABA_w"])

    assert gAMPA.ndim == 2, f"gAMPA_w ndim={gAMPA.ndim}, expected 2"
    assert gGABA.ndim == 2, f"gGABA_w ndim={gGABA.ndim}, expected 2"


def test_suite_no4_summary_is_json_safe():
    """Verify result.summary is JSON-safe (no Optax objects)."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)

    n_E = 16
    n_I = 4
    objective = jtfne.rate_targets(
        groups={"E": list(range(n_E)), "I": list(range(n_E, n_E + n_I))},
        targets_hz={"E": 12.0, "I": 8.0},
    )
    sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)

    optimizer = jtfne.agsdr(
        parameters={
            "gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.3, 3.0)),
            "gGABA_w": jtfne.matrix_parameter(mask="E_to_I", bounds=(0.1, 5.0)),
        },
        generations=2,
        population_size=3,
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=3,
        seed=42,
    )

    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)

    # Should not raise ValueError about NaN/Inf
    json_str = json.dumps(result.summary, allow_nan=False)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
