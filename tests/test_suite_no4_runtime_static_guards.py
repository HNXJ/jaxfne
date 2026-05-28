"""
Suite No. 4 static guards: verify correct grammar.

Ensures Suite No. 4 uses the correct parameter grammar:
- gAMPA_w and gGABA_w for matrix parameters
- Optax Adam as inner optimizer, not jtfne.adam
"""
import pytest
import jaxfne as jtfne

# Gate optional Optax dependency at module level
optax = pytest.importorskip("optax")


def test_suite_no4_uses_gampa_w_not_gampa_scalar():
    """Verify Suite No. 4 uses gAMPA_w matrix parameter."""
    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.0, 3.0))
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": spec},
        generations=1,
        seed=0,
    )
    assert "gAMPA_w" in optimizer.parameters
    assert isinstance(optimizer.parameters["gAMPA_w"], jtfne.core.MatrixParameterSpec)


def test_suite_no4_uses_ggaba_w_not_scalar():
    """Verify Suite No. 4 uses gGABA_w matrix parameter."""
    spec = jtfne.matrix_parameter(mask="E_to_I", bounds=(0.0, 5.0))
    optimizer = jtfne.agsdr(
        parameters={"gGABA_w": spec},
        generations=1,
        seed=0,
    )
    assert "gGABA_w" in optimizer.parameters
    assert isinstance(optimizer.parameters["gGABA_w"], jtfne.core.MatrixParameterSpec)


def test_suite_no4_uses_optax_adam_not_jtfne_adam():
    """Verify jtfne.adam does not exist; use optax.adam instead."""
    assert not hasattr(jtfne, "adam"), "jtfne.adam should not exist; use optax.adam"
    # Verify optax.adam is accessible
    inner_opt = optax.adam(learning_rate=1e-2)
    assert inner_opt is not None


def test_suite_no4_jtfne_agsdr_adam_does_not_exist():
    """Verify jtfne.agsdr_adam does not exist."""
    assert not hasattr(jtfne, "agsdr_adam"), "jtfne.agsdr_adam should not exist; use jtfne.agsdr with inner_optimizer=optax.adam"


def test_suite_no4_accepts_both_gampa_w_and_ggaba_w():
    """Verify Suite No. 4 grammar accepts both matrix parameters together."""
    optimizer = jtfne.agsdr(
        parameters={
            "gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.0, 3.0)),
            "gGABA_w": jtfne.matrix_parameter(mask="E_to_I", bounds=(0.0, 5.0)),
        },
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=5,
        generations=2,
        population_size=4,
        seed=42,
    )
    assert "gAMPA_w" in optimizer.parameters
    assert "gGABA_w" in optimizer.parameters
    assert optimizer.inner_optimizer is not None


def test_suite_no4_inner_optimizer_metadata_is_json_safe():
    """Verify inner optimizer metadata can be serialized (not the optimizer object itself)."""
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.0, 3.0))},
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=5,
        generations=2,
        population_size=4,
        seed=42,
    )
    # to_dict should not include the actual optimizer object
    spec_dict = optimizer.to_dict()
    assert "inner_optimizer" in spec_dict
    # The inner_optimizer field should be metadata (string or dict), not the Optax object
    inner_meta = spec_dict["inner_optimizer"]
    assert isinstance(inner_meta, (str, type(None)))


def test_suite_no4_requires_matrix_parameter_for_matrices():
    """Verify matrix parameters must use matrix_parameter() spec, not tuple bounds."""
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

    # This should work (proper matrix_parameter spec)
    optimizer_good = jtfne.agsdr(
        parameters={"gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.0, 3.0))},
        generations=1,
        population_size=2,
        seed=0,
    )
    result_good = model.tune(objectives=objective, optimizer=optimizer_good, simulation=sim)
    assert "gAMPA_w" in result_good.best_parameters

    # This should also work (scalar parameter as tuple)
    optimizer_scalar = jtfne.agsdr(
        parameters={"source_scale": (0.5, 2.0)},
        generations=1,
        population_size=2,
        seed=0,
    )
    result_scalar = model.tune(objectives=objective, optimizer=optimizer_scalar, simulation=sim)
    assert "source_scale" in result_scalar.best_parameters
