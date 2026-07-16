"""MatrixParameterSpec.target generalization (added alongside the
homeostatic_ei emitter, see jaxfne/emitters_homeostatic_ei.py): lets the
AGSDR matrix-tuning path address a field other than the Izhikevich `W`
matrix -- specifically the homeostatic_ei emitter's `G0` (initial
conductance matrix). Default target="W" keeps every existing caller
unaffected (see tests/test_optim_tune.py's gAMPA_w-based coverage for the
default-path regression).
"""
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne._model import MatrixParameterSpec, _model_with_matrix_parameter


def _homeostatic_ei_model():
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=100.0, dt_ms=0.5)
        .network(name="ei2", n=2)
        .set_emitter("homeostatic_ei")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    return jtfne.construct(cfg)


def test_target_field_scales_g0_not_w():
    model = _homeostatic_ei_model()
    spec = MatrixParameterSpec(mask="all", bounds=(0.1, 5.0), target="G0")

    new_model = _model_with_matrix_parameter(model, "G0", spec, 2.0)

    original_G0 = np.asarray(model.params["emitter"].G0, dtype=float)
    new_G0 = np.asarray(new_model.params["emitter"].G0, dtype=float)
    assert np.allclose(new_G0, original_G0 * 2.0)
    # Original model must be unchanged (frozen dataclass contract).
    assert np.allclose(np.asarray(model.params["emitter"].G0), original_G0)


def test_target_field_bounds_clip_the_scale_factor():
    model = _homeostatic_ei_model()
    spec = MatrixParameterSpec(mask="all", bounds=(0.5, 2.0), target="G0")
    original_G0 = np.asarray(model.params["emitter"].G0, dtype=float)

    clipped_hi = _model_with_matrix_parameter(model, "G0", spec, 100.0)  # scale clipped to 2.0
    clipped_lo = _model_with_matrix_parameter(model, "G0", spec, 0.01)  # scale clipped to 0.5

    assert np.allclose(np.asarray(clipped_hi.params["emitter"].G0), original_G0 * 2.0)
    assert np.allclose(np.asarray(clipped_lo.params["emitter"].G0), original_G0 * 0.5)


def test_target_defaults_to_w_and_leaves_izhikevich_path_unchanged():
    """Regression: a spec with no explicit target still hits W (Izhikevich)."""
    from jaxfne.core import _model_with_matrix_parameter as _model_with_matrix_parameter_core

    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=10.0, dt_ms=0.1)
        .column("izh", layers=["L2/3"], n=4)
        .set_emitter("izhikevich")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(modes=["V_m"])
    )
    model = jtfne.construct(cfg)
    spec = MatrixParameterSpec(mask="all", bounds=(0.1, 5.0))
    assert spec.target == "W"
    new_model = _model_with_matrix_parameter_core(model, "gAMPA_w", spec, 2.0)
    original_W = jnp.asarray(model.params["emitter"].W, dtype=float)
    new_W = jnp.asarray(new_model.params["emitter"].W, dtype=float)
    assert not jnp.allclose(original_W, new_W)


def test_matrix_parameter_helper_forwards_target():
    spec = jtfne.matrix_parameter(mask="all", bounds=(0.1, 5.0), target="G0")
    assert spec.target == "G0"
    default_spec = jtfne.matrix_parameter(mask="all", bounds=(0.1, 5.0))
    assert default_spec.target == "W"


def test_agsdr_tune_smoke_runs_without_crash_on_g0_target():
    """Full AGSDR convergence on G0 is deferred (see artifacts/developer/plans.json
    entry homeostatic-ei-milestones-4-6-regime-sweep) -- this only confirms the
    plumbing (Objective -> tune() -> AGSDR -> MatrixParameterSpec(target="G0"))
    runs end-to-end without an exception for a couple of generations."""
    model = _homeostatic_ei_model()
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.5, seed=0)
    objective = jtfne.rate_targets(groups={"E": [0], "I": [1]}, targets_hz={"E": 5.0, "I": 3.0})
    spec = jtfne.matrix_parameter(mask="all", bounds=(0.1, 5.0), target="G0")

    result = model.tune(
        objectives=objective, optimizer="AGSDR", simulation=sim,
        parameters={"G0": spec}, generations=1, population_size=2,
    )
    assert result.best_score is not None
    assert np.isfinite(result.best_score)
