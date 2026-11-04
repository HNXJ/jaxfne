"""Per-parameter log-scale sweep in the AGSDR loop.

A bound may be ``(lo, hi)`` (linear scale) or ``(lo, hi, "log")`` (the loop
runs in ``u = ln(theta)`` and records theta units). Linear-only runs reproduce
the baseline loop bit for bit (same z draws, same order).
"""

import math

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._signals import Objective
from jaxfne.optim.core import _run_agsdr_optimization_loop


def _log10_objective(candidate):
    return (math.log10(candidate["x"]) - 0.5) ** 2


# Baseline candidates, pinned by running the unmodified loop with
# parameter_bounds={"x": (0.1, 10.0)}, n_generations=2, n_population=4, seed=0.
_BASELINE_CANDIDATES = [
    5.050000190734863,
    8.659021377563477,
    4.277335166931152,
    4.9099040031433105,
    4.5477681159973145,
    2.815505266189575,
    3.6651458740234375,
    5.428750514984131,
]
_BASELINE_BEST = 2.815505266189575
_BASELINE_BEST_SCORE = 0.0025445624313771343


def test_linear_only_matches_baseline_bit_for_bit():
    result = _run_agsdr_optimization_loop(
        evaluate_fn=_log10_objective,
        parameter_bounds={"x": (0.1, 10.0)},
        n_generations=2,
        n_population=4,
        seed=0,
    )
    got = [c["x"] for c in result["all_candidates"]]
    np.testing.assert_allclose(got, _BASELINE_CANDIDATES, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        result["best_parameters"]["x"], _BASELINE_BEST, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        result["best_score"], _BASELINE_BEST_SCORE, rtol=1e-12, atol=0.0
    )


def test_log_bounds_symmetric_about_geometric_mean():
    result = _run_agsdr_optimization_loop(
        evaluate_fn=_log10_objective,
        parameter_bounds={"x": (0.01, 100.0, "log")},
        n_generations=1,
        n_population=200,
        seed=0,
    )
    thetas = np.array([c["x"] for c in result["all_candidates"]])
    assert np.all(thetas >= 0.01) and np.all(thetas <= 100.0)
    # Candidate 0 is the elitist anchor at the geometric mean sqrt(0.01 * 100).
    assert abs(math.log10(thetas[0])) < 1e-12
    log10_vals = np.log10(thetas)
    se = float(np.std(log10_vals, ddof=1)) / math.sqrt(log10_vals.size)
    assert abs(float(np.mean(log10_vals))) <= 3.0 * se


def test_log_bounds_put_mass_below_one():
    kwargs = dict(
        evaluate_fn=_log10_objective, n_generations=1, n_population=200, seed=0
    )
    linear = _run_agsdr_optimization_loop(
        parameter_bounds={"x": (0.01, 100.0)}, **kwargs
    )
    log = _run_agsdr_optimization_loop(
        parameter_bounds={"x": (0.01, 100.0, "log")}, **kwargs
    )
    linear_frac = float(
        np.mean([c["x"] < 1.0 for c in linear["all_candidates"]])
    )
    log_frac = float(np.mean([c["x"] < 1.0 for c in log["all_candidates"]]))
    assert linear_frac < 0.15
    assert 0.35 <= log_frac <= 0.65


@pytest.mark.parametrize(
    "bounds",
    [
        {"x": (0.0, 1.0, "log")},
        {"x": (-1.0, 1.0, "log")},
        {"x": (1.0, 1.0, "log")},
        {"x": (0.1, 10.0, "cubic")},
    ],
)
def test_log_bounds_refusals_name_the_path(bounds):
    with pytest.raises(ValueError) as excinfo:
        _run_agsdr_optimization_loop(
            evaluate_fn=_log10_objective,
            parameter_bounds=bounds,
            n_generations=1,
            n_population=2,
            seed=0,
        )
    assert "'x'" in str(excinfo.value)


def test_log_run_convergence_measurement():
    """Measurement (not a pass condition): evaluations for best-so-far to get
    within 10% of the optimum x = 10**0.5, log vs linear, same seed/budget."""
    optimum = 10.0**0.5

    def first_best_within(result):
        best_x = None
        best_score = float("inf")
        for i, candidate in enumerate(result["all_candidates"]):
            score = _log10_objective(candidate)
            if score < best_score:
                best_score = score
                best_x = candidate["x"]
            if abs(best_x - optimum) / optimum <= 0.10:
                return i + 1
        return None

    kwargs = dict(
        evaluate_fn=_log10_objective, n_generations=8, n_population=6, seed=0
    )
    linear = _run_agsdr_optimization_loop(
        parameter_bounds={"x": (0.01, 100.0)}, **kwargs
    )
    log = _run_agsdr_optimization_loop(
        parameter_bounds={"x": (0.01, 100.0, "log")}, **kwargs
    )
    linear_evals = first_best_within(linear)
    log_evals = first_best_within(log)
    print(f"MEASUREMENT log_evals={log_evals} linear_evals={linear_evals}")
    assert log_evals is not None, "log run never got within 10% of the optimum"
    assert linear_evals is not None, "linear run never got within 10% of the optimum"


# ---------------------------------------------------------------------------
# Public-entry coverage: every bounds-taking entry point either realizes "log"
# end to end or refuses a 3-element bound by name.
# ---------------------------------------------------------------------------


def _tiny_cfg(n=8):
    return (
        jtfne.configuration()
        .network(
            name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1}
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def _tiny_sim():
    return jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)


def _tiny_objective():
    return Objective(name="spectrolaminar_objective")


def _candidate_values(result, key):
    return [e["parameters"][key] for e in result.summary["candidate_evaluations"]]


def test_model_tune_scalar_log_reaches_loop():
    """Model.tune(parameters=) scalar log bounds reach the loop (anchor at the
    geometric mean; log10-mean closer to it than the same-seed linear run)."""
    model = jtfne.construct(_tiny_cfg())
    common = dict(objectives=_tiny_objective(), simulation=_tiny_sim())
    log = model.tune(
        optimizer=jtfne.agsdr(
            parameters={"source_scale": (0.5, 3.0, "log")},
            generations=2, population_size=4, seed=0,
        ),
        **common,
    )
    linear = model.tune(
        optimizer=jtfne.agsdr(
            parameters={"source_scale": (0.5, 3.0)},
            generations=2, population_size=4, seed=0,
        ),
        **common,
    )
    log_vals = _candidate_values(log, "source_scale")
    lin_vals = _candidate_values(linear, "source_scale")
    assert len(log_vals) == len(lin_vals) > 0
    geo = math.sqrt(0.5 * 3.0)
    assert min(abs(v - geo) for v in log_vals) < 1e-6 * geo
    assert min(abs(v - 1.75) for v in lin_vals) < 1e-9
    assert all(0.5 <= v <= 3.0 for v in log_vals)
    log_mean = float(np.mean(np.log10(log_vals)))
    lin_mean = float(np.mean(np.log10(lin_vals)))
    assert abs(log_mean - math.log10(geo)) < abs(lin_mean - math.log10(geo))


def test_tune_handles_log_reaches_loop():
    """tune_handles() log bounds reach the loop (same anchor/mean check)."""
    path = "cells.izhikevich.a[all]"
    model = jtfne.construct(_tiny_cfg())
    common = dict(
        objectives=_tiny_objective(), optimizer=jtfne.agsdr(), simulation=_tiny_sim()
    )
    log = jtfne.tune_handles(
        model, parameters={path: (0.005, 0.08, "log")},
        generations=2, population_size=4, seed=0, **common,
    )
    linear = jtfne.tune_handles(
        model, parameters={path: (0.005, 0.08)},
        generations=2, population_size=4, seed=0, **common,
    )
    log_vals = _candidate_values(log, path)
    lin_vals = _candidate_values(linear, path)
    assert len(log_vals) == len(lin_vals) > 0
    geo = math.sqrt(0.005 * 0.08)
    assert min(abs(v - geo) for v in log_vals) < 1e-6 * geo
    assert min(abs(v - 0.0425) for v in lin_vals) < 1e-9
    assert all(0.005 <= v <= 0.08 for v in log_vals)
    log_mean = float(np.mean(np.log10(log_vals)))
    lin_mean = float(np.mean(np.log10(lin_vals)))
    assert abs(log_mean - math.log10(geo)) < abs(lin_mean - math.log10(geo))


def test_model_tune_matrix_log_reaches_loop():
    """MatrixParameterSpec (lo, hi, "log") without an inner optimizer reaches
    the loop: the elitist anchor sits at the geometric mean."""
    model = jtfne.construct(_tiny_cfg())
    result = model.tune(
        objectives=_tiny_objective(),
        optimizer=jtfne.agsdr(
            parameters={"gain": jtfne.matrix_parameter(mask="all", bounds=(0.5, 3.0, "log"))},
            generations=2, population_size=4, seed=0,
        ),
        simulation=_tiny_sim(),
    )
    vals = _candidate_values(result, "gain")
    assert len(vals) > 0
    geo = math.sqrt(0.5 * 3.0)
    assert min(abs(v - geo) for v in vals) < 1e-6 * geo
    assert all(0.5 <= v <= 3.0 for v in vals)


def test_single_parameter_log_bounds_refused():
    """Legacy tune(parameter=, bounds=) is a linear scan: a 3-element bound is
    refused naming the parameter and the supported (lo, hi) form."""
    model = jtfne.construct(_tiny_cfg())
    with pytest.raises(ValueError) as excinfo:
        model.tune(
            objective=_tiny_objective(),
            parameter="source_scale",
            bounds=(0.5, 3.0, "log"),
            steps=2,
            seed=0,
            simulation=_tiny_sim(),
        )
    msg = str(excinfo.value)
    assert "source_scale" in msg and "(lo, hi)" in msg


def test_edge_parameter_log_bounds_refused():
    """EdgeParameterSpec strips nothing silently: a 3-element bound is refused
    naming EdgeParameterSpec.bounds and the supported (lower, upper) form."""
    with pytest.raises(ValueError) as excinfo:
        jtfne.edge_parameter(bounds=(0.1, 5.0, "log"), edge_indices=(0,))
    msg = str(excinfo.value)
    assert "EdgeParameterSpec.bounds" in msg and "(lower, upper)" in msg


def test_edge_parameter_two_tuple_bounds_still_accepted():
    spec = jtfne.edge_parameter(bounds=(0.1, 5.0), edge_indices=(0,))
    assert spec.bounds == (0.1, 5.0)


def test_handles_gradient_log_bounds_refused():
    """Gradient tune_handles loops are linear-only: log bounds are refused
    naming the path even with a valid surrogate declared."""
    model = jtfne.construct(_tiny_cfg())
    with pytest.raises(ValueError) as excinfo:
        jtfne.tune_handles(
            model,
            objectives=jtfne.rate_targets(
                groups={"all": list(range(8))}, targets_hz={"all": 5.0}
            ),
            optimizer=jtfne.optax_adam(
                learning_rate=0.05, differentiability_status="declared_surrogate"
            ),
            parameters={"cells.izhikevich.a[all]": (0.005, 0.08, "log")},
            steps=1,
            seed=0,
            simulation=_tiny_sim(),
            surrogate="soft_rate",
        )
    assert "cells.izhikevich.a[all]" in str(excinfo.value)


def test_matrix_inner_log_bounds_refused():
    """Matrix two-level (inner_optimizer) refinement clips in linear units, so
    log bounds are refused naming the parameter."""
    model = jtfne.construct(_tiny_cfg())
    opt = jtfne.agsdr(
        parameters={"gain": jtfne.matrix_parameter(mask="all", bounds=(0.5, 3.0, "log"))},
        generations=1, population_size=2, seed=0,
        inner_optimizer=object(), inner_steps=1,
    )
    with pytest.raises(ValueError) as excinfo:
        model.tune(objectives=_tiny_objective(), optimizer=opt, simulation=_tiny_sim())
    assert "gain" in str(excinfo.value)
