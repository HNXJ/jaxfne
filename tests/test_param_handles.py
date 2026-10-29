"""Lane 8 named parameter handles (gap G10).

Covers :mod:`jaxfne._param_handles` (parse/refuse, functional get/set,
``list_parameters`` flags) and :func:`jaxfne.optim.handles.tune_handles`
(black-box AGSDR on either family, optax exact/surrogate loops, concrete-edge
refusal), plus the shared single-handle optax dispatch through
:meth:`Model.tune` and byte-identity of the two pre-existing narrow tune
paths.
"""

import math
import re

import numpy as np
import pytest

import jaxfne as jtfne


def _izh_model(n=12):
    cfg = (
        jtfne.configuration()
        .network(n=n, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _izh_sim():
    return jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=0)


def _adex_model():
    cfg = (
        jtfne.Configuration()
        .network(name="adex8", n=8, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.5, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.5, weight=1.0, mechanism="ampa", delay_ms=1.0,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )
    return jtfne.construct(cfg)


def _adex_sim():
    return jtfne.Simulation(duration_ms=50.0, dt_ms=0.5, seed=0)


def _rate_objective(n, target_hz):
    return jtfne.rate_targets(
        groups={"all": list(range(n))}, targets_hz={"all": float(target_hz)}
    )


# ---------------------------------------------------------------------------
# parse / refuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "cells.izhikevich.a[all]",
        "cells.izhikevich.a[0]",
        "cells.izhikevich.a[2:8]",
        "cells.izhikevich.a[E]",
        "cells.izhikevich.a[cell_type=E]",
        "cells.izhikevich.source_scale[all]",
        "cells.adex.b_pA[all]",
        "edges.weight[all]",
        "edges.weight[0:4]",
        "edges.weight[receptor=ampa]",
        "drive.E.rate",
        "drive.all.rate",
    ],
)
def test_valid_handle_paths_parse(path):
    assert jtfne.is_handle_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "source_scale",  # legacy narrow-tune names are not handle paths
        "drive_gain",
        "synaptic_gain",
        "cells.izhikevich.a",  # missing selector brackets
        "cells.izhikevich.zzz[all]",  # unknown field
        "cells.lif.v[all]",  # unknown family
        "cells.izhikevich.a[voltage=E]",  # unknown selector key
        "edges.strength[all]",  # unknown edge field
        "edges.weight[cell_type=E]",  # cell-only selector key on edges
        "drive..rate",  # empty drive name
        "drive.E.gain",  # unknown drive field
        "",
        None,
    ],
)
def test_unknown_paths_refused_by_name(path):
    assert not jtfne.is_handle_path(path)
    with pytest.raises(ValueError, match=re.escape(repr(path))):
        jtfne.get_parameter(_izh_model(), path)


def test_tune_handles_unknown_path_refused_by_name():
    model = _izh_model()
    with pytest.raises(ValueError, match=re.escape(repr("cells.izhikevich.zzz[all]"))):
        jtfne.tune_handles(
            model,
            objectives=_rate_objective(12, 1.0),
            parameters={"cells.izhikevich.zzz[all]": (0.0, 1.0)},
            simulation=_izh_sim(),
        )


def test_family_mismatch_refused_by_name():
    izh = _izh_model()
    with pytest.raises(ValueError, match=re.escape("cells.adex.b_pA[all]")):
        jtfne.get_parameter(izh, "cells.adex.b_pA[all]")
    adex = _adex_model()
    with pytest.raises(ValueError, match=re.escape("cells.izhikevich.a[all]")):
        jtfne.get_parameter(adex, "cells.izhikevich.a[all]")
    with pytest.raises(ValueError, match=re.escape("drive.E.rate")):
        jtfne.get_parameter(adex, "drive.E.rate")


def test_scalar_source_scale_rejects_indexed_selectors():
    model = _izh_model()
    with pytest.raises(ValueError, match=re.escape("cells.izhikevich.source_scale[0]")):
        jtfne.get_parameter(model, "cells.izhikevich.source_scale[0]")
    with pytest.raises(ValueError, match=re.escape("cells.izhikevich.source_scale[E]")):
        jtfne.set_parameter(model, "cells.izhikevich.source_scale[E]", 1.0)


def test_dense_w_rejects_indexed_edge_selectors():
    model = _izh_model()
    with pytest.raises(ValueError, match=re.escape("edges.weight[0]")):
        jtfne.get_parameter(model, "edges.weight[0]")


# ---------------------------------------------------------------------------
# get / set round-trip (functional: input model never mutated)
# ---------------------------------------------------------------------------


def test_cell_get_set_roundtrip_is_functional():
    model = _izh_model()
    before = jtfne.get_parameter(model, "cells.izhikevich.a[all]")
    assert before.shape == (12,)
    assert np.allclose(before, 0.02)

    updated = jtfne.set_parameter(model, "cells.izhikevich.a[all]", 0.03)
    assert np.allclose(np.asarray(updated.params["emitter"].a), 0.03)
    # input model untouched
    assert np.allclose(np.asarray(model.params["emitter"].a), 0.02)

    # selection-size array sets per-element values
    per_neuron = np.linspace(0.01, 0.05, 12)
    updated = jtfne.set_parameter(model, "cells.izhikevich.a[all]", per_neuron)
    assert np.allclose(np.asarray(updated.params["emitter"].a), per_neuron)

    # full-length array applied at selected positions
    updated = jtfne.set_parameter(model, "cells.izhikevich.a[0:4]", per_neuron)
    got = np.asarray(updated.params["emitter"].a)
    assert np.allclose(got[:4], per_neuron[:4])
    assert np.allclose(got[4:], 0.02)


def test_drive_get_set_roundtrip():
    model = _izh_model()
    rates = jtfne.get_parameter(model, "drive.E.rate")
    assert rates.shape == (12,)
    updated = jtfne.set_parameter(model, "drive.E.rate", 7.0)
    assert np.all(np.asarray(updated.params["emitter"].drive) == 7.0)
    assert np.all(np.asarray(model.params["emitter"].drive) == 5.0)


def test_set_changes_executed_dynamics():
    model = _izh_model()
    sim = _izh_sim()
    base_spikes = np.asarray(model.simulate(sim).spikes).sum()
    assert base_spikes > 0, "fixture must spike"
    silenced = model.with_emitter_parameters(a=0.005)
    assert float(np.asarray(silenced.simulate(sim).spikes).sum()) < float(base_spikes)
    retuned = jtfne.set_parameter(model, "cells.izhikevich.a[all]", 0.08)
    assert float(np.asarray(retuned.simulate(sim).spikes).sum()) > float(base_spikes)


# ---------------------------------------------------------------------------
# list_parameters
# ---------------------------------------------------------------------------


def test_list_parameters_flags_only_source_scale_differentiable():
    entries = jtfne.list_parameters(_izh_model())
    by_path = {entry["path"]: entry for entry in entries}
    assert "cells.izhikevich.a[all]" in by_path
    assert "cells.izhikevich.source_scale[all]" in by_path
    assert "edges.weight[all]" in by_path
    assert "drive.E.rate" in by_path
    for path, entry in by_path.items():
        assert entry["n"] == int(np.prod(entry["shape"], dtype=int)) or entry["shape"] == []
        if path == "cells.izhikevich.source_scale[all]":
            assert entry["differentiable"] is True
            assert jtfne.is_differentiable_handle(path)
        else:
            assert entry["differentiable"] is False
            assert not jtfne.is_differentiable_handle(path)


# ---------------------------------------------------------------------------
# black-box fits
# ---------------------------------------------------------------------------


def test_blackbox_recovers_izhikevich_a():
    n = 12
    path = "cells.izhikevich.a[all]"
    a_true = 0.05
    model = _izh_model(n)
    sim = _izh_sim()
    duration_s = 200.0 / 1000.0
    truth = model.with_emitter_parameters(a=a_true)
    truth_spikes = np.asarray(truth.simulate(sim).spikes)
    assert truth_spikes.sum() > 0, "fixture must spike"
    # Target in Hz from a run at the truth with the same seed, so the
    # truth has (near-)zero loss under the squared-relative-error rate loss.
    target_hz = float(truth_spikes.sum() / (n * duration_s))
    objective = _rate_objective(n, target_hz)

    truth_report = truth.evaluate(truth.simulate(sim), objective)
    truth_score = float(truth_report["total_score"])
    assert truth_score <= 1e-9, "truth must have near-zero loss"

    def loss_at(a):
        cand = jtfne.set_parameter(model, path, float(a))
        rep = cand.evaluate(cand.simulate(sim), objective)
        return float(rep["total_score"])

    # Landscape check: the grid minimum is attained at the truth, and no
    # grid value more than 25 % away attains it (rate is monotone and
    # sensitive in a here: 36/48/60/63/72 spikes across the grid).
    grid = [0.5 * a_true, 0.75 * a_true, a_true, 1.25 * a_true, 1.5 * a_true]
    losses = [loss_at(a) for a in grid]
    assert min(losses) == losses[2]
    for a, loss in zip(grid, losses):
        if abs(a - a_true) > 0.25 * a_true:
            assert loss > losses[2]

    result = jtfne.tune_handles(
        model,
        objectives=objective,
        optimizer=jtfne.agsdr(),
        parameters={path: (0.005, 0.08)},
        generations=5,
        population_size=6,
        seed=0,
        simulation=sim,
    )
    assert result.summary["tuning_status"] == "handles_blackbox_agsdr_v0.1"
    assert result.best_score <= truth_score + 1e-9
    a_hat = result.best_parameters[path]
    assert abs(a_hat - a_true) <= 0.10 * a_true
    # returned model carries the fit and re-simulates
    assert math.isfinite(float(np.asarray(result.model.simulate(sim).spikes).sum()))

    # A second seed proposes a different candidate sequence.
    rerun = jtfne.tune_handles(
        model,
        objectives=objective,
        optimizer=jtfne.agsdr(),
        parameters={path: (0.005, 0.08)},
        generations=5,
        population_size=6,
        seed=1,
        simulation=sim,
    )
    seq0 = [e["parameters"][path] for e in result.summary["candidate_evaluations"]]
    seq1 = [e["parameters"][path] for e in rerun.summary["candidate_evaluations"]]
    assert seq0 != seq1, "second seed must give a different candidate sequence"


def test_model_tune_single_handle_blackbox():
    model = _izh_model()
    sim = _izh_sim()
    truth_spikes = np.asarray(model.with_emitter_parameters(a=0.03).simulate(sim).spikes)
    target_hz = float(truth_spikes.mean() / 0.2)
    result = model.tune(
        objectives=_rate_objective(12, target_hz),
        optimizer=jtfne.agsdr(),
        parameter="cells.izhikevich.a[all]",
        bounds=(0.005, 0.08),
        steps=4,
        seed=0,
        simulation=sim,
    )
    assert result.summary["tuning_status"] == "blackbox_loop_v0.0.6"
    assert len(result.history) == 4
    assert result.summary["candidate_values"]


def test_dense_edge_gain_blackbox_preserves_topology():
    model = _izh_model()
    before = np.asarray(model.params["emitter"].W)
    assert np.any(before == 0.0)
    sim = _izh_sim()
    target_hz = float(np.asarray(model.simulate(sim).spikes).mean() / 0.2) * 1.5
    result = jtfne.tune_handles(
        model,
        objectives=_rate_objective(12, target_hz),
        optimizer=jtfne.agsdr(),
        parameters={"edges.weight[all]": (0.5, 2.0)},
        generations=2,
        population_size=4,
        seed=0,
        simulation=sim,
    )
    assert result.summary["tuning_status"] == "handles_blackbox_agsdr_v0.1"
    after = np.asarray(result.model.params["emitter"].W)
    assert np.array_equal(after == 0.0, before == 0.0), "gain must preserve zero topology"


def test_adex_blackbox_fit_runs_where_model_tune_is_refused():
    model = _adex_model()
    sim = _adex_sim()
    with pytest.raises(NotImplementedError, match="adex"):
        model.tune(objectives=_rate_objective(8, 10.0), simulation=sim)
    result = jtfne.tune_handles(
        model,
        objectives=_rate_objective(8, 10.0),
        optimizer=jtfne.agsdr(),
        parameters={"cells.adex.b_pA[all]": (10.0, 200.0)},
        generations=2,
        population_size=4,
        seed=0,
        simulation=sim,
    )
    assert result.summary["tuning_status"] == "handles_blackbox_agsdr_v0.1"
    assert result.summary["n_candidates_evaluated"] == 8
    assert "cells.adex.b_pA[all]" in result.best_parameters


# ---------------------------------------------------------------------------
# gradient fits
# ---------------------------------------------------------------------------


def test_optax_missing_surrogate_raises_naming_path():
    model = _izh_model()
    with pytest.raises(ValueError, match=re.escape("cells.izhikevich.a[all]")):
        jtfne.tune_handles(
            model,
            objectives=_rate_objective(12, 1.0),
            optimizer=jtfne.optax_adam(
                learning_rate=0.05, differentiability_status="declared_surrogate"
            ),
            parameters={"cells.izhikevich.a[all]": (0.005, 0.08)},
            steps=3,
            seed=0,
            simulation=_izh_sim(),
        )


def test_model_tune_optax_handle_without_surrogate_refused():
    # Model.tune has no surrogate= entry point, so a gradient fit on a
    # spike-crossing handle must refuse naming the path (tune_handles keeps
    # its explicit surrogate= for this case).
    model = _izh_model()
    opt = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    with pytest.raises(ValueError, match=re.escape("cells.izhikevich.a[all]")):
        model.tune(
            objectives=_rate_objective(12, 10.0),
            optimizer=opt,
            parameter="cells.izhikevich.a[all]",
            bounds=(0.005, 0.08),
            steps=3,
            seed=0,
            simulation=_izh_sim(),
        )


def test_model_tune_handle_optax_unavailable_returns_revise(monkeypatch):
    # With optax unavailable a handle path keeps the pre-handles behaviour:
    # a REVISE report, not an ImportError from the gradient loop.
    import jaxfne.optim as optim_pkg
    import jaxfne.optim.handles as handles_mod

    def _missing():
        raise ImportError("This feature requires optional dependency 'optax'.")

    monkeypatch.setattr(optim_pkg, "require_optax", _missing)
    monkeypatch.setattr(handles_mod, "require_optax", _missing)
    model = _izh_model()
    opt = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(
        objectives=_rate_objective(12, 10.0),
        optimizer=opt,
        parameter="cells.izhikevich.a[all]",
        bounds=(0.005, 0.08),
        steps=3,
        seed=0,
        simulation=_izh_sim(),
    )
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
    assert result.summary["acceptance_decision"] == "REVISE"


def test_optax_surrogate_soft_rate_loop_runs():
    model = _izh_model()
    sim = _izh_sim()
    target_hz = float(np.asarray(model.simulate(sim).spikes).mean() / 0.2)
    result = jtfne.tune_handles(
        model,
        objectives=_rate_objective(12, target_hz),
        optimizer=jtfne.optax_adam(
            learning_rate=0.05, differentiability_status="declared_surrogate"
        ),
        parameters={"drive.E.rate": (1.0, 10.0)},
        steps=3,
        seed=0,
        simulation=sim,
        surrogate="soft_rate",
    )
    assert result.summary["tuning_status"] == "handles_differentiable_soft_rate_loop_v0.1"
    assert len(result.history) == 3
    assert "drive.E.rate" in result.best_parameters


def test_zero_learning_rate_refused_on_handle_soft_rate_path():
    """learning_rate=0.0 was silently replaced by 0.01 on the handle path."""
    model = _izh_model()
    with pytest.raises(ValueError, match="learning_rate"):
        jtfne.tune_handles(
            model,
            objectives=_rate_objective(12, 10.0),
            optimizer=jtfne.optax_adam(
                learning_rate=0.0, differentiability_status="declared_surrogate"
            ),
            parameters={"drive.E.rate": (1.0, 10.0)},
            steps=2,
            seed=0,
            simulation=_izh_sim(),
            surrogate="soft_rate",
        )


def test_concrete_edge_gradient_refused_up_front():
    model = _adex_model()
    with pytest.raises(ValueError, match=re.escape("edges.weight[all]")):
        jtfne.tune_handles(
            model,
            objectives=_rate_objective(8, 10.0),
            optimizer=jtfne.optax_adam(
                learning_rate=0.05, differentiability_status="declared_surrogate"
            ),
            parameters={"edges.weight[all]": (0.1, 2.0)},
            steps=3,
            seed=0,
            simulation=_adex_sim(),
            surrogate="soft_rate",
        )


# ---------------------------------------------------------------------------
# unchanged narrow paths (byte-identical numerics)
# ---------------------------------------------------------------------------


def test_narrow_drive_gain_blackbox_unchanged():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)
    obj = jtfne.rate_targets(groups={"all": list(range(20))}, targets_hz={"all": 15.0})
    result = model.tune(
        objectives=obj, optimizer=jtfne.agsdr(), simulation=sim,
        parameter="drive_gain", bounds=(0.5, 1.5), steps=4, seed=0,
    )
    assert set(result.best_parameters) == {"best_value"}
    # tune candidate seeds via SeedSequence (Hamm 2026-10-28); was 1.2579544029403025
    np.testing.assert_allclose(result.best_parameters["best_value"], 0.84114316166169, rtol=1e-12)
    # P-048 (u0 = b * v0), 2026-10-28; was 0.07111111111111111 (unchanged)
    np.testing.assert_allclose(result.best_score, 0.07111111111111111, rtol=1e-12)


def test_narrow_source_scale_optax_unchanged():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0)
    base_val = float(abs(model.simulate(sim).field.source_proxy).mean())
    obj = jtfne.objective().loss(
        "hit_target", metric="source_proxy_abs_mean", target=base_val * 2.0
    )
    opt = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(objectives=obj, optimizer=opt, simulation=sim, steps=5)
    history = result.summary["gradient_history"]
    assert len(history) == 5
    np.testing.assert_allclose(history[-1]["source_scale"], 1.2485201358795166, rtol=1e-6)
    np.testing.assert_allclose(history[-1]["loss"], 278.07080078125, rtol=1e-6)


# ---------------------------------------------------------------------------
# T-55 (lane C3): the AGSDR loop catches refusals only; seeds on gradient paths
# ---------------------------------------------------------------------------


def _edge_handle_call(**overrides):
    kwargs = dict(
        objectives=_rate_objective(12, 10.0),
        optimizer=jtfne.agsdr(),
        parameters={"edges.weight[all]": (0.5, 2.0)},
        generations=1,
        population_size=2,
        seed=0,
        simulation=_izh_sim(),
    )
    kwargs.update(overrides)
    return jtfne.tune_handles(_izh_model(), **kwargs)


def test_handles_loop_programming_error_propagates(monkeypatch):
    """A TypeError inside the AGSDR loop is a bug, not a REVISE report."""
    import jaxfne.optim.handles as handles_mod

    def _programming_error(**kwargs):
        raise TypeError("injected programming error")

    monkeypatch.setattr(handles_mod, "_run_agsdr_optimization_loop", _programming_error)
    with pytest.raises(TypeError, match="injected programming error"):
        _edge_handle_call()


def test_handles_loop_refusal_still_reports_revise(monkeypatch):
    """A ValueError refusal from the AGSDR loop keeps its REVISE report."""
    import jaxfne.optim.handles as handles_mod

    def _refused(**kwargs):
        raise ValueError("injected refusal")

    monkeypatch.setattr(handles_mod, "_run_agsdr_optimization_loop", _refused)
    result = _edge_handle_call()
    assert result.summary["tuning_status"] == "handles_agsdr_error"
    assert result.summary["acceptance_decision"] == "REVISE"
    assert "injected refusal" in result.summary["error"]


def _soft_rate_handle_call(**overrides):
    kwargs = dict(
        objectives=_rate_objective(12, 10.0),
        optimizer=jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate"),
        parameters={"drive.E.rate": (1.0, 10.0)},
        steps=2,
        seed=0,
        simulation=_izh_sim(),
        surrogate="soft_rate",
    )
    kwargs.update(overrides)
    return jtfne.tune_handles(_izh_model(), **kwargs)


def test_nondefault_seed_is_refused_on_the_handles_soft_rate_loop():
    """seed=7 against a seed-0 simulation: the loop reads no seed of its own, so it is refused."""
    with pytest.raises(ValueError, match="seed"):
        _soft_rate_handle_call(seed=7)


def test_handles_soft_rate_summary_records_the_simulation_seed_that_applied():
    result = _soft_rate_handle_call()
    assert result.summary["tuning_status"] == "handles_differentiable_soft_rate_loop_v0.1"
    assert result.summary["seed"] == 0
    assert result.summary["optimizer_seed"] is None


def test_handles_seed_without_simulation_builds_the_default_simulation_from_it():
    """tune_handles(seed=7) with no simulation: the default simulation takes seed 7; the summary records 7."""
    result = _soft_rate_handle_call(seed=7, simulation=None)
    assert result.summary["tuning_status"] == "handles_differentiable_soft_rate_loop_v0.1"
    assert result.summary["seed"] == 7
    assert result.summary["simulation_seed"] == 7


def test_handles_seed_that_differs_from_the_simulation_seed_is_refused():
    """simulation.seed=3 with seed=7: seed 7 would apply to nothing, so it is refused."""
    with pytest.raises(ValueError, match="seed"):
        _soft_rate_handle_call(seed=7, simulation=jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=3))


def test_handles_zero_seed_runs_on_a_seeded_simulation_and_records_its_seed():
    """simulation.seed=3 with seed=0: the loop runs, and the summary records the seed that applied (3)."""
    result = _soft_rate_handle_call(seed=0, simulation=jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=3))
    assert result.summary["tuning_status"] == "handles_differentiable_soft_rate_loop_v0.1"
    assert result.summary["seed"] == 3
