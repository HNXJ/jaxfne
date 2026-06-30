"""Regression tests for Etude No. 1's AGSDR tuning path.

These lock in the fix for a previously silent no-op: rate_synchrony_targets()
declared its loss on a metric name the engine did not compute
("mean_firing_rate_hz" instead of "spike_rate_hz_mean"), the synchrony metric
"kappa_synchrony" was not computed at all, and the claim_level gates
were evaluated against readout metrics (always unknown). The combined effect was
that every AGSDR candidate scored null, the optimizer could not select, and the
tuned model was returned unchanged.

Truth posture: proxy/computational-scaffold diagnostics only; no biological
calibration or mechanism claim is asserted by these tests.
"""
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.builders import Configuration
from jaxfne.core import _KNOWN_METRICS, _compute_all_metrics


def _default_spectrolaminar_config(
    areas=None,
    n_per_area: int = 100,
    seed=None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Local replacement for the removed jtfne.default_spectrolaminar_config.

    default_spectrolaminar_config was removed from jaxfne/builders.py
    (archived as legacy-schema JSON at
    jaxfne/configs/legacy/spectrolaminar_default.json). Reproduced here
    verbatim from the pre-removal source (commit 1715ec2^) using the
    still-public Configuration fluent API.
    """
    if areas is None:
        areas = ["V1", "V4"]

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .areas(areas)
    )

    for area in areas:
        cfg = cfg.column(area, layers=["L1", "L2/3", "L4", "L5", "L6"], n=n_per_area)

    cfg = (
        cfg.cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .area_layer_cell_types(
            "V1",
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )
    )

    if len(areas) > 1:
        cfg = cfg.area_layer_cell_types(
            areas[1],
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )

    cfg = (
        cfg.uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.35, edge_seed=seed or 42)
    )

    if len(areas) >= 2:
        cfg = cfg.inter_column_connectivity(
            source_area=areas[0],
            target_area=areas[1],
            mode="sparse",
            p_feedforward=0.3,
            p_feedback=0.2,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
        )

    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD", "EEG", "MEG", "EMM"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .objective(
            firing_rate_target={"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0},
            band_definitions={"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)},
        )
    )
    return cfg


def _small_model_and_sim(seed=20260530, n_per_area=20, duration_ms=120.0, dt_ms=0.5):
    cfg = _default_spectrolaminar_config(
        areas=["V1", "V4"], n_per_area=n_per_area, seed=seed,
        duration_ms=duration_ms, dt_ms=dt_ms,
    )
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=seed,
        record_sources=True, record_fields=True,
    )
    return model, sim


def test_kappa_synchrony_is_a_known_engine_metric():
    """kappa_synchrony must be computed by the engine, not only by the helper."""
    assert "kappa_synchrony" in _KNOWN_METRICS
    model, sim = _small_model_and_sim()
    signals = model.simulate(sim)
    metrics = _compute_all_metrics(signals)
    assert "kappa_synchrony" in metrics
    kappa = metrics["kappa_synchrony"]
    assert kappa is not None
    assert -1.0 <= float(kappa) <= 1.0


def test_engine_kappa_matches_public_helper():
    """The vectorized engine kappa must agree with tutorial_utils.kappa_synchrony."""
    model, sim = _small_model_and_sim()
    signals = model.simulate(sim)
    engine_kappa = _compute_all_metrics(signals)["kappa_synchrony"]
    helper_kappa = jtfne.kappa_synchrony(np.asarray(signals.spikes), dt_ms=0.5)
    assert engine_kappa == pytest.approx(helper_kappa, abs=1e-6)


def test_rate_synchrony_targets_produces_real_loss_and_passing_gates():
    """The shipped objective must score (non-null loss) and pass truth gates."""
    model, sim = _small_model_and_sim()
    signals = model.simulate(sim)
    obj = jtfne.rate_synchrony_targets(
        target_rate_hz=3.5, target_kappa_synchrony=0.0,
        rate_weight=1.0, synchrony_weight=0.25,
    )
    report = model.evaluate(signals, obj, strict=False)
    assert report["total_loss"] is not None
    assert report["all_gates_pass"] is True
    # No metric should be reported as unknown.
    assert not any("unknown_metric" in w for w in report["warnings"])
    for loss in report["losses"]:
        assert loss["status"] == "ok"
        assert loss["value"] is not None


def test_agsdr_drive_gain_tuning_actually_changes_the_model():
    """AGSDR on a supported parameter must converge toward a low rate target."""
    model, sim = _small_model_and_sim()
    baseline = float(model.simulate(sim).summary()["spike_rate_hz_mean"])
    obj = jtfne.rate_synchrony_targets(
        target_rate_hz=1.0, target_kappa_synchrony=0.0,
        rate_weight=1.0, synchrony_weight=0.25,
    )
    opt = jtfne.agsdr(
        parameters={"drive_gain": (0.1, 1.5)},
        generations=4, population_size=4, seed=20260530,
    )
    result = model.tune(objectives=obj, optimizer=opt, simulation=sim, seed=20260530)
    assert result.summary.get("acceptance_decision") == "ACCEPT_CANDIDATE"
    assert result.best_parameters  # non-empty
    assert np.isfinite(result.best_score)
    tuned_rate = float(result.model.simulate(sim).summary()["spike_rate_hz_mean"])
    # Target (1.0 Hz) is well below baseline; tuning must reduce the rate.
    assert tuned_rate < baseline


def test_unsupported_tunable_parameter_is_reported_not_silently_ignored():
    """Tuning an unsupported parameter must surface an error, not pretend success."""
    model, sim = _small_model_and_sim()
    obj = jtfne.rate_synchrony_targets(
        target_rate_hz=3.5, target_kappa_synchrony=0.0,
    )
    opt = jtfne.agsdr(
        parameters={"noise_amplitude": (0.1, 1.0)},
        generations=2, population_size=2, seed=20260530,
    )
    result = model.tune(objectives=obj, optimizer=opt, simulation=sim, seed=20260530)
    summary = result.summary
    # The engine must not claim a successful tune for an unsupported parameter.
    assert summary.get("acceptance_decision") != "ACCEPT_CANDIDATE"
