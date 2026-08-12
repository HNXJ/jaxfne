"""Canonical HDP population-restoring controller tests (no exploratory receipt paths)."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_tune import _edge_parameter_mask, _model_with_parameters

ROOT = Path(__file__).resolve().parents[1]
ETUDE_METRICS = ROOT / "artifacts/etudes/hdp_controllability_reachability/metrics.json"

# Frozen controller from Etude checkpoint (internal test constants; not MVC receipt paths).
ETUDE_CONTROLLER = {
    "B": [[-0.04728457976526459, 0.04948391122425175], [0.03850038382630078, 0.0]],
    "Lambda": 0.44719798993705917,
    "tau_H_s": 0.2,
    "tau_Theta_s": 2.0,
    "theta_S_init": [1.0567879676818848, 1.0],
    "theta_m_EI_bounds": (0.1, 5.0),
    "theta_eta_a_bounds": (0.25, 4.0),
    "theta_syn": {
        "m_EE": 2.317018508911133,
        "m_EI": 1.0567879676818848,
        "m_IE": 3.7993736267089844,
        "m_II": 4.9821672439575195,
    },
}


R_E_NORM = 15.0
R_I_NORM = 10.0
EPS_DRIVE = 0.15


def _fixed_pattern(n: int = 10) -> np.ndarray:
    raw = np.array(
        [0.12, -0.08, 0.05, -0.10, 0.06, -0.07, 0.09, -0.04, 0.03, -0.06],
        dtype=float,
    )
    return raw[:n] - raw[:n].mean()


def _apply_drive_heterogeneity(model: jtfne.Model, eps: float) -> jtfne.Model:
    pattern = _fixed_pattern()
    base = np.asarray(model.params["emitter"].drive, dtype=np.float32)
    scaled = base * (1.0 + eps * pattern).astype(np.float32)
    return jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(scaled))


def _windowed_population_rates(spikes: np.ndarray, dt_ms: float, window_ms: float = 200.0):
    win = max(1, int(round(window_ms / dt_ms)))
    kernel = np.ones(win, dtype=np.float64) / win
    scale = 1000.0 / dt_ms
    r_e = np.convolve(spikes[:, :5].mean(axis=1), kernel, mode="full")[: spikes.shape[0]] * scale
    r_i = np.convolve(spikes[:, 5:].mean(axis=1), kernel, mode="full")[: spikes.shape[0]] * scale
    return r_e, r_i


def _recovery_metrics(r_e, r_i, *, dt_ms, r0_e, r0_i, perturb_start_ms, final_window_ms):
    d = np.array([1.0 / R_E_NORM, 1.0 / R_I_NORM])
    t0 = int(round(perturb_start_ms / dt_ms))
    post_e, post_i = r_e[t0:], r_i[t0:]
    peak = np.array([float(np.max(post_e)), float(np.max(post_i))])
    i0 = int(round(final_window_ms[0] / dt_ms))
    i1 = int(round(final_window_ms[1] / dt_ms))
    r_final = np.array([float(np.mean(r_e[i0:i1])), float(np.mean(r_i[i0:i1]))])
    eps = 1e-6

    def _R(rf, rp, r0, norm):
        return 1.0 - abs((rf - r0) / norm) / (abs((rp - r0) / norm) + eps)

    R_E = _R(r_final[0], peak[0], r0_e, R_E_NORM)
    R_I = _R(r_final[1], peak[1], r0_i, R_I_NORM)
    e_peak = d * (peak - np.array([r0_e, r0_i]))
    e_final = d * (r_final - np.array([r0_e, r0_i]))
    R_EI = 1.0 - float(np.linalg.norm(e_final) / (np.linalg.norm(e_peak) + eps))
    terminal = float(np.linalg.norm(e_final))
    return R_EI, terminal


def _mcc3_model():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=10, cell_types={"E": 0.5, "PV": 0.5})
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    specs = {
        name: jtfne.edge_parameter(pre={"cell_type": pre}, post={"cell_type": post}, bounds=(0.1, 5.0))
        for name, pre, post in (("m_EE", "E", "E"), ("m_EI", "E", "PV"), ("m_IE", "PV", "E"), ("m_II", "PV", "PV"))
    }
    model = jtfne.construct(cfg)
    model = _apply_drive_heterogeneity(model, EPS_DRIVE)
    model = _model_with_parameters(model, ETUDE_CONTROLLER["theta_syn"], specs)
    mei_mask = _edge_parameter_mask(model, "m_EI", specs["m_EI"])
    e_mask = np.array([str(lbl).startswith("E") for lbl in model.params["emitter"].labels], dtype=bool)
    return model, mei_mask, e_mask


def _population_hdp_params(mei_mask, e_mask, *, r0_e: float, r0_i: float):
    ctrl = ETUDE_CONTROLLER
    return {
        "K_HDP": 0.0,
        "K_ctrl": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "delta": 0.0,
        "hdp_rule": "population_vector_restoring",
        "h_state_locality": "population",
        "h_state_dim": 2,
        "controller_B": ctrl["B"],
        "controller_lambda": ctrl["Lambda"],
        "controller_tau_H_s": ctrl["tau_H_s"],
        "controller_tau_theta_s": ctrl["tau_Theta_s"],
        # Etude protocol: controller setpoints match measured pre-perturbation r0.
        "controller_rate_setpoint_E_hz": float(r0_e),
        "controller_rate_setpoint_I_hz": float(r0_i),
        "controller_theta_S_init": ctrl["theta_S_init"],
        "m_ei_edge_mask": mei_mask.astype(bool),
        "e_neuron_mask": e_mask.astype(bool),
        "theta_m_EI_bounds": ctrl["theta_m_EI_bounds"],
        "theta_eta_a_bounds": ctrl["theta_eta_a_bounds"],
    }


@pytest.mark.skipif(not ETUDE_METRICS.exists(), reason="committed Etude metrics bundle required")
def test_population_restoring_structural_invariants():
    model, mei_mask, e_mask = _mcc3_model()
    hp = _population_hdp_params(mei_mask, e_mask, r0_e=10.9, r0_i=9.14)
    runtime = jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list", jit=False, hdp_params=hp)
    sig = model.simulate(jtfne.simulation(duration_ms=500.0, dt_ms=0.1, seed=17, runtime=runtime))
    assert jnp.all(jnp.isfinite(sig.spikes))
    diag = model.last_hdp_diagnostics()
    assert diag is not None
    assert np.asarray(diag["H_final"]).shape == (2,)
    assert np.asarray(diag["theta_S_final"]).shape == (2,)
    assert sig.metadata["hdp"]["h_state"]["h_state_locality"] == "population"
    B = np.asarray(hp["controller_B"])
    assert np.linalg.matrix_rank(B) == 2


def test_population_continuation_rejected_explicitly():
    model, mei_mask, e_mask = _mcc3_model()
    hp = _population_hdp_params(mei_mask, e_mask, r0_e=10.9, r0_i=9.14)
    runtime = jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list", jit=False, hdp_params=hp)
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=17, runtime=runtime)
    with pytest.raises(ValueError, match="population H-state locality is not supported"):
        model.simulate(sim, return_state=True)


@pytest.mark.skipif(not ETUDE_METRICS.exists(), reason="committed Etude metrics bundle required")
@pytest.mark.slow
def test_population_restoring_etude_regression_metrics():
    """Scientific regression against committed Etude metrics (MVC #2, alpha_U=1.2)."""
    metrics = json.loads(ETUDE_METRICS.read_text())
    expected = metrics["mvc2_recovery"]

    model, mei_mask, e_mask = _mcc3_model()
    dt_ms = 0.1
    duration_ms = 15000.0
    perturb_ms = 3000.0
    burn_ms = 20.0
    drive = np.asarray(model.params["emitter"].drive, dtype=np.float32)

    def paradigm(scale: float):
        extra = (scale - 1.0) * drive
        events = []
        for i in range(drive.shape[0]):
            if abs(extra[i]) < 1e-12:
                continue
            events.append(
                {
                    "onset_ms": perturb_ms,
                    "duration_ms": duration_ms - perturb_ms,
                    "amplitude": float(extra[i]),
                    "target_indices": [i],
                    "is_drive_event": True,
                }
            )
        return jtfne.StimulusSchedule(events=tuple(events), n_neurons=drive.shape[0])

    sig_off_measure = model.simulate(
        jtfne.simulation(
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            seed=17,
            runtime=jtfne.RuntimeConfig(enable_hdp=False, recurrent_backend="edge_list", jit=False),
        ),
        paradigm=paradigm(1.2),
    )
    r_e_m, r_i_m = _windowed_population_rates(np.asarray(sig_off_measure.spikes), dt_ms)
    i0 = int(round(burn_ms / dt_ms))
    i1 = int(round(perturb_ms / dt_ms))
    r0_e = float(np.mean(r_e_m[i0:i1]))
    r0_i = float(np.mean(r_i_m[i0:i1]))

    def run(enable_hdp: bool, hp=None):
        runtime = jtfne.RuntimeConfig(
            enable_hdp=enable_hdp,
            recurrent_backend="edge_list",
            jit=False,
            hdp_params=hp or {},
        )
        sig = model.simulate(
            jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=17, runtime=runtime),
            paradigm=paradigm(1.2),
        )
        sp = np.asarray(sig.spikes)
        r_e, r_i = _windowed_population_rates(sp, dt_ms)
        return _recovery_metrics(
            r_e,
            r_i,
            dt_ms=dt_ms,
            r0_e=r0_e,
            r0_i=r0_i,
            perturb_start_ms=perturb_ms,
            final_window_ms=(13000.0, 15000.0),
        )

    r_ei_off, term_off = run(False)
    r_ei_scalar, _ = run(True, {"K_HDP": 0.01, "h_state_dim": 1})
    r_ei_vec, term_vec = run(
        True, _population_hdp_params(mei_mask, e_mask, r0_e=r0_e, r0_i=r0_i)
    )

    assert r_ei_off == pytest.approx(expected["off"]["R_EI"], rel=0.08, abs=0.05)
    assert r_ei_scalar == pytest.approx(expected["scalar"]["R_EI"], rel=0.08, abs=0.05)
    assert r_ei_vec == pytest.approx(expected["vector"]["R_EI"], rel=0.05, abs=0.03)
    assert term_off == pytest.approx(expected["off"]["terminal_error_weighted"], rel=0.08, abs=0.05)
    assert term_vec == pytest.approx(expected["vector"]["terminal_error_weighted"], rel=0.15, abs=0.02)
