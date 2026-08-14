"""0.4.17-C1 — synthetic-field estimator validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxfne.protocol_c.c1_validation import run_c1_synthetic_validation, write_c1_receipt
from jaxfne.protocol_c.estimator import estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec
from jaxfne.protocol_c.synthetic import (
    make_positions_1d,
    make_positions_2d,
    make_time_axis,
    noise_only_field,
    planar_traveling_wave,
    random_spatial_phases,
    standing_wave_field,
    synchronous_oscillation,
)

RECEIPT = Path("artifacts/protocol_c/c1_synthetic_validation_receipt.json")


@pytest.fixture(scope="module")
def spec():
    return load_protocol_spec()


@pytest.fixture(scope="module")
def grid_1d():
    duration_ms = 2000.0
    dt_ms = 0.5
    return make_positions_1d(24), make_time_axis(duration_ms, dt_ms), dt_ms


def test_c1_positive_planar_recovers_kinematics(grid_1d, spec):
    pos, time_s, dt_ms = grid_1d
    f_hz = 10.0
    k_vec = np.array([2.0 * np.pi * 3.0])
    phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=f_hz)
    gt = {"frequency_hz": f_hz, "k_vector": k_vec, "phase_velocity": 2.0 * np.pi * f_hz / k_vec[0]}
    est = estimate_traveling_wave(phi, pos, dt_ms=dt_ms, ground_truth=gt)
    assert est.classification == "TRAVELING_WAVE"
    tol = spec["synthetic_controls"]["positive_planar_wave"]["tolerances"]
    assert est.errors["epsilon_theta_deg"] <= tol["direction_deg"]
    assert est.errors["epsilon_f"] / f_hz <= tol["relative_frequency_error"]
    assert est.errors["epsilon_v"] <= tol["relative_velocity_error"]


def test_c1_opposite_directions_both_traveling(grid_1d):
    pos, time_s, dt_ms = grid_1d
    f_hz = 10.0
    km = 2.0 * np.pi * 3.0
    for k_vec in (np.array([km]), np.array([-km])):
        phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=f_hz)
        est = estimate_traveling_wave(phi, pos, dt_ms=dt_ms)
        assert est.classification == "TRAVELING_WAVE"


def test_c1_2d_non_axis_aligned_traveling():
    pos = make_positions_2d(6, 6)
    time_s = make_time_axis(2000.0, 0.5)
    k_vec = np.array([2.0 * np.pi * 0.8, 2.0 * np.pi * 0.5])
    phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0)
    est = estimate_traveling_wave(phi, pos, dt_ms=0.5)
    assert est.classification == "TRAVELING_WAVE"


@pytest.mark.parametrize(
    "factory,case_id",
    [
        (lambda pos, t: synchronous_oscillation(pos, t, frequency_hz=10.0), "sync"),
        (lambda pos, t: random_spatial_phases(pos, t, frequency_hz=10.0, seed=1), "random"),
        (lambda pos, t: noise_only_field(pos, t, amplitude=0.02, seed=2), "noise"),
        (
            lambda pos, t: standing_wave_field(pos, t, k_scalar=2.0 * np.pi * 3.0, frequency_hz=10.0),
            "standing",
        ),
    ],
)
def test_c1_negative_controls_classified_no_wave(factory, case_id, grid_1d):
    pos, time_s, dt_ms = grid_1d
    phi = factory(pos, time_s)
    est = estimate_traveling_wave(phi, pos, dt_ms=dt_ms)
    assert est.classification == "NO_WAVE", f"{case_id}: {est.quality_reasons}"


def test_c1_invariance_global_phase_shift(grid_1d):
    pos, time_s, dt_ms = grid_1d
    k_vec = np.array([2.0 * np.pi * 2.5])
    base = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0, phi0=0.0)
    shifted = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0, phi0=1.2)
    e0 = estimate_traveling_wave(base, pos, dt_ms=dt_ms)
    e1 = estimate_traveling_wave(shifted, pos, dt_ms=dt_ms)
    assert e0.classification == e1.classification == "TRAVELING_WAVE"
    cos_d = abs(float(np.dot(e0.direction, e1.direction)))
    assert cos_d > 0.99
    assert abs(e0.phase_velocity - e1.phase_velocity) / abs(e0.phase_velocity) < 0.05


def test_c1_invariance_amplitude_scaling(grid_1d):
    pos, time_s, dt_ms = grid_1d
    k_vec = np.array([2.0 * np.pi * 2.5])
    phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0, amplitude=1.0)
    scaled = 3.5 * phi
    e0 = estimate_traveling_wave(phi, pos, dt_ms=dt_ms)
    e1 = estimate_traveling_wave(scaled, pos, dt_ms=dt_ms)
    assert e0.classification == e1.classification == "TRAVELING_WAVE"
    assert np.linalg.norm(e0.wave_vector - e1.wave_vector) / np.linalg.norm(e0.wave_vector) < 0.05


def test_c1_invariance_translation(grid_1d):
    pos, time_s, dt_ms = grid_1d
    offset = np.array([[0.37]])
    pos_shift = pos + offset
    k_vec = np.array([2.0 * np.pi * 2.0])
    phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0)
    phi_s = planar_traveling_wave(pos_shift, time_s, k_vector=k_vec, frequency_hz=10.0)
    e0 = estimate_traveling_wave(phi, pos, dt_ms=dt_ms)
    e1 = estimate_traveling_wave(phi_s, pos_shift, dt_ms=dt_ms)
    assert e0.classification == e1.classification == "TRAVELING_WAVE"
    assert abs(np.linalg.norm(e0.wave_vector) - np.linalg.norm(e1.wave_vector)) / np.linalg.norm(e0.wave_vector) < 0.08
    assert abs(e0.phase_velocity - e1.phase_velocity) / e0.phase_velocity < 0.08


def test_c1_invariance_rotation_2d():
    pos = make_positions_2d(6, 6)
    time_s = make_time_axis(2000.0, 0.5)
    k_vec = np.array([2.0 * np.pi * 0.8, 2.0 * np.pi * 0.5])
    phi = planar_traveling_wave(pos, time_s, k_vector=k_vec, frequency_hz=10.0)
    theta = np.pi / 6.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos_rot = pos @ rot.T
    k_rot = k_vec @ rot.T
    phi_rot = planar_traveling_wave(pos_rot, time_s, k_vector=k_rot, frequency_hz=10.0)
    gt0 = {"frequency_hz": 10.0, "k_vector": k_vec, "phase_velocity": 2.0 * np.pi * 10.0 / np.linalg.norm(k_vec)}
    gt1 = {"frequency_hz": 10.0, "k_vector": k_rot, "phase_velocity": 2.0 * np.pi * 10.0 / np.linalg.norm(k_rot)}
    e0 = estimate_traveling_wave(phi, pos, dt_ms=0.5, ground_truth=gt0)
    e1 = estimate_traveling_wave(phi_rot, pos_rot, dt_ms=0.5, ground_truth=gt1)
    assert e0.classification == e1.classification == "TRAVELING_WAVE"
    tol = load_protocol_spec()["synthetic_controls"]["positive_planar_wave"]["tolerances"]
    assert e0.errors["epsilon_k"] <= tol["relative_frequency_error"] * 2
    assert e1.errors["epsilon_k"] <= tol["relative_frequency_error"] * 2


def test_c1_full_validation_receipt_passes():
    receipt = run_c1_synthetic_validation(package_head="test_c1")
    assert receipt["summary"]["c1_pass"] is True
    for case in receipt["cases"]:
        if case["expected_classification"]:
            assert case["expected_match"], case["case_id"]


def test_c1_frozen_receipt_in_repo():
    assert RECEIPT.is_file()
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["checkpoint"] == "C1"
    assert receipt["status"] == "FROZEN"
    assert receipt["summary"]["c1_pass"] is True
