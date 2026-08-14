"""C1 prospective synthetic validation runner and receipt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from jaxfne.io import json_safe
from jaxfne.protocol_c.estimator import WaveEstimate, estimate_traveling_wave
from jaxfne.protocol_c.protocol import PROTOCOL_ID, load_protocol_spec
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

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "artifacts" / "protocol_c"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _ground_truth(f_hz: float, k_vector: np.ndarray) -> dict[str, Any]:
    k = np.asarray(k_vector, dtype=np.float64).reshape(-1)
    omega = 2.0 * np.pi * f_hz
    kn = float(np.linalg.norm(k))
    return {
        "frequency_hz": float(f_hz),
        "k_vector": k.tolist(),
        "phase_velocity": float(omega / kn) if kn > 1e-12 else float("inf"),
    }


def _case_result(case_id: str, est: WaveEstimate, *, expected: str | None = None) -> dict[str, Any]:
    row = {"case_id": case_id, "expected_classification": expected, **est.to_dict()}
    if expected is not None:
        row["expected_match"] = est.classification == expected
    return row


def run_c1_synthetic_validation(*, package_head: str | None = None) -> dict[str, Any]:
    spec = load_protocol_spec()
    tol = spec["synthetic_controls"]["positive_planar_wave"]["tolerances"]
    f_hz = 10.0
    duration_ms = 2000.0
    dt_ms = 0.5
    time_s = make_time_axis(duration_ms, dt_ms)
    results: list[dict[str, Any]] = []

    # --- Positive planar waves ---
    pos_1d = make_positions_1d(24, length=1.0)
    k_mag = 2.0 * np.pi * 3.0
    for case_id, k_vec in [
        ("planar_1d_plus_k", np.array([k_mag])),
        ("planar_1d_minus_k", np.array([-k_mag])),
    ]:
        phi = planar_traveling_wave(pos_1d, time_s, k_vector=k_vec, frequency_hz=f_hz)
        gt = _ground_truth(f_hz, k_vec)
        est = estimate_traveling_wave(phi, pos_1d, dt_ms=dt_ms, ground_truth=gt)
        results.append(_case_result(case_id, est, expected="TRAVELING_WAVE"))

    pos_2d = make_positions_2d(6, 6, lx=1.0, ly=1.0)
    k_diag = np.array([2.0 * np.pi * 0.8, 2.0 * np.pi * 0.5])
    phi2 = planar_traveling_wave(pos_2d, time_s, k_vector=k_diag, frequency_hz=f_hz)
    est2 = estimate_traveling_wave(
        phi2, pos_2d, dt_ms=dt_ms, ground_truth=_ground_truth(f_hz, k_diag)
    )
    results.append(_case_result("planar_2d_diagonal_k", est2, expected="TRAVELING_WAVE"))

    # --- Negative controls ---
    phi_sync = synchronous_oscillation(pos_1d, time_s, frequency_hz=f_hz)
    est_sync = estimate_traveling_wave(phi_sync, pos_1d, dt_ms=dt_ms)
    results.append(_case_result("sync_k_zero", est_sync, expected="NO_WAVE"))

    phi_rand = random_spatial_phases(pos_1d, time_s, frequency_hz=f_hz, seed=3)
    est_rand = estimate_traveling_wave(phi_rand, pos_1d, dt_ms=dt_ms)
    results.append(_case_result("spatially_random_phase", est_rand, expected="NO_WAVE"))

    phi_noise = noise_only_field(pos_1d, time_s, amplitude=0.02, seed=4)
    est_noise = estimate_traveling_wave(phi_noise, pos_1d, dt_ms=dt_ms)
    results.append(_case_result("noise_only", est_noise, expected="NO_WAVE"))

    phi_stand = standing_wave_field(pos_1d, time_s, k_scalar=k_mag, frequency_hz=f_hz, axis=0)
    est_stand = estimate_traveling_wave(phi_stand, pos_1d, dt_ms=dt_ms)
    results.append(_case_result("standing_wave", est_stand, expected="NO_WAVE"))

    positives = [r for r in results if r["case_id"].startswith("planar")]
    negatives = [r for r in results if not r["case_id"].startswith("planar")]

    pos_pass = all(r["expected_match"] for r in positives)
    neg_pass = all(r["expected_match"] for r in negatives)

    pos_metric_pass = True
    for r in positives:
        err = r.get("errors", {})
        if err.get("epsilon_theta_deg", 999) > tol["direction_deg"]:
            pos_metric_pass = False
        if err.get("epsilon_f", 999) / f_hz > tol["relative_frequency_error"]:
            pos_metric_pass = False
        if err.get("epsilon_v", 999) > tol["relative_velocity_error"]:
            pos_metric_pass = False

    receipt = {
        "schema": "jaxfne.protocol_c.c1_synthetic_validation_receipt.v1",
        "checkpoint": "C1",
        "status": "FROZEN",
        "protocol_id": PROTOCOL_ID,
        "package_head": package_head or _git_head(),
        "question": "Does W_hat correctly identify known wave and non-wave fields?",
        "scope": "synthetic_fields_only_no_neural_simulation",
        "tolerances_from_c0": tol,
        "cases": results,
        "summary": {
            "positive_expected_match": pos_pass,
            "negative_expected_match": neg_pass,
            "positive_metric_within_c0_tolerances": pos_metric_pass,
            "c1_pass": bool(pos_pass and neg_pass and pos_metric_pass),
        },
        "interpretation": (
            "C1 validates estimator discrimination on synthetic Phi only; "
            "it is not evidence that jaxfne neural dynamics generate waves."
        ),
        "next_checkpoint": "C2",
    }
    return json_safe(receipt)


def write_c1_receipt(path: Path | None = None) -> dict[str, Any]:
    receipt = run_c1_synthetic_validation()
    out = path or BUNDLE_ROOT / "c1_synthetic_validation_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
