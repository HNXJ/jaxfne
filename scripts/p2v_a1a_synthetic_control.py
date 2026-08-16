"""A-1a post-freeze estimator validation: synthetic recovery grid + negatives.

Runs the frozen publication estimator on injected synthetic fields with known
parameters over the frozen C3 geometry, verifying classification and recovery
of frequency, |k|, direction/sign, phase velocity, and fit quality.

Phase: post-freeze reviewer-motivated validation. No neural simulation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from jaxfne.protocol_c.estimator import estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec
from jaxfne.protocol_c.synthetic import (
    make_time_axis,
    noise_only_field,
    planar_traveling_wave,
    random_spatial_phases,
    standing_wave_field,
    synchronous_oscillation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a1a_synthetic_control" / "p2v_a1a_receipt.json"
SPEC_PATH = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a1a_synthetic_control" / "p2v_a1a_spec.json"

FROZEN_C3 = {
    "n_neurons": 24,
    "ring_radius_mm": 1.0,
    "dt_ms": 0.5,
    "duration_ms": 2000.0,
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def ordered_arc_positions(n: int, radius_mm: float) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)
    return (float(radius_mm) * theta).reshape(-1, 1)


def _fisher_yates_permutation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    perm = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _pass_tolerances(errors: dict[str, float], tol: dict) -> tuple[bool, dict[str, bool]]:
    f_err = abs(float(errors["epsilon_f"]))
    f_pass = f_err <= max(float(tol["absolute_frequency_error_hz"]), float(tol["relative_frequency_error"]) * abs(float(errors.get("f_true", 0.0))) if float(errors.get("f_true", 0.0)) > 0 else 0.0)
    k_pass = float(errors["epsilon_k"]) <= float(tol["relative_k_norm_error"])
    th_pass = float(errors["epsilon_theta_deg"]) <= float(tol["direction_deg_error"])
    v_pass = float(errors["epsilon_v"]) <= float(tol["relative_velocity_error"])
    checks = {
        "frequency": bool(f_pass),
        "k_norm": bool(k_pass),
        "direction": bool(th_pass),
        "velocity": bool(v_pass),
    }
    return all(checks.values()), checks


def run_a1a() -> dict:
    n = FROZEN_C3["n_neurons"]
    r = FROZEN_C3["ring_radius_mm"]
    dt_ms = FROZEN_C3["dt_ms"]
    dur_ms = FROZEN_C3["duration_ms"]
    positions = ordered_arc_positions(n, r)
    t = make_time_axis(dur_ms, dt_ms)
    spec = load_protocol_spec()

    tol = {
        "relative_frequency_error": 0.05,
        "absolute_frequency_error_hz": 0.5,
        "relative_k_norm_error": 0.1,
        "direction_deg_error": 15.0,
        "relative_velocity_error": 0.1,
        "phase_fit_r2_min": 0.6,
        "spatial_coherence_min": 0.55,
    }

    cases: list[dict] = []
    rng = np.random.default_rng(20260815)

    freqs = [8.5, 10.0, 12.5]
    modes = [1, 2]
    signs = [1, -1]
    phi0s = [0.0, 1.7]
    noise_sigmas = [0.0, 0.25]

    for f in freqs:
        for m in modes:
            for sgn in signs:
                for phi0 in phi0s:
                    for sigma_frac in noise_sigmas:
                        k_true = sgn * m / r
                        amp = 1.0
                        phi = planar_traveling_wave(
                            positions, t, k_vector=np.array([k_true]), frequency_hz=f,
                            amplitude=amp, phi0=phi0,
                        )
                        if sigma_frac > 0.0:
                            phi = phi + rng.normal(0.0, sigma_frac * amp, size=phi.shape)
                        gt = {
                            "frequency_hz": f,
                            "k_vector": np.array([k_true]),
                            "phase_velocity": 2.0 * np.pi * f / abs(k_true),
                        }
                        est = estimate_traveling_wave(phi, positions, dt_ms=dt_ms, spec=spec, ground_truth=gt)
                        met, checks = _pass_tolerances(est.errors, tol)
                        recovery = {
                            "classification_correct": est.classification == "TRAVELING_WAVE",
                            "finite": bool(est.finite_status),
                            "r2_gate": float(est.phase_fit_r2) >= float(tol["phase_fit_r2_min"]),
                            "coherence_gate": float(est.spatial_coherence) >= float(tol["spatial_coherence_min"]),
                            "direction_convention_consistent": bool(
                                est.classification == "TRAVELING_WAVE"
                                and float(est.wave_vector[0]) * k_true < 0.0
                            ),
                        }
                        ok = (
                            recovery["classification_correct"]
                            and recovery["finite"]
                            and recovery["r2_gate"]
                            and recovery["coherence_gate"]
                            and recovery["direction_convention_consistent"]
                            and met
                        )
                        cases.append({
                            "case_id": f"pos_f{f}_m{m}_s{sgn}_p{int(phi0*10)}_n{int(sigma_frac*100)}",
                            "kind": "positive",
                            "ground_truth": {"frequency_hz": f, "k_mm1": k_true, "mode": m, "sign": sgn, "phase_velocity_mm_per_s": 2.0 * np.pi * f / abs(k_true)},
                            "estimator": est.to_dict(),
                            "tolerance_checks": checks,
                            "recovery": recovery,
                            "pass": ok,
                        })

    negatives_spec = [
        ("sync_oscillation", lambda: synchronous_oscillation(positions, t, frequency_hz=10.0, amplitude=1.0)),
        ("standing_wave", lambda: standing_wave_field(positions, t, k_scalar=1.0, frequency_hz=10.0, amplitude=1.0, axis=0)),
        ("random_spatial_phases", lambda: random_spatial_phases(positions, t, frequency_hz=10.0, amplitude=1.0, seed=7)),
        ("noise_only", lambda: noise_only_field(positions, t, amplitude=0.05, seed=11)),
    ]
    for cid, gen in negatives_spec:
        est = estimate_traveling_wave(gen(), positions, dt_ms=dt_ms, spec=spec)
        cases.append({
            "case_id": cid,
            "kind": "negative",
            "expected_classification": "NO_WAVE",
            "observed_reason_finding": est.quality_reasons[0] if est.quality_reasons else "",
            "estimator": est.to_dict(),
            "pass": bool(est.classification == "NO_WAVE"),
        })

    shuffled = np.asarray(positions)[_fisher_yates_permutation(24, 4242)]
    est = estimate_traveling_wave(planar_traveling_wave(shuffled, t, k_vector=np.array([1.0]), frequency_hz=10.0), positions, dt_ms=dt_ms, spec=spec)
    cases.append({
        "case_id": "shuffled_coordinates_true_wave",
        "kind": "negative",
        "expected_classification": "NO_WAVE",
        "observed_reason_finding": est.quality_reasons[0] if est.quality_reasons else "",
        "estimator": est.to_dict(),
        "pass": bool(est.classification == "NO_WAVE"),
    })

    positives = [c for c in cases if c["kind"] == "positive"]
    negatives = [c for c in cases if c["kind"] == "negative"]
    receipt = {
        "schema": "jaxfne.protocol_c.p2v_a1a_receipt.v1",
        "protocol_id": "protocol_c_p2v_a1a",
        "phase": "post-freeze_reviewer_motivated_validation",
        "checkpoint": "A-1a",
        "status": "FROZEN",
        "write_once": True,
        "package_head": _git_head(),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "separation": "estimator-only; no neural simulation; not evidence of wave generation",
        "geometry": {"n_neurons": n, "ring_radius_mm": r, "coordinates": "ordered_arc_length_shape(N,1)"},
        "estimator_source": "artifacts/protocol_c/c0_wave_protocol_spec.json preregistered_estimator_parameters",
        "n_positive_cases": len(positives),
        "n_negative_cases": len(negatives),
        "recovery_tolerances": tol,
        "summary": {
            "all_positives_pass": bool(len(positives) > 0 and all(c["pass"] for c in positives)),
            "all_negatives_pass": bool(len(negatives) > 0 and all(c["pass"] for c in negatives)),
            "a1a_pass": bool(len(positives) > 0 and all(c["pass"] for c in positives) and all(c["pass"] for c in negatives)),
            "frequency_range_recovered": [min(freqs), max(freqs)],
            "velocity_range_recovered_mm_per_s": [
                round(2.0 * np.pi * 8.5 / 1.0, 3),
                round(2.0 * np.pi * 12.5 / 2.0, 3),
            ],
        },
        "cases": cases,
    }
    if not receipt["summary"]["a1a_pass"]:
        fail = [c["case_id"] for c in cases if not c["pass"]]
        raise AssertionError(f"A-1a cases failing: {fail}")
    return receipt


def write_a1a_receipt() -> dict:
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"write-once artifact exists: {RECEIPT_PATH}")
    receipt = run_a1a()
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    r = write_a1a_receipt()
    s = r["summary"]
    print(f"A-1a PASS | positives {s['n_positive_cases'] if False else len(r['cases'])} cases | all_pos {s['all_positives_pass']} all_neg {s['all_negatives_pass']}")
    print(f" wrote {RECEIPT_PATH}")