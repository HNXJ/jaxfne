"""A-2 detector sensitivity characterisation - frozen-lattice execution.

Runs the three predeclared stages from p2v_a2_spec.json against the
D1-repaired estimator with unchanged thresholds. Write-once receipt.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from jaxfne.protocol_c.c3_execution import _ordered_arc_positions_mm, replay_c3_cell
from jaxfne.protocol_c.c3_protocol import load_c3_spec
from jaxfne.protocol_c.estimator import _bandpass_rows, estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
A2_DIR = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a2_sensitivity_floor"
SPEC_PATH = A2_DIR / "p2v_a2_spec.json"
RECEIPT_PATH = A2_DIR / "p2v_a2_receipt.json"

RNG = np.random.default_rng(20260816)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def estimator_module_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD:jaxfne/protocol_c/estimator.py"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def synthetic_wave(positions: np.ndarray, *, mode_m: int, f_hz: float, phi0: float, dt_ms: float, duration_ms: float) -> np.ndarray:
    x = np.asarray(positions, dtype=np.float64).reshape(-1)
    R = 1.0
    k = float(mode_m) / R
    t = np.arange(int(duration_ms / dt_ms), dtype=np.float64) * dt_ms / 1000.0
    return np.cos(k * x[None, :] - 2.0 * np.pi * f_hz * t[:, None] + phi0)


def run_stage_s1(spec: dict, positions: np.ndarray, proto_spec: dict) -> list[dict]:
    wave = spec["stage_S1_amplitude_noise_floor"]["wave"]
    rows = []
    for m in wave["mode_m"]:
        for A in spec["stage_S1_amplitude_noise_floor"]["amplitude_A"]:
            for sigma in spec["stage_S1_amplitude_noise_floor"]["noise_sigma_abs"]:
                phi = A * synthetic_wave(
                    positions, mode_m=int(m), f_hz=10.0, phi0=0.0,
                    dt_ms=0.5, duration_ms=2000.0,
                )
                if sigma > 0.0:
                    phi = phi + RNG.normal(0.0, sigma, size=phi.shape)
                est = estimate_traveling_wave(
                    phi, positions, dt_ms=0.5, spec=proto_spec,
                    ground_truth={"frequency_hz": 10.0, "k_vector": np.array([float(m) / 1.0])},
                )
                rows.append(
                    {
                        "mode_m": int(m),
                        "amplitude": float(A),
                        "noise_sigma": float(sigma),
                        **est.to_dict(),
                    }
                )
    return rows


def run_stage_s2(spec: dict, proto_spec: dict, c3_spec: dict) -> dict:
    emb = spec["stage_S2_c3_regime_embedding"]
    E = 24
    positions = _ordered_arc_positions_mm(E, radius_mm=1.0)
    bands = {"band": tuple(proto_spec["preregistered_estimator_parameters"]["frequency_band_hz"])}
    fs_hz = 2000.0
    results = {"cells": []}
    for cell in emb["carrier_cells"]:
        rep = replay_c3_cell(cell["condition_id"], int(cell["seed"]), spec=c3_spec)
        phi_c3 = np.asarray(rep["V_m"], dtype=np.float64)
        phi_c3_bp = _bandpass_rows(phi_c3, fs_hz, bands["band"])
        rms = float(np.sqrt(np.mean(phi_c3_bp**2)))
        cell_rows = []
        for gamma in emb["embedding"]["gamma"]:
            for phi0 in emb["embedding"]["phi0_wave_rad"]:
                wave = gamma * rms * synthetic_wave(
                    positions, mode_m=1, f_hz=10.0, phi0=float(phi0),
                    dt_ms=0.5, duration_ms=2000.0,
                )
                phi_mix = phi_c3_bp + wave
                est = estimate_traveling_wave(
                    phi_mix, positions, dt_ms=0.5, spec=proto_spec,
                    ground_truth={"frequency_hz": 10.0, "k_vector": np.array([1.0])},
                )
                cell_rows.append(
                    {
                        "gamma": float(gamma),
                        "phi0_wave_rad": float(phi0),
                        "rms_c3_band": rms,
                        **est.to_dict(),
                    }
                )
        gamma_star = {}
        for phi0 in emb["embedding"]["phi0_wave_rad"]:
            tw = [r for r in cell_rows if r["phi0_wave_rad"] == float(phi0) and r["classification"] == "TRAVELING_WAVE"]
            gamma_star[f"phi0_{phi0}"] = min(r["gamma"] for r in tw) if tw else "NO_FLIP_WITHIN_LATTICE"
        results["cells"].append(
            {
                "condition_id": cell["condition_id"],
                "seed": int(cell["seed"]),
                "frozen_reason": cell["frozen_reason"],
                "rms_c3_band": rms,
                "gamma_star": gamma_star,
                "cases": cell_rows,
            }
        )
    groups = [c["gamma_star"] for c in results["cells"]]
    results["summary"] = {"cells": groups}
    return results


def run_stage_s3(spec: dict, positions: np.ndarray, proto_spec: dict) -> list[dict]:
    rows = []
    for dur in spec["stage_S3_duration_sites"]["duration_ms"]:
        for ns in spec["stage_S3_duration_sites"]["n_sites"]:
            stride = 24 // ns
            idx = np.arange(0, 24, stride, dtype=int)
            pos_sub = positions[idx]
            phi = synthetic_wave(positions, mode_m=1, f_hz=10.0, phi0=0.0, dt_ms=0.5, duration_ms=float(dur))
            phi_sub = phi[:, idx]
            est = estimate_traveling_wave(
                phi_sub, pos_sub, dt_ms=0.5, spec=proto_spec,
                ground_truth={"frequency_hz": 10.0, "k_vector": np.array([1.0])},
            )
            rows.append({"duration_ms": float(dur), "n_sites": int(ns), **est.to_dict()})
    return rows


def run_all() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    proto_spec = load_protocol_spec()
    c3_spec = load_c3_spec()
    positions = _ordered_arc_positions_mm(24, radius_mm=1.0)
    s1 = run_stage_s1(spec, positions, proto_spec)
    s2 = run_stage_s2(spec, proto_spec, c3_spec)
    s3 = run_stage_s3(spec, positions, proto_spec)

    s1_first_gate = {}
    for r in s1:
        if r["classification"] != "TRAVELING_WAVE":
            gate = r["quality_reasons"][0]
            s1_first_gate.setdefault((r["mode_m"], r["amplitude"], r["noise_sigma"]), gate)
    receipt = {
        "schema": "jaxfne.protocol_c.p2v_a2_receipt.v1",
        "protocol_id": "protocol_c_p2v_a2",
        "phase": "post-freeze reviewer-motivated validation",
        "checkpoint": "A-2",
        "status": "FROZEN",
        "write_once": True,
        "package_head": git_head(),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "estimator_module_sha": estimator_module_sha(),
        "stage_S1": {"n_cases": len(s1), "cases": s1},
        "stage_S2": s2,
        "stage_S3": {"n_cases": len(s3), "cases": s3},
        "title": "ESTIMATOR SENSITIVITY CHARACTERISATION: gates, C3-embedding flip thresholds, duration/site limits",
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {RECEIPT_PATH}")
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"A-2 done | S1 {len(s1)} cases | S2 cells: {[c['condition_id']+':'+str(c['seed']) for c in s2['cells']]} | S3 {len(s3)} cases")
    for c in s2["cells"]:
        print(f"  {c['condition_id']}:{c['seed']} rms={c['rms_c3_band']:.4f} gamma_star={c['gamma_star']}")
    return receipt


if __name__ == "__main__":
    run_all()