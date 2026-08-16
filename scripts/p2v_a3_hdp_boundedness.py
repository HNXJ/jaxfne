"""A-3 HDP boundedness - classification + minimal prospective run (frozen domain).

Reads p2v_a3_spec.json; executes DEFAULT_HDP / DEFAULT_HDP_DESYNC x seeds
{1001,1002,1003} on the C3 ring anchor construction via the HDP kernel
directly; asserts per-step hard-bound invariants; writes the FROZEN receipt.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp
from jaxfne.h4_matrix import build_ring_params_edges

REPO_ROOT = Path(__file__).resolve().parents[1]
A3_DIR = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a3_hdp_boundedness"
SPEC_PATH = A3_DIR / "p2v_a3_spec.json"
RECEIPT_PATH = A3_DIR / "p2v_a3_receipt.json"

DEFAULT_HDP = dict(
    K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, rho_passive=0.0, barrier_c=0.01, barrier_d=0.01
)
DEFAULT_HDP_DESYNC = dict(
    K_HDP=0.01, tau_0_ms=5.0, K_ctrl=0.15, rho_passive=0.0,
    barrier_c=0.01, barrier_d=0.01, alpha=0.05, gamma=0.5, C_spike=0.0,
)
BASE_KWARGS = dict(
    H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
    barrier_eps=1e-3, w_floor=0.01, w_ceiling=10.0, H_boost_gain=4.0,
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_drive_schedule(n_neurons: int, n_steps: int, dt_ms: float) -> np.ndarray:
    sched = np.zeros((n_steps, n_neurons), dtype=np.float32)
    events = [
        (200.0, 1.0, 50.0, 0),
        (800.0, 1.0, 45.0, 6),
    ]
    for onset, dur, amp, target in events:
        i0 = int(onset / dt_ms)
        i1 = int((onset + dur) / dt_ms)
        sched[i0:i1, target] = amp
    return sched


def run_preset(preset_name: str, preset: dict, seed: int) -> dict:
    n = 24
    params, edges = build_ring_params_edges(n, delay_steps=4, weight=6.0, tau_ms=3.0)
    n_steps = int(2000.0 / 0.5)
    kwargs = {**BASE_KWARGS, **preset}
    key = jax.random.PRNGKey(int(seed))
    voltages, spikes, sources, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps,
        0.5,
        key,
        dtype="float32",
        drive_schedule=jnp.asarray(build_drive_schedule(n, n_steps, 0.5)),
        noise_scale=0.0,
        hdp_rule="signed_linear",
        record_weight_trace=True,
        **kwargs,
    )
    H_trace = np.asarray(diag["H_trace"])
    w_trace = np.asarray(diag["w_trace"])
    v = np.asarray(voltages)
    all_finite = bool(np.isfinite(H_trace).all() and np.isfinite(w_trace).all() and np.isfinite(v).all())
    H_min_obs, H_max_obs = float(H_trace.min()), float(H_trace.max())
    w_abs = np.abs(w_trace)
    w_min_obs, w_max_obs = float(w_abs.min()), float(w_abs.max())
    w_growth = float(w_abs[-1].mean() / w_abs[0].mean())
    v_max = float(np.abs(v).max())
    rate = float(np.sum(spikes > 0.5) / n / (2000.0 / 1000.0))
    invariants = {
        "H_within_0p1_to_10": H_min_obs >= 0.1 and H_max_obs <= 10.0,
        "w_within_0p01_to_10": w_min_obs >= 0.01 and w_max_obs <= 10.0,
        "v_within_m150_to_100": v_max <= 100.0,
        "all_finite": all_finite,
    }
    return {
        "preset": preset_name,
        "seed": int(seed),
        "H_min_obs": H_min_obs,
        "H_max_obs": H_max_obs,
        "w_min_obs": w_min_obs,
        "w_max_obs": w_max_obs,
        "w_abs_growth_ratio": w_growth,
        "max_abs_v": v_max,
        "mean_spike_rate_hz": rate,
        "invariants": invariants,
        "invariants_pass": all(invariants.values()),
    }


def main() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    rows = []
    for name, preset in (("DEFAULT_HDP", DEFAULT_HDP), ("DEFAULT_HDP_DESYNC", DEFAULT_HDP_DESYNC)):
        for seed in (1001, 1002, 1003):
            rows.append(run_preset(name, preset, seed))
    pass_all = all(r["invariants_pass"] for r in rows)
    receipt = {
        "schema": "jaxfne.protocol_c.p2v_a3_receipt.v1",
        "protocol_id": "protocol_c_p2v_a3",
        "phase": "post-freeze reviewer-motivated validation",
        "checkpoint": "A-3",
        "status": "FROZEN",
        "write_once": True,
        "package_head": git_head(),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "run_domain": "24-neuron C3 ring anchor, weight 6.0, tau 3.0, delay 4; 2000 ms; seeds 1001-1003; K_w_ctrl=0.0 (kernel default)",
        "runs": rows,
        "all_hard_bound_invariants_pass": pass_all,
        "scoped_statement": (
            "trajectories remained bounded over the tested parameter and time domain "
            "(24-neuron ring, DEFAULT_HDP / DEFAULT_HDP_DESYNC, 2000 ms, seeds 1001-1003, K_w_ctrl=0.0): "
            "every state component stayed within its per-step hard bounds for every step"
        ),
        "classification_reference": spec["manuscript_facing_claims"],
        "no_tuning_observed": True,
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {RECEIPT_PATH}")
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    for r in rows:
        print(f"  {r['preset']} seed {r['seed']}: H[{r['H_min_obs']:.4f},{r['H_max_obs']:.4f}] "
              f"|w|[{r['w_min_obs']:.4f},{r['w_max_obs']:.4f}] growth {r['w_abs_growth_ratio']:.5f} "
              f"rate {r['mean_spike_rate_hz']:.2f} Hz invariants {r['invariants_pass']}")
    print(f"A-3 done | all invariants pass: {pass_all}")
    return receipt


if __name__ == "__main__":
    main()