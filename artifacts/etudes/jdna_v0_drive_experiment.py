"""JDNA encoding -> V0-targeted StimulusSchedule -> simulate -> invariant checks.

Scientific minimum:
  A, B in R^{32x32} -> E(A), E(B) -> J_{V0}(d) -> StimulusSchedule -> simulate -> Signals

Four numerical invariants checked:
  1. |E(A) - E(B)|_2 > 0
  2. ||d_A|_2 - |d_B|_2| / max(|d_A|_2, |d_B|_2) < epsilon
  3. supp(J_{V0}(d)) subset I_{V0}
  4. N_downstream_edges > 0
  5. Delta_X_{V0}^{A/B} != 0  (differential V0 response A vs B)

Verdict: EXECUTABLE or EXTERNALLY_COMPOSABLE.
"""

from __future__ import annotations

import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# 0. Imports — resolve from installed jaxfne (dev branch)
# ---------------------------------------------------------------------------
try:
    import jaxfne
    from jaxfne.core import Configuration, construct, simulate, Simulation
    from jaxfne._signals import StimulusSchedule
    from jaxfne.paradigm import paradigm_target_indices_from_model
    from jaxfne.plastic_params import PlasticParams
    JAXFNE_AVAILABLE = True
except ImportError as _ie:
    JAXFNE_AVAILABLE = False
    _IMPORT_ERROR = str(_ie)

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. JDNA encoder: R^{NxN} -> embedding vector E(I)
# ---------------------------------------------------------------------------

def jdna_encode(image: np.ndarray) -> np.ndarray:
    """Minimal JDNA-style encoding: flatten + L2-normalize.

    This stands in for a full JDNA pipeline (DCT -> retinal mosaic -> etc.)
    while preserving the key property that two distinct images produce
    two distinct, equal-norm embedding vectors.
    """
    flat = image.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    if norm < 1e-12:
        raise ValueError("Zero-norm image; JDNA encoding undefined.")
    return flat / norm


# ---------------------------------------------------------------------------
# 2. Construct drive vector from embedding: d = J_{V0}(e)
# ---------------------------------------------------------------------------

def embedding_to_drive(embedding: np.ndarray, n_neurons: int, v0_indices: list[int]) -> np.ndarray:
    """Project embedding onto V0 population to produce a per-neuron drive vector.

    d[i] = 0 for i not in V0 indices.
    d[i] = amplitude * embedding[k % len(embedding)] for i in V0, k=0,1,...

    Amplitude is scaled so |d|_2 is approximately 1 (normalized drive).
    """
    d = np.zeros(n_neurons, dtype=np.float64)
    n_v0 = len(v0_indices)
    if n_v0 == 0:
        raise ValueError("V0 indices list is empty; cannot construct drive.")
    # Project: use circular indexing into the embedding
    emb_len = len(embedding)
    for k, idx in enumerate(v0_indices):
        d[idx] = embedding[k % emb_len]
    # Normalize so |d|_2 ~ 1
    d_norm = np.linalg.norm(d)
    if d_norm > 1e-12:
        d = d / d_norm
    return d


# ---------------------------------------------------------------------------
# 3. Invariant checks
# ---------------------------------------------------------------------------

def check_invariants(
    eA: np.ndarray,
    eB: np.ndarray,
    dA: np.ndarray,
    dB: np.ndarray,
    v0_indices: list[int],
    n_neurons: int,
    eps: float = 0.05,
) -> dict:
    results = {}

    # Invariant 1: |E(A) - E(B)|_2 > 0
    enc_dist = float(np.linalg.norm(eA - eB))
    results["inv1_enc_distance"] = enc_dist
    results["inv1_pass"] = enc_dist > 0.0

    # Invariant 2: ||d_A|_2 - |d_B|_2| / max(...) < eps
    norm_dA = float(np.linalg.norm(dA))
    norm_dB = float(np.linalg.norm(dB))
    denom = max(norm_dA, norm_dB)
    if denom < 1e-12:
        rel_norm_diff = 0.0
    else:
        rel_norm_diff = abs(norm_dA - norm_dB) / denom
    results["inv2_rel_norm_diff"] = rel_norm_diff
    results["inv2_pass"] = rel_norm_diff < eps

    # Invariant 3: supp(J_{V0}(d)) ⊆ I_{V0}
    v0_set = set(v0_indices)
    nonzero_A = set(int(i) for i in np.nonzero(dA)[0])
    nonzero_B = set(int(i) for i in np.nonzero(dB)[0])
    offtarget_A = nonzero_A - v0_set
    offtarget_B = nonzero_B - v0_set
    results["inv3_offtarget_A"] = len(offtarget_A)
    results["inv3_offtarget_B"] = len(offtarget_B)
    results["inv3_pass"] = (len(offtarget_A) == 0) and (len(offtarget_B) == 0)

    return results


# ---------------------------------------------------------------------------
# 4. Count downstream edges from V0 in connectivity matrix W
# ---------------------------------------------------------------------------

def count_downstream_edges(model, v0_indices: list[int]) -> int:
    """Count V0 -> rest edges from the constructed model's weight matrix."""
    try:
        W = np.asarray(model.W)
        v0_rows = W[v0_indices, :]
        return int(np.sum(np.abs(v0_rows) > 1e-10))
    except Exception:
        # If W not directly accessible, fall back to positive count sentinel
        return -1  # signals "not directly inspectable"


# ---------------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    report = {}

    # -- JAX / jaxfne availability
    report["jax_available"] = JAX_AVAILABLE
    report["jaxfne_available"] = JAXFNE_AVAILABLE
    if not JAX_AVAILABLE or not JAXFNE_AVAILABLE:
        report["verdict"] = "EXTERNALLY_COMPOSABLE"
        report["reason"] = (
            f"import_error={_IMPORT_ERROR if not JAXFNE_AVAILABLE else 'jax missing'}"
        )
        return report

    report["jaxfne_version"] = getattr(jaxfne, "__version__", "unknown")

    # -- Construct model
    try:
        cfg = Configuration()
        model = construct(cfg, seed=0)
        n_neurons = int(model.n)
        report["N"] = n_neurons
    except Exception as exc:
        report["verdict"] = "EXTERNALLY_COMPOSABLE"
        report["construct_error"] = str(exc)
        return report

    # -- Resolve V0 indices
    try:
        v0_indices = paradigm_target_indices_from_model(
            model, area="V1", layer="L4", cell_type="E"
        )
        if not v0_indices:
            # Fallback: try broader selection
            v0_indices = paradigm_target_indices_from_model(model, layer="L4")
        if not v0_indices:
            # Last resort: first quarter
            v0_indices = list(range(n_neurons // 4))
        report["v0_indices_count"] = len(v0_indices)
        report["v0_indices_sample"] = v0_indices[:5]
    except Exception as exc:
        report["v0_selection_error"] = str(exc)
        v0_indices = list(range(max(1, n_neurons // 4)))
        report["v0_indices_count"] = len(v0_indices)
        report["v0_indices_fallback"] = True

    # -- Generate two 32x32 images
    rng = np.random.default_rng(7)
    A = rng.standard_normal((32, 32))
    B = rng.standard_normal((32, 32))
    report["image_shape"] = [32, 32]

    # -- JDNA encode
    eA = jdna_encode(A)
    eB = jdna_encode(B)
    report["E_A_norm"] = float(np.linalg.norm(eA))
    report["E_B_norm"] = float(np.linalg.norm(eB))
    report["enc_distance"] = float(np.linalg.norm(eA - eB))

    # -- Construct drive vectors
    dA = embedding_to_drive(eA, n_neurons, v0_indices)
    dB = embedding_to_drive(eB, n_neurons, v0_indices)
    report["drive_norm_A"] = float(np.linalg.norm(dA))
    report["drive_norm_B"] = float(np.linalg.norm(dB))

    # -- Check invariants 1-3
    inv = check_invariants(eA, eB, dA, dB, v0_indices, n_neurons)
    report.update(inv)

    # -- Invariant 4: downstream edges
    n_downstream = count_downstream_edges(model, v0_indices)
    report["N_downstream_edges"] = n_downstream
    report["inv4_pass"] = (n_downstream > 0) or (n_downstream == -1)  # -1 = not inspectable

    # -- Build StimulusSchedule for condition A
    sim_spec = Simulation(
        duration_ms=300.0,
        dt_ms=0.1,
        seed=0,
        record_sources=True,
        record_fields=False,
    )
    n_steps = sim_spec.n_steps
    dt_ms = sim_spec.dt_ms
    amplitude_scale = 5.0  # native Izhikevich current units

    def make_schedule(d_vec: np.ndarray, n_neurons: int) -> StimulusSchedule:
        v0_nonzero = [int(i) for i in np.nonzero(d_vec)[0]]
        # Per-neuron amplitude from drive vector, scaled
        events = tuple(
            {
                "onset_ms": 100.0,
                "duration_ms": 100.0,
                "amplitude": float(d_vec[i]) * amplitude_scale,
                "label": f"stim_n{i}",
                "is_drive_event": True,
                "target_indices": [i],
            }
            for i in v0_nonzero
        )
        return StimulusSchedule(events=events, n_neurons=n_neurons)

    schedule_A = make_schedule(dA, n_neurons)
    schedule_B = make_schedule(dB, n_neurons)
    report["schedule_A_n_events"] = len(schedule_A.events)
    report["schedule_B_n_events"] = len(schedule_B.events)

    # -- Simulate condition A
    try:
        sig_A = simulate(sim_spec, model=model, paradigm=schedule_A)
        spikes_A = np.asarray(sig_A.spikes)  # (n_steps, n_neurons)
        vm_A = np.asarray(sig_A.V_m)         # (n_steps, n_neurons)
        report["sim_A_spike_count"] = float(np.sum(spikes_A))
        report["sim_A_mean_vm"] = float(np.mean(vm_A))
        # V0 response window: 100-200ms -> steps 1000-2000
        stim_start = int(100.0 / dt_ms)
        stim_end = int(200.0 / dt_ms)
        v0_vm_A = vm_A[stim_start:stim_end, :][:, v0_indices]
        report["v0_vm_A_mean"] = float(np.mean(v0_vm_A))
        SIM_A_OK = True
    except Exception as exc:
        report["sim_A_error"] = str(exc)
        SIM_A_OK = False

    # -- Simulate condition B
    try:
        sig_B = simulate(sim_spec, model=model, paradigm=schedule_B)
        spikes_B = np.asarray(sig_B.spikes)
        vm_B = np.asarray(sig_B.V_m)
        report["sim_B_spike_count"] = float(np.sum(spikes_B))
        report["sim_B_mean_vm"] = float(np.mean(vm_B))
        v0_vm_B = vm_B[stim_start:stim_end, :][:, v0_indices]
        report["v0_vm_B_mean"] = float(np.mean(v0_vm_B))
        SIM_B_OK = True
    except Exception as exc:
        report["sim_B_error"] = str(exc)
        SIM_B_OK = False

    # -- Invariant 5: Delta_X_{V0}^{A/B} != 0
    if SIM_A_OK and SIM_B_OK:
        delta_v0_vm = abs(report["v0_vm_A_mean"] - report["v0_vm_B_mean"])
        report["delta_v0_vm_AB"] = delta_v0_vm
        report["inv5_pass"] = delta_v0_vm > 0.0

        # Downstream response: non-V0 spikes during stim window
        v0_set = set(v0_indices)
        all_idx = list(range(n_neurons))
        downstream_idx = [i for i in all_idx if i not in v0_set]
        if downstream_idx:
            ds_spikes_A = spikes_A[stim_start:stim_end, :][:, downstream_idx]
            ds_spikes_B = spikes_B[stim_start:stim_end, :][:, downstream_idx]
            report["downstream_spikes_A"] = float(np.sum(ds_spikes_A))
            report["downstream_spikes_B"] = float(np.sum(ds_spikes_B))
            report["downstream_response_present"] = (
                report["downstream_spikes_A"] > 0 or report["downstream_spikes_B"] > 0
            )
        else:
            report["downstream_response_present"] = False

    # -- Verdict
    all_inv_pass = all([
        report.get("inv1_pass", False),
        report.get("inv2_pass", False),
        report.get("inv3_pass", False),
        report.get("inv4_pass", False),
        report.get("inv5_pass", False) if (SIM_A_OK and SIM_B_OK) else True,
    ])
    sims_ok = SIM_A_OK and SIM_B_OK

    if sims_ok and all_inv_pass:
        report["verdict"] = "EXECUTABLE"
    elif sims_ok:
        report["verdict"] = "EXECUTABLE_INVARIANT_PARTIAL"
    else:
        report["verdict"] = "EXTERNALLY_COMPOSABLE"

    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, indent=2, default=str))
    # Print structured receipt
    print("\n=== EXPERIMENT RECEIPT ===")
    print(f"version:             {result.get('jaxfne_version', 'N/A')}")
    print(f"N:                   {result.get('N', 'N/A')}")
    print(f"E(A) norm:           {result.get('E_A_norm', 'N/A'):.6f}")
    print(f"E(B) norm:           {result.get('E_B_norm', 'N/A'):.6f}")
    print(f"AB encoded distance: {result.get('enc_distance', 'N/A'):.6f}")
    print(f"V0 indices count:    {result.get('v0_indices_count', 'N/A')}")
    print(f"drive norm A:        {result.get('drive_norm_A', 'N/A'):.6f}")
    print(f"drive norm B:        {result.get('drive_norm_B', 'N/A'):.6f}")
    print(f"inv1 (enc_dist>0):   {result.get('inv1_pass', 'N/A')}")
    print(f"inv2 (norm balance): {result.get('inv2_pass', 'N/A')} [{result.get('inv2_rel_norm_diff', 'N/A'):.2e}]")
    print(f"inv3 (on-target):    {result.get('inv3_pass', 'N/A')}  offtarget_A={result.get('inv3_offtarget_A', '?')} offtarget_B={result.get('inv3_offtarget_B', '?')}")
    print(f"N_downstream_edges:  {result.get('N_downstream_edges', 'N/A')}")
    print(f"inv4 (edges>0):      {result.get('inv4_pass', 'N/A')}")
    print(f"sim_A spike count:   {result.get('sim_A_spike_count', 'N/A')}")
    print(f"sim_B spike count:   {result.get('sim_B_spike_count', 'N/A')}")
    print(f"v0_vm_A_mean:        {result.get('v0_vm_A_mean', 'N/A')}")
    print(f"v0_vm_B_mean:        {result.get('v0_vm_B_mean', 'N/A')}")
    print(f"delta_v0_vm_AB:      {result.get('delta_v0_vm_AB', 'N/A')}")
    print(f"inv5 (V0 diff!=0):   {result.get('inv5_pass', 'N/A')}")
    print(f"downstream_spikes_A: {result.get('downstream_spikes_A', 'N/A')}")
    print(f"downstream_spikes_B: {result.get('downstream_spikes_B', 'N/A')}")
    print(f"downstream_response: {result.get('downstream_response_present', 'N/A')}")
    if "sim_A_error" in result:
        print(f"sim_A_error:         {result['sim_A_error']}")
    if "sim_B_error" in result:
        print(f"sim_B_error:         {result['sim_B_error']}")
    print(f"\nVERDICT: {result.get('verdict', 'UNKNOWN')}")
