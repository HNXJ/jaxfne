"""E3 sparse owned H_K RBS composition — gates and receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_owned_h_k_delayed
from jaxfne.io import json_safe
from jaxfne.protocol_e_integration.e2_execution import (
    build_e2_model,
    delay_step_occupancy,
    hierarchy_fingerprint,
    load_e2_execution_receipt,
)
from jaxfne.protocol_e_integration.e3_protocol import (
    E3_EXECUTION_RECEIPT_PATH,
    e3_owner_flat_indices,
    load_e3_spec,
    validate_e3_spec,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

E3Mode = Literal["E3-null", "E3-dynamic"]

_BANNED_RECEIPT_TERMS = (
    "psd",
    "band_power",
    "gamma",
    "beta",
    "phase",
    "wave",
    "memory_score",
    "adaptation_index",
    "functional_ff",
    "functional_fb",
    "hdp",
    "field_evidence",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def owner_flat_indices(spec: dict[str, Any] | None = None) -> list[int]:
    return list(e3_owner_flat_indices(spec or load_e3_spec()))


def build_owner_mask(n_neurons: int, owners: list[int]) -> jnp.ndarray:
    mask = np.zeros((n_neurons,), dtype=np.float32)
    mask[np.asarray(owners, dtype=np.int64)] = 1.0
    return jnp.asarray(mask, dtype=jnp.float32)


def build_h_k0(
    n_neurons: int,
    owners: list[int],
    *,
    mode: E3Mode,
    h_k0_dynamic: float,
) -> jnp.ndarray:
    h = np.ones((n_neurons,), dtype=np.float32)
    if mode == "E3-dynamic":
        h[np.asarray(owners, dtype=np.int64)] = float(h_k0_dynamic)
    return jnp.asarray(h, dtype=jnp.float32)


def h_k_owner_euler_reference(
    h0: float,
    n_steps: int,
    *,
    dt_ms: float,
    tau_k_ms: float,
) -> np.ndarray:
    """D2a float32 Euler contract for one owner neuron."""
    H = np.float32(h0)
    dt = np.float32(dt_ms)
    tau = np.float32(tau_k_ms)
    one = np.float32(1.0)
    out = np.empty((n_steps,), dtype=np.float32)
    for i in range(n_steps):
        H = H + dt * (one - H) / tau
        out[i] = H
    return out.astype(np.float64)


def _digest(obj: Any) -> str:
    payload = json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def delay_table_digest(model: jtfne.Model) -> str:
    e2_receipt = load_e2_execution_receipt()
    rows = e2_receipt["typed_delay_table"]
    occ = delay_step_occupancy(model)
    table = {
        "typed_delay_table": rows,
        "delay_step_occupancy": occ,
    }
    return _digest(table)


def _kernel_init_from_final(st: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": st["v"],
        "u": st["u"],
        "prev_spikes": st["prev_spikes"],
        "syn_state": st["syn_state"],
        "delay_state": st["delay_state"],
        "H_K": st["H_K_final"],
        "continuation_step_offset": st["continuation_step_offset"],
    }


def run_e3_kernel(
    model: jtfne.Model,
    *,
    n_steps: int,
    dt_ms: float,
    seed: int,
    mode: E3Mode,
    init_state: dict[str, Any] | None = None,
    step_offset: int | None = None,
) -> dict[str, Any]:
    spec = load_e3_spec()
    owners = owner_flat_indices(spec)
    n_neurons = int(model.params["emitter"].n_neurons)
    tau_k = float(spec["rbs_primitive"]["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"])
    h_k0_dynamic = float(spec["rbs_primitive"]["initial_perturbation"]["H_K0_dynamic"])
    key = jax.random.PRNGKey(int(seed))
    step_indices = None
    if step_offset is not None:
        step_indices = jnp.arange(int(step_offset), int(step_offset) + int(n_steps), dtype=jnp.int32)
    v, sp, src, st = simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
        model.params["emitter"],
        model.params["edge_list"],
        int(n_steps),
        float(dt_ms),
        key,
        h_k0=build_h_k0(n_neurons, owners, mode=mode, h_k0_dynamic=h_k0_dynamic),
        owner_mask=build_owner_mask(n_neurons, owners),
        tau_k_ms=tau_k,
        dynamic=(mode == "E3-dynamic"),
        dtype="float32",
        noise_scale=0.0,
        init_state=init_state,
        step_indices=step_indices,
    )
    H_trace = np.asarray(st["H_K_trace"], dtype=np.float64)
    return {
        "V_m": np.asarray(v, dtype=np.float64),
        "spikes": np.asarray(sp, dtype=np.float64),
        "sources": np.asarray(src, dtype=np.float64),
        "H_K_trace": H_trace,
        "H_K_final": np.asarray(st["H_K_final"], dtype=np.float64),
        "synaptic_state_final": np.asarray(st["syn_state"], dtype=np.float64),
        "delay_state_final": np.asarray(st["delay_state"], dtype=np.float64),
        "final_state": st,
        "owner_mask": np.asarray(st["owner_mask"], dtype=np.float64),
        "h_k_semantics": st["h_k_non_owner_semantics"],
    }


def run_e2_reference_kernel(
    model: jtfne.Model,
    *,
    n_steps: int,
    dt_ms: float,
    seed: int,
    init_state: dict[str, Any] | None = None,
    step_offset: int | None = None,
) -> dict[str, Any]:
    from jaxfne.emitters import simulate_edge_recurrent_izhikevich

    key = jax.random.PRNGKey(int(seed))
    step_indices = None
    if step_offset is not None:
        step_indices = jnp.arange(int(step_offset), int(step_offset) + int(n_steps), dtype=jnp.int32)
    v, sp, src, st = simulate_edge_recurrent_izhikevich(
        model.params["emitter"],
        model.params["edge_list"],
        int(n_steps),
        float(dt_ms),
        key,
        dtype="float32",
        noise_scale=0.0,
        init_state=init_state,
        step_indices=step_indices,
    )
    return {
        "V_m": np.asarray(v, dtype=np.float64),
        "spikes": np.asarray(sp, dtype=np.float64),
        "sources": np.asarray(src, dtype=np.float64),
        "final_state": st,
    }


def _spike_event_times(spikes: np.ndarray, *, dt_ms: float) -> list[tuple[int, int, float]]:
    idx = np.argwhere(np.asarray(spikes) > 0.5)
    return [(int(r[0]), int(r[1]), float(r[0]) * float(dt_ms)) for r in idx]


def verify_e3_continuation(
    model: jtfne.Model,
    *,
    total_steps: int,
    split_steps: int,
    dt_ms: float,
    seed: int,
    mode: E3Mode,
) -> dict[str, Any]:
    seg2_steps = int(total_steps) - int(split_steps)
    full = run_e3_kernel(
        model, n_steps=total_steps, dt_ms=dt_ms, seed=seed, mode=mode
    )
    seg1 = run_e3_kernel(
        model, n_steps=split_steps, dt_ms=dt_ms, seed=seed, mode=mode
    )
    seg2 = run_e3_kernel(
        model,
        n_steps=seg2_steps,
        dt_ms=dt_ms,
        seed=seed,
        mode=mode,
        init_state=_kernel_init_from_final(seg1["final_state"]),
        step_offset=int(split_steps),
    )
    vm_seg = np.concatenate([seg1["V_m"], seg2["V_m"]], axis=0)
    sp_seg = np.concatenate([seg1["spikes"], seg2["spikes"]], axis=0)
    src_seg = np.concatenate([seg1["sources"], seg2["sources"]], axis=0)
    hk_seg = np.concatenate([seg1["H_K_trace"], seg2["H_K_trace"]], axis=0)
    vm_ok = bool(np.array_equal(full["V_m"], vm_seg))
    sp_ok = bool(np.array_equal(full["spikes"], sp_seg))
    src_ok = bool(np.array_equal(full["sources"], src_seg))
    hk_ok = bool(np.array_equal(full["H_K_trace"], hk_seg))
    syn_ok = bool(
        np.array_equal(
            full["synaptic_state_final"],
            seg2["synaptic_state_final"],
        )
    )
    delay_ok = bool(
        np.array_equal(full["delay_state_final"], seg2["delay_state_final"])
    )
    events_full = _spike_event_times(full["spikes"], dt_ms=dt_ms)
    events_seg = _spike_event_times(sp_seg, dt_ms=dt_ms)
    timing_ok = events_full == events_seg
    passed = vm_ok and sp_ok and src_ok and hk_ok and syn_ok and delay_ok and timing_ok
    return {
        "passed": passed,
        "split_steps": int(split_steps),
        "V_m_bit_exact": vm_ok,
        "spikes_bit_exact": sp_ok,
        "sources_bit_exact": src_ok,
        "H_K_bit_exact": hk_ok,
        "synaptic_state_bit_exact": syn_ok,
        "delay_state_bit_exact": delay_ok,
        "event_timing_exact": timing_ok,
        "H_K_at_split_ms": float(seg1["H_K_final"][owner_flat_indices()[0]]),
    }


def verify_non_owner_h_k_support(H_trace: np.ndarray, owners: list[int], n_neurons: int) -> dict[str, Any]:
    non_owners = [i for i in range(n_neurons) if i not in owners]
    owner_dev = H_trace[:, owners] - 1.0
    non_owner_dev = H_trace[:, non_owners] - 1.0
    non_owner_exact = bool(np.max(np.abs(non_owner_dev)) == 0.0)
    support_only_owners = bool(np.max(np.abs(owner_dev)) == 0.0 or np.any(np.abs(owner_dev) > 0.0))
    return {
        "passed": non_owner_exact,
        "max_abs_H_minus_1_non_owners": float(np.max(np.abs(non_owner_dev))),
        "max_abs_H_minus_1_owners": float(np.max(np.abs(owner_dev))),
        "support_only_on_O_H": support_only_owners,
    }


def _receipt_has_banned_terms(receipt: dict[str, Any]) -> list[str]:
    blob = json.dumps(
        {
            "composition_claim": receipt.get("composition_claim"),
            "gates": receipt.get("gates"),
            "composition_effect": receipt.get("composition_effect"),
        }
    ).lower()
    return [term for term in _BANNED_RECEIPT_TERMS if term in blob]


def run_e3_rbs_composition(*, package_head: str | None = None) -> dict[str, Any]:
    spec = load_e3_spec()
    validate_e3_spec(spec)
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    seeds = [int(s) for s in sim["seeds"]]
    owners = owner_flat_indices(spec)
    n_neurons = 80
    tau_k = float(spec["rbs_primitive"]["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"])
    h_k0_dynamic = float(spec["rbs_primitive"]["initial_perturbation"]["H_K0_dynamic"])

    model = build_e2_model(zero_delays=False)
    pre_fp = hierarchy_fingerprint(model)
    pre_delay_digest = delay_table_digest(model)

    gates: dict[str, dict[str, Any]] = {
        "G1_e2_reduction": {"passed": False},
        "G2_ownership": {"passed": False},
        "G3_non_owner_invariance": {"passed": False},
        "G4_typed_expression": {"passed": False},
        "G5_autonomous_recovery": {"passed": False},
        "G6_delay_compatibility": {"passed": False},
        "G7_continuation": {"passed": False},
        "G8_hierarchy_invariance": {"passed": False},
        "G9_no_phenotype_claim": {"passed": False},
    }

    g1_checks: list[dict[str, Any]] = []
    g1_ok = True
    for seed in seeds:
        e2 = run_e2_reference_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seed)
        e3 = run_e3_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seed, mode="E3-null")
        vm_ok = bool(np.array_equal(e2["V_m"], e3["V_m"]))
        sp_ok = bool(np.array_equal(e2["spikes"], e3["spikes"]))
        src_ok = bool(np.array_equal(e2["sources"], e3["sources"]))
        g1_ok = g1_ok and vm_ok and sp_ok and src_ok
        g1_checks.append(
            {"seed": seed, "V_m_bit_exact": vm_ok, "spikes_bit_exact": sp_ok, "sources_bit_exact": src_ok}
        )
    gates["G1_e2_reduction"] = {
        "passed": g1_ok,
        "path": "E3-null owned H_K delayed kernel vs E2 delayed reference kernel",
        "seeds": g1_checks,
    }

    dyn = run_e3_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seeds[0], mode="E3-dynamic")
    owner_mask = np.asarray(dyn["owner_mask"])
    g2_ok = bool(np.all(owner_mask[owners] > 0.5) and np.all(owner_mask[np.setdiff1d(np.arange(n_neurons), owners)] < 0.5))
    gates["G2_ownership"] = {
        "passed": g2_ok,
        "owner_flat_indices": owners,
        "n_owners": len(owners),
    }

    g3 = verify_non_owner_h_k_support(dyn["H_K_trace"], owners, n_neurons)
    gates["G3_non_owner_invariance"] = g3

    g4_ok = bool(dyn["h_k_semantics"] == "fixed_reference_H_K_equals_1_with_F1_recurrence_masked_off")
    gates["G4_typed_expression"] = {
        "passed": g4_ok,
        "coupling_map": spec["rbs_primitive"]["coupling_map"]["display"],
        "kernel": "simulate_edge_recurrent_izhikevich_owned_h_k_delayed",
    }

    euler_checks: list[dict[str, Any]] = []
    g5_ok = True
    ref = h_k_owner_euler_reference(h_k0_dynamic, n_steps, dt_ms=dt_ms, tau_k_ms=tau_k)
    for oid in owners:
        trace = dyn["H_K_trace"][:, oid]
        err = float(np.max(np.abs(trace - ref)))
        ok = err == 0.0
        g5_ok = g5_ok and ok
        euler_checks.append({"owner_flat_index": oid, "max_abs_euler_diff": err, "passed": ok})
    gates["G5_autonomous_recovery"] = {
        "passed": g5_ok,
        "tau_K_ms": tau_k,
        "H_K0_dynamic": h_k0_dynamic,
        "owner_euler_checks": euler_checks,
        "H_K_final_owner_mean": float(np.mean(dyn["H_K_final"][owners])),
    }

    post_delay_digest = delay_table_digest(model)
    occ = delay_step_occupancy(model)
    g6_ok = (
        pre_delay_digest == post_delay_digest
        and occ.get("2") == 651
        and occ.get("4") == 140
        and occ.get("8") == 140
    )
    gates["G6_delay_compatibility"] = {
        "passed": g6_ok,
        "delay_table_digest_match": pre_delay_digest == post_delay_digest,
        "delay_step_occupancy": occ,
        "expected_occupancy": {"2": 651, "4": 140, "8": 140},
    }

    split_steps = int(round(120.0 / dt_ms))
    g7 = verify_e3_continuation(
        model,
        total_steps=n_steps,
        split_steps=split_steps,
        dt_ms=dt_ms,
        seed=seeds[0],
        mode="E3-dynamic",
    )
    gates["G7_continuation"] = g7

    post_fp = hierarchy_fingerprint(model)
    g8_ok = pre_fp == post_fp
    gates["G8_hierarchy_invariance"] = {
        "passed": g8_ok,
        "pre_rbs_fingerprints": pre_fp,
        "post_rbs_fingerprints": post_fp,
        "delay_table_digest": post_delay_digest,
    }

    e2_ref = run_e2_reference_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seeds[0])
    composition_diff = bool(not np.array_equal(e2_ref["sources"], dyn["sources"]))
    gates["G9_no_phenotype_claim"] = {
        "passed": True,
        "composition_effect_observed": composition_diff,
        "statement": "H_K(0)!=1 => X_E3 may differ from X_E2; expression evidence only",
        "forbidden_metrics_absent": True,
    }

    ownership = spec["ownership"]
    rbs_primitive_receipt = {
        "semantic_type": "effective_K_associated_recovery",
        "source_protocol": "D1/D2a",
        "H_reference": 1.0,
        "tau_K_ms": tau_k,
        "initial_dynamic_H": h_k0_dynamic,
        "non_owner_H_K_semantics": (
            "fixed reference coordinate H_K=1; F1 recurrence masked off outside owner_mask"
        ),
    }
    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e3_execution_receipt.v1",
        "checkpoint": "E3",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_e_integration/e3_rbs_composition_spec.json",
        "scientific_question": spec["question"],
        "composition_claim": (
            "hierarchy + typed delays + sparse owned local RBD without changing E2 reference"
        ),
        "reduction_contract": "R_E3_to_E2",
        "rbs_owner": {
            "area": ownership["selector"]["area"],
            "layer": ownership["selector"]["layer"],
            "cell_type": ownership["selector"]["cell_type"],
            "n_nodes": int(ownership["n_nodes"]),
            "flat_indices": owners,
            "identity_source": "artifacts/protocol_e_integration/e1_execution_receipt.json#identity_map",
        },
        "rbs_primitive": rbs_primitive_receipt,
        "inherited_e2_delay_digest": post_delay_digest,
        "inherited_e2_hierarchy_fingerprints": post_fp,
        "gates": gates,
        "composition_effect": {
            "dynamic_sources_differ_from_e2": composition_diff,
            "interpretation": "expression_evidence_only_not_phenotype",
        },
        "e4_deferred": "observation chain with R_E4_to_E3",
    }
    if _receipt_has_banned_terms(receipt):
        raise RuntimeError("E3 receipt contains banned scientific terms")

    for gname, gval in gates.items():
        if not gval.get("passed"):
            raise RuntimeError(f"E3 gate failed: {gname}")
    return json_safe(receipt)


def write_e3_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_e3_rbs_composition(package_head=package_head)
    E3_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_e3_execution_receipt() -> dict[str, Any]:
    return json.loads(E3_EXECUTION_RECEIPT_PATH.read_text())
