"""E2 typed provenance-class delays — attachment, gates, receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import _edge_delay_steps_host
from jaxfne.io import json_safe
from jaxfne.protocol_e_integration.e1_execution import (
    build_edge_provenance_table,
    build_e1_configuration,
    build_identity_map,
    identity_round_trip_ok,
    verify_connectivity_ownership,
)
from jaxfne.protocol_e_integration.e2_protocol import (
    E2_EXECUTION_RECEIPT_PATH,
    e2_delay_class_ids,
    load_e2_spec,
    validate_e2_spec,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

_BANNED_RECEIPT_TERMS = (
    "psd",
    "band_power",
    "gamma",
    "beta",
    "phase",
    "wave",
    "memory",
    "functional_ff",
    "functional_fb",
    "rbs",
    "hdp",
    "field_evidence",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _e2_runtime() -> jtfne.RuntimeConfig:
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=False,
        hdp_params={"noise_scale": 0.0},
    )


def _delay_steps_by_class(spec: dict[str, Any], *, zero: bool = False) -> dict[str, int]:
    if zero:
        return {edge_class: 0 for edge_class in e2_delay_class_ids(spec)}
    return {
        row["edge_class"]: int(row["delay_steps"])
        for row in spec["delay_values"]["classes"]
    }


def _expected_edge_class_counts(spec: dict[str, Any]) -> dict[str, int]:
    raw = spec["e1_receipt_derived_constants"]["expected_edge_class_counts"]
    return {str(k): int(v) for k, v in raw.items() if k != "source_receipt" and k != "source_field"}


def assert_no_compile_drift(model: jtfne.Model, spec: dict[str, Any]) -> dict[str, int]:
    """Halt on any deviation from frozen E1 edge realization."""
    provenance = build_edge_provenance_table(model)
    g3 = verify_connectivity_ownership(provenance)
    actual = {str(k): int(v) for k, v in g3["edge_class_counts"].items()}
    expected = _expected_edge_class_counts(spec)
    if actual != expected:
        raise RuntimeError(
            f"E2 compile drift: expected edge_class_counts {expected}, got {actual}"
        )
    total = sum(actual.values())
    if total != 931:
        raise RuntimeError(f"E2 compile drift: expected 931 edges, got {total}")
    return actual


def attach_provenance_class_delays(
    model: jtfne.Model,
    delay_steps_by_class: Mapping[str, int],
) -> jtfne.Model:
    """Attach per-edge delay_steps using E1 provenance rows (not index inference)."""
    provenance = build_edge_provenance_table(model)
    edges = model.params["edge_list"]
    delay_steps = np.zeros(int(edges.n_edges), dtype=np.int32)
    for row in provenance:
        edge_class = str(row["edge_class"])
        if edge_class not in delay_steps_by_class:
            raise ValueError(f"unknown edge_class {edge_class!r} for delay attachment")
        delay_steps[int(row["edge_index"])] = int(delay_steps_by_class[edge_class])
    new_edges = replace(edges, delay_steps=jnp.asarray(delay_steps, dtype=jnp.int32))
    object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
    return model


def build_e2_base_model() -> jtfne.Model:
    """Construct the shared E1 hierarchy before typed delay attachment."""
    return jtfne.construct(build_e1_configuration(include_inter_area=True))


def build_e2_model(*, zero_delays: bool = False) -> jtfne.Model:
    """E2 model path: compile hierarchy, verify drift, attach provenance delays."""
    spec = load_e2_spec()
    validate_e2_spec(spec)
    model = build_e2_base_model()
    assert_no_compile_drift(model, spec)
    delay_map = _delay_steps_by_class(spec, zero=zero_delays)
    return attach_provenance_class_delays(model, delay_map)


def _provenance_topology_rows(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "edge_index": int(row["edge_index"]),
            "edge_class": row["edge_class"],
            "pre_flat_index": int(row["pre_flat_index"]),
            "post_flat_index": int(row["post_flat_index"]),
        }
        for row in provenance
    ]


def hierarchy_fingerprint(model: jtfne.Model) -> dict[str, str]:
    """Digest neuron identity and edge provenance topology (delay-free)."""
    identity = build_identity_map(model.neuron_table())
    provenance = _provenance_topology_rows(build_edge_provenance_table(model))
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre, dtype=np.int64)
    post = np.asarray(edges.post, dtype=np.int64)
    weight = np.asarray(edges.weight, dtype=np.float64)
    receptor = np.asarray(edges.receptor_index, dtype=np.int64)
    edge_topology = [
        {
            "pre": int(p),
            "post": int(q),
            "weight": float(w),
            "receptor_index": int(r),
        }
        for p, q, w, r in zip(pre, post, weight, receptor, strict=True)
    ]

    def _digest(obj: Any) -> str:
        payload = json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "identity_digest": _digest(identity),
        "provenance_digest": _digest(provenance),
        "edge_topology_digest": _digest(edge_topology),
    }


def build_typed_delay_table(
    model: jtfne.Model,
    spec: dict[str, Any],
    *,
    actual_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delay_by_class = {
        row["edge_class"]: row for row in spec["delay_values"]["classes"]
    }
    expected = _expected_edge_class_counts(spec)
    for edge_class in e2_delay_class_ids(spec):
        expected_n = int(expected[edge_class])
        actual_n = int(actual_counts[edge_class])
        if actual_n != expected_n:
            raise RuntimeError(
                f"typed delay table drift for {edge_class}: expected {expected_n}, got {actual_n}"
            )
        meta = delay_by_class[edge_class]
        rows.append(
            {
                "edge_class": edge_class,
                "expected_edges": expected_n,
                "actual_edges": actual_n,
                "delay_ms": float(meta["tau_ms"]),
                "delay_steps": int(meta["delay_steps"]),
            }
        )
    return rows


def delay_step_occupancy(model: jtfne.Model) -> dict[str, int]:
    steps = np.asarray(_edge_delay_steps_host(model.params["edge_list"]), dtype=np.int64)
    uniq, counts = np.unique(steps, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq, counts, strict=True)}


def _simulate(
    model: jtfne.Model,
    *,
    duration_ms: float,
    dt_ms: float,
    seed: int,
    return_state: bool = False,
    continuation: Any | None = None,
) -> Any:
    sim = jtfne.Simulation(
        duration_ms=float(duration_ms),
        dt_ms=float(dt_ms),
        seed=int(seed),
        record_sources=True,
        runtime=_e2_runtime(),
    )
    return model.simulate(
        sim,
        continuation=continuation,
        return_state=return_state,
    )


def _spike_event_times(spikes: np.ndarray, *, dt_ms: float) -> list[tuple[int, int, float]]:
    idx = np.argwhere(np.asarray(spikes) > 0.5)
    return [(int(r[0]), int(r[1]), float(r[0]) * float(dt_ms)) for r in idx]


def _assert_bit_exact_continuation_states(st_full: Any, st_split: Any) -> None:
    for key in ("v", "u", "prev_spikes", "syn_state", "H", "w"):
        diff = float(jnp.max(jnp.abs(getattr(st_full.dynamic, key) - getattr(st_split.dynamic, key))))
        if diff != 0.0:
            raise RuntimeError(f"continuation dynamic.{key} mismatch: max abs diff {diff}")
    if st_full.delay_state is None or st_split.delay_state is None:
        raise RuntimeError("delayed E2 continuation requires non-None delay_state")
    b_diff = float(jnp.max(jnp.abs(st_full.delay_state - st_split.delay_state)))
    if b_diff != 0.0:
        raise RuntimeError(f"delay_state B_t mismatch: max abs diff {b_diff}")
    if int(np.asarray(st_full.step_index)) != int(np.asarray(st_split.step_index)):
        raise RuntimeError("continuation step_index mismatch")


def verify_delayed_continuation(
    model: jtfne.Model,
    *,
    total_ms: float,
    split_ms: float,
    dt_ms: float,
    seed: int,
) -> dict[str, Any]:
    seg1_ms = float(split_ms)
    seg2_ms = float(total_ms) - seg1_ms
    full_sig, st_full = _simulate(
        model, duration_ms=total_ms, dt_ms=dt_ms, seed=seed, return_state=True
    )
    seg1_sig, st1 = _simulate(
        model, duration_ms=seg1_ms, dt_ms=dt_ms, seed=seed, return_state=True
    )
    seg2_sig, st2 = _simulate(
        model,
        duration_ms=seg2_ms,
        dt_ms=dt_ms,
        seed=seed,
        continuation=st1,
        return_state=True,
    )

    vm_full = np.asarray(full_sig.V_m)
    vm_seg = np.concatenate([np.asarray(seg1_sig.V_m), np.asarray(seg2_sig.V_m)], axis=0)
    sp_full = np.asarray(full_sig.spikes)
    sp_seg = np.concatenate([np.asarray(seg1_sig.spikes), np.asarray(seg2_sig.spikes)], axis=0)
    src_full = np.asarray(full_sig.sources)
    src_seg = np.concatenate([np.asarray(seg1_sig.sources), np.asarray(seg2_sig.sources)], axis=0)

    vm_ok = bool(np.array_equal(vm_full, vm_seg))
    sp_ok = bool(np.array_equal(sp_full, sp_seg))
    src_ok = bool(np.array_equal(src_full, src_seg))
    syn_ok = bool(
        np.array_equal(
            np.asarray(st_full.dynamic.syn_state),
            np.asarray(st2.dynamic.syn_state),
        )
    )
    try:
        _assert_bit_exact_continuation_states(st_full, st2)
        state_ok = True
        state_error = None
    except RuntimeError as exc:
        state_ok = False
        state_error = str(exc)

    events_full = _spike_event_times(sp_full, dt_ms=dt_ms)
    events_seg = _spike_event_times(sp_seg, dt_ms=dt_ms)
    timing_ok = events_full == events_seg

    passed = vm_ok and sp_ok and src_ok and syn_ok and state_ok and timing_ok
    return {
        "passed": passed,
        "split_ms": seg1_ms,
        "vm_bit_exact": vm_ok,
        "spikes_bit_exact": sp_ok,
        "sources_bit_exact": src_ok,
        "synaptic_state_bit_exact": syn_ok,
        "delay_state_bit_exact": state_ok,
        "event_timing_exact": timing_ok,
        "n_spike_events": len(events_full),
        "state_error": state_error,
    }


def verify_delay_ownership(model: jtfne.Model, spec: dict[str, Any]) -> dict[str, Any]:
    expected = _delay_steps_by_class(spec, zero=False)
    provenance = build_edge_provenance_table(model)
    steps = np.asarray(_edge_delay_steps_host(model.params["edge_list"]), dtype=np.int64)
    violations: list[dict[str, Any]] = []
    for row in provenance:
        edge_index = int(row["edge_index"])
        edge_class = str(row["edge_class"])
        want = int(expected[edge_class])
        got = int(steps[edge_index])
        if got != want:
            violations.append(
                {
                    "edge_index": edge_index,
                    "edge_class": edge_class,
                    "expected_delay_steps": want,
                    "actual_delay_steps": got,
                }
            )
    return {
        "passed": not violations,
        "n_violations": len(violations),
        "violations_sample": violations[:8],
    }


def _receipt_has_banned_scientific_claims(receipt: dict[str, Any]) -> list[str]:
    evidence_blob = json.dumps(
        {
            "scientific_question": receipt.get("scientific_question"),
            "composition_claim": receipt.get("composition_claim"),
            "typed_delay_table": receipt.get("typed_delay_table"),
            "delay_step_occupancy": receipt.get("delay_step_occupancy"),
            "gates": receipt.get("gates"),
        }
    ).lower()
    return [term for term in _BANNED_RECEIPT_TERMS if term in evidence_blob]


def run_e2_delayed_coupling(*, package_head: str | None = None) -> dict[str, Any]:
    spec = load_e2_spec()
    validate_e2_spec(spec)
    sim = spec["simulation_policy"]
    duration_ms = float(sim["duration_ms"])
    dt_ms = float(sim["dt_ms"])
    seeds = [int(s) for s in sim["seeds"]]

    gates: dict[str, dict[str, Any]] = {
        "G1_e1_reduction": {"passed": False},
        "G2_provenance_preservation": {"passed": False},
        "G3_delay_ownership": {"passed": False},
        "G4_finite_delayed_execution": {"passed": False},
        "G5_delayed_continuation": {"passed": False},
        "G6_exact_event_timing": {"passed": False},
        "G7_hierarchy_invariance": {"passed": False},
        "G8_no_scientific_overinterpretation": {"passed": False},
    }

    base_model = build_e2_base_model()
    actual_counts = assert_no_compile_drift(base_model, spec)
    pre_delay_fp = hierarchy_fingerprint(base_model)
    pre_provenance = build_edge_provenance_table(base_model)

    zero_model = attach_provenance_class_delays(
        jtfne.construct(build_e1_configuration(include_inter_area=True)),
        _delay_steps_by_class(spec, zero=True),
    )
    assert_no_compile_drift(zero_model, spec)
    delayed_model = build_e2_model(zero_delays=False)
    post_delay_fp = hierarchy_fingerprint(delayed_model)

    e1_reference = jtfne.construct(build_e1_configuration(include_inter_area=True))
    g1_seed_checks: list[dict[str, Any]] = []
    g1_ok = True
    for seed in seeds:
        ref = _simulate(e1_reference, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
        zero = _simulate(zero_model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
        vm_ok = bool(np.array_equal(np.asarray(ref.V_m), np.asarray(zero.V_m)))
        sp_ok = bool(np.array_equal(np.asarray(ref.spikes), np.asarray(zero.spikes)))
        g1_ok = g1_ok and vm_ok and sp_ok
        g1_seed_checks.append({"seed": seed, "V_m_bit_exact": vm_ok, "spikes_bit_exact": sp_ok})
    gates["G1_e1_reduction"] = {
        "passed": g1_ok,
        "path": "E2 provenance delay attachment with tau=0",
        "seeds": g1_seed_checks,
    }

    post_provenance = build_edge_provenance_table(delayed_model)
    prov_ok = _provenance_topology_rows(pre_provenance) == _provenance_topology_rows(
        post_provenance
    )
    gates["G2_provenance_preservation"] = {
        "passed": prov_ok,
        "n_edges": len(post_provenance),
    }

    g3 = verify_delay_ownership(delayed_model, spec)
    gates["G3_delay_ownership"] = g3

    g4_seed_checks: list[dict[str, Any]] = []
    g4_ok = True
    for seed in seeds:
        sig, _ = _simulate(
            delayed_model,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            seed=seed,
            return_state=True,
        )
        finite = bool(
            np.isfinite(np.asarray(sig.V_m)).all()
            and np.isfinite(np.asarray(sig.spikes)).all()
            and np.isfinite(np.asarray(sig.sources)).all()
        )
        g4_ok = g4_ok and finite
        g4_seed_checks.append({"seed": seed, "finite": finite})
    gates["G4_finite_delayed_execution"] = {"passed": g4_ok, "seeds": g4_seed_checks}

    primary_split = float(spec["continuation_segmentation"]["primary_split"]["t_split_ms"])
    inflight_split = float(spec["continuation_segmentation"]["inflight_stress_split"]["t_split_ms"])
    g5_primary = verify_delayed_continuation(
        delayed_model,
        total_ms=duration_ms,
        split_ms=primary_split,
        dt_ms=dt_ms,
        seed=seeds[0],
    )
    g5_inflight = verify_delayed_continuation(
        delayed_model,
        total_ms=duration_ms,
        split_ms=inflight_split,
        dt_ms=dt_ms,
        seed=seeds[0],
    )
    g5_ok = bool(g5_primary["passed"] and g5_inflight["passed"])
    gates["G5_delayed_continuation"] = {
        "passed": g5_ok,
        "primary_split_ms": primary_split,
        "inflight_split_ms": inflight_split,
        "primary": g5_primary,
        "inflight": g5_inflight,
    }
    gates["G6_exact_event_timing"] = {
        "passed": bool(g5_primary["event_timing_exact"] and g5_inflight["event_timing_exact"]),
        "primary_n_spike_events": g5_primary["n_spike_events"],
        "inflight_n_spike_events": g5_inflight["n_spike_events"],
    }

    identity_unchanged = pre_delay_fp["identity_digest"] == post_delay_fp["identity_digest"]
    provenance_unchanged = pre_delay_fp["provenance_digest"] == post_delay_fp["provenance_digest"]
    topology_unchanged = pre_delay_fp["edge_topology_digest"] == post_delay_fp["edge_topology_digest"]
    g7_ok = identity_unchanged and provenance_unchanged and topology_unchanged
    gates["G7_hierarchy_invariance"] = {
        "passed": g7_ok,
        "identity_digest_match": identity_unchanged,
        "provenance_digest_match": provenance_unchanged,
        "edge_topology_digest_match": topology_unchanged,
        "pre_delay_fingerprints": pre_delay_fp,
        "post_delay_fingerprints": post_delay_fp,
    }

    typed_delay_table = build_typed_delay_table(
        delayed_model, spec, actual_counts=actual_counts
    )
    occupancy = delay_step_occupancy(delayed_model)
    receipt_draft: dict[str, Any] = {
        "typed_delay_table": typed_delay_table,
        "delay_step_occupancy": occupancy,
    }
    banned = _receipt_has_banned_scientific_claims(receipt_draft)
    gates["G8_no_scientific_overinterpretation"] = {
        "passed": not banned,
        "banned_terms_found": banned,
        "reported_evidence": "lag/delay structure and continuation contracts only",
    }

    p_local = spec["e1_receipt_derived_constants"]["p_local"]
    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e2_execution_receipt.v1",
        "checkpoint": "E2",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_e_integration/e2_delayed_coupling_spec.json",
        "scientific_question": spec["question"],
        "composition_claim": (
            "hierarchical biological identity + typed pathway delays + exact delayed continuation"
        ),
        "reduction_contract": "R_E2_to_E1",
        "p_local": {
            "value": float(p_local["value"]),
            "provenance": p_local["provenance"],
            "rule": p_local["rule"],
        },
        "typed_delay_table": typed_delay_table,
        "delay_step_occupancy": {
            "N_d2": int(occupancy.get("2", 0)),
            "N_d4": int(occupancy.get("4", 0)),
            "N_d8": int(occupancy.get("8", 0)),
            "full_histogram": occupancy,
        },
        "gates": gates,
        "e3_deferred": "RBS with H=H* => E3=E2",
        "explicit_exclusions": list(spec["explicit_prohibitions"]),
    }

    banned_final = _receipt_has_banned_scientific_claims(receipt)
    if banned_final:
        raise RuntimeError(f"E2 receipt contains banned scientific terms: {banned_final}")

    for gname, gval in gates.items():
        if not gval.get("passed"):
            raise RuntimeError(f"E2 gate failed: {gname}")
    return json_safe(receipt)


def write_e2_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_e2_delayed_coupling(package_head=package_head)
    E2_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_e2_execution_receipt() -> dict[str, Any]:
    return json.loads(E2_EXECUTION_RECEIPT_PATH.read_text())
