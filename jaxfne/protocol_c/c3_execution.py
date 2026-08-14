"""C3 prospective neural execution — frozen 60-cell geometry/delay experiment."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.core import StimulusSchedule
from jaxfne.h4_matrix import build_ring_params_edges
from jaxfne.io import json_safe
from jaxfne.protocol_c.c3_protocol import (
    C3_SPEC_PATH,
    c3_total_cells,
    load_c3_spec,
    validate_c3_spec,
)
from jaxfne.protocol_c.estimator import estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "artifacts" / "protocol_c"
C3_EXECUTION_RECEIPT_PATH = BUNDLE_ROOT / "c3_execution_receipt.json"
C3_CONDITION_SUMMARY_PATH = BUNDLE_ROOT / "c3_condition_summary.json"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _fisher_yates_permutation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    perm = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _ordered_arc_positions_mm(n: int, *, radius_mm: float) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)
    return (float(radius_mm) * theta).reshape(-1, 1)


def _ring_positions_3d(n: int, *, radius_mm: float) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)
    xy = radius_mm * np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    return xy.astype(np.float32)


def _geometry_shuffle_seed(spec: dict[str, Any], condition_index: int) -> int:
    g = spec["geometry_shuffle"]
    return int(g["base_geometry_seed"]) + int(condition_index) * 9973


def _delay_shuffle_seed(spec: dict[str, Any], condition_index: int) -> int:
    d = spec["delay_construction"]["delay_shuffle_control"]
    return int(d["base_delay_shuffle_seed"]) + int(condition_index) * 7919


def _base_delay_steps(spec: dict[str, Any]) -> np.ndarray:
    dc = spec["delay_construction"]
    n = int(spec["frozen_topology_and_weights"]["n_neurons"])
    step = int(dc["uniform_delay_steps"])
    return np.full((n,), step, dtype=np.int32)


def _delay_steps_for_condition(
    spec: dict[str, Any],
    condition: dict[str, Any],
    condition_index: int,
    *,
    reference_delays: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    policy = condition["delay_policy"]
    base = _base_delay_steps(spec)
    if policy == "uniform":
        return base.copy()
    if policy == "geometry_derived":
        return base.copy()
    if policy == "delay_shuffled":
        ref_id = condition["delay_shuffle_reference"]
        ref = reference_delays[ref_id]
        perm = _fisher_yates_permutation(ref.shape[0], _delay_shuffle_seed(spec, condition_index))
        return ref[perm].copy()
    raise ValueError(f"unknown delay_policy {policy!r}")


def _estimator_positions(
    spec: dict[str, Any],
    condition: dict[str, Any],
    condition_index: int,
) -> np.ndarray:
    n = int(spec["frozen_topology_and_weights"]["n_neurons"])
    r_mm = float(spec["frozen_spatial_coordinates"]["ring_radius_mm"])
    ordered = _ordered_arc_positions_mm(n, radius_mm=r_mm)
    if condition["geometry_layout"] == "ordered":
        return ordered
    perm = _fisher_yates_permutation(n, _geometry_shuffle_seed(spec, condition_index))
    return ordered[perm]


def _stimulus_schedule(spec: dict[str, Any]) -> StimulusSchedule:
    n = int(spec["frozen_topology_and_weights"]["n_neurons"])
    events = []
    for ev in spec["simulation_policy"]["drive"]["events"]:
        events.append(
            {
                "onset_ms": float(ev["onset_ms"]),
                "duration_ms": float(ev["duration_ms"]),
                "amplitude": float(ev["amplitude"]),
                "target_indices": [int(ev["target_neuron"])],
                "is_drive_event": bool(ev.get("is_drive_event", True)),
            }
        )
    return StimulusSchedule(events=tuple(events), n_neurons=n)


def _build_c3_model(
    spec: dict[str, Any],
    *,
    edges,
    emitter,
    positions_3d: np.ndarray,
    seed: int,
):
    n = int(spec["frozen_topology_and_weights"]["n_neurons"])
    sim_pol = spec["simulation_policy"]
    cfg = (
        jtfne.Configuration()
        .runtime(
            seed=int(seed),
            dtype=str(sim_pol["dtype"]),
            duration_ms=float(sim_pol["duration_ms"]),
            dt_ms=float(sim_pol["dt_ms"]),
            recurrent_backend="edge_list",
            enable_hdp=False,
            enable_homeostasis=False,
            hdp_params={"noise_scale": float(sim_pol["noise_scale"])},
        )
        .column("c3_ring", layers=["L4"], n=n)
        .cell_types({"E": 1.0})
        .uniform3d(radius_mm=float(spec["frozen_spatial_coordinates"]["ring_radius_mm"]), height_mm=0.1)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.0)
        .probes(["spikes", "V_m"], n_contacts=2)
    )
    model = jtfne.construct(cfg)
    object.__setattr__(
        model,
        "params",
        {
            **model.params,
            "emitter": emitter,
            "edge_list": edges,
            "positions": jnp.asarray(positions_3d, dtype=jnp.float32),
        },
    )
    return model


def _simulate_cell(
    spec: dict[str, Any],
    *,
    condition: dict[str, Any],
    condition_index: int,
    seed: int,
    reference_delays: dict[str, np.ndarray],
) -> dict[str, Any]:
    topo = spec["frozen_topology_and_weights"]
    n = int(topo["n_neurons"])
    sim_pol = spec["simulation_policy"]
    dt_ms = float(sim_pol["dt_ms"])
    delay_steps = _delay_steps_for_condition(
        spec, condition, condition_index, reference_delays=reference_delays
    )
    params, edges = build_ring_params_edges(
        n,
        delay_steps=delay_steps,
        weight=float(topo["edge_weight"]),
        tau_ms=float(topo["syn_tau_ms"]),
    )
    positions_3d = _ring_positions_3d(n, radius_mm=float(spec["frozen_spatial_coordinates"]["ring_radius_mm"]))
    model = _build_c3_model(spec, edges=edges, emitter=params, positions_3d=positions_3d, seed=seed)
    runtime = jtfne.RuntimeConfig(
        dtype=str(sim_pol["dtype"]),
        recurrent_backend="edge_list",
        enable_hdp=False,
        enable_homeostasis=False,
        hdp_params={"noise_scale": float(sim_pol["noise_scale"])},
    )
    sim = jtfne.Simulation(
        duration_ms=float(sim_pol["duration_ms"]),
        dt_ms=dt_ms,
        seed=int(seed),
        runtime=runtime,
        record_sources=bool(spec["analysis_substrates"]["secondary_optional"][0]["enabled"]),
        record_fields=bool(sim_pol["record_fields"]),
    )
    schedule = _stimulus_schedule(spec)
    signals = model.simulate(sim, paradigm=schedule)
    positions = _estimator_positions(spec, condition, condition_index)
    vm = np.asarray(signals.V_m, dtype=np.float64)
    est = estimate_traveling_wave(vm, positions, dt_ms=dt_ms, spec=load_protocol_spec())
    vc_diag = None
    if condition["delay_policy"] == "geometry_derived" and est.classification == "TRAVELING_WAVE":
        vc = float(spec["delay_construction"]["geometry_derived"]["conduction_velocity_mm_per_ms"])
        if math.isfinite(est.phase_velocity) and vc > 0:
            vc_diag = abs(float(est.phase_velocity) - vc) / vc
    row = {
        "condition_id": condition["id"],
        "seed": int(seed),
        "geometry_layout": condition["geometry_layout"],
        "delay_policy": condition["delay_policy"],
        "delay_steps": delay_steps.tolist(),
        "estimator": est.to_dict(),
        "substrate": "V_m",
        "n_steps": int(vm.shape[0]),
        "n_neurons": int(vm.shape[1]),
    }
    if vc_diag is not None:
        row["v_c_diagnostic_ratio"] = float(vc_diag)
    return row


def run_c3_prospective_execution(
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    """Execute all preregistered C3 cells and return the raw receipt payload."""
    spec = spec or load_c3_spec()
    validate_c3_spec(spec)
    conditions = spec["design_matrix"]["conditions"]
    seeds = [int(s) for s in spec["simulation_policy"]["seeds"]]
    reference_delays: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    for ci, condition in enumerate(conditions):
        if condition["delay_policy"] in ("uniform", "geometry_derived"):
            reference_delays[condition["id"]] = _delay_steps_for_condition(
                spec, condition, ci, reference_delays=reference_delays
            )
    for ci, condition in enumerate(conditions):
        for seed in seeds:
            cells.append(_simulate_cell(spec, condition=condition, condition_index=ci, seed=seed, reference_delays=reference_delays))
    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_c.c3_execution_receipt.v1",
        "checkpoint": "C3",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": str(C3_SPEC_PATH.relative_to(REPO_ROOT)),
        "n_cells": len(cells),
        "expected_n_cells": c3_total_cells(spec),
        "cells": cells,
        "interpretation_deferred_to": "C4",
    }
    if len(cells) != c3_total_cells(spec):
        raise RuntimeError(f"expected {c3_total_cells(spec)} cells, got {len(cells)}")
    return json_safe(receipt)


def write_c3_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_c3_prospective_execution(package_head=package_head)
    C3_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_c3_execution_receipt() -> dict[str, Any]:
    return json.loads(C3_EXECUTION_RECEIPT_PATH.read_text())
