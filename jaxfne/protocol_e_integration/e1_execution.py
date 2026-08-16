"""E1 two-area laminar hierarchy — construction, identity, gates, receipt."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import _edge_delay_steps_host
from jaxfne.io import json_safe
from jaxfne.protocol_e_integration.e1_protocol import (
    E1_EXECUTION_RECEIPT_PATH,
    load_e1_spec,
    validate_e1_spec,
)

REPO_ROOT = jtfne.__file__ and __import__("pathlib").Path(jtfne.__file__).resolve().parents[1]
REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

# Within-area recurrence probability (execution-only; not in frozen E1 spec).
_LOCAL_WITHIN_AREA_PROBABILITY = 0.2

_EDGE_CLASS_BY_RULE: dict[str, str] = {
    "local_A1": "local_A1",
    "local_A2": "local_A2",
    "FF": "FF_A1_to_A2",
    "FB": "FB_A2_to_A1",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _layer_names(spec: dict[str, Any]) -> list[str]:
    return list(spec["hierarchy"]["laminar_layers_per_area"])


def _neurons_per_layer(spec: dict[str, Any]) -> dict[str, int]:
    n = int(spec["simulation_policy"]["n_neurons_per_area"])
    layers = _layer_names(spec)
    if n % len(layers) != 0:
        raise ValueError("n_neurons_per_area must divide evenly across laminar layers")
    per = n // len(layers)
    return {layer: per for layer in layers}


def build_e1_configuration(*, include_inter_area: bool = True) -> jtfne.Configuration:
    """Build the frozen E1 hierarchy Configuration.

    ``include_inter_area=False`` omits FF/FB rules for G6 baseline reduction.
    """
    spec = load_e1_spec()
    validate_e1_spec(spec)
    layers = _layer_names(spec)
    neurons = _neurons_per_layer(spec)
    sim = spec["simulation_policy"]
    drive = spec["simulation_policy"]["drive"]
    layer_fracs = {layer: (i / len(layers), (i + 1) / len(layers)) for i, layer in enumerate(layers)}
    layer_cell_types = {layer: {"E": 0.7, "PV": 0.3} for layer in layers}

    cfg = (
        jtfne.Configuration()
        .areas(list(spec["hierarchy"]["areas"]))
        .update_metadata(connectivity_mode="explicit")
        .layer_fractions(layer_fracs, layer_cell_types=layer_cell_types)
    )
    for area in spec["hierarchy"]["areas"]:
        cfg = cfg.population(
            int(sim["n_neurons_per_area"]),
            dict(neurons),
            name=area,
            layers=layers,
        )
    cfg = (
        cfg.cell_type_drives({str(k): float(v) for k, v in drive.items()})
        .set_emitter("izhikevich", str(sim["emitter_preset"]))
        .probes(["spikes", "V_m"])
        .field(domain="laminar_column", conductivity="proxy")
        .runtime(
            dtype=str(sim["dtype"]),
            recurrent_backend="edge_list",
            noise_scale=float(sim["noise_scale"]),
        )
        .connections(
            name="local_A1",
            source={"area": "A1"},
            target={"area": "A1"},
            probability=_LOCAL_WITHIN_AREA_PROBABILITY,
            weight=0.3,
        )
        .connections(
            name="local_A2",
            source={"area": "A2"},
            target={"area": "A2"},
            probability=_LOCAL_WITHIN_AREA_PROBABILITY,
            weight=0.3,
        )
    )
    if include_inter_area:
        ff = spec["connectivity"]["feedforward"]
        fb = spec["connectivity"]["feedback"]
        cfg = (
            cfg.connections(
                name=str(ff["id"]),
                source=dict(ff["source"]),
                target=dict(ff["target"]),
                probability=1.0,
                weight=0.4,
            ).connections(
                name=str(fb["id"]),
                source=dict(fb["source"]),
                target=dict(fb["target"]),
                probability=1.0,
                weight=0.3,
            )
        )
    return cfg


def build_identity_map(neuron_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map flat index ``i`` to ``(area, layer, cell_type, local_index)``."""
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in neuron_table:
        nid = int(row["neuron_id"])
        key = (str(row["area"]), str(row["layer"]), str(row["cell_type"]))
        groups[key].append(nid)
    for ids in groups.values():
        ids.sort()

    identity: list[dict[str, Any]] = [None] * len(neuron_table)  # type: ignore[list-item]
    for (area, layer, cell_type), ids in sorted(groups.items()):
        for local_index, flat_index in enumerate(ids):
            identity[flat_index] = {
                "flat_index": flat_index,
                "area": area,
                "layer": layer,
                "cell_type": cell_type,
                "local_index": int(local_index),
            }
    if any(row is None for row in identity):
        raise ValueError("identity map failed to cover all neuron_table rows")
    return identity


def identity_round_trip_ok(identity_map: list[dict[str, Any]]) -> bool:
    """Verify ``i <-> (area, layer, cell_type, local_index)`` is bijective."""
    reverse: dict[tuple[str, str, str, int], int] = {}
    for row in identity_map:
        flat = int(row["flat_index"])
        if flat != len(reverse):
            return False
        key = (row["area"], row["layer"], row["cell_type"], int(row["local_index"]))
        if key in reverse:
            return False
        reverse[key] = flat
    return len(reverse) == len(identity_map)


def _row_at(neuron_table: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    return neuron_table[int(idx)]


def _matches_ff_semantics(pre: Mapping[str, Any], post: Mapping[str, Any]) -> bool:
    return (
        pre["area"] == "A1"
        and pre["layer"] in ("L2", "L3")
        and pre["cell_type"] == "E"
        and post["area"] == "A2"
        and post["layer"] == "L4"
    )


def _matches_fb_semantics(pre: Mapping[str, Any], post: Mapping[str, Any]) -> bool:
    return (
        pre["area"] == "A2"
        and pre["layer"] == "L5"
        and pre["cell_type"] == "E"
        and post["area"] == "A1"
        and post["layer"] in ("L2", "L3")
    )


def _rule_edge_slices(model: jtfne.Model) -> list[tuple[str, slice]]:
    rules = (model.cfg.metadata.get("circuit", {}) or {}).get("connections", [])
    start = 0
    slices: list[tuple[str, slice]] = []
    for rule in rules:
        count = int(rule.get("compiled_n_edges", 0))
        end = start + count
        slices.append((str(rule["name"]), slice(start, end)))
        start = end
    return slices


def build_edge_provenance_table(model: jtfne.Model) -> list[dict[str, Any]]:
    """Per-edge typed pathway metadata for E2 delay attachment."""
    neuron_table = model.neuron_table()
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre, dtype=np.int64)
    post = np.asarray(edges.post, dtype=np.int64)
    table: list[dict[str, Any]] = []
    for rule_name, sl in _rule_edge_slices(model):
        edge_class = _EDGE_CLASS_BY_RULE.get(rule_name, rule_name)
        for edge_index, (p, q) in enumerate(zip(pre[sl], post[sl], strict=True)):
            pre_row = _row_at(neuron_table, int(p))
            post_row = _row_at(neuron_table, int(q))
            table.append(
                {
                    "edge_index": int(sl.start + edge_index),
                    "rule_name": rule_name,
                    "edge_class": edge_class,
                    "pre_flat_index": int(p),
                    "post_flat_index": int(q),
                    "pre_area": pre_row["area"],
                    "pre_layer": pre_row["layer"],
                    "pre_cell_type": pre_row["cell_type"],
                    "post_area": post_row["area"],
                    "post_layer": post_row["layer"],
                    "post_cell_type": post_row["cell_type"],
                }
            )
    return table


def _edge_class_semantics_ok(
    edge_class: str,
    pre_row: Mapping[str, Any],
    post_row: Mapping[str, Any],
) -> bool:
    if edge_class == "local_A1":
        return pre_row["area"] == "A1" and post_row["area"] == "A1"
    if edge_class == "local_A2":
        return pre_row["area"] == "A2" and post_row["area"] == "A2"
    if edge_class == "FF_A1_to_A2":
        return _matches_ff_semantics(pre_row, post_row)
    if edge_class == "FB_A2_to_A1":
        return _matches_fb_semantics(pre_row, post_row)
    return False


def verify_connectivity_ownership(provenance: list[dict[str, Any]]) -> dict[str, Any]:
    """G3: every edge matches its declared class with no cross-class leakage."""
    violations: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in provenance:
        edge_class = str(row["edge_class"])
        counts[edge_class] += 1
        pre_row = {
            "area": row["pre_area"],
            "layer": row["pre_layer"],
            "cell_type": row["pre_cell_type"],
        }
        post_row = {
            "area": row["post_area"],
            "layer": row["post_layer"],
            "cell_type": row["post_cell_type"],
        }
        if not _edge_class_semantics_ok(edge_class, pre_row, post_row):
            violations.append({"edge_index": row["edge_index"], "edge_class": edge_class})
    ff_edges = [r for r in provenance if r["edge_class"] == "FF_A1_to_A2"]
    fb_edges = [r for r in provenance if r["edge_class"] == "FB_A2_to_A1"]
    return {
        "passed": not violations and bool(ff_edges) and bool(fb_edges),
        "edge_class_counts": dict(counts),
        "n_violations": len(violations),
        "violations_sample": violations[:8],
        "ff_edge_count": len(ff_edges),
        "fb_edge_count": len(fb_edges),
    }


def _simulate_arrays(model: jtfne.Model, *, duration_ms: float, dt_ms: float, seed: int) -> dict[str, np.ndarray]:
    sig = model.simulate(jtfne.Simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=int(seed)))
    return {
        "V_m": np.asarray(sig.V_m),
        "spikes": np.asarray(sig.spikes),
    }


def _structural_invariants(model: jtfne.Model) -> dict[str, Any]:
    edges = model.params["edge_list"]
    delay_steps = _edge_delay_steps_host(edges)
    metadata = model.cfg.metadata
    return {
        "connectivity_mode": metadata.get("connectivity_mode"),
        "default_edge_count": (metadata.get("connectivity_compilation") or {}).get("default_edge_count"),
        "rbs_enabled": False,
        "all_delay_steps_zero": bool(np.all(delay_steps == 0)),
        "emitter_family_uniform": "izhikevich",
        "heterogeneity_note": (
            "population_parameter_heterogeneity_not_different_emitter_equations"
        ),
    }


def run_e1_hierarchy_runtime(*, package_head: str | None = None) -> dict[str, Any]:
    """Execute G1–G6 structural gates and return the write-once receipt payload."""
    spec = load_e1_spec()
    validate_e1_spec(spec)
    sim = spec["simulation_policy"]
    duration_ms = float(sim["duration_ms"])
    dt_ms = float(sim["dt_ms"])
    seeds = [int(s) for s in sim["seeds"]]

    gates: dict[str, dict[str, Any]] = {
        "G1_construction": {"passed": False},
        "G2_identity_recovery": {"passed": False},
        "G3_ff_fb_ownership": {"passed": False},
        "G4_finite_deterministic_execution": {"passed": False},
        "G5_reproducibility": {"passed": False},
        "G6_baseline_structural_reduction": {"passed": False},
    }

    cfg = build_e1_configuration(include_inter_area=True)
    model = jtfne.construct(cfg)
    gates["G1_construction"] = {
        "passed": True,
        "n_neurons": len(model.neuron_table()),
        "areas": sorted({r["area"] for r in model.neuron_table()}),
        "layers": sorted({r["layer"] for r in model.neuron_table()}),
        "cell_types": sorted({r["cell_type"] for r in model.neuron_table()}),
        "n_edges": int(model.params["edge_list"].n_edges),
    }

    neuron_table = model.neuron_table()
    identity_map = build_identity_map(neuron_table)
    table_ids_match = all(int(row["neuron_id"]) == i for i, row in enumerate(neuron_table))
    round_trip = identity_round_trip_ok(identity_map)
    gates["G2_identity_recovery"] = {
        "passed": table_ids_match and round_trip,
        "neuron_table_ids_sequential": table_ids_match,
        "identity_round_trip": round_trip,
        "n_identity_rows": len(identity_map),
        "sample_identity": identity_map[:3] + identity_map[-1:],
    }

    provenance = build_edge_provenance_table(model)
    g3 = verify_connectivity_ownership(provenance)
    gates["G3_ff_fb_ownership"] = g3

    seed_runs: list[dict[str, Any]] = []
    g4_ok = True
    g5_ok = True
    for seed in seeds:
        arrays_a = _simulate_arrays(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
        arrays_b = _simulate_arrays(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
        finite = bool(np.isfinite(arrays_a["V_m"]).all() and np.isfinite(arrays_a["spikes"]).all())
        reproducible = bool(np.array_equal(arrays_a["V_m"], arrays_b["V_m"]))
        g4_ok = g4_ok and finite
        g5_ok = g5_ok and reproducible
        # Post-sim identity interpretation: flat index labels unchanged.
        interpreted = [
            {
                "flat_index": i,
                "area": identity_map[i]["area"],
                "layer": identity_map[i]["layer"],
                "cell_type": identity_map[i]["cell_type"],
                "local_index": identity_map[i]["local_index"],
                "V_m_final": float(arrays_a["V_m"][-1, i]),
            }
            for i in (0, len(identity_map) // 2, len(identity_map) - 1)
        ]
        seed_runs.append(
            {
                "seed": seed,
                "finite": finite,
                "reproducible": reproducible,
                "V_m_shape": list(arrays_a["V_m"].shape),
                "interpreted_samples": interpreted,
            }
        )

    gates["G4_finite_deterministic_execution"] = {"passed": g4_ok, "seeds": seed_runs}
    gates["G5_reproducibility"] = {"passed": g5_ok, "seeds": [r["seed"] for r in seed_runs]}

    baseline_cfg = build_e1_configuration(include_inter_area=False)
    baseline_model = jtfne.construct(baseline_cfg)
    baseline_table = baseline_model.neuron_table()
    baseline_identity = build_identity_map(baseline_table)
    baseline_prov = build_edge_provenance_table(baseline_model)
    cross_area = [
        row
        for row in baseline_prov
        if row["pre_area"] != row["post_area"]
    ]
    edge_classes = {row["edge_class"] for row in baseline_prov}
    identity_preserved = identity_map == baseline_identity and neuron_table == baseline_table
    g6_ok = (
        not cross_area
        and edge_classes <= {"local_A1", "local_A2"}
        and identity_preserved
        and identity_round_trip_ok(baseline_identity)
    )
    gates["G6_baseline_structural_reduction"] = {
        "passed": g6_ok,
        "inter_area_edges_disabled": True,
        "n_cross_area_edges": len(cross_area),
        "edge_classes": sorted(edge_classes),
        "identity_unchanged_vs_full_hierarchy": identity_preserved,
        "baseline_finite": bool(
            np.isfinite(
                _simulate_arrays(
                    baseline_model,
                    duration_ms=duration_ms,
                    dt_ms=dt_ms,
                    seed=seeds[0],
                )["V_m"]
            ).all()
        ),
    }

    invariants = _structural_invariants(model)
    if not invariants["all_delay_steps_zero"]:
        raise RuntimeError("E1 invariant failed: nonzero edge delay_steps")

    edge_class_counts = defaultdict(int)
    for row in provenance:
        edge_class_counts[row["edge_class"]] += 1

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e1_execution_receipt.v1",
        "checkpoint": "E1",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_e_integration/e1_hierarchy_runtime_spec.json",
        "scientific_question": (
            "Can jaxfne execute a two-area laminar hierarchy while preserving "
            "declared biological identity and directional connectivity?"
        ),
        "failure_condition": (
            "finite simulation with corrupted hierarchy identity is E1 failure"
        ),
        "emitter_note": (
            "heterogeneous area/layer/population identity and parameters; "
            "common izhikevich F_X family only"
        ),
        "local_within_area_probability": _LOCAL_WITHIN_AREA_PROBABILITY,
        "structural_invariants": invariants,
        "identity_map": identity_map,
        "edge_provenance_summary": {
            "edge_class_counts": dict(edge_class_counts),
            "n_edges": len(provenance),
        },
        "edge_provenance_table": provenance,
        "gates": gates,
        "e2_deferred": "typed delays on provenance edge classes",
    }
    for gname, gval in gates.items():
        if not gval.get("passed"):
            raise RuntimeError(f"E1 gate failed: {gname}")
    return json_safe(receipt)


def write_e1_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_e1_hierarchy_runtime(package_head=package_head)
    E1_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_e1_execution_receipt() -> dict[str, Any]:
    return json.loads(E1_EXECUTION_RECEIPT_PATH.read_text())
