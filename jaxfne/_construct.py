"""construct()/connect() hub: builds Model from Configuration, merges Models,
plus the remaining group-1 connectivity compiler helpers and group-8 paradigm/
receipt/manifest-adjacent module-level functions.

Split out of ``jaxfne/core.py`` (final slice, 5 of 5, of the core.py monolith
split, see ``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports
every symbol here for backward compatibility -- import from ``jaxfne.core``,
not this module, unless you are working on core.py itself.

This is the genuine hub: it imports Configuration (jaxfne/_config.py) and
Model (jaxfne/_model.py) -- both already independent leaf modules -- to
build/merge models. Nothing in _config.py/_runtime_config.py/_signals.py/
_model.py imports from this module, so there is no cycle.
"""

from __future__ import annotations

import contextlib
import json
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import (
    EdgeList,
    EIGNetwork,
    IzhikevichParams,
    make_edge_list_from_dense,
    make_eig_network,
    izhikevich_params_from_labels,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_homeostatic,
    simulate_edge_recurrent_izhikevich_hdp,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)

from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, load_json, manifest as build_manifest
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN
from ._runtime_config import (
    RuntimeConfig,
    SurrogateConfig,
    _ALLOWED_DTYPES,
    _ALLOWED_SYNAPTIC_KERNELS,
    _device_scope,
    _jaxlib_version,
    _resolve_backend_device,
)
from ._config import (
    Configuration,
    Config,
    _default_operator_status,
    _circuit_json_safe,
    _default_metadata,
    _counts_from_fractions,
    _reject_retired_like,
    _ProbeDeclarations,
)

# v0.3.29: canonical selector/identity types live in experimental_hpc.contracts.
# Re-export (do not duplicate) so the stable API exposes one definition.
from .experimental_hpc.contracts import NodeIdentity, SelectorSpec
from ._signals import (
    Simulation,
    Probe,
    Signals,
    Signal,
    Objective,
    DatasetSpec,
    TrialSpec,
    TrialBatch,
    TrialResult,
    TrialBatchResult,
    ReadoutSpec,
    ReadoutResult,
    ObjectiveReport,
    RunReceipt,
    StimulusSchedule,
    LaminarPopulation,
    LaminarSourceGeometry,
    AxisSpec,
    BasisSpec,
    default_basis_spec,
    ParadigmEvent,
    ParadigmCondition,
    Paradigm,
    paradigm,
    evoked_l4_drive_paradigm,
    omission_oddball_paradigm,
    coop_omission_oddball_paradigm,
    general_sequential_oddball_paradigm,
    general_delayed_match_to_sample_paradigm,
    _make_poisson_drive,
    _finite_or_none,
    _compute_kappa_synchrony_metric,
    _compute_all_metrics,
    _check_gate_criterion,
    _evaluate_loss_spec,
    _evaluate_regularizer_spec,
    _evaluate_gate_spec,
    _default_basis_dict,
    _normalize_manifest_readout,
    _KNOWN_METRICS,
    _KNOWN_LAYERS,
    _KNOWN_CONFIG_GATE_METRICS,
    _SIGNALS_GET_NEURON_AXIS_KEYS,
    _SIGNALS_GET_FIELD_KEYS,
    _SIGNALS_GET_KEY_ALIASES,
    _AXIS_STATUS_VALUES,
    _SPACE_BASIS_VALUES,
    _TIME_BASIS_VALUES,
    _FIELD_REGIME_VALUES,
    _FUTURE_FIELD_REGIMES,
    _SOURCE_MODE_BASIS_VALUES,
    _PROBE_BASIS_VALUES,
    _CONSERVATIVE_TRUTH_DEFAULTS,
)
from ._model import (
    Model,
    MatrixParameterSpec,
    matrix_parameter,
    TuneResult,
    stimulus_schedule,
    with_emitter_parameters,
    _model_with_scalar_parameter,
    _mask_for_parameter,
    _model_with_matrix_parameter,
    _model_with_parameters,
    _evaluate_soft_rate_targets,
    _JAXFNE_VERSION,
    _RECEIPT_SCHEMA_VERSION,
    _MANIFEST_SCHEMA_VERSION,
    _SOURCE_PROXY_METADATA,
    _KNOWN_READOUT_METRICS,
)




_SUITE2_LAYER_FRACTIONS = {
    "L1": (0.00, 0.10),
    "L2": (0.10, 0.25),
    "L3": (0.25, 0.45),
    "L4": (0.45, 0.55),
    "L5": (0.55, 0.85),
    "L6": (0.85, 1.00),
}

_SUITE2_LAYER_CELL_TYPES_V1 = {
    "L1": {"E": 0.75, "PV": 0.00, "SST": 0.00, "VIP": 0.25},
    "L2": {"E": 0.75, "PV": 0.05, "SST": 0.05, "VIP": 0.15},
    "L3": {"E": 0.75, "PV": 0.10, "SST": 0.10, "VIP": 0.05},
    "L4": {"E": 0.25, "PV": 0.45, "SST": 0.15, "VIP": 0.15},
    "L5": {"E": 0.15, "PV": 0.25, "SST": 0.30, "VIP": 0.30},
    "L6": {"E": 0.10, "PV": 0.20, "SST": 0.20, "VIP": 0.50},
}

_SUITE2_LAYER_CELL_TYPES_V4 = {
    "L1": {"E": 0.70, "PV": 0.00, "SST": 0.00, "VIP": 0.30},
    "L2": {"E": 0.70, "PV": 0.08, "SST": 0.06, "VIP": 0.16},
    "L3": {"E": 0.70, "PV": 0.12, "SST": 0.12, "VIP": 0.06},
    "L4": {"E": 0.30, "PV": 0.40, "SST": 0.15, "VIP": 0.15},
    "L5": {"E": 0.20, "PV": 0.25, "SST": 0.30, "VIP": 0.25},
    "L6": {"E": 0.15, "PV": 0.20, "SST": 0.25, "VIP": 0.40},
}

_SUITE2_PROXY_MODES = (
    "spikes", "V_m", "source", "LFP", "CSD", "EEG-proxy", "MEG-proxy", "EMM-proxy"
)


def _suite2_default_layer_cell_types() -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in _SUITE2_LAYER_CELL_TYPES_V1.items()}


def _layer_ranges_for(layers: Sequence[str], metadata: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    declared = metadata.get("layer_fractions") or _SUITE2_LAYER_FRACTIONS
    out: dict[str, tuple[float, float]] = {}
    missing = [str(layer) for layer in layers if str(layer) not in declared]
    if missing:
        n = len(layers)
        for idx, layer in enumerate(layers):
            out[str(layer)] = (idx / max(n, 1), (idx + 1) / max(n, 1))
        return out
    for layer in layers:
        z0, z1 = declared[str(layer)]
        out[str(layer)] = (float(z0), float(z1))
    return out


def _area_layer_cell_type_map(metadata: Mapping[str, Any], area: str) -> dict[str, dict[str, float]]:
    per_area = metadata.get("area_layer_cell_types", {}) or {}
    if area in per_area:
        return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in per_area[area].items()}
    base = metadata.get("layer_cell_types", None)
    if base:
        return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in base.items()}
    # A flat cell_types() declaration (metadata["cell_types"]) was previously
    # invisible here -- Configuration.cell_types() writes that key, but this
    # resolver only ever checked area_layer_cell_types/layer_cell_types, so a
    # bare .cell_types({...}) after .population()/.geometry() silently fell
    # through to the hardcoded _suite2_default_layer_cell_types() table below
    # instead (confirmed regression: {"E":0.5,"PV":0.5} at N=2 on layer "L4"
    # produced ["PV","VIP"], not ["E","PV"], because L4's hardcoded default is
    # {"E":0.25,"PV":0.45,"SST":0.15,"VIP":0.15}). Broadcasting the flat
    # declaration to every layer here closes that gap without touching
    # cell_types()'s own write behavior.
    flat = metadata.get("cell_types", None)
    if flat:
        clean = {str(k): float(v) for k, v in flat.items()}
        layers = metadata.get("layer_fractions") or _SUITE2_LAYER_FRACTIONS
        return {str(layer): dict(clean) for layer in layers}
    return _suite2_default_layer_cell_types()


def _area_layer_count_frac(metadata: Mapping[str, Any], area: str) -> dict[str, float] | None:
    """Per-layer neuron *population* fractions (count budget, NOT thickness).

    These are set by ``Configuration.population(...)`` and decouple how many
    neurons a layer gets from how thick it is (e.g. a thin but dense L2 holding
    more neurons than a thick but sparse L5). Resolution order: per-area override
    (``area_layer_count_frac``) -> global ``layer_count_frac`` -> ``None`` (caller
    falls back to the historical thickness-proportional allocation).
    """
    per_area = metadata.get("area_layer_count_frac", {}) or {}
    if area in per_area:
        return {str(k): float(v) for k, v in per_area[area].items()}
    base = metadata.get("layer_count_frac", None)
    if base:
        return {str(k): float(v) for k, v in base.items()}
    return None


def _suite2_neuron_population_from_config(cfg: "Configuration", *, dtype: str = "float32") -> tuple[IzhikevichParams, jax.Array, dict[str, Any]]:
    """Build explicit Suite No. 2 neuron metadata and reduced emitter arrays."""

    metadata = cfg.metadata
    net = cfg.networks[0]
    columns = [dict(c) for c in metadata.get("columns", [])]
    if not columns:
        columns = [{"name": str(net.get("name", "net1")), "layers": ["uniform_3d"], "n": int(net.get("n", 100)), "start_index": 0, "stop_index": int(net.get("n", 100))}]
    seed = int(metadata.get("seed", 0))
    uniform_3d = bool(metadata.get("uniform_3d", False))
    global_cell_types = {str(k): float(v) for k, v in net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}).items()}

    labels: list[str] = []
    layer_labels: list[str] = []
    area_labels: list[str] = []
    neuron_rows: list[dict[str, Any]] = []
    position_chunks: list[jax.Array] = []
    jdtype = jnp.float64 if dtype == "float64" and bool(jax.config.read("jax_enable_x64")) else jnp.float32
    neuron_id = 0

    for area_idx, column in enumerate(columns):
        area = str(column.get("name", f"area_{area_idx}"))
        n_col = int(column.get("n", 0))
        if n_col <= 0:
            continue
        layers = [str(x) for x in column.get("layers", ["uniform_3d"])] or ["uniform_3d"]
        key = jax.random.PRNGKey(seed + 1009 * (area_idx + 1))

        if uniform_3d or layers == ["uniform_3d"]:
            counts_by_type = _counts_from_fractions(n_col, global_cell_types)
            col_labels: list[str] = []
            for cell_type, count in counts_by_type.items():
                col_labels.extend([cell_type] * count)
            col_labels = col_labels[:n_col] + ["E"] * max(0, n_col - len(col_labels))
            x_key, y_key, z_key = jax.random.split(key, 3)
            radius = float(metadata.get("column_radius_mm", 0.25))
            height = float(metadata.get("column_height_mm", 1.60))
            x = jax.random.uniform(x_key, (n_col,), minval=-radius, maxval=radius, dtype=jdtype) + jnp.asarray(area_idx * 2.0, dtype=jdtype)
            y = jax.random.uniform(y_key, (n_col,), minval=-radius, maxval=radius, dtype=jdtype)
            z = jax.random.uniform(z_key, (n_col,), minval=0.0, maxval=height, dtype=jdtype)
            position_chunks.append(jnp.stack([x, y, z], axis=1))
            for local_idx, cell_type in enumerate(col_labels[:n_col]):
                labels.append(cell_type)
                layer_labels.append("uniform_3d")
                area_labels.append(area)
                neuron_rows.append({"neuron_id": neuron_id, "area": area, "layer": "uniform_3d", "cell_type": cell_type, "x": float(x[local_idx]), "y": float(y[local_idx]), "z": float(z[local_idx])})
                neuron_id += 1
            continue

        layer_ranges = _layer_ranges_for(layers, metadata)
        count_frac = _area_layer_count_frac(metadata, area)
        if count_frac is not None and sum(count_frac.get(layer, 0.0) for layer in layers) > 0.0:
            # Population-fraction allocation (decoupled from thickness).
            alloc = {layer: float(count_frac.get(layer, 0.0)) for layer in layers}
        else:
            # Historical fallback: allocate neurons proportional to layer thickness.
            alloc = {layer: max(0.0, layer_ranges[layer][1] - layer_ranges[layer][0]) for layer in layers}
        layer_counts = _counts_from_fractions(n_col, alloc)
        area_layer_map = _area_layer_cell_type_map(metadata, area)
        for layer_idx, layer in enumerate(layers):
            n_layer = int(layer_counts.get(layer, 0))
            if n_layer <= 0:
                continue
            z0, z1 = layer_ranges[layer]
            type_fracs = area_layer_map.get(layer, global_cell_types)
            type_counts = _counts_from_fractions(n_layer, type_fracs)
            layer_cell_labels: list[str] = []
            for cell_type, count in type_counts.items():
                layer_cell_labels.extend([cell_type] * count)
            layer_cell_labels = layer_cell_labels[:n_layer] + ["E"] * max(0, n_layer - len(layer_cell_labels))
            x_key, y_key, z_key = jax.random.split(jax.random.fold_in(key, layer_idx), 3)
            radius = float(metadata.get("column_radius_mm", 0.25))
            x = jax.random.uniform(x_key, (n_layer,), minval=-radius, maxval=radius, dtype=jdtype) + jnp.asarray(area_idx * 2.0, dtype=jdtype)
            y = jax.random.uniform(y_key, (n_layer,), minval=-radius, maxval=radius, dtype=jdtype)
            z = jax.random.uniform(z_key, (n_layer,), minval=float(z0), maxval=float(z1), dtype=jdtype)
            position_chunks.append(jnp.stack([x, y, z], axis=1))
            for local_idx, cell_type in enumerate(layer_cell_labels[:n_layer]):
                labels.append(cell_type)
                layer_labels.append(layer)
                area_labels.append(area)
                neuron_rows.append({"neuron_id": neuron_id, "area": area, "layer": layer, "cell_type": cell_type, "x": float(x[local_idx]), "y": float(y[local_idx]), "z": float(z[local_idx])})
                neuron_id += 1

    # Apply baseline_drive_by_cell_type from drive specification if present
    drive_spec = metadata.get("drive", {})
    baseline_drive = drive_spec.get("baseline_drive_by_cell_type") if isinstance(drive_spec, dict) else None

    params = izhikevich_params_from_labels(
        labels,
        layer_labels=layer_labels,
        dtype=dtype,
        drive_overrides=baseline_drive,
    )
    
    # Compile and apply declared cell_params overrides
    circuit = metadata.get("circuit", {})
    cell_param_decls = circuit.get("cell_params", [])
    if cell_param_decls:
        import numpy as np
        a_list = np.array(params.a)
        b_list = np.array(params.b)
        c_list = np.array(params.c)
        d_list = np.array(params.d)
        drive_list = np.array(params.drive)
        for decl in cell_param_decls:
            selector = decl.get("selector", {})
            param_overrides = decl.get("params", {})
            for i in range(len(labels)):
                match = True
                if "cell_type" in selector and labels[i] != selector["cell_type"]:
                    match = False
                if "layer" in selector and layer_labels[i] != selector["layer"]:
                    match = False
                if match:
                    if "a" in param_overrides:
                        a_list[i] = float(param_overrides["a"])
                    if "b" in param_overrides:
                        b_list[i] = float(param_overrides["b"])
                    if "c" in param_overrides:
                        c_list[i] = float(param_overrides["c"])
                    if "d" in param_overrides:
                        d_list[i] = float(param_overrides["d"])
                    if "drive" in param_overrides:
                        drive_list[i] = float(param_overrides["drive"])
        params = replace(
            params,
            a=jnp.asarray(a_list, dtype=jdtype),
            b=jnp.asarray(b_list, dtype=jdtype),
            c=jnp.asarray(c_list, dtype=jdtype),
            d=jnp.asarray(d_list, dtype=jdtype),
            drive=jnp.asarray(drive_list, dtype=jdtype),
        )

    positions = jnp.concatenate(position_chunks, axis=0) if position_chunks else jnp.zeros((0, 3), dtype=jdtype)
    params, _prebuilt_edges = _suite2_apply_connectivity(params, area_labels, layer_labels, labels, metadata, seed=seed, dtype=dtype)
    geometry_meta = {
        "neuron_rows": neuron_rows,
        "area_labels": area_labels,
        "layer_labels": layer_labels,
        "cell_type_labels": labels,
        "geometry_mode": metadata.get("geometry_mode", "declared_metadata_not_solved_3d_pde_grid"),
        "position_units": "mm_declared_metadata",
        "uniform_3d": bool(uniform_3d),
    }
    return params, positions, geometry_meta, _prebuilt_edges


# Above this neuron count, a dense all-to-all within-area weight matrix is flagged
# as an O(N^2) cost (memory + edges). Sparse connectivity (p_connect<1) avoids it.
_DENSE_CONNECTIVITY_WARN_N = 5000


# At/above this neuron count, a SPARSE within-area connectivity request (p_connect<1
# with no special-structure specs) is built as an edge list DIRECTLY, skipping the
# dense (n,n) weight matrix entirely. Below it, the dense-then-mask path is kept so
# small-N behavior (and the W-inspecting tests) stay exact.
_SPARSE_DIRECT_N = 5000


def _make_sparse_within_area_edges(area_labels, sign, n, *, within_gain, p_connect, seed, jdtype):
    """Build a within-area sparse EdgeList directly (no dense (n,n) materialization).

    Erdos-Renyi-style: each within-area ordered pair (j->i, i!=j) is included with
    probability ``p_connect``. Edge weight matches the dense path's connected-edge
    magnitude ``within_gain * U(0.25,1) * sign[pre] / (sqrt(n) * p_connect)`` so the
    expected total input is preserved. Edges are excitatory/inhibitory by presyn
    sign (receptor_index + tau follow make_edge_list_from_dense). Deterministic from
    ``seed``. Statistically equivalent to the dense path; NOT bit-identical.
    """
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 4242)
    codes = _np.unique(_np.asarray(area_labels), return_inverse=True)[1]
    sign_np = _np.asarray(sign, dtype=_np.float64)
    sqrt_n = float(max(n, 1)) ** 0.5
    p = float(p_connect)
    pre_chunks, post_chunks, w_chunks = [], [], []
    for code in _np.unique(codes):
        idx = _np.where(codes == code)[0]
        m = idx.size
        if m < 2:
            continue
        n_edges = int(rng.binomial(m * (m - 1), p))
        if n_edges == 0:
            continue
        pre_loc = rng.integers(0, m, n_edges)
        post_loc = rng.integers(0, m, n_edges)
        self_loop = pre_loc == post_loc
        while self_loop.any():                      # resample i==j (no self-edges)
            post_loc[self_loop] = rng.integers(0, m, int(self_loop.sum()))
            self_loop = pre_loc == post_loc
        pre_g = idx[pre_loc]
        post_g = idx[post_loc]
        rnd = rng.uniform(0.25, 1.0, n_edges)
        w = within_gain * rnd * sign_np[pre_g] / (sqrt_n * p)
        pre_chunks.append(pre_g); post_chunks.append(post_g); w_chunks.append(w)
    if pre_chunks:
        pre = _np.concatenate(pre_chunks); post = _np.concatenate(post_chunks)
        w = _np.concatenate(w_chunks)
    else:
        pre = _np.zeros(0, int); post = _np.zeros(0, int); w = _np.zeros(0, float)
    receptor = (w < 0).astype(_np.int32)
    tau = _np.where(receptor == 0, 2.0, 5.0)
    return EdgeList(
        pre=jnp.asarray(pre, jnp.int32),
        post=jnp.asarray(post, jnp.int32),
        weight=jnp.asarray(w, jdtype),
        receptor_index=jnp.asarray(receptor, jnp.int32),
        tau_ms=jnp.asarray(tau, jdtype),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def _suite2_apply_connectivity(params: IzhikevichParams, area_labels: Sequence[str], layer_labels: Sequence[str], cell_labels: Sequence[str], metadata: Mapping[str, Any], *, seed: int, dtype: str):
    n = len(cell_labels)
    if n == 0:
        return params, None
    jdtype = jnp.float64 if dtype == "float64" and bool(jax.config.read("jax_enable_x64")) else jnp.float32
    connectivity_spec = metadata.get("connectivity", {}) or {}
    p_connect = connectivity_spec.get("p_connect")
    base_gain = float(connectivity_spec.get("within_gain", 0.45))

    # ── Sparse-direct path ────────────────────────────────────────────────────
    # For a sparse within-area request at scale (p_connect<1, n>=threshold) with no
    # special-structure specs (TCM / suite2-interarea / inter-column), build the
    # EdgeList DIRECTLY and skip the dense (n,n) weight matrix entirely. Returns a
    # (0,0) placeholder W; the model is run on the edge_list backend. This is the
    # O(N^2)-memory escape for large sparse columns. Small-N / all-to-all / special
    # specs keep the exact dense path below.
    _tcm = bool(metadata.get("tcm_v1_6pop", False)) or bool(connectivity_spec.get("tcm_v1_6pop", False))
    _interarea = bool(metadata.get("suite2_interarea", False))
    _inter_col = bool(metadata.get("inter_column_connectivity"))
    if (p_connect is not None and 0.0 < float(p_connect) < 1.0
            and n >= _SPARSE_DIRECT_N and not (_tcm or _interarea or _inter_col)):
        edges = _make_sparse_within_area_edges(
            area_labels, params.sign, n,
            within_gain=base_gain, p_connect=float(p_connect), seed=seed, jdtype=jdtype)
        placeholder_W = jnp.zeros((0, 0), dtype=jdtype)
        return replace(params, W=placeholder_W), edges

    key = jax.random.PRNGKey(seed + 4242)
    rnd = jax.random.uniform(key, (n, n), minval=0.25, maxval=1.0, dtype=jdtype)
    sign = params.sign.astype(jdtype)
    # Vectorized same-area mask: O(N^2) numpy broadcast instead of an O(N^2)
    # pure-Python double comprehension (~100M ops at N=10k dominated construct cost).
    import numpy as _np
    _area_codes = _np.unique(_np.asarray(area_labels), return_inverse=True)[1]
    same_area = jnp.asarray(_area_codes[:, None] == _area_codes[None, :], dtype=jdtype)
    eye = jnp.eye(n, dtype=jdtype)
    W = base_gain * rnd * sign[None, :] * same_area * (1.0 - eye) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))

    # Transparency: dense all-to-all within-area connectivity materializes an (n,n)
    # weight matrix and ~n^2 edges. That is inherent to all-to-all topology, not a
    # bug, but it is O(N^2) in memory and edge count. Make the cost visible at scale
    # and point to the sparse lever (p_connect < 1) rather than failing silently.
    if n >= _DENSE_CONNECTIVITY_WARN_N and (p_connect is None or float(p_connect) >= 1.0):
        import warnings as _warnings
        _mb = (n * n * (8 if jdtype == jnp.float64 else 4)) / 1e6
        _warnings.warn(
            f"dense all-to-all within-area connectivity at N={n} materializes an "
            f"({n}x{n}) weight matrix (~{_mb:.0f} MB) and ~{n*n} edges (O(N^2)). "
            f"For large columns set connectivity p_connect<1 (or a sparse rule) to "
            f"reduce memory and edge count.",
            RuntimeWarning,
            stacklevel=2,
        )
    if p_connect is not None:
        p_val = float(p_connect)
        if 0.0 < p_val < 1.0:
            mask_key = jax.random.fold_in(key, 999)
            mask = jax.random.bernoulli(mask_key, p_val, (n, n)).astype(jdtype)
            W = W * mask / p_val

    # Apply TCM_V1_6POP specific population-based connection mask if enabled
    tcm_active = bool(metadata.get("tcm_v1_6pop", False)) or bool(connectivity_spec.get("tcm_v1_6pop", False))
    if tcm_active:
        import numpy as np
        pop = []
        for idx in range(n):
            lyr = layer_labels[idx]
            cell = cell_labels[idx]
            is_e = (cell == "E")
            if lyr in {"L2/3", "L2", "L3"}:
                pop.append("SP" if is_e else "SI")
            elif lyr == "L4":
                pop.append("SS" if is_e else "other")
            elif lyr == "L5":
                pop.append("DP" if is_e else "DI")
            elif lyr == "L6":
                pop.append("TP" if is_e else "other")
            else:
                pop.append("other")
        allowed_pairs = {
            ("SS", "SP"),
            ("SS", "SI"),
            ("SP", "SP"),
            ("SP", "SI"),
            ("SI", "SP"),
            ("DP", "DP"),
            ("DP", "DI"),
            ("DI", "DP"),
            ("SP", "DP"),
            ("DP", "SP"),
            ("SP", "TP"),
            ("TP", "DP")
        }
        # Vectorized population-pair mask (bit-identical to the prior O(N^2) Python
        # double loop, which was ~N^2 Python-level iterations — ~1e8 at N=1e4).
        # tcm_mask[i, j] = 1 iff (pop[j] (pre), pop[i] (post)) in allowed_pairs.
        uniq = list(dict.fromkeys(pop))
        code = {p: c for c, p in enumerate(uniq)}
        K = len(uniq)
        allowed_code = np.zeros((K, K), dtype=float)   # [src_code, dst_code]
        for src, dst in allowed_pairs:
            if src in code and dst in code:
                allowed_code[code[src], code[dst]] = 1.0
        codes = np.fromiter((code[p] for p in pop), dtype=np.intp, count=n)
        tcm_mask = allowed_code[codes[None, :], codes[:, None]]  # [i,j] = allowed[codes[j], codes[i]]
        W = W * jnp.asarray(tcm_mask, dtype=jdtype)

    if bool(metadata.get("suite2_interarea", False)):
        pre_v1_l23 = jnp.asarray([area_labels[j] == "V1" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v4_l4 = jnp.asarray([area_labels[i] == "V4" and layer_labels[i] == "L4" and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        pre_v4_l23 = jnp.asarray([area_labels[j] == "V4" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v1_deep_l1 = jnp.asarray([area_labels[i] == "V1" and layer_labels[i] in {"L1", "L5", "L6"} and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        ff_gain = float((metadata.get("connectivity", {}) or {}).get("feedforward_gain", 0.65)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        fb_gain = float((metadata.get("connectivity", {}) or {}).get("feedback_gain", 0.50)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        W = W + ff_gain * (post_v4_l4[:, None] * pre_v1_l23[None, :])
        W = W + fb_gain * (post_v1_deep_l1[:, None] * pre_v4_l23[None, :])

    # Declarative inter-column connectivity (from Configuration.inter_column_connectivity).
    # Materializes real cross-area edges using anatomical routing: feedforward
    # from source L2/3 E-cells -> target L4; feedback from source L6 -> target
    # L1/L5. An explicit layer_to_layer_map overrides these defaults. Proxy
    # scaffold connectivity, not a calibrated axonal projection.
    inter = metadata.get("inter_column_connectivity")
    if inter:
        specs = inter if isinstance(inter, list) else [inter]
        for spec in specs:
            W = W + _interarea_W(spec, area_labels, layer_labels, cell_labels, n, jdtype, seed)

    return replace(params, W=W), None


def _interarea_layer_set(name: str) -> set[str]:
    """Resolve a routing layer name to the actual labels it should match.

    Tolerant of the merged 5-layer scheme (``"L2/3"``) and split columns
    (``"L2"``, ``"L3"``): any superficial name matches all superficial variants,
    so inter-area feedforward (source L2/3 -> target L4) wires regardless of
    whether a column declares ``"L2/3"`` or separate ``"L2"``/``"L3"`` layers.
    """
    superficial = {"L2", "L3", "L2/3", "L23"}
    return set(superficial) if name in superficial else {name}


def _interarea_W(
    spec: Mapping[str, Any],
    area_labels: Sequence[str],
    layer_labels: Sequence[str],
    cell_labels: Sequence[str],
    n: int,
    jdtype: Any,
    default_seed: int,
) -> jax.Array:
    """Build the additive cross-area weight contribution for one inter-column spec.

    Anatomical routing rules (default-to-rules, allow override):
      * feedforward: source L2/3 excitatory -> target L4
      * feedback:    source L6 -> target L1/L5
    A target layer of L4 is treated as feedforward (uses ``p_feedforward`` and
    ``feedforward_weight_range``); any other target layer is feedback. An
    explicit ``layer_to_layer_map`` replaces the default pairs. Edges are
    excitatory (positive) since the projecting source is an excitatory cell.
    """
    src_area = str(spec.get("source_area", "V1"))
    dst_area = str(spec.get("target_area", "V4"))
    l2l = spec.get("layer_to_layer_map")
    if l2l:
        pairs = [(str(s), str(t)) for s, t in dict(l2l).items()]
    else:
        # Anatomical routing. "L2/3" matches both merged and split schemes
        # (see _interarea_layer_set), so one feedforward pair suffices.
        pairs = [("L2/3", "L4"), ("L6", "L1"), ("L6", "L5")]
    p_ff = float(spec.get("p_feedforward", 0.3) or 0.0)
    p_fb = float(spec.get("p_feedback", 0.2) or 0.0)
    ff_range = tuple(spec.get("feedforward_weight_range") or (0.5, 2.0))
    fb_range = tuple(spec.get("feedback_weight_range") or (0.3, 1.5))
    seed = spec.get("seed")
    seed = int(seed) if seed is not None else int(default_seed)
    key = jax.random.PRNGKey(seed + 7777)
    inv_sqrt = 1.0 / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
    Wadd = jnp.zeros((n, n), dtype=jdtype)
    for src_layer, dst_layer in pairs:
        src_layers = _interarea_layer_set(src_layer)
        dst_layers = _interarea_layer_set(dst_layer)
        is_ff = bool(dst_layers & {"L4"})
        p = p_ff if is_ff else p_fb
        if p <= 0.0:
            continue
        wlo, whi = (ff_range if is_ff else fb_range)
        pre = jnp.asarray(
            [area_labels[j] == src_area and layer_labels[j] in src_layers and cell_labels[j] == "E" for j in range(n)],
            dtype=jdtype,
        )
        post = jnp.asarray(
            [area_labels[i] == dst_area and layer_labels[i] in dst_layers for i in range(n)],
            dtype=jdtype,
        )
        pair_mask = post[:, None] * pre[None, :]
        key, k_bern, k_w = jax.random.split(key, 3)
        bern = (jax.random.uniform(k_bern, (n, n), dtype=jdtype) < p).astype(jdtype)
        wval = jax.random.uniform(k_w, (n, n), minval=float(wlo), maxval=float(whi), dtype=jdtype)
        Wadd = Wadd + pair_mask * bern * wval * inv_sqrt
    return Wadd


# Candidate (pre x post) product above which a connection rule falls back to
# binomial sampling instead of the exact cartesian product, so an unselective
# "any -> any" rule cannot materialize an O(N^2) index grid.
_CONNECTIONS_EXACT_PRODUCT_CAP = 4_000_000


def _connection_selector_mask(selector, area_labels, layer_labels, cell_labels, model_labels=None):
    """Boolean neuron mask for a ``.connections()`` source/target selector.

    Recognized keys: ``area`` (exact), ``layer``/``layers`` (tolerant of merged
    ``"L2/3"`` vs split ``"L2"``/``"L3"`` via :func:`_interarea_layer_set`),
    ``cell_type``/``cell_types``. A missing key means "any" (no constraint). When
    ``model_labels`` is supplied (the :func:`connect` ensemble path), a ``model``
    key additionally restricts the selection to one member model by integer index.
    """
    import numpy as _np

    n = len(cell_labels)
    mask = _np.ones(n, dtype=bool)
    area = selector.get("area")
    if area is not None:
        mask &= _np.asarray(area_labels) == str(area)
    layers = selector.get("layers", selector.get("layer"))
    if layers is not None:
        layers = [layers] if isinstance(layers, str) else list(layers)
        allowed: set[str] = set()
        for lyr in layers:
            allowed |= _interarea_layer_set(str(lyr))
        mask &= _np.isin(_np.asarray(layer_labels), list(allowed))
    cts = selector.get("cell_type", selector.get("cell_types"))
    if cts is not None:
        cts = [cts] if isinstance(cts, str) else list(cts)
        mask &= _np.isin(_np.asarray(cell_labels), [str(c) for c in cts])
    mdl = selector.get("model")
    if mdl is not None and model_labels is not None:
        mask &= _np.asarray(model_labels) == int(mdl)
    return mask


def _apply_edge_sign_policy(sign_str, magnitude, pre_idx, sign_intrinsic):
    """Resolve a connection rule's declared ``sign`` into signed edge weight(s).

    Shared by :func:`_compile_connection_rules` (sign-only fallback) and
    :func:`_compile_mechanism_aware_connection_rules` so the
    excitatory/inhibitory/signed policy is defined exactly once instead of
    twice. ``magnitude`` is the per-edge weight magnitude (sign discarded via
    ``abs``); ``pre_idx`` indexes into ``sign_intrinsic`` (each neuron's
    intrinsic +1/-1 sign) for the ``"signed"``/unset case.
    """
    import numpy as _np

    magnitude = _np.abs(_np.asarray(magnitude, dtype=_np.float64))
    if sign_str == "excitatory":
        return magnitude
    elif sign_str == "inhibitory":
        return -magnitude
    else:  # "signed" or unset -> presynaptic intrinsic sign
        return magnitude * _np.asarray(sign_intrinsic)[pre_idx]


def _compile_connection_rules(
    rules, area_labels, layer_labels, cell_labels, sign, n, jdtype, default_seed, model_labels=None
):
    """Compile declarative ``.connections()`` rules into a sparse EdgeList.

    Each rule selects a source and target neuron set (see
    :func:`_connection_selector_mask`) and adds edges source->target. ``probability``
    is Bernoulli inclusion over candidate pairs (1.0 = full, 0.0 = none; default 1.0),
    ``weight`` is the native edge magnitude (default ``1/sqrt(n)``), and ``sign`` maps
    ``"excitatory"``->positive, ``"inhibitory"``->negative, ``"signed"``/unset->the
    presynaptic neuron's intrinsic sign. Self-edges are dropped. Edge construction is
    sparse: the exact (pre x post) product is used for selective rules, falling back to
    binomial sampling past :data:`_CONNECTIONS_EXACT_PRODUCT_CAP` candidates so an
    unselective rule never builds an O(N^2) grid.

    Returns ``(edge_list_or_None, per_rule_edge_counts)``.

    This is the sign-only fallback compiler: receptor inferred from weight
    sign, tau hardcoded exc=2 ms/inh=5 ms, ignoring any declared ``mechanism``.
    ``construct()`` calls this only when the rule set does NOT fully resolve
    against declared ``.mechanisms()`` (see
    :func:`_all_connection_rules_declare_resolvable_mechanism`); when it does,
    :func:`_compile_mechanism_aware_connection_rules` runs instead, wrapping
    ``connectivity.compile_connection_rules`` → ``ConnectionCompileResult``
    for real per-edge tau from the declared mechanism.
    """
    import numpy as _np

    sign_np = _np.asarray(sign, dtype=_np.float64)
    sqrt_n = float(max(n, 1)) ** 0.5
    pre_all, post_all, w_all = [], [], []
    counts: list[int] = []
    for ri, rule in enumerate(rules):
        src = rule.get("source", {}) or {}
        tgt = rule.get("target", {}) or {}
        pre_idx = _np.where(_connection_selector_mask(src, area_labels, layer_labels, cell_labels, model_labels))[0]
        post_idx = _np.where(_connection_selector_mask(tgt, area_labels, layer_labels, cell_labels, model_labels))[0]
        p = rule.get("probability")
        p = 1.0 if p is None else float(p)
        if pre_idx.size == 0 or post_idx.size == 0 or p <= 0.0:
            counts.append(0)
            continue
        w_spec = rule.get("weight")
        w_mag = (1.0 / sqrt_n) if w_spec is None else float(w_spec)
        rseed = rule.get("seed")
        rseed = int(rseed) if rseed is not None else (int(default_seed) + 7919 * (ri + 1))
        rng = _np.random.default_rng(rseed)

        n_cand = int(pre_idx.size) * int(post_idx.size)
        if n_cand <= _CONNECTIONS_EXACT_PRODUCT_CAP:
            # Exact cartesian product -> exact p=1 (full) / p=0 (none) semantics.
            pre_grid, post_grid = _np.meshgrid(pre_idx, post_idx, indexing="ij")
            pre_flat = pre_grid.ravel(); post_flat = post_grid.ravel()
            non_self = pre_flat != post_flat
            pre_flat, post_flat = pre_flat[non_self], post_flat[non_self]
            keep = _np.ones(pre_flat.size, bool) if p >= 1.0 else (rng.random(pre_flat.size) < p)
            pre_g, post_g = pre_flat[keep], post_flat[keep]
        else:
            # Sparse fallback for huge unselective rules (statistical, not exact).
            import warnings as _warnings
            _warnings.warn(
                f"connection rule {rule.get('name')!r} has {n_cand} candidate pairs "
                f"(> {_CONNECTIONS_EXACT_PRODUCT_CAP}); using binomial sampling instead of "
                "the exact product (add area/layer/cell_type selectors to narrow it).",
                RuntimeWarning, stacklevel=2,
            )
            n_edges = int(rng.binomial(n_cand, min(p, 1.0)))
            pre_g = pre_idx[rng.integers(0, pre_idx.size, n_edges)]
            post_g = post_idx[rng.integers(0, post_idx.size, n_edges)]
            non_self = pre_g != post_g
            pre_g, post_g = pre_g[non_self], post_g[non_self]

        if pre_g.size == 0:
            counts.append(0)
            continue
        sgn = rule.get("sign")
        wv = _apply_edge_sign_policy(sgn, _np.full(pre_g.size, w_mag), pre_g, sign_np)
        pre_all.append(pre_g); post_all.append(post_g); w_all.append(wv)
        counts.append(int(pre_g.size))

    if not pre_all:
        return None, counts
    pre = _np.concatenate(pre_all); post = _np.concatenate(post_all); w = _np.concatenate(w_all)
    receptor = (w < 0).astype(_np.int32)
    tau = _np.where(receptor == 0, 2.0, 5.0)
    edges = EdgeList(
        pre=jnp.asarray(pre, jnp.int32),
        post=jnp.asarray(post, jnp.int32),
        weight=jnp.asarray(w, jdtype),
        receptor_index=jnp.asarray(receptor, jnp.int32),
        tau_ms=jnp.asarray(tau, jdtype),
    )
    return edges, counts


def _all_connection_rules_declare_resolvable_mechanism(rules, mechanisms):
    """Gate for the mechanism-aware connection-rule compiler.

    True only when every rule has a non-None ``mechanism`` that resolves
    against a declared name in ``mechanisms`` -- the same requirement
    :func:`connectivity.compile_connection_rules`/``_resolve_mechanism``
    enforces (it raises on an unresolvable/missing mechanism). Mixed or
    partially-declared rule sets fall back to the sign-only compiler so this
    switch never silently changes simulated dynamics for a model that did
    not fully opt in via ``.mechanisms()`` + ``.connections(mechanism=...)``.
    """
    if not rules or not mechanisms:
        return False
    known = {m.get("name") for m in mechanisms}
    return all(rule.get("mechanism") in known and rule.get("mechanism") is not None for rule in rules)


def _compile_mechanism_aware_connection_rules(
    rules, mechanisms, area_labels, layer_labels, cell_labels, sign, n, jdtype, default_seed,
    positions=None,
):
    """Mechanism-aware connection-rule compiler (Synaptic Tensor switch, gated path).

    Selected by :func:`construct` instead of :func:`_compile_connection_rules`
    only when :func:`_all_connection_rules_declare_resolvable_mechanism` is
    True. Wraps :func:`connectivity.compile_connection_rules` +
    :meth:`ConnectionCompileResult.to_edge_list` for mechanism-correct
    ``receptor_index``/``tau_ms``, then applies the SAME sign semantics as
    :func:`_compile_connection_rules` (excitatory/inhibitory/signed) on top --
    ``compile_connection_rules`` itself stores a rule's declared ``sign`` as
    metadata only and never applies it to ``edge_weight`` (verified: an
    inhibitory-mechanism rule otherwise keeps a positive weight), so without
    this post-correction the switch would silently break inhibition.

    Validated for exact parity with :func:`_compile_connection_rules` when a
    declared mechanism's tau mirrors the legacy hardcoded default (AMPA-like
    exc=2ms, GABA_A-like inh=5ms) at ``probability=1.0`` -- see
    ``tests/test_mechanism_aware_connection_compiler.py``.
    """
    from .connectivity import compile_connection_rules

    import numpy as _np

    if positions is not None:
        neuron_rows = [
            {"neuron_id": i, "area": area_labels[i], "layer": layer_labels[i], "cell_type": cell_labels[i],
             "x": float(positions[i, 0]), "y": float(positions[i, 1]), "z": float(positions[i, 2])}
            for i in range(n)
        ]
    else:
        neuron_rows = [
            {"neuron_id": i, "area": area_labels[i], "layer": layer_labels[i], "cell_type": cell_labels[i]}
            for i in range(n)
        ]
    dtype_name = "float64" if jdtype == jnp.float64 else "float32"
    result = compile_connection_rules(
        neuron_rows, rules, mechanisms, seed=int(default_seed), allow_empty=True, dtype=dtype_name
    )
    counts = [int(row.get("n_edges", 0)) for row in result.connection_table]
    if result.n_edges == 0:
        return None, counts

    edges = result.to_edge_list(dtype=dtype_name)
    edge_rule_id = _np.asarray(result.edge_rule_id)
    pre_np = _np.asarray(edges.pre)
    raw_weight = _np.asarray(edges.weight, dtype=_np.float64)
    out_weight = raw_weight.copy()
    sign_np = _np.asarray(sign, dtype=_np.float64)
    for ri, rule in enumerate(rules):
        mask = edge_rule_id == ri
        if not _np.any(mask):
            continue
        sgn = rule.get("sign")
        out_weight[mask] = _apply_edge_sign_policy(sgn, raw_weight[mask], pre_np[mask], sign_np)
    edges = replace(edges, weight=jnp.asarray(out_weight, dtype=jdtype))
    return edges, counts


def _concat_edge_lists(a: "EdgeList", b: "EdgeList") -> "EdgeList":
    """Concatenate two EdgeLists (preserving the first's calibration status)."""
    return EdgeList(
        pre=jnp.concatenate([a.pre, b.pre]),
        post=jnp.concatenate([a.post, b.post]),
        weight=jnp.concatenate([a.weight, b.weight]),
        receptor_index=jnp.concatenate([a.receptor_index, b.receptor_index]),
        tau_ms=jnp.concatenate([a.tau_ms, b.tau_ms.astype(a.tau_ms.dtype)]),
        source_calibration_status=a.source_calibration_status,
    )


def _mark_connections_compiled(cfg: "Configuration", counts: Sequence[int]) -> "Configuration":
    """Flip each connection rule's status from declared to compiled.

    Honestly lifts the ``declared_not_compiled`` fence: a rule that produced edges
    becomes ``"compiled"`` (with ``compiled_n_edges``); a rule that matched no
    neurons/edges becomes ``"compiled_no_matching_edges"`` so the no-op is visible.
    """
    metadata = {**cfg.metadata}
    circuit = {**metadata.get("circuit", {})}
    conns = list(circuit.get("connections", []))
    updated = []
    for i, rule in enumerate(conns):
        c = int(counts[i]) if i < len(counts) else 0
        updated.append({
            **rule,
            "status": "compiled" if c > 0 else "compiled_no_matching_edges",
            "compiled_n_edges": c,
        })
    circuit["connections"] = updated
    metadata["circuit"] = circuit
    return replace(cfg, metadata=metadata)


# _default_operator_status, _circuit_json_safe, _default_metadata,
# _counts_from_fractions, _reject_retired_like, _ProbeDeclarations,
# Configuration, Config moved to jaxfne/_config.py and re-exported above.

# RuntimeConfig, SurrogateConfig, _resolve_backend_device, _device_scope,
# _jaxlib_version moved to jaxfne/_runtime_config.py and re-exported above.

# Simulation, Probe, Signals, Objective, DatasetSpec, TrialSpec, TrialBatch,
# TrialResult, TrialBatchResult, ReadoutSpec, ReadoutResult, ObjectiveReport,
# RunReceipt, StimulusSchedule, LaminarPopulation, LaminarSourceGeometry, the
# metric/gate-evaluation helpers, AxisSpec, BasisSpec moved to
# jaxfne/_signals.py and re-exported above.


def configuration() -> Configuration:
    """Return a fresh, empty :class:`Configuration` builder.

    Entry point for the chainable configuration grammar
    (``configuration().network(...).emitter(...)...``). Equivalent to calling
    ``Configuration()`` directly.
    """
    return Configuration()


def runtime(
    backend: str = "auto",
    dtype: str = "float32",
    jit: bool = False,
    vmap: bool = False,
    precision: str = "default",
    seed: int = 0,
    n_steps: int = 0,
    recurrent_backend: str = "dense",
    synaptic_kernel: str = "exponential",
    # v0.0.3 compatibility names.
    device_type: str | None = None,
    dtype_primary: str | None = None,
    x64_enabled: bool | None = None,
) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` (JAX backend/dtype/jit/vmap policy).

    Thin keyword factory for :class:`RuntimeConfig`; ``device_type``,
    ``dtype_primary`` and ``x64_enabled`` are accepted as v0.0.3-compatibility
    aliases. This configures execution only — it makes no numerical or
    scientific claim about the simulation.
    """
    return RuntimeConfig(
        backend=backend,
        dtype=dtype,
        jit=jit,
        vmap=vmap,
        precision=precision,
        seed=seed,
        n_steps=n_steps,
        recurrent_backend=recurrent_backend,
        synaptic_kernel=synaptic_kernel,
        device_type=device_type,
        dtype_primary=dtype_primary,
        x64_enabled=x64_enabled,
    )


def runtime_report(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Return a JSON-safe dict describing the resolved JAX runtime.

    Reports the effective backend, dtype policy, and jit/vmap flags for the
    given :class:`RuntimeConfig` (or the default if ``None``). Use it to confirm
    ``actual_dtype`` and that jit/vmap are enabled before profiling.
    """
    return (runtime_config or RuntimeConfig()).runtime_report()


def simulation(**kwargs: Any) -> Simulation:
    """Build a :class:`Simulation` spec from keyword arguments.

    Thin factory for :class:`Simulation` (``duration_ms``, ``dt_ms``, ``seed``,
    ``plasticity``, recording flags, ...). Equivalent to ``Simulation(**kwargs)``.
    """
    return Simulation(**kwargs)


def objective() -> Objective:
    """Return a fresh, empty :class:`Objective` specification.

    Starting point for declaring losses, regularizers, and diagnostic gates.
    Equivalent to ``Objective()``.
    """
    return Objective()


def rate_targets(
    groups: dict[str, Any],
    targets_hz: dict[str, float],
    weights: Optional[dict[str, float]] = None,
) -> Objective:
    """Create a multi-group firing-rate objective.

    This factory creates an Objective with kind="group_rate_targets" that
    encodes group-wise firing-rate targets. When passed to Model.tune(),
    the optimization loop computes group-wise rates and minimizes
    squared-relative-error loss.

    Parameters
    ----------
    groups : dict[str, Any]
        Mapping from group names to neuron indices.
        E.g., {"first_half": np.arange(0, 24), "second_half": np.arange(24, 48)}.
    targets_hz : dict[str, float]
        Mapping from group names to target firing rates in Hz.
        E.g., {"first_half": 5.0, "second_half": 10.0}.
    weights : Optional[dict[str, float]]
        Mapping from group names to loss weights (default: 1.0 each).

    Returns
    -------
    Objective
        Objective with kind="group_rate_targets", storing groups and targets
        in metadata for use by optimization loops.

    Example
    -------
    >>> import numpy as np
    >>> import jaxfne as jtfne
    >>> objectives = jtfne.rate_targets(
    ...     groups={"first": np.arange(0, 24), "second": np.arange(24, 48)},
    ...     targets_hz={"first": 5.0, "second": 10.0},
    ... )
    >>> optimizer = jtfne.agsdr(parameters={"drive_scale_a": (0.3, 2.0)}, generations=8)
    >>> result = model.tune(objectives=objectives, optimizer=optimizer)
    >>> result.best_score
    """
    import numpy as np

    # Validate
    if not groups or not targets_hz:
        raise ValueError("groups and targets_hz must be non-empty")
    if set(groups.keys()) != set(targets_hz.keys()):
        raise ValueError("Group names must match between groups and targets_hz")

    if weights is None:
        weights = {name: 1.0 for name in groups.keys()}

    # Convert to JSON-safe lists
    groups_lists = {}
    for name, indices in groups.items():
        arr = np.asarray(indices, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Group '{name}' indices must be 1D")
        groups_lists[name] = arr.tolist()

    # Create objective with group metadata
    return Objective(
        name="rate_targets",
        kind="group_rate_targets",
        losses=[],
        regularizers=[],
        gates=[],
    ).gate(
        name="rate_targets_metadata",
        threshold=0,  # Threshold unused for optimizer-computed score
        criterion="below",
        metadata={
            "groups": groups_lists,
            "targets_hz": {k: float(v) for k, v in targets_hz.items()},
            "weights": {k: float(weights.get(k, 1.0)) for k in groups.keys()},
        },
    )


def suite2_celltype_presets() -> dict[str, dict[str, float | str]]:
    """Return compact E/PV/SST/VIP reduced-emitter preset metadata."""
    return {
        "E": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0, "drive": 5.0, "role": "excitatory_pyramidal_like"},
        "PV": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0, "drive": 3.0, "role": "fast_inhibitory_like"},
        "SST": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0, "drive": 3.5, "role": "somatostatin_like"},
        "VIP": {"a": 0.02, "b": -0.10, "c": -55.0, "d": 6.0, "drive": 3.0, "role": "disinhibitory_like"},
    }


def suite2_single_neuron_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, cell_type: str = "E") -> Configuration:
    """Build the Suite No. 2 one-emitter configuration."""
    fractions = {"E": 0.0, "PV": 0.0, "SST": 0.0, "VIP": 0.0}
    if cell_type not in fractions:
        raise ValueError(f"cell_type must be one of {tuple(fractions)}")
    fractions[cell_type] = 1.0
    return (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("single", layers=["uniform_3d"], n=1)
        .cell_types(fractions)
        .uniform3d(radius_mm=0.010, height_mm=0.010)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=4)
    )


def suite2_four_celltype_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> Configuration:
    """Build the Suite No. 2 four-emitter E/PV/SST/VIP configuration."""
    return (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("celltype_panel", layers=["uniform_3d"], n=4)
        .cell_types({"E": 0.25, "PV": 0.25, "SST": 0.25, "VIP": 0.25})
        .uniform3d(radius_mm=0.030, height_mm=0.10)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=4)
    )


def suite2_net1_config(
    *,
    seed: int = 7,
    n: int = 100,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    drives: Mapping[str, float] | None = None,
) -> Configuration:
    """Build net1: a uniformly sampled 3D E/PV/SST/VIP column."""
    cfg = (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("net1", layers=["uniform_3d"], n=int(n))
        .cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .uniform3d(radius_mm=0.25, height_mm=1.60)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45)
        .probes(_SUITE2_PROXY_MODES, n_contacts=16)
    )
    cfg = cfg.cell_type_drives(drives or {"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
    return cfg


def suite2_v1_v4_config(
    *,
    seed: int = 7,
    n_per_area: int = 400,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    v1_layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
    v4_layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
) -> Configuration:
    """Build the Suite No. 2 V1-V4 laminar scaffold with six layers per area."""
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]
    cfg = (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("V1", layers=layers, n=int(n_per_area))
        .column("V4", layers=layers, n=int(n_per_area))
        .cell_types({"E": 0.50, "PV": 0.20, "SST": 0.15, "VIP": 0.15})
        .layer_fractions(_SUITE2_LAYER_FRACTIONS, _suite2_default_layer_cell_types())
        .area_layer_cell_types("V1", v1_layer_cell_types or _SUITE2_LAYER_CELL_TYPES_V1)
        .area_layer_cell_types("V4", v4_layer_cell_types or _SUITE2_LAYER_CELL_TYPES_V4)
        .connectivity(
            within_area="all_to_all_uniform_random",
            within_gain=0.35,
            feedforward="V1_L2L3_E_to_V4_L4_E",
            feedback="V4_L2L3_E_to_V1_L5L6_L1_E",
            feedforward_gain=0.65,
            feedback_gain=0.50,
        )
        .suite2_interarea(True)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=24)
    )
    return cfg


def suite2_simulation(
    *,
    seed: int = 7,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    noise_amplitude: float | None = None,
    noise_rate_hz: float = 2.0,
    target: str = "all",
    jit: bool = False,
    recurrent_backend: str = "dense",
) -> Simulation:
    """Create a Suite No. 2 simulation with deterministic runtime metadata."""
    runtime_cfg = RuntimeConfig(dtype="float32", seed=seed, jit=jit, recurrent_backend=recurrent_backend)
    poisson = None
    if noise_amplitude is not None:
        poisson = {"rate_hz": float(noise_rate_hz), "amplitude": float(noise_amplitude), "target": target, "seed": int(seed) + 7919}
    return Simulation(
        duration_ms=float(duration_ms),
        dt_ms=float(dt_ms),
        seed=int(seed),
        record_sources=True,
        record_fields=True,
        runtime=runtime_cfg,
        poisson_drive=poisson,
    )


def suite2_tune_noise_agsdr_adam(
    model: Model,
    *,
    simulation: Simulation | None = None,
    target_rate_hz: tuple[float, float] = (5.0, 10.0),
    amplitudes: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    poisson_rate_hz: float = 2.0,
    adam_steps: int = 2,
    learning_rate: float = 0.20,
    finite_difference_eps: float = 0.05,
    seed: int = 7,
) -> TuneResult:
    """Tune Poisson-drive amplitude toward a target mean firing-rate range.

    Multi-stage approach: outer stage evaluates an AGSDR-style candidate population
    (black-box sweep over noise amplitudes). Inner stage applies finite-difference
    Adam updates for local refinement. Current implementation: finite-difference Adam
    with black-box candidate sweep. Variance-balanced adaptive self-supervision is
    reserved for a future optimizer path with adaptive alpha/learning-rate schedules.
    This path tunes a solver-tuned drive parameter and preserves relative
    proxy-unit metadata.
    """
    sim0 = simulation or suite2_simulation(seed=seed, duration_ms=1000.0, dt_ms=0.1)
    lo, hi = float(target_rate_hz[0]), float(target_rate_hz[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError("target_rate_hz must be a positive (low, high) tuple")
    target_mid = 0.5 * (lo + hi)

    def run_amp(amp: float, run_seed: int) -> tuple[float, float, Signals]:
        """Documented public function `run_amp`."""
        sim = replace(
            sim0,
            seed=int(run_seed),
            poisson_drive={"rate_hz": float(poisson_rate_hz), "amplitude": float(max(0.0, amp)), "target": "all", "seed": int(run_seed) + 7919},
        )
        sig = model.simulate(sim)
        rate = float(sig.summary()["spike_rate_hz_mean"] or 0.0)
        if lo <= rate <= hi:
            loss = 0.0
        else:
            loss = (rate - target_mid) * (rate - target_mid)
        return loss, rate, sig

    history: list[dict[str, Any]] = []
    best_amp = float(amplitudes[0])
    best_loss, best_rate, best_sig = run_amp(best_amp, seed)
    history.append({"stage": "agsdr_population", "amplitude": best_amp, "rate_hz": best_rate, "loss": best_loss})
    for idx, amp in enumerate(amplitudes[1:], start=1):
        loss, rate, sig = run_amp(float(amp), seed + idx)
        history.append({"stage": "agsdr_population", "amplitude": float(amp), "rate_hz": rate, "loss": loss})
        if loss < best_loss:
            best_amp, best_loss, best_rate, best_sig = float(amp), loss, rate, sig

    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    amp = best_amp
    for step in range(1, int(adam_steps) + 1):
        loss0, rate0, _ = run_amp(amp, seed + 100 + step * 2)
        loss1, _, _ = run_amp(amp + finite_difference_eps, seed + 101 + step * 2)
        grad = (loss1 - loss0) / max(float(finite_difference_eps), 1e-6)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        amp = max(0.0, amp - float(learning_rate) * m_hat / (math.sqrt(v_hat) + eps))
        loss_new, rate_new, sig_new = run_amp(amp, seed + 200 + step)
        history.append({"stage": "finite_difference_adam", "step": step, "amplitude": amp, "rate_hz": rate_new, "loss": loss_new, "gradient_estimate": grad})
        if loss_new < best_loss:
            best_amp, best_loss, best_rate, best_sig = amp, loss_new, rate_new, sig_new

    summary = json_safe({
        "optimizer": "AGSDR_outer_finite_difference_Adam_inner",
        "target_rate_hz": [lo, hi],
        "best_noise_amplitude": best_amp,
        "best_rate_hz": best_rate,
        "best_loss": best_loss,
        "history_length": len(history),
        "tuned_parameter": "simulation.poisson_drive.amplitude",
        "units_or_status": "reduced_native_drive_units_relative_proxy",
        "field_solver_status": model.cfg.metadata.get("field_solver_status", "linear_solver"),
        "physical_amplitude_calibrated": False,
    })
    return TuneResult(
        best_parameters={"noise_amplitude": float(best_amp), "poisson_rate_hz": float(poisson_rate_hz)},
        best_score=float(best_loss),
        history=json_safe(history),
        summary=summary,
        model=model,
    )


def suite2_run_bundle(model: Model, *, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, noise_amplitude: float | None = None) -> dict[str, Any]:
    """Run simulation, readouts, manifest, and receipt for Suite No. 2 notebooks."""
    sim = suite2_simulation(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, noise_amplitude=noise_amplitude)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=list(_SUITE2_PROXY_MODES))
    manifest = model.manifest(signals=signals, readout=readout)
    return {"simulation": sim, "signals": signals, "readout": readout, "manifest": manifest, "neuron_table": model.neuron_table()}


def _runtime_config_from_metadata(metadata: Mapping[str, Any]) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from ``Configuration.runtime(...)`` metadata.

    ``Configuration.runtime(**kwargs)`` stores its keys in ``cfg.metadata`` rather
    than compiling a runtime object. This maps the runtime-relevant keys back onto
    a ``RuntimeConfig`` so that ``.runtime(dtype=..., recurrent_backend=..., jit=...)``
    set on a Configuration is actually honored by the high-level ``simulate`` path.
    Only non-None known keys are applied, so a config without explicit runtime
    knobs yields the default RuntimeConfig (float32, dense, eager) — unchanged
    behavior. ``dtype="float64"`` still requires JAX x64 enabled to take effect
    (see ``RuntimeConfig.actual_dtype``).
    """
    kw: dict[str, Any] = {}
    for k in ("dtype", "recurrent_backend", "jit", "vmap", "backend",
              "synaptic_kernel", "precision", "device_type",
              "enable_homeostasis", "homeostasis_params",
              "enable_hdp", "hdp_params"):
        v = metadata.get(k)
        if v is not None:
            kw[k] = v
    return RuntimeConfig(**kw)


def simulate(
    model: Model,
    sim: Optional[Simulation] = None,
    paradigm: Optional[Any] = None,
    **kwargs: Any,
) -> Signals:
    """Run a simulation with the given model.

    Allows passing either an explicit :class:`Simulation` object, or passing
    simulation parameters (such as ``duration_ms``, ``dt_ms``, ``seed``,
    ``record_sources``, ``record_fields``, ``runtime``, ``dtype``) as direct
    keyword arguments.

    When no explicit ``runtime``/``Simulation`` is given, the runtime declared on
    the model's :class:`Configuration` via ``.runtime(...)`` (``dtype``,
    ``recurrent_backend``, ``jit``, ``vmap``, ``backend``, ``synaptic_kernel``) is
    inherited — so ``cfg.runtime(dtype="float64")`` / ``recurrent_backend="edge_list"``
    actually take effect. A ``dtype=`` keyword overrides the inherited dtype.
    """
    if sim is None:
        if "runtime" not in kwargs:
            override_dtype = kwargs.pop("dtype", None)
            cfg_meta = getattr(getattr(model, "cfg", None), "metadata", None) or {}
            runtime_cfg = _runtime_config_from_metadata(cfg_meta)
            if override_dtype is not None:
                runtime_cfg = replace(runtime_cfg, dtype=str(override_dtype))
            kwargs["runtime"] = runtime_cfg
        elif "dtype" in kwargs:
            raise ValueError(
                "Specify dtype via runtime=RuntimeConfig(dtype=...), not both "
                "runtime= and dtype=."
            )
        sim = Simulation(**kwargs)
    elif kwargs:
        raise ValueError(
            "Cannot specify both a Simulation object and individual simulation parameters as keyword arguments."
        )
    return model.simulate(sim, paradigm=paradigm)


def compute_fields(model: "Model", signals: "Signals") -> "FieldOutput":
    """Canonical field-stage entry point: ``fields = jtfne.compute_fields(model, signals)``.

    This is a thin accessor, not a new computation -- ``simulate()`` already
    builds ``signals.field`` internally (via :func:`project_laminar_sources`)
    whenever field-capable probe modes (``"source"``, ``"CSD"``, ``"LFP"``)
    were declared on the model's :class:`Configuration`. ``compute_fields``
    validates presence and returns that existing :class:`FieldOutput` rather
    than fabricating one; it raises if no field was computed, instead of
    silently returning ``None`` or synthesizing a placeholder.

    ``model`` is accepted (not just ``signals``) to match the canonical
    pipeline signature; the current thin-accessor implementation does not
    read it.
    """
    if signals.field is None:
        raise ValueError(
            "signals.field is None -- no field-capable probe modes (e.g. "
            "'source', 'CSD', 'LFP') were declared before simulate(). "
            "compute_fields() is a thin accessor over the field already "
            "computed inside simulate(); it does not synthesize a new one."
        )
    return signals.field


def _apply_canonical_biophysics(emitter, positions, edge_list, cfg):
    """Apply canonical cortical-column biophysics at construct time.

    Three effects, all reproducible from ``cfg`` seed:

    * **Random ``v0``** ~ Uniform(-70, 0) mV per neuron — ALWAYS applied. Identical
      initial potentials cause a synchronized t=0 onset volley (raster banding);
      randomizing removes that onset-response synchrony bias. Disable with
      ``cfg.runtime(random_v0=False)``.
    * **Deep-E "larger" grading** (laminar columns only): excitatory neurons get a
      depth gradient — ``source_scale`` 1.0->1.8 (bigger dipole), ``a`` 0.020->0.015
      (slower recovery), ``d`` 8->10 (wider/stronger). Larger deep pyramidals fire
      sparser, slower, wider.
    * **PV<->E local connectivity x3** (laminar columns only): the fast feedback-
      inhibition (PING) loop is strengthened, distance-gated by laminar depth.

    The two laminar effects are gated on the presence of cell-type + layer labels
    and a non-degenerate depth axis, so non-laminar/test configs only get random v0.
    Returns ``(emitter, edge_list)``. Proxy/scaffold truth status is unchanged.
    """
    import numpy as _np
    jdtype = emitter.v0.dtype
    n = int(emitter.v0.shape[0])
    seed = int(cfg.metadata.get("seed", 0) or 0)

    if cfg.metadata.get("random_v0", True):
        vkey = jax.random.PRNGKey((seed * 1_000_003 + 11) % (2**31 - 1))
        v0 = jax.random.uniform(vkey, (n,), dtype=jdtype, minval=-70.0, maxval=0.0)
        emitter = replace(emitter, v0=v0)

    # Deep-E grading + PV<->E strengthening are canonical-column features: they only
    # apply when the canonical biophysics profile is requested (build_laminar_column
    # ei_profile="canonical"), so generic/test configs and explicit user cell_params
    # are left untouched. Random v0 above is always applied.
    if not cfg.metadata.get("canonical_biophysics", False):
        return emitter, edge_list

    cts = emitter.labels
    if cts is None or emitter.layer_labels is None:
        return emitter, edge_list
    lab = _np.asarray([str(x) for x in cts])
    Z = _np.asarray(positions)[:, 2]
    if Z.shape[0] != n or float(_np.ptp(Z)) == 0.0:
        return emitter, edge_list
    zd = (Z - Z.min()) / (float(_np.ptp(Z)) + 1e-9)
    isE = lab == "E"

    # Deep-E "larger" grading (E only, by depth).
    if isE.any():
        a = _np.asarray(emitter.a, dtype=float).copy()
        d = _np.asarray(emitter.d, dtype=float).copy()
        ss = _np.asarray(emitter.source_scale)
        ss = (_np.full(n, float(ss)) if ss.ndim == 0 else _np.asarray(ss, dtype=float)).copy()
        a[isE] = 0.020 - 0.005 * zd[isE]
        d[isE] = 8.0 + 2.0 * zd[isE]
        ss[isE] = 1.0 + 0.8 * zd[isE]
        emitter = replace(emitter, a=jnp.asarray(a, dtype=jdtype),
                          d=jnp.asarray(d, dtype=jdtype),
                          source_scale=jnp.asarray(ss, dtype=jdtype))

    # PV<->E local connectivity x3 (distance-gated by laminar depth).
    pre = _np.asarray(edge_list.pre); post = _np.asarray(edge_list.post)
    pc = lab[pre]; qc = lab[post]
    pv_e = ((pc == "PV") & (qc == "E")) | ((pc == "E") & (qc == "PV"))
    if pv_e.any():
        w = _np.asarray(edge_list.weight, dtype=float).copy()
        gate = _np.exp(-((_np.abs(Z[pre] - Z[post]) / 0.15) ** 2))
        w[pv_e] = w[pv_e] * (1.0 + 2.0 * gate[pv_e])
        edge_list = replace(edge_list, weight=jnp.asarray(w, dtype=jdtype))

    return emitter, edge_list


def _np_isscalar_param(v: Any) -> bool:
    """True if a homeostasis param is a JSON-safe scalar (not a per-neuron array)."""
    return isinstance(v, (int, float, bool, str)) or v is None


def _simulate_homeostasis_metadata(
    runtime_cfg: "RuntimeConfig", diag: "dict[str, Any] | None"
) -> dict[str, Any]:
    """``Model.simulate()`` helper: the homeostasis metadata sub-block.

    Called only when ``runtime_cfg.enable_homeostasis`` is True. Framing: a
    minimal homeostatic resource/adaptation controller (intrinsic
    excitability bias + optional homeostatic synaptic scaling) -- a
    COMPUTATIONAL method, NOT a biological mechanism. Mechanism support
    requires nulls/ablations/repeated seeds and empirical comparison;
    objective success alone does not imply it.
    """
    _hp_meta = dict(runtime_cfg.homeostasis_params or {})
    _plastic_on = float(_hp_meta.get("eta", 0.0) or 0.0) != 0.0
    homeo_meta: dict[str, Any] = {
        "enabled": True,
        "params": {
            k: (v if _np_isscalar_param(v) else "per_neuron_array")
            for k, v in _hp_meta.items()
        },
        "method": "minimal_homeostatic_resource_adaptation_controller",
        "claim_status": "computational_control_proxy_not_biological_mechanism",
        "synaptic_plasticity_enabled": bool(_plastic_on),
        "biological_learning_claim": False,
        "mechanism_claim_status": "not_claimed",
        "diagnostics_passthrough": "Signals.metadata['homeostasis'] summary; "
                                   "full per-step g_bias/r_trace (and, when "
                                   "synaptic_plasticity_enabled, w_final/w_trace) via "
                                   "Model.last_homeostasis_diagnostics()",
    }
    if diag is not None:
        g = diag["g_bias"]; r = diag["r_trace"]
        homeo_meta["g_bias_summary"] = {
            "min": float(jnp.min(g)), "max": float(jnp.max(g)),
            "mean": float(jnp.mean(g)), "shape": list(g.shape),
        }
        homeo_meta["r_trace_summary"] = {
            "min": float(jnp.min(r)), "max": float(jnp.max(r)),
            "mean": float(jnp.mean(r)), "shape": list(r.shape),
        }
        if "w_final" in diag:
            wf = diag["w_final"]
            homeo_meta["w_final_summary"] = {
                "min": float(jnp.min(wf)), "max": float(jnp.max(wf)),
                "mean": float(jnp.mean(wf)), "shape": list(wf.shape),
            }
    return homeo_meta


def _simulate_hdp_metadata(
    runtime_cfg: "RuntimeConfig", diag: "dict[str, Any] | None"
) -> dict[str, Any]:
    """``Model.simulate()`` helper: the HDP metadata sub-block.

    Called only when ``runtime_cfg.enable_hdp`` is True. Framing: a single
    per-neuron master-state (H_i) plasticity controller -- a COMPUTATIONAL
    method, NOT a biological mechanism, matching the homeostasis controller's
    claim discipline.
    """
    _hp_meta = dict(runtime_cfg.hdp_params or {})
    hdp_meta: dict[str, Any] = {
        "enabled": True,
        "params": {
            k: (v if _np_isscalar_param(v) else "per_neuron_array")
            for k, v in _hp_meta.items()
        },
        "method": "homeostasis_dependent_plasticity_master_state_controller",
        "claim_status": "computational_control_proxy_not_biological_mechanism",
        "biological_learning_claim": False,
        "mechanism_claim_status": "not_claimed",
        "diagnostics_passthrough": "Signals.metadata['hdp'] summary; full "
                                    "per-step H_trace/w_trace via "
                                    "Model.last_hdp_diagnostics()",
    }
    if diag is not None:
        H = diag["H_trace"]; wf = diag["w_final"]
        hdp_meta["H_trace_summary"] = {
            "min": float(jnp.min(H)), "max": float(jnp.max(H)),
            "mean": float(jnp.mean(H)), "std": float(jnp.std(H)),
            "shape": list(H.shape),
        }
        hdp_meta["w_final_summary"] = {
            "min": float(jnp.min(wf)), "max": float(jnp.max(wf)),
            "mean": float(jnp.mean(wf)), "shape": list(wf.shape),
        }
    return hdp_meta


def _resolve_homeostasis_k_gain(hp: Mapping[str, Any], emitter) -> Any:
    """Resolve the homeostatic ``k_gain`` to a scalar or per-neuron array.

    If ``homeostasis_params["k_gain_size_scaled"]`` is true, return a per-neuron
    array ``k_gain_base / size`` where size is the emitter ``source_scale`` (so
    larger/deep-E neurons are taxed less aggressively, per the canonical
    size-aware homeostasis). Otherwise return the scalar base ``k_gain``.
    A directly-supplied array ``k_gain`` is passed through unchanged.
    """
    base = hp.get("k_gain", 1.0)
    if not hp.get("k_gain_size_scaled", False):
        return base
    import numpy as _np
    n = int(emitter.v0.shape[0])
    ss = _np.asarray(emitter.source_scale)
    ss = _np.full(n, float(ss)) if ss.ndim == 0 else _np.asarray(ss, dtype=float)
    return jnp.asarray(float(base) / _np.clip(ss, 1e-3, None),
                       dtype=emitter.v0.dtype)


def _homeostasis_params_cache_fingerprint(hp: Mapping[str, Any]) -> tuple:
    """Build a hashable fingerprint of ``homeostasis_params`` for a JIT
    cache key, so a reused ``Model`` that varies any homeostasis parameter
    (``r_star``, ``k_gain``, ``eta``, ...) at identical shapes never replays a
    stale compiled closure built for different parameter values. Scalars pass
    through as-is; array-valued params (e.g. a directly-supplied per-neuron
    ``k_gain``) are reduced to (shape, dtype, content-hash) since arrays
    aren't hashable.
    """
    import numpy as _np
    items: list[tuple] = []
    for k in sorted(hp.keys()):
        v = hp[k]
        if isinstance(v, (bool, int, float, str)) or v is None:
            items.append((k, v))
        elif isinstance(v, Mapping):
            items.append((k, tuple(sorted(v.items()))))
        else:
            arr = _np.asarray(v)
            items.append((k, "array", arr.shape, str(arr.dtype), hash(arr.tobytes())))
    return tuple(items)


_SUPPORTED_EMITTER_FAMILIES = frozenset({"izhikevich"})


def _construct_validate_config(cfg: "Configuration") -> None:
    """``construct()`` stage: validate config + fail loudly on unsupported emitter family."""
    validation = cfg.validate()
    if not validation["valid"]:
        raise ValueError(f"Invalid jaxfne configuration: {validation['issues']}")
    # Fail loudly on explicitly declared unsupported emitter families. The
    # construct/simulate path only implements the Izhikevich kernel; any other
    # explicitly named family (e.g. "lif", "glif") must raise rather than
    # silently running Izhikevich. A config that omits "family" (or sets it to
    # None) uses the documented default Izhikevich kernel by design.
    for _emitter in cfg.emitters:
        _family = _emitter.get("family")
        if _family is not None and _family not in _SUPPORTED_EMITTER_FAMILIES:
            _supported = ", ".join(sorted(_SUPPORTED_EMITTER_FAMILIES))
            raise ValueError(
                f"Unsupported emitter family {_family!r} for construct/simulate. "
                f"Supported families: {_supported}. "
                "An explicitly declared unsupported family does not silently "
                "fall back to another emitter."
            )


def _construct_build_network(
    cfg: "Configuration", net: Mapping[str, Any], dtype_name_cfg: str
) -> "tuple[EIGNetwork, jax.Array, dict[str, Any] | None, int, EdgeList | None]":
    """``construct()`` stage: build the EIGNetwork (suite2-style or plain) ->
    ``(network, positions, geometry_meta, n, prebuilt_edges)``."""
    n = int(net.get("n", 100))
    _prebuilt_edges = None
    if cfg.metadata.get("columns") or cfg.metadata.get("layer_cell_types") or cfg.metadata.get("uniform_3d"):
        params, positions, geometry_meta, _prebuilt_edges = _suite2_neuron_population_from_config(cfg, dtype=dtype_name_cfg)
        network = EIGNetwork(
            params=params,
            positions=positions,
            metadata={
                "emitter_family": "izhikevich",
                "source_calibration_status": params.source_calibration_status,
                "position_units": geometry_meta.get("position_units", "mm_declared_metadata"),
            },
        )
        n = int(params.n_neurons)
    else:
        cell_types = net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.1})
        network = make_eig_network(n=n, cell_type_fractions=cell_types)
        positions = network.positions
        geometry_meta = None
    return network, positions, geometry_meta, n, _prebuilt_edges


def _construct_resolve_edge_list(
    cfg: "Configuration", network: "EIGNetwork", prebuilt_edges: "EdgeList | None"
) -> "tuple[Configuration, EdgeList]":
    """``construct()`` stage: prebuilt sparse edges OR dense-W conversion ->
    ``(cfg, edge_list)``. Avoids materializing/scanning the (n, n) dense matrix
    on the sparse-direct path; otherwise converts the dense ``network.params.W``.
    """
    if prebuilt_edges is not None:
        edge_list = prebuilt_edges
        # The dense W is a placeholder; this model must run on the edge_list backend.
        cfg = cfg.runtime(recurrent_backend="edge_list")
    else:
        edge_list = make_edge_list_from_dense(network.params.W, dtype=network.params.v0.dtype.name)
    return cfg, edge_list


def _construct_apply_geometry_override(
    geometry: "LaminarSourceGeometry | None", network: "EIGNetwork", positions: "jax.Array",
    geometry_meta: "dict[str, Any] | None", n: int,
) -> "tuple[jax.Array, dict[str, Any] | None]":
    """``construct()`` stage: optional explicit geometry override -> ``(positions, geometry_meta)``."""
    if geometry is not None:
        if geometry.n_units_total != n:
            raise ValueError(
                f"geometry_n_units_total_mismatch: "
                f"geometry.n_units_total={geometry.n_units_total} but cfg network n={n}"
            )
        positions = geometry.positions_array(dtype=network.params.v0.dtype.name)
        geometry_meta = geometry.to_dict()
    return positions, geometry_meta


def _construct_compile_connections(
    cfg: "Configuration", network: "EIGNetwork", n: int, geometry_meta: "dict[str, Any] | None",
    net: Mapping[str, Any], edge_list: "EdgeList", positions: "jax.Array | None" = None,
) -> "tuple[Configuration, EdgeList]":
    """``construct()`` stage: compile declarative ``.connections()`` rules into
    real edges -> ``(cfg, edge_list)``. They append to ``edge_list``; because the
    dense backend runs on ``emitter.W`` (which does not carry these edges), force
    the edge_list backend whenever any rule materializes so the connections
    actually drive dynamics. Rule statuses flip declared->compiled.
    """
    _conn_rules = (cfg.metadata.get("circuit", {}) or {}).get("connections", [])
    _conn_mechanisms = (cfg.metadata.get("circuit", {}) or {}).get("mechanisms", [])
    if _conn_rules:
        import numpy as _np
        _ep = network.params
        _cell_labels = list(_ep.labels)
        _layer_labels = list(getattr(_ep, "layer_labels", None) or [""] * n)
        if geometry_meta and geometry_meta.get("area_labels"):
            _area_labels = list(geometry_meta["area_labels"])
        else:
            _area_labels = [str(net.get("name", "V1"))] * n
        # Mechanism-aware path only when EVERY rule fully opts in via a
        # resolvable .mechanisms() declaration; otherwise the sign-only
        # compiler runs unchanged (see _all_connection_rules_declare_resolvable_mechanism).
        if _all_connection_rules_declare_resolvable_mechanism(_conn_rules, _conn_mechanisms):
            _needs_positions = any(r.get("max_in_degree") is not None for r in _conn_rules)
            if _needs_positions and positions is None:
                raise ValueError(
                    "connection rule(s) declare max_in_degree (spatially-localized sampling) "
                    "but no neuron positions are available at this construct() stage"
                )
            _positions_np = _np.asarray(positions) if _needs_positions else None
            _conn_edges, _counts = _compile_mechanism_aware_connection_rules(
                _conn_rules, _conn_mechanisms, _area_labels, _layer_labels, _cell_labels,
                _np.asarray(_ep.sign), n, edge_list.weight.dtype,
                int(cfg.metadata.get("seed", 0) or 0), positions=_positions_np,
            )
        else:
            _conn_edges, _counts = _compile_connection_rules(
                _conn_rules, _area_labels, _layer_labels, _cell_labels,
                _np.asarray(_ep.sign), n, edge_list.weight.dtype,
                int(cfg.metadata.get("seed", 0) or 0),
            )
        if _conn_edges is not None and _conn_edges.n_edges > 0:
            edge_list = _concat_edge_lists(edge_list, _conn_edges)
            cfg = cfg.runtime(recurrent_backend="edge_list")
        cfg = _mark_connections_compiled(cfg, _counts)
    return cfg, edge_list


def _construct_build_static(cfg: "Configuration", geometry_meta: "dict[str, Any] | None") -> "dict[str, Any]":
    """``construct()`` stage: resolve ``n_contacts`` + assemble the model's static dict."""
    n_contacts: int = 16
    if cfg.probes:
        _nc = cfg.probes[0].get("n_contacts", 16)
        try:
            _nc = int(_nc)
        except (TypeError, ValueError):
            _nc = 16
        if _nc < 2:
            raise ValueError(
                f"probe n_contacts must be >= 2; got {_nc!r} in first probe"
            )
        n_contacts = _nc
    static: dict[str, Any] = {"n_contacts": n_contacts, "operator_status": operator_status()}
    if geometry_meta is not None:
        static["geometry"] = geometry_meta
        if "neuron_rows" in geometry_meta:
            static["neuron_metadata"] = list(geometry_meta["neuron_rows"])
            static["neuron_metadata_summary"] = {
                "n_rows": len(geometry_meta["neuron_rows"]),
                "areas": sorted(set(geometry_meta.get("area_labels", []))),
                "layers": sorted(set(geometry_meta.get("layer_labels", []))),
                "cell_types": sorted(set(geometry_meta.get("cell_type_labels", []))),
            }
    return static


def construct(
    cfg: "Configuration | Any",
    runtime: "Any | None" = None,
    *,
    geometry: "LaminarSourceGeometry | None" = None,
) -> Model:
    """Construct a runnable :class:`Model`. Two call forms:

    - ``construct(cfg)`` / ``construct(cfg, geometry=...)`` -- the
      :class:`Configuration`-based path (unchanged; this is the original
      signature and remains fully backward compatible).
    - ``construct(tensor, runtime)`` -- the canonical NeuronalTensor path
      (0.4.7+): ``tensor`` is a :class:`jaxfne.neuronal_tensor.NeuronalTensor`,
      ``runtime`` a :class:`jaxfne.neuronal_tensor.RuntimeConfiguration`
      (defaults to ``RuntimeConfiguration()`` if omitted). This is the same
      construction logic as the compatibility wrapper
      :func:`jaxfne.neuronal_tensor.construct_neuronal_tensor` -- both reach
      the same internal implementation.

    The returned model is a computational scaffold; its field/probe outputs
    are proxy readouts, not calibrated physical signals.
    """
    from .neuronal_tensor import NeuronalTensor, RuntimeConfiguration, _construct_neuronal_tensor_impl

    if isinstance(cfg, NeuronalTensor):
        if runtime is None:
            runtime = RuntimeConfiguration()
        elif not isinstance(runtime, RuntimeConfiguration):
            raise TypeError(
                "construct(tensor, runtime) requires a RuntimeConfiguration "
                f"for the second argument when the first is a NeuronalTensor; "
                f"got {type(runtime).__name__}"
            )
        if geometry is not None:
            raise ValueError(
                "geometry= is not supported on the NeuronalTensor path -- "
                "geometry comes from each Layer's declared Geometry3D + "
                "Area.pose instead."
            )
        return _construct_neuronal_tensor_impl(
            cfg, seed=runtime.seed, duration_ms=runtime.duration_ms,
            dt_ms=runtime.dt_ms, emitter=runtime.emitter,
        )

    if runtime is not None:
        raise ValueError(
            "construct(cfg, runtime=...) is not supported for a Configuration "
            "-- a Configuration already carries its own runtime via "
            ".runtime(...). runtime= is only used with a NeuronalTensor "
            "first argument."
        )
    return _construct_from_configuration(cfg, geometry=geometry)


def _construct_from_configuration(cfg: Configuration, *, geometry: "LaminarSourceGeometry | None" = None) -> Model:
    """Validate a :class:`Configuration` and build a runnable :class:`Model`.

    Raises ``ValueError`` if the configuration is invalid or names an
    unsupported emitter family (only the Izhikevich kernel is implemented; an
    explicitly declared family such as ``"lif"``/``"glif"`` fails loudly rather
    than silently substituting Izhikevich). An optional
    :class:`LaminarSourceGeometry` overrides geometry derived from the config.
    The returned model is a computational scaffold; its field/probe outputs are
    proxy readouts, not calibrated physical signals.
    """
    _construct_validate_config(cfg)
    net = cfg.networks[0]
    dtype_name_cfg = str(cfg.metadata.get("dtype", "float32"))

    network, positions, geometry_meta, n, _prebuilt_edges = _construct_build_network(cfg, net, dtype_name_cfg)
    cfg, edge_list = _construct_resolve_edge_list(cfg, network, _prebuilt_edges)
    positions, geometry_meta = _construct_apply_geometry_override(geometry, network, positions, geometry_meta, n)

    # Canonical-column biophysics: random v0 (always) + deep-E grading and PV<->E
    # local strengthening (laminar columns). Reproducible from cfg seed.
    emitter_params, edge_list = _apply_canonical_biophysics(
        network.params, positions, edge_list, cfg
    )
    network = replace(network, params=emitter_params)

    cfg, edge_list = _construct_compile_connections(cfg, network, n, geometry_meta, net, edge_list, positions=positions)
    static = _construct_build_static(cfg, geometry_meta)

    return Model(
        cfg=cfg,
        params={"emitter": network.params, "positions": positions, "edge_list": edge_list},
        static=static,
    )


def _empty_edge_list(jdtype: Any) -> "EdgeList":
    """An EdgeList with zero edges (no recurrence)."""
    z_i = jnp.zeros((0,), dtype=jnp.int32)
    z_f = jnp.zeros((0,), dtype=jdtype)
    return EdgeList(pre=z_i, post=z_i, weight=z_f, receptor_index=z_i, tau_ms=z_f)


def _model_edge_list(model: "Model", jdtype: Any) -> "EdgeList":
    """Return a model's recurrence as an EdgeList.

    Uses ``params["edge_list"]`` when present; otherwise converts the dense
    ``emitter.W`` (the dense and edge_list backends are numerically equivalent),
    or returns an empty list when the model has no real recurrence.
    """
    el = model.params.get("edge_list")
    if el is not None:
        return el
    emitter: IzhikevichParams = model.params["emitter"]
    W = emitter.W
    if W.shape[0] == emitter.n_neurons and W.shape[0] > 0:
        return make_edge_list_from_dense(W, dtype=("float64" if jdtype == jnp.float64 else "float32"))
    return _empty_edge_list(jdtype)


def _connect_validate_models(models: "tuple[Model, ...]") -> "list[IzhikevichParams]":
    """``connect()`` stage: argument validation -> per-model emitter list."""
    if len(models) < 2:
        raise ValueError("connect() requires at least two models")
    for i, m in enumerate(models):
        if not isinstance(m, Model):
            raise TypeError(f"connect() argument {i} is not a Model (got {type(m).__name__})")
    emitters: list[IzhikevichParams] = []
    for i, m in enumerate(models):
        em = m.params.get("emitter")
        if not isinstance(em, IzhikevichParams):
            raise TypeError(
                f"connect() supports IzhikevichParams emitters only; model {i} has "
                f"{type(em).__name__}"
            )
        emitters.append(em)
    return emitters


def _connect_reconcile_runtime(
    models: "tuple[Model, ...]", emitters: "list[IzhikevichParams]", strict: bool
) -> Any:
    """``connect()`` stage: dtype + dt_ms must agree (strict) -> merged jdtype."""
    dtypes = {str(em.v0.dtype) for em in emitters}
    if len(dtypes) > 1:
        if strict:
            raise ValueError(f"connect() models have mismatched dtypes: {sorted(dtypes)}")
        warnings.warn(f"connect() mismatched dtypes {sorted(dtypes)}; using the first.", RuntimeWarning, stacklevel=2)
    jdtype = emitters[0].v0.dtype
    dts = {float(m.cfg.metadata.get("dt_ms")) for m in models if m.cfg.metadata.get("dt_ms") is not None}
    if len(dts) > 1:
        if strict:
            raise ValueError(f"connect() models have mismatched dt_ms: {sorted(dts)}")
        warnings.warn(f"connect() mismatched dt_ms {sorted(dts)}; sim uses the duration/dt passed to simulate().", RuntimeWarning, stacklevel=2)
    return jdtype


def _connect_resolve_namespace(
    models: "tuple[Model, ...]", namespace: "Sequence[str] | None", strict: bool
) -> "tuple[list[str] | None, list[list[dict[str, Any]]]]":
    """``connect()`` stage: namespace validation + area-collision check -> (ns, tables)."""
    ns = list(namespace) if namespace is not None else None
    if ns is not None and len(ns) != len(models):
        raise ValueError(f"namespace must have one entry per model ({len(models)}), got {len(ns)}")
    tables = [m.neuron_table() for m in models]
    if ns is None:
        seen: set[str] = set()
        for t in tables:
            areas_k = {str(r.get("area")) for r in t}
            clash = seen & areas_k
            if clash:
                msg = (f"connect() area label collision across models ({sorted(clash)}); "
                       f"pass namespace=(...) to disambiguate")
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            seen |= areas_k
    return ns, tables


def _connect_merge_emitter(emitters: "list[IzhikevichParams]", jdtype: Any) -> "IzhikevichParams":
    """``connect()`` stage: concat per-neuron arrays, reconcile scalars -> merged emitter."""
    def _cat(attr: str) -> jax.Array:
        return jnp.concatenate([getattr(em, attr).astype(jdtype) for em in emitters], axis=0)

    sign_cat = _cat("sign")
    labels = tuple(lbl for em in emitters for lbl in em.labels)
    layer_labels = tuple(
        (em.layer_labels[i] if em.layer_labels is not None else "unspecified")
        for em in emitters for i in range(em.n_neurons)
    )
    scs = {em.source_calibration_status for em in emitters}
    source_cal = emitters[0].source_calibration_status if len(scs) == 1 else "mixed_uncalibrated_proxy"
    return IzhikevichParams(
        a=_cat("a"), b=_cat("b"), c=_cat("c"), d=_cat("d"),
        drive=_cat("drive"), sign=sign_cat, W=jnp.zeros((0, 0), dtype=jdtype),
        v0=_cat("v0"), u0=_cat("u0"), source_scale=emitters[0].source_scale,
        labels=labels, layer_labels=layer_labels, source_calibration_status=source_cal,
    )


def _connect_merge_edges(models: "tuple[Model, ...]", offsets: Any, jdtype: Any) -> "EdgeList":
    """``connect()`` stage: member edges (offset), block-concatenated, sparse.

    Members couple ONLY through explicit cross-model edges (no spurious
    cross-recurrence -- the sparse realization of a block-diagonal W).
    """
    merged_edges = _empty_edge_list(jdtype)
    for k, m in enumerate(models):
        el = _model_edge_list(m, jdtype)
        off = int(offsets[k])
        el = replace(el, pre=el.pre + off, post=el.post + off)
        merged_edges = _concat_edge_lists(merged_edges, el)
    return merged_edges


def _connect_merge_positions(models: "tuple[Model, ...]", layout: str, jdtype: Any) -> "jax.Array":
    """``connect()`` stage: merged positions (offset_x keeps columns spatially distinct)."""
    pos_parts: list[jax.Array] = []
    xcursor = 0.0
    for m in models:
        p = m.params["positions"].astype(jdtype)
        if layout == "offset_x" and p.shape[0] > 0:
            xmin = float(jnp.min(p[:, 0])); xmax = float(jnp.max(p[:, 0]))
            p = p + jnp.asarray([xcursor - xmin, 0.0, 0.0], dtype=jdtype)
            xcursor += (xmax - xmin) + 1.0
        pos_parts.append(p)
    return jnp.concatenate(pos_parts, axis=0) if pos_parts else jnp.zeros((0, 3), dtype=jdtype)


def _connect_merge_neuron_metadata(
    tables: "list[list[dict[str, Any]]]", ns: "list[str] | None", counts: "list[int]"
) -> "tuple[list[dict[str, Any]], list[str], list[str], list[str], Any]":
    """``connect()`` stage: reindex ids, namespace areas, tag model index.

    Returns ``(merged_rows, orig_area, layer_lab, cell_lab, model_labels)``.
    """
    import numpy as np

    merged_rows: list[dict[str, Any]] = []
    orig_area: list[str] = []
    layer_lab: list[str] = []
    cell_lab: list[str] = []
    model_labels = np.concatenate([np.full(c, k, dtype=int) for k, c in enumerate(counts)]) if counts else np.zeros(0, int)
    nid = 0
    for k, t in enumerate(tables):
        prefix = (ns[k] + "/") if ns is not None else ""
        for r in t:
            row = dict(r)
            a_orig = str(row.get("area"))
            orig_area.append(a_orig)
            layer_lab.append(str(row.get("layer")))
            cell_lab.append(str(row.get("cell_type")))
            row["neuron_id"] = nid
            row["model"] = k
            row["area"] = prefix + a_orig
            merged_rows.append(row)
            nid += 1
    return merged_rows, orig_area, layer_lab, cell_lab, model_labels


def _connect_compile_cross_edges(
    edges: "Sequence[Mapping[str, Any]] | None",
    models: "tuple[Model, ...]",
    orig_area: "list[str]", layer_lab: "list[str]", cell_lab: "list[str]",
    sign_cat: Any, n_total: int, jdtype: Any, model_labels: Any,
) -> "tuple['EdgeList | None', list[int]]":
    """``connect()`` stage: compile cross-model edges (selectors match original area + model idx)."""
    import numpy as np

    if not edges:
        return None, []
    default_seed = int(models[0].cfg.metadata.get("seed", 0) or 0)
    return _compile_connection_rules(
        list(edges), orig_area, layer_lab, cell_lab, np.asarray(sign_cat),
        n_total, jdtype, default_seed, model_labels=model_labels,
    )


def _connect_merge_cfg(
    models: "tuple[Model, ...]", ns: "list[str] | None", name: "str | None", layout: str,
    counts: "list[int]", n_total: int, edges: "Sequence[Mapping[str, Any]] | None",
    cross_counts: "list[int]",
) -> "Configuration":
    """``connect()`` stage: merged cfg -- conservative truth gates, ensemble marker, cross rules."""
    cfg2 = models[0].cfg
    try:
        cfg2 = cfg2.runtime(recurrent_backend="edge_list")
    except Exception:
        pass
    md = {**cfg2.metadata}
    md["claim_level"] = "computational_scaffold"
    md["field_solver_status"] = "linear_solver"
    md["physical_amplitude_calibrated"] = all(
        bool(m.cfg.metadata.get("physical_amplitude_calibrated", False)) for m in models
    )
    all_area_names: list[str] = []
    for k, m in enumerate(models):
        for ar in (m.cfg.metadata.get("column_names") or []):
            all_area_names.append((ns[k] + "/" + str(ar)) if ns is not None else str(ar))
    if all_area_names:
        md["column_names"] = all_area_names
        md["column_count"] = len(all_area_names)
    total_cross = int(sum(cross_counts)) if cross_counts else 0
    md["ensemble"] = {
        "name": str(name) if name is not None else None,
        "n_models": len(models),
        "model_sizes": [int(x) for x in counts],
        "n_neurons_total": n_total,
        "namespace": list(ns) if ns is not None else None,
        "cross_model_edges": total_cross,
        "layout": str(layout),
    }
    if edges:
        circuit = {**md.get("circuit", {})}
        conns = list(circuit.get("connections", []))
        for i, rule in enumerate(edges):
            c = int(cross_counts[i]) if i < len(cross_counts) else 0
            conns.append({
                **dict(rule),
                "scope": "cross_model",
                "status": "compiled" if c > 0 else "compiled_no_matching_edges",
                "compiled_n_edges": c,
            })
        circuit["connections"] = conns
        md["circuit"] = circuit
    return replace(cfg2, metadata=md)


def _connect_merge_static(
    models: "tuple[Model, ...]", merged_rows: "list[dict[str, Any]]", strict: bool, ensemble: dict,
) -> "dict[str, Any]":
    """``connect()`` stage: merged static -- reconcile n_contacts, most-conservative operator_status."""
    n_contacts_set = {int(m.static.get("n_contacts", 16)) for m in models}
    if len(n_contacts_set) > 1 and strict:
        raise ValueError(
            f"connect() models have mismatched n_contacts {sorted(n_contacts_set)}; "
            f"field projection needs equal contacts"
        )
    if len(n_contacts_set) > 1:
        warnings.warn(f"connect() mismatched n_contacts {sorted(n_contacts_set)}; using the first.", RuntimeWarning, stacklevel=2)
    rank = {"not_implemented": 0, "prototype_api": 1, "experimental": 2, "validated": 3}
    op: dict[str, str] = {}
    for m in models:
        for key, val in (m.static.get("operator_status") or {}).items():
            if key not in op or rank.get(val, 1) < rank.get(op[key], 1):
                op[key] = val
    summary = {
        "n_rows": len(merged_rows),
        "areas": sorted({str(r.get("area")) for r in merged_rows}),
        "layers": sorted({str(r.get("layer")) for r in merged_rows}),
        "cell_types": sorted({str(r.get("cell_type")) for r in merged_rows}),
    }
    return {
        **models[0].static,
        "n_contacts": int(models[0].static.get("n_contacts", 16)),
        "operator_status": op,
        "geometry": {**models[0].static.get("geometry", {}), "neuron_rows": merged_rows},
        "neuron_metadata": merged_rows,
        "neuron_metadata_summary": summary,
        "ensemble": ensemble,
    }


def connect(
    *models: "Model",
    edges: "Sequence[Mapping[str, Any]] | None" = None,
    namespace: "Sequence[str] | None" = None,
    layout: str = "offset_x",
    strict: bool = True,
    name: "str | None" = None,
) -> "Model":
    """Fuse two or more constructed :class:`Model` s into one ensemble Model.

    This is the model-to-model composition operator of the jaxfne grammar
    (``construct`` builds one model; ``connect`` joins several). The result is a
    normal ``Model`` — ``simulate`` / ``tune`` / ``manifest`` / ``vis.*`` all
    work on it unchanged. The merge is purely structural and stays on the sparse
    edge_list backend: each member's neurons, per-neuron emitter parameters,
    positions, and recurrence are concatenated with index offsets, and optional
    cross-model edges are compiled from ``edges``. No simulation numerics of a
    member change; the math/physics layer stays valid and the biological claim is
    only ever as strong as the member configurations back.

    Parameters
    ----------
    *models:
        Two or more Models built by :func:`construct`. Each must use an
        ``IzhikevichParams`` emitter (the only family supported in this version).
    edges:
        Optional list of cross-model connection rules using the
        :meth:`Configuration.connections` selector grammar, extended with a
        ``"model"`` key (integer member index). Example::

            edges=[dict(source={"model": 0, "area": "V1", "cell_type": "E",
                                "layers": ["L2/3"]},
                        target={"model": 1, "area": "V2", "layers": ["L4"]},
                        probability=0.1, weight=0.5, sign="excitatory")]

        ``probability`` / ``weight`` / ``sign`` behave exactly as in
        :func:`_compile_connection_rules`. Selectors match members' original
        (pre-``namespace``) area labels, so a rule reads the same whether or not
        ``namespace`` is set.
    namespace:
        Optional per-model area prefix (one string per model), e.g.
        ``("A", "B")``. Areas become ``"A/V1"``, ``"B/V1"`` in the merged neuron
        table so two members sharing an area name don't collide. Without it, an
        area-label collision across members raises (``strict=True``).
    layout:
        ``"offset_x"`` (default) translates each member along x so columns are
        spatially distinct for field projection (depth/z preserved). ``"keep"``
        leaves member coordinates untouched.
    strict:
        When True, mismatched ``dt_ms`` / dtype / ``n_contacts`` or an
        un-namespaced area collision raise. When False they warn and coerce
        (take the first member's value).
    name:
        Optional label recorded in ``metadata["ensemble"]["name"]``.

    Returns
    -------
    Model
        The ensemble model. Recurrence lives entirely in ``params["edge_list"]``
        (member-internal edges, offset, plus cross-model edges); ``emitter.W`` is
        the ``(0, 0)`` placeholder, so the edge_list backend is auto-selected.
    """
    import numpy as np

    emitters = _connect_validate_models(models)
    jdtype = _connect_reconcile_runtime(models, emitters, strict)

    counts = [int(em.n_neurons) for em in emitters]
    offsets = np.cumsum([0, *counts])[:-1]
    n_total = int(sum(counts))

    ns, tables = _connect_resolve_namespace(models, namespace, strict)
    merged_emitter = _connect_merge_emitter(emitters, jdtype)
    sign_cat = merged_emitter.sign
    merged_edges = _connect_merge_edges(models, offsets, jdtype)
    merged_positions = _connect_merge_positions(models, layout, jdtype)
    merged_rows, orig_area, layer_lab, cell_lab, model_labels = _connect_merge_neuron_metadata(
        tables, ns, counts
    )

    cross_el, cross_counts = _connect_compile_cross_edges(
        edges, models, orig_area, layer_lab, cell_lab, sign_cat, n_total, jdtype, model_labels
    )
    if cross_el is not None:
        merged_edges = _concat_edge_lists(merged_edges, cross_el)

    cfg2 = _connect_merge_cfg(models, ns, name, layout, counts, n_total, edges, cross_counts)
    merged_static = _connect_merge_static(models, merged_rows, strict, cfg2.metadata["ensemble"])

    return Model(
        cfg=cfg2,
        params={"emitter": merged_emitter, "positions": merged_positions, "edge_list": merged_edges},
        static=merged_static,
    )


def operator_status() -> dict[str, str]:
    """Return the current operator status registry for all declared operators.

    Returns a dict mapping operator symbol names to their readiness strings
    (e.g., ``"prototype_api"``, ``"not_implemented"``). This is a scaffold
    declaration; no operator has been empirically validated.

    Returns
    -------
    dict[str, str]
        Operator name to status string mapping.
    """
    return _default_operator_status()


def standard_visual_omission() -> Paradigm:
    """Construct a Paradigm with standard visual oddball/omission task conditions.

    12 core conditions:
      - AAAB, AXAB, AAXB, AAAX (omission in p2, p3, p4, and p4 respectively)
      - BBBA, BXBA, BBXA, BBBX (omission in p2, p3, p4, and p4 respectively)
      - RRRR, RXRR, RRXR, RRRX (random-control stimuli, omissions in p2, p3, p4)

    Event codes:
      - fx: 10 (fixation)
      - p1: 101 (standard visual P1)
      - p2: 103 (standard visual P2)
      - p3: 105 (standard visual P3)
      - p4: 107 (standard visual P4)
      - rw: 96 (reward marker)

    Analysis windows:
      - baseline: -500 to 0 ms (pre-stimulus)
      - event: 0 to 500 ms (post-stimulus)
      - post_event: 500 to 1000 ms (post-stimulus)

    Comparison: P1 onset (code 101) at t=0.
    Pre-stimulus buffer: 1000 ms.
    """
    # Define event code mapping (immutable, hardcoded).
    event_codes = {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    }

    # Standard stimulus identifiers.
    std_A = "stimulus_A"
    std_B = "stimulus_B"
    std_X = "stimulus_omitted"
    std_R = "random_stimulus"

    # Define conditions with condition numbers and omission metadata.
    conditions = [
        # A-sequence (AAAB family): oddball in position 4.
        ParadigmCondition(
            name="AAAB",
            sequence=(std_A, std_A, std_A, std_B),
            omission_position=None,
            probability=None,
            condition_numbers=(1, 2),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AXAB",
            sequence=(std_A, std_X, std_A, std_B),
            omission_position="p2",
            probability=None,
            condition_numbers=(3,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AAXB",
            sequence=(std_A, std_A, std_X, std_B),
            omission_position="p3",
            probability=None,
            condition_numbers=(4,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AAAX",
            sequence=(std_A, std_A, std_A, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(5,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
            ),
        ),
        # B-sequence (BBBA family): oddball in position 4.
        ParadigmCondition(
            name="BBBA",
            sequence=(std_B, std_B, std_B, std_A),
            omission_position=None,
            probability=None,
            condition_numbers=(6, 7),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BXBA",
            sequence=(std_B, std_X, std_B, std_A),
            omission_position="p2",
            probability=None,
            condition_numbers=(8,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BBXA",
            sequence=(std_B, std_B, std_X, std_A),
            omission_position="p3",
            probability=None,
            condition_numbers=(9,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BBBX",
            sequence=(std_B, std_B, std_B, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(10,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
            ),
        ),
        # R-sequence (random-control family): random stimulus identity, omissions in p2, p3, p4.
        ParadigmCondition(
            name="RRRR",
            sequence=(std_R, std_R, std_R, std_R),
            omission_position=None,
            probability=None,
            condition_numbers=tuple(range(11, 27)),  # [11-26]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RXRR",
            sequence=(std_R, std_X, std_R, std_R),
            omission_position="p2",
            probability=None,
            condition_numbers=tuple(range(27, 35)),  # [27-34]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RRXR",
            sequence=(std_R, std_R, std_X, std_R),
            omission_position="p3",
            probability=None,
            condition_numbers=(35, 37, 39, 41),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RRRX",
            sequence=(std_R, std_R, std_R, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(36, 38, 40) + tuple(range(42, 51)),  # [36, 38, 40, 42-50]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
    ]

    return Paradigm(
        name="standard_visual_omission",
        conditions=tuple(conditions),
        comparison_code=event_codes["p1"],
        comparison_label="p1",
        pre_stimulus_buffer_ms=1000.0,
        analysis_windows={
            "baseline": (-500.0, 0.0),
            "event": (0.0, 500.0),
            "post_event": (500.0, 1000.0),
        },
        event_codes=event_codes,
        metadata={
            "task_type": "visual_oddball_omission",
            "n_conditions": 12,
            "n_trials_per_condition": {c.name: len(c.condition_numbers) for c in conditions},
        },
    )


def trial_batch(
    conditions: Sequence[ParadigmCondition],
    n_reps: int = 1,
    seed: int = 0,
    seed_policy: str = "paired_by_replicate",
    batch_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TrialBatch:
    """Create a TrialBatch by repeating conditions.

    Correctly iterates reps then conditions to ensure deterministic ordering.
    Assigns unique trial_id in format "trial_{index:04d}_{condition_name}".

    Seed policy:
      - "paired_by_replicate" (default): seed = base_seed + replicate_index
      - "unique_per_trial": seed = base_seed + trial_index
    """
    if seed_policy not in {"unique_per_trial", "paired_by_replicate"}:
        raise ValueError(
            f"invalid_seed_policy: {seed_policy!r}; "
            "must be one of {'paired_by_replicate', 'unique_per_trial'}"
        )

    trials: list[TrialSpec] = []
    idx = 0
    for r in range(n_reps):
        for cond in conditions:
            t_id = f"trial_{idx:04d}_{cond.name}"
            if seed_policy == "unique_per_trial":
                trial_seed = seed + idx
            else:  # paired_by_replicate
                trial_seed = seed + r
            trials.append(
                TrialSpec(
                    trial_id=t_id,
                    condition=cond,
                    seed=trial_seed,
                    metadata={"rep": r},
                )
            )
            idx += 1
    return TrialBatch(
        trials=tuple(trials),
        batch_id=batch_id or f"batch_{seed}",
        metadata=metadata or {},
    )


def run_trials(
    model: Model, batch: TrialBatch, sim: Simulation, *, collect_errors: bool = False
) -> TrialBatchResult:
    """Execute a batch of trials using the model.

    Args:
        model: Model instance to run trials on.
        batch: TrialBatch with trial specifications.
        sim: Simulation parameters for each trial.
        collect_errors: If False (default), raise immediately on first trial failure.
                       If True, record failures in TrialResult and continue.

    Delegates to model.run_trials() for the actual execution.
    """
    return model.run_trials(batch, sim, collect_errors=collect_errors)


def run_receipt(
    model: "Model", signals: Signals, *, tags: Optional[dict[str, Any]] = None
) -> RunReceipt:
    """Build a RunReceipt for a completed simulation run.

    Convenience wrapper around Model.run_receipt().

    Args:
        model: Model that produced the signals.
        signals: Signals returned by model.simulate().
        tags: Optional user-supplied key-value metadata.

    Returns:
        RunReceipt with frozen truth gates and deterministic receipt_id.
    """
    return model.run_receipt(signals, tags=tags)


def provenance_receipt(
    branch: str = "unknown",
    sha: str = "unknown",
    dirty: bool = False,
) -> dict[str, Any]:
    """Capture release provenance atomically.

    Freezes branch, SHA, dirty flag, and jaxfne version metadata into a single
    JSON-safe dict for release auditing and reproducibility.

    Args:
        branch: Git branch name (e.g., "main", "feat/something"). Default: "unknown".
        sha: Git commit SHA (short or full). Default: "unknown".
        dirty: True if working tree has uncommitted changes. Default: False.

    Returns:
        dict[str, Any] (JSON-safe) with keys:
        - branch (str)
        - sha (str)
        - dirty (bool)
        - jaxfne_version (str, from _JAXFNE_VERSION)
        - config_schema_version (str, from _JAXFNE_CONFIG_SCHEMA_VERSION)
        - manifest_schema_version (str, from _MANIFEST_SCHEMA_VERSION)
        - timestamp (str, ISO-8601 format)

    Example:
        receipt = provenance_receipt(branch="main", sha="abc123def456", dirty=False)
        json.dumps(receipt, allow_nan=False)  # Always succeeds
    """
    import json
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    receipt = {
        "branch": branch,
        "sha": sha,
        "dirty": dirty,
        "jaxfne_version": _JAXFNE_VERSION,
        "config_schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
        "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
        "timestamp": timestamp,
    }

    # Verify JSON-safe by attempting serialization
    try:
        json.dumps(receipt, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"provenance_receipt produced non-JSON-safe dict: {e}"
        ) from e

    return receipt


def get_signal(obj: Any, key: str, **kwargs: Any) -> Any:
    """Thin free-function accessor that delegates to :meth:`Signals.get`.

    This is a convenience wrapper only; it does not implement any signal logic
    of its own. ``obj`` must be a :class:`Signals` instance.
    """
    if isinstance(obj, Signals):
        return obj.get(key, **kwargs)
    raise TypeError(
        f"get_signal expects a Signals instance, got {type(obj).__name__}"
    )


def readout_spec(
    name: str,
    metric: str,
    *,
    time_window_ms: Optional[tuple[float, float]] = None,
    n_contacts_slice: Optional[tuple[int, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ReadoutSpec:
    """Build a ReadoutSpec for declarative feature extraction.

    Args:
        name: Unique label for this readout spec.
        metric: One of _KNOWN_READOUT_METRICS (spike_rate_hz, spike_count,
                mean_V_m, csd_abs_mean, lfp_abs_mean, source_abs_mean).
        time_window_ms: Optional (start_ms, end_ms) temporal slice.
        n_contacts_slice: Optional (start, end) contact-depth slice for field modes.
        metadata: Optional user-supplied metadata dict.

    Returns:
        ReadoutSpec (frozen, JSON-safe).
    """
    return ReadoutSpec(
        name=name,
        metric=metric,
        time_window_ms=time_window_ms,
        n_contacts_slice=n_contacts_slice,
        metadata=metadata or {},
    )


def dataset_spec(**kwargs: Any) -> DatasetSpec:
    """Return a DatasetSpec schema declaration."""
    return DatasetSpec(**kwargs)


def surrogate_config(**kwargs: Any) -> SurrogateConfig:
    """Return a SurrogateConfig declaration for an Optax gradient path."""
    return SurrogateConfig(**kwargs)


def laminar_source_geometry(
    populations: Sequence["LaminarPopulation"],
) -> "LaminarSourceGeometry":
    """Build a :class:`LaminarSourceGeometry` from an ordered population sequence.

    Depth overlap between populations is allowed; co-located cell types sharing
    a layer band are anatomically expected. Hard validation errors are raised only
    for invalid depth ranges, zero n_units, or empty population list.
    No physical-amplitude or calibration claim is made.
    """
    pops = tuple(populations)
    if not pops:
        raise ValueError("laminar_source_geometry requires at least one LaminarPopulation")
    issues: list[str] = []
    for p in pops:
        v = p.validate()
        if not v["valid"]:
            issues.extend([f"{p.name}:{i}" for i in v["issues"]])
    if issues:
        raise ValueError(f"Invalid LaminarPopulation(s): {issues}")
    n_total = sum(p.n_units for p in pops)
    return LaminarSourceGeometry(populations=pops, n_units_total=n_total)


def enable_x64() -> dict[str, Any]:
    """Enable JAX float64 mode before constructing arrays and report status."""
    jax.config.update("jax_enable_x64", True)
    return {"x64_enabled": bool(jax.config.read("jax_enable_x64")), "status": "enabled"}


# ──────────────────────────────────────────────────────────────
# v0.0.17 readout spec
# ──────────────────────────────────────────────────────────────

_OBJECTIVE_REPORT_SCHEMA_VERSION = "objective_report.v0.0.18"


# ──────────────────────────────────────────────────────────────
# v0.0.15 config foundation
# ───────────────────────────────────────────────��──────────────

_JAXFNE_CONFIG_SCHEMA_VERSION = "jaxfne.config.v0.0.16"

_REQUIRED_CONFIG_SECTIONS = frozenset(
    {"schema_version", "run", "truth", "network", "emitter", "field", "probes"}
)

_RECOGNIZED_OPTIONAL_CONFIG_SECTIONS = frozenset({
    "runtime",
    "geometry",
    "paradigm",
    "trials",
    "stimulus",
    "features",
    "objective",
    "targets",
    "validation",
    "output",
    "metadata",
})

# _CONSERVATIVE_TRUTH_DEFAULTS moved to jaxfne/_signals.py (only consumer is
# _evaluate_gate_spec, which lives there) and re-exported above.

#: Canonical field-solver-status values. ``linear_solver`` is the shipped laminar
#: proxy (a linear readout operator, no PDE); ``pde_solver`` is reserved for a future
#: elliptic/volume-conductor solve gated on boundary/gauge/residual/convergence tests.
_VALID_FIELD_SOLVER_STATUS = (None, "linear_solver", "pde_solver")


def migrate_schema(meta: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy truth/metadata dict to the canonical truth-gate schema.

    Pure, JSON-safe rewrite applied on load of older manifests/configs. The legacy
    key/value strings are built by concatenation so the repository stays free of
    literal occurrences even here.

    - legacy physical-amplitude key -> ``physical_amplitude_calibrated`` (bool kept)
    - legacy laminar field-solver value -> ``linear_solver``
    - legacy proxy field-claim value -> ``proxy_readout``
    - drop the legacy truth-mode key

    Returns a new dict; the input is not mutated.
    """
    _amp = "physical_amplitude_" + "claim_allowed"
    _solver = "laminar_proxy" + "_no_pde"
    _claim = "proxy_readout" + "_only"
    _tmode = "truth" + "_mode"
    out: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if key == _tmode:
            continue
        if key == _amp:
            out["physical_amplitude_calibrated"] = value
            continue
        if key == "field_solver_status" and value == _solver:
            out[key] = "linear_solver"
            continue
        if key == "field_claim_level" and value == _claim:
            out[key] = "proxy_readout"
            continue
        out[key] = value
    from ._config import clamp_truth_gate_metadata

    return clamp_truth_gate_metadata(out)


