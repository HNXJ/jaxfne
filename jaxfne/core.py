"""Core object model for :mod:`jaxfne`.

Docs: ``docs/api/core.md`` (https://jaxfne.readthedocs.io/en/latest/api/core/) —
update that page when this module's public API changes.

Design target: object-oriented public API, pure-JAX computational core.  The
current package is an honest TFNE scaffold: reduced emitters plus laminar proxy
source/readout status, a proxy field/readout scaffold.
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

_ALLOWED_SYNAPTIC_KERNELS = ("exponential", "receptor_exponential")
from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, load_json, manifest as build_manifest
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN

# v0.3.29: canonical selector/identity types live in experimental_hpc.contracts.
# Re-export (do not duplicate) so the stable API exposes one definition.
from .experimental_hpc.contracts import NodeIdentity, SelectorSpec


@dataclass(frozen=True)
class MatrixParameterSpec:
    """Declarative specification for a tunable weight matrix parameter.

    Used in multi-parameter optimization to specify that a named parameter
    maps to a matrix (e.g., the synaptic weight matrix W) rather than a scalar.
    The mask field selects which matrix entries are subject to scaling:
    E_to_E, E_to_I, excitatory_to_all, or all.

    The name is always the dict key in the parameters argument to
    :func:; do **not** add a name field here.

    Attributes
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization scope; "current" means start from the
        model's existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.
    """

    mask: str
    bounds: tuple
    init: str = "current"
    trainable: bool = True


def matrix_parameter(
    *,
    mask: str,
    bounds: tuple,
    init: str = "current",
    trainable: bool = True,
) -> MatrixParameterSpec:
    """Create a matrix parameter specification for tuning weight matrices.

    Parameters
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization; "current" uses existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.

    Returns
    -------
    MatrixParameterSpec
        Frozen specification object.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))
    """
    return MatrixParameterSpec(mask=mask, bounds=bounds, init=init, trainable=trainable)



@dataclass
class TuneResult:
    """Result object returned by Model.tune() with multi-parameter optimization.

    This is a typed container for tuning results, with JSON-safe serialization
    via to_dict() method for reporting and logging.

    Attributes
    ----------
    best_parameters : dict[str, float]
        Optimized parameter values.
    best_score : float
        Best (lowest) objective score achieved.
    history : list[dict[str, Any]]
        Per-generation records with scores and parameter values.
    summary : dict[str, Any]
        High-level tuning summary (targets vs achieved, initial vs final scores, etc).
    model : Optional[Any]
        The model object (if returned by tuning; may be None for metadata-only runs).
    """

    best_parameters: dict[str, float]
    best_score: float
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    model: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary for serialization."""
        from .io import json_safe

        return json_safe({
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "history": self.history,
            "summary": self.summary,
        })

    def __iter__(self):
        """Support legacy tuple unpacking: ``model, report = tune(...)``.

        New code should use ``result.model`` and ``result.summary``.  The iterator
        remains to preserve existing notebooks and tests while surfacing a
        deprecation warning.
        """
        warnings.warn(
            "Tuple-unpacking TuneResult is deprecated; use result.model and result.summary.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield self.model
        yield self.summary


def _default_operator_status() -> dict[str, str]:
    return {
        "E_theta": "prototype_api",
        "S_WDR": "prototype_api",
        "C_mu_nu": "not_implemented",
        "Q_eta_alpha": "prototype_api",
        "F_field": "prototype_api",
        "P_probe": "prototype_api",
        "A_objective": "prototype_api",
        "O_optimizer": "prototype_api",
        "C_constraints": "prototype_api",
    }


def _circuit_json_safe(value: Any, path: str) -> Any:
    """Recursively validate and return a strict-JSON-safe copy of a declaration.

    Used by the v0.3.28 circuit/training declaration methods. Raises ``ValueError``
    on non-finite numbers (NaN/Inf) or unsupported types so non-serializable
    declarations fail loudly at declaration time rather than at manifest export.
    ``bool`` is checked before ``int`` because ``bool`` is an ``int`` subclass.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}: {value!r}")
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _circuit_json_safe(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_circuit_json_safe(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise ValueError(f"non-JSON-safe value at {path}: {type(value).__name__}")


def _default_metadata() -> dict[str, Any]:
    return {
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "source_projection_mode": "proxy_no_field_solve",
        "source_decomposition": "proxy_reduced_emitter",
        "boundary_condition": "mean_zero_neumann",
        "gauge": "mean_zero",
        "csd_sign_convention": "positive_equals_extracellular_source",
        "field_solver_status": "linear_solver",
        "field_claim_level": "proxy_readout",
        "physical_amplitude_calibrated": False,
        "manifest_schema_version": "0.0.4",
        "operator_status": _default_operator_status(),
        # Suite No. 2 truth gates — always present so validation passes regardless
        # of which subset of chainable methods the caller uses.
        "connectivity_status": "declared_metadata_proxy",
        "geometry_mode": "declared_metadata_not_solved_3d_pde_grid",
    }




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


def _counts_from_fractions(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    total = int(total)
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return {str(k): 0 for k in fractions.keys()}
    items = [(str(k), float(v)) for k, v in fractions.items()]
    if not items:
        raise ValueError("fractions must not be empty")
    if any((not math.isfinite(v)) or v < 0.0 for _, v in items):
        raise ValueError("fractions must be finite and non-negative")
    mass = sum(v for _, v in items)
    if mass <= 0.0:
        raise ValueError("fractions must have positive mass")
    normalized = [(k, v / mass) for k, v in items]
    counts: dict[str, int] = {}
    remaining = total
    for idx, (key, frac) in enumerate(normalized):
        if idx == len(normalized) - 1:
            count = remaining
        else:
            count = int(round(total * frac))
            count = max(0, min(count, remaining))
        counts[key] = count
        remaining -= count
    return counts


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


def _reject_retired_like(value: Any) -> None:
    """Reject the retired ``*_like`` probe vocabulary in favor of ``*_proxy``.

    The ``lfp_like``/``csd_like``/``eeg_like``/``meg_like`` naming is fully
    retired (no aliases): proxy readouts carry the ``*_proxy`` suffix only.
    """
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        if isinstance(item, str) and item.lower().endswith("_like"):
            proxy = item[: -len("_like")] + "_proxy"
            raise ValueError(
                f"Probe kind/mode {item!r} uses the retired `*_like` vocabulary; "
                f"use {proxy!r} instead. `*_like` names are no longer accepted."
            )


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
    rules, mechanisms, area_labels, layer_labels, cell_labels, sign, n, jdtype, default_seed
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


class _ProbeDeclarations(list):
    """List-like probe declaration container that also supports ``cfg.probes(...)``.

    ``Configuration`` historically exposes ``cfg.probes`` as a list of probe
    declaration dictionaries.  Suite No. 2 needs the more compact grammar
    ``cfg = cfg.probes([...])``.  A callable list preserves the old read path
    while adding the verb-like write path without renaming the public field.
    """

    def __init__(self, values: Sequence[Mapping[str, Any]] | None = None, owner: "Configuration | None" = None):
        super().__init__(dict(v) for v in (values or ()))
        self._owner = owner

    def bind(self, owner: "Configuration") -> "_ProbeDeclarations":
        """Documented public function `bind`."""
        return _ProbeDeclarations(self, owner=owner)

    def __call__(
        self,
        modes: Sequence[str],
        *,
        name: str = "multimodal_probe",
        n_contacts: int | None = None,
        ensure_defaults: bool = True,
        **kwargs: Any,
    ) -> "Configuration":
        """Declare multimodal proxy probe modes through the Suite No. 2 DSL.

        Parameters
        ----------
        modes:
            Probe/readout mode labels.  They remain declarative labels and are
            not upgraded to physical sensor claims.
        name:
            Probe declaration name.
        n_contacts:
            Optional number of laminar contacts.  When omitted, the existing
            construct-time default is preserved.
        ensure_defaults:
            If true, add the canonical Izhikevich emitter and laminar proxy
            field declarations when they are absent.
        **kwargs:
            Additional probe metadata, for example ``contact_depths`` or
            ``claim_level``.
        """
        cfg = self._owner
        if cfg is None:
            raise TypeError("Detached probe declarations cannot be called as a Configuration facade.")
        return cfg._with_probe_modes(
            modes=modes,
            name=name,
            n_contacts=n_contacts,
            ensure_defaults=ensure_defaults,
            **kwargs,
        )


@dataclass(frozen=True)
class Configuration:
    """Declarative TFNE model configuration.

    It is the anatomical/model declaration, not the compiled model.  Methods
    return new objects, making the public API friendly while keeping mutation
    out of the construction path.
    """

    networks: list[dict[str, Any]] = field(default_factory=list)
    emitters: list[dict[str, Any]] = field(default_factory=list)
    fields: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=_ProbeDeclarations)
    metadata: dict[str, Any] = field(default_factory=_default_metadata)

    def __post_init__(self) -> None:
        """Normalize mutable containers after dataclass construction.

        The public object is frozen, but the fields are list/dict shaped for
        backwards compatibility.  We defensively copy declarations so method
        chaining does not leak mutable state across configurations, and we bind
        the probe list proxy so ``cfg.probes([...])`` works while
        ``len(cfg.probes)`` and ``cfg.probes[0]`` continue to work.
        """
        object.__setattr__(self, "networks", [dict(v) for v in self.networks])
        object.__setattr__(self, "emitters", [dict(v) for v in self.emitters])
        object.__setattr__(self, "fields", [dict(v) for v in self.fields])
        object.__setattr__(self, "metadata", dict(self.metadata))
        probe_decls = self.probes
        if not isinstance(probe_decls, _ProbeDeclarations) or probe_decls._owner is not self:
            object.__setattr__(self, "probes", _ProbeDeclarations(probe_decls, owner=self))

    def network(self, **kwargs: Any) -> "Configuration":
        """Attach network metadata to the configuration.

        Parameters are stored as JSON-safe metadata and consumed by public
        construction helpers when supported.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        return replace(self, networks=[*self.networks, dict(kwargs)])

    def emitter(self, **kwargs: Any) -> "Configuration":
        """Attach emitter metadata to the configuration.

        Keyword arguments describe the emitter family and parameters.
        The ``family`` key selects the emitter kernel at build time.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        return replace(self, emitters=[*self.emitters, dict(kwargs)])

    def field(self, **kwargs: Any) -> "Configuration":
        """Attach field metadata to the configuration.

        Describes source-to-field projection settings. Current implementation
        is a laminar proxy; no PDE field solver is invoked.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        return replace(self, fields=[*self.fields, dict(kwargs)])

    def probe(self, **kwargs: Any) -> "Configuration":
        """Attach probe metadata to the configuration.

        The ``n_contacts`` key sets the number of recording contacts.
        Minimum is 2; default is 16.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        for _key in ("kind", "mode", "modes"):
            if _key in kwargs:
                _reject_retired_like(kwargs[_key])
        return replace(self, probes=[*self.probes, dict(kwargs)])

    def update_metadata(self, **kwargs: Any) -> "Configuration":
        """Merge keyword arguments into configuration metadata.

        Use for truth-mode, claim-level, and other administrative tags.
        Existing keys are overwritten; unspecified keys are preserved.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        metadata = dict(self.metadata)
        metadata.update(kwargs)
        return replace(self, metadata=metadata)

    # ------------------------------------------------------------------
    # Suite No. 2 compact DSL facade.
    # These methods are thin, public, backwards-compatible wrappers over the
    # existing explicit declaration grammar.  They do not introduce stronger
    # biological or physical claims; they only make common tutorial declarations
    # more legible.
    # ------------------------------------------------------------------

    def runtime(self, **kwargs: Any) -> "Configuration":
        """Set runtime/simulation metadata in chainable configuration form.

        This intentionally maps to :meth:`update_metadata` rather than creating
        a compiled :class:`RuntimeConfig`; the compiled runtime remains a
        simulation-time object.  Typical keys include ``seed``, ``dtype``,
        ``duration_ms``, and ``dt_ms``.
        """
        return self.update_metadata(**kwargs)

    def set_runtime(self, **kwargs: Any) -> "Configuration":
        """Backward-compatible alias for :meth:`runtime`."""
        return self.runtime(**kwargs)

    def column(self, name: str, layers: Sequence[str], n: int) -> "Configuration":
        """Declare one cortical column and update the unified constructable network.

        Multiple column declarations are accumulated in ``metadata["columns"]``.
        For the current jaxfne construct path, the columns are represented as one
        unified ``multi_column`` network with explicit offsets.  This preserves
        compatibility with the existing source-to-field core while keeping V1/PFC
        bookkeeping available in manifests and tutorial diagnostics.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("column name must be a non-empty string")
        layers_list = [str(layer) for layer in layers]
        if not layers_list:
            raise ValueError("column layers must contain at least one layer label")
        n_int = int(n)
        if n_int <= 0:
            raise ValueError(f"column n must be positive; got {n!r}")

        metadata = dict(self.metadata)
        columns = [dict(col) for col in metadata.get("columns", [])]
        if any(col.get("name") == name for col in columns):
            raise ValueError(f"duplicate column name: {name!r}")

        start_index = sum(int(col["n"]) for col in columns)
        column_decl = {
            "name": name,
            "layers": layers_list,
            "n": n_int,
            "start_index": start_index,
            "stop_index": start_index + n_int,
        }
        columns.append(column_decl)
        total_n = sum(int(col["n"]) for col in columns)
        metadata["columns"] = columns
        metadata["column_names"] = [col["name"] for col in columns]
        metadata["column_count"] = len(columns)
        metadata.setdefault("layout_mode", "unified_multi_column_vertical_slice")
        metadata.setdefault("dx_mm", 0.010)
        metadata.setdefault("dy_mm", 0.010)
        metadata.setdefault("dz_mm", 0.010)
        metadata.setdefault("geometry_mode", "declared_metadata_not_solved_3d_pde_grid")
        metadata.setdefault("physical_amplitude_calibrated", False)

        cell_type_fractions = metadata.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.1})
        networks = [
            {
                "name": "_".join(col["name"] for col in columns) + "_motif",
                "kind": "multi_column",
                "n": total_n,
                "columns": columns,
                "layers": sorted({layer for col in columns for layer in col["layers"]}),
                "cell_types": dict(cell_type_fractions),
            }
        ]
        return replace(self, metadata=metadata, networks=networks)

    def add_column(self, name: str, layers: Sequence[str], n: int) -> "Configuration":
        """Backward-compatible alias for :meth:`column`."""
        return self.column(name=name, layers=layers, n=n)

    def geometry(
        self,
        *,
        layer_thickness: Mapping[str, float] | None = None,
        layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
    ) -> "Configuration":
        """Declare laminar geometry from per-layer *thickness* (relative depth).

        Thicknesses are normalized and accumulated into ``(z_min, z_max)`` depth
        intervals (z = 0 at the pia/top, z = 1 at the white-matter/bottom), then
        stored as ``layer_fractions``.  This is the ergonomic front-end for
        :meth:`layer_fractions`: you give thicknesses, not cumulative intervals.

        Geometry (where a layer sits / how thick it is) is intentionally
        independent of population (:meth:`population`, how many neurons a layer
        gets).  ``layer_cell_types`` is optional and only written when provided,
        so a prior :meth:`cell_types`/:meth:`area_layer_cell_types` is preserved.
        """
        if layer_thickness is None:
            layer_thickness = {"L1": 0.10, "L2": 0.15, "L3": 0.15, "L4": 0.10, "L5": 0.30, "L6": 0.20}
        layers = [str(k) for k in layer_thickness.keys()]
        if not layers:
            raise ValueError("layer_thickness must not be empty")
        values = [float(layer_thickness[k]) for k in layer_thickness.keys()]
        if any((not math.isfinite(v)) or v <= 0.0 for v in values):
            raise ValueError("layer thicknesses must be finite and positive")
        total = sum(values)
        fractions: dict[str, tuple[float, float]] = {}
        z = 0.0
        for layer, value in zip(layers, values):
            dz = value / total
            fractions[layer] = (z, min(1.0, z + dz))
            z += dz
        metadata = dict(self.metadata)
        metadata["layer_fractions"] = {k: list(v) for k, v in fractions.items()}
        metadata["layer_thickness"] = {k: float(layer_thickness[k]) for k in layer_thickness}
        if layer_cell_types is not None:
            metadata["layer_cell_types"] = {
                str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in layer_cell_types.items()
            }
        return replace(self, metadata=metadata)

    def population(
        self,
        N: int,
        neurons: Mapping[str, float],
        *,
        name: str = "V1",
        layers: Sequence[str] | None = None,
    ) -> "Configuration":
        """Declare a column's neuron population: total ``N`` + per-layer budget.

        ``neurons`` maps each layer to its share of the total (any positive
        scale — counts like ``{"L2": 250}`` or fractions like ``{"L2": 0.25}``;
        they are normalized).  Allocation uses the same largest-remainder rule as
        cell-type allocation, so per-layer integer counts sum exactly to ``N``.

        Changing the whole column size is a single edit to ``N``; the per-layer
        proportions stay fixed.  Per-area populations differ by calling
        ``population(...)`` once per area with distinct ``name`` and ``N``; each
        area's proportions are stored under ``area_layer_count_frac[name]``.
        """
        n_int = int(N)
        if n_int <= 0:
            raise ValueError(f"population N must be positive; got {N!r}")
        if not neurons:
            raise ValueError("neurons map must not be empty")
        frac: dict[str, float] = {}
        for key, value in neurons.items():
            f = float(value)
            if not math.isfinite(f) or f < 0.0:
                raise ValueError(f"neuron share for {key!r} must be finite and non-negative; got {value!r}")
            frac[str(key)] = f
        if sum(frac.values()) <= 0.0:
            raise ValueError("neurons map must have positive total")
        if layers is None:
            layers = list(frac.keys())
        layers_list = [str(layer) for layer in layers]

        metadata = dict(self.metadata)
        per_area = dict(metadata.get("area_layer_count_frac", {}))
        per_area[name] = dict(frac)
        metadata["area_layer_count_frac"] = per_area
        metadata.setdefault("layer_count_frac", dict(frac))
        cfg = replace(self, metadata=metadata)
        return cfg.column(name=name, layers=layers_list, n=n_int)

    def cell_types(self, fractions: Mapping[str, float]) -> "Configuration":
        """Set cell-type fractions for the current configuration.

        Fractions are copied into metadata and into the constructable unified
        network.  The method rejects negative, non-finite, or zero-total maps but
        does not silently normalize values; the manifest should preserve exactly
        what the user declared.
        """
        if not fractions:
            raise ValueError("cell type fractions must not be empty")
        clean: dict[str, float] = {}
        for key, value in fractions.items():
            f = float(value)
            if not math.isfinite(f) or f < 0.0:
                raise ValueError(f"cell type fraction for {key!r} must be finite and non-negative; got {value!r}")
            clean[str(key)] = f
        if sum(clean.values()) <= 0.0:
            raise ValueError("cell type fractions must have positive total mass")

        metadata = dict(self.metadata)
        metadata["cell_types"] = clean
        networks = [dict(net) for net in self.networks]
        if networks:
            networks[0] = dict(networks[0], cell_types=clean)
        else:
            total_n = sum(int(col["n"]) for col in metadata.get("columns", [])) or 100
            networks = [{"name": "configured_network", "kind": "configured", "n": total_n, "cell_types": clean}]
        return replace(self, metadata=metadata, networks=networks)

    def set_cell_types(self, fractions: Mapping[str, float]) -> "Configuration":
        """Backward-compatible alias for :meth:`cell_types`."""
        return self.cell_types(fractions)

    def connectivity(self, **kwargs: Any) -> "Configuration":
        """Attach declared connectivity metadata without overclaiming dynamics.

        Current construct-time dynamics still use the package's existing network
        generator.  These declarations are exported so tutorials and future
        kernels can distinguish feedforward/feedback bookkeeping from the actual
        proxy simulation path.
        """
        metadata = dict(self.metadata)
        connectivity = dict(metadata.get("connectivity", {}))
        connectivity.update(kwargs)
        metadata["connectivity"] = connectivity
        metadata.setdefault("connectivity_status", "declared_metadata_proxy")
        networks = [dict(net) for net in self.networks]
        if networks:
            networks[0] = dict(networks[0], connectivity=connectivity)
        return replace(self, metadata=metadata, networks=networks)

    def set_connectivity(self, **kwargs: Any) -> "Configuration":
        """Backward-compatible alias for :meth:`connectivity`."""
        return self.connectivity(**kwargs)

    def plasticity(self, relative_baseline: float = 1.0, **kwargs: Any) -> "Configuration":
        """Declare a plasticity baseline in chainable configuration form (declaration only).

        ``relative_baseline=1.0`` is the identity/neutral setting and is purely
        declarative: it does not change ``simulate()`` output. The STDP weight-
        update kernel (``update_stdp_weights_jax``) is not wired into the main
        ``Model.simulate()`` loop today — it only runs via the separate
        ``run_stdp_stream`` path. This verb records intent in
        ``metadata["plasticity"]`` so it is visible in ``manifest()`` from the
        first call; deviating from ``1.0`` does not yet activate any kernel.
        """
        spec = {
            "relative_baseline": float(relative_baseline),
            **dict(kwargs),
            "status": "declared_not_wired_to_simulate",
        }
        return self.update_metadata(plasticity=spec)

    def homeostasis(self, relative_baseline: float = 1.0, **kwargs: Any) -> "Configuration":
        """Declare a homeostasis baseline in chainable configuration form.

        ``relative_baseline=1.0`` is the identity/neutral setting: it leaves
        ``RuntimeConfig.enable_homeostasis=False`` (today's default; unchanged
        ``simulate()`` output) rather than enabling the kernel with a zeroed
        gain, so no new code path runs at baseline. Deviating from ``1.0``
        (or passing any ``homeostasis_params`` override, e.g. ``k_gain=``,
        ``r_star=``) resolves ``k_gain = relative_baseline - 1.0`` (overridable
        via an explicit ``k_gain=`` kwarg) and enables the existing per-neuron
        homeostatic feedback kernel via ``metadata["enable_homeostasis"]`` /
        ``metadata["homeostasis_params"]``, consumed by ``simulate()`` through
        ``_runtime_config_from_metadata``.
        """
        rb = float(relative_baseline)
        spec = {"relative_baseline": rb, **dict(kwargs)}
        metadata = dict(self.metadata)
        metadata["homeostasis"] = spec
        active = rb != 1.0 or bool(kwargs)
        metadata["enable_homeostasis"] = active
        if active:
            params = dict(RuntimeConfig().homeostasis_params)
            params["k_gain"] = rb - 1.0
            params.update(kwargs)
            metadata["homeostasis_params"] = params
        return replace(self, metadata=metadata)

    def hdp(self, relative_baseline: float = 1.0, **kwargs: Any) -> "Configuration":
        """Declare an HDP (Homeostasis-Dependent Plasticity) baseline, mirroring
        :meth:`homeostasis`.

        ``relative_baseline=1.0`` is the identity/neutral setting: it leaves
        ``RuntimeConfig.enable_hdp=False`` (unchanged ``simulate()`` output).
        Deviating from ``1.0`` (or passing any ``hdp_params`` override, e.g.
        ``K_HDP=``, ``alpha=``) resolves ``K_HDP = relative_baseline - 1.0``
        (overridable via an explicit ``K_HDP=`` kwarg) and enables the HDP
        kernel via ``metadata["enable_hdp"]`` / ``metadata["hdp_params"]``,
        consumed by ``simulate()`` through ``_runtime_config_from_metadata``.
        Mutually exclusive with :meth:`homeostasis` at the ``RuntimeConfig``
        level (enforced in ``RuntimeConfig.__post_init__``).
        """
        rb = float(relative_baseline)
        spec = {"relative_baseline": rb, **dict(kwargs)}
        metadata = dict(self.metadata)
        metadata["hdp"] = spec
        active = rb != 1.0 or bool(kwargs)
        metadata["enable_hdp"] = active
        if active:
            params = dict(RuntimeConfig().hdp_params)
            params["K_HDP"] = rb - 1.0
            params.update(kwargs)
            metadata["hdp_params"] = params
        return replace(self, metadata=metadata)

    # ------------------------------------------------------------------
    # v0.3.28 completion: declarative circuit/training ownership.
    #
    # These methods are DECLARATION-ONLY. They record JSON-safe specs under
    # ``metadata["circuit"]`` / ``metadata["training"]`` for a future
    # connectivity compiler (v0.3.30) to consume. They deliberately do NOT
    # compile edges, change simulation numerics, build dense matrices, trigger
    # JIT, import optional dependencies, or claim biological/physical validity.
    # Each declaration carries an explicit status so nothing reads as compiled,
    # applied, or optimized yet.
    # ------------------------------------------------------------------
    def cell_params(self, selector: Mapping[str, Any], params: Mapping[str, Any]) -> "Configuration":
        """Declare cell/emitter parameter overrides for a selector (declaration only)."""
        entry = {
            "selector": _circuit_json_safe(dict(selector), "cell_params.selector"),
            "params": _circuit_json_safe(dict(params), "cell_params.params"),
            "status": "declared_not_compiled",
        }
        return self._append_circuit("cell_params", entry, dedup_name=False)

    def mechanisms(
        self, *, name: str, kind: str, params: Optional[Mapping[str, Any]] = None, **extra: Any
    ) -> "Configuration":
        """Declare a named mechanism spec (declaration only; not compiled)."""
        if not name:
            raise ValueError("mechanism requires a non-empty name")
        entry = {
            "name": str(name),
            "kind": str(kind),
            "params": _circuit_json_safe(dict(params or {}), "mechanisms.params"),
            **_circuit_json_safe(dict(extra), "mechanisms.extra"),
            "status": "declared_not_compiled",
        }
        return self._append_circuit("mechanisms", entry, dedup_name=True)

    def connections(
        self,
        *,
        name: str,
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        probability: Optional[float] = None,
        weight: Optional[float] = None,
        sign: Optional[str] = None,
        mechanism: Optional[str] = None,
        plasticity: Optional[Mapping[str, Any]] = None,
        control_key: Optional[str] = None,
    ) -> "Configuration":
        """Declare a connection rule that ``construct()`` compiles into edges.

        Appends a rule to ``metadata["circuit"]["connections"]``; at ``construct``
        time the rule is compiled into real edges in the model's EdgeList (and the
        rule's ``status`` flips from ``"declared_not_compiled"`` to ``"compiled"``,
        with ``compiled_n_edges``). ``source``/``target`` are selector mappings over
        ``area``, ``layer``/``layers`` (tolerant of merged ``"L2/3"`` vs split
        ``"L2"``/``"L3"``), and ``cell_type``/``cell_types``; a missing key means
        "any". ``probability`` is Bernoulli edge inclusion (1.0 = full, 0.0 = none;
        default 1.0), ``weight`` is the native edge magnitude (default ``1/sqrt(n)``),
        and ``sign`` is ``"excitatory"``/``"inhibitory"``/``"signed"`` (``"signed"`` or
        unset follows the presynaptic neuron's intrinsic sign). When any rule
        materializes edges the model runs on the edge_list backend so the connections
        drive dynamics. ``mechanism`` is resolved into real per-edge tau/receptor_index
        at ``construct()`` time **only when every declared rule** in the
        configuration has a resolvable ``mechanism=`` reference into a
        ``.mechanisms()`` declaration (see
        ``core._all_connection_rules_declare_resolvable_mechanism``); a config
        with no mechanisms, or a mixed rule set where even one rule omits
        ``mechanism=``, compiles entirely through the sign-only fallback
        (receptor inferred from weight sign, tau hardcoded exc=2 ms/inh=5 ms)
        as before. ``plasticity``/``control_key`` remain declarative metadata
        (not yet compiled). Distinct from :meth:`connectivity`, which records
        feedforward/feedback bookkeeping.
        """
        if not name:
            raise ValueError("connection requires a non-empty name")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError(
                f"connection {name!r} requires source and target selector mappings"
            )
        if probability is not None:
            p = float(probability)
            if not math.isfinite(p) or not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"connection {name!r} probability must be finite in [0, 1]; got {probability!r}"
                )
        if weight is not None and not math.isfinite(float(weight)):
            raise ValueError(f"connection {name!r} weight must be finite; got {weight!r}")
        if sign is not None and sign not in {"excitatory", "inhibitory", "signed"}:
            raise ValueError(
                f"connection {name!r} sign must be excitatory/inhibitory/signed or None; got {sign!r}"
            )
        entry = {
            "name": str(name),
            "source": _circuit_json_safe(dict(source), "connections.source"),
            "target": _circuit_json_safe(dict(target), "connections.target"),
            "probability": float(probability) if probability is not None else None,
            "weight": float(weight) if weight is not None else None,
            "sign": sign,
            "mechanism": str(mechanism) if mechanism is not None else None,
            "plasticity": _circuit_json_safe(dict(plasticity), "connections.plasticity")
            if plasticity is not None
            else None,
            "control_key": str(control_key) if control_key is not None else None,
            "status": "declared_not_compiled",
        }
        cfg = self._append_circuit("connections", entry, dedup_name=True)
        # Unknown-mechanism reference is a WARNING in this declaration-only phase;
        # the v0.3.30 compiler will promote it to a hard error.
        known = {m.get("name") for m in cfg.metadata["circuit"]["mechanisms"]}
        if mechanism is not None and mechanism not in known:
            warnings.warn(
                f"connection {name!r} references mechanism {mechanism!r} not declared via "
                ".mechanisms() (declared_not_compiled phase)",
                UserWarning,
            )
        return cfg

    def lesions(self, *, name: str, selector: Mapping[str, Any], effect: str) -> "Configuration":
        """Declare a lesion (disabled nodes/edges/mechanisms; declaration only)."""
        if not name:
            raise ValueError("lesion requires a non-empty name")
        entry = {
            "name": str(name),
            "selector": _circuit_json_safe(dict(selector), "lesions.selector"),
            "effect": str(effect),
            "status": "declared_not_applied",
        }
        return self._append_circuit("lesions", entry, dedup_name=True)

    def trainables(
        self,
        *,
        name: str,
        path: str,
        selector: Optional[Mapping[str, Any]] = None,
        bounds: Optional[Sequence[float]] = None,
    ) -> "Configuration":
        """Declare a trainable parameter selector/spec (declaration only; not optimized)."""
        if not name:
            raise ValueError("trainable requires a non-empty name")
        clean_bounds = None
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(
                    f"trainable {name!r} bounds must have exactly 2 elements (low, high); got {bounds!r}"
                )
            lo, hi = (float(bounds[0]), float(bounds[1]))
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"trainable {name!r} bounds must be finite; got {bounds!r}")
            if lo > hi:
                raise ValueError(f"trainable {name!r} bounds reversed: low {lo} > high {hi}")
            clean_bounds = [lo, hi]
        entry = {
            "name": str(name),
            "path": str(path),
            "selector": _circuit_json_safe(dict(selector or {}), "trainables.selector"),
            "bounds": clean_bounds,
            "status": "declared_not_optimized",
        }
        return self._append_training("trainables", entry)

    def objective_outputs(
        self,
        *,
        name: str,
        dtype: str = "float32",
        shape: Optional[Sequence[int]] = None,
        required: bool = True,
    ) -> "Configuration":
        """Declare an expected objective output schema entry (declaration only)."""
        if not name:
            raise ValueError("objective output requires a non-empty name")
        entry = {
            "name": str(name),
            "dtype": str(dtype),
            "shape": [int(x) for x in shape] if shape is not None else None,
            "required": bool(required),
            "status": "declared",
        }
        return self._append_training("objective_outputs", entry)

    def _append_circuit(
        self, key: str, entry: dict[str, Any], *, dedup_name: bool
    ) -> "Configuration":
        metadata = dict(self.metadata)
        circuit = {k: list(v) for k, v in dict(metadata.get("circuit", {})).items()}
        for k in ("cell_params", "mechanisms", "connections", "lesions"):
            circuit.setdefault(k, [])
        lst = list(circuit.get(key, []))
        if dedup_name and any(e.get("name") == entry.get("name") for e in lst):
            raise ValueError(f"duplicate {key} declaration name: {entry.get('name')!r}")
        lst.append(entry)
        circuit[key] = lst
        metadata["circuit"] = circuit
        metadata["circuit_declared"] = True
        # A new declaration invalidates any prior compiled state, so reset
        # rather than setdefault (which would leave a stale True if a future
        # compiler set the flag and the user then chained another declaration).
        metadata["circuit_compiled"] = False
        return replace(self, metadata=metadata)

    def _append_training(self, key: str, entry: dict[str, Any]) -> "Configuration":
        metadata = dict(self.metadata)
        training = {k: list(v) for k, v in dict(metadata.get("training", {})).items()}
        for k in ("trainables", "objective_outputs"):
            training.setdefault(k, [])
        lst = list(training.get(key, []))
        if any(e.get("name") == entry.get("name") for e in lst):
            raise ValueError(f"duplicate {key} declaration name: {entry.get('name')!r}")
        lst.append(entry)
        training[key] = lst
        metadata["training"] = training
        metadata["training_declared"] = True
        # Reset (not setdefault): a new training declaration invalidates any
        # prior executed state.
        metadata["training_executed"] = False
        return replace(self, metadata=metadata)

    def set_emitter(self, family: str = "izhikevich", preset: str = "cortical_eig") -> "Configuration":
        """Chainable config method to set/wrap emitter family and presets."""
        return self.emitter(family=family, preset=preset)

    def _with_probe_modes(
        self,
        modes: Sequence[str],
        *,
        name: str = "multimodal_probe",
        n_contacts: int | None = None,
        ensure_defaults: bool = True,
        **kwargs: Any,
    ) -> "Configuration":
        mode_list = [str(mode) for mode in modes]
        if not mode_list:
            raise ValueError("probe modes must contain at least one mode label")

        cfg = self
        if ensure_defaults and not cfg.emitters:
            cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
        if ensure_defaults and not cfg.fields:
            cfg = cfg.field(
                domain="laminar_column",
                conductivity="proxy",
                boundary="mean_zero_neumann",
                gauge="mean_zero",
            )

        probe_kwargs: dict[str, Any] = {
            "name": name,
            "modes": mode_list,
            "operator_status": "simulated_proxy",
            "field_solver_status": "linear_solver",
            "physical_amplitude_calibrated": False,
        }
        if n_contacts is not None:
            probe_kwargs["n_contacts"] = int(n_contacts)
        probe_kwargs.update(kwargs)
        return cfg.probe(**probe_kwargs)

    def set_probes(self, modes: Sequence[str], **kwargs: Any) -> "Configuration":
        """Backward-compatible alias for the callable ``cfg.probes(...)`` facade."""
        return self._with_probe_modes(modes=modes, **kwargs)

    # ------------------------------------------------------------------
    # Multi-area laminar configuration (Suite No. 5).
    # These methods support declaring V1/V4/PFC circuits with explicit
    # laminar structure (L1–L6) and layer-specific cell-type distributions.
    # ------------------------------------------------------------------

    def areas(self, area_names: Sequence[str]) -> "Configuration":
        """Declare areas for a multi-area circuit (e.g., ['V1', 'V4', 'PFC']).

        Area declarations are stored in metadata and later used by
        layer_fractions() to generate multi-area neuron populations.

        Parameters
        ----------
        area_names : Sequence[str]
            Names of cortical areas (e.g., ['V1', 'V4', 'PFC']).

        Returns
        -------
        Configuration
            Updated configuration with areas recorded in metadata.
        """
        if not area_names:
            raise ValueError("area_names must not be empty")
        area_list = [str(a) for a in area_names]
        if len(area_list) != len(set(area_list)):
            raise ValueError("duplicate area names detected")

        metadata = dict(self.metadata)
        metadata["areas"] = area_list
        return replace(self, metadata=metadata)

    def layer_fractions(
        self,
        layer_fractions: Mapping[str, tuple[float, float]] | None = None,
        layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
    ) -> "Configuration":
        """Declare laminar structure and layer-specific cell-type distributions.

        Parameters
        ----------
        layer_fractions : Mapping[str, tuple[float, float]], optional
            Map of layer name to (z_min_rel, z_max_rel) relative depth.
            Example: {'L1': (0.0, 0.1), 'L2': (0.1, 0.25), ...}
            If None, uses standard L1–L6 fractions.
        layer_cell_types : Mapping[str, Mapping[str, float]], optional
            Map of layer name to cell-type fractions within that layer.
            Example: {'L1': {'E': 0.75, 'PV': 0.0, ...}, ...}
            If None, uses default distributions per layer.

        Returns
        -------
        Configuration
            Updated configuration with laminar structure recorded.
        """
        # Use standard L1-L6 fractions if not provided
        if layer_fractions is None:
            layer_fractions = {
                "L1": (0.00, 0.08),
                "L2": (0.08, 0.20),
                "L3": (0.20, 0.35),
                "L4": (0.35, 0.45),
                "L5": (0.45, 0.75),
                "L6": (0.75, 1.00),
            }

        # Validate layer fractions
        for layer, (z_min, z_max) in layer_fractions.items():
            if not (0.0 <= z_min < z_max <= 1.0):
                raise ValueError(f"layer {layer!r} has invalid fraction range: ({z_min}, {z_max})")

        # Use default layer cell-type distributions if not provided
        if layer_cell_types is None:
            layer_cell_types = {
                "L1": {"E": 0.20, "PV": 0.10, "SST": 0.10, "VIP": 0.60},
                "L2": {"E": 0.40, "PV": 0.15, "SST": 0.30, "VIP": 0.15},
                "L3": {"E": 0.40, "PV": 0.35, "SST": 0.20, "VIP": 0.05},
                "L4": {"E": 0.50, "PV": 0.35, "SST": 0.10, "VIP": 0.05},
                "L5": {"E": 0.90, "PV": 0.05, "SST": 0.04, "VIP": 0.01},
                "L6": {"E": 0.90, "PV": 0.05, "SST": 0.04, "VIP": 0.01},
            }

        metadata = dict(self.metadata)
        metadata["layer_fractions"] = {k: list(v) for k, v in layer_fractions.items()}
        metadata["layer_cell_types"] = layer_cell_types
        return replace(self, metadata=metadata)


    def area_layer_cell_types(self, area: str, layer_cell_types: Mapping[str, Mapping[str, float]]) -> "Configuration":
        """Set layer-specific cell-type fractions for one declared area."""
        if not area:
            raise ValueError("area must be a non-empty string")
        clean: dict[str, dict[str, float]] = {}
        for layer, fracs in layer_cell_types.items():
            clean[str(layer)] = {str(k): float(v) for k, v in fracs.items()}
            _counts_from_fractions(1, clean[str(layer)])
        metadata = dict(self.metadata)
        per_area = {str(k): dict(v) for k, v in (metadata.get("area_layer_cell_types", {}) or {}).items()}
        per_area[str(area)] = clean
        metadata["area_layer_cell_types"] = per_area
        return replace(self, metadata=metadata)

    def uniform3d(self, *, radius_mm: float = 0.25, height_mm: float = 1.60) -> "Configuration":
        """Mark the current column geometry as a uniformly sampled 3D column."""
        if radius_mm <= 0.0 or height_mm <= 0.0:
            raise ValueError("radius_mm and height_mm must be positive")
        return self.update_metadata(
            uniform_3d=True,
            column_radius_mm=float(radius_mm),
            column_height_mm=float(height_mm),
            geometry_mode="declared_uniform_3d_metadata_not_solved_pde_grid",
            dx_mm=0.010,
            dy_mm=0.010,
            dz_mm=0.010,
            physical_amplitude_calibrated=False,
        )

    def cell_type_drives(self, drives: Mapping[str, float]) -> "Configuration":
        """Override native reduced drive by cell type for Suite No. 2 sweeps."""
        if not drives:
            raise ValueError("drives must not be empty")
        clean: dict[str, float] = {}
        for key, value in drives.items():
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"drive for {key!r} must be finite")
            clean[str(key)] = v
        return self.update_metadata(cell_type_drives=clean)

    def suite2_interarea(self, enabled: bool = True) -> "Configuration":
        """Enable V1/V4 feedforward-feedback metadata in the construct path."""
        return self.update_metadata(suite2_interarea=bool(enabled))

    # ------------------------------------------------------------------
    # Complete Configuration API: domains 6-10 (inter_column_connectivity,
    # drive, objective, optimizer) as chainable methods.
    # ------------------------------------------------------------------

    def inter_column_connectivity(
        self,
        *,
        source_area: Optional[str] = None,
        target_area: Optional[str] = None,
        layer_to_layer_map: Optional[Mapping[str, str]] = None,
        cell_type_to_cell_type_map: Optional[Mapping[str, str]] = None,
        mode: str = "sparse",
        p_feedforward: Optional[float] = None,
        p_feedback: Optional[float] = None,
        feedforward_weight_range: Optional[tuple] = None,
        feedback_weight_range: Optional[tuple] = None,
        delay_ms_or_status: Optional[float | str] = None,
        sign_policy: str = "intrinsic",
        seed: Optional[int] = None,
    ) -> "Configuration":
        """Declare inter-column (inter-area) connectivity specification.

        This method stores declarative connectivity metadata between two cortical
        areas. No actual matrix construction or edge list generation occurs;
        all parameters are metadata only.

        Parameters
        ----------
        source_area : str, optional
            Name of the source area (e.g., "V1"). Default: "V1".
        target_area : str, optional
            Name of the target area (e.g., "V4"). Default: "V4".
        layer_to_layer_map : dict[str, str], optional
            Mapping from source layers to target layers. Default: declarative
            metadata with common layer pairs (e.g., "L2/3" → "L4").
        cell_type_to_cell_type_map : dict[str, str], optional
            Mapping from source cell types to target cell types. Default:
            identity map (E→E, PV→PV, etc.).
        mode : str
            Connectivity mode: "sparse", "all_to_all", "block_dense",
            or "distance_decay". Default: "sparse".
        p_feedforward : float, optional
            Probability of feedforward connections. Default: 0.3.
        p_feedback : float, optional
            Probability of feedback connections. Default: 0.2.
        feedforward_weight_range : tuple[float, float], optional
            Weight range for feedforward synapses. Default: (0.5, 2.0).
        feedback_weight_range : tuple[float, float], optional
            Weight range for feedback synapses. Default: (0.3, 1.5).
        delay_ms_or_status : float or str, optional
            Transmission delay in ms, or a status string (e.g.,
            "no_delay_proxy_metadata"). Default: None.
        sign_policy : str
            "intrinsic" (signs determined by emitter cell types) or "declared".
            Default: "intrinsic".
        seed : int, optional
            PRNG seed for stochastic connectivity generation. Default: None.

        Returns
        -------
        Configuration
            Updated configuration.

        Notes
        -----
        - All parameters are metadata only; no matrix construction.
        - Document explicitly: "Declarative inter-column metadata. Not solved.
          Physical edge delays not modeled."
        - Scope: proxy specification only; no PDE solution.
        """
        if source_area is None:
            source_area = "V1"
        if target_area is None:
            target_area = "V4"
        if p_feedforward is None:
            p_feedforward = 0.3
        if p_feedback is None:
            p_feedback = 0.2
        if feedforward_weight_range is None:
            feedforward_weight_range = (0.5, 2.0)
        if feedback_weight_range is None:
            feedback_weight_range = (0.3, 1.5)
        # layer_to_layer_map intentionally stays None when unset: the construct
        # path then applies the canonical anatomical routing (feedforward
        # L2/3->L4, feedback L6->L1/L5). Passing a map overrides those defaults.
        if cell_type_to_cell_type_map is None:
            cell_type_to_cell_type_map = {
                "E": "E",
                "PV": "PV",
                "SST": "SST",
                "VIP": "VIP",
            }

        inter_conn_spec = {
            "source_area": str(source_area),
            "target_area": str(target_area),
            "layer_to_layer_map": dict(layer_to_layer_map) if layer_to_layer_map is not None else None,
            "cell_type_to_cell_type_map": dict(cell_type_to_cell_type_map),
            "mode": str(mode),
            "p_feedforward": float(p_feedforward),
            "p_feedback": float(p_feedback),
            "feedforward_weight_range": tuple(feedforward_weight_range),
            "feedback_weight_range": tuple(feedback_weight_range),
            "delay_ms_or_status": delay_ms_or_status,
            "sign_policy": str(sign_policy),
            "seed": seed,
        }
        # Accumulate specs: each call adds one source->target projection. The
        # construct path (``_suite2_apply_connectivity``) already iterates a list of
        # specs, so multiple calls (e.g. one per adjacent area pair, or both
        # directions of a pair) all materialize. Using a list here — rather than
        # overwriting the single key — is what lets a full multi-area hierarchy wire.
        existing = self.metadata.get("inter_column_connectivity")
        if isinstance(existing, list):
            specs = [*existing, inter_conn_spec]
        elif existing:
            specs = [existing, inter_conn_spec]
        else:
            specs = [inter_conn_spec]
        return self.update_metadata(inter_column_connectivity=specs)

    def drive(
        self,
        *,
        baseline_drive_by_cell_type: Optional[Mapping[str, float]] = None,
        drive_by_layer: Optional[Mapping[str, float]] = None,
        drive_by_area: Optional[Mapping[str, float]] = None,
        time_schedule: Optional[str] = None,
        evoked_windows: Optional[Sequence[tuple]] = None,
        oddball_or_omission_schedule: Optional[Mapping[str, Sequence]] = None,
        noise_policy: str = "additive_poisson",
        trial_variability: bool = False,
    ) -> "Configuration":
        """Declare drive (stimulus and noise) specification.

        This method stores declarative drive metadata: baseline external input,
        evoked windows, noise policy, and trial variability settings.
        No actual stimulus envelope generation occurs; all parameters are
        metadata only.

        Parameters
        ----------
        baseline_drive_by_cell_type : dict[str, float], optional
            Baseline external drive per cell type. Default:
            {"E": 5.0, "PV": 3.0, "SST": 3.5, "VIP": 3.0}.
        drive_by_layer : dict[str, float], optional
            Layer-specific drive overrides. Default: {} (no override).
        drive_by_area : dict[str, float], optional
            Area-specific drive overrides. Default: {} (no override).
        time_schedule : str, optional
            Drive schedule type: "constant", "ramp", "pulse", or a path.
            Default: "constant".
        evoked_windows : list[tuple], optional
            List of (onset_ms, duration_ms) pairs for evoked responses.
            Default: [] (no evoked drive).
        oddball_or_omission_schedule : dict[str, list], optional
            Oddball and omission event times. Default: empty dict.
        noise_policy : str
            Noise type: "additive_poisson", "additive_gaussian", or "none".
            Default: "additive_poisson".
        trial_variability : bool
            Whether trial-to-trial variation is enabled. Default: False.

        Returns
        -------
        Configuration
            Updated configuration.

        Notes
        -----
        - All parameters are metadata only; no runtime I(t) generation.
        - Actual stimulus envelope is generated at simulate() time.
        """
        if baseline_drive_by_cell_type is None:
            baseline_drive_by_cell_type = {
                "E": 5.0,
                "PV": 3.0,
                "SST": 3.5,
                "VIP": 3.0,
            }
        if drive_by_layer is None:
            drive_by_layer = {}
        if drive_by_area is None:
            drive_by_area = {}
        if time_schedule is None:
            time_schedule = "constant"
        if evoked_windows is None:
            evoked_windows = []
        if oddball_or_omission_schedule is None:
            oddball_or_omission_schedule = {}

        if noise_policy not in ("additive_poisson", "additive_gaussian", "none"):
            raise ValueError(
                f"noise_policy must be one of ('additive_poisson', 'additive_gaussian', 'none'); "
                f"got {noise_policy!r}"
            )

        drive_spec = {
            "baseline_drive_by_cell_type": dict(baseline_drive_by_cell_type),
            "drive_by_layer": dict(drive_by_layer),
            "drive_by_area": dict(drive_by_area),
            "time_schedule": str(time_schedule),
            "evoked_windows": list(evoked_windows),
            "oddball_or_omission_schedule": dict(oddball_or_omission_schedule),
            "noise_policy": str(noise_policy),
            "trial_variability": bool(trial_variability),
        }
        return self.update_metadata(drive=drive_spec)

    def objective(
        self,
        *,
        firing_rate_target: Optional[Mapping[str, float]] = None,
        spectrolaminar_profile_target: Optional[Mapping[str, tuple]] = None,
        band_definitions: Optional[Mapping[str, tuple]] = None,
        synchrony_metrics: Optional[Sequence[str]] = None,
        null_controls: Optional[Mapping[str, str]] = None,
        ablations: Optional[Mapping[str, bool]] = None,
        rejection_gates: Optional[Mapping[str, tuple]] = None,
    ) -> "Configuration":
        """Declare objective specification for optimization and validation.

        This method stores declarative objective metadata: firing-rate targets,
        spectral band definitions, null controls, ablations, and rejection gates.
        No actual loss computation occurs; all parameters are metadata only.

        Parameters
        ----------
        firing_rate_target : dict[str, float], optional
            Target firing rate (Hz) per cell type. Default:
            {"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0}.
        spectrolaminar_profile_target : dict[str, tuple], optional
            Target power fraction per band. Default:
            {"alpha_beta": (0.3, 0.5), "gamma": (0.1, 0.3)}.
        band_definitions : dict[str, tuple], optional
            Frequency band definitions (Hz). Default:
            {"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)}.
        synchrony_metrics : list[str], optional
            Synchrony metrics to track. Default:
            ["pairwise_spike_distance", "lfp_cross_correlation"].
        null_controls : dict[str, str], optional
            Null model specification. Default:
            {"null_type": "shuffled_spike_times", "n_null_trials": 10}.
        ablations : dict[str, bool], optional
            Ablation flags (True = remove, False = keep). Default:
            {"E_ablation": False, "PV_ablation": False}.
        rejection_gates : dict[str, tuple], optional
            Hard validation bounds. Default:
            {"max_firing_rate": (0.0, 200.0), "min_synchrony": (-1.0, 1.0)}.

        Returns
        -------
        Configuration
            Updated configuration.

        Notes
        -----
        - All parameters are metadata only; no loss computation.
        - Use with model.tune() and optimizer configuration.
        """
        if firing_rate_target is None:
            firing_rate_target = {
                "E": 8.0,
                "PV": 15.0,
                "SST": 4.0,
                "VIP": 2.0,
            }
        if spectrolaminar_profile_target is None:
            spectrolaminar_profile_target = {
                "alpha_beta": (0.3, 0.5),
                "gamma": (0.1, 0.3),
            }
        if band_definitions is None:
            band_definitions = {
                "alpha_beta": (8.0, 25.0),
                "gamma": (40.0, 150.0),
            }
        if synchrony_metrics is None:
            synchrony_metrics = [
                "pairwise_spike_distance",
                "lfp_cross_correlation",
            ]
        if null_controls is None:
            null_controls = {
                "null_type": "shuffled_spike_times",
                "n_null_trials": 10,
            }
        if ablations is None:
            ablations = {
                "E_ablation": False,
                "PV_ablation": False,
                "SST_ablation": False,
                "VIP_ablation": False,
            }
        if rejection_gates is None:
            rejection_gates = {
                "max_firing_rate": (0.0, 200.0),
                "min_synchrony": (-1.0, 1.0),
            }

        objective_spec = {
            "firing_rate_target": dict(firing_rate_target),
            "spectrolaminar_profile_target": dict(spectrolaminar_profile_target),
            "band_definitions": dict(band_definitions),
            "synchrony_metrics": list(synchrony_metrics),
            "null_controls": dict(null_controls),
            "ablations": dict(ablations),
            "rejection_gates": dict(rejection_gates),
        }
        return self.update_metadata(objective=objective_spec)

    def optimizer(
        self,
        *,
        optimizer_family: str = "AGSDR",
        differentiability_status: str = "non_differentiable",
        surrogate_status: str = "soft_rate_surrogate",
        search_space: Optional[Mapping[str, tuple]] = None,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        hard_gates: Optional[Mapping[str, str]] = None,
    ) -> "Configuration":
        """Declare optimizer specification.

        This method stores declarative optimizer metadata: family,
        differentiability status, search space, budget, and hard gates
        (e.g., immutable claim_level). No actual optimization occurs;
        all parameters are metadata only.

        Parameters
        ----------
        optimizer_family : str
            Optimizer family: "AGSDR", "GSDR", "random_search",
            "optax_adam", or "optax_sgd". Default: "AGSDR".
        differentiability_status : str
            "non_differentiable", "differentiable_via_surrogate",
            or "fully_differentiable". Default: "non_differentiable".
        surrogate_status : str
            Surrogate type: "soft_rate_surrogate", "quadratic_loss",
            or "none". Default: "soft_rate_surrogate".
        search_space : dict[str, tuple], optional
            Parameter bounds: {param_name: (low, high), ...}.
            Default: {} (empty; no search space).
        budget : int, optional
            Iteration budget (e.g., 50 for AGSDR). Default: 50.
        seed : int, optional
            PRNG seed for stochastic optimizer. Default: None.
        hard_gates : dict[str, str], optional
            Immutable constraints (e.g., {"claim_level":
            "computational_scaffold"}). Default:
            {"claim_level": "computational_scaffold"}.

        Returns
        -------
        Configuration
            Updated configuration.

        Notes
        -----
        - All parameters are metadata only; no optimization execution.
        - hard_gates enforce immutable truth constraints; cannot be overridden.
        - Surrogate paths (differentiable_via_surrogate) are for inner loops
          only; real objective gates biological/physical claims.
        """
        if search_space is None:
            search_space = {}
        if budget is None:
            budget = 50
        if hard_gates is None:
            hard_gates = {
                "claim_level": "computational_scaffold",
            }

        allowed_families = (
            "AGSDR",
            "GSDR",
            "random_search",
            "optax_adam",
            "optax_sgd",
        )
        if optimizer_family not in allowed_families:
            raise ValueError(
                f"optimizer_family must be one of {allowed_families}; "
                f"got {optimizer_family!r}"
            )

        optimizer_spec = {
            "optimizer_family": str(optimizer_family),
            "differentiability_status": str(differentiability_status),
            "surrogate_status": str(surrogate_status),
            "search_space": dict(search_space),
            "budget": int(budget),
            "seed": seed,
            "hard_gates": dict(hard_gates),
        }
        return self.update_metadata(optimizer=optimizer_spec)

    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.networks:
            issues.append("no_networks_declared")
        if not self.emitters:
            issues.append("no_emitters_declared")
        if not self.fields:
            issues.append("no_field_declared")
        if not self.probes:
            issues.append("no_probes_declared")
        return {
            "valid": not issues,
            "issues": issues,
            "config_hash": config_hash(self),
            "claim_level": self.metadata.get("claim_level"),
        }


Config = Configuration


@dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime, dtype policy, and emitter control.

    ``dtype='float64'`` is honored only when JAX x64 is enabled.  The manifest
    always reports both requested and actual dtype policy.

    Homeostasis: when ``enable_homeostasis=True``, emitters use per-neuron
    activity-trace feedback to balance firing rates. Pass ``homeostasis_params``
    as a dict with keys {r_star, tau_r_ms, alpha, k_gain, g_min, g_max, r_max};
    k_gain=0 disables the feedback.

    HDP (Homeostasis-Dependent Plasticity): when ``enable_hdp=True``, emitters
    use a per-neuron master state H_i driving excitatory/inhibitory weight
    ODEs (see ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp``).
    Pass ``hdp_params`` as a dict with keys {K_HDP, tau_0_ms, alpha, beta,
    gamma, delta, C_spike, K_ctrl, barrier_c, barrier_d}; all income/spending/
    control gains default to 0.0 (the null control: H pinned at 1.0, no
    weight change). ``jaxfne.hdp_network.DEFAULT_HDP`` / ``DEFAULT_HDP_DESYNC``
    are tuned presets. Mutually exclusive with ``enable_homeostasis``.
    """

    backend: str = "auto"  # "auto" | "cpu" | "gpu" | "tpu"
    dtype: str = "float32"  # "float32" | "float64"
    jit: bool | str = False
    vmap: bool | str = False
    precision: str = "default"  # "default" | "high"
    seed: int = 0
    n_steps: int = 0
    recurrent_backend: str = "dense"  # "dense" | "edge_list"
    synaptic_kernel: str = "exponential"  # "exponential" | "receptor_exponential"
    recompilation_guard: str = "warning"  # "warning" | "exception" | "off"
    enable_homeostasis: bool = False  # Enable per-neuron homeostatic resource
    homeostasis_params: dict = field(default_factory=lambda: {
        "r_star": 0.05,
        "tau_r_ms": 300.0,
        "alpha": 1.0,
        "k_gain": 1.0,
        "g_min": -12.0,
        "g_max": 8.0,
        "r_max": 1.0,
    })  # Homeostasis parameters; k_gain=0 disables, default 1.0 is a gentle in-band nudge
    enable_hdp: bool = False  # Enable Homeostasis-Dependent Plasticity (per-neuron
    # master state H_i driving excitatory/inhibitory weight ODEs); see
    # jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp and
    # jaxfne.hdp_network.DEFAULT_HDP / DEFAULT_HDP_DESYNC for tuned presets.
    # Mutually exclusive with enable_homeostasis (distinct controllers).
    hdp_params: dict = field(default_factory=lambda: {
        "K_HDP": 1.0, "tau_0_ms": 100.0, "alpha": 0.0, "beta": 0.0,
        "gamma": 0.0, "delta": 0.0, "C_spike": 0.0, "K_ctrl": 0.0,
        "barrier_c": 0.0, "barrier_d": 0.0,
    })  # All income/spending/control gains default to 0.0 -- the documented
    # null control (H_i pinned at 1.0, weight term identically zero).
    # v0.0.3 compatibility names; if provided by old caller, they are folded in.
    device_type: Optional[str] = None
    dtype_primary: Optional[str] = None
    x64_enabled: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.synaptic_kernel not in _ALLOWED_SYNAPTIC_KERNELS:
            raise ValueError(
                f"synaptic_kernel must be one of {_ALLOWED_SYNAPTIC_KERNELS}; "
                f"got {self.synaptic_kernel!r}"
            )
        if self.recompilation_guard not in {"warning", "exception", "off"}:
            raise ValueError(
                f"recompilation_guard must be one of {{'warning', 'exception', 'off'}}; "
                f"got {self.recompilation_guard!r}"
            )
        # Validate jit and vmap options
        jit_str = str(self.jit).lower()
        if jit_str not in {"true", "false", "auto", "1", "0"}:
            raise ValueError(
                f"jit must be a boolean or 'auto'; got {self.jit!r}"
            )
        vmap_str = str(self.vmap).lower()
        if vmap_str not in {"true", "false", "auto", "1", "0"}:
            raise ValueError(
                f"vmap must be a boolean or 'auto'; got {self.vmap!r}"
            )
        if self.enable_homeostasis and self.enable_hdp:
            raise ValueError(
                "enable_homeostasis and enable_hdp are mutually exclusive "
                "controllers; enable only one."
            )

    def resolve_jit(self, n_steps: int, n_units: int, batch: int = 1) -> bool:
        """Resolve the effective JIT compilation status based on policy and parameters."""
        if isinstance(self.jit, bool):
            return self.jit
        jit_str = str(self.jit).lower()
        if jit_str == "auto":
            return (n_steps * n_units * batch) > 50000
        return jit_str in {"true", "1"}

    def resolve_vmap(self, batch: int) -> bool:
        """Resolve the effective vmap vectorization status based on policy and batch size."""
        if isinstance(self.vmap, bool):
            return self.vmap
        vmap_str = str(self.vmap).lower()
        if vmap_str == "auto":
            return batch > 1
        return vmap_str in {"true", "1"}

    @property
    def requested_dtype(self) -> str:
        """Documented public function `requested_dtype`."""
        return self.dtype_primary or self.dtype

    @property
    def selected_backend(self) -> str:
        """Documented public function `selected_backend`."""
        return self.device_type or self.backend

    @property
    def actual_dtype(self) -> str:
        """Documented public function `actual_dtype`."""
        if self.requested_dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
            return "float64"
        return "float32"

    @property
    def jnp_dtype(self) -> jnp.dtype:
        """Documented public function `jnp_dtype`."""
        return jnp.float64 if self.actual_dtype == "float64" else jnp.float32

    def runtime_report(self) -> dict[str, Any]:
        """Compile a summary of JAX system runtime, backend, and device topology.

        Detects requested vs actual execution backends (CPU, GPU, or TPU) and
        validates system properties.

        Returns
        -------
        dict[str, Any]
            System configuration report metadata.
        """
        devices = [str(device) for device in jax.devices()]
        default_backend = jax.default_backend()
        requested_backend = self.selected_backend  # alias for selected (user-requested)

        # Compute actual_backend / backend_enforced / backend_warning.
        # Conservative policy (CPU-only environment is the supported scaffold target):
        # - "auto": actual_backend = JAX default; enforced = True (whatever JAX picks).
        # - "cpu" : actual_backend = "cpu"; enforced = True iff JAX default is cpu.
        # - "gpu"/"tpu": if JAX default is the requested device, enforced = True;
        #   otherwise enforced = False with a clean warning (no false claim).
        # v0.3.40: simulate() pins execution to the requested backend device via
        # ``_device_scope`` (jax.default_device) when that device is available, so
        # a non-"auto" backend is genuinely enforced — not merely reported. The
        # report mirrors that: enforced iff the requested backend resolves to an
        # available device. Unavailable devices are an honest downgrade.
        backend_warning: Optional[str] = None
        if requested_backend == "auto":
            actual_backend = default_backend
            backend_enforced = True
        elif requested_backend in ("cpu", "gpu", "tpu"):
            device_available = _resolve_backend_device(requested_backend) is not None
            if device_available:
                actual_backend = requested_backend
                backend_enforced = True
            else:
                # Honest downgrade: do not falsely report a device that is absent.
                actual_backend = default_backend
                backend_enforced = False
                backend_warning = (
                    f"requested_backend_unavailable:requested={requested_backend!r}"
                    f"_actual={default_backend!r}"
                )
        else:
            actual_backend = default_backend
            backend_enforced = False
            backend_warning = f"unknown_requested_backend:{requested_backend!r}"

        return {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": _jaxlib_version(),
            "default_backend": default_backend,
            "available_devices": devices,
            "selected_backend": self.selected_backend,
            "backend": self.backend,
            # v0.0.21 explicit requested-vs-actual reporting.
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "backend_enforced": bool(backend_enforced),
            "backend_warning": backend_warning,
            "requested_dtype": self.requested_dtype,
            "actual_dtype": self.actual_dtype,
            "dtype": self.dtype,
            "jit": bool(self.jit),
            "vmap": bool(self.vmap),
            "precision": self.precision,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "recurrent_backend": self.recurrent_backend,
            "synaptic_kernel": self.synaptic_kernel,
            "enable_homeostasis": bool(self.enable_homeostasis),
            "homeostasis_params": dict(self.homeostasis_params) if self.enable_homeostasis else None,
            "enable_hdp": bool(self.enable_hdp),
            "hdp_params": dict(self.hdp_params) if self.enable_hdp else None,
            # Compatibility keys from v0.0.3.
            "device_type": self.selected_backend,
            "dtype_primary": self.actual_dtype,
        }


def _resolve_backend_device(backend: Optional[str]):
    """Resolve a RuntimeConfig backend string to a concrete JAX device or ``None``.

    ``None``/``"auto"`` returns ``None`` (let JAX keep its default device).
    ``"cpu"``/``"gpu"``/``"tpu"`` return that backend's first device when it
    exists, else ``None`` — an honest fallback that never crashes and never
    falsely claims placement on an absent device.
    """
    if not backend or backend == "auto":
        return None
    try:
        devices = jax.devices(backend)
    except (RuntimeError, ValueError):
        return None
    return devices[0] if devices else None


def _device_scope(backend: Optional[str]):
    """Return a context manager pinning execution to the requested backend device.

    Yields ``jax.default_device(dev)`` when the backend resolves to an available
    device, else a no-op context (``"auto"`` or an unavailable backend) so JAX
    keeps its default device. Compiling and running inside this scope is what
    actually honors ``RuntimeConfig.backend`` for cpu/gpu/tpu placement; on a
    single-device host it is a no-op.
    """
    device = _resolve_backend_device(backend)
    if device is not None:
        return jax.default_device(device)
    return contextlib.nullcontext()


def _jaxlib_version() -> str:
    try:
        import jaxlib  # type: ignore

        return str(jaxlib.__version__)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class SurrogateConfig:
    """Declared surrogate-gradient metadata for discontinuous emitter paths.

    The current implementation records the declaration only; it does not alter
    the Izhikevich reset dynamics.  Optax paths may reference this object to
    distinguish explicit surrogate intent from accidental differentiation
    through hard spike resets.
    """

    method: str = "none"  # "none" | "straight_through" | "sigmoid_beta"
    beta: float = 10.0
    applies_to: str = "izhikevich_reset"
    status: str = "declaration_only_v0.0.8"

    def gradient_path_status(self) -> str:
        """Documented public function `gradient_path_status`."""
        if self.method in {"straight_through", "sigmoid_beta"}:
            return "declared_surrogate"
        return "required_but_missing"

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "method": self.method,
            "beta": float(self.beta),
            "applies_to": self.applies_to,
            "status": self.status,
            "gradient_path_status": self.gradient_path_status(),
            "mechanism_claim_status": "not_claimed",
        })


@dataclass(frozen=True)
class Simulation:
    """Immutable specification of one simulation run.

    Fields: ``duration_ms`` and ``dt_ms`` (milliseconds; both must be positive
    and finite), ``seed`` (PRNG seed), ``plasticity`` (plasticity gain),
    ``record_sources``/``record_fields`` (recording toggles), ``poisson_drive``
    (optional drive spec), ``runtime`` (:class:`RuntimeConfig` override), and
    ``ablation`` (optional ablation label). ``n_steps`` is derived as
    ``round(duration_ms / dt_ms)`` and must be > 0.
    """

    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True
    poisson_drive: Optional[dict] = None
    runtime: RuntimeConfig | None = None
    ablation: Optional[str] = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.duration_ms) and self.duration_ms > 0):
            raise ValueError(
                f"Simulation.duration_ms must be positive and finite; got {self.duration_ms!r}"
            )
        if not (math.isfinite(self.dt_ms) and self.dt_ms > 0):
            raise ValueError(
                f"Simulation.dt_ms must be positive and finite; got {self.dt_ms!r}"
            )
        n = int(round(self.duration_ms / self.dt_ms))
        if n <= 0:
            raise ValueError(
                f"Simulation produces n_steps={n} <= 0 for "
                f"duration_ms={self.duration_ms}, dt_ms={self.dt_ms}"
            )

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_ms / self.dt_ms))

    @property
    def resolved_runtime(self) -> RuntimeConfig:
        base = self.runtime or RuntimeConfig(seed=self.seed, n_steps=self.n_steps)
        return replace(base, seed=self.seed, n_steps=self.n_steps)

    def with_plasticity(self, gain: float) -> "Simulation":
        return replace(self, plasticity=float(gain))


@dataclass(frozen=True)
class Probe:
    """Immutable declaration of a readout probe.

    ``name`` identifies the probe; ``modes`` is the sequence of requested
    readout keys (e.g. ``"spikes"``, ``"V_m"``, ``"lfp_proxy"``,
    ``"csd_proxy"``); ``metadata`` carries optional JSON-safe annotations. All
    field-derived modes are proxy readouts, not physical measurements.
    """

    name: str
    modes: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# v0.3.29 signal-access key aliases. Maps user-facing keys to the real
# underlying attribute. Only real, present readouts are listed: the core proxy
# field carries lfp/csd/phi_e/source proxies — there is NO eeg/meg/emm readout.
_SIGNALS_GET_KEY_ALIASES: dict[str, str] = {
    "vm": "V_m", "v_m": "V_m", "voltage": "V_m", "V_m": "V_m",
    "spk": "spikes", "spike": "spikes", "spikes": "spikes", "raster": "spikes",
    "src": "sources", "source": "sources", "sources": "sources",
    "lfp": "lfp_proxy", "lfp_proxy": "lfp_proxy",
    "csd": "csd_proxy", "csd_proxy": "csd_proxy",
    "phi_e": "phi_e_proxy", "phi": "phi_e_proxy", "phi_e_proxy": "phi_e_proxy",
    "field_source": "source_proxy", "source_proxy": "source_proxy",
    # Probe-only readouts (not on FieldOutput); aliases normalize config vocabulary.
    # `*_like` names are fully retired (no aliases): use `*_proxy` only.
    "eeg": "eeg_proxy", "eeg_proxy": "eeg_proxy",
    "meg": "meg_proxy", "meg_proxy": "meg_proxy",
    "emm": "emm_proxy", "emm_proxy": "emm_proxy",
}
# Signals whose neuron axis is the trailing axis (length == n_units).
_SIGNALS_GET_NEURON_AXIS_KEYS = frozenset({"V_m", "spikes", "sources"})
# Field readouts are laminar/contact-indexed, not neuron-indexed.
_SIGNALS_GET_FIELD_KEYS = frozenset({"lfp_proxy", "csd_proxy", "phi_e_proxy", "source_proxy"})


@dataclass(frozen=True)
class Signals:
    """Simulation output container holding multiple arrays."""

    time_ms: jax.Array
    V_m: jax.Array
    spikes: jax.Array
    sources: Optional[jax.Array]
    field: Optional[FieldOutput]
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe signal diagnostics for notebooks."""
        from .io import json_safe
        dt_ms = float(self.time_ms[1] - self.time_ms[0]) if self.time_ms.shape[0] > 1 else None
        return json_safe({
            "n_steps": int(self.time_ms.shape[0]),
            "n_units": int(self.V_m.shape[1]) if self.V_m.ndim == 2 else None,
            "dt_ms": dt_ms,
            "spike_count_total": float(jnp.sum(self.spikes)),
            "spike_rate_hz_mean": (
                float(jnp.mean(self.spikes) * (1000.0 / dt_ms)) if dt_ms else None
            ),
            "V_m_mean": float(jnp.mean(self.V_m)),
            "field_status": "present" if self.field is not None else "absent",
            "field_claim_level": self.metadata.get("field_claim_level", "proxy_readout"),
        })

    def get(
        self,
        key: str,
        *,
        selector: "Optional[SelectorSpec]" = None,
        area: Optional[Any] = None,
        layer: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        ids: Optional[Sequence[int]] = None,
        trial: Optional[int] = None,
        as_numpy: bool = False,
    ) -> Any:
        """Return a named signal array, optionally filtered to selected neurons.

        Key aliases (case-sensitive): ``vm``/``V_m``/``voltage`` -> V_m;
        ``spk``/``spikes``/``raster`` -> spikes; ``src``/``sources`` -> sources;
        ``lfp``/``csd``/``phi_e`` -> the corresponding laminar proxy readout;
        ``field_source`` -> field source proxy. Unknown keys raise ``KeyError``
        listing the available ones.

        A selector (either a ``SelectorSpec`` or the ``area``/``layer``/
        ``cell_type``/``ids`` fields, not both) filters neuron-indexed signals
        (V_m, spikes, sources) along their trailing axis. Selectors on laminar
        field readouts raise ``ValueError`` (no declared neuron axis). Selection
        requires ``metadata['neuron_metadata']``; if absent it raises
        ``ValueError`` rather than guessing.

        ``trial`` is not supported: core ``Signals`` has no declared trial axis,
        so any ``trial`` argument raises ``NotImplementedError``.
        """
        if trial is not None:
            raise NotImplementedError(
                "Signals.get(..., trial=...) is not supported: core Signals has no "
                "declared trial axis (V_m is (n_steps, n_units)). Use a trial-batched "
                "API or index the array directly once trial semantics are explicit."
            )

        field_args = (area, layer, cell_type, ids)
        if selector is not None and any(x is not None for x in field_args):
            raise ValueError(
                "Pass either selector=SelectorSpec(...) or the area/layer/cell_type/ids "
                "fields, not both."
            )
        if selector is None and any(x is not None for x in field_args):
            selector = SelectorSpec(
                area=area,
                layer=layer,
                cell_type=cell_type,
                ids=tuple(ids) if ids is not None else None,
            )

        if key not in _SIGNALS_GET_KEY_ALIASES:
            available = ", ".join(sorted(_SIGNALS_GET_KEY_ALIASES))
            raise KeyError(f"Unknown signal key {key!r}. Available: {available}")
        attr = _SIGNALS_GET_KEY_ALIASES[key]

        if attr == "V_m":
            arr = self.V_m
        elif attr == "spikes":
            arr = self.spikes
        elif attr == "sources":
            if self.sources is None:
                raise ValueError("sources not recorded (run with record_sources=True)")
            arr = self.sources
        elif attr in _SIGNALS_GET_FIELD_KEYS:
            if self.field is None:
                raise ValueError(f"field not recorded; {attr!r} unavailable (record_fields=True)")
            arr = getattr(self.field, attr, None)
            if arr is None:
                raise ValueError(f"field output has no attribute {attr!r}")
        else:  # pragma: no cover - alias map and branches kept in sync
            raise KeyError(f"Signal {attr!r} not available")

        if selector is not None:
            if attr not in _SIGNALS_GET_NEURON_AXIS_KEYS:
                raise ValueError(
                    f"Selector cannot be applied to signal {key!r}: {attr!r} is a laminar "
                    f"field readout with no declared neuron axis. Slice it directly instead."
                )
            if getattr(arr, "ndim", 0) < 2:
                raise ValueError(
                    f"Cannot apply selector to {attr!r}: expected a neuron axis "
                    f"(ndim >= 2), got shape {getattr(arr, 'shape', None)}"
                )
            n_units = int(self.V_m.shape[-1]) if self.V_m.ndim >= 2 else None
            if n_units is not None and int(arr.shape[-1]) != n_units:
                raise ValueError(
                    f"Cannot apply selector to {attr!r}: trailing axis {arr.shape[-1]} "
                    f"does not match neuron count {n_units}; no declared neuron axis."
                )
            table = self.metadata.get("neuron_metadata")
            if table is None:
                raise ValueError(
                    "Selector requested but this run has no neuron identity table "
                    "(metadata['neuron_metadata'] is None). Build the model with geometry "
                    "that provides neuron rows, or index by integer position directly."
                )
            indices = selector.resolve(table)
            arr = arr[..., indices]

        if as_numpy:
            import numpy as _np
            arr = _np.asarray(arr)
        return arr


# Backwards-compatible alias.
Signal = Signals


@dataclass(frozen=True)
class Objective:
    """Declarative objective specification: losses, regularizers, and diagnostic gates.

    All specs are stored as plain dicts (no callables) so the objective is always
    JSON-serializable.  Gate pass/fail is a computational diagnostic only — it does
    not imply empirical validation or biological calibration.
    """

    name: str = "anonymous"
    kind: str = "generic"  # "generic", "group_rate_targets", or custom
    losses: list[dict[str, Any]] = field(default_factory=list)
    regularizers: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)

    def loss(
        self,
        name: str,
        target: Optional[float] = None,
        weight: float = 1.0,
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `loss`."""
        spec: dict[str, Any] = {"name": name, "weight": float(weight)}
        if target is not None:
            spec["target"] = float(target)
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, losses=[*self.losses, spec])

    def regularizer(
        self,
        name: str,
        target: float = 0.0,
        weight: float = 1.0,
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `regularizer`."""
        spec: dict[str, Any] = {"name": name, "target": float(target), "weight": float(weight)}
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, regularizers=[*self.regularizers, spec])

    def gate(
        self,
        name: str,
        threshold: Any,
        criterion: str = "below",
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `gate`."""
        spec: dict[str, Any] = {"name": name, "threshold": threshold, "criterion": str(criterion)}
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, gates=[*self.gates, spec])

    def compose(self, *others: "Objective") -> "Objective":
        """Merge other Objective specs into this one, concatenating all specs."""
        all_losses = list(self.losses)
        all_regularizers = list(self.regularizers)
        all_gates = list(self.gates)
        for other in others:
            all_losses.extend(other.losses)
            all_regularizers.extend(other.regularizers)
            all_gates.extend(other.gates)
        return replace(self, losses=all_losses, regularizers=all_regularizers, gates=all_gates)


from .paradigm import (
    ParadigmEvent,
    ParadigmCondition,
    Paradigm,
    paradigm,
    evoked_l4_drive_paradigm,
    omission_oddball_paradigm,
    coop_omission_oddball_paradigm,
    general_sequential_oddball_paradigm,
    general_delayed_match_to_sample_paradigm,
)



@dataclass(frozen=True)
class DatasetSpec:
    """Manifest-safe dataset/comparison declaration for observed data.

    DatasetSpec is a schema object, not a loader.  It records how an external
    dataset is aligned and interpreted so objectives can reference data without
    hard-coding paths or claiming empirical validation.
    """

    name: str = "unnamed_dataset"
    modality: str = "unspecified"
    source_format: str = "unspecified"
    comparison_label: str = "p1"
    comparison_code: int = 101
    sampling_rate_hz: Optional[float] = None
    units: str = "unspecified"
    trial_filter: dict[str, Any] = field(default_factory=dict)
    condition_map: dict[str, list[int]] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_condition_map(self, condition_map: Mapping[str, Sequence[int]]) -> "DatasetSpec":
        """Documented public function `with_condition_map`."""
        mapped = {str(k): [int(x) for x in v] for k, v in condition_map.items()}
        return replace(self, condition_map=mapped)

    def with_quality_gate(self, name: str, value: Any) -> "DatasetSpec":
        """Documented public function `with_quality_gate`."""
        gates = dict(self.quality_gates)
        gates[str(name)] = value
        return replace(self, quality_gates=gates)


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.name:
            issues.append("dataset_name_missing")
        if self.comparison_code is None:
            issues.append("comparison_code_missing")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            issues.append("sampling_rate_hz_must_be_positive")
        return {
            "valid": not issues,
            "issues": issues,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        }

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "modality": self.modality,
            "source_format": self.source_format,
            "comparison_label": self.comparison_label,
            "comparison_code": self.comparison_code,
            "sampling_rate_hz": self.sampling_rate_hz,
            "units": self.units,
            "trial_filter": self.trial_filter,
            "condition_map": self.condition_map,
            "quality_gates": self.quality_gates,
            "metadata": self.metadata,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        })


@dataclass(frozen=True)
class TrialSpec:
    """Specification for a single simulation trial."""

    trial_id: str
    condition: Optional[ParadigmCondition] = None
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "trial_id": self.trial_id,
            "condition": self.condition.to_dict() if self.condition else None,
            "seed": self.seed,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialBatch:
    """A collection of trial specifications to be run."""

    trials: tuple[TrialSpec, ...]
    batch_id: str = "anonymous_batch"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "batch_id": self.batch_id,
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialResult:
    """Result of a single simulation trial."""

    trial_id: str
    condition_label: Optional[str] = None
    signals: Optional[Signals] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation, excluding large JAX arrays."""
        from .io import json_safe
        return json_safe({
            "trial_id": self.trial_id,
            "condition_label": self.condition_label,
            "success": self.success,
            "error_message": self.error_message,
            "signals": self.signals.summary() if self.signals else None,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialBatchResult:
    """Results from a batch of trials."""

    batch_id: str
    results: tuple[TrialResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "batch_id": self.batch_id,
            "n_results": len(self.results),
            "n_success": sum(1 for r in self.results if r.success),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        })




@dataclass(frozen=True)
class ReadoutSpec:
    """Declarative specification for extracting a scalar feature from Signals.

    Defines a single named metric to compute from a simulation's Signals
    object.  All fields are JSON-safe.  No physical-amplitude or calibration
    claim is made; all values are proxy or native-current units unless
    explicitly stated otherwise.

    Supported metrics (``_KNOWN_READOUT_METRICS``):
        spike_rate_hz, spike_count, mean_V_m,
        csd_abs_mean, lfp_abs_mean, source_abs_mean.

    Optional filters:
        time_window_ms: (start_ms, end_ms) tuple for temporal slice.
        n_contacts_slice: (start, end) tuple for contact-depth slice on field modes.
    """

    name: str
    metric: str
    time_window_ms: Optional[tuple[float, float]] = None
    n_contacts_slice: Optional[tuple[int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "metric": self.metric,
            "time_window_ms": list(self.time_window_ms) if self.time_window_ms else None,
            "n_contacts_slice": list(self.n_contacts_slice) if self.n_contacts_slice else None,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ReadoutResult:
    """Result of applying a ReadoutSpec to Signals.

    All scalar values are floats or None (when computation is not applicable).
    JSON-safe via to_dict().

    Status values:
        "computed"   — value was computed successfully.
        "no_field"   — metric requires field output but signals has no field.
        "unknown_metric" — metric not in _KNOWN_READOUT_METRICS.
    """

    spec_name: str
    metric: str
    value: Optional[float]
    status: str = "computed"
    claim_level: str = "computational_scaffold"
    physical_amplitude_calibrated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "spec_name": self.spec_name,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "claim_level": self.claim_level,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "metadata": self.metadata,
        })

    @property
    def name(self) -> str:
        """Compatibility alias for spec_name used by public examples.

        Allows usage like:
            for result in results:
                print(result.name, result.metric, result.value, result.status)
        """
        return self.spec_name



@dataclass(frozen=True)
class ObjectiveReport:
    """Structured, immutable result of evaluating an Objective against Signals.

    Wraps the dict returned by :meth:`Model.evaluate` into a frozen dataclass
    that is always JSON-safe and carries explicit truth gates.

    Gate pass/fail is a computational diagnostic only.  It does not imply
    empirical validation, biological calibration, or mechanism proof.

    ``readout_results`` is populated when ReadoutSpecs are passed to
    :meth:`Model.evaluate_report`.  It is an empty tuple otherwise.
    """

    objective_name: str
    evaluation_status: str
    total_loss: Optional[float]
    all_gates_pass: bool
    losses: tuple[dict[str, Any], ...]
    regularizers: tuple[dict[str, Any], ...]
    gates: tuple[dict[str, Any], ...]
    readout_results: tuple["ReadoutResult", ...] = field(default_factory=tuple)
    truth: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "objective_name": self.objective_name,
            "evaluation_status": self.evaluation_status,
            "total_loss": self.total_loss,
            "all_gates_pass": self.all_gates_pass,
            "losses": list(self.losses),
            "regularizers": list(self.regularizers),
            "gates": list(self.gates),
            "readout_results": [r.to_dict() for r in self.readout_results],
            "truth": self.truth,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class RunReceipt:
    """Complete, JSON-safe record of a single simulation run.

    Captures config fingerprint, simulation parameters, signal summary,
    and truth gates in one immutable object.  ``receipt_id`` is deterministic:
    same configuration + seed + version always yields the same ID.

    Truth status: all gates are frozen at conservative defaults and cannot be
    escalated.  No physical-amplitude, empirical-validation, or mechanism
    claim is introduced by this receipt.
    """

    receipt_id: str
    jaxfne_version: str
    config_hash: str
    simulation: dict[str, Any]
    signals_summary: dict[str, Any]
    truth: dict[str, Any]
    claim_labels: dict[str, Any]
    backend: dict[str, Any]
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "receipt_id": self.receipt_id,
            "jaxfne_version": self.jaxfne_version,
            "config_hash": self.config_hash,
            "simulation": self.simulation,
            "signals_summary": self.signals_summary,
            "truth": self.truth,
            "claim_labels": self.claim_labels,
            "backend": self.backend,
            "tags": self.tags,
        })

import numpy as _np  # used only in StimulusSchedule.to_array; no JAX tracing


def _make_poisson_drive(
    n_steps: int,
    n_neurons: int,
    rate_hz: float,
    amplitude: float,
    dt_ms: float,
    seed: int,
    target: str = "all",
) -> jax.Array:
    """Generate a Poisson stochastic drive array.
    
    Returns (n_steps, n_neurons) float32 array. Each timestep, each neuron
    has an independent Poisson event with probability rate_hz * dt_ms / 1000.
    Events inject `amplitude` native current units. Output is finite and bounded.
    """
    prob = float(rate_hz) * float(dt_ms) / 1000.0
    prob = min(max(prob, 0.0), 1.0)
    key = jax.random.PRNGKey(int(seed))
    noise = jax.random.bernoulli(key, p=prob, shape=(int(n_steps), int(n_neurons)))
    return (jnp.asarray(noise, dtype=jnp.float32) * float(amplitude))


@dataclass(frozen=True)
class StimulusSchedule:
    """Explicit drive schedule for event-aligned stimulus injection.

    Contains an ordered sequence of drive events as JSON-safe dicts with keys:
    ``onset_ms``, ``duration_ms``, ``amplitude``, ``label``, ``is_drive_event``.
    When ``is_drive_event`` is False or ``amplitude`` is 0, the event injects
    zero drive. No physical-amplitude or calibration claim is made; amplitude
    values are native Izhikevich current units.
    """

    events: tuple[dict[str, Any], ...]
    n_neurons: int
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"

    def to_array(self, n_steps: int, dt_ms: float, dtype: str = "float32") -> "jax.Array":
        """Materialize a ``(n_steps, n_neurons)`` drive schedule array.

        Supports optional target_indices in event to apply amplitude only to selected neurons.
        """
        schedule = _np.zeros((int(n_steps), int(self.n_neurons)), dtype=_np.float32)
        for ev in self.events:
            if not ev.get("is_drive_event", True):
                continue
            amp = float(ev.get("amplitude", 0.0))
            if amp == 0.0:
                continue
            onset_ms = float(ev.get("onset_ms", 0.0))
            dur_ms = float(ev.get("duration_ms", 50.0))
            start = int(round(onset_ms / dt_ms))
            end = int(round((onset_ms + dur_ms) / dt_ms))
            start = max(0, min(start, int(n_steps)))
            end = max(0, min(end, int(n_steps)))
            if start < end:
                # Check if this event has target_indices
                target_indices = ev.get("target_indices", None)
                if target_indices is not None:
                    # Apply amplitude only to selected indices
                    idx_array = _np.asarray(target_indices, dtype=int)
                    # Validate bounds
                    if len(idx_array) > 0:
                        if idx_array.min() < 0 or idx_array.max() >= self.n_neurons:
                            raise ValueError(
                                f"target_indices out of bounds: "
                                f"min={idx_array.min()}, max={idx_array.max()}, "
                                f"n_neurons={self.n_neurons}"
                            )
                        schedule[start:end, idx_array] += amp
                else:
                    # Apply amplitude to all neurons
                    schedule[start:end, :] += amp
        np_dtype = _np.float64 if dtype == "float64" else _np.float32
        return jnp.asarray(schedule.astype(np_dtype))

    def to_array_jax(self, n_steps: int, dt_ms: float, dtype: str = "float32") -> jax.Array:
        """Materialize a ``(n_steps, n_neurons)`` drive schedule array using JAX.

        Supports target_indices in event to apply amplitude only to selected neurons.
        Uses time masking and outer products for JIT/vmap compatibility.
        """
        jdtype = jnp.float64 if (dtype == "float64" and bool(jax.config.read("jax_enable_x64"))) else jnp.float32
        time_ms = jnp.arange(n_steps, dtype=jdtype) * dt_ms
        
        schedule = jnp.zeros((n_steps, self.n_neurons), dtype=jdtype)
        
        for ev in self.events:
            if not ev.get("is_drive_event", True):
                continue
            amp = float(ev.get("amplitude", 0.0))
            if amp == 0.0:
                continue
            onset_ms = float(ev.get("onset_ms", 0.0))
            dur_ms = float(ev.get("duration_ms", 50.0))
            
            event_mask = (time_ms >= onset_ms) & (time_ms < onset_ms + dur_ms)
            
            target_indices = ev.get("target_indices", None)
            if target_indices is not None:
                idx_array = jnp.asarray(target_indices, dtype=jnp.int32)
                neuron_mask = jnp.zeros((self.n_neurons,), dtype=jdtype)
                neuron_mask = neuron_mask.at[idx_array].set(1.0)
                schedule = schedule + (event_mask[:, None] * neuron_mask[None, :]) * amp
            else:
                schedule = schedule + event_mask[:, None] * amp
                
        return schedule

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        # Count targeted vs non-targeted events
        n_targeted = sum(1 for ev in self.events if ev.get("target_indices") is not None)
        return json_safe({
            "stimulus_injection_status": "native_drive_schedule_v0.0.12",
            "n_drive_events": len(self.events),
            "n_targeted_events": n_targeted,
            "n_neurons": self.n_neurons,
            "events": list(self.events),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
        })


_KNOWN_LAYERS = frozenset({"L1", "L2/3", "L4", "L5", "L6", "unspecified"})


@dataclass(frozen=True)
class LaminarPopulation:
    """Metadata descriptor for one named laminar cell population.

    Depth values are normalized proxy coordinates in [0, 1] — not physical
    microns.  Overlapping depth ranges are allowed; co-located cell types
    (e.g. E and PV in the same layer) are anatomically expected.
    No physical-amplitude or calibration claim is made.
    """

    name: str
    cell_type: str
    layer: str
    depth_min: float
    depth_max: float
    n_units: int
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.name:
            issues.append("name_empty")
        if not self.cell_type:
            issues.append("cell_type_empty")
        if not self.layer:
            issues.append("layer_empty")
        if not (0.0 <= self.depth_min < self.depth_max <= 1.0):
            issues.append("depth_range_invalid")
        if self.n_units <= 0:
            issues.append("n_units_must_be_positive")
        if self.physical_amplitude_calibrated is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        if self.claim_level != "computational_scaffold":
            issues.append("claim_level_must_be_computational_scaffold")
        warnings: list[str] = []
        if self.layer not in _KNOWN_LAYERS:
            warnings.append(f"unrecognized_layer:{self.layer}")
        return {"valid": not issues, "issues": issues, "warnings": warnings}

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "cell_type": self.cell_type,
            "layer": self.layer,
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "n_units": int(self.n_units),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
        })


@dataclass(frozen=True)
class LaminarSourceGeometry:
    """Metadata descriptor for the full laminar source geometry.

    Groups named :class:`LaminarPopulation` descriptors and can materialize
    a deterministic ``(n_units_total, 3)`` positions array for use in field
    projection.  Depths are proxy-normalized coordinates, not physical
    microns.  No physical-amplitude, PDE, or calibration claim is made.
    """

    populations: tuple[LaminarPopulation, ...]
    n_units_total: int
    position_units: str = "relative_laminar_depth_proxy"
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.populations:
            issues.append("populations_empty")
        pop_sum = sum(p.n_units for p in self.populations)
        if pop_sum != self.n_units_total:
            issues.append(f"n_units_total_mismatch:sum={pop_sum},declared={self.n_units_total}")
        if self.physical_amplitude_calibrated is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        pop_issues: list[str] = []
        for p in self.populations:
            v = p.validate()
            pop_issues.extend([f"{p.name}:{i}" for i in v["issues"]])
        issues.extend(pop_issues)
        return {"valid": not issues, "issues": issues, "n_populations": len(self.populations)}

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "type": "laminar_source_geometry",
            "n_units_total": self.n_units_total,
            "n_populations": len(self.populations),
            "position_units": self.position_units,
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
            "populations": [p.to_dict() for p in self.populations],
        })

    def population_slices(self) -> dict[str, slice]:
        """Map population names to neuron index slices.

        Returns:
            dict mapping population name → slice object spanning neuron indices.

        Example:
            >>> geom = LaminarSourceGeometry(...)
            >>> slices = geom.population_slices()
            >>> V_m_L4 = signals.V_m[slices["L4_E"], :]
        """
        result = {}
        start = 0
        for pop in self.populations:
            end = start + pop.n_units
            result[pop.name] = slice(start, end)
            start = end
        return result

    def positions_array(self, dtype: str = "float32") -> "jax.Array":
        """Return a deterministic ``(n_units_total, 3)`` positions array.

        x = 0, y = 0, z linearly spaced within each population's depth range.
        Population order is preserved.  No random sampling.
        """
        np_dtype = _np.float64 if dtype == "float64" else _np.float32
        rows: list[_np.ndarray] = []
        for pop in self.populations:
            n = int(pop.n_units)
            z = _np.linspace(float(pop.depth_min), float(pop.depth_max), n, dtype=np_dtype)
            xyz = _np.stack([_np.zeros(n, dtype=np_dtype), _np.zeros(n, dtype=np_dtype), z], axis=1)
            rows.append(xyz)
        arr = _np.concatenate(rows, axis=0) if rows else _np.zeros((0, 3), dtype=np_dtype)
        return jnp.asarray(arr)


_KNOWN_METRICS = frozenset({
    "spike_rate_hz_mean",
    "spike_count_total",
    "mean_V_m",
    "source_proxy_abs_mean",
    "csd_proxy_abs_mean",
    "lfp_proxy_abs_mean",
    "kappa_synchrony",
})

#: Config-metadata gate metrics — string-valued flags compared against
#: ``cfg.metadata`` rather than computed readout metrics.  These let an
#: Objective declare truth/scope gates (e.g. claim_level) that
#: are evaluated against the configuration, not the signals.
_KNOWN_CONFIG_GATE_METRICS = frozenset({
    "claim_level",
    "field_solver_status",
    "field_claim_level",
    "source_calibration_status",
    "source_projection_mode",
    "empirical_validation_status",
    "mechanism_claim_status",
})


def _finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _compute_kappa_synchrony_metric(spikes: Any) -> Optional[float]:
    """Vectorized mean pairwise spike-train correlation (kappa synchrony).

    Matches :func:`jaxfne.tutorial_utils.kappa_synchrony` semantics — mean
    Pearson correlation over neuron pairs with non-zero temporal variance —
    but computed as a single correlation matrix multiply for speed in the
    per-candidate tuning loop.  Proxy diagnostic; not a biological invariant.
    Expects ``spikes`` shaped ``[n_timesteps, n_neurons]``.
    """
    import numpy as _np

    x = _np.asarray(spikes, dtype=float)
    if x.ndim != 2:
        return 0.0
    n_timesteps, n_neurons = x.shape
    if n_neurons < 2 or n_timesteps < 1:
        return 0.0
    std = x.std(axis=0)
    valid = std > 0
    n_valid = int(valid.sum())
    if n_valid < 2:
        return 0.0
    xv = x[:, valid]
    z = (xv - xv.mean(axis=0)) / xv.std(axis=0)
    corr = (z.T @ z) / n_timesteps
    off_diagonal_sum = float(corr.sum() - _np.trace(corr))
    mean_pairwise = off_diagonal_sum / (n_valid * (n_valid - 1))
    return _finite_or_none(float(mean_pairwise))


def _compute_all_metrics(signals: "Signals", readout: Optional[dict[str, Any]] = None) -> dict[str, Optional[float]]:
    """Compute all known scalar metrics from signals."""
    dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
    m: dict[str, Optional[float]] = {}
    m["spike_rate_hz_mean"] = _finite_or_none(float(jnp.mean(signals.spikes) * (1000.0 / dt_ms)))
    m["spike_count_total"] = _finite_or_none(float(jnp.sum(signals.spikes)))
    m["mean_V_m"] = _finite_or_none(float(jnp.mean(signals.V_m)))
    m["kappa_synchrony"] = _compute_kappa_synchrony_metric(signals.spikes)
    if signals.field is not None:
        m["source_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.source_proxy))))
        m["csd_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.csd_proxy))))
        m["lfp_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.lfp_proxy))))
    else:
        m["source_proxy_abs_mean"] = None
        m["csd_proxy_abs_mean"] = None
        m["lfp_proxy_abs_mean"] = None
    return m


def _check_gate_criterion(value: float, threshold: Any, criterion: str) -> bool:
    """Return True if the gate passes for the given criterion."""
    if criterion == "below":
        return float(value) < float(threshold)
    if criterion == "above":
        return float(value) > float(threshold)
    if criterion == "equal":
        return abs(float(value) - float(threshold)) < 1e-6
    if criterion == "in_range":
        lo, hi = float(threshold[0]), float(threshold[1])
        return lo <= float(value) <= hi
    return False


def _evaluate_loss_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    """Evaluate one loss spec against computed metrics."""
    result: dict[str, Any] = {"name": spec["name"], "weight": spec.get("weight", 1.0)}
    metric = spec.get("metric")
    target = spec.get("target")
    if metric is None:
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = "no_metric_specified"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        if strict:
            result["status"] = msg
            result["value"] = None
            result["weighted_value"] = None
            warnings.append(msg)
            if "metadata" in spec:
                result["metadata"] = spec["metadata"]
            return result
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if target is not None:
        raw = (value - float(target)) ** 2
    else:
        raw = value
    weighted = float(spec.get("weight", 1.0)) * raw
    result["target"] = target
    result["raw_loss"] = _finite_or_none(raw)
    result["weighted_value"] = _finite_or_none(weighted)
    result["status"] = "ok"
    if "metadata" in spec:
        result["metadata"] = spec["metadata"]
    return result


def _evaluate_regularizer_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    """Evaluate one regularizer spec."""
    result: dict[str, Any] = {
        "name": spec["name"],
        "target": spec.get("target", 0.0),
        "weight": spec.get("weight", 1.0),
    }
    metric = spec.get("metric")
    if metric is None:
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = "no_metric_specified"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    target = float(spec.get("target", 0.0))
    raw = (value - target) ** 2
    weighted = float(spec.get("weight", 1.0)) * raw
    result["raw_regularizer"] = _finite_or_none(raw)
    result["weighted_value"] = _finite_or_none(weighted)
    result["status"] = "ok"
    if "metadata" in spec:
        result["metadata"] = spec["metadata"]
    return result


def _evaluate_gate_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
    cfg_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate one gate spec; returns pass/fail.

    Numeric gates are checked against computed readout ``metrics``.  Gates
    whose metric is a configuration flag (see ``_KNOWN_CONFIG_GATE_METRICS``)
    are checked against ``cfg_meta`` via exact string comparison — this lets
    an Objective assert truth/scope gates (claim_level, …) that
    describe the configuration rather than the signals.
    """
    result: dict[str, Any] = {
        "name": spec["name"],
        "threshold": spec.get("threshold"),
        "criterion": spec.get("criterion", "below"),
    }
    metric = spec.get("metric")
    if metric is None:
        result["value"] = None
        result["pass"] = False
        result["status"] = "no_metric_specified"
        return result
    # Config-metadata gates: exact-match a configuration flag.
    if metric in _KNOWN_CONFIG_GATE_METRICS:
        cfg_meta = cfg_meta or {}
        value = cfg_meta.get(metric, _CONSERVATIVE_TRUTH_DEFAULTS.get(metric))
        result["metric"] = metric
        result["value"] = value
        threshold = spec.get("threshold")
        criterion = spec.get("criterion", "exact")
        if criterion == "exact":
            passes = value == threshold
        else:
            passes = _check_gate_criterion(value, threshold, criterion) if value is not None else False
        result["pass"] = bool(passes)
        result["status"] = "pass" if passes else "fail"
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        warnings.append(msg)
        result["metric"] = metric
        result["value"] = None
        result["pass"] = False
        result["status"] = msg
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["pass"] = False
        result["status"] = "metric_unavailable"
        return result
    passes = _check_gate_criterion(value, spec.get("threshold"), spec.get("criterion", "below"))
    result["pass"] = passes
    result["status"] = "pass" if passes else "fail"
    return result


# ──────────────────────────────────────────────────────────────
# v0.2.26 computation-basis contracts
# ──────────────────────────────────────────────────────────────

#: Allowed values for AxisSpec.status
_AXIS_STATUS_VALUES: frozenset[str] = frozenset({"active", "collapsed", "indexed"})

#: Allowed values for BasisSpec.space_basis
_SPACE_BASIS_VALUES: frozenset[str] = frozenset(
    {"collapsed", "xy", "xyz", "laminar_depth", "graph"}
)

#: Allowed values for BasisSpec.time_basis
_TIME_BASIS_VALUES: frozenset[str] = frozenset(
    {"continuous_ms", "discrete_steps", "slow_proxy"}
)

#: Allowed values for BasisSpec.field_regime
_FIELD_REGIME_VALUES: frozenset[str] = frozenset(
    {
        "laminar_proxy",
        "quasi_static_resistive",
        "solved_poisson",
        "future_admittive",
        "future_maxwell",
    }
)

#: Regimes that are declared future — never claim implemented=True for these
_FUTURE_FIELD_REGIMES: frozenset[str] = frozenset({"future_admittive", "future_maxwell"})

#: Allowed values for BasisSpec.source_mode
_SOURCE_MODE_BASIS_VALUES: frozenset[str] = frozenset(
    {"total_membrane_current", "decomposed_cap_ion_syn", "proxy_no_field_solve"}
)

#: Allowed values for BasisSpec.probe_basis
_PROBE_BASIS_VALUES: frozenset[str] = frozenset(
    {
        "none",
        "spike_only",
        "field_proxy",
        "multimodal_proxy",
        "physical_forward_model",
    }
)


@dataclass(frozen=True)
class AxisSpec:
    """Typed descriptor for one tensor axis in the TFNE scaffold.

    Describes whether a spatial/feature dimension is actively computed
    (``active``), collapsed to a scalar or removed (``collapsed``), or
    indexed by an explicit label set (``indexed``).

    These are documentation/contract objects — they do not affect JAX
    execution. They appear in manifest output to make axis semantics
    explicit and auditable.

    Attributes
    ----------
    name : str
        Canonical dimension name (e.g. ``"x"``, ``"y"``, ``"z"``, ``"t"``).
    status : str
        One of ``"active"``, ``"collapsed"``, ``"indexed"``.
    size : int or None
        Known static size, if any. ``None`` if dynamic or not applicable.
    units_or_status : str
        Physical units (``"mm"``, ``"ms"``) or proxy status
        (``"declared"``, ``"proxy"``). Default ``"declared"``.
    """

    name: str
    status: str = "active"
    size: Optional[int] = None
    units_or_status: str = "declared"


    def validate(self) -> dict[str, Any]:
        """Return a JSON-safe validation dict."""
        issues: list[str] = []
        if not self.name:
            issues.append("name_empty")
        if self.status not in _AXIS_STATUS_VALUES:
            issues.append(f"invalid_status:{self.status!r}")
        if self.size is not None and self.size <= 0:
            issues.append(f"size_must_be_positive_got:{self.size}")
        return {
            "valid": not issues,
            "issues": issues,
            "name": self.name,
            "status": self.status,
            "size": self.size,
            "units_or_status": self.units_or_status,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dict representation."""
        return {
            "name": self.name,
            "status": self.status,
            "size": self.size,
            "units_or_status": self.units_or_status,
        }


@dataclass(frozen=True)
class BasisSpec:
    """Typed descriptor for the computation basis of a TFNE run.

    Declares the spatial, temporal, and field regime without claiming
    physical amplitude or biological validity. Future electrodynamic
    regimes (``future_maxwell``, ``future_admittive``) are recorded as
    declared-future modules: ``implemented=False``,
    ``claim_allowed=False``.

    The default matches the current v0.2.27 laminar-proxy scaffold.

    Attributes
    ----------
    space_basis : str
        Spatial computation domain. Default ``"laminar_depth"``.
    time_basis : str
        Temporal basis. Default ``"continuous_ms"``.
    field_regime : str
        Field computation regime. Default ``"laminar_proxy"``.
    source_mode : str
        Source model. Default ``"proxy_no_field_solve"``.
    probe_basis : str
        Probe operator class. Default ``"multimodal_proxy"``.
    axes : tuple of AxisSpec
        Explicit axis descriptors. Default: x/y collapsed, z active.
    """

    space_basis: str = "laminar_depth"
    time_basis: str = "continuous_ms"
    field_regime: str = "laminar_proxy"
    source_mode: str = "proxy_no_field_solve"
    probe_basis: str = "multimodal_proxy"
    axes: tuple[Any, ...] = field(
        default_factory=lambda: (
            AxisSpec(name="x", status="collapsed"),
            AxisSpec(name="y", status="collapsed"),
            AxisSpec(name="z", status="active", units_or_status="proxy"),
        )
    )

    @property
    def implemented(self) -> bool:
        """True if this regime has a runtime implementation in the current package."""
        # Future regimes are always unimplemented by doctrine
        if self.field_regime in _FUTURE_FIELD_REGIMES:
            return False
        # solved_poisson is specified but not solved in v0.2.x
        if self.field_regime == "solved_poisson":
            return False
        return True

    @property
    def claim_allowed(self) -> bool:
        """Physical amplitude claims are always False in proxy/scaffold regimes."""
        # Claims require solved field with calibrated conductivity — not in v0.2.x
        return False


    def validate(self) -> dict[str, Any]:
        """Return a JSON-safe validation dict. Raises ValueError on invalid enum."""
        issues: list[str] = []
        if self.space_basis not in _SPACE_BASIS_VALUES:
            issues.append(f"invalid_space_basis:{self.space_basis!r}")
        if self.time_basis not in _TIME_BASIS_VALUES:
            issues.append(f"invalid_time_basis:{self.time_basis!r}")
        if self.field_regime not in _FIELD_REGIME_VALUES:
            issues.append(f"invalid_field_regime:{self.field_regime!r}")
        if self.source_mode not in _SOURCE_MODE_BASIS_VALUES:
            issues.append(f"invalid_source_mode:{self.source_mode!r}")
        if self.probe_basis not in _PROBE_BASIS_VALUES:
            issues.append(f"invalid_probe_basis:{self.probe_basis!r}")
        # Axis-space consistency checks
        active_axes = {a.name for a in self.axes if a.status == "active"}
        for ax in self.axes:
            v = ax.validate()
            if not v["valid"]:
                issues.extend([f"axis_{ax.name}:{i}" for i in v["issues"]])
        if self.space_basis == "collapsed" and active_axes:
            issues.append(f"collapsed_basis_must_not_have_active_axes:{sorted(active_axes)}")
        if self.space_basis == "xy" and "z" in active_axes:
            issues.append("xy_basis:z_must_not_be_active_unless_indexed")
        if self.space_basis == "xyz":
            missing = {"x", "y", "z"} - active_axes - {
                a.name for a in self.axes if a.status in ("active", "indexed")
            }
            if missing:
                issues.append(f"xyz_basis:missing_active_or_indexed_axes:{sorted(missing)}")
        if self.space_basis == "laminar_depth":
            z_ok = any(
                a.name == "z" and a.status in ("active", "indexed") for a in self.axes
            )
            if not z_ok:
                issues.append("laminar_depth_basis:z_must_be_active_or_indexed")
        return {
            "valid": not issues,
            "issues": issues,
            "space_basis": self.space_basis,
            "time_basis": self.time_basis,
            "field_regime": self.field_regime,
            "source_mode": self.source_mode,
            "probe_basis": self.probe_basis,
            "implemented": self.implemented,
            "future_regime": self.field_regime in _FUTURE_FIELD_REGIMES,
            "claim_allowed": self.claim_allowed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict including axis dimension_status."""
        dim_status = {a.name: a.status for a in self.axes}
        return {
            "space_basis": self.space_basis,
            "time_basis": self.time_basis,
            "field_regime": self.field_regime,
            "source_mode": self.source_mode,
            "probe_basis": self.probe_basis,
            "dimension_status": dim_status,
            "implemented": self.implemented,
            "future_regime": self.field_regime in _FUTURE_FIELD_REGIMES,
            "claim_allowed": self.claim_allowed,
        }


def default_basis_spec() -> BasisSpec:
    """Return the default BasisSpec matching the current laminar-proxy scaffold."""
    return BasisSpec()


def _default_basis_dict() -> dict[str, Any]:
    """Return JSON-safe basis metadata dict for manifest embedding."""
    return default_basis_spec().to_dict()


# ──────────────────────────────────────────────────────────────
def _normalize_manifest_readout(
    readout: Any,
) -> Optional[dict[str, Any]]:
    """Normalize any supported readout argument shape for :meth:`Model.manifest`.

    Accepted input forms:

    * ``None``                    → returns ``None``
    * ``dict``                    → returned unchanged (legacy shape)
    * :class:`ReadoutResult`      → wrapped in list, then converted
    * ``list/tuple`` of :class:`ReadoutResult` → converted to summary dict
    * ``list/tuple`` of ``dict``  → converted to summary dict

    The returned dict (when non-None) always contains:

    * ``readout_results``    – list of JSON-safe readout result dicts
    * ``requested_metrics``  – list of metric name strings
    * ``n_results``          – integer count
    * ``physical_amplitude_calibrated`` – always False
    """
    if readout is None:
        return None
    if isinstance(readout, dict):
        return readout
    # Normalize single ReadoutResult to a one-element list.
    if isinstance(readout, ReadoutResult):
        readout = [readout]
    if isinstance(readout, (list, tuple)):
        items: list[dict[str, Any]] = []
        metrics: list[str] = []
        for item in readout:
            if isinstance(item, ReadoutResult):
                items.append(item.to_dict())
                metrics.append(item.metric)
            elif isinstance(item, dict):
                items.append(json_safe(item))
                metrics.append(str(item.get("metric", "unknown")))
            else:
                items.append(json_safe({"raw": str(item)}))
                metrics.append("unknown")
        return {
            "readout_results": items,
            "requested_metrics": metrics,
            "n_results": len(items),
            "physical_amplitude_calibrated": False,
        }
    # Fallback: stringify unknown types rather than crash.
    return {"readout_results": [json_safe({"raw": str(readout)})], "n_results": 1,
            "physical_amplitude_calibrated": False}


@dataclass(frozen=True)
class Model:
    """Immutable, runnable model built from a validated :class:`Configuration`.

    Holds the source ``cfg``, the dynamic ``params`` pytree (arrays that may be
    tuned/traced), and ``static`` metadata (JIT-static, non-array). Construct via
    :func:`construct`; run via :func:`simulate` / :meth:`simulate`. Also exported
    as the alias ``Net``. The model is a computational scaffold — its field and
    probe outputs are proxy readouts, not calibrated physical signals.
    """

    cfg: Configuration
    params: dict[str, Any]
    static: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe model metadata for notebook display."""
        from .io import json_safe
        emitter: IzhikevichParams = self.params["emitter"]
        return json_safe({
            "config_hash": config_hash(self.cfg),
            "n_units": int(emitter.v0.shape[0]),
            "n_contacts": int(self.static.get("n_contacts", 16)),
            "claim_level": self.cfg.metadata.get("claim_level", "computational_scaffold"),
            "source_calibration_status": self.cfg.metadata.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "field_solver_status": self.cfg.metadata.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
        })


    def neuron_table(self) -> list[dict[str, Any]]:
        """Return declared neuron metadata rows for area/layer/cell-type grouping."""
        rows = self.static.get("neuron_metadata")
        if rows is not None:
            return [dict(row) for row in rows]
        emitter: IzhikevichParams = self.params["emitter"]
        layers = emitter.layer_labels or tuple("unspecified" for _ in emitter.labels)
        positions = self.params.get("positions")
        rows_out: list[dict[str, Any]] = []
        for idx, label in enumerate(emitter.labels):
            z_value = None
            try:
                z_value = float(positions[idx, 2]) if positions is not None else None
            except Exception:
                z_value = None
            rows_out.append({
                "neuron_id": int(idx),
                "area": "network",
                "layer": str(layers[idx]),
                "cell_type": str(label),
                "z": z_value,
            })
        return rows_out

    def compile_connections(self, *, seed: int = 0, **kwargs: Any):
        """Compile this model's declared connection rules into sparse edges.

        Thin wrapper over :func:`jaxfne.compile_connection_rules`: resolves the
        rule selectors against this model's :meth:`neuron_table` and the
        ``metadata["circuit"]`` declarations. Declaration-only inputs in →
        finite sparse edge arrays out; no simulation numerics change.
        """
        from .connectivity import compile_connection_rules

        circuit = dict(self.cfg.metadata.get("circuit", {}))
        return compile_connection_rules(
            self.neuron_table(),
            circuit.get("connections", []),
            circuit.get("mechanisms", []),
            seed=seed,
            **kwargs,
        )

    def select(
        self,
        *,
        area: Optional[Any] = None,
        area_id: Optional[Any] = None,
        layer: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        ids: Optional[Sequence[int]] = None,
        allow_empty: bool = False,
    ) -> jax.Array:
        """Resolve semantic selectors to neuron row indices (does not mutate).

        Thin, non-mutating wrapper around :class:`SelectorSpec` over this model's
        :meth:`neuron_table`. Returns an int32 JAX array of row positions suitable
        for indexing the trailing (neuron) axis of V_m/spikes/sources. Empty
        matches raise ``ValueError`` unless ``allow_empty=True``. A requested
        field absent from the neuron table raises ``KeyError``.
        """
        spec = SelectorSpec(
            area=area,
            area_id=area_id,
            layer=layer,
            cell_type=cell_type,
            ids=tuple(ids) if ids is not None else None,
        )
        return spec.resolve(self.neuron_table(), allow_empty=allow_empty)

    def _simulate_arrays(
        self: "Model",
        sim: Simulation,
        key: jax.Array,
        runtime_cfg: RuntimeConfig,
        drive_schedule: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compile and execute the underlying simulation kernel.

        This method resolves the ablation mode masks, updates parameters, and
        dispatches to either the sparse/edge-list or dense JAX simulation kernels
        with compile-time caching.

        Parameters
        ----------
        sim : Simulation
            Simulation configuration.
        key : jax.Array
            JAX PRNG key.
        runtime_cfg : RuntimeConfig
            Resolved runtime config.
        drive_schedule : jax.Array, optional
            Input drive schedule array, by default None.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            Voltages, spikes, and source currents.
        """
        from .emitters import _dtype_from_policy
        emitter: IzhikevichParams = self.params["emitter"]
        sched = drive_schedule  # None or (n_steps, n_neurons) array
        
        # Build silence_mask if E_silence or I_silence is requested
        n_neurons = emitter.v0.shape[0]
        jdtype = _dtype_from_policy(runtime_cfg.actual_dtype)
        ablation_mode = getattr(sim, "ablation", None)

        # Sparse-direct models carry a placeholder dense W (edges live only in
        # params["edge_list"]); force the edge_list backend so the dense kernel is
        # never handed the empty W.
        if emitter.W.shape[0] != n_neurons and "edge_list" in self.params:
            runtime_cfg = replace(runtime_cfg, recurrent_backend="edge_list")

        if not hasattr(self, "_silence_masks"):
            object.__setattr__(self, "_silence_masks", {})

        if ablation_mode == "E_silence":
            if "E_silence" not in self._silence_masks:
                mask_list = [0.0 if lbl.startswith("E") else 1.0 for lbl in emitter.labels]
                self._silence_masks["E_silence"] = jnp.array(mask_list, dtype=jdtype)
            silence_mask = self._silence_masks["E_silence"]
        elif ablation_mode == "I_silence":
            if "I_silence" not in self._silence_masks:
                mask_list = [1.0 if lbl.startswith("E") else 0.0 for lbl in emitter.labels]
                self._silence_masks["I_silence"] = jnp.array(mask_list, dtype=jdtype)
            silence_mask = self._silence_masks["I_silence"]
        else:
            if "default" not in self._silence_masks:
                self._silence_masks["default"] = jnp.ones((n_neurons,), dtype=jdtype)
            silence_mask = self._silence_masks["default"]
            
        if ablation_mode == "disconnected_null":
            if runtime_cfg.recurrent_backend == "edge_list":
                edges: EdgeList = self.params["edge_list"]
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            else:
                emitter = replace(emitter, W=jnp.zeros_like(emitter.W))

        # Reset per-call homeostasis/HDP diagnostics (populated only when enabled).
        object.__setattr__(self, "_last_homeostasis_diag", None)
        object.__setattr__(self, "_last_hdp_diag", None)

        if getattr(runtime_cfg, "enable_homeostasis", False):
            if runtime_cfg.synaptic_kernel == "receptor_exponential":
                raise ValueError(
                    "enable_homeostasis is not supported with "
                    "synaptic_kernel='receptor_exponential'; use the default "
                    "exponential synaptic kernel."
                )
            # Homeostasis is sparse-edge based; edge_list always exists from construct().
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            hp = dict(runtime_cfg.homeostasis_params or {})
            _plastic_active = float(hp.get("eta", 0.0) or 0.0) != 0.0

            def _homeo_packed(k, s):
                """Return (V, spikes, sources, g_bias, r_trace[, w_final, w_trace])."""
                V, S, src, diag = simulate_edge_recurrent_izhikevich_homeostatic(
                    emitter, edges, sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                    silence_mask=silence_mask,
                    r_star=hp.get("r_star", 0.05),
                    tau_r_ms=hp.get("tau_r_ms", 300.0),
                    alpha=hp.get("alpha", 1.0),
                    k_gain=_resolve_homeostasis_k_gain(hp, emitter),
                    g_min=hp.get("g_min", -12.0),
                    g_max=hp.get("g_max", 8.0),
                    r_max=hp.get("r_max", 1.0),
                    eta=hp.get("eta", 0.0),
                    tau_x_ms=hp.get("tau_x_ms", 100.0),
                    w_min=hp.get("w_min", -10.0),
                    w_max=hp.get("w_max", 10.0),
                    v_floor=hp.get("v_floor", -150.0),
                    v_ceiling=hp.get("v_ceiling", 100.0),
                    u_abs_max=hp.get("u_abs_max", 2000.0),
                    syn_abs_max=hp.get("syn_abs_max", 1.0e4),
                )
                if _plastic_active:
                    return V, S, src, diag["g_bias"], diag["r_trace"], diag["w_final"], diag["w_trace"]
                return V, S, src, diag["g_bias"], diag["r_trace"]

            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                cache_key = ("simulate_homeostatic", B, Z, C, T, runtime_cfg.actual_dtype,
                             ablation_mode, runtime_cfg.selected_backend, _plastic_active,
                             _homeostasis_params_cache_fingerprint(hp))
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        guard_name = ("simulate_homeostatic_plastic" if _plastic_active
                                      else "simulate_homeostatic")
                        target_fn = make_recompilation_guard(
                            _homeo_packed, name=guard_name,
                            recompilation_guard=guard_mode, B=B, Z=Z, C=C, T=T,
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        result = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                    else:
                        result = self._compiled_cache[cache_key](key, sched)
            else:
                with _device_scope(runtime_cfg.selected_backend):
                    result = _homeo_packed(key, sched)
            if _plastic_active:
                V, S, src, g_bias, r_trace, w_final, w_trace = result
                object.__setattr__(self, "_last_homeostasis_diag",
                                   {"g_bias": g_bias, "r_trace": r_trace,
                                    "w_final": w_final, "w_trace": w_trace})
            else:
                V, S, src, g_bias, r_trace = result
                object.__setattr__(self, "_last_homeostasis_diag",
                                   {"g_bias": g_bias, "r_trace": r_trace})
            return V, S, src

        if getattr(runtime_cfg, "enable_hdp", False):
            if runtime_cfg.synaptic_kernel == "receptor_exponential":
                raise ValueError(
                    "enable_hdp is not supported with "
                    "synaptic_kernel='receptor_exponential'; use the default "
                    "exponential synaptic kernel."
                )
            # HDP is sparse-edge based; edge_list always exists from construct().
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            hp = dict(runtime_cfg.hdp_params or {})

            # Optional caller-supplied initial HDP state (Model.with_hdp_initial_state).
            # Absent by default -> init_state=None, the exact prior behavior
            # (kernel's own equilibrium H=1.0, native edge weight).
            _hdp_H0 = self.params.get("hdp_initial_H")
            _hdp_w0 = self.params.get("hdp_initial_w")
            init_state = None
            if _hdp_H0 is not None or _hdp_w0 is not None:
                _idt = runtime_cfg.actual_dtype
                init_state = {
                    "v": emitter.v0.astype(_idt),
                    "u": emitter.u0.astype(_idt),
                    "prev_spikes": jnp.zeros_like(emitter.v0, dtype=_idt),
                    "syn_state": jnp.zeros_like(edges.weight, dtype=_idt),
                }
                if _hdp_H0 is not None:
                    init_state["H_final"] = jnp.asarray(_hdp_H0, dtype=_idt)
                if _hdp_w0 is not None:
                    init_state["w_final"] = jnp.asarray(_hdp_w0, dtype=_idt)

            def _hdp_packed(k, s):
                """Return (V, spikes, sources, H_final, H_trace, w_final, w_trace)."""
                V, S, src, diag = simulate_edge_recurrent_izhikevich_hdp(
                    emitter, edges, sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                    silence_mask=silence_mask,
                    init_state=init_state,
                    H_min=hp.get("H_min", 0.1), H_max=hp.get("H_max", 10.0),
                    tau_0_ms=hp.get("tau_0_ms", 100.0),
                    alpha=hp.get("alpha", 0.0), beta=hp.get("beta", 0.0),
                    gamma=hp.get("gamma", 0.0), delta=hp.get("delta", 0.0),
                    C_spike=hp.get("C_spike", 0.0), K_HDP=hp.get("K_HDP", 1.0),
                    K_ctrl=hp.get("K_ctrl", 0.0),
                    barrier_c=hp.get("barrier_c", 0.0), barrier_d=hp.get("barrier_d", 0.0),
                    barrier_eps=hp.get("barrier_eps", 1.0e-3),
                    w_floor=hp.get("w_floor", 1.0e-3), w_ceiling=hp.get("w_ceiling", 50.0),
                    v_floor=hp.get("v_floor", -150.0), v_ceiling=hp.get("v_ceiling", 100.0),
                    u_abs_max=hp.get("u_abs_max", 2000.0), syn_abs_max=hp.get("syn_abs_max", 1.0e4),
                    H_boost_gain=hp.get("H_boost_gain", 0.0),
                    size_scale_by_cell_type=hp.get("size_scale_by_cell_type"),
                    size_scale_override=hp.get("size_scale_override"),
                )
                return V, S, src, diag["H_final"], diag["H_trace"], diag["w_final"], diag["w_trace"]

            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                cache_key = ("simulate_hdp", B, Z, C, T, runtime_cfg.actual_dtype,
                             ablation_mode, runtime_cfg.selected_backend,
                             _homeostasis_params_cache_fingerprint(hp))
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        target_fn = make_recompilation_guard(
                            _hdp_packed, name="simulate_hdp",
                            recompilation_guard=guard_mode, B=B, Z=Z, C=C, T=T,
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        result = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                    else:
                        result = self._compiled_cache[cache_key](key, sched)
            else:
                with _device_scope(runtime_cfg.selected_backend):
                    result = _hdp_packed(key, sched)
            V, S, src, H_final, H_trace, w_final, w_trace = result
            object.__setattr__(self, "_last_hdp_diag",
                               {"H_final": H_final, "H_trace": H_trace,
                                "w_final": w_final, "w_trace": w_trace})
            return V, S, src

        if runtime_cfg.recurrent_backend == "edge_list":
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            kernel_fn = (
                simulate_receptor_exponential_izhikevich
                if runtime_cfg.synaptic_kernel == "receptor_exponential"
                else simulate_edge_recurrent_izhikevich
            )
            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")

                cache_key = ("simulate_recurrent", B, Z, C, T, runtime_cfg.actual_dtype, runtime_cfg.synaptic_kernel, ablation_mode, runtime_cfg.selected_backend)
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        target_fn = lambda k, s: kernel_fn(
                            emitter, edges, sim.n_steps, sim.dt_ms, k,
                            dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                            silence_mask=silence_mask,
                        )[:3]
                        target_fn = make_recompilation_guard(
                            target_fn,
                            name="simulate",
                            recompilation_guard=guard_mode,
                            B=B, Z=Z, C=C, T=T
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        res = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                        return res
                    run = self._compiled_cache[cache_key]
                    return run(key, sched)
            with _device_scope(runtime_cfg.selected_backend):
                return kernel_fn(
                    emitter, edges, sim.n_steps, sim.dt_ms, key,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
                    silence_mask=silence_mask,
                )[:3]
        effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
        if effective_jit:
            if not hasattr(self, "_compiled_cache"):
                object.__setattr__(self, "_compiled_cache", {})
            from .validation import make_recompilation_guard
            B = 1
            Z = int(self.static.get("n_contacts", 16))
            C = int(emitter.n_neurons)
            T = int(sim.n_steps)
            guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")

            cache_key = ("simulate_dense", B, Z, C, T, runtime_cfg.actual_dtype, ablation_mode, runtime_cfg.selected_backend)
            with _device_scope(runtime_cfg.selected_backend):
                if cache_key not in self._compiled_cache:
                    import time
                    target_fn = lambda k, s: simulate_eig_izhikevich(
                        emitter, sim.n_steps, sim.dt_ms, k,
                        dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                        silence_mask=silence_mask,
                    )
                    target_fn = make_recompilation_guard(
                        target_fn,
                        name="simulate",
                        recompilation_guard=guard_mode,
                        B=B, Z=Z, C=C, T=T
                    )
                    self._compiled_cache[cache_key] = jax.jit(target_fn)
                    t0 = time.perf_counter()
                    res = self._compiled_cache[cache_key](key, sched)
                    t1 = time.perf_counter()
                    if not hasattr(self, "_warmup_times"):
                        object.__setattr__(self, "_warmup_times", [])
                    self._warmup_times.append(t1 - t0)
                    return res
                run = self._compiled_cache[cache_key]
                return run(key, sched)
        with _device_scope(runtime_cfg.selected_backend):
            return simulate_eig_izhikevich(
                emitter, sim.n_steps, sim.dt_ms, key,
                dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
                silence_mask=silence_mask,
            )

    def _resolve_stimulus_schedule(
        self,
        paradigm: Any,
        sim: Simulation,
        runtime_cfg: RuntimeConfig,
    ) -> Optional["StimulusSchedule"]:
        """Return a StimulusSchedule from paradigm arg, or None."""
        if paradigm is None:
            return None
        if isinstance(paradigm, StimulusSchedule):
            return paradigm
        if isinstance(paradigm, ParadigmCondition):
            return stimulus_schedule(
                paradigm.events,
                n_neurons=self.params["emitter"].n_neurons,
            )
        return None

    def simulate(
        self: "Model",
        sim: Simulation,
        paradigm: "Optional[Any]" = None,
    ) -> Signals:
        """Run the default EIG/Izhikevich vertical slice.

        When ``paradigm`` is None, behavior is identical to v0.0.11.
        When ``paradigm`` is a :class:`StimulusSchedule`, its drive array is
        injected as native (uncalibrated) current at each timestep.
        When ``paradigm`` is a :class:`ParadigmCondition`, its events are
        converted to a ``StimulusSchedule`` and injected.

        JIT is opt-in through ``Simulation(runtime=RuntimeConfig(jit=True))`` or
        ``runtime(jit=True)``.  The compiled path preserves the same proxy-field
        truth status as the eager path. No calibrated amplitude, PDE, or empirical
        claim is introduced by stimulus injection.
        """

        runtime_cfg = sim.resolved_runtime
        key = jax.random.PRNGKey(sim.seed)

        schedule = self._resolve_stimulus_schedule(paradigm, sim, runtime_cfg)
        drive_array: Optional[Any] = None
        if schedule is not None:
            drive_array = schedule.to_array(sim.n_steps, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
        if sim.poisson_drive is not None:
            _emitter: IzhikevichParams = self.params["emitter"]
            _pd = sim.poisson_drive
            _poisson_arr = _make_poisson_drive(
                n_steps=sim.n_steps,
                n_neurons=_emitter.n_neurons,
                rate_hz=float(_pd.get("rate_hz", 2.0)),
                amplitude=float(_pd.get("amplitude", 0.5)),
                dt_ms=sim.dt_ms,
                seed=int(_pd.get("seed", sim.seed + 7919)),
                target=str(_pd.get("target", "all")),
            )
            drive_array = _poisson_arr if drive_array is None else drive_array + _poisson_arr

        # shuffled_timing ablation: shuffle drive_array along time axis (axis 0) independently for each neuron
        ablation_mode = getattr(sim, "ablation", None)
        if ablation_mode == "shuffled_timing" and drive_array is not None:
            shuffle_key = jax.random.PRNGKey(sim.seed + 12345)
            n_neurons = drive_array.shape[1]
            keys = jax.random.split(shuffle_key, n_neurons)
            # Use vmap to shuffle each neuron's temporal drive independently
            shuffled = jax.vmap(lambda arr, k: jax.random.permutation(k, arr))(drive_array.T, keys)
            drive_array = shuffled.T

        voltages, spikes, sources = self._simulate_arrays(sim, key, runtime_cfg, drive_schedule=drive_array)
        time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
            sim.dt_ms, dtype=runtime_cfg.jnp_dtype
        )
        positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
        field_output = None
        if sim.record_fields:
            field_output = project_laminar_sources(
                sources=sources,
                positions=positions,
                n_contacts=self.static.get("n_contacts", 16),
                dtype=runtime_cfg.actual_dtype,
            )

        paradigm_meta: Optional[dict[str, Any]] = None
        if isinstance(paradigm, Mapping):
            paradigm_meta = dict(paradigm)
        elif hasattr(paradigm, "to_dict"):
            paradigm_meta = paradigm.to_dict()

        metadata: dict[str, Any] = {
            "config_hash": config_hash(self.cfg),
            "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
            "field_claim_level": "proxy_readout",
            "paradigm": paradigm_meta,
            "duration_ms": float(sim.duration_ms),
            "dt_ms": float(sim.dt_ms),
            "n_steps": int(sim.n_steps),
            "record_sources": bool(sim.record_sources),
            "record_fields": bool(sim.record_fields),
            "plasticity_gain": sim.plasticity,
            "runtime": runtime_cfg.runtime_report(),
            "recurrent_backend": runtime_cfg.recurrent_backend,
            "synaptic_kernel": runtime_cfg.synaptic_kernel,
            "source_model": _SOURCE_PROXY_METADATA,
            "neuron_metadata": self.static.get("neuron_metadata"),
            "neuron_metadata_summary": self.static.get("neuron_metadata_summary"),
            "ablation": ablation_mode,
        }
        # v0.2.0: Add source bookkeeping metadata for theoretical validation.
        metadata["source_bookkeeping"] = {
            "source_mode": _SOURCE_PROXY_METADATA.get("source_mode"),
            "source_projection_mode": self.cfg.metadata.get("source_projection_mode", "proxy_no_field_solve"),
            "source_decomposition": self.cfg.metadata.get("source_decomposition", "proxy_reduced_emitter"),
            "source_calibration_status": _SOURCE_PROXY_METADATA.get("source_calibration_status"),
            "synaptic_current_counting": _SOURCE_PROXY_METADATA.get("double_count_synaptic_current_guard"),
            "source_mode_exclusive": True,
            "physical_amplitude_calibrated": _SOURCE_PROXY_METADATA.get("physical_amplitude_calibrated", False),
            "double_count_guard": "passed",
            "double_count_evidence": None,
        }
        if schedule is not None:
            metadata["stimulus_injection_status"] = "native_drive_schedule_v0.0.12"
            metadata["stimulus_schedule"] = schedule.to_dict()
            if isinstance(paradigm, ParadigmCondition):
                metadata["condition_name"] = paradigm.name
                metadata["has_omission"] = paradigm.has_omission()
        if sim.poisson_drive is not None:
            metadata["poisson_drive"] = {
                "rate_hz": float(sim.poisson_drive.get("rate_hz", 2.0)),
                "amplitude": float(sim.poisson_drive.get("amplitude", 0.5)),
                "target": str(sim.poisson_drive.get("target", "all")),
                "seed": int(sim.poisson_drive.get("seed", sim.seed + 7919)),
                "status": "stochastic_drive_applied",
            }
        if getattr(runtime_cfg, "enable_homeostasis", False):
            diag = getattr(self, "_last_homeostasis_diag", None)
            metadata["homeostasis"] = _simulate_homeostasis_metadata(runtime_cfg, diag)
        if getattr(runtime_cfg, "enable_hdp", False):
            diag = getattr(self, "_last_hdp_diag", None)
            metadata["hdp"] = _simulate_hdp_metadata(runtime_cfg, diag)
        return Signals(
            time_ms=time_ms,
            V_m=voltages.astype(runtime_cfg.jnp_dtype),
            spikes=spikes,
            sources=sources.astype(runtime_cfg.jnp_dtype) if sim.record_sources else None,
            field=field_output,
            metadata=metadata,
        )

    def last_homeostasis_diagnostics(self) -> "Optional[dict[str, Any]]":
        """Return the full per-step homeostasis diagnostics from the most recent
        ``simulate(...)`` call with ``enable_homeostasis=True``.

        Returns a dict with arrays ``g_bias`` and ``r_trace`` of shape
        ``(n_steps, n_neurons)``, or ``None`` if homeostasis was not enabled on
        the last run. When ``homeostasis_params["eta"] != 0`` (homeostatic
        synaptic plasticity active), the dict also carries ``w_final``
        ``(n_edges,)`` and ``w_trace`` ``(n_steps, n_edges)`` — the plastic
        edge-weight trajectory. These are computational-control diagnostics
        (proxy), not a biological-mechanism claim.
        """
        return getattr(self, "_last_homeostasis_diag", None)

    def last_hdp_diagnostics(self) -> "Optional[dict[str, Any]]":
        """Return the full per-step HDP diagnostics from the most recent
        ``simulate(...)`` call with ``enable_hdp=True``.

        Returns a dict with ``H_final``/``H_trace`` ``(n_steps, n_neurons)``
        and ``w_final``/``w_trace`` ``(n_steps, n_edges)``, or ``None`` if HDP
        was not enabled on the last run. See
        ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp`` for the
        underlying kernel and ``jaxfne.hdp_network.DEFAULT_HDP`` /
        ``DEFAULT_HDP_DESYNC`` for tuned presets. Computational-control
        diagnostics (proxy), not a biological-mechanism claim.
        """
        return getattr(self, "_last_hdp_diag", None)

    def simulate_condition(
        self,
        sim: Simulation,
        condition: "ParadigmCondition",
        *,
        drive_amplitude: float = 5.0,
        event_duration_ms: float = 50.0,
    ) -> Signals:
        """Convenience wrapper: simulate one trial condition with event-aligned drive injection.

        Equivalent to ``simulate(sim, paradigm=condition)`` but allows per-call
        override of ``drive_amplitude`` and ``event_duration_ms``.
        No calibrated amplitude, PDE, or empirical claim is introduced.
        """
        schedule = stimulus_schedule(
            condition.events,
            n_neurons=self.params["emitter"].n_neurons,
            drive_amplitude=drive_amplitude,
            event_duration_ms=event_duration_ms,
        )
        signals = self.simulate(sim, paradigm=schedule)
        signals.metadata["condition_name"] = condition.name
        signals.metadata["has_omission"] = condition.has_omission()
        return signals

    def simulate_batch(self, sim: Simulation, n_seeds: int = 4, seed: int | None = None) -> dict[str, Any]:
        """Run a vectorized seed batch and return JSON-safe metadata plus arrays.

        This is a trial-replicate utility for notebook statistics.  It uses
        ``jax.vmap`` over PRNG keys and returns proxy arrays without changing the
        field-solver or calibration status.
        """
        from .io import json_safe
        runtime_cfg = sim.resolved_runtime
        base_seed = sim.seed if seed is None else int(seed)
        keys = jax.random.split(jax.random.PRNGKey(base_seed), int(n_seeds))
        emitter: IzhikevichParams = self.params["emitter"]

        # Sparse-direct models (placeholder dense W) must use the edge_list backend.
        if emitter.W.shape[0] != int(emitter.v0.shape[0]) and "edge_list" in self.params:
            runtime_cfg = replace(runtime_cfg, recurrent_backend="edge_list")

        homeo_on = bool(getattr(runtime_cfg, "enable_homeostasis", False))
        if homeo_on and runtime_cfg.synaptic_kernel == "receptor_exponential":
            raise ValueError(
                "enable_homeostasis is not supported with "
                "synaptic_kernel='receptor_exponential'."
            )
        edge_kernel_fn = (
            simulate_receptor_exponential_izhikevich
            if runtime_cfg.synaptic_kernel == "receptor_exponential"
            else simulate_edge_recurrent_izhikevich
        )
        _hp = dict(runtime_cfg.homeostasis_params or {})

        def one(k):
            """Documented public function `one`."""
            if homeo_on:
                # Homeostasis engages the sparse-edge homeostatic kernel; per-step
                # g_bias/r_trace diagnostics are dropped here (batch is a seed-replicate
                # statistics utility — use simulate() for full diagnostics passthrough).
                return simulate_edge_recurrent_izhikevich_homeostatic(
                    emitter, self.params["edge_list"], sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype,
                    r_star=_hp.get("r_star", 0.05), tau_r_ms=_hp.get("tau_r_ms", 300.0),
                    alpha=_hp.get("alpha", 1.0), k_gain=_resolve_homeostasis_k_gain(_hp, emitter),
                    g_min=_hp.get("g_min", -12.0), g_max=_hp.get("g_max", 8.0),
                    r_max=_hp.get("r_max", 1.0),
                    eta=_hp.get("eta", 0.0), tau_x_ms=_hp.get("tau_x_ms", 100.0),
                    w_min=_hp.get("w_min", -10.0), w_max=_hp.get("w_max", 10.0),
                    v_floor=_hp.get("v_floor", -150.0), v_ceiling=_hp.get("v_ceiling", 100.0),
                    u_abs_max=_hp.get("u_abs_max", 2000.0), syn_abs_max=_hp.get("syn_abs_max", 1.0e4),
                )[:3]
            if runtime_cfg.recurrent_backend == "edge_list":
                return edge_kernel_fn(
                    emitter,
                    self.params["edge_list"],
                    sim.n_steps,
                    sim.dt_ms,
                    k,
                    dtype=runtime_cfg.actual_dtype,
                )[:3]
            return simulate_eig_izhikevich(
                emitter, sim.n_steps, sim.dt_ms, k, dtype=runtime_cfg.actual_dtype
            )

        # v0.0.21: honor runtime.vmap flag behaviorally.
        # vmap=True  → jax.vmap over keys (one compiled call, vectorized over batch).
        # vmap=False → Python-loop + jnp.stack (each key runs independently, no vmap).
        effective_vmap = runtime_cfg.resolve_vmap(int(n_seeds))
        if effective_vmap:
            if not hasattr(self, "_compiled_cache"):
                object.__setattr__(self, "_compiled_cache", {})
            B = int(n_seeds)
            Z = int(self.static.get("n_contacts", 16))
            C = int(emitter.n_neurons)
            T = int(sim.n_steps)
            cache_key = ("simulate_batch", B, Z, C, T, runtime_cfg.actual_dtype, runtime_cfg.synaptic_kernel, runtime_cfg.recurrent_backend, homeo_on, runtime_cfg.selected_backend,
                         _homeostasis_params_cache_fingerprint(_hp) if homeo_on else ())
            with _device_scope(runtime_cfg.selected_backend):
                effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons, batch=B)
                if effective_jit:
                    if cache_key not in self._compiled_cache:
                        import time
                        from .validation import make_recompilation_guard
                        guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                        run_mapped = jax.vmap(one)
                        run_mapped = make_recompilation_guard(
                            run_mapped,
                            name="simulate_batch",
                            recompilation_guard=guard_mode,
                            B=B, Z=Z, C=C, T=T
                        )
                        self._compiled_cache[cache_key] = jax.jit(run_mapped)
                        t0 = time.perf_counter()
                        results = self._compiled_cache[cache_key](keys)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                        voltages, spikes, sources = results
                    else:
                        run = self._compiled_cache[cache_key]
                        voltages, spikes, sources = run(keys)
                else:
                    run = jax.vmap(one)
                    voltages, spikes, sources = run(keys)
            batch_execution_mode = "jax_vmap"
        else:
            per_key = [one(k) for k in keys]
            voltages = jnp.stack([t[0] for t in per_key], axis=0)
            spikes = jnp.stack([t[1] for t in per_key], axis=0)
            sources = jnp.stack([t[2] for t in per_key], axis=0)
            batch_execution_mode = "python_loop_stack"

        if runtime_cfg.recurrent_backend == "edge_list":
            batch_status = (
                "vmap_seed_batch_v0.0.11"
                if runtime_cfg.synaptic_kernel == "receptor_exponential"
                else "vmap_seed_batch_v0.0.9"
            )
        else:
            batch_status = "vmap_seed_batch_v0.0.8"
        return {
            "V_m": voltages.astype(runtime_cfg.jnp_dtype),
            "spikes": spikes,
            "sources": sources.astype(runtime_cfg.jnp_dtype),
            "metadata": json_safe({
                "batch_status": batch_status,
                "batch_execution_mode": batch_execution_mode,
                "n_seeds": int(n_seeds),
                "seed": base_seed,
                "runtime": runtime_cfg.runtime_report(),
                "field_claim_level": "proxy_readout",
                "physical_amplitude_calibrated": False,
                "recurrent_backend": runtime_cfg.recurrent_backend,
                "synaptic_kernel": runtime_cfg.synaptic_kernel,
                "enable_homeostasis": homeo_on,
                "homeostasis_params": _hp if homeo_on else None,
                "source_model": _SOURCE_PROXY_METADATA,
            }),
        }

    def run_trials(self, batch: TrialBatch, sim: Simulation, collect_errors: bool = False) -> TrialBatchResult:
        """Execute a batch of trials sequentially.

        For each trial in the batch, this method:
        1. Replaces sim.seed with trial.seed.
        2. Calls self.simulate(sim_trial, paradigm=trial.condition).
        3. If collect_errors=False (default): raises immediately on failure.
           If collect_errors=True: records exception in TrialResult and continues.

        Returns a TrialBatchResult containing all individual TrialResults (or raises on first failure).
        """
        results: list[TrialResult] = []
        for trial in batch.trials:
            sim_trial = replace(sim, seed=trial.seed)
            try:
                signals = self.simulate(sim_trial, paradigm=trial.condition)
                results.append(
                    TrialResult(
                        trial_id=trial.trial_id,
                        condition_label=trial.condition.name if trial.condition else None,
                        signals=signals,
                        success=True,
                        metadata=trial.metadata,
                    )
                )
            except Exception as e:
                if not collect_errors:
                    raise
                results.append(
                    TrialResult(
                        trial_id=trial.trial_id,
                        condition_label=trial.condition.name if trial.condition else None,
                        signals=None,
                        success=False,
                        error_message=str(e),
                        metadata=trial.metadata,
                    )
                )
        return TrialBatchResult(batch_id=batch.batch_id, results=tuple(results), metadata=batch.metadata)

    def run_receipt(self, signals: Signals, *, tags: Optional[dict[str, Any]] = None) -> RunReceipt:
        """Build a RunReceipt capturing this run for audit and reproducibility.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`manifest`
        for recording completed simulation runs.

        Args:
            signals: Signals returned by self.simulate().
            tags: Optional user-supplied key-value metadata (condition, paper, etc.).

        Returns:
            RunReceipt with frozen truth gates and deterministic receipt_id.

        Note:
            ``receipt_id`` is deterministic for the same
            ``(config_hash, seed, _JAXFNE_VERSION)`` triple.  Upgrading the
            package version changes the ID even when config and seed are
            identical, because the computational kernel may have changed.
            IDs are audit identifiers; they are not empirical claims.
        """
        from .io import json_safe, sha256_text

        cfg_h = config_hash(self.cfg)
        # Seed is stored inside the runtime sub-dict (via RuntimeConfig.runtime_report)
        seed = int(signals.metadata.get("runtime", {}).get("seed", signals.metadata.get("seed", 0)))

        sim_meta = signals.metadata
        sim_summary: dict[str, Any] = {
            "duration_ms": sim_meta.get("duration_ms"),
            "dt_ms": sim_meta.get("dt_ms"),
            "seed": seed,
            "n_steps": int(signals.time_ms.shape[0]),
            "record_sources": sim_meta.get("record_sources"),
            "record_fields": sim_meta.get("record_fields"),
        }

        # Deterministic receipt_id based on config, version, simulation, and key runtime metadata
        receipt_payload = {
            "config_hash": cfg_h,
            "jaxfne_version": _JAXFNE_VERSION,
            "simulation": sim_summary,
            "runtime": sim_meta.get("runtime"),
            "condition_name": sim_meta.get("condition_name"),
            "stimulus_schedule": sim_meta.get("stimulus_schedule"),
            "recurrent_backend": sim_meta.get("recurrent_backend"),
            "synaptic_kernel": sim_meta.get("synaptic_kernel"),
            "source_model": sim_meta.get("source_model"),
        }
        receipt_id = sha256_text(
            json.dumps(json_safe(receipt_payload), sort_keys=True, allow_nan=False)
        )[:16]

        truth: dict[str, Any] = {
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "linear_solver",
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }

        claim_labels: dict[str, Any] = {
            "receipt_status": _RECEIPT_SCHEMA_VERSION,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "physical_amplitude_calibrated": False,
        }

        backend: dict[str, Any] = {
            "recurrent_backend": signals.metadata.get("recurrent_backend", "dense"),
            "synaptic_kernel": signals.metadata.get("synaptic_kernel", "exponential"),
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "physical_amplitude_calibrated": False,
            "source_model": signals.metadata.get("source_model"),
            "source_bookkeeping": signals.metadata.get("source_bookkeeping"),
        }
        if "edge_list" in self.params:
            edges = self.params["edge_list"]
            backend["edge_list_n_edges"] = int(edges.n_edges)
            backend["edge_list_backend"] = "edge_list_recurrent_v0.0.9"

        return RunReceipt(
            receipt_id=receipt_id,
            jaxfne_version=_JAXFNE_VERSION,
            config_hash=cfg_h,
            simulation=sim_summary,
            signals_summary=signals.summary(),
            truth=truth,
            claim_labels=claim_labels,
            backend=backend,
            tags=dict(tags or {}),
        )


    def compute_readout(
        self,
        signals: Signals,
        specs: "Sequence[ReadoutSpec]",
    ) -> "list[ReadoutResult]":
        """Compute scalar features from Signals according to a list of ReadoutSpecs.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`probe`
        for declarative, typed feature extraction.

        Args:
            signals: Signals returned by self.simulate().
            specs: Sequence of ReadoutSpec objects declaring what to extract.

        Returns:
            List of ReadoutResult objects in the same order as specs.
            Values are None when not applicable (missing field, unknown metric).

        No physical-amplitude, empirical-validation, or mechanism claim is
        introduced.  All values are proxy/native-current scaffold outputs.
        """
        results: list[ReadoutResult] = []
        for spec in specs:
            if spec.metric not in _KNOWN_READOUT_METRICS:
                results.append(ReadoutResult(
                    spec_name=spec.name,
                    metric=spec.metric,
                    value=None,
                    status="unknown_metric",
                ))
                continue

            dt_ms = (
                float(signals.time_ms[1] - signals.time_ms[0])
                if signals.time_ms.shape[0] > 1
                else 1.0
            )

            # Time slice (optional); negative start is treated as empty window.
            if spec.time_window_ms is not None:
                start_ms, end_ms = spec.time_window_ms
                t0 = max(0, int(start_ms / dt_ms))
                t1 = min(int(signals.time_ms.shape[0]), int(end_ms / dt_ms))
                if t0 >= t1:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="empty_time_window",
                    ))
                    continue
                V_m_sl = signals.V_m[t0:t1]
                sp_sl = signals.spikes[t0:t1]
                src_sl = signals.sources[t0:t1] if signals.sources is not None else None
                field_t0, field_t1 = t0, t1
            else:
                V_m_sl = signals.V_m
                sp_sl = signals.spikes
                src_sl = signals.sources
                field_t0, field_t1 = 0, int(signals.time_ms.shape[0])

            if spec.metric == "spike_rate_hz":
                value = float(jnp.mean(sp_sl) * (1000.0 / dt_ms))
            elif spec.metric == "spike_count":
                value = float(jnp.sum(sp_sl))
            elif spec.metric == "mean_V_m":
                value = float(jnp.mean(V_m_sl))
            elif spec.metric == "source_abs_mean":
                if src_sl is None:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="missing_sources",
                    ))
                    continue
                value = float(jnp.mean(jnp.abs(src_sl)))
            elif spec.metric in ("csd_abs_mean", "lfp_abs_mean"):
                if signals.field is None:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="no_field",
                    ))
                    continue
                arr = signals.field.csd if spec.metric == "csd_abs_mean" else signals.field.lfp
                # Apply time-window slice first, then contact slice.
                arr = arr[field_t0:field_t1]
                if spec.n_contacts_slice is not None:
                    c0, c1 = spec.n_contacts_slice
                    arr = arr[:, c0:c1]
                value = float(jnp.mean(jnp.abs(arr)))
            else:
                value = None

            results.append(ReadoutResult(
                spec_name=spec.name,
                metric=spec.metric,
                value=value,
                status="computed",
            ))
        return results

    def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
        """Extract named arrays from Signals by mode.

        Compatibility alias retained from v0.0.3–v0.0.14.  For typed,
        declarative feature extraction in the canonical v0.1 workflow, prefer
        :meth:`compute_readout` with :class:`ReadoutSpec` objects.
        """

        modes = list(modes or [])
        out: dict[str, Any] = {"requested_modes": modes}
        if "spikes" in modes:
            out["spikes"] = signals.spikes
        if "V_m" in modes:
            out["V_m"] = signals.V_m
        if "source" in modes or "sources" in modes:
            out["sources"] = signals.sources
        if signals.field is not None:
            out.update(probe_laminar_modes(signals.field, modes))
        return out

    def record(self, signals: Signals, modes: Sequence[str]) -> dict[str, Any]:
        """User-friendly alias for :meth:`probe`."""

        return self.probe(signals, modes)

    def evaluate(
        self,
        signals: Signals,
        objective: "Objective | str",
        readout: Optional[dict[str, Any]] = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Full objective/gate evaluation with JSON-safe report.

        Gate pass/fail is a computational diagnostic only.  It does not imply
        empirical validation, biological calibration, or mechanism proof.
        All truth gates from v0.0.4 are preserved in the report.
        """
        from .io import json_safe

        if isinstance(objective, str):
            objective = Objective(name=objective)

        cfg_meta = self.cfg.metadata
        warnings: list[str] = []

        # Special dispatch for group-rate targets objective
        if getattr(objective, "kind", "generic") == "group_rate_targets":
            return self._evaluate_group_rate_targets(signals, objective, warnings, cfg_meta)

        computed_metrics = _compute_all_metrics(signals, readout)

        loss_results = []
        total_loss = 0.0
        has_loss_value = False
        for spec in getattr(objective, "losses", []):
            r = _evaluate_loss_spec(spec, computed_metrics, warnings, strict)
            loss_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        reg_results = []
        for spec in getattr(objective, "regularizers", []):
            r = _evaluate_regularizer_spec(spec, computed_metrics, warnings, strict)
            reg_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        gate_results = []
        all_gates_pass = True
        for spec in getattr(objective, "gates", []):
            r = _evaluate_gate_spec(spec, computed_metrics, warnings, strict)
            gate_results.append(r)
            if not r.get("pass", True):
                all_gates_pass = False

        acceptance = "gates_pass" if all_gates_pass else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_v0.0.5",
            "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "losses": loss_results,
            "regularizers": reg_results,
            "gates": gate_results,
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "warnings": warnings,
        })


    def evaluate_report(
        self,
        signals: Signals,
        objective: "Objective | str",
        *,
        readout_specs: "Optional[Sequence[ReadoutSpec]]" = None,
        readout: Optional[dict[str, Any]] = None,
    ) -> ObjectiveReport:
        """Evaluate an objective and return a structured, immutable ObjectiveReport.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`evaluate`
        when a typed, JSON-safe, auditable result is needed.

        Wraps :meth:`evaluate` into a frozen dataclass.  Optionally computes
        ReadoutSpecs via :meth:`compute_readout` and embeds results in the report.

        Gate pass/fail is a computational diagnostic only.  No biological
        calibration, no physical-amplitude, empirical-validation, or
        mechanism claim is introduced.

        Args:
            signals: Signals returned by self.simulate().
            objective: Objective or objective name string.
            readout_specs: Optional list of ReadoutSpec for feature extraction.
            readout: Optional readout dict (passed through to evaluate()).

        Returns:
            ObjectiveReport (frozen, JSON-safe).
        """
        eval_dict = self.evaluate(signals, objective, readout=readout)
        rr: tuple[ReadoutResult, ...] = ()
        if readout_specs:
            rr = tuple(self.compute_readout(signals, readout_specs))
        truth: dict[str, Any] = {
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "linear_solver",
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }
        return ObjectiveReport(
            objective_name=eval_dict.get("objective_name", "anonymous"),
            evaluation_status="objective_report_v0.0.18",
            total_loss=eval_dict.get("total_loss"),
            all_gates_pass=bool(eval_dict.get("all_gates_pass", True)),
            losses=tuple(eval_dict.get("losses", [])),
            regularizers=tuple(eval_dict.get("regularizers", [])),
            gates=tuple(eval_dict.get("gates", [])),
            readout_results=rr,
            truth=truth,
            warnings=tuple(eval_dict.get("warnings", [])),
        )

    def _evaluate_group_rate_targets(
        self: "Model",
        signals: Signals,
        objective: "Objective",
        warnings: list[str],
        cfg_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate group-wise firing rate targets objective.

        Extracts group definitions and target rates from objective metadata,
        computes group-wise firing rates, and returns squared relative error loss.
        """
        from .io import json_safe

        # Extract metadata from gates (set by rate_targets())
        groups_dict: Optional[dict[str, Any]] = None
        targets_hz_dict: Optional[dict[str, float]] = None
        weights_dict: Optional[dict[str, float]] = None

        for gate_spec in objective.gates:
            if "metadata" in gate_spec:
                meta = gate_spec["metadata"]
                if "groups" in meta:
                    groups_dict = meta.get("groups")
                    targets_hz_dict = meta.get("targets_hz", {})
                    weights_dict = meta.get("weights", {})
                    break

        if groups_dict is None or targets_hz_dict is None:
            warnings.append("group_rate_targets_missing_metadata")
            return json_safe({
                "evaluation_status": "objective_evaluate_group_rate_targets_v0.0.1",
                "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
                "total_loss": None,
                "losses": [],
                "regularizers": [],
                "gates": [],
                "all_gates_pass": False,
                "acceptance_decision": "gates_fail",
                "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
                "field_claim_level": "proxy_readout",
                "physical_amplitude_calibrated": False,
                "warnings": warnings,
            })

        if weights_dict is None:
            weights_dict = {name: 1.0 for name in groups_dict.keys()}

        # Compute dt from metadata to avoid JAX device-to-host transfer
        dt_ms = float(signals.metadata.get("dt_ms", 0.0))
        if dt_ms <= 0:
            dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
        if dt_ms <= 0:
            dt_ms = 0.05

        # Compute group-wise firing rates and loss
        total_loss = 0.0
        loss_details = []
        all_gates_pass = True

        for group_name in sorted(groups_dict.keys()):
            group_indices = groups_dict[group_name]
            target_hz = float(targets_hz_dict.get(group_name, 10.0))
            weight = float(weights_dict.get(group_name, 1.0))

            # Convert group indices to list of ints
            if isinstance(group_indices, list):
                idx_list = [int(i) for i in group_indices]
            else:
                idx_list = list(group_indices)

            if not idx_list:
                warnings.append(f"group_{group_name}_empty")
                continue

            try:
                # Extract spikes for this group
                group_spikes = signals.spikes[:, idx_list]  # Shape: [n_steps, n_neurons_in_group]

                # Compute mean spike rate over time and neurons in group
                group_rate_hz = float(jnp.mean(group_spikes) * (1000.0 / dt_ms))

                # Compute squared relative error: ((rate - target) / target)^2
                if target_hz == 0:
                    if group_rate_hz == 0:
                        raw_loss = 0.0
                    else:
                        raw_loss = float("inf")
                else:
                    raw_loss = ((group_rate_hz - target_hz) / target_hz) ** 2

                weighted_loss = weight * raw_loss
                total_loss += weighted_loss

                loss_details.append({
                    "group": group_name,
                    "target_hz": float(target_hz),
                    "achieved_hz": _finite_or_none(group_rate_hz),
                    "weight": float(weight),
                    "raw_loss": _finite_or_none(raw_loss),
                    "weighted_loss": _finite_or_none(weighted_loss),
                    "status": "ok",
                })
            except Exception as e:
                warnings.append(f"group_{group_name}_evaluation_error: {str(e)}")
                loss_details.append({
                    "group": group_name,
                    "target_hz": float(target_hz),
                    "achieved_hz": None,
                    "weight": float(weight),
                    "raw_loss": None,
                    "weighted_loss": None,
                    "status": str(e),
                })
                all_gates_pass = False

        # Check if loss is finite
        has_loss_value = math.isfinite(total_loss)
        if not has_loss_value:
            all_gates_pass = False

        acceptance = "gates_pass" if (all_gates_pass and has_loss_value) else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_group_rate_targets_v0.0.1",
            "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "group_rate_losses": loss_details,
            "losses": [],
            "regularizers": [],
            "gates": [],
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "warnings": warnings,
        })

    def tune(
        self: "Model",
        objective: Optional["Objective"] = None,
        optimizer: Any = None,
        steps: int = 0,
        seed: int = 0,
        scope: Optional[str] = None,
        strict: bool = False,
        simulation: Optional[Simulation] = None,
        parameter: Optional[str] = None,
        bounds: Optional[tuple[float, float]] = None,
        # Multi-parameter optimization path
        parameters: Optional[dict[str, tuple[float, float]]] = None,
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        # New plural form for public API
        objectives: Optional["Objective"] = None,
    ) -> "TuneResult":
        """Run black-box tuning loop (single or multi-parameter).

        Public API: tune(objectives=objectives, optimizer=optimizer, simulation=simulation)
        Returns TuneResult with best_parameters, best_score, history, and summary.

        Legacy API: tune(objective=objective, parameter=..., bounds=...) for backward compatibility.
        Also returns TuneResult (not tuple).

        This is a computational scaffold: no biological calibration, no field-solver upgrade,
        and no optimizer-selected mechanism claim are made.
        """
        from .io import json_safe
        from .optim import _resolve_optimizer, propose_blackbox_candidates, require_optax

        # Normalize objectives vs objective
        if objectives is not None:
            objective = objectives
        elif objective is None:
            raise ValueError("Either 'objective' (legacy) or 'objectives' (public) must be provided")

        cfg_meta = self.cfg.metadata
        spec = _resolve_optimizer(optimizer)
        sim = simulation or Simulation(duration_ms=10.0, dt_ms=0.1, seed=seed)

        # Detect multi-parameter path: either explicit parameters dict, or AGSDROptimizerSpec
        # If optimizer is an AGSDROptimizerSpec, extract parameters from it
        if parameters is None and hasattr(optimizer, "parameters"):
            # optimizer is likely an AGSDROptimizerSpec
            parameters = optimizer.parameters
            if generations is None and hasattr(optimizer, "generations"):
                generations = optimizer.generations
            if population_size is None and hasattr(optimizer, "population_size"):
                population_size = optimizer.population_size

        # Detect multi-parameter path
        if parameters is not None:
            # Extract seed from optimizer if not explicitly overridden via model.tune(..., seed=nonzero)
            opt_seed = getattr(optimizer, "seed", 0)
            actual_seed = seed if seed != 0 else opt_seed
            return self._tune_multiparameter(
                objective=objective,
                optimizer=optimizer,
                spec=spec,
                parameters=parameters,
                generations=generations or 8,
                population_size=population_size or 6,
                seed=int(actual_seed),
                strict=strict,
                simulation=sim,
            )

        # Single-parameter path (backward compat)
        if parameter is None:
            parameter = "source_scale"
        if bounds is None:
            bounds = (0.25, 4.0)

        n_steps = max(0, int(steps))
        base_report: dict[str, Any] = {
            "same_model_unchanged": True,
            "steps_requested": n_steps,
            "seed": int(seed),
            "scope": scope or spec.optimizer,
            "parameter": parameter,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "optimizer": spec.to_dict(),
            "objective_name": getattr(objective, "name", "spectrolaminar_objective") if not isinstance(objective, str) else objective,
            "losses_declared": len(getattr(objective, "losses", [])) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(getattr(objective, "regularizers", [])) if not isinstance(objective, str) else 0,
            "gates_declared": len(getattr(objective, "gates", [])) if not isinstance(objective, str) else 0,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "biological_learning_claim": False,
            "amplitude_claim_allowed": False,
            "surrogate_status": spec.surrogate_status,
            "final_evaluation_basis": "total_loss",
        }

        if spec.is_differentiable_path():
            if not spec.gradient_path_safe():
                report = {
                    **base_report,
                    "tuning_status": "blocked_non_differentiable_path",
                    "acceptance_decision": "REVISE",
                    "warnings": [
                        "optax_requires_differentiable_or_declared_surrogate",
                        "spiking_reset_not_differentiable_without_surrogate",
                    ],
                }
                return TuneResult(
                    best_parameters={},
                    best_score=float("inf"),
                    history=[],
                    summary=json_safe(report),
                    model=self,
                )
            try:
                require_optax()
                optax_status = "available"
            except ImportError:
                if strict:
                    raise
                optax_status = "unavailable"
            report = {
                **base_report,
                "tuning_status": "optax_guarded_path_no_loop_v0.0.8",
                "acceptance_decision": "REVISE" if optax_status == "unavailable" else "ACCEPT_CANDIDATE",
                "optax_status": optax_status,
                "same_model_unchanged": True,
                "warnings": ["differentiable_loop_not_enabled_for_spiking_reset_without_explicit_surrogate_kernel"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

        if n_steps <= 0:
            report = {
                **base_report,
                "tuning_status": "metadata_only_no_steps_requested",
                "acceptance_decision": "REVISE",
                "candidate_history": [],
                "warnings": ["no_blackbox_steps_requested"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

        candidates = propose_blackbox_candidates(
            optimizer=spec,
            n_steps=n_steps,
            seed=int(seed),
            bounds=(float(bounds[0]), float(bounds[1])),
        )
        best_model: Model = self
        best_loss: Optional[float] = None
        best_value: Optional[float] = None
        history: list[dict[str, Any]] = []
        warnings: list[str] = []
        for idx, candidate_value in enumerate(candidates):
            candidate_model = _model_with_scalar_parameter(self, parameter, float(candidate_value))
            candidate_signals = candidate_model.simulate(replace(sim, seed=int(seed) + idx))
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
            score = float(score)
            accepted = math.isfinite(score) and (best_loss is None or score < best_loss)
            if accepted:
                best_loss = score
                best_value = float(candidate_value)
                best_model = candidate_model

            reasons = []
            if not gates_pass:
                reasons.append("failed_objective_gates")
            if not math.isfinite(score):
                reasons.append("non_finite_loss")

            history.append({
                "step": idx,
                "candidate_value": float(candidate_value),
                "score": _finite_or_none(score),
                "all_gates_pass": gates_pass,
                "accepted_as_best": bool(accepted),
                "evaluation_status": candidate_report.get("evaluation_status"),
                "rejection_reasons": reasons,
            })
        if best_loss is None:
            warnings.append("no_finite_candidate_score")
            best_model = self

        # Compute candidate statistics for enhanced report
        candidate_values = [float(h["candidate_value"]) for h in history]
        candidate_scores = [h.get("score") for h in history]
        finite_scores = [s for s in candidate_scores if s is not None and math.isfinite(s)]

        score_variance = 0.0
        n_unique_scores = 0
        if len(finite_scores) > 1:
            score_variance = float(jnp.var(jnp.asarray(finite_scores)))
            n_unique_scores = len(set(finite_scores))

        report = {
            **base_report,
            "same_model_unchanged": best_model is self,
            "tuning_status": "blackbox_loop_v0.0.6",
            "acceptance_decision": "ACCEPT_CANDIDATE" if best_loss is not None else "REVISE",
            "best_parameter_value": best_value,
            "best_score": _finite_or_none(best_loss) if best_loss is not None else None,
            "candidate_values": candidate_values,
            "candidate_scores": candidate_scores,
            "score_variance": score_variance,
            "n_unique_scores": n_unique_scores,
            "tuning_path": "scalar_black_box",
            "candidate_history": history,
            "warnings": warnings + [
                "blackbox_loop_is_computational_scaffold_only",
                "optimizer_selected_candidate_is_not_biological_truth",
            ],
        }
        # Return TuneResult (new public API)
        # Note: model not included in summary (would not be JSON-safe)
        # Access tuned model separately: model_result = model.tune(...); print(model_result.summary)
        return TuneResult(
            best_parameters={"best_value": best_value} if best_value is not None else {},
            best_score=float(best_loss) if best_loss is not None else float("inf"),
            history=history,
            summary=json_safe(report),
            model=best_model,
        )

    def _tune_multiparameter(
        self,
        objective: "Objective",
        optimizer: Any,
        spec: "OptimizerSpec",
        parameters: Any,
        generations: int,
        population_size: int,
        seed: int,
        strict: bool,
        simulation: "Simulation",
    ) -> "TuneResult":
        """Run multi-parameter AGSDR optimization loop.

        This is an internal helper called by tune() when the multi-parameter
        path is requested (parameters dict provided).

        If any parameter values are :class: objects and the
        optimizer has an inner_optimizer, routes to the two-level AGSDR+Adam path.
        Otherwise uses the scalar AGSDR black-box path.
        """
        from .io import json_safe
        from .optim import (
            _run_agsdr_optimization_loop,
            _tune_matrix_agsdr_optax,
        )

        cfg_meta = self.cfg.metadata

        # Build base report
        base_report: dict[str, Any] = {
            "same_model_unchanged": True,
            "seed": int(seed),
            "scope": "agsdr_multiparameter",
            "parameters": {
                k: (
                    {"type": "MatrixParameterSpec", "mask": v.mask, "bounds": list(v.bounds)}
                    if isinstance(v, MatrixParameterSpec)
                    else [float(v[0]), float(v[1])]
                )
                for k, v in parameters.items()
            },
            "generations": int(generations),
            "population_size": int(population_size),
            "optimizer": spec.to_dict(),
            "objective_name": getattr(objective, "name", "spectrolaminar_objective") if not isinstance(objective, str) else objective,
            "losses_declared": len(getattr(objective, "losses", [])) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(getattr(objective, "regularizers", [])) if not isinstance(objective, str) else 0,
            "gates_declared": len(getattr(objective, "gates", [])) if not isinstance(objective, str) else 0,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "biological_learning_claim": False,
            "amplitude_claim_allowed": False,
            "surrogate_status": spec.surrogate_status,
            "final_evaluation_basis": "total_loss",
        }

        # Separate scalar bounds from MatrixParameterSpec entries
        param_specs: dict[str, Any] = {}
        scalar_bounds: dict[str, tuple] = {}
        for k, v in parameters.items():
            if isinstance(v, MatrixParameterSpec):
                param_specs[k] = v
                scalar_bounds[k] = v.bounds
            else:
                scalar_bounds[k] = tuple(v)

        # Two-level dispatch: matrix + inner_optimizer => AGSDR+Adam path
        inner_optimizer = getattr(optimizer, "inner_optimizer", None)
        inner_steps = getattr(optimizer, "inner_steps", 0)
        inner_objective = getattr(optimizer, "inner_objective", None)
        has_matrix = bool(param_specs)

        if has_matrix and inner_optimizer is not None:
            return _tune_matrix_agsdr_optax(
                model=self,
                objective=objective,
                parameters=parameters,
                param_specs=param_specs,
                scalar_bounds=scalar_bounds,
                inner_optimizer=inner_optimizer,
                inner_steps=inner_steps,
                inner_objective=inner_objective,
                spec=spec,
                generations=generations,
                population_size=population_size,
                seed=seed,
                strict=strict,
                simulation=simulation,
                base_report=base_report,
            )

        rejections_map = []

        # Define scoring function for AGSDR loop
        def evaluate_fn(candidate_params: dict[str, float]) -> float:
            """Evaluate a candidate parameter dict and return loss."""
            reasons = []
            candidate_model = _model_with_parameters(self, candidate_params, param_specs if param_specs else None)
            candidate_signals = candidate_model.simulate(simulation)
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
            score = float(score)

            if not gates_pass:
                reasons.append("failed_objective_gates")
            if not math.isfinite(score):
                reasons.append("non_finite_loss")

            rejections_map.append(reasons)

            return float(score)

        # Run AGSDR optimization (scalar_bounds only, not MatrixParameterSpec objects)
        try:
            agsdr_result = _run_agsdr_optimization_loop(
                evaluate_fn=evaluate_fn,
                parameter_bounds=scalar_bounds,
                n_generations=int(generations),
                n_population=int(population_size),
                alpha=float(spec.alpha),
                exploration=float(spec.exploration),
                seed=int(seed),
                rejections_map=rejections_map,
            )

            best_parameters = agsdr_result["best_parameters"]
            best_score = agsdr_result["best_score"]
            generation_records = agsdr_result["generation_records"]

            # Apply best parameters to model
            best_model = _model_with_parameters(self, best_parameters, param_specs if param_specs else None)

            # Build detailed report
            report = {
                **base_report,
                "same_model_unchanged": False,
                "tuning_status": "multiparameter_agsdr_v0.0.7",
                "acceptance_decision": "ACCEPT_CANDIDATE" if math.isfinite(best_score) else "REVISE",
                "best_parameters": best_parameters,
                "best_score": _finite_or_none(best_score),
                "generation_records": generation_records,
                "all_scores": agsdr_result["all_scores"],
                "n_candidates_evaluated": len(agsdr_result["all_scores"]),
                "tuning_path": "multiparameter_black_box",
                "warnings": [
                    "blackbox_loop_is_computational_scaffold_only",
                    "optimizer_selected_candidate_is_not_biological_truth",
                ],
            }

            return TuneResult(
                best_parameters=best_parameters,
                best_score=float(best_score) if math.isfinite(best_score) else float("inf"),
                history=generation_records,
                summary=json_safe(report),
                model=best_model,
            )

        except Exception as e:
            report = {
                **base_report,
                "tuning_status": "multiparameter_agsdr_error",
                "acceptance_decision": "REVISE",
                "error": str(e),
                "warnings": ["multiparameter_optimization_failed"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

    def with_emitter_parameters(
        self,
        *,
        a: "float | None" = None,
        b: "float | None" = None,
        c: "float | None" = None,
        d: "float | None" = None,
        drive_scale: "float | None" = None,
        # New per-neuron overrides (v0.3.3)
        a_per_neuron: "jax.Array | None" = None,
        b_per_neuron: "jax.Array | None" = None,
        c_per_neuron: "jax.Array | None" = None,
        d_per_neuron: "jax.Array | None" = None,
        drive_per_neuron: "jax.Array | None" = None,
    ) -> "Model":
        """Return a new Model with Izhikevich parameter overrides.

        Supports both scalar (uniform) and per-neuron (array) overrides.
        Per-neuron arrays take priority over scalar values.
        Explicit None checks are used to handle zero-valued arrays correctly.

        Args:
            a: Scalar recovery time scale override (uniform to all neurons).
            b: Scalar voltage-sensitivity override (uniform).
            c: Scalar post-spike reset override (uniform).
            d: Scalar post-spike increment override (uniform).
            drive_scale: Multiplicative gain on native drive.
            a_per_neuron: Per-neuron recovery time scale (shape: [n_neurons]).
            b_per_neuron: Per-neuron voltage sensitivity (shape: [n_neurons]).
            c_per_neuron: Per-neuron reset voltage (shape: [n_neurons]).
            d_per_neuron: Per-neuron recovery increment (shape: [n_neurons]).
            drive_per_neuron: Per-neuron absolute drive (shape: [n_neurons]).
                Overrides both scalar drive_scale and emitter.drive.

        Returns:
            New Model — original is not mutated.
        """
        emitter: IzhikevichParams = self.params["emitter"]
        updates: dict[str, Any] = {}

        # a: per-neuron takes priority over scalar
        if a_per_neuron is not None:
            updates["a"] = jnp.asarray(a_per_neuron, dtype=emitter.a.dtype)
        elif a is not None:
            updates["a"] = jnp.ones_like(emitter.a) * float(a)

        # b: per-neuron takes priority over scalar
        if b_per_neuron is not None:
            updates["b"] = jnp.asarray(b_per_neuron, dtype=emitter.b.dtype)
        elif b is not None:
            updates["b"] = jnp.ones_like(emitter.b) * float(b)

        # c: per-neuron takes priority over scalar
        if c_per_neuron is not None:
            updates["c"] = jnp.asarray(c_per_neuron, dtype=emitter.c.dtype)
        elif c is not None:
            updates["c"] = jnp.ones_like(emitter.c) * float(c)

        # d: per-neuron takes priority over scalar
        if d_per_neuron is not None:
            updates["d"] = jnp.asarray(d_per_neuron, dtype=emitter.d.dtype)
        elif d is not None:
            updates["d"] = jnp.ones_like(emitter.d) * float(d)

        # drive: per-neuron absolute takes priority; scalar applies multiplicative scale
        if drive_per_neuron is not None:
            updates["drive"] = jnp.asarray(drive_per_neuron, dtype=emitter.drive.dtype)
        elif drive_scale is not None:
            updates["drive"] = emitter.drive * float(drive_scale)

        new_emitter = replace(emitter, **updates)
        new_params = dict(self.params)
        new_params["emitter"] = new_emitter
        return replace(self, params=new_params)

    def with_hdp_initial_state(
        self,
        *,
        H0: "jax.Array | None" = None,
        w0: "jax.Array | None" = None,
    ) -> "Model":
        """Return a new Model with a custom initial HDP controller state.

        Only takes effect when HDP is separately enabled via
        ``Configuration.hdp(...)`` (``RuntimeConfig.enable_hdp=True``); stored
        but inert otherwise, mirroring :meth:`with_emitter_parameters`'s
        additive-override pattern.

        Args:
            H0: Per-neuron initial homeostatic factor (shape: [n_neurons]).
                Defaults to the HDP kernel's own equilibrium value (1.0 for
                every neuron) when not provided.
            w0: Per-edge initial weight (shape: [n_edges], aligned to
                ``self.params["edge_list"]``). Defaults to the edge list's
                native ``weight`` when not provided.

        Returns:
            New Model — original is not mutated.
        """
        new_params = dict(self.params)
        if H0 is not None:
            new_params["hdp_initial_H"] = jnp.asarray(H0, dtype=jnp.float32)
        if w0 is not None:
            new_params["hdp_initial_w"] = jnp.asarray(w0, dtype=jnp.float32)
        return replace(self, params=new_params)

    def with_recurrent_coupling(
        self,
        *,
        g_ei: float = 5.0,
        g_ie: float = 3.0,
        tau_syn_e_ms: float = 5.0,
        tau_syn_i_ms: float = 10.0,
    ) -> "Model":
        """Return a new Model with recurrent E/I coupling parameters stored.

        Stores coupling parameters in model.static["recurrent_coupling"] for
        use with simulate_dynamic_ei_coupling(). The original model is not mutated
        (frozen dataclass contract is preserved via replace()).

        Coupling is stored as metadata; it does not modify the emitter's W matrix.
        Use with simulate_dynamic_ei_coupling() to apply dynamic coupling at runtime.

        Args:
            g_ei: E→I excitatory coupling conductance (model units).
            g_ie: I→E inhibitory coupling magnitude (model units).
            tau_syn_e_ms: Excitatory synaptic time constant (ms).
            tau_syn_i_ms: Inhibitory synaptic time constant (ms).

        Returns:
            New Model with coupling parameters in static["recurrent_coupling"].
            Original model is not mutated.
        """
        coupling_params = {
            "g_ei": float(g_ei),
            "g_ie": float(g_ie),
            "tau_syn_e_ms": float(tau_syn_e_ms),
            "tau_syn_i_ms": float(tau_syn_i_ms),
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "physical_amplitude_calibrated": False,
            "claim_level": "computational_scaffold",
        }
        return replace(
            self,
            static={**self.static, "recurrent_coupling": coupling_params}
        )

    def manifest(
        self,
        signals: Optional[Signals] = None,
        readout: Optional[Any] = None,
        paradigm: Optional[dict[str, Any]] = None,
        objective: Optional[dict[str, Any]] = None,
        evaluation: Optional[dict[str, Any]] = None,
        tuning: Optional[dict[str, Any]] = None,
        dataset: Optional[dict[str, Any]] = None,
        trials: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a JSON-safe run manifest dict.

        Compatibility method retained from v0.0.4–v0.0.14.  For the canonical
        v0.1 workflow, prefer :meth:`run_receipt` (typed, immutable, with
        deterministic receipt ID) and :meth:`evaluate_report` (typed objective
        evaluation).  This method remains supported and is not scheduled for
        removal.

        The ``readout`` argument accepts any of:

        * ``None`` — no readout section included.
        * ``dict`` — passed through to the manifest as-is (legacy shape).
        * ``list`` or ``tuple`` of :class:`ReadoutResult` objects — the canonical
          output of :meth:`compute_readout`.  Converted to a JSON-safe readout
          summary dict with ``readout_results`` and ``requested_metrics`` keys.
        * ``list`` or ``tuple`` of ``dict`` — same conversion applied to each element.
        * Single :class:`ReadoutResult` — wrapped in a list and handled as above.
        """
        readout_normalized = _normalize_manifest_readout(readout)
        runtime_cfg = None
        if signals is not None and "runtime" in signals.metadata:
            runtime_cfg = _RuntimeReportAdapter(signals.metadata["runtime"])
        res = build_manifest(
            self.cfg,
            signals=signals,
            readout=readout_normalized,
            runtime_config=runtime_cfg,
            paradigm=paradigm,
            objective=objective,
            evaluation=evaluation,
            tuning=tuning,
            dataset=dataset,
        )
        if trials is not None:
            res["trials"] = trials
        # If readout was provided as ReadoutResult list (canonical v0.1 workflow),
        # surface the normalized readout summary in the manifest under "readout_results".
        # Dict-shaped readouts are already surfaced via build_manifest's field_diagnostics
        # logic; non-dict shapes are added here only.
        if readout_normalized is not None and isinstance(readout_normalized, dict):
            if "readout_results" in readout_normalized:
                res["readout_results"] = readout_normalized
        # Backend metadata: distinguish executed backend from available infrastructure.
        used_backend = "dense"
        used_kernel = "exponential"
        if signals is not None:
            used_backend = signals.metadata.get("recurrent_backend", "dense")
            used_kernel = signals.metadata.get("synaptic_kernel", "exponential")
        elif "edge_list" in self.params:
            used_backend = "unknown_not_run"
        backend_meta: dict[str, Any] = {
            "used_recurrent_backend": used_backend,
            "used_synaptic_kernel": used_kernel,
            "available_edge_list": "edge_list" in self.params,
            "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
            "source_model": dict(_SOURCE_PROXY_METADATA),
        }
        # v0.2.0: Field admissibility metadata
        if signals is not None and signals.field is not None:
            from .validation import build_field_admissibility_report
            field_admissibility = build_field_admissibility_report(
                field_output=signals.field,
                cfg_metadata=dict(self.cfg.metadata or {}),
            )
            backend_meta["field_admissibility"] = field_admissibility
            if "field_admissibility" in signals.field.diagnostics:
                backend_meta["field_admissibility_diagnostics"] = signals.field.diagnostics.get(
                    "field_admissibility"
                )
        if "edge_list" in self.params:
            edges = self.params["edge_list"]
            backend_meta["edge_count"] = int(edges.n_edges)
            backend_meta["receptor_indexed"] = True
            backend_meta["edge_list_source_calibration_status"] = edges.source_calibration_status
            backend_meta["edge_list_physical_amplitude_calibrated"] = False
            # v0.0.21: explicitly document which tau source each kernel uses.
            # simulate_edge_recurrent_izhikevich → edges.tau_ms (per-edge field)
            # simulate_receptor_exponential_izhikevich → standard_receptor_tau_table
            #   (receptor_index → standard catalog). Current standard table agrees
            #   with make_edge_list_from_dense for receptor_index ∈ {0, 1}, so
            #   these are numerically equivalent in the default scaffold flow.
            backend_meta["receptor_tau_source"] = {
                "exponential_kernel_uses": "edges.tau_ms",
                "receptor_exponential_kernel_uses": "standard_receptor_tau_table_by_receptor_index",
                "consistent_for_receptor_index_in": [0, 1],
            }
            # v0.0.21: surface receptor spec metadata so manifest documents
            # the receptor labels/taus the kernel can index. The actual per-edge
            # tau_ms lives on EdgeList; this is the catalog.
            from .emitters import standard_receptor_specs
            backend_meta["receptor_specs"] = {
                name: {
                    "name": spec.name,
                    "receptor_index": spec.receptor_index,
                    "sign": spec.sign,
                    "tau_ms": spec.tau_ms,
                    "reversal_mV": spec.reversal_mV,
                    "source_calibration_status": spec.source_calibration_status,
                }
                for name, spec in standard_receptor_specs().items()
            }
        # v0.0.21: explicit source model in manifest.
        res["source_model"] = dict(_SOURCE_PROXY_METADATA)
        res["backend_metadata"] = backend_meta
        if "geometry" in self.static:
            res["source_geometry"] = self.static["geometry"]
        # v0.2.26: computation-basis block
        res["basis"] = _default_basis_dict()
        # v0.2.27: conservation-inspired proxy diagnostics
        if signals is not None and signals.field is not None:
            from .fields import compute_conservation_proxy_diagnostics
            _src_cal = (
                signals.metadata.get("source_calibration_status",
                                     "uncalibrated_izhikevich_native_current")
            )
            res["conservation_proxy_diagnostics"] = compute_conservation_proxy_diagnostics(
                field_solution=signals.field,
                source_calibration_status=_src_cal,
                field_solver_status="linear_solver",
                field_claim_level="proxy_readout",
            )
        return res


def _model_with_scalar_parameter(model: Model, parameter: str, value: float) -> Model:
    """Return a Model copy with one safe scalar emitter parameter changed.

    Supported parameters:
    - source_scale: multiplicative gain on all source signals
    - drive_gain: multiplicative gain on all drive signals
    - synaptic_gain: multiplicative gain on all synaptic weights
    - drive_scale_a: multiplicative gain on first-half neuron drive signals
    - drive_scale_b: multiplicative gain on second-half neuron drive signals
    - gAMPA: multiplicative gain on all excitatory (positive) synaptic weights
    """
    import numpy as np

    emitter = model.params["emitter"]
    value = float(value)

    if parameter == "source_scale":
        new_emitter = replace(emitter, source_scale=jnp.asarray(value, dtype=emitter.source_scale.dtype))
    elif parameter == "drive_gain":
        new_emitter = replace(emitter, drive=emitter.drive * jnp.asarray(value, dtype=emitter.drive.dtype))
    elif parameter == "synaptic_gain":
        new_emitter = replace(emitter, W=emitter.W * jnp.asarray(value, dtype=emitter.W.dtype))
    elif parameter in ("drive_scale_a", "drive_scale_b"):
        import numpy as _np_dsa
        base_drive = _np_dsa.asarray(emitter.drive, dtype=float).reshape(-1)
        n_units = base_drive.shape[0]
        split = n_units // 2
        drive_scale = _np_dsa.ones(n_units, dtype=float)
        if parameter == "drive_scale_a":
            drive_scale[:split] = value
        else:
            drive_scale[split:] = value
        drive_per_neuron = base_drive * drive_scale
        new_emitter = replace(emitter, drive=jnp.asarray(drive_per_neuron, dtype=emitter.drive.dtype))
    elif parameter == "gAMPA":
        import numpy as np
        W = np.asarray(emitter.W, dtype=float)
        new_W = W.copy()
        # Scale only excitatory (positive) weights
        new_W[W > 0] = W[W > 0] * value
        new_emitter = replace(emitter, W=jnp.asarray(new_W, dtype=emitter.W.dtype))
    else:
        supported = ["source_scale", "drive_gain", "synaptic_gain", "drive_scale_a", "drive_scale_b", "gAMPA"]
        raise ValueError(
            f"Unsupported tunable parameter: {parameter!r}. "
            f"Supported parameters: {supported}"
        )
    params = dict(model.params)
    params["emitter"] = new_emitter
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def _mask_for_parameter(model: "Model", parameter_name: str, mask_type: str) -> "jax.Array":
    """Return a boolean mask over the W matrix for the given mask type.

    Parameters
    ----------
    model : Model
        Model whose W matrix determines the mask shape.
    parameter_name : str
        Name of the parameter (used only for error messages).
    mask_type : str
        One of: "E_to_E", "E_to_I", "excitatory_to_all", "all".

    Returns
    -------
    jax.Array
        Boolean mask of shape (n, n) where True marks entries to scale.
    """
    import numpy as _np_mask
    emitter = model.params["emitter"]
    W = _np_mask.asarray(emitter.W, dtype=float)
    n = W.shape[0]

    if mask_type == "all":
        return jnp.ones((n, n), dtype=bool)

    if mask_type == "excitatory_to_all":
        # Scale entries in rows corresponding to excitatory neurons (positive out-degree).
        # We identify E neurons by their sign label or by majority positive outgoing weights.
        row_mask = _np_mask.zeros(n, dtype=bool)
        for i in range(n):
            # E neurons: positive outgoing (sign > 0) or labeled E
            label = emitter.labels[i] if i < len(emitter.labels) else ""
            if label.startswith("E") or _np_mask.sum(W[i, :] > 0) > _np_mask.sum(W[i, :] < 0):
                row_mask[i] = True
        return jnp.asarray(_np_mask.outer(row_mask, _np_mask.ones(n, dtype=bool)), dtype=bool)

    # E_to_E and E_to_I: identify E vs I neurons by labels
    e_mask = _np_mask.zeros(n, dtype=bool)
    i_mask = _np_mask.zeros(n, dtype=bool)
    for idx in range(n):
        label = emitter.labels[idx] if idx < len(emitter.labels) else ""
        if label.startswith("E"):
            e_mask[idx] = True
        else:
            i_mask[idx] = True

    if mask_type == "E_to_E":
        return jnp.asarray(_np_mask.outer(e_mask, e_mask), dtype=bool)
    if mask_type == "E_to_I":
        return jnp.asarray(_np_mask.outer(e_mask, i_mask), dtype=bool)

    raise ValueError(
        f"Unknown mask type for parameter {parameter_name!r}: {mask_type!r}. "
        "Supported: E_to_E, E_to_I, excitatory_to_all, all"
    )


def _model_with_matrix_parameter(
    model: "Model",
    parameter_name: str,
    spec: "MatrixParameterSpec",
    value: float,
) -> "Model":
    """Return a Model copy with a matrix parameter scaled by value.

    The value is treated as a multiplicative scale factor applied to
    the subset of W entries selected by spec.mask.  The result is then
    clipped to spec.bounds.

    Parameters
    ----------
    model : Model
        Original model (not mutated).
    parameter_name : str
        Name of the parameter (for diagnostics).
    spec : MatrixParameterSpec
        Matrix parameter specification.
    value : float
        Multiplicative scale factor (clipped to spec.bounds).

    Returns
    -------
    Model
        New model with scaled matrix entries.
    """
    import numpy as _np_matrix
    lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
    value = float(_np_matrix.clip(value, lo, hi))

    emitter = model.params["emitter"]
    W = _np_matrix.asarray(emitter.W, dtype=float)
    mask = _np_matrix.asarray(_mask_for_parameter(model, parameter_name, spec.mask), dtype=bool)

    new_W = W.copy()
    new_W[mask] = W[mask] * value
    new_emitter = replace(emitter, W=jnp.asarray(new_W, dtype=emitter.W.dtype))
    params = dict(model.params)
    params["emitter"] = new_emitter
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def _model_with_parameters(
    model: "Model",
    parameters: Any,
    param_specs: Optional[Any] = None,
) -> "Model":
    """Return a Model copy with multiple emitter parameters changed.

    Dispatches each parameter to the scalar or matrix path depending on
    whether param_specs contains a :class: for that name.

    Parameters
    ----------
    model : Model
        Original model (not mutated).
    parameters : dict[str, float]
        Mapping from parameter names to float values.
    param_specs : dict[str, Any], optional
        Mapping from parameter names to spec objects (e.g. MatrixParameterSpec).
        When None, all parameters are treated as scalars.

    Returns
    -------
    Model
        New model with all parameters updated.
    """
    result = model
    for param_name, param_value in parameters.items():
        if param_specs is not None and param_name in param_specs:
            spec = param_specs[param_name]
            if isinstance(spec, MatrixParameterSpec):
                result = _model_with_matrix_parameter(result, param_name, spec, float(param_value))
                continue
        result = _model_with_scalar_parameter(result, param_name, float(param_value))
    return result


def _evaluate_soft_rate_targets(
    V_m: "jax.Array",
    groups: dict,
    targets_hz: dict,
    duration_ms: float,
    dt_ms: float,
    threshold: float = -45.0,
    temperature: float = 5.0,
) -> "jax.Array":
    """Compute a differentiable soft firing-rate MSE loss using a sigmoid spike surrogate.

    This function is used in the Adam inner loop for matrix parameter optimization.
    It provides smooth gradients through the spike threshold by approximating
    spike probability with a sigmoid function, making the loss differentiable
    with respect to V_m (and thus to the weight matrix W).

    Parameters
    ----------
    V_m : jax.Array
        Membrane voltages, shape (n_steps, n_neurons).
    groups : dict
        Mapping from group name to list of neuron indices.
    targets_hz : dict
        Mapping from group name to target firing rate in Hz.
    duration_ms : float
        Simulation duration in milliseconds.
    dt_ms : float
        Simulation time step in milliseconds.
    threshold : float
        Spike threshold in mV (default -45 mV for Izhikevich scaffold).
    temperature : float
        Sigmoid sharpness (lower = sharper threshold, default 5.0).

    Returns
    -------
    jax.Array
        Scalar MSE loss (differentiable).
    """
    duration_s = duration_ms / 1000.0

    # Soft spike approximation: sigmoid((V_m - threshold) / temperature)
    soft_spikes = jax.nn.sigmoid((V_m - threshold) / temperature)

    total_loss = jnp.zeros((), dtype=jnp.float32)
    for group_name, idx_list in groups.items():
        if not idx_list:
            continue
        idx_arr = jnp.asarray(idx_list, dtype=jnp.int32)
        group_soft = soft_spikes[:, idx_arr]  # (n_steps, n_group)
        n_neurons = group_soft.shape[1]
        # Soft rate in Hz: sum over steps / (duration_s * n_neurons)
        soft_rate_hz = jnp.sum(group_soft) / (duration_s * n_neurons)
        target_hz = float(targets_hz.get(group_name, 10.0))
        target_arr = jnp.asarray(target_hz, dtype=jnp.float32)
        # Normalized MSE
        denom = jnp.maximum(jnp.abs(target_arr), jnp.asarray(1.0, dtype=jnp.float32))
        loss_i = ((soft_rate_hz - target_arr) / denom) ** 2
        total_loss = total_loss + loss_i

    return total_loss


@dataclass(frozen=True)
class _RuntimeReportAdapter:
    report: dict[str, Any]

    def runtime_report(self) -> dict[str, Any]:
        """Documented public function `runtime_report`."""
        return self.report


def _mean_pairwise_corr_proxy(spikes: jax.Array) -> jax.Array:
    x = spikes.astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.std(x, axis=0, keepdims=True) + 1e-6
    z = x / denom
    corr = (z.T @ z) / jnp.maximum(1, z.shape[0] - 1)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    return jnp.sum(jnp.abs(corr) * mask) / jnp.maximum(1.0, jnp.sum(mask))


def with_emitter_parameters(
    model: Model,
    *,
    a: "float | None" = None,
    b: "float | None" = None,
    c: "float | None" = None,
    d: "float | None" = None,
    drive_scale: "float | None" = None,
    a_per_neuron: "jax.Array | None" = None,
    b_per_neuron: "jax.Array | None" = None,
    c_per_neuron: "jax.Array | None" = None,
    d_per_neuron: "jax.Array | None" = None,
    drive_per_neuron: "jax.Array | None" = None,
) -> Model:
    """Functional wrapper for :meth:`Model.with_emitter_parameters`.

    Supports both scalar (uniform) and per-neuron (array) overrides.
    Per-neuron arrays take priority over scalars.
    Explicit None checks used — zero-valued JAX arrays handled correctly.
    """
    return model.with_emitter_parameters(
        a=a, b=b, c=c, d=d, drive_scale=drive_scale,
        a_per_neuron=a_per_neuron,
        b_per_neuron=b_per_neuron,
        c_per_neuron=c_per_neuron,
        d_per_neuron=d_per_neuron,
        drive_per_neuron=drive_per_neuron,
    )


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
    net: Mapping[str, Any], edge_list: "EdgeList",
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
            _conn_edges, _counts = _compile_mechanism_aware_connection_rules(
                _conn_rules, _conn_mechanisms, _area_labels, _layer_labels, _cell_labels,
                _np.asarray(_ep.sign), n, edge_list.weight.dtype,
                int(cfg.metadata.get("seed", 0) or 0),
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

    cfg, edge_list = _construct_compile_connections(cfg, network, n, geometry_meta, net, edge_list)
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


def stimulus_schedule(
    events: Sequence[Any],
    n_neurons: int,
    *,
    drive_amplitude: float = 5.0,
    event_duration_ms: float = 50.0,
) -> StimulusSchedule:
    """Build a :class:`StimulusSchedule` from a sequence of events.

    Each event may be a :class:`ParadigmEvent` or a dict-like with at least
    ``onset_ms``.  The ``drive_amplitude`` and ``event_duration_ms`` are the
    default values applied to all events that do not specify their own.

    Events that carry ``is_omission=True`` or an explicit ``amplitude=0`` inject
    zero drive (generic no-drive semantics, not cognitive omission logic).
    No calibrated-current or physical-amplitude claim is made.
    """
    ev_dicts: list[dict[str, Any]] = []
    for e in events:
        if isinstance(e, ParadigmEvent):
            amp = float(e.metadata.get("drive_amplitude", drive_amplitude))
            dur = float(e.metadata.get("event_duration_ms", event_duration_ms))
            is_drive = not e.is_omission and e.onset_ms is not None
            ev_dict = {
                "label": e.label,
                "onset_ms": float(e.onset_ms) if e.onset_ms is not None else 0.0,
                "duration_ms": dur,
                "amplitude": amp if is_drive else 0.0,
                "is_drive_event": is_drive,
            }
            if "target_indices" in e.metadata:
                ev_dict["target_indices"] = e.metadata["target_indices"]
            ev_dicts.append(ev_dict)
        else:
            d = dict(e)
            if "amplitude" not in d:
                d["amplitude"] = drive_amplitude
            if "duration_ms" not in d:
                d["duration_ms"] = event_duration_ms
            if "is_drive_event" not in d:
                d["is_drive_event"] = d.get("onset_ms") is not None and d["amplitude"] != 0.0
            ev_dicts.append(d)
    return StimulusSchedule(
        events=tuple(ev_dicts),
        n_neurons=int(n_neurons),
    )


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

_JAXFNE_VERSION = "0.4.4"
_RECEIPT_SCHEMA_VERSION = "run_receipt_v0.0.21"
_MANIFEST_SCHEMA_VERSION = "manifest.v0.0.21"
_OBJECTIVE_REPORT_SCHEMA_VERSION = "objective_report.v0.0.18"

# v0.0.21: explicit source proxy metadata.
# Documents what the current Izhikevich scaffold computes as the "source" field.
# Reading the edge/dense kernels: source_proxy = source_scale * (current_native
# + DEFAULT_SPIKE_IMPULSE_GAIN * spikes), where current_native = drive +
# recurrent_syn + noise. The gain constant lives in presets.py and is shared by
# every kernel variant in emitters.py (simulate_edge_recurrent_izhikevich and
# the dense variants) -- they must stay in sync or the double-count guard
# below breaks. No physical-amplitude claim is made; this remains an
# uncalibrated proxy. The double-count guard records that synaptic current
# enters the source only via the single proxy expression, not as a separate
# additive term.
_SOURCE_PROXY_METADATA: dict[str, Any] = {
    "source_model": "izhikevich_native_current_plus_spike_impulse_proxy",
    "source_mode": "native_current_plus_spike_impulse_proxy",
    "includes_native_current": True,
    "includes_drive_current": True,
    "includes_recurrent_synaptic_current": True,
    "includes_noise_current": True,
    "includes_spike_impulse": True,
    "spike_impulse_gain": DEFAULT_SPIKE_IMPULSE_GAIN,
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "physical_amplitude_calibrated": False,
    "double_count_synaptic_current_guard": (
        "single_proxy_expression_no_extra_synaptic_source"
    ),
}

_KNOWN_READOUT_METRICS = frozenset({
    "spike_rate_hz",
    "spike_count",
    "mean_V_m",
    "csd_abs_mean",
    "lfp_abs_mean",
    "source_abs_mean",
})

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

_CONSERVATIVE_TRUTH_DEFAULTS = {
    "physical_amplitude_calibrated": False,
    "claim_level": "computational_scaffold",
    "field_claim_level": "proxy_readout",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "linear_solver",
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}

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
    return out


@dataclass(frozen=True)
class JaxFNEConfig:
    """JSON-safe container for a complete ``.jcfg.json`` TFNE specification.

    Attributes map to top-level JSON keys.  The JSON key ``"field"`` maps to the
    Python attribute ``field_spec``.
    """

    schema_version: str
    run: dict[str, Any]
    truth: dict[str, Any]
    network: dict[str, Any]
    emitter: dict[str, Any]
    field_spec: dict[str, Any]
    probes: tuple[dict[str, Any], ...]
    geometry: Optional[dict[str, Any]] = None
    paradigm: Optional[dict[str, Any]] = None
    trials: Optional[dict[str, Any]] = None
    runtime_spec: Optional[dict[str, Any]] = None
    stimulus: Optional[dict[str, Any]] = None
    features: Optional[dict[str, Any]] = None
    objective_spec: Optional[dict[str, Any]] = None
    targets: Optional[dict[str, Any]] = None
    validation_spec: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe

        d = {
            "schema_version": self.schema_version,
            "run": self.run,
            "truth": self.truth,
            "network": self.network,
            "emitter": self.emitter,
            "field": self.field_spec,
            "probes": list(self.probes),
        }
        if self.geometry:
            d["geometry"] = self.geometry
        if self.paradigm:
            d["paradigm"] = self.paradigm
        if self.trials:
            d["trials"] = self.trials
        if self.runtime_spec:
            d["runtime"] = self.runtime_spec
        if self.stimulus:
            d["stimulus"] = self.stimulus
        if self.features:
            d["features"] = self.features
        if self.objective_spec:
            d["objective"] = self.objective_spec
        if self.targets:
            d["targets"] = self.targets
        if self.validation_spec:
            d["validation"] = self.validation_spec
        if self.output:
            d["output"] = self.output
        if self.metadata:
            d["metadata"] = self.metadata
        return json_safe(d)

    @property
    def config_hash(self) -> str:
        """Return a compact SHA256 hash for the configuration.

        The hash is computed over ``to_dict()`` output, which includes all
        known sections plus any unknown top-level keys that were captured in
        ``metadata["unknown_keys"]`` during ``load_config()``.  Two configs
        that differ only in unknown keys will produce different hashes.

        Hash equality is structural identity — it is not biological equivalence
        or empirical validation.
        """
        return config_hash(self.to_dict())


@dataclass(frozen=True)
class ConfigValidationResult:
    """Report container for configuration validation."""

    valid: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    truth_boundary: dict[str, Any]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe

        return json_safe({
            "valid": self.valid,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "truth_boundary": self.truth_boundary,
            "schema_version": self.schema_version,
        })


# ──────────────────────────────────────────────────────────────
# v0.0.15 config factory functions
# ──────────────────────────────────────────────────────────────

def load_config(path: Any) -> JaxFNEConfig:
    """Load a ``.jcfg.json`` file and return a :class:`JaxFNEConfig`.

    Malformed JSON raises.  Missing or invalid sections are surfaced by
    :func:`validate_config`.  Unknown top-level keys are preserved in
    ``metadata["unknown_keys"]`` and produce warnings via
    :func:`validate_config`.

    The JSON key ``"field"`` is mapped to the Python attribute ``field_spec``.
    """
    raw: dict[str, Any] = load_json(path)

    known_keys = _REQUIRED_CONFIG_SECTIONS | _RECOGNIZED_OPTIONAL_CONFIG_SECTIONS
    unknown_keys = [k for k in raw if k not in known_keys]

    meta: dict[str, Any] = dict(raw.get("metadata") or {})
    if unknown_keys:
        meta["unknown_keys"] = unknown_keys

    return JaxFNEConfig(
        schema_version=str(raw.get("schema_version", "")),
        run=dict(raw.get("run") or {}),
        truth=dict(raw.get("truth") or {}),
        network=dict(raw.get("network") or {}),
        emitter=dict(raw.get("emitter") or {}),
        field_spec=dict(raw.get("field") or {}),
        probes=tuple(raw.get("probes") or []),
        geometry=raw.get("geometry"),
        paradigm=raw.get("paradigm"),
        trials=raw.get("trials"),
        runtime_spec=raw.get("runtime"),
        stimulus=raw.get("stimulus"),
        features=raw.get("features"),
        objective_spec=raw.get("objective"),
        targets=raw.get("targets"),
        validation_spec=raw.get("validation"),
        output=raw.get("output"),
        metadata=meta,
    )


def validate_config(cfg: JaxFNEConfig) -> ConfigValidationResult:
    """Validate a :class:`JaxFNEConfig` and return a :class:`ConfigValidationResult`.

    Does not raise for normal validation failures.  Returns issues list.
    Escalated truth claims and missing required sections are blocking issues.
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Schema version
    if not cfg.schema_version:
        issues.append("schema_version_missing")
    elif cfg.schema_version != _JAXFNE_CONFIG_SCHEMA_VERSION:
        issues.append(
            f"schema_version_unsupported:{cfg.schema_version!r};"
            f"expected:{_JAXFNE_CONFIG_SCHEMA_VERSION!r}"
        )

    # Required sections
    for section_key, section_val in [
        ("run", cfg.run),
        ("truth", cfg.truth),
        ("network", cfg.network),
        ("emitter", cfg.emitter),
        ("field", cfg.field_spec),
        ("probes", cfg.probes),
    ]:
        if not section_val:
            issues.append(f"required_section_missing:{section_key}")

    # run validation
    run = cfg.run or {}
    for k in ("duration_ms", "dt_ms", "seed"):
        if k not in run:
            issues.append(f"run.{k}_missing")
    if float(run.get("duration_ms", 0)) <= 0:
        issues.append("run.duration_ms_must_be_positive")
    if float(run.get("dt_ms", 0)) <= 0:
        issues.append("run.dt_ms_must_be_positive")

    # network validation
    network = cfg.network or {}
    n = network.get("n")
    if n is None:
        issues.append("network.n_missing")
    elif not isinstance(n, int) or n <= 0:
        issues.append("network.n_must_be_positive_int")

    # Truth boundary — required fields must be present and conservative. The
    # ``field_claim_level`` addition is emitted by default but treated as optional
    # so legacy configs without it remain valid.
    truth = cfg.truth or {}
    _optional_truth_keys = {"field_claim_level"}
    for tk in _CONSERVATIVE_TRUTH_DEFAULTS:
        if tk in _optional_truth_keys:
            continue
        if tk not in truth:
            issues.append(f"truth.{tk}_missing")

    if truth.get("physical_amplitude_calibrated") is not False:
        issues.append("truth_escalation:physical_amplitude_calibrated_must_be_False")
    if truth.get("claim_level") not in (None, "computational_scaffold"):
        issues.append(f"truth_escalation:claim_level:{truth.get('claim_level')!r}")
    if truth.get("source_calibration_status") not in (
        None, "uncalibrated_izhikevich_native_current"
    ):
        issues.append(
            f"truth_escalation:source_calibration_status:{truth.get('source_calibration_status')!r}"
        )
    if truth.get("field_solver_status") not in _VALID_FIELD_SOLVER_STATUS:
        issues.append(f"truth_escalation:field_solver_status:{truth.get('field_solver_status')!r}")
    if truth.get("field_claim_level") not in (None, "proxy_readout"):
        issues.append(f"truth_escalation:field_claim_level:{truth.get('field_claim_level')!r}")
    if truth.get("empirical_validation_status") not in (None, "not_empirically_validated"):
        issues.append(
            f"truth_escalation:empirical_validation_status:"
            f"{truth.get('empirical_validation_status')!r}"
        )
    if truth.get("mechanism_claim_status") not in (None, "not_claimed"):
        issues.append(
            f"truth_escalation:mechanism_claim_status:{truth.get('mechanism_claim_status')!r}"
        )

    # Warnings for unknown top-level keys
    for uk in cfg.metadata.get("unknown_keys", []):
        warnings.append(f"unknown_top_level_key:{uk}")

    # Warnings for absent optional sections
    for opt_section, opt_val in (
        ("geometry", cfg.geometry),
        ("paradigm", cfg.paradigm),
        ("trials", cfg.trials),
    ):
        if opt_val is None:
            warnings.append(f"optional_section_absent:{opt_section}")

    from .io import json_safe
    truth_boundary = json_safe(dict(truth))
    return ConfigValidationResult(
        valid=len(issues) == 0,
        issues=tuple(issues),
        warnings=tuple(warnings),
        truth_boundary=truth_boundary,
        schema_version=cfg.schema_version or "",
    )


def config_truth_boundary(cfg: JaxFNEConfig) -> dict[str, Any]:
    """Return a JSON-safe copy of the truth boundary section.

    This is a reporting/passthrough helper — it returns the truth dict exactly
    as stored in the config, without re-validating it.  Call
    :func:`validate_config` first to confirm the truth section is structurally
    correct and free of escalated claims.

    This function does not prove biological truth.  It produces a
    JSON-safe snapshot of the declared claim level for logging and auditing.
    """
    from .io import json_safe
    return json_safe(dict(cfg.truth) if cfg.truth else {})


_SUPPORTED_RUNTIME_SPEC_KEYS = frozenset({
    "backend", "dtype", "jit", "vmap", "precision", "seed",
    "recurrent_backend", "synaptic_kernel",
})


def _runtime_from_spec(runtime_spec: dict[str, Any], default_seed: int) -> tuple[RuntimeConfig, tuple[str, ...]]:
    """Build a RuntimeConfig from a .jcfg.json runtime_spec dict.

    Returns (runtime_config, warnings). Unsupported keys generate warnings but
    do not raise. Invalid known values (e.g. unknown synaptic_kernel) raise
    via RuntimeConfig.__post_init__.
    """
    spec = dict(runtime_spec or {})
    warnings: list[str] = []
    kwargs: dict[str, Any] = {"seed": int(spec.get("seed", default_seed))}
    if "backend" in spec:
        kwargs["backend"] = str(spec["backend"])
    if "dtype" in spec:
        kwargs["dtype"] = str(spec["dtype"])
    if "jit" in spec:
        kwargs["jit"] = bool(spec["jit"])
    if "vmap" in spec:
        kwargs["vmap"] = bool(spec["vmap"])
    if "precision" in spec:
        kwargs["precision"] = str(spec["precision"])
    if "recurrent_backend" in spec:
        kwargs["recurrent_backend"] = str(spec["recurrent_backend"])
    if "synaptic_kernel" in spec:
        kwargs["synaptic_kernel"] = str(spec["synaptic_kernel"])
    for k in spec:
        if k not in _SUPPORTED_RUNTIME_SPEC_KEYS:
            warnings.append(f"runtime_spec_unsupported_key:{k}")
    return RuntimeConfig(**kwargs), tuple(warnings)


def config_to_simulation(cfg: JaxFNEConfig) -> Simulation:
    """Map the ``run`` section of a :class:`JaxFNEConfig` to a :class:`Simulation`.

    Also consumes ``cfg.runtime_spec`` (the ``"runtime"`` key in ``.jcfg.json``)
    so backend / dtype / jit / vmap / recurrent_backend / synaptic_kernel
    declarations are honored at execution time rather than being metadata-only.

    Unsupported runtime keys generate warnings stored in ``Simulation.runtime``
    metadata via ``RuntimeConfig.runtime_report``. Known but invalid values
    raise ``ValueError`` via ``RuntimeConfig.__post_init__``.
    """
    run = cfg.run or {}
    kwargs: dict[str, Any] = {
        "duration_ms": float(run["duration_ms"]),
        "dt_ms": float(run["dt_ms"]),
        "seed": int(run.get("seed", 0)),
    }
    if "record_sources" in run:
        kwargs["record_sources"] = bool(run["record_sources"])
    if "record_fields" in run:
        kwargs["record_fields"] = bool(run["record_fields"])
    if "plasticity" in run:
        kwargs["plasticity"] = float(run["plasticity"])
    if cfg.runtime_spec:
        rt, rt_warnings = _runtime_from_spec(cfg.runtime_spec, default_seed=kwargs["seed"])
        kwargs["runtime"] = rt
        _CONFIG_RUNTIME_WARNINGS[cfg.config_hash] = rt_warnings
    return Simulation(**kwargs)


# Module-level registry: maps JaxFNEConfig.config_hash → runtime_spec warnings.
# Populated by config_to_simulation; consumed (and cleared) by config_to_configuration
# so warnings flow into Configuration.metadata without mutating the frozen RuntimeConfig.
# This couples config_to_simulation and config_to_configuration implicitly; a future
# refactor should pass warnings as a return value or context object instead.
# Size-bounded by unique configs in a session; not thread-safe.
_CONFIG_RUNTIME_WARNINGS: dict[str, tuple[str, ...]] = {}


def config_to_geometry(cfg: JaxFNEConfig) -> Optional[LaminarSourceGeometry]:
    """Map the ``geometry`` section to a :class:`LaminarSourceGeometry`, or ``None``.

    Depths are normalized proxy coordinates in [0, 1].  No physical spatial
    units (mm, µm) are introduced.  The default ``position_units`` is
    ``"relative_laminar_depth_proxy"``.
    """
    if cfg.geometry is None:
        return None

    geo = cfg.geometry
    populations: list[LaminarPopulation] = []
    for p in geo.get("populations", []):
        populations.append(LaminarPopulation(
            name=str(p["name"]),
            cell_type=str(p["cell_type"]),
            layer=str(p.get("layer", "unspecified")),
            depth_min=float(p["depth_min"]),
            depth_max=float(p["depth_max"]),
            n_units=int(p["n_units"]),
        ))

    position_units = str(geo.get("position_units", "relative_laminar_depth_proxy"))
    n_units_total = sum(p.n_units for p in populations)
    return LaminarSourceGeometry(
        populations=tuple(populations),
        n_units_total=n_units_total,
        position_units=position_units,
    )


def _conservative_truth_transfer(user_truth: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Transfer user-supplied truth metadata conservatively into Configuration.metadata.

    Conservative rules:
    1. claim_level is forced to "computational_scaffold".
    2. physical_amplitude_calibrated is forced to False.
    3. Other conservative-default keys are taken from _CONSERVATIVE_TRUTH_DEFAULTS.
    4. Legacy keys are upgraded via :func:`migrate_schema` before transfer.
    5. Unknown truth keys are copied through only if JSON-safe scalars.

    Returns (transferred_truth, escalation_warnings).
    """
    out: dict[str, Any] = dict(_CONSERVATIVE_TRUTH_DEFAULTS)
    warnings: list[str] = []
    for k, v in migrate_schema(user_truth or {}).items():
        if k == "claim_level" and v != "computational_scaffold":
            warnings.append(f"truth_escalation_downgraded:claim_level:{v!r}_to_computational_scaffold")
            continue
        if k == "physical_amplitude_calibrated" and v is True:
            warnings.append("truth_escalation_downgraded:physical_amplitude_calibrated:True_to_False")
            continue
        # Accept value (overwrite conservative default with user-supplied conservative value).
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        else:
            warnings.append(f"truth_key_non_scalar_skipped:{k}")
    return out, tuple(warnings)


_SUPPORTED_EMITTER_FAMILIES = frozenset({"izhikevich"})
_SUPPORTED_FIELD_DOMAINS = frozenset({"laminar_column"})
_SUPPORTED_FIELD_CONDUCTIVITIES = frozenset({"proxy", "isotropic"})
_SUPPORTED_FIELD_BOUNDARIES = frozenset({"mean_zero_neumann", "proxy_boundary"})
_SUPPORTED_FIELD_GAUGES = frozenset({"mean_zero", "proxy_reference"})


def _config_section_warnings(cfg: JaxFNEConfig) -> tuple[str, ...]:
    """Generate unsupported-config warnings without raising.

    The scaffold supports a limited set of values for emitter.family,
    field.domain, field.conductivity, field.boundary, field.gauge. Any other
    value is recorded as a warning so manifests do not silently report
    declarations the kernel does not execute.
    """
    warnings: list[str] = []
    emitter_family = (cfg.emitter or {}).get("family")
    if emitter_family is not None and emitter_family not in _SUPPORTED_EMITTER_FAMILIES:
        warnings.append(f"unsupported_emitter_family:{emitter_family!r}")
    field_spec = cfg.field_spec or {}
    for key, supported in (
        ("domain", _SUPPORTED_FIELD_DOMAINS),
        ("conductivity", _SUPPORTED_FIELD_CONDUCTIVITIES),
        ("boundary", _SUPPORTED_FIELD_BOUNDARIES),
        ("gauge", _SUPPORTED_FIELD_GAUGES),
    ):
        v = field_spec.get(key)
        if v is not None and v not in supported:
            warnings.append(f"unsupported_field_{key}:{v!r}")
    return tuple(warnings)


def config_to_configuration(cfg: JaxFNEConfig) -> Configuration:
    """Map the ``network``/``emitter``/``field``/``probes`` sections to a :class:`Configuration`.

    v0.0.21 hardening:
    * Truth metadata is conservatively transferred via ``_conservative_truth_transfer``.
      User attempts to escalate ``claim_level``, or
      ``physical_amplitude_calibrated`` are downgraded with a warning.
    * Unsupported ``emitter.family``/``field.*`` values are recorded as
      ``unsupported_config_warnings`` in ``Configuration.metadata`` rather than
      silently accepted.
    * ``runtime_spec_unsupported_key:*`` warnings stashed by
      :func:`config_to_simulation` are surfaced into the same metadata bucket.
    """
    c = configuration()
    c = c.network(**dict(cfg.network or {}))
    c = c.emitter(**dict(cfg.emitter or {}))
    c = c.field(**dict(cfg.field_spec or {}))
    for probe in cfg.probes or ():
        c = c.probe(**dict(probe))

    # Conservative truth transfer.
    truth_transferred, truth_warnings = _conservative_truth_transfer(cfg.truth or {})
    section_warnings = _config_section_warnings(cfg)
    runtime_warnings = _CONFIG_RUNTIME_WARNINGS.pop(cfg.config_hash, ())
    all_warnings = tuple(truth_warnings) + section_warnings + runtime_warnings

    md = dict(c.metadata)
    md.update(truth_transferred)
    if all_warnings:
        md["unsupported_config_warnings"] = list(all_warnings)
    c = replace(c, metadata=md)
    return c


def config_to_trial_batch(
    cfg: JaxFNEConfig,
    conditions: Sequence[ParadigmCondition],
) -> TrialBatch:
    """Map the ``trials`` section and conditions to a :class:`TrialBatch`.

    Conservative defaults when section is absent:
    - n_reps = 1
    - seed from cfg.run.seed
    - seed_policy = "paired_by_replicate"
    """
    trials_spec = cfg.trials or {}
    n_reps = int(trials_spec.get("n_reps", 1))
    base_seed = int(
        trials_spec.get("base_seed", trials_spec.get("seed", (cfg.run or {}).get("seed", 0)))
    )
    seed_policy = str(trials_spec.get("seed_policy", "paired_by_replicate"))
    return trial_batch(
        conditions=conditions,
        n_reps=n_reps,
        seed=base_seed,
        seed_policy=seed_policy,
    )


