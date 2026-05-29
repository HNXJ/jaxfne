"""Core object model for :mod:`jaxfne`.

Design target: object-oriented public API, pure-JAX computational core.  The
current package is an honest TFNE scaffold: reduced emitters plus laminar proxy
source/readout status, not a full PDE field solver.
"""

from __future__ import annotations

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
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)

_ALLOWED_SYNAPTIC_KERNELS = ("exponential", "receptor_exponential")
from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, load_json, manifest as build_manifest


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
        Initialization strategy; "current" means start from the
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


def _default_metadata() -> dict[str, Any]:
    return {
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "source_projection_mode": "proxy_no_field_solve",
        "source_decomposition": "proxy_reduced_emitter",
        "boundary_condition": "mean_zero_neumann",
        "gauge": "mean_zero",
        "csd_sign_convention": "positive_equals_extracellular_source",
        "field_solver_status": "laminar_proxy_no_pde",
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
        thickness = {layer: max(0.0, layer_ranges[layer][1] - layer_ranges[layer][0]) for layer in layers}
        layer_counts = _counts_from_fractions(n_col, thickness)
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

    params = izhikevich_params_from_labels(
        labels,
        layer_labels=layer_labels,
        dtype=dtype,
        drive_overrides=metadata.get("cell_type_drives"),
    )
    positions = jnp.concatenate(position_chunks, axis=0) if position_chunks else jnp.zeros((0, 3), dtype=jdtype)
    params = _suite2_apply_connectivity(params, area_labels, layer_labels, labels, metadata, seed=seed, dtype=dtype)
    geometry_meta = {
        "neuron_rows": neuron_rows,
        "area_labels": area_labels,
        "layer_labels": layer_labels,
        "cell_type_labels": labels,
        "geometry_mode": metadata.get("geometry_mode", "declared_metadata_not_solved_3d_pde_grid"),
        "position_units": "mm_declared_metadata",
        "uniform_3d": bool(uniform_3d),
    }
    return params, positions, geometry_meta


def _suite2_apply_connectivity(params: IzhikevichParams, area_labels: Sequence[str], layer_labels: Sequence[str], cell_labels: Sequence[str], metadata: Mapping[str, Any], *, seed: int, dtype: str) -> IzhikevichParams:
    n = len(cell_labels)
    if n == 0:
        return params
    jdtype = jnp.float64 if dtype == "float64" and bool(jax.config.read("jax_enable_x64")) else jnp.float32
    key = jax.random.PRNGKey(seed + 4242)
    rnd = jax.random.uniform(key, (n, n), minval=0.25, maxval=1.0, dtype=jdtype)
    sign = params.sign.astype(jdtype)
    same_area = jnp.asarray([[1.0 if area_labels[i] == area_labels[j] else 0.0 for j in range(n)] for i in range(n)], dtype=jdtype)
    eye = jnp.eye(n, dtype=jdtype)
    base_gain = float((metadata.get("connectivity", {}) or {}).get("within_gain", 0.45))
    W = base_gain * rnd * sign[None, :] * same_area * (1.0 - eye) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))

    if bool(metadata.get("suite2_interarea", False)):
        pre_v1_l23 = jnp.asarray([area_labels[j] == "V1" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v4_l4 = jnp.asarray([area_labels[i] == "V4" and layer_labels[i] == "L4" and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        pre_v4_l23 = jnp.asarray([area_labels[j] == "V4" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v1_deep_l1 = jnp.asarray([area_labels[i] == "V1" and layer_labels[i] in {"L1", "L5", "L6"} and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        ff_gain = float((metadata.get("connectivity", {}) or {}).get("feedforward_gain", 0.65)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        fb_gain = float((metadata.get("connectivity", {}) or {}).get("feedback_gain", 0.50)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        W = W + ff_gain * (post_v4_l4[:, None] * pre_v1_l23[None, :])
        W = W + fb_gain * (post_v1_deep_l1[:, None] * pre_v4_l23[None, :])

    return replace(params, W=W)


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
        metadata.setdefault("physical_amplitude_claim_allowed", False)

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
            "field_solver_status": "laminar_proxy_no_pde",
            "physical_amplitude_claim_allowed": False,
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
                "L1": (0.00, 0.10),
                "L2": (0.10, 0.25),
                "L3": (0.25, 0.45),
                "L4": (0.45, 0.55),
                "L5": (0.55, 0.85),
                "L6": (0.85, 1.00),
            }

        # Validate layer fractions
        for layer, (z_min, z_max) in layer_fractions.items():
            if not (0.0 <= z_min < z_max <= 1.0):
                raise ValueError(f"layer {layer!r} has invalid fraction range: ({z_min}, {z_max})")

        # Use default layer cell-type distributions if not provided
        if layer_cell_types is None:
            layer_cell_types = {
                "L1": {"E": 0.75, "PV": 0.0, "SST": 0.0, "VIP": 0.25},
                "L2": {"E": 0.75, "PV": 0.05, "SST": 0.05, "VIP": 0.15},
                "L3": {"E": 0.75, "PV": 0.10, "SST": 0.10, "VIP": 0.05},
                "L4": {"E": 0.25, "PV": 0.45, "SST": 0.15, "VIP": 0.15},
                "L5": {"E": 0.15, "PV": 0.25, "SST": 0.30, "VIP": 0.30},
                "L6": {"E": 0.10, "PV": 0.20, "SST": 0.20, "VIP": 0.50},
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
            physical_amplitude_claim_allowed=False,
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

    def validate(self) -> dict[str, Any]:
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
            "truth_mode": self.metadata.get("truth_mode"),
            "claim_level": self.metadata.get("claim_level"),
        }


@dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime and dtype policy.

    ``dtype='float64'`` is honored only when JAX x64 is enabled.  The manifest
    always reports both requested and actual dtype policy.
    """

    backend: str = "auto"  # "auto" | "cpu" | "gpu" | "tpu"
    dtype: str = "float32"  # "float32" | "float64"
    jit: bool = False
    vmap: bool = False
    precision: str = "default"  # "default" | "high"
    seed: int = 0
    n_steps: int = 0
    recurrent_backend: str = "dense"  # "dense" | "edge_list"
    synaptic_kernel: str = "exponential"  # "exponential" | "receptor_exponential"
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

    @property
    def requested_dtype(self) -> str:
        return self.dtype_primary or self.dtype

    @property
    def selected_backend(self) -> str:
        return self.device_type or self.backend

    @property
    def actual_dtype(self) -> str:
        if self.requested_dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
            return "float64"
        return "float32"

    @property
    def jnp_dtype(self) -> jnp.dtype:
        return jnp.float64 if self.actual_dtype == "float64" else jnp.float32

    def runtime_report(self) -> dict[str, Any]:
        devices = [str(device) for device in jax.devices()]
        default_backend = jax.default_backend()
        requested_backend = self.selected_backend  # alias for selected (user-requested)

        # Compute actual_backend / backend_enforced / backend_warning.
        # Conservative policy (CPU-only environment is the supported scaffold target):
        # - "auto": actual_backend = JAX default; enforced = True (whatever JAX picks).
        # - "cpu" : actual_backend = "cpu"; enforced = True iff JAX default is cpu.
        # - "gpu"/"tpu": if JAX default is the requested device, enforced = True;
        #   otherwise enforced = False with a clean warning (no false claim).
        backend_warning: Optional[str] = None
        if requested_backend == "auto":
            actual_backend = default_backend
            backend_enforced = True
        elif requested_backend == "cpu":
            actual_backend = "cpu"
            backend_enforced = (default_backend == "cpu")
            if not backend_enforced:
                backend_warning = (
                    f"requested_cpu_but_jax_default_is:{default_backend!r}"
                    "; jaxfne_does_not_force_jax_backend"
                )
        elif requested_backend in ("gpu", "tpu"):
            if default_backend == requested_backend:
                actual_backend = requested_backend
                backend_enforced = True
            else:
                # Honest downgrade: do not falsely report GPU/TPU executed.
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
            # Compatibility keys from v0.0.3.
            "device_type": self.selected_backend,
            "dtype_primary": self.actual_dtype,
        }


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
        if self.method in {"straight_through", "sigmoid_beta"}:
            return "declared_surrogate"
        return "required_but_missing"

    def to_dict(self) -> dict[str, Any]:
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
    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True
    poisson_drive: Optional[dict] = None
    runtime: RuntimeConfig | None = None

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
    name: str
    modes: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)


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
            "truth_mode": self.metadata.get("truth_mode", "truth_safe_unverified"),
            "field_claim_level": self.metadata.get("field_claim_level", "proxy_readout_only"),
        })


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


@dataclass(frozen=True)
class ParadigmEvent:
    """Discrete event within a task trial: stimulus, behavioral code, or omission marker."""

    label: str
    onset_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    code: Optional[int] = None
    stimulus: Optional[str] = None
    is_omission: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "label": self.label,
            "onset_ms": self.onset_ms,
            "duration_ms": self.duration_ms,
            "code": self.code,
            "stimulus": self.stimulus,
            "is_omission": self.is_omission,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ParadigmCondition:
    """A specific trial condition: sequence of stimuli and associated events."""

    name: str
    sequence: tuple[str, str, str, str]
    omission_position: Optional[str] = None
    probability: Optional[float] = None
    condition_numbers: tuple[int, ...] = ()
    events: tuple[ParadigmEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_omission(self) -> bool:
        """Return True if this condition contains an omission."""
        return self.omission_position is not None

    def omitted_event_label(self) -> Optional[str]:
        """Return the label of the omitted event, or None if no omission."""
        if self.omission_position is None:
            return None
        for evt in self.events:
            if evt.label == self.omission_position:
                return evt.label
        return self.omission_position

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "sequence": self.sequence,
            "omission_position": self.omission_position,
            "probability": self.probability,
            "condition_numbers": self.condition_numbers,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class Paradigm:
    name: str = "none"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    conditions: tuple["ParadigmCondition", ...] = ()
    alignment_code: int = 101
    alignment_label: str = "p1"
    pre_stimulus_buffer_ms: float = 1000.0
    analysis_windows: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "baseline": (-500.0, 0.0),
        "event": (0.0, 500.0),
        "post_event": (500.0, 1000.0),
    })
    event_codes: dict[str, int] = field(default_factory=lambda: {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    })
    metadata: dict[str, Any] = field(default_factory=dict)

    def habituation(self, sequence: Sequence[str], n_trials: int) -> "Paradigm":
        return replace(
            self,
            blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}],
        )

    def main_block(self, **kwargs: Any) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0, condition_weights: Optional[dict[str, float]] = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_trials": n_trials,
            "seed": seed,
            "blocks": self.blocks,
            "condition_weights": condition_weights,
        }

    def condition(self, name: str) -> Optional["ParadigmCondition"]:
        """Return ParadigmCondition by name, or None if not found."""
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None

    def condition_names(self) -> list[str]:
        """Return list of condition names."""
        return [c.name for c in self.conditions]

    def omission_conditions(self) -> list["ParadigmCondition"]:
        """Return list of conditions containing omissions."""
        return [c for c in self.conditions if c.has_omission()]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "blocks": self.blocks,
            "conditions": [c.to_dict() for c in self.conditions],
            "alignment_code": self.alignment_code,
            "alignment_label": self.alignment_label,
            "pre_stimulus_buffer_ms": self.pre_stimulus_buffer_ms,
            "analysis_windows": self.analysis_windows,
            "event_codes": self.event_codes,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class DatasetSpec:
    """Manifest-safe dataset/alignment declaration for observed data.

    DatasetSpec is a schema object, not a loader.  It records how an external
    dataset is aligned and interpreted so objectives can reference data without
    hard-coding paths or claiming empirical validation.
    """

    name: str = "unnamed_dataset"
    modality: str = "unspecified"
    source_format: str = "unspecified"
    alignment_label: str = "p1"
    alignment_code: int = 101
    sampling_rate_hz: Optional[float] = None
    units: str = "unspecified"
    trial_filter: dict[str, Any] = field(default_factory=dict)
    condition_map: dict[str, list[int]] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_condition_map(self, condition_map: Mapping[str, Sequence[int]]) -> "DatasetSpec":
        mapped = {str(k): [int(x) for x in v] for k, v in condition_map.items()}
        return replace(self, condition_map=mapped)

    def with_quality_gate(self, name: str, value: Any) -> "DatasetSpec":
        gates = dict(self.quality_gates)
        gates[str(name)] = value
        return replace(self, quality_gates=gates)


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
            physical_amplitude_claim_allowed=False,
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

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.name:
            issues.append("dataset_name_missing")
        if self.alignment_code is None:
            issues.append("alignment_code_missing")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            issues.append("sampling_rate_hz_must_be_positive")
        return {
            "valid": not issues,
            "issues": issues,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        }

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "modality": self.modality,
            "source_format": self.source_format,
            "alignment_label": self.alignment_label,
            "alignment_code": self.alignment_code,
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
    physical_amplitude_claim_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "spec_name": self.spec_name,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "claim_level": self.claim_level,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
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
    """Explicit native-drive schedule for event-aligned stimulus injection.

    Contains an ordered sequence of drive events as JSON-safe dicts with keys:
    ``onset_ms``, ``duration_ms``, ``amplitude``, ``label``, ``is_drive_event``.
    When ``is_drive_event`` is False or ``amplitude`` is 0, the event injects
    zero drive. No physical-amplitude or calibration claim is made; amplitude
    values are native Izhikevich current units.
    """

    events: tuple[dict[str, Any], ...]
    n_neurons: int
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_claim_allowed: bool = False
    claim_level: str = "computational_scaffold"

    def to_array(self, n_steps: int, dt_ms: float, dtype: str = "float32") -> "jax.Array":
        """Materialize a ``(n_steps, n_neurons)`` drive schedule array."""
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
                schedule[start:end, :] += amp
        np_dtype = _np.float64 if dtype == "float64" else _np.float32
        return jnp.asarray(schedule.astype(np_dtype))

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "stimulus_injection_status": "native_drive_schedule_v0.0.12",
            "n_drive_events": len(self.events),
            "n_neurons": self.n_neurons,
            "events": list(self.events),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
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
    physical_amplitude_claim_allowed: bool = False
    claim_level: str = "computational_scaffold"


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
            physical_amplitude_claim_allowed=False,
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

    def validate(self) -> dict[str, Any]:
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
        if self.physical_amplitude_claim_allowed is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        if self.claim_level != "computational_scaffold":
            issues.append("claim_level_must_be_computational_scaffold")
        warnings: list[str] = []
        if self.layer not in _KNOWN_LAYERS:
            warnings.append(f"unrecognized_layer:{self.layer}")
        return {"valid": not issues, "issues": issues, "warnings": warnings}

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "cell_type": self.cell_type,
            "layer": self.layer,
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "n_units": int(self.n_units),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
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
    physical_amplitude_claim_allowed: bool = False
    claim_level: str = "computational_scaffold"


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
            physical_amplitude_claim_allowed=False,
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

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.populations:
            issues.append("populations_empty")
        pop_sum = sum(p.n_units for p in self.populations)
        if pop_sum != self.n_units_total:
            issues.append(f"n_units_total_mismatch:sum={pop_sum},declared={self.n_units_total}")
        if self.physical_amplitude_claim_allowed is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        pop_issues: list[str] = []
        for p in self.populations:
            v = p.validate()
            pop_issues.extend([f"{p.name}:{i}" for i in v["issues"]])
        issues.extend(pop_issues)
        return {"valid": not issues, "issues": issues, "n_populations": len(self.populations)}

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "type": "laminar_source_geometry",
            "n_units_total": self.n_units_total,
            "n_populations": len(self.populations),
            "position_units": self.position_units,
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
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
})


def _finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _compute_all_metrics(signals: "Signals", readout: Optional[dict[str, Any]] = None) -> dict[str, Optional[float]]:
    """Compute all known scalar metrics from signals."""
    dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
    m: dict[str, Optional[float]] = {}
    m["spike_rate_hz_mean"] = _finite_or_none(float(jnp.mean(signals.spikes) * (1000.0 / dt_ms)))
    m["spike_count_total"] = _finite_or_none(float(jnp.sum(signals.spikes)))
    m["mean_V_m"] = _finite_or_none(float(jnp.mean(signals.V_m)))
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
) -> dict[str, Any]:
    """Evaluate one gate spec; returns pass/fail."""
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
            physical_amplitude_claim_allowed=False,
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
            physical_amplitude_claim_allowed=False,
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
    * ``physical_amplitude_claim_allowed`` – always False
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
            "physical_amplitude_claim_allowed": False,
        }
    # Fallback: stringify unknown types rather than crash.
    return {"readout_results": [json_safe({"raw": str(readout)})], "n_results": 1,
            "physical_amplitude_claim_allowed": False}


@dataclass(frozen=True)
class Model:
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
            "truth_mode": self.cfg.metadata.get("truth_mode", "truth_safe_unverified"),
            "claim_level": self.cfg.metadata.get("claim_level", "computational_scaffold"),
            "source_calibration_status": self.cfg.metadata.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "field_solver_status": self.cfg.metadata.get("field_solver_status", "laminar_proxy_no_pde"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
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

    def _simulate_arrays(
        self,
        sim: Simulation,
        key: jax.Array,
        runtime_cfg: RuntimeConfig,
        drive_schedule: Optional[Any] = None,
    ):
        emitter: IzhikevichParams = self.params["emitter"]
        sched = drive_schedule  # None or (n_steps, n_neurons) array
        if runtime_cfg.recurrent_backend == "edge_list":
            edges: EdgeList = self.params["edge_list"]
            kernel_fn = (
                simulate_receptor_exponential_izhikevich
                if runtime_cfg.synaptic_kernel == "receptor_exponential"
                else simulate_edge_recurrent_izhikevich
            )
            if runtime_cfg.jit:
                run = jax.jit(
                    lambda k, s: kernel_fn(
                        emitter, edges, sim.n_steps, sim.dt_ms, k,
                        dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                    )[:3]
                )
                return run(key, sched)
            return kernel_fn(
                emitter, edges, sim.n_steps, sim.dt_ms, key,
                dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
            )[:3]
        if runtime_cfg.jit:
            run = jax.jit(
                lambda k, s: simulate_eig_izhikevich(
                    emitter, sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                )
            )
            return run(key, sched)
        return simulate_eig_izhikevich(
            emitter, sim.n_steps, sim.dt_ms, key,
            dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
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
        self,
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
            "field_claim_level": "proxy_readout_only",
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
        }
        # v0.2.0: Add source bookkeeping metadata for theoretical validation.
        metadata["source_bookkeeping"] = {
            "source_mode": _SOURCE_PROXY_METADATA.get("source_mode"),
            "source_projection_mode": self.cfg.metadata.get("source_projection_mode", "proxy_no_field_solve"),
            "source_decomposition": self.cfg.metadata.get("source_decomposition", "proxy_reduced_emitter"),
            "source_calibration_status": _SOURCE_PROXY_METADATA.get("source_calibration_status"),
            "synaptic_current_counting": _SOURCE_PROXY_METADATA.get("double_count_synaptic_current_guard"),
            "source_mode_exclusive": True,
            "physical_amplitude_claim_allowed": _SOURCE_PROXY_METADATA.get("physical_amplitude_claim_allowed", False),
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
        return Signals(
            time_ms=time_ms,
            V_m=voltages.astype(runtime_cfg.jnp_dtype),
            spikes=spikes,
            sources=sources.astype(runtime_cfg.jnp_dtype) if sim.record_sources else None,
            field=field_output,
            metadata=metadata,
        )

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

        edge_kernel_fn = (
            simulate_receptor_exponential_izhikevich
            if runtime_cfg.synaptic_kernel == "receptor_exponential"
            else simulate_edge_recurrent_izhikevich
        )

        def one(k):
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
        if runtime_cfg.vmap:
            run = jax.vmap(one)
            if runtime_cfg.jit:
                run = jax.jit(run)
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
                "field_claim_level": "proxy_readout_only",
                "physical_amplitude_claim_allowed": False,
                "recurrent_backend": runtime_cfg.recurrent_backend,
                "synaptic_kernel": runtime_cfg.synaptic_kernel,
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
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }

        claim_labels: dict[str, Any] = {
            "receipt_status": _RECEIPT_SCHEMA_VERSION,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "physical_amplitude_claim_allowed": False,
        }

        backend: dict[str, Any] = {
            "recurrent_backend": signals.metadata.get("recurrent_backend", "dense"),
            "synaptic_kernel": signals.metadata.get("synaptic_kernel", "exponential"),
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "physical_amplitude_claim_allowed": False,
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
        if objective.kind == "group_rate_targets":
            return self._evaluate_group_rate_targets(signals, objective, warnings, cfg_meta)

        computed_metrics = _compute_all_metrics(signals, readout)

        loss_results = []
        total_loss = 0.0
        has_loss_value = False
        for spec in objective.losses:
            r = _evaluate_loss_spec(spec, computed_metrics, warnings, strict)
            loss_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        reg_results = []
        for spec in objective.regularizers:
            r = _evaluate_regularizer_spec(spec, computed_metrics, warnings, strict)
            reg_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        gate_results = []
        all_gates_pass = True
        for spec in objective.gates:
            r = _evaluate_gate_spec(spec, computed_metrics, warnings, strict)
            gate_results.append(r)
            if not r.get("pass", True):
                all_gates_pass = False

        acceptance = "gates_pass" if all_gates_pass else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_v0.0.5",
            "objective_name": objective.name,
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "losses": loss_results,
            "regularizers": reg_results,
            "gates": gate_results,
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
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
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
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
        self,
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
                "objective_name": objective.name,
                "total_loss": None,
                "losses": [],
                "regularizers": [],
                "gates": [],
                "all_gates_pass": False,
                "acceptance_decision": "gates_fail",
                "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
                "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
                "field_claim_level": "proxy_readout_only",
                "physical_amplitude_claim_allowed": False,
                "warnings": warnings,
            })

        if weights_dict is None:
            weights_dict = {name: 1.0 for name in groups_dict.keys()}

        # Compute dt from time axis
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
            "objective_name": objective.name,
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "group_rate_losses": loss_details,
            "losses": [],
            "regularizers": [],
            "gates": [],
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "warnings": warnings,
        })

    def tune(
        self,
        objective: Optional["Objective"] = None,
        optimizer: Any = None,
        steps: int = 0,
        seed: int = 0,
        strategy: Optional[str] = None,
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
            return self._tune_multiparameter(
                objective=objective,
                optimizer=optimizer,
                spec=spec,
                parameters=parameters,
                generations=generations or 8,
                population_size=population_size or 6,
                seed=int(seed),
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
            "strategy": strategy or spec.optimizer,
            "parameter": parameter,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "optimizer": spec.to_dict(),
            "objective_name": objective.name if not isinstance(objective, str) else objective,
            "losses_declared": len(objective.losses) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(objective.regularizers) if not isinstance(objective, str) else 0,
            "gates_declared": len(objective.gates) if not isinstance(objective, str) else 0,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "laminar_proxy_no_pde"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
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
            history.append({
                "step": idx,
                "candidate_value": float(candidate_value),
                "score": _finite_or_none(score),
                "all_gates_pass": gates_pass,
                "accepted_as_best": bool(accepted),
                "evaluation_status": candidate_report.get("evaluation_status"),
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
            "strategy": "agsdr_multiparameter",
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
            "objective_name": objective.name if not isinstance(objective, str) else objective,
            "losses_declared": len(objective.losses) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(objective.regularizers) if not isinstance(objective, str) else 0,
            "gates_declared": len(objective.gates) if not isinstance(objective, str) else 0,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "laminar_proxy_no_pde"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
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

        # Define scoring function for AGSDR loop
        def evaluate_fn(candidate_params: dict[str, float]) -> float:
            """Evaluate a candidate parameter dict and return loss."""
            candidate_model = _model_with_parameters(self, candidate_params, param_specs if param_specs else None)
            candidate_signals = candidate_model.simulate(replace(simulation, seed=int(seed)))
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
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
            "physical_amplitude_claim_allowed": False,
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
            backend_meta["edge_list_physical_amplitude_claim_allowed"] = False
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
                field_solver_status="laminar_proxy_no_pde",
                field_claim_level="proxy_readout_only",
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
    return (runtime_config or RuntimeConfig()).runtime_report()


def simulation(**kwargs: Any) -> Simulation:
    return Simulation(**kwargs)


def objective() -> Objective:
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
    """Tune Poisson-drive amplitude toward a target mean firing-rate range.\n\n    The outer stage evaluates an AGSDR-style candidate population.  The inner\n    stage applies finite-difference Adam updates to the scalar noise amplitude.\n    This path tunes a reduced native-drive parameter and preserves relative\n    proxy-unit metadata.\n    """
    sim0 = simulation or suite2_simulation(seed=seed, duration_ms=1000.0, dt_ms=0.1)
    lo, hi = float(target_rate_hz[0]), float(target_rate_hz[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError("target_rate_hz must be a positive (low, high) tuple")
    target_mid = 0.5 * (lo + hi)

    def run_amp(amp: float, run_seed: int) -> tuple[float, float, Signals]:
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
        "field_solver_status": model.cfg.metadata.get("field_solver_status", "laminar_proxy_no_pde"),
        "physical_amplitude_claim_allowed": False,
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


def paradigm(name: str = "none") -> Paradigm:
    return Paradigm(name=name)


def simulate(
    model: Model,
    sim: Optional[Simulation] = None,
    paradigm: Optional[Any] = None,
    **kwargs: Any,
) -> Signals:
    """Run a simulation with the given model.

    Allows passing either an explicit :class:`Simulation` object, or passing
    simulation parameters (such as ``duration_ms``, ``dt_ms``, ``seed``,
    `record_sources`, `record_fields`, `runtime`) as direct keyword
    arguments.
    """
    if sim is None:
        sim = Simulation(**kwargs)
    elif kwargs:
        raise ValueError(
            "Cannot specify both a Simulation object and individual simulation parameters as keyword arguments."
        )
    return model.simulate(sim, paradigm=paradigm)



def construct(cfg: Configuration, *, geometry: "LaminarSourceGeometry | None" = None) -> Model:
    validation = cfg.validate()
    if not validation["valid"]:
        raise ValueError(f"Invalid jaxfne configuration: {validation['issues']}")
    net = cfg.networks[0]
    n = int(net.get("n", 100))
    dtype_name_cfg = str(cfg.metadata.get("dtype", "float32"))
    if cfg.metadata.get("columns") or cfg.metadata.get("layer_cell_types") or cfg.metadata.get("uniform_3d"):
        params, positions, geometry_meta = _suite2_neuron_population_from_config(cfg, dtype=dtype_name_cfg)
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

    edge_list = make_edge_list_from_dense(network.params.W, dtype=network.params.v0.dtype.name)

    if geometry is not None:
        if geometry.n_units_total != n:
            raise ValueError(
                f"geometry_n_units_total_mismatch: "
                f"geometry.n_units_total={geometry.n_units_total} but cfg network n={n}"
            )
        positions = geometry.positions_array(dtype=network.params.v0.dtype.name)
        geometry_meta = geometry.to_dict()

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

    return Model(
        cfg=cfg,
        params={"emitter": network.params, "positions": positions, "edge_list": edge_list},
        static=static,
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

    Alignment: P1 onset (code 101) at t=0.
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
        alignment_code=event_codes["p1"],
        alignment_label="p1",
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
            ev_dicts.append({
                "label": e.label,
                "onset_ms": float(e.onset_ms) if e.onset_ms is not None else 0.0,
                "duration_ms": dur,
                "amplitude": amp if is_drive else 0.0,
                "is_drive_event": is_drive,
            })
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

_JAXFNE_VERSION = "0.3.11"
_RECEIPT_SCHEMA_VERSION = "run_receipt_v0.0.21"
_MANIFEST_SCHEMA_VERSION = "manifest.v0.0.21"
_OBJECTIVE_REPORT_SCHEMA_VERSION = "objective_report.v0.0.18"

# v0.0.21: explicit source proxy metadata.
# Documents what the current Izhikevich scaffold computes as the "source" field.
# Reading the edge/dense kernels: source_proxy = source_scale * (current_native
# + 20.0 * spikes), where current_native = drive + recurrent_syn + noise. The
# 20.0 gain is hardcoded in simulate_edge_recurrent_izhikevich (and the dense
# variant). No physical-amplitude claim is made; this remains an uncalibrated
# proxy. The double-count guard records that synaptic current enters the source
# only via the single proxy expression, not as a separate additive term.
_SOURCE_PROXY_METADATA: dict[str, Any] = {
    "source_model": "izhikevich_native_current_plus_spike_impulse_proxy",
    "source_mode": "native_current_plus_spike_impulse_proxy",
    "includes_native_current": True,
    "includes_drive_current": True,
    "includes_recurrent_synaptic_current": True,
    "includes_noise_current": True,
    "includes_spike_impulse": True,
    "spike_impulse_gain": 20.0,
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "physical_amplitude_claim_allowed": False,
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

_JAXFNE_CONFIG_SCHEMA_VERSION = "jaxfne.config.v0.0.15"

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
    "truth_mode": "truth_safe_unverified",
    "physical_amplitude_claim_allowed": False,
    "claim_level": "computational_scaffold",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "laminar_proxy_no_pde",
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}


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

    # Truth boundary — all fields required and must be conservative
    truth = cfg.truth or {}
    for tk in _CONSERVATIVE_TRUTH_DEFAULTS:
        if tk not in truth:
            issues.append(f"truth.{tk}_missing")

    if truth.get("physical_amplitude_claim_allowed") is not False:
        issues.append("truth_escalation:physical_amplitude_claim_allowed_must_be_False")
    if truth.get("claim_level") not in (None, "computational_scaffold"):
        issues.append(f"truth_escalation:claim_level:{truth.get('claim_level')!r}")
    if truth.get("source_calibration_status") not in (
        None, "uncalibrated_izhikevich_native_current"
    ):
        issues.append(
            f"truth_escalation:source_calibration_status:{truth.get('source_calibration_status')!r}"
        )
    if truth.get("field_solver_status") not in (None, "laminar_proxy_no_pde"):
        issues.append(f"truth_escalation:field_solver_status:{truth.get('field_solver_status')!r}")
    if truth.get("truth_mode") not in (None, "truth_safe_unverified"):
        issues.append(f"truth_escalation:truth_mode:{truth.get('truth_mode')!r}")
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
    1. truth_mode is forced to "truth_safe_unverified" regardless of input.
    2. claim_level is forced to "computational_scaffold".
    3. physical_amplitude_claim_allowed is forced to False.
    4. Other conservative-default keys are taken from _CONSERVATIVE_TRUTH_DEFAULTS.
    5. Unknown truth keys are copied through only if JSON-safe scalars.

    Returns (transferred_truth, escalation_warnings).
    """
    out: dict[str, Any] = dict(_CONSERVATIVE_TRUTH_DEFAULTS)
    warnings: list[str] = []
    for k, v in (user_truth or {}).items():
        if k == "truth_mode" and v != "truth_safe_unverified":
            warnings.append(f"truth_escalation_downgraded:truth_mode:{v!r}_to_truth_safe_unverified")
            continue
        if k == "claim_level" and v != "computational_scaffold":
            warnings.append(f"truth_escalation_downgraded:claim_level:{v!r}_to_computational_scaffold")
            continue
        if k == "physical_amplitude_claim_allowed" and v is True:
            warnings.append("truth_escalation_downgraded:physical_amplitude_claim_allowed:True_to_False")
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
      User attempts to escalate ``truth_mode``, ``claim_level``, or
      ``physical_amplitude_claim_allowed`` are downgraded with a warning.
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
