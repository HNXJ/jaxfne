"""Configuration: the declarative TFNE model-declaration object.

Split out of ``jaxfne/core.py`` (slice 3 of the core.py monolith split, see
``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports every symbol
here for backward compatibility -- import from ``jaxfne.core``, not this
module, unless you are working on core.py itself.

``Configuration`` is purely declarative: its methods record metadata, they
do not touch dynamics. ``construct()``/``Model`` (still in core.py) consume
it one-directionally -- nothing here calls back into ``Model`` or
``construct()``. A handful of module-level helpers (``_default_operator_status``,
``_counts_from_fractions``) are also used by group-1 connectivity code that
remains in core.py; core.py imports them back from here rather than
duplicating them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from .io import config_hash
from ._runtime_config import RuntimeConfig


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


# Conservative claim defaults. Public metadata merges cannot escalate these.
_ALLOWED_CLAIM_LEVELS = frozenset({"computational_scaffold"})
_ALLOWED_FIELD_CLAIM_LEVELS = frozenset({"proxy_readout"})
_ALLOWED_FIELD_SOLVER_STATUS = frozenset(
    {
        "linear_solver",
        "not_computed",
        "experimental_pde_solver",
        "pde_solver",  # reserved token; amplitude still forced False
    }
)


def clamp_truth_gate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``metadata`` with non-escalatable truth gates pinned.

    Forces ``physical_amplitude_calibrated=False``, pins ``claim_level`` and
    ``field_claim_level`` to scaffold/proxy defaults when missing or escalated,
    and coerces unknown ``field_solver_status`` values to ``linear_solver``.
    Does not mutate the input mapping.
    """
    out = dict(metadata or {})
    claim = out.get("claim_level", "computational_scaffold")
    if claim not in _ALLOWED_CLAIM_LEVELS:
        out["claim_level"] = "computational_scaffold"
    else:
        out["claim_level"] = str(claim)

    field_claim = out.get("field_claim_level", "proxy_readout")
    if field_claim not in _ALLOWED_FIELD_CLAIM_LEVELS:
        out["field_claim_level"] = "proxy_readout"
    else:
        out["field_claim_level"] = str(field_claim)

    solver = out.get("field_solver_status", "linear_solver")
    if solver not in _ALLOWED_FIELD_SOLVER_STATUS:
        out["field_solver_status"] = "linear_solver"
    else:
        out["field_solver_status"] = str(solver)

    out["physical_amplitude_calibrated"] = False
    return out


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

        Use for administrative tags. Truth-gate fields
        (``claim_level``, ``field_claim_level``, ``field_solver_status``,
        ``physical_amplitude_calibrated``) are clamped after merge so callers
        cannot escalate claim surfaces via this API.

        Returns
        -------
        Configuration
            Updated configuration.
        """
        metadata = dict(self.metadata)
        metadata.update(kwargs)
        return replace(self, metadata=clamp_truth_gate_metadata(metadata))

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
        max_in_degree: Optional[int] = None,
        spatial_sigma: Optional[float] = None,
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
        if max_in_degree is not None and probability is not None:
            raise ValueError(
                f"connection {name!r}: max_in_degree and probability are mutually exclusive "
                f"sampling modes -- probability targets an O(n_pre*n_post) density, "
                f"max_in_degree targets a constant per-post-neuron cap; pick one."
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
            "max_in_degree": int(max_in_degree) if max_in_degree is not None else None,
            "spatial_sigma": float(spatial_sigma) if spatial_sigma is not None else None,
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

    def set_emitter(
        self,
        family: str = "izhikevich",
        preset: str = "cortical_eig",
        *,
        activation_rule: str = "cubic",
        conductance_rule: str = "hebbian",
        homeostasis_rule: str = "linear",
    ) -> "Configuration":
        """Chainable config method to set/wrap emitter family and presets.

        For ``family="homeostatic_ei"`` (the second canonical HDP sanity
        circuit -- see ``jaxfne/emitters_homeostatic_ei.py``), the three rule
        kwargs select the update-rule *names* from that module's registries
        (``ACTIVATION_RULES``/``CONDUCTANCE_RULES``/``HOMEOSTASIS_RULES``).
        Only registry names are accepted here -- a custom Python callable
        cannot pass through ``Configuration`` (which must stay JSON-safe for
        ``jtfne.io.json_safe``/``config_hash``); call
        :func:`jaxfne.emitters_homeostatic_ei.simulate_homeostatic_ei`
        directly to use a custom callable rule. Validity of the names is
        checked at ``construct()`` time, not here (this method only records
        metadata).
        """
        if family == "homeostatic_ei":
            return self.emitter(
                family=family,
                homeostatic_ei_rules={
                    "activation_rule": str(activation_rule),
                    "conductance_rule": str(conductance_rule),
                    "homeostasis_rule": str(homeostasis_rule),
                },
            )
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

