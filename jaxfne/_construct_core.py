"""``construct()``/``simulate()``/``compute_fields()`` hub: builds a ``Model``
from a ``Configuration``, resolves homeostasis/HDP metadata, and applies
canonical biophysics.

Split out of ``jaxfne/_construct.py`` (Phase 2 defragmentation, 2026-07-20,
part of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). ``jaxfne/_construct.py``
re-exports every symbol here for backward compatibility. Depends on
``_construct_connectivity`` and ``_construct_population`` (one direction only
-- neither of those depends back on this module), which is what keeps this
split acyclic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from .emitters import EdgeList, EIGNetwork, make_edge_list_from_dense, make_eig_network
from .emitters_homeostatic_ei import ACTIVATION_RULES, CONDUCTANCE_RULES, HOMEOSTASIS_RULES, HomeostaticEIParams
from .fields import FieldOutput
from ._config import Configuration
from ._runtime_config import RuntimeConfig
from ._signals import Simulation, Signals, LaminarSourceGeometry
from ._model import Model
from ._construct_connectivity import (
    _all_connection_rules_declare_resolvable_mechanism,
    _compile_connection_rules,
    _compile_mechanism_aware_connection_rules,
    _concat_edge_lists,
    _mark_connections_compiled,
    _empty_edge_list,
)
from ._construct_population import _DENSE_CONNECTIVITY_WARN_N, _neuron_population_from_config
from ._construct_extras import operator_status


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
    *,
    continuation: Any = None,
    return_state: bool = False,
    **kwargs: Any,
) -> "Signals | tuple[Signals, Any]":
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
    return model.simulate(
        sim,
        paradigm=paradigm,
        continuation=continuation,
        return_state=return_state,
    )


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


def _canonical_biophysics_depth_gating(emitter, positions):
    """Shared gating precondition for the two canonical-column laminar effects
    (deep-E grading and PV<->E strengthening): both require cell-type labels,
    layer labels, and a non-degenerate depth (Z) axis.

    Returns ``None`` if the precondition fails (caller should no-op), else
    ``(lab, Z, zd, isE, n)`` -- the shared label/depth arrays both effects read.
    """
    import numpy as _np
    cts = emitter.labels
    if cts is None or emitter.layer_labels is None:
        return None
    n = int(emitter.v0.shape[0])
    lab = _np.asarray([str(x) for x in cts])
    Z = _np.asarray(positions)[:, 2]
    if Z.shape[0] != n or float(_np.ptp(Z)) == 0.0:
        return None
    zd = (Z - Z.min()) / (float(_np.ptp(Z)) + 1e-9)
    isE = lab == "E"
    return lab, Z, zd, isE, n


def _apply_canonical_dynamics_init(emitter, positions, cfg):
    """DYNAMICS-INIT half of canonical biophysics: random ``v0`` + deep-E grading.

    * **Random ``v0``** ~ Uniform(-70, 0) mV per neuron — ALWAYS applied (keyed off
      ``cfg.metadata["seed"]``). Identical initial potentials cause a synchronized
      t=0 onset volley (raster banding); randomizing removes that onset-response
      synchrony bias. Disable with ``cfg.runtime(random_v0=False)``.
    * **Deep-E "larger" grading** (laminar columns only, gated on
      ``cfg.metadata["canonical_biophysics"]``): excitatory neurons get a depth
      gradient — ``source_scale`` 1.0->1.8 (bigger dipole), ``a`` 0.020->0.015
      (slower recovery), ``d`` 8->10 (wider/stronger).

    Touches only the emitter (v0/a/d/source_scale) -- never edge_list. Returns
    the (possibly replaced) ``emitter``. Proxy/scaffold truth status unchanged.
    """
    import numpy as _np
    jdtype = emitter.v0.dtype
    n = int(emitter.v0.shape[0])
    seed = int(cfg.metadata.get("seed", 0) or 0)

    if cfg.metadata.get("random_v0", True):
        vkey = jax.random.PRNGKey((seed * 1_000_003 + 11) % (2**31 - 1))
        v0 = jax.random.uniform(vkey, (n,), dtype=jdtype, minval=-70.0, maxval=0.0)
        emitter = replace(emitter, v0=v0)

    if not cfg.metadata.get("canonical_biophysics", False):
        return emitter

    gating = _canonical_biophysics_depth_gating(emitter, positions)
    if gating is None:
        return emitter
    _lab, _Z, zd, isE, _n = gating

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

    return emitter


def _apply_canonical_edge_strengthening(emitter, positions, edge_list, cfg):
    """GRAPH-EDIT half of canonical biophysics: PV<->E local connectivity x3
    (laminar columns only, gated on ``cfg.metadata["canonical_biophysics"]``).

    The fast feedback-inhibition (PING) loop is strengthened, distance-gated by
    laminar depth. Reads ``emitter.labels``/``positions`` (post deep-E grading or
    not -- labels/positions are unaffected by dynamics-init) but touches only
    ``edge_list.weight`` -- never the emitter. Returns the (possibly replaced)
    ``edge_list``. Proxy/scaffold truth status unchanged.
    """
    import numpy as _np
    if not cfg.metadata.get("canonical_biophysics", False):
        return edge_list

    gating = _canonical_biophysics_depth_gating(emitter, positions)
    if gating is None:
        return edge_list
    lab, Z, _zd, _isE, _n = gating
    jdtype = emitter.v0.dtype

    pre = _np.asarray(edge_list.pre)
    post = _np.asarray(edge_list.post)
    pc = lab[pre]
    qc = lab[post]
    pv_e = ((pc == "PV") & (qc == "E")) | ((pc == "E") & (qc == "PV"))
    if pv_e.any():
        w = _np.asarray(edge_list.weight, dtype=float).copy()
        gate = _np.exp(-((_np.abs(Z[pre] - Z[post]) / 0.15) ** 2))
        w[pv_e] = w[pv_e] * (1.0 + 2.0 * gate[pv_e])
        edge_list = replace(edge_list, weight=jnp.asarray(w, dtype=jdtype))

    return edge_list


def _apply_canonical_biophysics(emitter, positions, edge_list, cfg):
    """Apply canonical cortical-column biophysics at construct time.

    Backward-compatible wrapper: dispatches to the DYNAMICS-INIT half
    (:func:`_apply_canonical_dynamics_init` -- random v0 + deep-E grading) and
    the GRAPH-EDIT half (:func:`_apply_canonical_edge_strengthening` -- PV<->E
    local strengthening) in sequence, in this order because the edge-strengthening
    gate reads ``emitter.labels`` (unaffected by dynamics-init) but not any
    dynamics-init output, so ordering does not change the result. Split out
    2026-07-24 so a future ``tensor_to_graph()`` can call the graph-edit half
    alone without touching emitter dynamics state. Returns ``(emitter,
    edge_list)``, identical to the pre-split behavior. Proxy/scaffold truth
    status unchanged.
    """
    emitter = _apply_canonical_dynamics_init(emitter, positions, cfg)
    edge_list = _apply_canonical_edge_strengthening(emitter, positions, edge_list, cfg)
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
        g = diag["g_bias"]
        r = diag["r_trace"]
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

    Called only when ``runtime_cfg.enable_hdp`` is True. Framing: a
    finite-dimensional per-neuron H-state plasticity controller -- a
    COMPUTATIONAL method, NOT a biological mechanism, matching the
    homeostasis controller's claim discipline.
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
    h_state_dim = int(_hp_meta.get("h_state_dim", 1))
    readout = _hp_meta.get("h_state_readout")
    coupling = _hp_meta.get("h_state_coupling")
    hdp_meta["h_state"] = {
        "h_state_dim": h_state_dim,
        "readout": {
            "kind": "linear",
            "shape": [h_state_dim],
            "configured": readout is not None,
        },
        "coupling": {
            "enabled": coupling is not None,
            "shape": [h_state_dim, h_state_dim],
        },
    }
    if diag is not None:
        H = diag["H_trace"]
        wf = diag["w_final"]
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


_SUPPORTED_EMITTER_FAMILIES = frozenset({"izhikevich", "homeostatic_ei"})


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
        params, positions, geometry_meta, _prebuilt_edges = _neuron_population_from_config(cfg, dtype=dtype_name_cfg)
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
        # This branch's builder (make_eig_network -> _default_eig_connectivity)
        # unconditionally materializes a dense (n,n) all-to-all weight matrix --
        # it has no sparse-direct escape, unlike _neuron_population_from_config's
        # _apply_connectivity above. A Configuration.connectivity(p_connect=...)
        # call has NO effect here: p_connect is only read from cfg.metadata by
        # _apply_connectivity, which this branch never reaches (routing above
        # requires columns/layer_cell_types/uniform_3d in cfg.metadata -- a plain
        # .network(kind=..., layers=[...]) call, without one of those three keys
        # also set, silently falls through to here regardless of `layers`).
        # Confirmed 2026-07-21: a 100,000-neuron .network(layers=[...]) config
        # with .connectivity(p_connect=0.0005) still built a dense (n,n) matrix
        # (~40GB) and ran ~370s -- p_connect was silently inert.
        cell_types = net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.1})
        if n >= _DENSE_CONNECTIVITY_WARN_N:
            import warnings as _warnings
            _mb = (n * n * 4) / 1e6
            _warnings.warn(
                f"dense all-to-all connectivity at N={n} materializes an "
                f"({n}x{n}) weight matrix (~{_mb:.0f} MB) -- this Configuration "
                f"did not set columns/layer_cell_types/uniform_3d metadata, so it "
                f"routed to the non-sparse-aware network builder. Any "
                f"connectivity(p_connect=...) call has NO effect on this path. "
                f"Use build_laminar_column/laminar_cortex_config (or set one of "
                f"those metadata keys directly) to reach the sparse-aware "
                f"population builder at this scale.",
                RuntimeWarning,
                stacklevel=2,
            )
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


def _record_connectivity_compilation(
    cfg: "Configuration",
    *,
    mode: str,
    default_edge_count: int,
    declared_edge_count: int,
    total_edge_count: int,
) -> "Configuration":
    """Attach mechanically derived tensor connectivity counts to the config."""
    metadata = dict(cfg.metadata)
    metadata["connectivity_compilation"] = {
        "connectivity_mode": mode,
        "default_edge_count": int(default_edge_count),
        "declared_rule_edge_count": int(declared_edge_count),
        "total_compiled_edge_count": int(total_edge_count),
    }
    return replace(cfg, metadata=metadata)


def _reject_duplicate_explicit_edges(edge_list: "EdgeList") -> None:
    """Reject accidental duplicate explicit synaptic identities.

    Distinct receptor/mechanism indices are valid parallel edges. The identity
    rejected here is exactly ``(pre, post, receptor_index)``.
    """
    import numpy as _np

    if edge_list.n_edges < 2:
        return
    identities = _np.stack(
        (
            _np.asarray(edge_list.pre, dtype=_np.int64),
            _np.asarray(edge_list.post, dtype=_np.int64),
            _np.asarray(edge_list.receptor_index, dtype=_np.int64),
        ),
        axis=1,
    )
    if len(_np.unique(identities, axis=0)) != edge_list.n_edges:
        raise ValueError(
            "explicit NeuronalTensor connectivity produced duplicate "
            "executable synaptic identities; identity is "
            "(pre, post, receptor_index)"
        )


def _construct_compile_connections(
    cfg: "Configuration", network: "EIGNetwork", n: int, geometry_meta: "dict[str, Any] | None",
    net: Mapping[str, Any], edge_list: "EdgeList", positions: "jax.Array | None" = None,
) -> "tuple[Configuration, EdgeList]":
    """``construct()`` stage: compile declarative ``.connections()`` rules into
    real edges -> ``(cfg, edge_list)``. Explicit tensor mode starts from an
    empty prebuilt graph; unspecified mode retains the configured graph.
    Because the dense backend runs on ``emitter.W`` (which does not carry
    declared rules), force the edge_list backend whenever a rule materializes.
    Rule statuses flip declared->compiled.
    """
    connectivity_mode = cfg.metadata.get("connectivity_mode")
    if connectivity_mode not in (None, "unspecified", "explicit"):
        raise ValueError(
            "unsupported connectivity_mode for construction: "
            f"{connectivity_mode!r}"
        )
    default_edge_count = int(edge_list.n_edges)
    declared_edge_count = 0
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
        declared_edge_count = sum(int(count) for count in _counts)
        if connectivity_mode == "explicit" and _conn_edges is not None:
            _reject_duplicate_explicit_edges(_conn_edges)
        if _conn_edges is not None and _conn_edges.n_edges > 0:
            edge_list = _concat_edge_lists(edge_list, _conn_edges)
            cfg = cfg.runtime(recurrent_backend="edge_list")
        cfg = _mark_connections_compiled(cfg, _counts)
    if connectivity_mode is not None:
        cfg = _record_connectivity_compilation(
            cfg,
            mode=connectivity_mode,
            default_edge_count=(
                0 if connectivity_mode == "explicit" else default_edge_count
            ),
            declared_edge_count=declared_edge_count,
            total_edge_count=int(edge_list.n_edges),
        )
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


# Canonical, empirically-verified-stable defaults for the homeostatic_ei
# circuit (see tests/test_homeostatic_ei_*.py for the receipts: "linear"
# activation is stable at fixed G but diverges once G-adaptation is enabled;
# "cubic" activation's self-damping is required for that regime). These
# numeric constants are fixed for this pass -- only the three rule names and
# the network size (n_neurons, via Configuration.network(n=...)) are
# configurable via the grammar; exposing per-neuron overrides is deferred
# (see artifacts/developer/plans.json entry
# homeostatic-ei-milestones-4-6-regime-sweep).
_HOMEOSTATIC_EI_CANONICAL_DEFAULTS: dict[str, Any] = {
    "x0_value": 0.1,
    "G0_e_value": 0.5,
    "G0_i_value": -0.5,
    "H0_value": 0.3,
    "drive_e_value": 0.5,
    "drive_i_value": 0.3,
    "tau_x_ms": 5.0,
    "tau_G_ms": 200.0,
    "tau_H_ms": 1000.0,
    "G_min": -5.0,
    "G_max": 5.0,
    "H_min": 0.1,
    "H_max": 10.0,
    "source_scale_value": 1.0,
    "activation_rule": "cubic",
    "conductance_rule": "hebbian",
    "homeostasis_rule": "linear",
}


def _homeostatic_ei_cell_type_split(n: int) -> "tuple[str, ...]":
    """E/I label split for the homeostatic_ei canonical circuit: 1E/1I at
    n=2 (the original 2-neuron circuit, kept bit-identical), else a 75/25
    E/I split (matching ``scripts/major_sanity_test.py``'s 6E/2I convention
    at n=8) with at least 1 I neuron. E neurons come first, I neurons last.
    """
    if n <= 2:
        n_e = max(1, n - 1)
    else:
        n_e = max(1, min(n - 1, round(n * 0.75)))
    n_i = n - n_e
    return tuple(["E"] * n_e + ["I"] * n_i)


def _construct_homeostatic_ei_model(cfg: Configuration) -> Model:
    """Build a :class:`Model` for the ``homeostatic_ei`` emitter family --
    the second canonical HDP sanity circuit (a minimal E/I circuit with an
    explicit differentiable conductance matrix and HDP state; see
    ``jaxfne/emitters_homeostatic_ei.py``'s module docstring). Network size
    comes from ``cfg.networks[0]["n"]`` (default 2, the original minimal
    circuit); at ``n=2`` every value below reduces exactly to the original
    hardcoded 2-neuron defaults (verified bit-identical).

    This is a separate, self-contained build path -- it does not run
    ``_construct_build_network``/``_construct_resolve_edge_list``/
    ``_apply_canonical_biophysics`` (those are Izhikevich/edge-list/laminar-
    geometry specific and would build the wrong thing for a dense-``G``
    E/I circuit). ``model.params["edge_list"]`` is populated with a real
    empty :class:`EdgeList` (via :func:`_empty_edge_list`), not ``None`` --
    several other ``Model`` methods (e.g. ``neuron_table``, ``checkpoint``)
    unconditionally type-annotate and unpack ``params["edge_list"]`` and are
    not yet generalized for this family (see the ``NotImplementedError``
    guards added to those methods in ``jaxfne/_model.py``).
    """
    rules = dict((cfg.emitters[0].get("homeostatic_ei_rules") if cfg.emitters else None) or {})
    activation_rule = str(rules.get("activation_rule", _HOMEOSTATIC_EI_CANONICAL_DEFAULTS["activation_rule"]))
    conductance_rule = str(rules.get("conductance_rule", _HOMEOSTATIC_EI_CANONICAL_DEFAULTS["conductance_rule"]))
    homeostasis_rule = str(rules.get("homeostasis_rule", _HOMEOSTATIC_EI_CANONICAL_DEFAULTS["homeostasis_rule"]))
    bound_mode = str((cfg.emitters[0].get("homeostatic_ei_bound_mode") if cfg.emitters else None) or "minimal")
    if bound_mode not in ("minimal", "stable"):
        raise ValueError(f"unknown bound_mode {bound_mode!r} for homeostatic_ei; expected 'minimal' or 'stable'")
    for name, registry, kind in (
        (activation_rule, ACTIVATION_RULES, "activation"),
        (conductance_rule, CONDUCTANCE_RULES, "conductance"),
        (homeostasis_rule, HOMEOSTASIS_RULES, "homeostasis"),
    ):
        if name not in registry:
            raise ValueError(f"unknown {kind}_rule {name!r} for homeostatic_ei; expected one of {sorted(registry)}")

    dtype_name_cfg = str(cfg.metadata.get("dtype", "float32"))
    jdtype = jnp.dtype(dtype_name_cfg)
    d = _HOMEOSTATIC_EI_CANONICAL_DEFAULTS

    n = int(cfg.networks[0].get("n", 2)) if cfg.networks else 2
    if n < 2:
        raise ValueError(f"homeostatic_ei requires at least 2 neurons; got n={n}")
    labels = _homeostatic_ei_cell_type_split(n)
    is_e = jnp.asarray([label == "E" for label in labels])
    n_e = int(is_e.sum())
    n_i = n - n_e

    x0 = jnp.full((n,), d["x0_value"], dtype=jdtype)
    H0 = jnp.full((n,), d["H0_value"], dtype=jdtype)
    source_scale = jnp.full((n,), d["source_scale_value"], dtype=jdtype)
    drive = jnp.where(is_e, jnp.asarray(d["drive_e_value"], dtype=jdtype), jnp.asarray(d["drive_i_value"], dtype=jdtype))
    # G0[i, j] = (per-type value) / (count of that type) for every row i --
    # normalizes total incoming drive so it matches the original 2-neuron
    # circuit's magnitude regardless of population size (1 E + 1 I there).
    col_value = jnp.where(
        is_e, jnp.asarray(d["G0_e_value"], dtype=jdtype) / max(n_e, 1),
        jnp.asarray(d["G0_i_value"], dtype=jdtype) / max(n_i, 1),
    )
    G0 = jnp.broadcast_to(col_value[None, :], (n, n))

    emitter_params = HomeostaticEIParams(
        x0=x0,
        G0=G0,
        H0=H0,
        drive=drive,
        tau_x_ms=jnp.asarray(d["tau_x_ms"], dtype=jdtype),
        tau_G_ms=jnp.asarray(d["tau_G_ms"], dtype=jdtype),
        tau_H_ms=jnp.asarray(d["tau_H_ms"], dtype=jdtype),
        G_min=jnp.asarray(d["G_min"], dtype=jdtype),
        G_max=jnp.asarray(d["G_max"], dtype=jdtype),
        H_min=jnp.asarray(d["H_min"], dtype=jdtype),
        H_max=jnp.asarray(d["H_max"], dtype=jdtype),
        source_scale=source_scale,
        labels=labels,
        activation_rule_name=activation_rule,
        conductance_rule_name=conductance_rule,
        homeostasis_rule_name=homeostasis_rule,
        bound_mode=bound_mode,
    )
    z = jnp.linspace(0.0, 1.0, n, dtype=jdtype) if n > 1 else jnp.zeros((1,), dtype=jdtype)
    positions = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=1)
    edge_list = _empty_edge_list(jdtype)
    static = _construct_build_static(cfg, geometry_meta=None)
    return Model(
        cfg=cfg,
        params={"emitter": emitter_params, "positions": positions, "edge_list": edge_list},
        static=static,
    )


def _construct_from_configuration(cfg: Configuration, *, geometry: "LaminarSourceGeometry | None" = None) -> Model:
    """Validate a :class:`Configuration` and build a runnable :class:`Model`.

    Raises ``ValueError`` if the configuration is invalid or names an
    unsupported emitter family (only the Izhikevich and homeostatic_ei
    kernels are implemented; an explicitly declared family such as
    ``"lif"``/``"glif"`` fails loudly rather than silently substituting
    Izhikevich). An optional :class:`LaminarSourceGeometry` overrides
    geometry derived from the config. The returned model is a computational
    scaffold; its field/probe outputs are proxy readouts, not calibrated
    physical signals.
    """
    _construct_validate_config(cfg)
    if cfg.emitters and cfg.emitters[0].get("family") == "homeostatic_ei":
        return _construct_homeostatic_ei_model(cfg)
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


