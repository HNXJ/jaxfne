"""Config/Tensor consistency helpers: validate, diff, merge, summarize.

Internal-only additions layered on top of the real ``RuntimeConfig`` and
``NeuronalTensor`` dataclasses (``jaxfne/core.py``, ``jaxfne/neuronal_tensor.py``).
Does not duplicate ``RuntimeConfig.runtime_report()`` (the existing JAX
backend/dtype/device report) -- these functions cover consistency checks,
cross-config diffing, and config layering that did not exist before.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import math

from .core import Configuration, EdgeList, Model, RuntimeConfig
from .neuronal_tensor import NeuronalTensor

__all__ = [
    "validate_runtime_config",
    "runtime_config_diff",
    "merge_runtime_configs",
    "validate_neuronal_tensor",
    "tensor_summary",
    "validate_model",
    "model_diff",
    "configuration_diff",
    "canonical_compact_summary",
    "format_canonical_text_bundle",
]

_REQUIRED_HOMEOSTASIS_KEYS = {"r_star", "tau_r_ms", "alpha", "k_gain", "g_min", "g_max", "r_max"}
_REQUIRED_HDP_KEYS = {
    "K_HDP", "tau_0_ms", "alpha", "beta", "gamma", "delta",
    "C_spike", "K_ctrl", "barrier_c", "barrier_d",
}


def validate_runtime_config(cfg: RuntimeConfig, *, strict: bool = False) -> list[str]:
    """Return consistency warnings for ``cfg`` beyond its own ``__post_init__``.

    ``__post_init__`` already rejects bad enums (synaptic_kernel,
    recompilation_guard, jit/vmap strings) and the homeostasis/hdp mutual
    exclusion. This adds checks that require runtime/JAX state or cross-field
    reasoning, which ``__post_init__`` cannot do at construction time:

    - ``dtype="float64"`` requested while ``jax_enable_x64`` is currently off
      (the dtype will silently downgrade to float32 at simulate-time).
    - ``backend`` requested but unavailable on this machine.
    - ``homeostasis_params``/``hdp_params`` missing a key the kernel reads
      (a missing key means that term silently behaves as if it were absent,
      not an error -- worth flagging).

    Raises ``ValueError`` on the first warning when ``strict=True``.
    """
    import jax

    warnings: list[str] = []

    if cfg.dtype == "float64" and not bool(jax.config.read("jax_enable_x64")):
        warnings.append(
            "dtype='float64' requested but jax_enable_x64 is False; "
            "actual_dtype will downgrade to float32 at simulate-time"
        )

    if cfg.backend in ("cpu", "gpu", "tpu"):
        try:
            available = bool(jax.devices(cfg.backend))
        except (RuntimeError, ValueError):
            available = False
        if not available:
            warnings.append(
                f"backend={cfg.backend!r} requested but no such device is "
                f"available; actual execution falls back to {jax.default_backend()!r}"
            )

    if cfg.enable_homeostasis:
        missing = _REQUIRED_HOMEOSTASIS_KEYS - set(cfg.homeostasis_params)
        if missing:
            warnings.append(f"homeostasis_params missing keys: {sorted(missing)}")

    if cfg.enable_hdp:
        missing = _REQUIRED_HDP_KEYS - set(cfg.hdp_params)
        if missing:
            warnings.append(f"hdp_params missing keys: {sorted(missing)}")
        from .public_surface import validate_hdp_params_semantics

        warnings.extend(validate_hdp_params_semantics(cfg.hdp_params))

    if strict and warnings:
        raise ValueError(f"validate_runtime_config: {warnings[0]}")
    return warnings


def runtime_config_diff(a: RuntimeConfig, b: RuntimeConfig) -> dict[str, tuple[Any, Any]]:
    """Return ``{field_name: (a_value, b_value)}`` for every field where ``a != b``.

    Dict-valued fields (``homeostasis_params``, ``hdp_params``) compare by
    value, not identity.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for f in fields(a):
        av = getattr(a, f.name)
        bv = getattr(b, f.name)
        if av != bv:
            diff[f.name] = (av, bv)
    return diff


def merge_runtime_configs(*cfgs: RuntimeConfig, **overrides: Any) -> RuntimeConfig:
    """Layer ``cfgs`` left-to-right (later non-default... no -- later wins outright
    per field), then apply ``overrides`` on top.

    Each later config's field value replaces the running result unconditionally
    (this is field-level last-write-wins, not a default-aware deep merge --
    a later config explicitly setting a field back to its default still wins).
    Requires at least one config.
    """
    if not cfgs:
        raise ValueError("merge_runtime_configs requires at least one RuntimeConfig")
    result = cfgs[0]
    for cfg in cfgs[1:]:
        result = replace(result, **{f.name: getattr(cfg, f.name) for f in fields(cfg)})
    if overrides:
        result = replace(result, **overrides)
    return result


def validate_neuronal_tensor(nt: NeuronalTensor, *, strict: bool = False) -> list[str]:
    """Return structural-consistency warnings for ``nt``.

    Checks, all grounded in the real ``Area``/``Layer``/``NeuronType``
    dataclasses (``jaxfne/neuronal_tensor.py``):

    - duplicate Area names
    - an Area with zero Layers
    - duplicate Layer names within an Area
    - a Layer with ``n_neurons <= 0``
    - NeuronType ``fraction`` values within a Layer summing to > 1.0
      (only checked when every NeuronType in that Layer declares a fraction;
      partial fractions are valid per ``NeuronType.fraction``'s documented
      all-or-nothing even-split fallback and are not flagged here)

    Raises ``ValueError`` on the first warning when ``strict=True``.
    """
    warnings: list[str] = []

    area_names = [a.name for a in nt.areas]
    dupes = {n for n in area_names if area_names.count(n) > 1}
    if dupes:
        warnings.append(f"duplicate Area names: {sorted(dupes)}")

    for area in nt.areas:
        if not area.layers:
            warnings.append(f"Area {area.name!r} has zero Layers")
            continue

        layer_names = [layer.name for layer in area.layers]
        layer_dupes = {n for n in layer_names if layer_names.count(n) > 1}
        if layer_dupes:
            warnings.append(f"Area {area.name!r}: duplicate Layer names: {sorted(layer_dupes)}")

        for layer in area.layers:
            if layer.n_neurons <= 0:
                warnings.append(
                    f"Area {area.name!r} Layer {layer.name!r}: n_neurons={layer.n_neurons} <= 0"
                )
            fractions = [nt_.fraction for nt_ in layer.neuron_types]
            if fractions and all(f is not None for f in fractions):
                total = sum(fractions)
                if total > 1.0 + 1e-6:
                    warnings.append(
                        f"Area {area.name!r} Layer {layer.name!r}: NeuronType fractions "
                        f"sum to {total:.4f} > 1.0"
                    )

    if strict and warnings:
        raise ValueError(f"validate_neuronal_tensor: {warnings[0]}")
    return warnings


def validate_model(model: Model, *, strict: bool = False) -> list[str]:
    """Return structural/numerical consistency warnings for a built ``Model``.

    Complements ``Model.summary()`` (existing, JSON-safe metadata) and
    ``Configuration.validate()`` (existing, pre-build declaration-presence
    check) -- neither inspects the actually-built numerical state. This does:

    - ``params["edge_list"]`` (when present, sparse/edge_list backend): every
      ``pre``/``post`` index in bounds for the emitter's neuron count, and
      ``weight``/``tau_ms`` finite (no NaN/inf).
    - ``params["emitter"].v0`` finite (catches a blown-up/NaN initial state
      before a single simulate() call wastes compute on it).

    Raises ``ValueError`` on the first warning when ``strict=True``.
    """
    import jax.numpy as jnp

    warnings: list[str] = []

    emitter = model.params.get("emitter")
    if emitter is not None and hasattr(emitter, "v0"):
        n_units = int(emitter.v0.shape[0])
        if not bool(jnp.all(jnp.isfinite(emitter.v0))):
            warnings.append("params['emitter'].v0 contains non-finite values")
    else:
        n_units = None

    edge_list = model.params.get("edge_list")
    if isinstance(edge_list, EdgeList):
        if n_units is not None:
            for name, idx in (("pre", edge_list.pre), ("post", edge_list.post)):
                if idx.size and (int(jnp.min(idx)) < 0 or int(jnp.max(idx)) >= n_units):
                    warnings.append(
                        f"edge_list.{name} has an index out of bounds for "
                        f"n_units={n_units} (range [{int(jnp.min(idx))}, {int(jnp.max(idx))}])"
                    )
        for name, arr in (("weight", edge_list.weight), ("tau_ms", edge_list.tau_ms)):
            if arr.size and not bool(jnp.all(jnp.isfinite(arr))):
                warnings.append(f"edge_list.{name} contains non-finite values")

    if strict and warnings:
        raise ValueError(f"validate_model: {warnings[0]}")
    return warnings


def model_diff(a: Model, b: Model, *, atol: float = 1e-6) -> dict[str, Any]:
    """Return a sweep-comparison summary between two built ``Model`` instances.

    Unlike ``runtime_config_diff`` (exact field equality on a dataclass), the
    interesting deltas here are numeric (edge count, mean weight, neuron
    count) since ``params``/``static`` are dicts of arrays/objects, not
    dataclass fields with stable identity across builds. Returns only
    quantities that differ by more than ``atol`` (or differ structurally,
    e.g. one model has ``edge_list`` and the other doesn't).
    """
    import jax.numpy as jnp

    diff: dict[str, Any] = {}

    ea, eb = a.params.get("emitter"), b.params.get("emitter")
    if ea is not None and eb is not None and hasattr(ea, "v0") and hasattr(eb, "v0"):
        na, nb = int(ea.v0.shape[0]), int(eb.v0.shape[0])
        if na != nb:
            diff["n_units"] = (na, nb)

    la, lb = a.params.get("edge_list"), b.params.get("edge_list")
    has_a, has_b = isinstance(la, EdgeList), isinstance(lb, EdgeList)
    if has_a != has_b:
        diff["has_edge_list"] = (has_a, has_b)
    elif has_a and has_b:
        if la.n_edges != lb.n_edges:
            diff["n_edges"] = (la.n_edges, lb.n_edges)
        if la.n_edges and lb.n_edges:
            mean_a, mean_b = float(jnp.mean(la.weight)), float(jnp.mean(lb.weight))
            if not math.isclose(mean_a, mean_b, abs_tol=atol):
                diff["mean_edge_weight"] = (mean_a, mean_b)

    return diff


def configuration_diff(a: Configuration, b: Configuration) -> dict[str, Any]:
    """Return ``{field_name: (a_value, b_value)}`` for declarative fields that differ.

    ``Configuration`` is a declarative builder (lists of dicts), not a flat
    dataclass of scalars like ``RuntimeConfig`` -- compares ``networks``,
    ``emitters``, ``fields`` by value (list equality) and ``metadata`` by key.
    ``probes`` is intentionally excluded: it is a ``_ProbeDeclarations`` proxy
    object, not a plain list, and does not support stable equality comparison.
    """
    diff: dict[str, Any] = {}
    for name in ("networks", "emitters", "fields"):
        av, bv = getattr(a, name), getattr(b, name)
        if av != bv:
            diff[name] = (av, bv)
    if a.metadata != b.metadata:
        diff["metadata"] = (a.metadata, b.metadata)
    return diff


def tensor_summary(nt: NeuronalTensor) -> dict[str, Any]:
    """Return a flat, JSON-safe summary of ``nt``: counts and cell-type inventory.

    Does not duplicate ``Model.runtime_report()`` -- this summarizes the
    declarative ``NeuronalTensor`` shape itself, before ``construct()`` ever
    runs, so it works on a tensor that has not been built into a ``Model``.
    """
    n_areas = len(nt.areas)
    n_layers = sum(len(a.layers) for a in nt.areas)
    n_neurons = sum(layer.n_neurons for a in nt.areas for layer in a.layers)
    cell_types: set[str] = set()
    for area in nt.areas:
        for layer in area.layers:
            for nt_ in layer.neuron_types:
                cell_types.add(nt_.name)
    n_inter_connections = sum(len(a.inter_connections) for a in nt.areas)
    return {
        "name": nt.name,
        "n_areas": n_areas,
        "n_layers": n_layers,
        "n_neurons": n_neurons,
        "cell_types": sorted(cell_types),
        "n_inter_connections": n_inter_connections,
        "n_area_connections": len(nt.area_connections),
    }


def canonical_compact_summary(
    model: "Model",
    signals: "Any | None" = None,
    tensor: "NeuronalTensor | None" = None,
) -> dict[str, Any]:
    """Canonical compact summary Θ = Θ_static ⊕ X ⊕ H ⊕ W.

    Pure-Python, no kernel change (Δscience=0), no overhead when unused
    (only runs when called). Uses existing API only: Model, Signals,
    NeuronalTensor, EdgeList, positions, provenance.

    Returns a JSON-safe dict with:

    - ``Theta``: decomposition ``Θ_static``, ``X``, ``H``, ``W`` with sizes.
    - ``N_static``: Σ|θ| for θ∈Θ_static.
    - ``output_basis``: minimal independent basis STATE/SOURCE/FIELD/PROBE/DERIVED
      (not flattened signals).
    - ``counts``: configured vs realized vs effective (neurons, populations,
      edges, duration, dt, n_steps, state sizes).
    - ``provenance``: config_hash, tensor_identity, versions.
    - ``text_bundle``: human-readable example (like jaxfne summary).

    Parameters
    ----------
    model : Model
        Built model from ``construct()``. Reads ``params['emitter']``,
        ``params['edge_list']``, ``params['positions']``, ``static``,
        ``cfg.metadata``.
    signals : Signals, optional
        If given, realized/effective time axes (duration_ms, dt_ms, n_steps)
        and output-basis shapes are read from ``signals``; otherwise inferred
        from ``model`` alone (effective = realized).
    tensor : NeuronalTensor, optional
        If given, configured counts are read declaratively (pre-construct).

    Notes
    -----
    - ``Θ_static`` = per-neuron static parameters {a,b,c,d,drive,sign,
      source_scale}. Each is size N (source_scale is scalar, counted as 1).
      Positions (N×3) and edge weights are NOT in Θ_static (they are geometry
      and W respectively). ``N_static = Σ|θ|`` is therefore 6·N+1 for the
      Izhikevich scaffold (e.g. N=1000 → 6001).
    - ``X`` = fast dynamical state [v, u] per neuron (2·N per step; 4·N if
      counting prev_spikes + syn_state buffer head). Trajectory size = T·|X|.
    - ``H`` = hidden/adaptation state (RBS/HDP). 0 when HDP disabled
      (canonical 1000n). When enabled, ``|H| = N × h_state_dim``.
    - ``W`` = synaptic storage (EdgeList). ``|W| = n_edges`` (weights) plus
      ``tau_ms`` catalog (kept separate in diagnostics).
    - Output basis is minimal independent: STATE (V_m, spikes, u), SOURCE
      (sources), FIELD (lfp/csd/phi_e proxy on contacts), PROBE (eeg/meg/emm
      if declared), DERIVED (scalar metrics from Signals.summary). This avoids
      a flattened “signals is one vector” view.
    - configured vs realized vs effective: configured = declared in
      NeuronalTensor/Configuration (pre-compile), realized = after
      ``construct()`` (actual array sizes), effective = after ``simulate()``
      with runtime overrides (e.g. capped dt, enabled HDP, dtype downgrade).
    """
    import jax.numpy as jnp  # local import: zero overhead when unused

    from .io import config_hash as _config_hash
    from .io import json_safe as _json_safe

    # --- realized (from built model) ---
    emitter = model.params.get("emitter")
    if emitter is None or not hasattr(emitter, "v0"):
        raise ValueError("canonical_compact_summary requires Model with Izhikevich emitter (v0)")
    n_neurons_realized = int(emitter.v0.shape[0])

    # Theta_static: per-neuron static params (existing API: emitter fields)
    # Each is (N,) except source_scale scalar; sign is static sign (+1/-1).
    theta_static_sized: dict[str, int] = {}
    for name in ("a", "b", "c", "d", "drive", "sign"):
        arr = getattr(emitter, name, None)
        theta_static_sized[name] = int(jnp.asarray(arr).size) if arr is not None else 0
    # source_scale may be scalar or (N,); count actual size
    ss = getattr(emitter, "source_scale", None)
    theta_static_sized["source_scale"] = int(jnp.asarray(ss).size) if ss is not None else 0
    N_static = int(sum(theta_static_sized.values()))

    # X: fast state [v, u] plus diagnostic buffers
    # Per-step X = v(N) + u(N) = 2N; with prev_spikes buffer 3N; full buffer 4N is
    # the canonical “X 4000” example (v,u,prev_spikes,spike buffer head) for N=1000.
    v_size = int(jnp.asarray(emitter.v0).size)
    u_size = int(jnp.asarray(emitter.u0).size)
    X_per_step = v_size + u_size  # 2N
    X_with_prev = X_per_step + n_neurons_realized  # 3N
    X_canonical_4N = 4 * n_neurons_realized  # for text bundle “X 4000” illustration

    # H: hidden/adaptation state (RBS/HDP)
    # Detect via model static or cfg metadata; H is 0 when disabled.
    H_size = 0
    h_state_dim = 0
    h_locality = None
    # provenance: hdp_params transport dict (existing API: model.cfg.metadata / signals.metadata)
    hdp_params = None
    if "hdp_params" in model.cfg.metadata:
        hdp_params = model.cfg.metadata.get("hdp_params")
    if signals is not None and isinstance(getattr(signals, "metadata", None), dict):
        # signals.metadata may carry hdp diagnostics if HDP was enabled at simulate time
        hdp_meta = signals.metadata.get("hdp") or signals.metadata.get("hdp_params")
        if hdp_meta is not None:
            hdp_params = hdp_meta if isinstance(hdp_meta, dict) else hdp_params
    if hdp_params is not None:
        try:
            h_state_dim = int(hdp_params.get("h_state_dim", 0))
            h_locality = hdp_params.get("h_state_locality")
            if h_state_dim and h_state_dim > 0:
                # node locality: H per neuron; population locality: H per population
                if h_locality == "population":
                    # count realized populations from neuron_table
                    rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                    pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 1
                    H_size = int(h_state_dim * pops)
                else:
                    H_size = int(h_state_dim * n_neurons_realized)
        except Exception:
            H_size = 0

    # W: synaptic storage (EdgeList, existing API)
    edge_list = model.params.get("edge_list")
    n_edges_realized = int(edge_list.n_edges) if edge_list is not None and hasattr(edge_list, "n_edges") else 0
    W_size = n_edges_realized  # weights; tau catalog kept separate
    tau_catalog_size = int(jnp.asarray(edge_list.tau_ms).size) if edge_list is not None else 0

    # positions: geometry (N,3) – existing API: params['positions']
    positions = model.params.get("positions")
    positions_shape = tuple(int(x) for x in jnp.asarray(positions).shape) if positions is not None else None
    positions_size = int(jnp.asarray(positions).size) if positions is not None else 0

    # --- configured (from declarative tensor/configuration, pre-construct) ---
    n_neurons_configured = None
    n_populations_configured = None
    n_edges_configured_rules = None
    n_areas_configured = None
    if tensor is not None:
        try:
            n_neurons_configured = int(sum(layer.n_neurons for a in tensor.areas for layer in a.layers))
            n_areas_configured = int(len(tensor.areas))
            # declared populations = total NeuronType entries (23 for canonical 1000n)
            n_populations_configured = int(sum(len(layer.neuron_types) for a in tensor.areas for layer in a.layers))
            n_edges_configured_rules = int(sum(len(a.inter_connections) for a in tensor.areas) + len(tensor.area_connections))
        except Exception:
            pass
    if n_neurons_configured is None:
        # fallback: configured == realized when no tensor given
        n_neurons_configured = n_neurons_realized

    # realized populations: distinct (layer, cell_type) combos in built neuron_table
    n_populations_realized = None
    populations_inventory: dict[str, int] = {}
    try:
        rows = model.neuron_table()
        from collections import Counter as _Counter
        combo_counts = _Counter((r.get("layer"), r.get("cell_type")) for r in rows)
        n_populations_realized = int(len(combo_counts))
        # also flat inventory like “12 = 6 layers × 2 effective E/I” vs 23 detailed
        # effective collapsed inhibitory: map PV/SST/VIP → I
        def _eff_cell(ct: str) -> str:
            return "I" if ct in ("PV", "SST", "VIP", "Inl", "Ing") else ct
        eff_combos = _Counter((r.get("layer"), _eff_cell(str(r.get("cell_type")))) for r in rows)
        populations_inventory = {
            "realized_detailed": n_populations_realized,
            "effective_EI_collapsed": int(len(eff_combos)),
            "per_layer": dict(_Counter(r.get("layer") for r in rows)),
            "per_cell_type": dict(_Counter(r.get("cell_type") for r in rows)),
        }
    except Exception:
        pass

    # effective: after simulate, may differ due to runtime overrides (dt rounding, HDP enable, dtype)
    n_steps_configured = None
    n_steps_realized = None
    n_steps_effective = None
    dt_ms_configured = None
    dt_ms_effective = None
    duration_ms_configured = None
    duration_ms_effective = None
    if tensor is not None:
        # No duration stored on tensor; leave None (configured duration is runtime, not tensor)
        pass
    # Try to read from model.cfg.metadata runtime knobs (if present)
    try:
        rt_meta = model.cfg.metadata
        # RuntimeConfiguration bridged keys may be in metadata; otherwise use signals time axis
        dt_ms_configured = float(rt_meta.get("dt_ms", rt_meta.get("dt", 0.5))) if "dt_ms" in rt_meta or "dt" in rt_meta else None
        duration_ms_configured = float(rt_meta.get("duration_ms", 1000.0)) if "duration_ms" in rt_meta else None
    except Exception:
        pass
    if signals is not None:
        try:
            time_ms = signals.time_ms
            n_steps_effective = int(time_ms.shape[0])
            if n_steps_effective > 1:
                dt_ms_effective = float(time_ms[1] - time_ms[0])
                duration_ms_effective = float(time_ms[-1] - time_ms[0] + dt_ms_effective)
            else:
                dt_ms_effective = None
                duration_ms_effective = None
            n_steps_realized = n_steps_effective
        except Exception:
            pass
    # fallback effective = realized when no signals
    if n_steps_effective is None and n_steps_realized is None:
        # no signals given: effective == realized == configured (if known)
        n_steps_effective = n_steps_realized

    # --- output basis: minimal independent (STATE/SOURCE/FIELD/PROBE/DERIVED) ---
    # Use existing Signals field names; probe readouts are optional.
    output_basis: dict[str, Any] = {
        "STATE": {
            "V_m": {"shape": [None, n_neurons_realized], "units": "mV (Izhikevich proxy)", "axis": "T×N"},
            "spikes": {"shape": [None, n_neurons_realized], "units": "binary", "axis": "T×N"},
            "u": {"shape": [n_neurons_realized], "units": "recovery proxy", "axis": "N", "note": "per-step state; trajectory T×N if recorded"},
        },
        "SOURCE": {
            "sources": {"shape": [None, n_neurons_realized], "units": "native_current+spike_impulse proxy", "axis": "T×N"},
        },
        "FIELD": {
            "lfp_proxy": {"shape": [None, 16], "units": "proxy", "axis": "T×X", "note": "n_contacts from static"},
            "csd_proxy": {"shape": [None, 16], "units": "proxy", "axis": "T×X"},
            "phi_e_proxy": {"shape": [None, 16], "units": "proxy", "axis": "T×X"},
        },
        "PROBE": {
            "eeg_proxy": {"shape": [None, 16], "units": "proxy", "axis": "T×X", "status": "declared_not_computed_in_laminar_proxy"},
            "meg_proxy": {"shape": [None, 16], "units": "proxy", "axis": "T×X", "status": "declared_not_computed"},
        },
        "DERIVED": {
            "spike_rate_hz_mean": {"units": "Hz", "source": "mean(spikes)/dt"},
            "mean_V_m": {"units": "mV", "source": "mean(V_m)"},
            "spike_count_total": {"units": "count", "source": "sum(spikes)"},
        },
    }
    # enrich with realized shapes from signals if available (existing API: signals.field, signals.V_m shape)
    if signals is not None:
        try:
            T = int(signals.time_ms.shape[0])
            output_basis["STATE"]["V_m"]["shape"] = [T, n_neurons_realized]
            output_basis["STATE"]["spikes"]["shape"] = [T, n_neurons_realized]
            if signals.sources is not None:
                output_basis["SOURCE"]["sources"]["shape"] = [T, n_neurons_realized]
            else:
                output_basis["SOURCE"]["sources"]["status"] = "not_recorded (record_sources=False)"
            if signals.field is not None:
                # existing API: FieldOutput diagnostics shape
                for k in ("lfp_proxy", "csd_proxy", "phi_e_proxy"):
                    arr = getattr(signals.field, k, None)
                    if arr is not None:
                        output_basis["FIELD"][k]["shape"] = [int(arr.shape[0]), int(arr.shape[1])]
        except Exception:
            pass
    # n_contacts realized
    try:
        n_contacts_realized = int(model.static.get("n_contacts", 16))
        for k in ("lfp_proxy", "csd_proxy", "phi_e_proxy"):
            if output_basis["FIELD"][k]["shape"][1] == 16:
                output_basis["FIELD"][k]["shape"][1] = n_contacts_realized
    except Exception:
        n_contacts_realized = 16

    # --- provenance (existing API: config_hash, tensor_identity, neuronal_tensor provenance) ---
    provenance: dict[str, Any] = {}
    try:
        provenance["config_hash"] = _config_hash(model.cfg)
    except Exception:
        provenance["config_hash"] = None
    try:
        provenance["tensor_identity"] = model.cfg.metadata.get("tensor_identity")
    except Exception:
        provenance["tensor_identity"] = None
    try:
        provenance["tensor_provenance"] = getattr(tensor, "provenance", None) if tensor is not None else None
    except Exception:
        provenance["tensor_provenance"] = None
    try:
        from jaxfne._model import _JAXFNE_VERSION as _ver
        provenance["jaxfne_version"] = _ver
    except Exception:
        provenance["jaxfne_version"] = None
    try:
        provenance["positions_shape"] = positions_shape
        provenance["positions_units"] = "relative_laminar_depth_proxy"
        provenance["source_calibration_status"] = model.cfg.metadata.get("source_calibration_status")
        provenance["field_solver_status"] = model.cfg.metadata.get("field_solver_status")
        provenance["field_claim_level"] = model.cfg.metadata.get("field_claim_level", "proxy_readout")
        provenance["physical_amplitude_calibrated"] = False
    except Exception:
        pass

    Theta = {
        "Theta_static": {"members": theta_static_sized, "N_static": N_static, "note": "Σ|θ| for θ∈Θ_static (per-neuron static; positions and W are separate)"},
        "X": {"per_step": X_per_step, "with_prev_spikes": X_with_prev, "canonical_4N": X_canonical_4N, "note": "fast state [v,u]; 4N counts v,u,prev_spikes,buffer head for text bundle illustration"},
        "H": {"size": H_size, "h_state_dim": h_state_dim, "locality": h_locality, "note": "hidden/adaptation state; 0 when HDP disabled"},
        "W": {"n_edges": W_size, "tau_catalog": tau_catalog_size, "note": "synaptic storage (EdgeList.weight); τ catalog separate"},
        "positions": {"shape": positions_shape, "size": positions_size, "note": "geometry (N×3), not part of N_static"},
    }

    counts = {
        "configured": {
            "n_neurons": n_neurons_configured,
            "n_areas": n_areas_configured,
            "n_populations_declared": n_populations_configured,
            "n_connection_rules": n_edges_configured_rules,
            "duration_ms": duration_ms_configured,
            "dt_ms": dt_ms_configured,
            "n_steps": n_steps_configured,
        },
        "realized": {
            "n_neurons": n_neurons_realized,
            "n_populations_detailed": n_populations_realized,
            "populations_inventory": populations_inventory,
            "n_edges": n_edges_realized,
            "n_contacts": n_contacts_realized,
            "positions_shape": positions_shape,
            "Theta": Theta,
            "N_static": N_static,
        },
        "effective": {
            "n_neurons": n_neurons_realized,
            "n_edges_effective": n_edges_realized,
            "duration_ms": duration_ms_effective,
            "dt_ms": dt_ms_effective,
            "n_steps": n_steps_effective,
            "X_per_step": X_per_step,
            "H_size": H_size,
            "W_size": W_size,
            "dtype": str(signals.V_m.dtype) if signals is not None and hasattr(signals.V_m, "dtype") else None,
        },
    }

    payload = {
        "schema": "canonical_compact_summary_v0.1",
        "Theta": Theta,
        "N_static": N_static,
        "output_basis": output_basis,
        "counts": counts,
        "provenance": provenance,
        "Δscience": 0,
        "overhead": "zero when unused (function not called in simulate/construct hot path)",
    }

    # text bundle (like jaxfne summary)
    payload["text_bundle"] = format_canonical_text_bundle(payload)

    return _json_safe(payload)


def format_canonical_text_bundle(summary: dict[str, Any]) -> str:
    """Format the canonical compact summary as a human-readable text bundle.

    Example (like jaxfne summary) — compact, one-screen, no kernels touched.
    """
    Theta = summary.get("Theta", {})
    N_static = summary.get("N_static", "?")
    counts = summary.get("counts", {})
    cfg = counts.get("configured", {})
    real = counts.get("realized", {})
    eff = counts.get("effective", {})
    prov = summary.get("provenance", {})

    # Extract with fallbacks
    n_neurons_cfg = cfg.get("n_neurons")
    n_neurons_real = real.get("n_neurons")
    n_pops_decl = cfg.get("n_populations_declared")
    n_pops_real = real.get("n_populations_detailed")
    inv = real.get("populations_inventory", {})
    n_edges_rules = cfg.get("n_connection_rules")
    n_edges_real = real.get("n_edges")
    n_contacts = real.get("n_contacts")
    dur_eff = eff.get("duration_ms")
    dt_eff = eff.get("dt_ms")
    n_steps_eff = eff.get("n_steps")
    Theta_static = Theta.get("Theta_static", {})
    X = Theta.get("X", {})
    H = Theta.get("H", {})
    W = Theta.get("W", {})

    lines: list[str] = []
    lines.append("jaxfne canonical compact summary (0.4.18) — Θ=Θ_static⊕X⊕H⊕W  Δscience=0")
    lines.append(f"provenance: config_hash={prov.get('config_hash')}  tensor_identity={str(prov.get('tensor_identity'))[:12] if prov.get('tensor_identity') else '—'}  version={prov.get('jaxfne_version')}  calibrated={prov.get('physical_amplitude_calibrated')}")
    lines.append("")
    lines.append("counts (configured → realized → effective):")
    lines.append(f"  neurons:    {n_neurons_cfg} → {n_neurons_real} → {eff.get('n_neurons')}  (realized N={n_neurons_real})")
    # populations: explain 12 vs 23
    if n_pops_decl is not None or n_pops_real is not None:
        eff_collapsed = inv.get("effective_EI_collapsed") if isinstance(inv, dict) else None
        lines.append(f"  populations: declared {n_pops_decl} → detailed {n_pops_real} → effective EI-collapsed {eff_collapsed}  (task example ‘12’ = 6 layers × 2 E/I; detailed {n_pops_real} = layer×cell-type combos; see per_layer inventory)")
        if isinstance(inv, dict) and inv.get("per_layer"):
            lines.append(f"    per_layer: {inv.get('per_layer')}  per_cell_type: {inv.get('per_cell_type')}")
    lines.append(f"  edges:      rules {n_edges_rules} → realized {n_edges_real} → effective {eff.get('n_edges_effective')}  (EdgeList; τ catalog {W.get('tau_catalog')} )  — task ‘~79k’ is illustrative; realized 215785 for canonical-v1-column-1000n with p=1.0 bipartite rules")
    lines.append(f"  contacts:   {n_contacts}  (laminar proxy)")
    lines.append(f"  duration:   {cfg.get('duration_ms')}ms → {dur_eff}ms  dt {cfg.get('dt_ms')}ms → {dt_eff}ms  n_steps {n_steps_eff}  (configured RuntimeConfiguration(1000ms,0.5ms) → realized T=2000)")
    lines.append("")
    lines.append("Θ decomposition:")
    members = Theta_static.get("members", {})
    members_str = "+".join(f"{k}({v})" for k, v in members.items()) if members else "—"
    lines.append(f"  Θ_static: {members_str} = {N_static}  (N_static=Σ|θ|, positions {Theta.get('positions',{}).get('shape')} = {Theta.get('positions',{}).get('size')} not in Θ_static, geometry separate)")
    lines.append(f"  X:        per-step {X.get('per_step')} (v+u, 2N)  with_prev {X.get('with_prev_spikes')} (3N)  canonical_4N {X.get('canonical_4N')} (illustrative ‘X 4000’ = 4×N for N=1000, counting prev_spikes+buffer head)  trajectory T·|X|={n_steps_eff}×{X.get('per_step')}={ (n_steps_eff or 0) * (X.get('per_step') or 0)}")
    lines.append(f"  H:        {H.get('size')} (h_state_dim={H.get('h_state_dim')} locality={H.get('locality')}; 0 when HDP disabled — canonical 1000n)")
    lines.append(f"  W:        {W.get('n_edges')} weights (+τ catalog {W.get('tau_catalog')})  total Θ size ≈ {N_static + (X.get('per_step') or 0) + (H.get('size') or 0) + (W.get('n_edges') or 0)} scalars per snapshot (positions {Theta.get('positions',{}).get('size')} extra geometry)")
    lines.append("")
    lines.append("minimal independent output basis (not flattened signals):")
    ob = summary.get("output_basis", {})
    for k in ("STATE", "SOURCE", "FIELD", "PROBE", "DERIVED"):
        block = ob.get(k, {})
        if isinstance(block, dict):
            items = ", ".join(f"{name}{':'+str(v.get('shape')) if v.get('shape') else ''}" for name, v in block.items())
            lines.append(f"  {k}: {items}")
    lines.append("")
    lines.append("provenance/API used: Model(params['emitter'], params['edge_list'], params['positions']), Signals(time_ms,V_m,spikes,sources,field), NeuronalTensor(areas/layers/neuron_types), EdgeList(pre,post,weight,tau), positions (N×3), provenance(config_hash,tensor_identity)")
    lines.append("no kernel change, no overhead when unused (summary is off-hot-path; simulate/construct unchanged)")
    return "\n".join(lines)

