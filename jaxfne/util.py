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

    - ``taxonomy``: universal six-class split — Fixed parameters (constant),
      Dynamic state (v,u), History state (delay/continuation buffers),
      Mutable parameters (coordinates permitted to change: EdgeList.weight),
      Trainable/free parameters (optimizer-exposed subset), Recorded outputs
      (X→Q→Φ→Y trajectory). Stored ≠ plastic (permitted) ≠ free ≠ optimizer.
    - ``Theta``: legacy decomposition ``Θ_static``, ``X``, ``H``, ``W`` with sizes
      (kept for backward compat; maps onto taxonomy).
    - ``N_static``: Σ|θ| for θ∈fixed per-neuron set.
    - ``N_static_counterexamples``: 4 concrete mis-counts showing why
      stored/plastic/free/optimizer cannot be substituted for N_static.
    - ``output_basis``: typed observation graph ``X→Q→Φ→Y`` (not minimal
      independent; not flattened signals). Each node carries
      ``depends_on``, ``shape``, ``recorded``, ``calibrated``; roles
      X=STATE, Q=SOURCE, Φ=FIELD, Y=PROBE/DERIVED.
    - ``counts``: configured vs realized vs executed (neurons, populations,
      edges, duration, dt, n_steps, state sizes); ``effective`` is reserved
      for causal evidence ``ΔX`` under intervention, not runtime.
    - ``provenance``: config_hash, tensor_identity, versions.
    - ``text_bundle``: human-readable example (like jaxfne summary).
      ``depends_on``, ``shape``, ``recorded``, ``calibrated``; roles
      X=STATE, Q=SOURCE, Φ=FIELD, Y=PROBE/DERIVED.
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
    - ``Θ_static`` = per-neuron static (fixed) parameters present on the
      emitter (e.g. {a,b,c,d,drive,sign,source_scale} for Izhikevich).
      Generic rule: ``N_static = Σ|θ_i|`` over actual fixed arrays
      (sum of sizes, not a fixed formula). Positions (N×3) and edge
      weights are NOT in Θ_static (they are geometry and W respectively);
      source_scale contributes its realized size (scalar→1, per-neuron→N).
    - ``X`` = fast dynamical state [v, u] per neuron (2·N per step; 4·N if
      counting prev_spikes + syn_state buffer head). Trajectory size = T·|X|.
    - ``H`` = Relative Biophysical State (RBS). Exists independently of HDP;
      HDP (``dot W = F_W(H)``) is only one possible parameter-dynamics map
      involving H. RBD (e.g. Protocol H1, ``dot W=0``) demonstrates H
      without plasticity. ``|H|`` is derived from actual state/schema/traces
      (``N × h_state_dim`` or per-population), not solely from ``hdp_params``.
      Summary reports ``H_status`` in {absent, present_static, present_dynamic, unknown}.
    - ``W`` = synaptic storage (EdgeList). ``|W| = n_edges`` (weights) plus
      ``tau_ms`` catalog (kept separate in diagnostics).
    - Observation graph is typed ``X→Q→Φ→Y`` (deterministic): ``X``=STATE (V_m, spikes, u; depends_on=[]), ``Q``=SOURCE
      (sources; depends_on=[X]), ``Φ``=FIELD (lfp/csd/phi_e proxy on contacts;
      depends_on=[Q]), ``Y``=PROBE/DERIVED (eeg/meg/emm + scalar metrics;
      depends_on=[Φ] transitively X→Q→Φ→Y). Each node/member reports
      ``depends_on``, ``shape``, ``recorded``, ``calibrated``
      (``calibrated=False`` while ``physical_amplitude_calibrated=False``);
      this distinguishes state/source/field/probe/derived correctly and avoids
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
    # Per-step X = v(N) + u(N) = 2N; with prev_spikes buffer 3N; full buffer
    # 4N counts v,u,prev_spikes,buffer head (generic: 4·N derived from realized N).
    v_size = int(jnp.asarray(emitter.v0).size)
    u_size = int(jnp.asarray(emitter.u0).size)
    X_per_step = v_size + u_size  # 2N
    X_with_prev = X_per_step + n_neurons_realized  # 3N
    X_canonical_4N = 4 * n_neurons_realized  # generic 4·N (illustrative for text bundle)

    # H: Relative Biophysical State (RBS) — exists independently of HDP.
    # HDP (dot W = F_W(H)) is only one possible dynamics map involving H;
    # RBD (e.g. Protocol H1, dot W=0) demonstrates H without plasticity.
    # Do NOT infer H solely from hdp_params. Detect actual H from
    # state/schema/traces in priority order:
    #   1) model.params['hdp_initial_H'] (actual per-neuron initial RBS),
    #   2) tensor PlasticParams.H schema,
    #   3) signals traces/diagnostics (H_trace/H_final),
    #   4) hdp_params only as fallback for dim/locality when H already evidenced
    #      or to detect present_dynamic via enable_hdp dynamics.
    # Report H_status in {absent, present_static, present_dynamic, unknown}.
    H_size = 0
    h_state_dim = 0
    h_locality = None
    H_status: str = "unknown"
    H_evidence: list[str] = []
    H_dynamic = False
    hdp_params_probe: Any = None
    try:
        if "hdp_params" in model.cfg.metadata:
            hdp_params_probe = model.cfg.metadata.get("hdp_params")
    except Exception:
        hdp_params_probe = None
    enable_hdp = False
    try:
        enable_hdp = bool(model.cfg.metadata.get("enable_hdp"))
        if enable_hdp:
            H_dynamic = True
            H_evidence.append("model.cfg.metadata enable_hdp (dynamics)")
    except Exception:
        enable_hdp = False
    try:
        hdp_H = model.params.get("hdp_initial_H")
        if hdp_H is not None:
            arr = jnp.asarray(hdp_H)
            if arr.ndim == 2:
                h_state_dim = int(arr.shape[1])
                H_size = int(arr.size)
                if H_size != n_neurons_realized * h_state_dim:
                    try:
                        rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                        pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 0
                        if pops and H_size == h_state_dim * pops:
                            h_locality = "population"
                        elif h_state_dim == H_size and h_state_dim > 0:
                            h_locality = "population"
                        else:
                            if isinstance(hdp_params_probe, dict) and hdp_params_probe.get("h_state_locality") == "population":
                                h_locality = "population"
                            else:
                                h_locality = "node"
                    except Exception:
                        h_locality = "node"
                else:
                    h_locality = "node"
            elif arr.ndim == 1:
                h_state_dim = 1
                H_size = int(arr.size)
                if isinstance(hdp_params_probe, dict) and hdp_params_probe.get("h_state_locality") == "population":
                    try:
                        rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                        pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 0
                        if pops and H_size == int(hdp_params_probe.get("h_state_dim", 1)) * pops:
                            h_locality = "population"
                        elif H_size != n_neurons_realized and H_size <= 4:
                            h_locality = "population"
                        else:
                            h_locality = "node"
                    except Exception:
                        h_locality = "node"
                else:
                    h_locality = "node"
            else:
                H_size = int(arr.size)
                h_state_dim = int(H_size // max(n_neurons_realized, 1)) if n_neurons_realized else 0
                h_locality = "node"
            H_evidence.append("model.params['hdp_initial_H']")
            H_status = "present_dynamic" if H_dynamic else "present_static"
        if hdp_H is not None and isinstance(hdp_params_probe, dict):
            try:
                if h_locality is None:
                    h_locality = hdp_params_probe.get("h_state_locality")
                if h_state_dim == 0 and hdp_params_probe.get("h_state_dim"):
                    h_state_dim = int(hdp_params_probe.get("h_state_dim", 0))
            except Exception:
                pass
    except Exception:
        pass
    if H_size == 0 and tensor is not None:
        try:
            vals: list[float] = []
            for area in tensor.areas:
                for ic in area.inter_connections:
                    try:
                        vals.append(float(ic.plastic.H))
                    except Exception:
                        pass
            for ac in getattr(tensor, "area_connections", []):
                try:
                    vals.append(float(ac.plastic.H))
                except Exception:
                    pass
            if vals:
                H_evidence.append("tensor PlasticParams.H schema")
                if h_state_dim == 0:
                    if isinstance(hdp_params_probe, dict) and hdp_params_probe.get("h_state_dim"):
                        h_state_dim = int(hdp_params_probe.get("h_state_dim", 1))
                    else:
                        h_state_dim = 1
                if h_locality is None:
                    if isinstance(hdp_params_probe, dict) and hdp_params_probe.get("h_state_locality"):
                        h_locality = hdp_params_probe.get("h_state_locality")
                    else:
                        h_locality = "node"
                if h_locality == "population":
                    try:
                        rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                        pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 1
                        H_size = int(h_state_dim * pops)
                    except Exception:
                        H_size = int(h_state_dim)
                else:
                    H_size = int(h_state_dim * n_neurons_realized)
                if H_status == "unknown":
                    H_status = "present_dynamic" if H_dynamic else "present_static"
        except Exception:
            pass
    if signals is not None and isinstance(getattr(signals, "metadata", None), dict):
        try:
            hdp_meta = signals.metadata.get("hdp") or signals.metadata.get("hdp_params")
            if isinstance(hdp_meta, dict) and H_size == 0:
                H_trace = hdp_meta.get("H_trace") if "H_trace" in hdp_meta else hdp_meta.get("H_final")
                if H_trace is not None:
                    try:
                        arr = jnp.asarray(H_trace)
                        if arr.ndim == 3:
                            h_state_dim = int(arr.shape[2])
                            h_locality = "node"
                            H_size = int(arr.shape[1] * h_state_dim)
                        elif arr.ndim == 2:
                            if arr.shape[1] == n_neurons_realized:
                                h_state_dim = 1
                                h_locality = "node"
                                H_size = int(n_neurons_realized)
                            else:
                                h_state_dim = int(arr.shape[1])
                                h_locality = "population"
                                try:
                                    rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                                    pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 1
                                    H_size = int(h_state_dim * pops) if pops else int(h_state_dim)
                                except Exception:
                                    H_size = int(h_state_dim)
                        elif arr.ndim == 1:
                            h_state_dim = 1
                            H_size = int(arr.size)
                            h_locality = "node"
                        H_evidence.append("signals.metadata['hdp'].H_trace/H_final")
                        if H_status == "unknown":
                            H_status = "present_dynamic"
                        else:
                            H_dynamic = True
                            if H_status == "present_static":
                                H_status = "present_dynamic"
                    except Exception:
                        pass
                if isinstance(hdp_meta, dict) and hdp_meta.get("enabled"):
                    H_dynamic = True
                    if "signals.metadata hdp enabled" not in H_evidence:
                        H_evidence.append("signals.metadata hdp enabled")
                    if H_status in ("present_static", "unknown"):
                        H_status = "present_dynamic"
        except Exception:
            pass
    if H_size == 0:
        try:
            diag = getattr(model, "_last_hdp_diag", None)
            if isinstance(diag, dict) and ("H_trace" in diag or "H_final" in diag):
                H_trace = diag.get("H_trace") if "H_trace" in diag else diag.get("H_final")
                if H_trace is not None:
                    arr = jnp.asarray(H_trace)
                    if arr.ndim == 3:
                        h_state_dim = int(arr.shape[2])
                        H_size = int(arr.shape[1] * h_state_dim) if arr.ndim == 3 else int(arr.size)
                        h_locality = "node"
                    elif arr.ndim == 2:
                        if arr.shape[1] == n_neurons_realized:
                            h_state_dim = 1
                            H_size = int(n_neurons_realized)
                            h_locality = "node"
                        else:
                            h_state_dim = int(arr.shape[1])
                            h_locality = "population"
                            try:
                                rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                                pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 1
                                H_size = int(h_state_dim * pops) if pops else int(h_state_dim)
                            except Exception:
                                H_size = int(h_state_dim)
                    else:
                        H_size = int(arr.size)
                        h_state_dim = 1
                        h_locality = "node"
                    H_evidence.append("model._last_hdp_diag H_trace/H_final")
                    if H_status == "unknown":
                        H_status = "present_dynamic"
                    else:
                        H_dynamic = True
                        if H_status == "present_static":
                            H_status = "present_dynamic"
        except Exception:
            pass
    if H_size == 0 and enable_hdp and isinstance(hdp_params_probe, dict):
        try:
            h_state_dim = int(hdp_params_probe.get("h_state_dim", 1))
            h_locality = hdp_params_probe.get("h_state_locality")
            if h_state_dim and h_state_dim > 0:
                if h_locality == "population":
                    rows = model.neuron_table() if callable(getattr(model, "neuron_table", None)) else []
                    pops = len({(r.get("layer"), r.get("cell_type")) for r in rows}) if rows else 1
                    H_size = int(h_state_dim * pops)
                    if not h_locality:
                        h_locality = "population"
                else:
                    H_size = int(h_state_dim * n_neurons_realized)
                    if not h_locality:
                        h_locality = "node"
                H_evidence.append("hdp_params fallback for size (HDP enabled, H dynamics)")
                H_status = "present_dynamic"
                H_dynamic = True
        except Exception:
            pass
    if H_status == "unknown":
        if H_size == 0:
            H_status = "absent"
            H_evidence.append("no H state/schema/trace found")
            h_locality = None
            h_state_dim = 0
        else:
            H_status = "present_dynamic" if H_dynamic else "present_static"
    if H_status == "absent":
        H_size = 0
        h_state_dim = 0
        h_locality = None
    if H_size > 0 and h_locality is None:
        h_locality = "node"
        if h_state_dim == 0:
            h_state_dim = int(H_size // max(n_neurons_realized, 1)) if n_neurons_realized else 1

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
            # declared populations = total NeuronType entries (derived from tensor)
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
            "EI_collapsed": int(len(eff_combos)),
            "effective_EI_collapsed": int(len(eff_combos)),
            "per_layer": dict(_Counter(r.get("layer") for r in rows)),
            "per_cell_type": dict(_Counter(r.get("cell_type") for r in rows)),
        }
    except Exception:
        pass

    # effective: after simulate, may differ due to runtime overrides (dt rounding, HDP enable, dtype)
    n_steps_configured = None
    n_steps_realized = None
    n_steps_executed = None
    dt_ms_configured = None
    dt_ms_executed = None
    duration_ms_configured = None
    duration_ms_executed = None
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
            n_steps_executed = int(time_ms.shape[0])
            if n_steps_executed > 1:
                dt_ms_executed = float(time_ms[1] - time_ms[0])
                duration_ms_executed = float(time_ms[-1] - time_ms[0] + dt_ms_executed)
            else:
                dt_ms_executed = None
                duration_ms_executed = None
            n_steps_realized = n_steps_executed
        except Exception:
            pass
    # fallback executed = realized when no signals (effective=causal ΔX)
    if n_steps_executed is None and n_steps_realized is None:
        # no signals given: executed == realized == configured (if known)
        n_steps_executed = n_steps_realized

    # --- output basis: typed observation graph X→Q→Φ→Y (dependent chain, not flattened) ---
    # Deterministic structure: X (STATE) -> Q (SOURCE) -> Φ (FIELD) -> Y (PROBE/DERIVED)
    # Each node/member carries depends_on, shape, recorded, calibrated.
    # Roles: X=STATE (V_m, spikes, u), Q=SOURCE (sources), Φ=FIELD (lfp/csd/phi_e), Y=PROBE/DERIVED
    try:
        n_contacts_realized = int(model.static.get("n_contacts", 16))
    except Exception:
        n_contacts_realized = 16
    # realized time steps
    T_eff: Any = None
    if signals is not None:
        try:
            T_eff = int(signals.time_ms.shape[0])
        except Exception:
            T_eff = None
    # recorded flags (existing API: signals.field, signals.sources)
    sources_recorded = bool(signals is not None and getattr(signals, "sources", None) is not None)
    field_recorded: dict[str, bool] = {}
    if signals is not None and getattr(signals, "field", None) is not None:
        for _k in ("lfp_proxy", "csd_proxy", "phi_e_proxy"):
            try:
                field_recorded[_k] = getattr(signals.field, _k, None) is not None
            except Exception:
                field_recorded[_k] = False
    else:
        for _k in ("lfp_proxy", "csd_proxy", "phi_e_proxy"):
            field_recorded[_k] = False
    # shapes
    _shape_TN = [T_eff, n_neurons_realized] if T_eff is not None else [None, n_neurons_realized]
    _shape_TN_u = [n_neurons_realized]
    _shape_TX = [T_eff, n_contacts_realized] if T_eff is not None else [None, n_contacts_realized]
    _cal = False
    _state_members: dict[str, Any] = {
        "V_m": {"shape": list(_shape_TN), "units": "mV (Izhikevich proxy)", "axis": "T×N", "depends_on": [], "recorded": T_eff is not None, "calibrated": _cal},
        "spikes": {"shape": list(_shape_TN), "units": "binary", "axis": "T×N", "depends_on": [], "recorded": T_eff is not None, "calibrated": _cal},
        "u": {"shape": list(_shape_TN_u), "units": "recovery proxy", "axis": "N", "note": "per-step state; trajectory T×N if recorded", "depends_on": [], "recorded": T_eff is not None, "calibrated": _cal},
    }
    _source_members: dict[str, Any] = {
        "sources": {"shape": list(_shape_TN), "units": "native_current+spike_impulse proxy", "axis": "T×N", "depends_on": ["X"], "recorded": sources_recorded, "calibrated": _cal, **({"status": "not_recorded (record_sources=False)"} if not sources_recorded and T_eff is not None else {})},
    }
    _field_members: dict[str, Any] = {
        "lfp_proxy": {"shape": list(_shape_TX), "units": "proxy", "axis": "T×X", "note": "n_contacts from static", "depends_on": ["Q"], "recorded": field_recorded["lfp_proxy"], "calibrated": _cal},
        "csd_proxy": {"shape": list(_shape_TX), "units": "proxy", "axis": "T×X", "depends_on": ["Q"], "recorded": field_recorded["csd_proxy"], "calibrated": _cal},
        "phi_e_proxy": {"shape": list(_shape_TX), "units": "proxy", "axis": "T×X", "depends_on": ["Q"], "recorded": field_recorded["phi_e_proxy"], "calibrated": _cal},
    }
    _probe_members: dict[str, Any] = {
        "eeg_proxy": {"shape": list(_shape_TX), "units": "proxy", "axis": "T×X", "status": "declared_not_computed_in_laminar_proxy", "depends_on": ["Phi"], "recorded": False, "calibrated": _cal},
        "meg_proxy": {"shape": list(_shape_TX), "units": "proxy", "axis": "T×X", "status": "declared_not_computed", "depends_on": ["Phi"], "recorded": False, "calibrated": _cal},
    }
    _derived_members: dict[str, Any] = {
        "spike_rate_hz_mean": {"units": "Hz", "source": "mean(spikes)/dt", "depends_on": ["X"], "shape": [], "recorded": T_eff is not None, "calibrated": _cal},
        "mean_V_m": {"units": "mV", "source": "mean(V_m)", "depends_on": ["X"], "shape": [], "recorded": T_eff is not None, "calibrated": _cal},
        "spike_count_total": {"units": "count", "source": "sum(spikes)", "depends_on": ["X"], "shape": [], "recorded": T_eff is not None, "calibrated": _cal},
    }
    if signals is not None and field_recorded:
        try:
            for _k in ("lfp_proxy", "csd_proxy", "phi_e_proxy"):
                if field_recorded[_k]:
                    arr = getattr(signals.field, _k, None)
                    if arr is not None:
                        _field_members[_k]["shape"] = [int(arr.shape[0]), int(arr.shape[1])]
        except Exception:
            pass
    X_node: dict[str, Any] = {"role": "STATE", "depends_on": [], "shape": list(_shape_TN), "recorded": T_eff is not None, "calibrated": _cal, "members": _state_members}
    Q_node: dict[str, Any] = {"role": "SOURCE", "depends_on": ["X"], "shape": list(_shape_TN), "recorded": sources_recorded, "calibrated": _cal, "members": _source_members}
    Phi_node: dict[str, Any] = {"role": "FIELD", "depends_on": ["Q"], "shape": list(_shape_TX), "recorded": any(field_recorded.values()), "calibrated": _cal, "members": _field_members}
    Y_node: dict[str, Any] = {"role": "PROBE/DERIVED", "depends_on": ["Phi"], "shape": list(_shape_TX), "recorded": any(field_recorded.values()), "calibrated": _cal, "members": {**_probe_members, **_derived_members}, "probe": _probe_members, "derived": _derived_members}
    output_basis: dict[str, Any] = {
        "structure": "X->Q->Phi->Y",
        "deterministic_structure": "X->Q->Phi->Y",
        "deterministic": True,
        "note": "typed observation graph X->Q->Phi->Y; Phi depends_on Q, Q depends_on X, Y depends_on Phi (dependent, not an independent basis); each node/member has depends_on, shape, recorded, calibrated; calibrated=False in proxy regime",
        "nodes": {"X": X_node, "Q": Q_node, "Phi": Phi_node, "Y": Y_node},
        "X": X_node,
        "Q": Q_node,
        "Phi": Phi_node,
        "Y": Y_node,
        "STATE": _state_members,
        "SOURCE": _source_members,
        "FIELD": _field_members,
        "PROBE": _probe_members,
        "DERIVED": _derived_members,
    }

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

    # --- universal taxonomy: Fixed / Dynamic / History / Mutable / Trainable / Recorded ---
    _history_syn_state = int(W_size)
    _history_prev_spikes = int(n_neurons_realized)
    _history_delay_max = 0
    _history_delay_buffer = 0
    try:
        import numpy as _np_hist
        if edge_list is not None and getattr(edge_list, "delay_steps", None) is not None:
            _dh = _np_hist.asarray(edge_list.delay_steps)
            if _dh.size:
                _history_delay_max = int(_np_hist.max(_dh))
                _history_delay_buffer = int((_history_delay_max + 1) * n_neurons_realized) if _history_delay_max > 0 else 0
    except Exception:
        _history_delay_max = 0
        _history_delay_buffer = 0
    _history_total = int(_history_syn_state + _history_prev_spikes + _history_delay_buffer)
    _n_trainable = 0
    _trainable_note = "0 unless EdgeParameterSpec/MatrixParameterSpec declared (optimizer-exposed subset of mutable weight coordinates); stored ≠ free ≠ optimizer"
    try:
        _opt_meta = model.cfg.metadata.get("optimizer") if isinstance(getattr(model.cfg, "metadata", None), dict) else None
        if isinstance(_opt_meta, dict) and _opt_meta.get("n_trainable"):
            _n_trainable = int(_opt_meta.get("n_trainable", 0))
    except Exception:
        pass
    N_static_counterexamples = [
        {
            "misconception": "N_static includes stored EdgeList.weight (n_edges)",
            "wrong": int(N_static + W_size),
            "correct": int(N_static),
            "explanation": f"weights are mutable storage (|W|=n_edges={W_size}), not fixed per-neuron params; counting them inflates {N_static}→{N_static + W_size} (N_static=Σ|θ_i|)",
            "class_confused": "stored weights (mutable) vs fixed parameters",
        },
        {
            "misconception": "N_static includes geometry positions (N×3)",
            "wrong": int(N_static + positions_size),
            "correct": int(N_static),
            "explanation": f"positions are fixed geometry {positions_shape} size {positions_size}, not per-neuron params; counted separately as geometry (N_static=Σ|θ_i|)",
            "class_confused": "geometry vs fixed parameters",
        },
        {
            "misconception": "N_static equals number of mutable coordinates (n_edges) or plastic rule count",
            "wrong_rules": int(n_edges_configured_rules) if n_edges_configured_rules is not None else 0,
            "wrong_edges": int(W_size),
            "correct": int(N_static),
            "explanation": f"plastic rule count ({n_edges_configured_rules} declared inter/area connections) and realized mutable storage ({W_size} edges) are W, not Θ_static; N_static={N_static}=Σ|θ_i|",
            "class_confused": "plastic declaration (rule) / mutable storage (edge) vs fixed",
        },
        {
            "misconception": "N_static equals trainable/free optimizer coordinates",
            "wrong": int(_n_trainable),
            "correct": int(N_static),
            "explanation": "trainable/free is optimizer-exposed subset of mutable (0 when no EdgeParameterSpec/MatrixParameterSpec); N_static is fixed count, independent of optimizer declaration",
            "class_confused": "optimizer/free (trainable subset) vs fixed",
        },
        {
            "misconception": "N_static includes history buffers (syn_state+prev_spikes+B_t)",
            "wrong": int(N_static + _history_total),
            "correct": int(N_static),
            "explanation": f"history state (syn_state {W_size}+prev_spikes {n_neurons_realized}+B_t {_history_delay_buffer}={_history_total}) is carry, not fixed parameter",
            "class_confused": "history buffers vs fixed",
        },
    ]
    taxonomy: dict[str, Any] = {
        "fixed_parameters": {
            "members": dict(theta_static_sized),
            "N_static": int(N_static),
            "tau_catalog": int(tau_catalog_size),
            "positions_excluded": {"shape": positions_shape, "size": int(positions_size)},
            "note": "constant per-neuron params (e.g. {a,b,c,d,drive,sign}+source_scale for Izhikevich); tau/receptor catalog and pre/post structure are fixed discrete; positions are fixed geometry kept separate; N_static=Σ|θ_i| generic (sum of realized fixed-array sizes)",
            "counterexamples": "see N_static_counterexamples",
        },
        "dynamic_state": {
            "v": int(v_size),
            "u": int(u_size),
            "per_step": int(X_per_step),
            "with_prev_spikes": int(X_with_prev),
            "canonical_4N": int(X_canonical_4N),
            "note": "fast ODE state [v,u] (2·N per step); trajectory T·|X|; H is separate hidden state, not part of X",
        },
        "history_state": {
            "syn_state": int(_history_syn_state),
            "prev_spikes": int(_history_prev_spikes),
            "delay_max_steps": int(_history_delay_max),
            "delay_buffer_Bt": int(_history_delay_buffer),
            "continuation_prng_key": "scalar (not counted in element count)",
            "continuation_step_offset": "scalar",
            "total_buffer_elements": int(_history_total),
            "note": "delay/continuation buffers carried between steps; O(E)+O(N)+O((Dmax+1)·N); distinct from dynamic X and mutable W",
        },
        "mutable_parameters": {
            "weight": int(W_size),
            "total_mutable": int(W_size),
            "tau_catalog_mutable": False,
            "positions_mutable": False,
            "note": "coordinates permitted to change: EdgeList.weight (n_edges) may evolve via HDP dW/dt=F_W(H) or optimizer; unchanged when HDP disabled and no tune, but *permitted*",
        },
        "trainable_parameters": {
            "n_trainable": int(_n_trainable),
            "n_free": int(_n_trainable),
            "note": _trainable_note,
            "declaration_api": "EdgeParameterSpec / MatrixParameterSpec (trainable=True)",
            "relation_to_mutable": f"subset of mutable (≤{W_size}); stored ({W_size}) ≠ plastic (permitted) ≠ free/trainable ({_n_trainable})",
        },
        "recorded_outputs": {
            "structure": "X->Q->Phi->Y",
            "X_state": ["V_m", "spikes", "u"],
            "Q_source": ["sources"],
            "Phi_field": ["lfp_proxy", "csd_proxy", "phi_e_proxy"],
            "Y_probe_derived": ["eeg_proxy", "meg_proxy", "spike_rate_hz_mean", "mean_V_m", "spike_count_total"],
            "note": "saved trajectory / observation graph; recorded outputs are not state carried forward (|Y| is evidence, not dynamics)",
        },
        "hidden_state": {
            "H_size": int(H_size),
            "h_state_dim": int(h_state_dim),
            "locality": h_locality,
            "H_status": H_status,
            "evidence": list(H_evidence),
            "note": "RBS (Relative Biophysical State) — exists independently of HDP; HDP is only one map dot W=F_W(H) (cf. RBD dot W=0); absent=0, present_static=stored but not evolving, present_dynamic=evolving (RBS/RBD or HDP), unknown=indeterminate; distinct from plastic W and history",
        },
    }
    Theta = {
        "Theta_static": {"members": theta_static_sized, "N_static": N_static, "note": "Σ|θ| for θ∈Θ_static (per-neuron static; positions and W are separate)"},
        "X": {"per_step": X_per_step, "with_prev_spikes": X_with_prev, "canonical_4N": X_canonical_4N, "note": "fast state [v,u]; 4N counts v,u,prev_spikes,buffer head for text bundle illustration"},
        "H": {"size": H_size, "h_state_dim": h_state_dim, "locality": h_locality, "H_status": H_status, "evidence": list(H_evidence), "note": "RBS; H_status in {absent,present_static,present_dynamic,unknown}; size 0 when absent, else N*h_state_dim (or per-population); HDP only one possible F_W(H)"},
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
        "executed": {
            "n_neurons": n_neurons_realized,
            "n_edges_executed": n_edges_realized,
            "n_edges_effective": n_edges_realized,
            "duration_ms": duration_ms_executed,
            "dt_ms": dt_ms_executed,
            "n_steps": n_steps_executed,
            "X_per_step": X_per_step,
            "H_size": H_size,
            "H_status": H_status,
            "h_state_dim": int(h_state_dim),
            "locality": h_locality,
            "W_size": W_size,
            "dtype": str(signals.V_m.dtype) if signals is not None and hasattr(signals.V_m, "dtype") else None,
        },
    }

    # alias: counts["effective"] for one-release compat (effective reserved for causal ΔX, runtime is executed)
    counts["effective"] = counts["executed"]

    payload = {
        "schema": "canonical_compact_summary_v0.1",
        "Theta": Theta,
        "N_static": N_static,
        "N_static_counterexamples": N_static_counterexamples,
        "taxonomy": taxonomy,
        "output_basis": output_basis,
        "counts": counts,
        "provenance": provenance,
        "H_status": H_status,
        "H_evidence": list(H_evidence),
        "H": {"size": H_size, "h_state_dim": int(h_state_dim), "locality": h_locality, "H_status": H_status, "evidence": list(H_evidence)},
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
    eff = counts.get("executed", counts.get("effective", {}))
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
    lines.append("counts (configured → realized → executed):")
    lines.append(f"  neurons:    {n_neurons_cfg} → {n_neurons_real} → {eff.get('n_neurons')}  (realized N={n_neurons_real})")
    # populations: explain 12 vs 23
    if n_pops_decl is not None or n_pops_real is not None:
        eff_collapsed = inv.get("executed_EI_collapsed", inv.get("effective_EI_collapsed", inv.get("EI_collapsed"))) if isinstance(inv, dict) else None
        lines.append(f"  populations: declared {n_pops_decl} → detailed {n_pops_real} → executed EI-collapsed {eff_collapsed}  (task example ‘12’ = 6 layers × 2 E/I; detailed {n_pops_real} = layer×cell-type combos; see per_layer inventory)")
        if isinstance(inv, dict) and inv.get("per_layer"):
            lines.append(f"    per_layer: {inv.get('per_layer')}  per_cell_type: {inv.get('per_cell_type')}")
    n_edges_exec = eff.get('n_edges_executed', eff.get('n_edges_effective'))
    lines.append(f"  edges:      rules {n_edges_rules} → realized {n_edges_real} → executed {n_edges_exec}  (EdgeList; τ catalog {W.get('tau_catalog')} )")
    lines.append(f"  contacts:   {n_contacts}  (laminar proxy)")
    lines.append(f"  duration:   {cfg.get('duration_ms')}ms → {dur_eff}ms  dt {cfg.get('dt_ms')}ms → {dt_eff}ms  n_steps {n_steps_eff}")
    lines.append("")
    lines.append("Θ decomposition:")
    members = Theta_static.get("members", {})
    members_str = "+".join(f"{k}({v})" for k, v in members.items()) if members else "—"
    lines.append(f"  Θ_static: {members_str} = {N_static}  (N_static=Σ|θ_i| generic sum of realized fixed sizes, positions {Theta.get('positions',{}).get('shape')} = {Theta.get('positions',{}).get('size')} not in Θ_static, geometry separate)")
    lines.append(f"  X:        per-step {X.get('per_step')} (v+u, 2N)  with_prev {X.get('with_prev_spikes')} (3N)  canonical_4N {X.get('canonical_4N')} (4·N generic)  trajectory T·|X|={n_steps_eff}×{X.get('per_step')}={ (n_steps_eff or 0) * (X.get('per_step') or 0)}")
    H_status_txt = summary.get("H_status") or H.get("H_status") or summary.get("H", {}).get("H_status") or "unknown"
    H_ev = summary.get("H_evidence") or H.get("evidence") or []
    ev_str = f" evidence={','.join(H_ev[:2])}" if H_ev else ""
    lines.append(f"  H:        {H.get('size')} (h_state_dim={H.get('h_state_dim')} locality={H.get('locality')} H_status={H_status_txt}{ev_str}; RBS exists independently of HDP — HDP is only one F_W(H); 0 when absent)")
    lines.append(f"  W:        {W.get('n_edges')} weights (+τ catalog {W.get('tau_catalog')})  total Θ size ≈ {N_static + (X.get('per_step') or 0) + (H.get('size') or 0) + (W.get('n_edges') or 0)} scalars per snapshot (positions {Theta.get('positions',{}).get('size')} extra geometry)")
    tax = summary.get("taxonomy", {})
    if tax:
        lines.append("")
        lines.append("universal taxonomy — stored ≠ plastic ≠ free ≠ optimizer (Δscience=0):")
        fp = tax.get("fixed_parameters", {})
        ds = tax.get("dynamic_state", {})
        hs = tax.get("history_state", {})
        mp = tax.get("mutable_parameters", {})
        tp = tax.get("trainable_parameters", {})
        ro = tax.get("recorded_outputs", {})
        hidden = tax.get("hidden_state", {})
        lines.append(f"  Fixed (constant):       N_static={fp.get('N_static', N_static)}  members a,b,c,d,drive,sign,source_scale  tau_catalog={fp.get('tau_catalog')}  positions_excluded {Theta.get('positions',{}).get('shape')}={Theta.get('positions',{}).get('size')}")
        lines.append(f"  Dynamic state (v,u):    per_step {ds.get('per_step', X.get('per_step'))} (v+u, 2N)  canonical_4N {ds.get('canonical_4N', X.get('canonical_4N'))}")
        lines.append(f"  History state (B_t):    syn_state {hs.get('syn_state')} + prev_spikes {hs.get('prev_spikes')} + B_t ring {hs.get('delay_buffer_Bt')} (Dmax={hs.get('delay_max_steps')}) = total {hs.get('total_buffer_elements')}")
        h_status_hidden = hidden.get('H_status', H.get('H_status', H_status_txt))
        ev_hidden = ','.join((hidden.get('evidence') or H_ev)[:2]) if (hidden.get('evidence') or H_ev) else ''
        ev_hidden_str = f" evidence={ev_hidden}" if ev_hidden else ""
        lines.append(f"  Hidden state (H):       {hidden.get('H_size', H.get('size'))} (h_state_dim={hidden.get('h_state_dim', H.get('h_state_dim'))} locality={hidden.get('locality', H.get('locality'))} H_status={h_status_hidden}{ev_hidden_str}) — RBS exists without HDP (HDP only one F_W(H)); not plastic, not optimizer coord")
        lines.append(f"  Mutable (permitted):    weight array {mp.get('weight')} (EdgeList.weight) — stored weights permitted to change via HDP or optimizer; tau/positions not mutable")
        lines.append(f"  Trainable/free (opt):   n_trainable={tp.get('n_trainable')} (optimizer-exposed subset ≤ mutable; 0 unless EdgeParameterSpec/MatrixParameterSpec trainable=True) — free ≠ stored")
        lines.append(f"  Recorded outputs:       {ro.get('structure','X->Q->Phi->Y')}  X={ro.get('X_state')}  Q={ro.get('Q_source')}  Phi={ro.get('Phi_field')}  Y={ro.get('Y_probe_derived')}")
        ces = summary.get("N_static_counterexamples", [])
        if ces:
            lines.append("  N_static counterexamples (why stored/plastic/free ≠ N_static):")
            for ce in ces[:5]:
                lines.append(f"    - {ce.get('misconception')}: wrong={ce.get('wrong', ce.get('wrong_edges', ce.get('wrong_rules')))} correct={ce.get('correct')} — {ce.get('explanation','')[:90]}")
        lines.append(f"  rule: N_static=Σ|θ_i| fixed only (generic sum of fixed-array sizes); history/mutable/trainable/recorded are disjoint classes; stored={W.get('n_edges', '?')} ≠ plastic(=permitted) ≠ free/trainable(=declared) ≠ fixed(=N_static)")
    lines.append("")
    ob = summary.get("output_basis", {})
    struct = ob.get("structure") or ob.get("deterministic_structure") or "X->Q->Phi->Y"
    lines.append(f"typed observation graph {struct} (dependent, not flattened signals):")
    nodes = ob.get("nodes", {})
    if nodes:
        for nk in ("X", "Q", "Phi", "Y"):
            nd = nodes.get(nk, {})
            if isinstance(nd, dict):
                dep = nd.get("depends_on", [])
                shape = nd.get("shape", [])
                rec = nd.get("recorded", False)
                cal = nd.get("calibrated", False)
                members = nd.get("members", {})
                mitems = ", ".join(f"{name}{':'+str(v.get('shape')) if v.get('shape') else ''}" for name, v in members.items()) if isinstance(members, dict) else ""
                lines.append(f"  {nk} ({nd.get('role','')}): depends_on={dep} shape={shape} recorded={rec} calibrated={cal}  members: {mitems}")
    else:
        for k in ("STATE", "SOURCE", "FIELD", "PROBE", "DERIVED"):
            block = ob.get(k, {})
            if isinstance(block, dict):
                items = ", ".join(f"{name}{':'+str(v.get('shape')) if v.get('shape') else ''}" for name, v in block.items())
                lines.append(f"  {k}: {items}")
    for k in ("STATE", "SOURCE", "FIELD", "PROBE", "DERIVED"):
        block = ob.get(k, {})
        if isinstance(block, dict) and k not in (nodes or {}):
            if nodes:
                continue
            items = ", ".join(f"{name}{':'+str(v.get('shape')) if v.get('shape') else ''}" for name, v in block.items())
            lines.append(f"  {k}: {items}")
    if nodes:
        for k in ("STATE", "SOURCE", "FIELD", "PROBE", "DERIVED"):
            block = ob.get(k, {})
            if isinstance(block, dict):
                items = ", ".join(f"{name}{':'+str(v.get('shape')) if v.get('shape') else ''}" for name, v in block.items())
                lines.append(f"  {k} (alias of { {'STATE':'X','SOURCE':'Q','FIELD':'Phi','PROBE':'Y','DERIVED':'Y'} .get(k,k)}): {items}")
    lines.append("")
    lines.append("provenance/API used: Model(params['emitter'], params['edge_list'], params['positions']), Signals(time_ms,V_m,spikes,sources,field), NeuronalTensor(areas/layers/neuron_types), EdgeList(pre,post,weight,tau), positions (N×3), provenance(config_hash,tensor_identity)")
    lines.append("no kernel change, no overhead when unused (summary is off-hot-path; simulate/construct unchanged)")
    return "\n".join(lines)

