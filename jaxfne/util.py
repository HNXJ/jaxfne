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
