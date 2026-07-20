"""Connectivity-rule compilation, inter-area weight construction, small
EdgeList helpers, and the multi-model ``connect()`` merge machinery.

Split out of ``jaxfne/_construct.py`` (Phase 2 defragmentation, 2026-07-20,
part of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). ``jaxfne/_construct.py``
re-exports every symbol here for backward compatibility. This module has no
dependency on ``_construct_core``/``_construct_population``/``_construct_presets``
-- it is a leaf in the internal dependency graph (they depend on it, not the
reverse), which is what keeps this split acyclic.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from .emitters import EdgeList, IzhikevichParams, make_edge_list_from_dense
from ._config import (
    Configuration,
)
from ._model import Model


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
            pre_flat = pre_grid.ravel()
            post_flat = post_grid.ravel()
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
        pre_all.append(pre_g)
        post_all.append(post_g)
        w_all.append(wv)
        counts.append(int(pre_g.size))

    if not pre_all:
        return None, counts
    pre = _np.concatenate(pre_all)
    post = _np.concatenate(post_all)
    w = _np.concatenate(w_all)
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
            xmin = float(jnp.min(p[:, 0]))
            xmax = float(jnp.max(p[:, 0]))
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


