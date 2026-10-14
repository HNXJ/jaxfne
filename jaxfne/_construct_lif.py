"""``construct()``/``simulate()`` support for the ``lif`` emitter family (PCL LIF).

The ``lif`` route reuses the current-based (Izhikevich) bridge population and
edge sampling bit-exactly, then swaps the emitter object for
:class:`jaxfne.emitters_lif.LIFNetworkParams`: same neurons, same signed
edge list, same posed positions and velocity delays — only the executed
point-neuron equations change. ``simulate()`` dispatches on the emitter type
(see ``jaxfne._model_simulate._simulate_lif``) into
:func:`jaxfne.emitters_lif.simulate_lif_edge`.

Reference: N'dri, Barbier, Teuliere, Triesch, Nat Commun 16:8880 (2025),
doi:10.1038/s41467-025-64234-z. Per-type LIF rows and defaults live in
:mod:`jaxfne.emitters_lif`; the registered HDP rule ``pcl_stdp`` lives in
:mod:`jaxfne.hdp_rules_pcl` and is ensured registered before any rule run.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np


def apply_lif_triple_param_overrides(
    params: "Any",
    rows: Sequence[dict],
    param_overrides: Mapping[tuple, Mapping[str, float]] | None,
) -> "Any":
    """Apply per-(area, layer, type) LIF overrides onto per-neuron columns.

    ``param_overrides`` maps an ``(area, layer, cell_type)`` triple to a
    ``{field: value}`` mapping (the shape
    :func:`jaxfne.neuronal_tensor._resolve_tensor_emitter_family` returns for
    the ``lif`` family): each entry overrides the named LIF fields on every
    neuron of exactly that triple, matching drive semantics. Unknown fields,
    non-finite values (bools refused, never coerced) and entries matching no
    neuron are refused. ``None`` (or empty) returns ``params`` unchanged.
    """
    from .emitters_lif import _PARAM_FIELDS

    if not param_overrides:
        return params
    if not isinstance(param_overrides, Mapping):
        raise ValueError(
            f"param_overrides {param_overrides!r} is not realized: it takes "
            "a mapping of (area, layer, cell_type) triple to {field: value}"
        )
    columns: dict[str, Any] = {}
    for key, fields in param_overrides.items():
        triple: tuple[str, str, str] | None = None
        if isinstance(key, (tuple, list)) and len(key) == 3:
            try:
                triple = (str(key[0]), str(key[1]), str(key[2]))
            except (TypeError, ValueError):
                triple = None
        if triple is None:
            raise ValueError(
                f"param_overrides key {key!r} is not realized: it takes an "
                "(area, layer, cell_type) triple"
            )
        if not isinstance(fields, Mapping):
            raise ValueError(
                f"param_overrides[{key!r}] {fields!r} is not realized: it takes "
                "a mapping of LIF field name to value"
            )
        for field, value in fields.items():
            if field not in _PARAM_FIELDS:
                raise ValueError(
                    f"param_overrides[{key!r}] names unknown LIF parameter field "
                    f"{field!r}; expected one of {list(_PARAM_FIELDS)}"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"param_overrides[{key!r}][{field!r}]={value!r} is not "
                    "realized: it takes a finite real number"
                )
        matched = [
            i
            for i, row in enumerate(rows)
            if (str(row.get("area")), str(row.get("layer")), str(row.get("cell_type")))
            == triple
        ]
        if not matched:
            raise ValueError(
                f"param_overrides[{key!r}] is not realized: it matches no neuron "
                "of the built population"
            )
        for field, value in fields.items():
            column = columns.get(field)
            if column is None:
                column = np.asarray(getattr(params, field), dtype=np.float64).copy()
                columns[field] = column
            column[matched] = float(value)
    if not columns:
        return params
    return replace(
        params,
        **{field: jnp.asarray(column, dtype=params.drive.dtype) for field, column in columns.items()},
    )


def construct_lif_tensor_model(
    tensor: "Any",
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    dtype: str = "float32",
    backend: "str | None" = None,
    jit: "bool | str | None" = None,
    vmap: "bool | str | None" = None,
    n_contacts: int = 16,
    param_overrides: Mapping[tuple, Mapping[str, float]] | None = None,
) -> "Any":
    """Build a PCL-LIF model from a tensor resolving to ``"lif"``.

    Population size, labels, positions, edges and delays come from the
    current-based bridge (built with emitter resolution skipped, so no
    family switch fires); the emitter object is then swapped for
    :class:`jaxfne.emitters_lif.LIFNetworkParams` carrying the same drive
    channel (baseline plus declared triple drives) and one PCL LIF row per
    neuron (:func:`jaxfne.emitters_lif.lif_cell_type_of`), with
    ``param_overrides`` applied per triple. Posing, HDP initial-state
    seeding and velocity delays run exactly as on the current-based path.
    The model records ``recurrent_backend="edge_list"`` in its
    configuration metadata (edges live only in ``params["edge_list"]``).
    """
    from .core import construct
    from .emitters_lif import (
        LIF_DEFAULT_DRIVE_MV,
        LIF_LABEL_SIGN,
        LIFNetworkParams,
        lif_cell_type_of,
        make_lif_params,
    )
    from .neuronal_tensor import (
        _apply_velocity_delays_izhi,
        _pose_positions_and_rows,
        _replace_model_positions,
        neuronal_tensor_to_configuration,
    )

    cfg = neuronal_tensor_to_configuration(
        tensor,
        seed=seed,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        emitter="izhikevich",
        dtype=dtype,
        backend=backend,
        jit=jit,
        vmap=vmap,
        n_contacts=n_contacts,
        _skip_emitter_resolution=True,
    )
    model = construct(cfg)
    rows = model.neuron_table()
    izhi = model.params["emitter"]
    labels = [str(x) for x in izhi.labels]
    layer_labels = (
        tuple(str(x) for x in izhi.layer_labels)
        if izhi.layer_labels is not None
        else None
    )
    lif_params = make_lif_params(
        [lif_cell_type_of(lab) for lab in labels],
    )
    keep = izhi.drive.dtype
    drive = jnp.asarray(izhi.drive, dtype=keep)
    if bool(jnp.any(~jnp.isfinite(drive))):
        raise ValueError("lif route inherits a non-finite drive from the bridge")
    sign = jnp.asarray(izhi.sign, dtype=keep)
    fallback = sorted({lab for lab in labels if lif_cell_type_of(lab) != lab})
    def col(name):
        return jnp.asarray(
            np.asarray(getattr(lif_params, name), dtype=np.float64), dtype=keep)
    lif_model_params = LIFNetworkParams(
        tau_m_ms=col("tau_m_ms"),
        v_thresh_mV=col("v_thresh_mV"),
        v_reset_mV=col("v_reset_mV"),
        v_min_mV=col("v_min_mV"),
        eta_rp_mV=col("eta_rp_mV"),
        tau_rp_ms=col("tau_rp_ms"),
        tau_sra_ms=col("tau_sra_ms"),
        delta_sra_mV=col("delta_sra_mV"),
        drive=drive,
        sign=sign,
        W=jnp.zeros((0, 0), dtype=keep),
        labels=tuple(labels),
        layer_labels=layer_labels,
    )
    lif_model_params = apply_lif_triple_param_overrides(
        lif_model_params, rows, param_overrides
    )
    new_params = dict(model.params)
    new_params["emitter"] = lif_model_params
    model = replace(model, params=new_params)
    # Edges live only in params["edge_list"] (the dense kernel would drop
    # them), so the model declares its backend need once here: a plain
    # jtfne.simulate(model) inherits recurrent_backend="edge_list" from this
    # metadata, while an explicit dense backend is still refused at
    # simulate time (refuse_lif_simulation names it).
    model = replace(
        model, cfg=model.cfg.update_metadata(recurrent_backend="edge_list")
    )
    new_static = dict(model.static)
    new_static["lif"] = {
        "emitter_family": "lif",
        "label_fallback_to_simple": fallback,
        "triples_with_param_overrides": (
            0 if not param_overrides else len(param_overrides)
        ),
        "default_drive_note": (
            "drive inherits the current-based bridge channel "
            f"(undeclared labels default to {LIF_DEFAULT_DRIVE_MV} mV/step "
            "only when the bridge default does)"
        ),
        "label_sign": {lab: float(LIF_LABEL_SIGN.get(lab, 1.0)) for lab in sorted(set(labels))},
    }
    declared_inhibition = []
    for area in tensor.areas:
        for ic in area.inter_connections:
            spec = getattr(ic, "local_inhibition", None)
            if spec is not None:
                declared_inhibition.append({
                    "rule": (
                        f"{area.name}.{ic.source_layer}.{ic.source_neuron_type}"
                        f"->{area.name}.{ic.target_layer}.{ic.target_neuron_type}"
                    ),
                    "rounds": int(spec["rounds"]),
                    "order": str(spec["order"]),
                })
    for ac in tensor.area_connections:
        spec = getattr(ac, "local_inhibition", None)
        if spec is not None:
            declared_inhibition.append({
                "rule": (
                    f"{ac.source_area}.{ac.source_layer}.{ac.source_neuron_type}"
                    f"->{ac.target_area}.{ac.target_layer}.{ac.target_neuron_type}"
                ),
                "rounds": int(spec["rounds"]),
                "order": str(spec["order"]),
            })
    if declared_inhibition:
        # One spec executes as a whole-population cap; several distinct
        # specs (different rounds or order) are refused by name — the kernel
        # runs a single intra-step fixed point per step.
        _distinct = sorted(
            {(e["rounds"], e["order"]) for e in declared_inhibition})
        if len(_distinct) > 1:
            raise ValueError(
                "local_inhibition declares distinct specs "
                f"{[{'rounds': r, 'order': o} for r, o in _distinct]}; the lif route "
                "executes one whole-population cap per step — declare one spec only"
            )
        # Recorded (and echoed onto the connection table by the compiler, so
        # it rides the manifest); the lif edge kernel executes the intra-step
        # fixed point (R rounds in the declared order per step).
        new_static["lif"]["local_inhibition"] = declared_inhibition
    sharing_groups: list[int] = []
    _next_singleton = 0
    _n_shared_rules = 0
    _rows = [r for r in (model.cfg.metadata.get("circuit", {}) or {}).get("connections", [])
             if isinstance(r, Mapping)]
    _compiled_total = sum(int(r.get("compiled_n_edges", 0) or 0) for r in _rows)
    _n_total = int(model.params["edge_list"].n_edges)
    _n_prebuilt = _n_total - _compiled_total
    if _n_prebuilt < 0:
        raise ValueError(
            "lif sharing groups disagree with the edge list: "
            f"{_n_total} executed edges but {_compiled_total} compiled rule edges"
        )
    # Executed order is prebuilt (dense-derived) edges first, then compiled
    # rules in declaration order (_concat_edge_lists appends, and class
    # compaction keeps order and count); prebuilt edges are unshared
    # singletons. Per-rule groups come from the compiler's own
    # sharing_group_of_edge echo (same order as that rule's executed edges),
    # remapped onto dense global ids so rules never share a group.
    sharing_groups.extend(range(_next_singleton, _next_singleton + _n_prebuilt))
    _next_singleton += _n_prebuilt
    for _row in _rows:
        _n = int(_row.get("compiled_n_edges", 0) or 0)
        if _row.get("weight_sharing") is not None:
            _g = _row.get("sharing_group_of_edge")
            if _g is None or len([int(g) for g in _g]) != _n:
                raise ValueError(
                    f"connection rule {_row.get('name')!r} declares weight_sharing but its "
                    f"compiled sharing_group_of_edge disagrees with compiled_n_edges ({_n})"
                )
            _g = [int(g) for g in _g]
            _base = _next_singleton
            _remapped = [g + _base for g in _g]
            sharing_groups.extend(_remapped)
            _next_singleton = (max(_remapped) + 1) if _remapped else _base
            _n_shared_rules += 1
        else:
            sharing_groups.extend(range(_next_singleton, _next_singleton + _n))
            _next_singleton += _n
    if len(sharing_groups) != _n_total:
        raise ValueError(
            "lif sharing groups disagree with the edge list: "
            f"{len(sharing_groups)} groups for {_n_total} edges"
        )
    if _n_shared_rules:
        # Per-edge sharing-group ids in executed edge order (global, dense:
        # unshared rules contribute one singleton group per edge); the kernel
        # collapses each group's weights to one shared value.
        new_static["lif"]["sharing_group_of_edge"] = [int(g) for g in sharing_groups]
        new_static["lif"]["n_sharing_groups"] = int(_next_singleton)
    model = replace(model, static=new_static)

    h_sum: dict[tuple[str, str, str], float] = {}
    h_count: dict[tuple[str, str, str], int] = {}
    for area in tensor.areas:
        for ic in area.inter_connections:
            key = (area.name, ic.target_layer, ic.target_neuron_type)
            h_sum[key] = h_sum.get(key, 0.0) + float(ic.plastic.H)
            h_count[key] = h_count.get(key, 0) + 1
    for ac in tensor.area_connections:
        key = (ac.target_area, ac.target_layer, ac.target_neuron_type)
        h_sum[key] = h_sum.get(key, 0.0) + float(ac.plastic.H)
        h_count[key] = h_count.get(key, 0) + 1
    if h_count:
        h_mean = {key: h_sum[key] / h_count[key] for key in h_count}
        H0 = jnp.asarray(
            [h_mean.get((row["area"], row["layer"], row["cell_type"]), 1.0) for row in rows],
            dtype=jnp.float32,
        )
        model = model.with_hdp_initial_state(H0=H0)

    posed, updated_rows = _pose_positions_and_rows(model.params["positions"], rows, tensor)
    model = _replace_model_positions(model, posed, updated_rows)
    model = _apply_velocity_delays_izhi(model, tensor, updated_rows)
    return model


def lif_edge_arrays(model: "Any") -> dict[str, Any]:
    """Materialize signed edge arrays for the LIF kernel, flooring delays.

    Zero-step edges (the default topology compiles ``delay_steps == 0``)
    cannot advance the kernel's bool ring, so they floor to 1 step here;
    the count rides the returned dict (``n_floored_to_one_step``) for the
    simulate-time receipt. Constructed delays are untouched.

    With declared ``weight_sharing`` (``static["lif"]["sharing_group_of_edge"]``),
    each group's executed weights collapse to the group mean, so one shared
    weight runs per group; the per-edge group ids ride ``"group"`` for the
    kernel's shared HDP accumulation. Without a declaration every edge is
    its own group and weights pass through unchanged.
    """
    from .emitters import resolve_edge_delay_steps, resolve_edge_weight

    edges = model.params.get("edge_list")
    if edges is None:
        raise ValueError(
            "lif route needs params['edge_list']; the model carries none "
            "(the lif tensor route always compiles one)"
        )
    emitter = model.params.get("emitter")
    presynaptic_sign = (
        emitter.sign if emitter is not None and hasattr(emitter, "sign") else None
    )
    pre = np.asarray(edges.pre).astype(np.int64).ravel()
    post = np.asarray(edges.post).astype(np.int64).ravel()
    weight = np.asarray(
        resolve_edge_weight(edges, np.float64, presynaptic_sign=presynaptic_sign),
        dtype=np.float64,
    ).ravel()
    delay = np.asarray(resolve_edge_delay_steps(edges)).astype(np.int64).ravel()
    if not (pre.shape == post.shape == weight.shape == delay.shape):
        raise ValueError(
            "lif route edge arrays disagree: "
            f"{pre.shape} vs {post.shape} vs {weight.shape} vs {delay.shape}"
        )
    group = np.asarray(
        (model.static.get("lif", {}) or {}).get("sharing_group_of_edge",
                                                np.arange(pre.shape[0])),
        dtype=np.int64,
    ).ravel()
    if group.shape != pre.shape:
        raise ValueError(
            "lif route sharing groups disagree with the edge list: "
            f"{group.shape} vs {pre.shape}"
        )
    if bool((group < 0).any()):
        raise ValueError("lif route sharing groups must be >= 0")
    shared = bool((model.static.get("lif", {}) or {}).get("sharing_group_of_edge"))
    if shared and pre.shape[0]:
        # One executed weight per group: the group mean (per-edge magnitudes
        # from the compiler differ only by the shared scalar-weight convention,
        # so the mean is the shared row the HDP rule accumulates onto).
        mean = np.zeros(int(group.max()) + 1, dtype=np.float64)
        count = np.zeros(int(group.max()) + 1, dtype=np.float64)
        np.add.at(mean, group, weight)
        np.add.at(count, group, 1.0)
        weight = (mean / np.maximum(count, 1.0))[group]
    floored = int(np.sum(delay < 1))
    return {
        "pre": pre,
        "post": post,
        "weight": weight,
        "delay_steps": np.maximum(delay, 1),
        "n_floored_to_one_step": floored,
        "group": group,
        "weight_sharing": shared,
    }


def refuse_lif_simulation(sim: "Any", runtime_cfg: "Any") -> None:
    """Refuse the simulate() inputs the lif route does not consume, naming each."""
    if getattr(sim, "ablation", None) is not None:
        raise ValueError(
            f"Simulation(ablation={sim.ablation!r}) is not realized for emitter "
            "family 'lif': the route has no silence/disconnect mask; leave it None"
        )
    if bool(getattr(runtime_cfg, "enable_homeostasis", False)):
        raise ValueError(
            "RuntimeConfig(enable_homeostasis=True) is not realized for emitter "
            "family 'lif': the route has no homeostasis controller"
        )
    if str(getattr(runtime_cfg, "synaptic_kernel", "exponential")) != "exponential":
        raise ValueError(
            f"RuntimeConfig(synaptic_kernel={runtime_cfg.synaptic_kernel!r}) is not "
            "realized for emitter family 'lif': the route runs the exponential "
            "edge kernel only"
        )
    requested = getattr(runtime_cfg, "recurrent_backend", None)
    if requested is not None and requested != "edge_list":
        raise ValueError(
            f"RuntimeConfig(recurrent_backend={requested!r}) is not realized for "
            "emitter family 'lif': edges live only in params['edge_list'] "
            "(the dense kernel would drop them); use 'edge_list'"
        )
    if bool(runtime_cfg.resolve_vmap(1)):
        raise ValueError(
            f"RuntimeConfig(vmap={runtime_cfg.vmap!r}) is not realized for emitter "
            "family 'lif': single-trial simulate() runs one trial with no batch "
            "to vectorize; leave vmap False"
        )
    if not bool(getattr(runtime_cfg, "enable_hdp", False)):
        from ._runtime_config import RuntimeConfig

        null_keys = frozenset(RuntimeConfig().hdp_params)
        foreign = sorted(
            set(dict(getattr(runtime_cfg, "hdp_params", None) or {})) - null_keys,
            key=repr,
        )
        if foreign:
            raise ValueError(
                f"hdp_params keys {foreign} are not consumed on emitter family 'lif' "
                "without enable_hdp: the lif kernel runs no noise stream without a "
                "registered rule (only the default null-control gains are admitted "
                "there); pass enable_hdp=True with hdp_params={'hdp_rule': ...} "
                "or leave hdp_params at its default"
            )
