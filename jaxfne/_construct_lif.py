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


def _position_group_of_neuron(
    rows: Sequence[Mapping[str, Any]], *, label: str,
) -> list[int]:
    """Per-neuron retinotopic position group from posed ``(x, y)`` geometry.

    Neurons sharing an exact ``(x, y)`` compete (ids in first-appearance
    order); ``z`` distinguishes the competitors within a group. Missing or
    non-finite positions are refused, naming ``label`` — without geometry
    there is nothing to group by.
    """
    seen: dict[tuple[float, float], int] = {}
    groups: list[int] = []
    for i, row in enumerate(rows):
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"{label} needs posed (x, y) positions to group competitors; "
                f"neuron row {i} has none"
            )
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(
                f"{label} needs finite posed (x, y) positions; "
                f"neuron row {i} has ({x}, {y})"
            )
        key = (x, y)
        if key not in seen:
            seen[key] = len(seen)
        groups.append(seen[key])
    return groups


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


def _lif_bridge_alias_tensor(
    tensor: "Any",
) -> "tuple[Any, dict[str, str]]":
    """Bridge copy aliasing lif-only cell types onto Izhikevich names.

    The lif route samples its population through the current-based bridge,
    which refuses cell types outside
    :data:`jaxfne.emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS` — including the
    PCL natives ``"simple"``/``"complex"``. Each such lif-only type is
    renamed 1:1 onto a free Izhikevich name (counts, fractions, drives,
    geometry and connection structure untouched, so every RNG draw is
    preserved), and every connection selector follows the rename. ``"simple"``
    prefers ``"E"`` when free, so excitatory (AMPA) sources keep their P-023
    validity on the bridge. Per-type emitter declarations are dropped on the
    renamed types only (the family is forced past resolution and
    ``param_overrides`` were already extracted from the original tensor).
    Names the bridge knows (``"E"``, ``"PV"``, ...) and names neither family
    knows pass through untouched — a typo still fails in the bridge.

    Returns ``(bridge_tensor, restore)`` with ``restore`` mapping
    alias -> original. Without lif-only types the tensor is returned as-is
    with an empty map (the historical path byte-identically).

    Connections declaring ``local_inhibition`` ``resolution="event_scan"``
    are stripped from the bridge (they compile no edges: the scan replays
    the declared event stream, which has no edge-list expression); each
    such connection must carry no edge-semantics keys (anything that
    would change compiled edges is refused by name, since stripping
    would silently drop it). The spec itself is recorded from the
    original tensor in :func:`construct_lif_tensor_model`.
    """
    from dataclasses import replace as _replace

    from .emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS
    from .emitters_lif import LIF_CELL_TYPE_DEFAULTS
    from .neuronal_tensor import Area, NeuronalTensor, NeuronType

    _SCAN_EDGE_KEYS = (
        "footprint", "weight_sharing", "max_in_degree", "sampling",
        "spatial_sigma", "delay_ms", "conduction_velocity_m_per_s",
        "allow_self",
    )

    def _is_scan_connection(conn: "Any") -> bool:
        spec = getattr(conn, "local_inhibition", None)
        return (
            isinstance(spec, dict)
            and spec.get("resolution") == "event_scan"
        )

    def _refuse_scan_edge_keys(conn: "Any") -> None:
        for key in _SCAN_EDGE_KEYS:
            if getattr(conn, key, None) is not None:
                raise ValueError(
                    f"scan connection {key}={getattr(conn, key)!r} is "
                    "unmapped: resolution 'event_scan' compiles no edges, "
                    f"so declaring {key} would silently change nothing — "
                    "leave it undeclared"
                )

    used: list[str] = []
    for area in tensor.areas:
        for layer in area.layers:
            for ntype in layer.neuron_types:
                if ntype.name not in used:
                    used.append(ntype.name)
    lif_only = [
        name for name in used
        if name in LIF_CELL_TYPE_DEFAULTS and name not in IZHIKEVICH_CELL_TYPE_DEFAULTS
    ]
    if not lif_only:
        return tensor, {}
    free = [name for name in IZHIKEVICH_CELL_TYPE_DEFAULTS if name not in used]
    alias: dict[str, str] = {}
    if "simple" in lif_only and "E" in free:
        alias["simple"] = "E"
        free.remove("E")
    for name in lif_only:
        if name in alias:
            continue
        if not free:
            raise ValueError(
                f"lif route has no free Izhikevich bridge name for {name!r} "
                f"(types in use: {sorted(used)}); the bridge samples the "
                "population by name — free a bridge name or rename the type"
            )
        alias[name] = free.pop(0)
    restore = {v: k for k, v in alias.items()}

    def _bridge_type(ntype: "NeuronType") -> "NeuronType":
        if ntype.name not in alias:
            return ntype
        return NeuronType(
            name=alias[ntype.name],
            relative_size=ntype.relative_size,
            fraction=ntype.fraction,
            value_tag=ntype.value_tag,
            emitter=None,
            emitter_params=None,
            drive=ntype.drive,
        )

    def _bridge_area(area: "Area") -> "Area":
        layers = [
            _replace(
                layer,
                neuron_types=[_bridge_type(ntype) for ntype in layer.neuron_types],
            )
            for layer in area.layers
        ]
        if area.connectivity_mode == "unspecified":
            return Area(
                name=area.name,
                layers=layers,
                pose=area.pose if bool(getattr(area, "_pose_specified", False)) else None,
                connectivity_mode="unspecified",
            )
        conns = []
        for conn in area.inter_connections:
            if _is_scan_connection(conn):
                # Declared scan replay: no bridge edges; edge-semantics
                # keys would silently change nothing, so they are refused.
                _refuse_scan_edge_keys(conn)
                continue
            conns.append(_replace(
                conn,
                source_neuron_type=alias.get(conn.source_neuron_type, conn.source_neuron_type),
                target_neuron_type=alias.get(conn.target_neuron_type, conn.target_neuron_type),
            ))
        return Area(
            name=area.name,
            layers=layers,
            inter_connections=conns,
            pose=area.pose if bool(getattr(area, "_pose_specified", False)) else None,
        )

    new_areas = [_bridge_area(area) for area in tensor.areas]
    if tensor.connectivity_mode == "unspecified":
        return (
            NeuronalTensor(name=tensor.name, areas=new_areas, provenance=tensor.provenance),
            restore,
        )
    new_ac = []
    for conn in tensor.area_connections:
        if _is_scan_connection(conn):
            _refuse_scan_edge_keys(conn)
            continue
        new_ac.append(_replace(
            conn,
            source_neuron_type=alias.get(conn.source_neuron_type, conn.source_neuron_type),
            target_neuron_type=alias.get(conn.target_neuron_type, conn.target_neuron_type),
        ))
    return (
        NeuronalTensor(
            name=tensor.name, areas=new_areas,
            area_connections=new_ac, provenance=tensor.provenance,
        ),
        restore,
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

    Lif-only cell types (``"simple"``/``"complex"``) ride the bridge under
    aliased Izhikevich names (:func:`_lif_bridge_alias_tensor`) and are
    restored everywhere user-visible afterwards (emitter labels, drive/sign
    of the renamed neurons, neuron-table rows); the alias map rides the
    ``static["lif"]`` receipt.
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

    bridge_tensor, _restore = _lif_bridge_alias_tensor(tensor)
    cfg = neuronal_tensor_to_configuration(
        bridge_tensor,
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
    if _restore:
        # The bridge sampled lif-only types under aliased names; restore
        # the declared vocabulary (the adex harvest does the same).
        rows = [
            dict(
                row,
                cell_type=_restore.get(str(row.get("cell_type")), str(row.get("cell_type"))),
            )
            for row in rows
        ]
    izhi = model.params["emitter"]
    labels = [str(x) for x in izhi.labels]
    # Lif-only types sampled under aliased bridge names read their PCL row
    # under the declared name; untouched labels pass through identically.
    true_labels = [_restore.get(lab, lab) for lab in labels] if _restore else list(labels)
    layer_labels = (
        tuple(str(x) for x in izhi.layer_labels)
        if izhi.layer_labels is not None
        else None
    )
    lif_params = make_lif_params(
        [lif_cell_type_of(lab) for lab in true_labels],
    )
    keep = izhi.drive.dtype
    drive = jnp.asarray(izhi.drive, dtype=keep)
    if bool(jnp.any(~jnp.isfinite(drive))):
        raise ValueError("lif route inherits a non-finite drive from the bridge")
    sign = jnp.asarray(izhi.sign, dtype=keep)
    if _restore:
        # The bridge channel of a renamed neuron belongs to its alias, not
        # its declared type: recompute it from the declared triple (or the
        # lif default when undeclared) with the lif sign. Untouched labels
        # keep the inherited channel bit-exactly.
        triple_drive: dict[tuple[str, str, str], float] = {}
        for area in tensor.areas:
            for layer in area.layers:
                for ntype in layer.neuron_types:
                    if ntype.name in _restore.values() and ntype.drive is not None:
                        triple_drive[(area.name, layer.name, ntype.name)] = float(ntype.drive)
        d = np.asarray(drive, dtype=np.float64).copy()
        s = np.asarray(sign, dtype=np.float64).copy()
        for i, (lab, row) in enumerate(zip(labels, rows)):
            if lab in _restore:
                true = _restore[lab]
                key = (str(row.get("area")), str(row.get("layer")), true)
                d[i] = triple_drive.get(key, LIF_DEFAULT_DRIVE_MV)
                s[i] = float(LIF_LABEL_SIGN.get(true, 1.0))
        drive = jnp.asarray(d, dtype=keep)
        sign = jnp.asarray(s, dtype=keep)
    fallback = sorted({lab for lab in true_labels if lif_cell_type_of(lab) != lab})
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
        labels=tuple(true_labels),
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
        "bridge_alias_of_label": dict(_restore),
        "triples_with_param_overrides": (
            0 if not param_overrides else len(param_overrides)
        ),
        "default_drive_note": (
            "drive inherits the current-based bridge channel "
            f"(undeclared labels default to {LIF_DEFAULT_DRIVE_MV} mV/step "
            "only when the bridge default does; renamed lif-only types "
            "read the declared triple or the lif default)"
        ),
        "label_sign": {lab: float(LIF_LABEL_SIGN.get(lab, 1.0)) for lab in sorted(set(true_labels))},
    }
    declared_inhibition = []
    for area in tensor.areas:
        for ic in area.inter_connections:
            spec = getattr(ic, "local_inhibition", None)
            if spec is not None:
                entry = {
                    "rule": (
                        f"{area.name}.{ic.source_layer}.{ic.source_neuron_type}"
                        f"->{area.name}.{ic.target_layer}.{ic.target_neuron_type}"
                    ),
                    "order": str(spec["order"]),
                }
                if spec.get("rounds") is not None:
                    entry["rounds"] = int(spec["rounds"])
                if "resolution" in spec:
                    entry["resolution"] = str(spec["resolution"])
                declared_inhibition.append(entry)
    for ac in tensor.area_connections:
        spec = getattr(ac, "local_inhibition", None)
        if spec is not None:
            entry = {
                "rule": (
                    f"{ac.source_area}.{ac.source_layer}.{ac.source_neuron_type}"
                    f"->{ac.target_area}.{ac.target_layer}.{ac.target_neuron_type}"
                ),
                "order": str(spec["order"]),
            }
            if spec.get("rounds") is not None:
                entry["rounds"] = int(spec["rounds"])
            if "resolution" in spec:
                entry["resolution"] = str(spec["resolution"])
            declared_inhibition.append(entry)
    if declared_inhibition:
        # One spec executes per step; several distinct specs (different
        # rounds, order or resolution) are refused by name — the kernel runs
        # a single intra-step fixed point per step, grouped per position below.
        # The scan replay carries no rounds (rounds-free specs record None).
        _distinct = sorted(
            {(e.get("rounds"), e["order"], e.get("resolution", "whole_step"))
             for e in declared_inhibition}, key=repr)
        if len(_distinct) > 1:
            raise ValueError(
                "local_inhibition declares distinct specs "
                f"{[{'rounds': r, 'order': o, 'resolution': s} for r, o, s in _distinct]}; "
                "the lif route executes one grouped cap per step — declare one spec only"
            )
        # Recorded (and echoed onto the connection table by the compiler, so
        # it rides the manifest); the lif edge kernel executes the intra-step
        # fixed point (R rounds in the declared order per step).
        new_static["lif"]["local_inhibition"] = declared_inhibition
        # Position groups from A's posed geometry: neurons sharing an (x, y)
        # compete (retinotopic position; z distinguishes the competitors).
        # neuron_table() runs in population order, like labels/drive above.
        new_static["lif"]["local_inhibition_group_of_neuron"] = (
            _position_group_of_neuron(rows, label="local_inhibition")
        )
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
