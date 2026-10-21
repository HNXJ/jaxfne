"""``construct()``/``simulate()`` support for the ``rate`` emitter family.

The ``rate`` route reuses the current-based (Izhikevich) bridge population
and edge sampling bit-exactly, then swaps the emitter object for
:class:`jaxfne.emitters_rate.RateNetworkParams`: same neurons, same signed
edge list, same posed positions and velocity delays — only the executed
point-neuron equations change (Wilson-Cowan-type rate dynamics,
:func:`jaxfne.emitters_rate.simulate_rate_edge`). ``simulate()``
dispatches on the emitter type (see ``jaxfne._model_simulate._simulate_rate``).

Per-type rate rows come from ``NeuronType.emitter_params`` (validated by
:func:`jaxfne.emitters_rate.check_rate_params`); undeclared triples run the
module defaults. The constant drive channel is inherited from the bridge
(declared triple drives plus the Izhikevich defaults), matching drive
semantics — the lane declares its input mapping in its PORT_SPEC.
``weight_sharing`` and ``local_inhibition`` are lif-only declarations and
are refused here by name; a bound trained W0 is refused by the caller
(izhikevich-only, as for lif/adex).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np


def apply_rate_triple_param_overrides(
    params: "Any",
    rows: Sequence[dict],
    param_overrides: Mapping[tuple, Mapping[str, Any]] | None,
) -> "Any":
    """Apply per-(area, layer, type) rate overrides onto per-neuron columns.

    ``param_overrides`` maps an ``(area, layer, cell_type)`` triple to a
    ``{field: value}`` mapping (the shape
    :func:`jaxfne.neuronal_tensor._resolve_tensor_emitter_family` returns for
    the ``rate`` family): each entry overrides the named rate fields on every
    neuron of exactly that triple, matching drive semantics. Unknown fields,
    non-finite values (bools refused, never coerced), unknown transfer names
    and entries matching no neuron are refused. ``None`` (or empty) returns
    ``params`` unchanged.
    """
    from .emitters_rate import check_rate_params

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
                "a mapping of rate parameter field name to value"
            )
        checked = check_rate_params(dict(fields))
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
        for field, value in checked.items():
            if field == "transfer":
                column = columns.get(field)
                if column is None:
                    column = list(getattr(params, field))
                    columns[field] = column
                for i in matched:
                    column[i] = str(value)
            else:
                column = columns.get(field)
                if column is None:
                    column = np.asarray(getattr(params, field), dtype=np.float64).copy()
                    columns[field] = column
                column[matched] = float(value)
    if not columns:
        return params
    transfer = columns.get("transfer")
    return replace(
        params,
        **{f: v for f, v in columns.items() if f != "transfer"},
        **({"transfer": tuple(transfer)} if transfer is not None else {}),
    )


def _refuse_rate_only_declarations(tensor: "Any") -> None:
    """Refuse lif-only connection declarations on the rate route, by name."""
    seen: list[Any] = []
    for area in tensor.areas:
        seen.extend(getattr(area, "inter_connections", []) or [])
    seen.extend(getattr(tensor, "area_connections", []) or [])
    for conn in seen:
        if getattr(conn, "weight_sharing", None) is not None:
            raise ValueError(
                f"weight_sharing={conn.weight_sharing!r} is not realized for emitter "
                "family 'rate': shared weights execute on the lif route only "
                "(only None is admitted here) — build the lif route, or drop "
                "weight_sharing"
            )
        if getattr(conn, "local_inhibition", None) is not None:
            raise ValueError(
                "local_inhibition is not realized for emitter family 'rate': "
                "the grouped intra-step cap executes on the lif route only — "
                "build the lif route, or drop local_inhibition"
            )


def construct_rate_tensor_model(
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
    param_overrides: Mapping[tuple, Mapping[str, Any]] | None = None,
) -> "Any":
    """Build a rate-population model from a tensor resolving to ``"rate"``.

    Population size, labels, positions, edges and delays come from the
    current-based bridge (built with emitter resolution skipped, so no
    family switch fires); the emitter object is then swapped for
    :class:`jaxfne.emitters_rate.RateNetworkParams` carrying the same drive
    channel (baseline plus declared triple drives) and one rate row per
    neuron (module defaults plus ``param_overrides`` per triple). Posing
    and velocity delays run exactly as on the current-based path. The model
    records ``recurrent_backend="edge_list"`` in its configuration metadata
    (edges live only in ``params["edge_list"]``).
    """
    from .core import construct
    from .emitters_rate import make_rate_params
    from .neuronal_tensor import (
        _apply_velocity_delays_izhi,
        _pose_positions_and_rows,
        _replace_model_positions,
        neuronal_tensor_to_configuration,
    )

    _refuse_rate_only_declarations(tensor)
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
    keep = np.asarray(izhi.drive, dtype=np.float64)
    if not bool(np.isfinite(keep).all()):
        raise ValueError("rate route inherits a non-finite drive from the bridge")
    sign = np.asarray(izhi.sign, dtype=np.float64)
    rate_params = make_rate_params(labels)
    rate_params = apply_rate_triple_param_overrides(rate_params, rows, param_overrides)
    rate_params = replace(
        rate_params, drive=keep, sign=sign, layer_labels=layer_labels,
    )
    new_params = dict(model.params)
    new_params["emitter"] = rate_params
    model = replace(model, params=new_params)
    # Edges live only in params["edge_list"] (the dense kernel would drop
    # them), so the model declares its backend need once here: a plain
    # jtfne.simulate(model) inherits recurrent_backend="edge_list" from this
    # metadata, while an explicit dense backend is still refused at
    # simulate time (refuse_rate_simulation names it).
    model = replace(
        model, cfg=model.cfg.update_metadata(recurrent_backend="edge_list")
    )
    new_static = dict(model.static)
    new_static["rate"] = {
        "emitter_family": "rate",
        "transfers": sorted(set(rate_params.transfer)),
        "triples_with_param_overrides": (
            0 if not param_overrides else len(param_overrides)
        ),
        "default_drive_note": (
            "drive inherits the current-based bridge channel "
            "(declared triple drives plus the bridge defaults)"
        ),
    }
    model = replace(model, static=new_static)

    posed, updated_rows = _pose_positions_and_rows(model.params["positions"], rows, tensor)
    model = _replace_model_positions(model, posed, updated_rows)
    model = _apply_velocity_delays_izhi(model, tensor, updated_rows)
    return model


def rate_edge_arrays(model: "Any") -> dict[str, Any]:
    """Materialize signed edge arrays for the rate kernel, flooring delays.

    Zero-step edges (the default topology compiles ``delay_steps == 0``)
    cannot advance the kernel's delay ring, so they floor to 1 step here;
    the count rides the returned dict (``n_floored_to_one_step``) for the
    simulate-time receipt. Constructed delays are untouched.
    """
    from .emitters import resolve_edge_delay_steps, resolve_edge_weight

    edges = model.params.get("edge_list")
    if edges is None:
        raise ValueError(
            "rate route needs params['edge_list']; the model carries none "
            "(the rate tensor route always compiles one)"
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
            "rate route edge arrays disagree: "
            f"{pre.shape} vs {post.shape} vs {weight.shape} vs {delay.shape}"
        )
    floored = int(np.sum(delay < 1))
    return {
        "pre": pre,
        "post": post,
        "weight": weight,
        "delay_steps": np.maximum(delay, 1),
        "n_floored_to_one_step": floored,
    }


def refuse_rate_simulation(sim: "Any", runtime_cfg: "Any") -> None:
    """Refuse the simulate() inputs the rate route does not consume, naming each."""
    if getattr(sim, "ablation", None) is not None:
        raise ValueError(
            f"Simulation(ablation={sim.ablation!r}) is not realized for emitter "
            "family 'rate': the route has no silence/disconnect mask; leave it None"
        )
    if bool(getattr(runtime_cfg, "enable_homeostasis", False)):
        raise ValueError(
            "RuntimeConfig(enable_homeostasis=True) is not realized for emitter "
            "family 'rate': the route has no homeostasis controller"
        )
    if str(getattr(runtime_cfg, "synaptic_kernel", "exponential")) != "exponential":
        raise ValueError(
            f"RuntimeConfig(synaptic_kernel={runtime_cfg.synaptic_kernel!r}) is not "
            "realized for emitter family 'rate': the route runs the rate edge "
            "kernel only"
        )
    requested = getattr(runtime_cfg, "recurrent_backend", None)
    if requested is not None and requested != "edge_list":
        raise ValueError(
            f"RuntimeConfig(recurrent_backend={requested!r}) is not realized for "
            "emitter family 'rate': edges live only in params['edge_list'] "
            "(the dense kernel would drop them); use 'edge_list'"
        )
    if bool(runtime_cfg.resolve_vmap(1)):
        raise ValueError(
            f"RuntimeConfig(vmap={runtime_cfg.vmap!r}) is not realized for emitter "
            "family 'rate': single-trial simulate() runs one trial with no batch "
            "to vectorize; leave vmap False"
        )
    if bool(getattr(runtime_cfg, "enable_hdp", False)):
        raise ValueError(
            "RuntimeConfig(enable_hdp=True) is not realized for emitter family "
            "'rate': the route runs fixed weights (no registered plasticity "
            "rule); leave enable_hdp False"
        )
    from ._runtime_config import RuntimeConfig

    null_keys = frozenset(RuntimeConfig().hdp_params)
    foreign = sorted(
        set(dict(getattr(runtime_cfg, "hdp_params", None) or {})) - null_keys,
        key=repr,
    )
    if foreign:
        raise ValueError(
            f"hdp_params keys {foreign} are not consumed on emitter family 'rate' "
            "without enable_hdp: the rate kernel runs no plasticity stream "
            "(only the default null-control gains are admitted "
            "there); leave hdp_params at its default"
        )
