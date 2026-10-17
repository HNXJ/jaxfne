"""Named parameter handles over a constructed :class:`Model` (Lane 8, gap G10).

A handle is a stable string path to one declared degree of freedom of the
core model, with functional get/set semantics::

    edges.weight[<selector>]                  executable recurrent weights
    cells.<family>.<field>[<group selector>]  per-neuron cell parameters
    drive.<name>.rate                         per-group injected drive

``get_parameter(model, path)`` returns a host-side copy of the selected
values; ``set_parameter(model, path, value)`` returns a new ``Model`` (the
input model is never mutated). ``list_parameters(model)`` enumerates every
fittable path with its shape, current value and whether it sits on a
differentiable path. Unknown paths are refused by name (``ValueError``
naming the path).

Selectors use ``(area, layer, neuron type)`` labels where the model carries
them (``static["neuron_metadata"]``, else the emitter's own labels), else
integer indices:

- ``[all]`` -- every element.
- ``[3]`` / ``[2:8]`` -- integer index / slice (cells: neuron rows;
  edge-list edges: executable edge rows).
- ``[E]`` -- bare cell-type label (cells only).
- ``[cell_type=E]``, ``[layer=L4]``, ``[area=V1]`` -- comma-separated
  ``key=value`` row matches (cells only).
- edges additionally accept ``receptor=<name|index>``, ``pre=<cell type>``,
  ``post=<cell type>`` (comma-separated AND).

Value semantics of :func:`set_parameter`:

- cells / drive: a scalar broadcasts uniformly over the selection; an array
  of selection size sets per-element values; a full-length array is applied
  at the selected positions. The write uses ``jnp`` scatter only, so a
  ``jax.grad`` tracer flows through (needed by the optax handles path).
- scalar cell fields (``cells.izhikevich.source_scale``, a 0-d array): only
  the ``[all]`` selector is accepted, with a scalar value; any indexed or
  labelled selector is refused by name.
- ``edges.weight`` on edge-list-authoritative models (placeholder dense ``W``
  or the adex route): a scalar is a sign-preserving per-edge magnitude
  (same convention as ``EdgeParameterSpec``); an array restores exact
  signed weights. Host-side numpy: concrete values only, not traceable --
  the optax handles loop refuses these paths (use the black-box loop).
- ``edges.weight[all]`` on dense-``W`` models: a scalar is a multiplicative
  gain on ``emitter.W`` (zeros stay zero, topology preserved); a full-shape
  array replaces ``W`` exactly. Indexed selectors are refused on dense
  ``W``. Mirrors ``_model_with_matrix_parameter``: the non-executed mirror
  (``edge_list``) is left untouched.

Only ``cells.izhikevich.source_scale[*]`` is marked differentiable (a linear
post-processing rescale that never crosses the spike/reset boundary,
``jaxfne/_model_tune.py`` source_scale path). Every other handle crosses
spiking dynamics and needs an explicit ``surrogate=`` for gradient fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ._model import Model

HANDLE_SCHEME_VERSION = "param_handles_v0.1"

IZH_CELL_FIELDS = ("a", "b", "c", "d", "drive", "source_scale")
ADEx_CELL_FIELDS = (
    "C_pF",
    "gL_nS",
    "EL_mV",
    "VT_mV",
    "Delta_mV",
    "Vreset_mV",
    "tauw_ms",
    "a_nS",
    "b_pA",
    "Tref_ms",
)

_CELL_MATCH_KEYS = frozenset({"area", "layer", "cell_type", "type"})
_EDGE_MATCH_KEYS = frozenset({"receptor", "pre", "post"})


@dataclass(frozen=True)
class _Handle:
    """Parsed parameter path (internal; use :func:`parse_handle`)."""

    kind: str  # "cell" | "edge" | "drive"
    family: str  # cell: "izhikevich" | "adex"; edge: "edge"; drive: "drive"
    field: str  # cell field | "weight" | "rate"
    selector: str  # raw bracket/name text


def is_handle_path(path: Any) -> bool:
    """Return True when ``path`` parses as a parameter handle (never raises)."""
    try:
        parse_handle(path)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def parse_handle(path: Any) -> _Handle:
    """Parse ``path`` into a :class:`_Handle`; refuse unknown paths by name."""
    if not isinstance(path, str) or not path:
        raise ValueError(f"Unknown parameter path {path!r}: expected a handle string")
    if path.startswith("cells."):
        rest = path[len("cells.") :]
        head, bracket = _split_bracket(rest, path)
        parts = head.split(".")
        if len(parts) != 2 or parts[0] not in ("izhikevich", "adex") or not parts[1]:
            raise ValueError(
                f"Unknown parameter path {path!r}: expected "
                "'cells.<izhikevich|adex>.<field>[<selector>]'"
            )
        family, field = parts
        allowed = IZH_CELL_FIELDS if family == "izhikevich" else ADEx_CELL_FIELDS
        if field not in allowed:
            raise ValueError(
                f"Unknown parameter path {path!r}: family {family!r} has no "
                f"field {field!r} (expected one of {list(allowed)})"
            )
        _check_bracket(bracket, path, kind="cell")
        return _Handle(kind="cell", family=family, field=field, selector=bracket)
    if path.startswith("edges."):
        rest = path[len("edges.") :]
        head, bracket = _split_bracket(rest, path)
        if head != "weight":
            raise ValueError(
                f"Unknown parameter path {path!r}: expected 'edges.weight[<selector>]'"
            )
        _check_bracket(bracket, path, kind="edge")
        return _Handle(kind="edge", family="edge", field="weight", selector=bracket)
    if path.startswith("drive."):
        rest = path[len("drive.") :]
        if not rest.endswith(".rate") or len(rest) <= len(".rate"):
            raise ValueError(
                f"Unknown parameter path {path!r}: expected 'drive.<name>.rate'"
            )
        name = rest[: -len(".rate")]
        if not name:
            raise ValueError(
                f"Unknown parameter path {path!r}: expected 'drive.<name>.rate'"
            )
        return _Handle(kind="drive", family="drive", field="rate", selector=name)
    raise ValueError(
        f"Unknown parameter path {path!r}: expected one of "
        "'edges.weight[...]', 'cells.<family>.<field>[...]', 'drive.<name>.rate'"
    )


def _split_bracket(rest: str, path: str) -> tuple[str, str]:
    if "[" not in rest or not rest.endswith("]"):
        raise ValueError(
            f"Unknown parameter path {path!r}: missing '[<selector>]' brackets"
        )
    head, bracket = rest.split("[", 1)
    bracket = bracket[:-1]
    if not head or not bracket:
        raise ValueError(
            f"Unknown parameter path {path!r}: missing '[<selector>]' brackets"
        )
    return head, bracket


def _check_bracket(bracket: str, path: str, kind: str) -> None:
    if bracket == "all" or _is_int(bracket) or _is_slice(bracket):
        return
    if "," in bracket:
        pairs = [chunk.split("=") for chunk in bracket.split(",")]
        if any(len(pair) != 2 or not pair[0].strip() or not pair[1].strip() for pair in pairs):
            raise ValueError(
                f"Unknown parameter path {path!r}: selector {bracket!r} must be "
                "'all', an index, a start:stop slice, a label, or key=value pairs"
            )
        allowed = _CELL_MATCH_KEYS if kind == "cell" else _EDGE_MATCH_KEYS
        for key, _ in pairs:
            if key.strip() not in allowed:
                raise ValueError(
                    f"Unknown parameter path {path!r}: selector key {key.strip()!r} "
                    f"not in {sorted(allowed)}"
                )
        return
    if "=" in bracket:
        key, _, value = bracket.partition("=")
        allowed = _CELL_MATCH_KEYS if kind == "cell" else _EDGE_MATCH_KEYS
        if not key.strip() or not value.strip() or key.strip() not in allowed:
            raise ValueError(
                f"Unknown parameter path {path!r}: selector {bracket!r} not in "
                f"{sorted(allowed)}"
            )
        return
    if kind == "cell":
        return  # bare cell-type label, matched against rows at resolve time
    raise ValueError(
        f"Unknown parameter path {path!r}: edges selector {bracket!r} must be "
        "'all', an index, a start:stop slice, or key=value pairs "
        f"({sorted(_EDGE_MATCH_KEYS)})"
    )


def _is_int(text: str) -> bool:
    try:
        int(text)
    except (ValueError, TypeError):
        return False
    return True


def _is_slice(text: str) -> bool:
    if text.count(":") != 1:
        return False
    lo, _, hi = text.partition(":")
    return (lo == "" or _is_int(lo)) and (hi == "" or _is_int(hi))


def is_differentiable_handle(path: Any) -> bool:
    """Return True only for handles on a genuinely differentiable path.

    Only ``cells.izhikevich.source_scale[*]`` -- a linear post-processing
    rescale that never crosses the spike/reset boundary. Everything else
    needs an explicit ``surrogate=`` for gradient fits.
    """
    try:
        handle = parse_handle(path)
    except (ValueError, TypeError, AttributeError):
        return False
    return (
        handle.kind == "cell"
        and handle.family == "izhikevich"
        and handle.field == "source_scale"
    )


def _is_adex_model(model: "Model") -> bool:
    return model.static.get("adex") is not None


def _emitter_family(model: "Model") -> str:
    if _is_adex_model(model):
        return "adex"
    from .emitters import IzhikevichParams

    if isinstance(model.params["emitter"], IzhikevichParams):
        return "izhikevich"
    raise ValueError(
        f"Parameter handles are not realized for emitter type "
        f"{type(model.params['emitter']).__name__!r}"
    )


def _neuron_rows(model: "Model") -> list[dict[str, Any]]:
    rows = model.static.get("neuron_metadata")
    if rows is not None:
        return [dict(row) for row in rows]
    emitter = model.params["emitter"]
    layers = getattr(emitter, "layer_labels", None)
    if layers is None:
        layers = tuple("unspecified" for _ in emitter.labels)
    return [
        {
            "neuron_id": int(idx),
            "area": "network",
            "layer": str(layers[idx]),
            "cell_type": str(label),
        }
        for idx, label in enumerate(emitter.labels)
    ]


def _match_dict(selector: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in selector.split(","):
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key == "type":
            key = "cell_type"
        out[key] = value.strip()
    return out


def _resolve_cell_index(
    rows: list[dict[str, Any]], n: int, selector: str, path: str
) -> np.ndarray:
    if selector == "all":
        return np.arange(n, dtype=np.int64)
    if _is_int(selector):
        idx = int(selector)
        if not 0 <= idx < n:
            raise ValueError(
                f"Parameter path {path!r} indexes neuron {idx} of {n}"
            )
        return np.asarray([idx], dtype=np.int64)
    if _is_slice(selector):
        lo_s, _, hi_s = selector.partition(":")
        lo = 0 if lo_s == "" else int(lo_s)
        hi = n if hi_s == "" else int(hi_s)
        if not 0 <= lo <= hi <= n:
            raise ValueError(
                f"Parameter path {path!r} slice {selector!r} is out of range for {n} neurons"
            )
        return np.arange(lo, hi, dtype=np.int64)
    match = _match_dict(selector) if ("=" in selector or "," in selector) else {"cell_type": selector}
    if "type" in match:
        match["cell_type"] = match.pop("type")
    unknown = set(match) - set(_CELL_MATCH_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown parameter path {path!r}: selector keys {sorted(unknown)} "
            f"not in {sorted(_CELL_MATCH_KEYS)}"
        )
    idx = np.asarray(
        [
            row["neuron_id"]
            for row in rows
            if all(str(row.get(key)) == value for key, value in match.items())
        ],
        dtype=np.int64,
    )
    if idx.size == 0:
        raise ValueError(f"Parameter path {path!r} matched no neurons")
    return idx


def _edge_store(model: "Model") -> dict[str, Any]:
    """Describe the executable recurrent-weight store of ``model``."""
    if _is_adex_model(model):
        st = model.static["adex"]
        return {
            "kind": "adex",
            "pre": np.asarray(st["pre"], dtype=np.int64),
            "post": np.asarray(st["post"], dtype=np.int64),
            "receptor": np.asarray(st["receptor"], dtype=np.int64),
            "receptor_names": tuple(st.get("receptor_names") or ()),
            "weights": np.asarray(st["weight_nS"], dtype=float),
        }
    from .emitters import is_placeholder_dense_W, resolve_edge_weight

    emitter = model.params["emitter"]
    if "edge_list" in model.params and is_placeholder_dense_W(emitter.W, emitter.n_neurons):
        edges = model.params["edge_list"]
        return {
            "kind": "edge_list",
            "pre": np.asarray(edges.pre, dtype=np.int64),
            "post": np.asarray(edges.post, dtype=np.int64),
            "receptor": np.asarray(edges.receptor_index, dtype=np.int64),
            "receptor_names": (),
            "weights": np.asarray(
                resolve_edge_weight(edges, edges.weight.dtype, presynaptic_sign=emitter.sign),
                dtype=float,
            ),
        }
    return {
        "kind": "dense",
        "weights": np.asarray(emitter.W, dtype=float),
    }


def _resolve_edge_index(
    store: dict[str, Any],
    rows: list[dict[str, Any]],
    selector: str,
    path: str,
) -> np.ndarray:
    n_edges = int(store["weights"].shape[0]) if store["kind"] != "dense" else None
    if store["kind"] == "dense" and selector != "all":
        raise ValueError(
            f"Parameter path {path!r}: dense-W models only support "
            "'edges.weight[all]' (indexed edge selectors need an edge_list model)"
        )
    if selector == "all":
        if store["kind"] == "dense":
            return np.arange(int(store["weights"].size), dtype=np.int64)
        return np.arange(n_edges, dtype=np.int64)
    if store["kind"] == "dense":
        raise ValueError(
            f"Parameter path {path!r}: dense-W models only support 'edges.weight[all]'"
        )
    if _is_int(selector):
        idx = int(selector)
        if not 0 <= idx < n_edges:
            raise ValueError(f"Parameter path {path!r} indexes edge {idx} of {n_edges}")
        return np.asarray([idx], dtype=np.int64)
    if _is_slice(selector):
        lo_s, _, hi_s = selector.partition(":")
        lo = 0 if lo_s == "" else int(lo_s)
        hi = n_edges if hi_s == "" else int(hi_s)
        if not 0 <= lo <= hi <= n_edges:
            raise ValueError(
                f"Parameter path {path!r} slice {selector!r} is out of range for {n_edges} edges"
            )
        return np.arange(lo, hi, dtype=np.int64)
    match = _match_dict(selector)
    unknown = set(match) - set(_EDGE_MATCH_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown parameter path {path!r}: selector keys {sorted(unknown)} "
            f"not in {sorted(_EDGE_MATCH_KEYS)}"
        )
    mask = np.ones(n_edges, dtype=bool)
    lookup = {row["neuron_id"]: row for row in rows}
    for key, value in match.items():
        if key == "receptor":
            names = list(store.get("receptor_names") or ())
            if value in names:
                mask &= store["receptor"] == names.index(value)
            elif _is_int(value):
                mask &= store["receptor"] == int(value)
            else:
                raise ValueError(
                    f"Parameter path {path!r} names unknown receptor {value!r}"
                )
        elif key in ("pre", "post"):
            side = store["pre"] if key == "pre" else store["post"]
            mask &= np.asarray(
                [str(lookup.get(int(i), {}).get("cell_type")) == value for i in side]
            )
    if not np.any(mask):
        raise ValueError(f"Parameter path {path!r} matched no executable edges")
    return np.nonzero(mask)[0].astype(np.int64)


def _require_family_match(handle: _Handle, model: "Model", path: str) -> None:
    actual = _emitter_family(model)
    if handle.kind == "cell" and handle.family != actual:
        raise ValueError(
            f"Parameter path {path!r} addresses family {handle.family!r} but the "
            f"model is family {actual!r}"
        )
    if handle.kind == "edge" and actual == "adex":
        return  # adex edges live in static["adex"]; handled by the store
    if handle.kind == "drive" and actual != "izhikevich":
        raise ValueError(
            f"Parameter path {path!r}: the adex route carries no per-neuron "
            "drive array to fit"
        )


def get_parameter(model: "Model", path: str) -> np.ndarray:
    """Return a host-side copy of the values selected by ``path``."""
    handle = parse_handle(path)
    _require_family_match(handle, model, path)
    if handle.kind == "cell":
        emitter = model.params["emitter"]
        current = np.asarray(getattr(emitter, handle.field))
        if current.ndim == 0:
            if handle.selector != "all":
                raise ValueError(
                    f"Parameter path {path!r} selects into a model-wide scalar "
                    f"({handle.field!r}); only '[all]' is supported"
                )
            return current.copy()
        rows = _neuron_rows(model)
        idx = _resolve_cell_index(rows, int(current.shape[0]), handle.selector, path)
        return np.asarray(getattr(emitter, handle.field), dtype=float)[np.asarray(idx)].copy()
    if handle.kind == "edge":
        store = _edge_store(model)
        rows = _neuron_rows(model)
        idx = _resolve_edge_index(store, rows, handle.selector, path)
        if store["kind"] == "dense":
            return np.asarray(store["weights"], dtype=float).copy()
        return np.asarray(store["weights"], dtype=float)[np.asarray(idx)].copy()
    rows = _neuron_rows(model)
    emitter = model.params["emitter"]
    n = int(np.asarray(emitter.drive).shape[0])
    idx = _resolve_cell_index(rows, n, handle.selector, path)
    return np.asarray(emitter.drive, dtype=float)[np.asarray(idx)].copy()


def set_parameter(model: "Model", path: str, value: Any) -> "Model":
    """Return a new ``Model`` with the values selected by ``path`` replaced.

    Scalar values broadcast uniformly; selection-size and full-length arrays
    set per-element values. Functional: neither ``model`` nor ``value`` is
    mutated. Cell/drive writes stay ``jnp``-traceable in ``value``; edge-list
    and adex-edge writes are concrete-only (refused by the optax handles
    loop -- use the black-box loop for those).
    """
    from dataclasses import replace

    from . import _model as _model_mod

    handle = parse_handle(path)
    _require_family_match(handle, model, path)
    if handle.kind == "cell":
        emitter = model.params["emitter"]
        current = getattr(emitter, handle.field)
        if np.ndim(current) == 0:
            if handle.selector != "all":
                raise ValueError(
                    f"Parameter path {path!r} selects into a model-wide scalar "
                    f"({handle.field!r}); only '[all]' is supported"
                )
            new_values = jnp.asarray(value, dtype=jnp.asarray(current).dtype)
            if new_values.ndim != 0:
                raise ValueError(
                    f"Parameter path {path!r} addresses a scalar; value shape "
                    f"{tuple(new_values.shape)} is not a scalar"
                )
        else:
            rows = _neuron_rows(model)
            idx = _resolve_cell_index(rows, int(np.asarray(current).shape[0]), handle.selector, path)
            new_values = _scatter_values(current, idx, value, path)
        if hasattr(emitter, "_replace"):  # AdExParams NamedTuple
            new_emitter = emitter._replace(**{handle.field: new_values})
        else:  # IzhikevichParams frozen dataclass
            new_emitter = replace(emitter, **{handle.field: new_values})
        params = dict(model.params)
        params["emitter"] = new_emitter
        return _model_mod.Model(
            cfg=model.cfg,
            params=params,
            static=dict(model.static),
        )
    if handle.kind == "edge":
        return _set_edge_parameter(model, handle, path, value)
    emitter = model.params["emitter"]
    rows = _neuron_rows(model)
    idx = _resolve_cell_index(rows, int(np.asarray(emitter.drive).shape[0]), handle.selector, path)
    new_drive = _scatter_values(emitter.drive, idx, value, path)
    new_emitter = replace(emitter, drive=new_drive)
    params = dict(model.params)
    params["emitter"] = new_emitter
    return _model_mod.Model(
        cfg=model.cfg,
        params=params,
        static=dict(model.static),
    )


def _scatter_values(current: Any, idx: np.ndarray, value: Any, path: str) -> Any:
    """Scatter ``value`` into a copy of ``current`` at ``idx`` (jnp-traceable)."""
    cur = jnp.asarray(current)
    n = int(cur.shape[0])
    idx_arr = jnp.asarray(np.asarray(idx, dtype=np.int64))
    val = jnp.asarray(value, dtype=cur.dtype)
    if val.ndim == 0:
        update = jnp.broadcast_to(val, (int(idx.shape[0]),))
    elif val.shape == (int(idx.shape[0]),):
        update = val
    elif val.shape == (n,):
        update = val[idx_arr]
    else:
        raise ValueError(
            f"Parameter path {path!r}: value shape {tuple(val.shape)} fits neither "
            f"the selection ({int(idx.shape[0])}) nor the full array ({n})"
        )
    return cur.at[idx_arr].set(update)


def _set_edge_parameter(model: "Model", handle: _Handle, path: str, value: Any) -> "Model":
    from dataclasses import replace

    from . import _model as _model_mod

    store = _edge_store(model)
    rows = _neuron_rows(model)
    idx = _resolve_edge_index(store, rows, handle.selector, path)
    if store["kind"] == "dense":
        from .emitters import IzhikevichParams  # noqa: F401 (family guard)

        emitter = model.params["emitter"]
        val = jnp.asarray(value, dtype=emitter.W.dtype)
        if val.ndim == 0:
            new_W = emitter.W * val
        elif tuple(val.shape) == tuple(np.asarray(emitter.W).shape):
            new_W = val
        else:
            raise ValueError(
                f"Parameter path {path!r}: dense-W value must be a scalar gain "
                f"or a full {tuple(np.asarray(emitter.W).shape)} array, got "
                f"{tuple(val.shape)}"
            )
        params = dict(model.params)
        params["emitter"] = replace(emitter, W=new_W)
        return _model_mod.Model(
            cfg=model.cfg,
            params=params,
            static=dict(model.static),
        )
    weights = np.asarray(store["weights"], dtype=float).copy()
    selected = weights[np.asarray(idx)]
    val = np.asarray(value, dtype=float)
    if val.ndim == 0:
        magnitude = float(val)
        if not np.isfinite(magnitude):
            raise ValueError(f"Parameter path {path!r} requires a finite magnitude")
        if not np.all(np.isfinite(selected)) or np.any(selected == 0.0):
            raise ValueError(
                f"Parameter path {path!r} requires finite nonzero signed weights "
                "to preserve edge identity"
            )
        weights[np.asarray(idx)] = np.sign(selected) * magnitude
    elif val.shape == (int(idx.shape[0]),):
        weights[np.asarray(idx)] = val
    elif val.shape == weights.shape:
        weights = val.copy()
    else:
        raise ValueError(
            f"Parameter path {path!r}: value shape {tuple(val.shape)} fits neither "
            f"the selection ({int(idx.shape[0])}) nor the full edge vector "
            f"{tuple(weights.shape)}"
        )
    if store["kind"] == "adex":
        static = dict(model.static)
        adex = dict(static["adex"])
        adex["weight_nS"] = np.asarray(weights, dtype=np.float64)
        static["adex"] = adex
        return _model_mod.Model(
            cfg=model.cfg,
            params=dict(model.params),
            static=static,
        )
    edges = model.params["edge_list"]
    params = dict(model.params)
    params["edge_list"] = replace(
        edges,
        weight=jnp.asarray(weights, dtype=edges.weight.dtype),
        weight_storage="per_edge",
        weight_magnitude=None,
        mechanism_weight_magnitude_table=None,
    )
    return _model_mod.Model(
        cfg=model.cfg,
        params=params,
        static=dict(model.static),
    )


def list_parameters(model: "Model") -> list[dict[str, Any]]:
    """Enumerate every fittable handle path with shape, value and gradient leg.

    Each entry holds ``path``, ``kind``, ``family``, ``field``, ``shape``,
    ``dtype``, ``value`` (current values as nested lists), ``n`` (selection
    size) and ``differentiable`` (True only on a genuinely differentiable
    path -- ``cells.izhikevich.source_scale[*]``).
    """
    _require_family_match(
        _Handle(kind="cell", family=_emitter_family(model), field="a", selector="all"),
        model,
        "list_parameters",
    )
    family = _emitter_family(model)
    rows = _neuron_rows(model)
    labels: list[str] = []
    for row in rows:
        label = str(row.get("cell_type"))
        if label not in labels:
            labels.append(label)
    entries: list[dict[str, Any]] = []
    fields = IZH_CELL_FIELDS if family == "izhikevich" else ADEx_CELL_FIELDS
    emitter = model.params["emitter"]
    for field in fields:
        current = np.asarray(getattr(emitter, field), dtype=float)
        entries.append(
            _entry(f"cells.{family}.{field}[all]", "cell", family, field, current)
        )
        if current.ndim == 0:
            continue  # model-wide scalar: no per-group rows exist
        for label in labels:
            idx = _resolve_cell_index(rows, int(current.shape[0]), f"cell_type={label}", f"cells.{family}.{field}[all]")
            entries.append(
                _entry(
                    f"cells.{family}.{field}[cell_type={label}]",
                    "cell",
                    family,
                    field,
                    current[np.asarray(idx)],
                )
            )
    store = _edge_store(model)
    if store["kind"] == "dense":
        entries.append(
            _entry("edges.weight[all]", "edge", "dense", "weight", np.asarray(store["weights"], dtype=float))
        )
    else:
        weights = np.asarray(store["weights"], dtype=float)
        entries.append(_entry("edges.weight[all]", "edge", store["kind"], "weight", weights))
        for receptor in sorted(np.unique(store["receptor"]).tolist()):
            names = list(store.get("receptor_names") or ())
            token = names[int(receptor)] if int(receptor) < len(names) else str(int(receptor))
            sel = np.nonzero(store["receptor"] == int(receptor))[0]
            entries.append(
                _entry(
                    f"edges.weight[receptor={token}]",
                    "edge",
                    store["kind"],
                    "weight",
                    weights[sel],
                )
            )
    if family == "izhikevich":
        drive = np.asarray(model.params["emitter"].drive, dtype=float)
        entries.append(_entry("drive.all.rate", "drive", "izhikevich", "rate", drive))
        for label in labels:
            idx = _resolve_cell_index(rows, int(drive.shape[0]), f"cell_type={label}", "drive.all.rate")
            entries.append(
                _entry(f"drive.{label}.rate", "drive", "izhikevich", "rate", drive[np.asarray(idx)])
            )
    return entries


def _entry(path: str, kind: str, family: str, field: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "path": path,
        "kind": kind,
        "family": family,
        "field": field,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "value": values.tolist(),
        "n": int(values.size),
        "differentiable": bool(is_differentiable_handle(path)),
    }
