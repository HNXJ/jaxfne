"""``construct()`` for the ``adex`` emitter family and the kernel-argument helper it shares
with ``simulate()`` (jaxfne G-2 F2-F4, part B).

``Configuration.emitter(family="adex")`` builds conductance-based AdEx cells
(``jaxfne/emitters_adex.py``) wired by the ``connections()`` rules: the population is
``network(n=, cell_types=)``, the receptors are the ``mechanisms()`` in declaration order, and
the edges are compiled by :func:`jaxfne.connectivity.compile_connection_rules`.

The route is an allowlist. Every field of ``Configuration``, ``RuntimeConfig`` and ``Simulation``
must equal its default unless ``ADEX_CONSUMED`` lists it; construct and simulate refuse the rest
by name (:func:`refuse_adex_configuration`, :func:`refuse_adex_simulation`).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._config import Configuration, _counts_from_fractions, edge_seed_from_metadata
from ._model import Model
from .connectivity import compile_connection_rules, delay_steps_from_ms
from .emitters_adex import AdExParams, make_adex_params
from .emitters_adex import _PARAM_FIELDS as _ADEX_PARAM_FIELDS

# The allowlist. A path starts at its root (Configuration, Simulation or RuntimeConfig). "*"
# matches one list index or dict key; "**" matches the whole subtree below it. A field that is
# not listed here must equal its default.
ADEX_CONSUMED = (
    # Simulation: duration and step set n_steps and the kernel's step; runtime is checked field by
    # field below.
    ("Simulation", "duration_ms"),
    ("Simulation", "dt_ms"),
    ("Simulation", "runtime"),
    ("Simulation", "poisson_drive"),
    # RuntimeConfig seed and n_steps are owned by Simulation (resolved_runtime replaces them, and
    # Simulation.__post_init__ refuses a conflicting value).
    ("RuntimeConfig", "seed"),
    ("RuntimeConfig", "n_steps"),
    # network(): name labels the run (signals.metadata["network"]); n and cell_types set the
    # population; connectivity(edge_seed=) is copied onto the first network.
    ("Configuration", "networks", "*", "name"),
    ("Configuration", "networks", "*", "n"),
    ("Configuration", "networks", "*", "cell_types", "**"),
    ("Configuration", "networks", "*", "connectivity", "edge_seed"),
    # emitter(): family selects the route; the adex keys set the kernel.
    ("Configuration", "emitters", "*", "family"),
    ("Configuration", "emitters", "*", "adex_decay"),
    ("Configuration", "emitters", "*", "adex_step_order"),
    # mechanisms(): kind must be "receptor" (checked in _adex_mechanisms); tau_ms and reversal_mV
    # set the receptor; status is written by the builder.
    ("Configuration", "metadata", "circuit", "mechanisms", "*", "name"),
    ("Configuration", "metadata", "circuit", "mechanisms", "*", "kind"),
    ("Configuration", "metadata", "circuit", "mechanisms", "*", "params", "tau_ms"),
    ("Configuration", "metadata", "circuit", "mechanisms", "*", "params", "reversal_mV"),
    ("Configuration", "metadata", "circuit", "mechanisms", "*", "status"),
    # connections(): the rule fields the compiler reads for this route; status is the builder's.
    ("Configuration", "metadata", "circuit", "connections", "*", "name"),
    ("Configuration", "metadata", "circuit", "connections", "*", "source", "**"),
    ("Configuration", "metadata", "circuit", "connections", "*", "target", "**"),
    ("Configuration", "metadata", "circuit", "connections", "*", "probability"),
    ("Configuration", "metadata", "circuit", "connections", "*", "weight"),
    ("Configuration", "metadata", "circuit", "connections", "*", "mechanism"),
    ("Configuration", "metadata", "circuit", "connections", "*", "delay_ms"),
    ("Configuration", "metadata", "circuit", "connections", "*", "intra_step_rounds"),
    ("Configuration", "metadata", "circuit", "connections", "*", "allow_self_connections"),
    ("Configuration", "metadata", "circuit", "connections", "*", "sampling"),
    ("Configuration", "metadata", "circuit", "connections", "*", "status"),
    # Builder bookkeeping: mechanisms() and connections() write these; _BUILDER_FLAGS checks the value.
    ("Configuration", "metadata", "circuit_declared"),
    ("Configuration", "metadata", "circuit_compiled"),
    # Edge seed: connectivity(edge_seed=), else the runtime seed (edge_seed_from_metadata).
    ("Configuration", "metadata", "connectivity", "edge_seed"),
    ("Configuration", "metadata", "seed"),
    # Per-(area, layer, type) drive declarations bridged as tensor_triple_drive
    # (neuronal_tensor_to_adex_configuration records them; construct_adex_model
    # realizes them as constant external current).
    ("Configuration", "metadata", "tensor_triple_drive", "**"),
    # runtime(dt_ms=, duration_ms=) are checked against the Simulation's values (one source).
    ("Configuration", "metadata", "dt_ms"),
    ("Configuration", "metadata", "duration_ms"),
)

# connections() writes these rule fields as None when the user leaves them unset. None is the
# compiler's default for each, so the builder's placeholder is accepted; any other value is refused
# like any other field the route does not consume.
ADEX_UNSET_IS_DEFAULT = (
    ("Configuration", "metadata", "circuit", "connections", "*", "sign"),
    ("Configuration", "metadata", "circuit", "connections", "*", "plasticity"),
    ("Configuration", "metadata", "circuit", "connections", "*", "control_key"),
    ("Configuration", "metadata", "circuit", "connections", "*", "max_in_degree"),
    ("Configuration", "metadata", "circuit", "connections", "*", "spatial_sigma"),
)

# The flags _append_circuit writes: True once a circuit is declared, False until it is compiled.
# A user cannot set them by hand, so refuse any other value (see refuse_adex_configuration).
_BUILDER_FLAGS = (("circuit_declared", True), ("circuit_compiled", False))

# Route defaults that differ from RuntimeConfig's. The adex route runs an edge-list kernel, so
# RuntimeConfig's default "dense" would describe a run it does not make; a RuntimeConfig passed
# to simulate() must set recurrent_backend="edge_list".
ADEX_DEFAULT_OVERRIDES = {"recurrent_backend": "edge_list"}

_ABSENT = object()  # the default of a field that the default configuration does not carry
_MAX_SHOWN = 120


def _finite_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _matches(pattern: tuple, path: tuple) -> bool:
    for i, want in enumerate(pattern):
        if want == "**":
            return True
        if i >= len(path):
            return False
        if want != "*" and want != path[i]:
            return False
    return len(pattern) == len(path)


def _consumed(path: tuple) -> bool:
    return any(_matches(pattern, path) for pattern in ADEX_CONSUMED)


def _show(path: tuple) -> str:
    out = str(path[0])
    for part in path[1:]:
        out += f"[{part!r}]" if isinstance(part, str) else f"[{part}]"
    return out


def _refuse(path: tuple, value: Any) -> None:
    shown = repr(value)
    if len(shown) > _MAX_SHOWN:
        shown = shown[: _MAX_SHOWN - 3] + "..."
    raise ValueError(
        f"{_show(path)} = {shown} is set away from its default, and emitter family 'adex' does not "
        "consume it; remove it or leave it at its default"
    )


def _unconsumed(value: Any, default: Any, path: tuple, found: list, declared: bool = False) -> None:
    """Collect ``(path, value)`` for each leaf set away from its default and not consumed.

    ``declared`` marks a list entry that the default does not carry: an empty container there is
    still a declaration (``field()`` with no arguments stores ``{}``), so it is collected too.
    """
    if default is not _ABSENT and value == default:
        return
    if _consumed(path):
        return
    if isinstance(value, Mapping):
        if declared and not value:
            found.append((path, value))
            return
        for key, sub in value.items():
            child = default.get(key, _ABSENT) if isinstance(default, Mapping) else _ABSENT
            _unconsumed(sub, child, path + (str(key),), found)
        return
    if isinstance(value, (list, tuple)):
        if declared and not value:
            found.append((path, value))
            return
        seq = default if isinstance(default, (list, tuple)) else ()
        for i, sub in enumerate(value):
            is_declared = i >= len(seq)
            _unconsumed(sub, _ABSENT if is_declared else seq[i], path + (i,), found, is_declared)
        return
    unset_ok = any(_matches(pattern, path) for pattern in ADEX_UNSET_IS_DEFAULT)
    if default is _ABSENT and value is None and unset_ok:
        return
    found.append((path, value))


def _runtime_defaults() -> dict[str, Any]:
    """RuntimeConfig's default for each field, with the adex route's overrides applied."""
    from ._runtime_config import RuntimeConfig

    base = RuntimeConfig()
    return {
        f.name: ADEX_DEFAULT_OVERRIDES[f.name] if f.name in ADEX_DEFAULT_OVERRIDES else getattr(base, f.name)
        for f in dataclasses.fields(RuntimeConfig)
    }


def refuse_adex_configuration(cfg: Configuration) -> None:
    """Refuse the first Configuration field the adex route does not consume (see ADEX_CONSUMED)."""
    circuit = "circuit" in cfg.metadata
    for key, written in _BUILDER_FLAGS:
        if key in cfg.metadata and not (circuit and cfg.metadata[key] is written):
            _refuse(("Configuration", "metadata", key), cfg.metadata[key])
    defaults = Configuration()
    metadata_default = {**_runtime_defaults(), **defaults.metadata}
    found: list[tuple[tuple, Any]] = []
    for f in dataclasses.fields(Configuration):
        default = metadata_default if f.name == "metadata" else getattr(defaults, f.name)
        _unconsumed(getattr(cfg, f.name), default, ("Configuration", f.name), found)
    if found:
        _refuse(*found[0])


def refuse_adex_simulation(sim: Any, cfg: Configuration) -> None:
    """Refuse the first Simulation or RuntimeConfig field the adex route does not consume, and a
    ``runtime(dt_ms=, duration_ms=)`` that conflicts with the Simulation's own values."""
    from ._runtime_config import RuntimeConfig
    from ._signals import Simulation

    defaults = Simulation()
    found: list[tuple[tuple, Any]] = []
    for f in dataclasses.fields(Simulation):
        _unconsumed(getattr(sim, f.name), getattr(defaults, f.name), ("Simulation", f.name), found)
    if sim.runtime is not None:
        runtime_defaults = _runtime_defaults()
        for f in dataclasses.fields(RuntimeConfig):
            _unconsumed(
                getattr(sim.runtime, f.name), runtime_defaults[f.name], ("RuntimeConfig", f.name), found
            )
    if found:
        _refuse(*found[0])
    for key, want in (("dt_ms", sim.dt_ms), ("duration_ms", sim.duration_ms)):
        have = cfg.metadata.get(key)
        if have is not None and float(have) != float(want):
            raise ValueError(
                f"runtime({key}={have!r}) conflicts with Simulation({key}={want!r}): "
                f"set {key} on Simulation only"
            )


def _adex_population(cfg: Configuration) -> tuple[int, tuple[str, ...]]:
    """``(n, labels)``: labels in the cell_types dict's order, counts from the plain route's rule."""
    net = cfg.networks[0] if cfg.networks else {}
    if net.get("n") is None:
        raise ValueError("network(n=...) is required for family 'adex': it sets the population size")
    if net.get("cell_types") is None:
        raise ValueError(
            "network(cell_types=...) is required for family 'adex': it maps each cell type to "
            "its fraction (types: E, PV, SOM)"
        )
    n = int(net["n"])
    fractions = dict(net["cell_types"])
    counts = _counts_from_fractions(n, fractions)
    if sum(counts.values()) != n:
        raise ValueError(
            f"network(cell_types={fractions!r}) gives counts {counts} that do not sum to n={n}"
        )
    labels = tuple(label for label, count in counts.items() for _ in range(count))
    return n, labels


def _adex_mechanisms(cfg: Configuration) -> tuple[list[str], tuple[float, ...], tuple[float, ...]]:
    """Receptor names, tau (ms) and reversal (mV) per declared mechanism, in declaration order."""
    names: list[str] = []
    tau: list[float] = []
    reversal: list[float] = []
    for mech in (cfg.metadata.get("circuit") or {}).get("mechanisms", []):
        name = mech.get("name")
        if mech.get("kind") != "receptor":
            raise ValueError(
                f"mechanism {name!r} has kind={mech.get('kind')!r}; the adex route realizes "
                "kind 'receptor' only"
            )
        params = dict(mech.get("params") or {})
        tau_ms = params.get("tau_ms")
        rev_mV = params.get("reversal_mV")
        if not (_finite_real(tau_ms) and tau_ms > 0) or not _finite_real(rev_mV):
            raise ValueError(
                f"mechanism {name!r} is not realized on the adex route: it needs params "
                f"tau_ms > 0 and a finite reversal_mV; got {params!r}"
            )
        names.append(str(name))
        tau.append(float(tau_ms))
        reversal.append(float(rev_mV))
    return names, tuple(tau), tuple(reversal)


def _adex_rules(cfg: Configuration, receptor_names: list[str]) -> list[dict[str, Any]]:
    """The connections() rules, each checked for the fields the adex compiler needs."""
    rules = list((cfg.metadata.get("circuit") or {}).get("connections", []))
    for rule in rules:
        rname = rule.get("name")
        if rule.get("mechanism") is None:
            raise ValueError(
                f"connections(name={rname!r}) has no mechanism: the adex route needs mechanism=... "
                "naming a declared mechanism"
            )
        if rule["mechanism"] not in receptor_names:
            raise ValueError(
                f"connections(name={rname!r}, mechanism={rule['mechanism']!r}) names an undeclared "
                f"mechanism; declared: {receptor_names}"
            )
        if rule.get("weight") is None:
            raise ValueError(f"connections(name={rname!r}) needs weight (nS) on the adex route")
        if rule.get("delay_ms") is None:
            raise ValueError(f"connections(name={rname!r}) needs delay_ms on the adex route")
    return rules


def _adex_triple_drive_pA(
    metadata: Mapping[str, Any], rows: "Sequence[dict]", labels: "tuple[str, ...]"
) -> "np.ndarray | None":
    """Per-neuron constant current (pA) from bridged ``tensor_triple_drive``.

    One declared drive unit reads as one pA on every neuron of exactly the
    declared (area, layer, type) triple; undeclared triples carry none.
    Absent metadata (every tensor without a declared drive, and every
    non-tensor config) returns None, so the kernel path stays unchanged.
    An entry naming no built neuron is refused unless its cell type is in
    the built composition (declared but rounded away: accepted, applies to
    no neuron); a non-finite drive (metadata surgery bypassing the bridge)
    fails as ValueError. Mirrors ``_apply_tensor_triple_drive`` in
    ``jaxfne/_construct_population.py`` (current-based route).
    """
    from .inputs import triple_drive_entries

    entries = (metadata or {}).get("tensor_triple_drive")
    if not entries:
        return None
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            f"tensor_triple_drive {entries!r} is not realized: it takes "
            "a list of {area, layer, cell_type, drive}"
        )
    # Values and order come from the compiled triple entries (metadata
    # order); membership refusal below keeps this route's exact behavior.
    compiled = triple_drive_entries(metadata, unit="pA")
    drive = np.zeros(len(rows), dtype=np.float64)
    declared = set(labels)
    for compiled_entry in compiled:
        area, layer, cell_type = compiled_entry.targets
        value: float = float(compiled_entry.value)
        matched = [
            i
            for i, row in enumerate(rows)
            if str(row.get("area")) == area
            and str(row.get("layer")) == layer
            and str(row.get("cell_type")) == cell_type
        ]
        if not matched:
            if cell_type in declared:
                continue
            raise ValueError(
                f"tensor_triple_drive(area={area!r}, layer={layer!r}, "
                f"cell_type={cell_type!r}) is not realized: it matches no neuron "
                "of the built population"
            )
        for i in matched:
            drive[i] = float(value)
    return drive


def _cast_params(params: AdExParams, dtype) -> AdExParams:
    fields = {f: jnp.asarray(getattr(params, f), dtype=dtype) for f in params._fields if f != "labels"}
    return params._replace(**fields)


def _apply_triple_param_overrides(
    params: AdExParams,
    rows: "Sequence[dict]",
    param_overrides: "Mapping[tuple[str, str, str], Mapping[str, float]]",
) -> AdExParams:
    """Apply per-(area, layer, type) param overrides onto per-neuron columns.

    ``param_overrides`` maps an ``(area, layer, cell_type)`` triple to a
    ``{field: value}`` mapping (the shape
    :func:`jaxfne.neuronal_tensor._resolve_tensor_emitter_family` returns):
    each entry overrides the named AdEx fields on every harvested neuron of
    exactly that triple, matching drive semantics — two triples sharing a
    cell type may carry different values. Unknown fields, non-finite
    values (bools refused, never coerced) and entries matching no harvested
    neuron are refused. A ``Vpeak_mV`` override must lie above that neuron's
    (post-override) ``EL_mV`` — absence is omitting the key (NaN), never an
    explicit value at or below rest. Returns the params with the overridden columns
    replaced (same dtype handling as the caller: float64 here, cast later).
    """
    if not isinstance(param_overrides, Mapping):
        raise ValueError(
            f"param_overrides {param_overrides!r} is not realized: it takes "
            "a mapping of (area, layer, cell_type) triple to {field: value}"
        )
    columns: dict[str, Any] = {}
    for key, fields in param_overrides.items():
        triple: "tuple[str, str, str] | None" = None
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
                "a mapping of AdEx field name to value"
            )
        for field, value in fields.items():
            if field not in _ADEX_PARAM_FIELDS:
                raise ValueError(
                    f"param_overrides[{key!r}] names unknown AdEx parameter field "
                    f"{field!r}; expected one of {list(_ADEX_PARAM_FIELDS)}"
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
                "of the harvested population"
            )
        for field, value in fields.items():
            column = columns.get(field)
            if column is None:
                column = np.asarray(getattr(params, field), dtype=np.float64).copy()
                columns[field] = column
            column[matched] = float(value)
    if "Vpeak_mV" in columns:
        el_column = columns.get("EL_mV")
        el_arr = np.asarray(
            el_column if el_column is not None else getattr(params, "EL_mV"),
            dtype=np.float64,
        )
        vp_arr = np.asarray(columns["Vpeak_mV"], dtype=np.float64)
        bad = [
            i
            for i in range(vp_arr.shape[0])
            if math.isfinite(float(vp_arr[i])) and not float(vp_arr[i]) > float(el_arr[i])
        ]
        if bad:
            first = bad[0]
            raise ValueError(
                f"param_overrides leaves Vpeak_mV={float(vp_arr[first])} at or below "
                f"EL_mV={float(el_arr[first])} on neuron {first}; Vpeak_mV must be "
                "above EL (absent = the key omitted = the reference VT + 5*Delta)"
            )
    if not columns:
        return params
    return params._replace(
        **{field: jnp.asarray(column, dtype=jnp.float64) for field, column in columns.items()}
    )


def construct_adex_model(
    cfg: Configuration,
    *,
    neuron_rows: "Sequence[dict] | None" = None,
    positions: Any = None,
    param_overrides: "Mapping[tuple[str, str, str], Mapping[str, float]] | None" = None,
) -> Model:
    """Build a :class:`Model` for the ``adex`` family (see the module docstring).

    The edges live in ``static["adex"]`` (``params["edge_list"]`` is None: the adex route runs
    its own kernel). ``static["adex"]`` keeps the declared per-rule delays; the delay in steps
    is realized at simulate time, when dt is known (:func:`adex_kernel_args`).

    The tensor path passes its harvested ``neuron_rows`` (real area/layer/cell
    types, so the selector rules compile against them), the posed
    ``positions``, and per-triple ``param_overrides`` (one
    ``{field: value}`` mapping per ``(area, layer, cell_type)`` triple,
    applied onto the per-neuron param columns by
    :func:`_apply_triple_param_overrides`); the pair comes together or
    not at all. Without them the model keeps the historical behavior
    (fraction-ordered labels, ``"adex"`` rows, zero positions).
    """
    # Local import: _construct_core imports this module at load time.
    from ._construct_core import _construct_build_static

    refuse_adex_configuration(cfg)
    if (neuron_rows is None) != (positions is None):
        raise ValueError(
            "neuron_rows and positions come as a harvested pair; pass both or neither"
        )
    if param_overrides is not None and neuron_rows is None:
        raise ValueError(
            "param_overrides needs the harvested neuron_rows it overrides; pass both or neither"
        )
    emitter = cfg.emitters[0]
    decay = emitter.get("adex_decay")
    if decay is None:
        raise ValueError(
            "emitter(family='adex') needs adex_decay ('exp' or 'euler'): the kernel has no default"
        )
    step_order = emitter.get("adex_step_order")
    step_order = "default" if step_order is None else step_order

    n, labels = _adex_population(cfg)
    dtype_name = str(cfg.metadata.get("dtype", "float32"))
    mechanisms = list((cfg.metadata.get("circuit") or {}).get("mechanisms", []))
    receptor_names, receptor_tau, receptor_reversal = _adex_mechanisms(cfg)
    rules = _adex_rules(cfg, receptor_names)
    network_name = cfg.networks[0].get("name") if cfg.networks else None

    if neuron_rows is None:
        params = _cast_params(make_adex_params(labels), jnp.dtype(dtype_name))
        rows = [
            {"neuron_id": i, "area": "adex", "layer": "adex", "cell_type": labels[i]} for i in range(n)
        ]
        area_names = ["adex"]
        layer_names = ["adex"]
        model_positions = jnp.zeros((n, 3), dtype=jnp.dtype(dtype_name))
    else:
        rows = [dict(row) for row in neuron_rows]
        if len(rows) != n:
            raise ValueError(
                f"neuron_rows holds {len(rows)} rows but network(n={n}); pass the harvested pair unchanged"
            )
        # Row order (construction order), not the fraction-grouped order of
        # _adex_population: the kernel indexes params by neuron, so params
        # follow the rows. The multiset must still match the declared
        # fractions exactly.
        labels = tuple(str(row["cell_type"]) for row in rows)
        net = cfg.networks[0] if cfg.networks else {}
        want = _counts_from_fractions(n, dict(net.get("cell_types") or {}))
        from collections import Counter as _Counter

        want_labels = [lab for lab, count in want.items() for _ in range(count)]
        if _Counter(labels) != _Counter(want_labels):
            raise ValueError(
                "neuron_rows cell types disagree with network(cell_types=...): "
                f"{sorted(set(labels))} vs {sorted(set(want))}"
            )
        params = _cast_params(
            _apply_triple_param_overrides(
                make_adex_params(labels), rows,
                {} if param_overrides is None else param_overrides,
            ),
            jnp.dtype(dtype_name),
        )
        area_names = sorted({str(row["area"]) for row in rows})
        layer_names = sorted({str(row["layer"]) for row in rows})
        model_positions = jnp.asarray(np.asarray(positions, dtype=np.float64).reshape(n, 3))
    # The edge seed is the plain route's: connectivity(edge_seed=) or else the runtime seed.
    edge_seed = edge_seed_from_metadata(cfg.metadata)
    # Per-(area, layer, type) drive declarations (tensor_triple_drive metadata):
    # one unit reads as one pA of constant external current on every neuron
    # of the declaring triple; undeclared triples carry none, and absent
    # metadata (every tensor without a declared drive, every non-tensor
    # config) keeps the historical path (None: no I_ext_pA downstream).
    triple_drive = _adex_triple_drive_pA(
        cfg.metadata, rows, tuple(sorted(set(labels)))
    )
    result = compile_connection_rules(
        rows, rules, mechanisms, seed=edge_seed, dtype=dtype_name,
        allow_intra_step_rounds=True,
    )

    static = _construct_build_static(cfg, geometry_meta=None)
    static["neuron_metadata"] = rows
    static["neuron_metadata_summary"] = {
        "n_rows": n,
        "areas": area_names,
        "layers": layer_names,
        "cell_types": sorted(set(labels)),
    }
    static["adex"] = {
        "pre": np.asarray(result.edge_pre, dtype=np.int64),
        "post": np.asarray(result.edge_post, dtype=np.int64),
        "weight_nS": np.asarray(result.edge_weight, dtype=np.float64),
        "receptor": np.asarray(result.edge_mechanism, dtype=np.int64),
        "delay_ms": np.asarray(result.edge_delay_ms, dtype=np.float64),
        "rules": [(str(r.get("name")), float(r["delay_ms"])) for r in rules],
        "rule_id": np.asarray(result.edge_rule_id, dtype=np.int64),
        "intra_step_rounds": {
            str(r.get("name")): int(r["intra_step_rounds"])
            for r in rules
            if r.get("intra_step_rounds") is not None
        },
        "network": None if network_name is None else str(network_name),
        "receptor_names": tuple(receptor_names),
        "receptor_reversal_mV": receptor_reversal,
        "receptor_tau_ms": receptor_tau,
        "decay": str(decay),
        "step_order": str(step_order),
        "edge_seed": int(edge_seed),
        "triple_drive_pA": (
            None if triple_drive is None
            else np.asarray(triple_drive, dtype=np.float64)
        ),
    }
    return Model(
        cfg=cfg,
        params={
            "emitter": params,
            "positions": model_positions,
            "edge_list": None,
        },
        static=static,
    )


def _rule_delay_steps(name: str, delay_ms: float, dt_ms: float) -> int:
    """``round(delay_ms / dt_ms)`` (the plain route's rule); zero-delay refused.

    The kernel reads each edge's spikes from a ring of past steps, so a rule
    realizing to 0 steps has no slot to read from: anything below one step is
    refused rather than run undelayed.
    """
    try:
        steps = delay_steps_from_ms(delay_ms, dt_ms)
    except ValueError as exc:
        raise ValueError(
            f"connections(name={name!r}) delay_ms={delay_ms} at dt_ms={dt_ms} rounds to 0 steps; "
            "zero-delay edges are refused on the adex route (it needs >= 1 step)"
        ) from exc
    if steps < 1:
        raise ValueError(
            f"connections(name={name!r}) delay_ms={delay_ms} at dt_ms={dt_ms} is {steps} steps; "
            "zero-delay edges are refused on the adex route (it needs >= 1 step)"
        )
    return steps


def _rule_delay_steps_allow_zero(name: str, delay_ms: float, dt_ms: float) -> int:
    """``round(delay_ms / dt_ms)`` admitting 0 steps (the intra-step fixed point's rule).

    Same realization as :func:`_rule_delay_steps`, except a delay rounding to
    0 steps returns 0 instead of raising; negative and non-finite delays still
    raise. The caller admits 0 only on a rule declaring ``intra_step_rounds``.
    """
    try:
        return int(delay_steps_from_ms(delay_ms, dt_ms))
    except ValueError as exc:
        try:
            ms = float(delay_ms)
        except (TypeError, ValueError):
            raise ValueError(
                f"connections(name={name!r}) delay_ms={delay_ms} is not realized: "
                "it takes a finite number in ms >= 0"
            ) from exc
        if math.isfinite(ms) and ms >= 0.0:
            return 0
        raise


def adex_kernel_args(model: Model, dt_ms: float, dtype: str | None = None) -> dict[str, Any]:
    """Keyword arguments for :func:`jaxfne.emitters_adex.simulate_adex` on ``model``.

    Returns ``params``, the five ``edge_*`` arrays (delays in steps at ``dt_ms``), the
    receptor reversal/tau tuples, ``decay`` and ``step_order``. The caller adds ``n_steps``,
    ``dt_ms``, ``record_v`` and optionally ``V0_mV``. ``I_ext_pA`` is included exactly
    when the model was built with a declared per-triple drive (``static["adex"]``
    ``["triple_drive_pA"]``); models built without one take the historical path
    (no key: the kernel runs undriven). ``dtype`` casts the cell
    parameters (and so the kernel's dtype); None keeps them as built.

    A rule declaring ``intra_step_rounds`` admits a delay realizing to 0 steps
    on that rule only: its edges ride the kernel's ``edge0_*`` set with
    ``intra_step_rounds`` (recorded in the manifest); without it a delay
    rounding to 0 steps is refused as before. A declared R with a delay
    realizing to >= 1 step, distinct R values across rules, or a declared R
    realizing no delay-0 edge is refused rather than silently unconsumed.
    Without any delay-0 edge the returned dict is unchanged.
    """
    emitter = model.params["emitter"]
    if dtype is not None:
        emitter = _cast_params(emitter, jnp.dtype(dtype))
    st = model.static["adex"]
    dt = float(dt_ms)
    declared_rounds = dict(st.get("intra_step_rounds") or {})
    if not declared_rounds:
        for name, delay_ms in st["rules"]:
            _rule_delay_steps(name, delay_ms, dt)
        steps = np.asarray(
            [delay_steps_from_ms(float(ms), dt) for ms in st["delay_ms"]], dtype=np.int64
        )
        out: dict[str, Any] = {
            "params": emitter,
            "edge_pre": st["pre"],
            "edge_post": st["post"],
            "edge_weight_nS": st["weight_nS"],
            "edge_receptor": st["receptor"],
            "edge_delay_steps": steps,
            "receptor_reversal_mV": st["receptor_reversal_mV"],
            "receptor_tau_ms": st["receptor_tau_ms"],
            "decay": st["decay"],
            "step_order": st["step_order"],
        }
        if st.get("triple_drive_pA", None) is not None:
            out["I_ext_pA"] = st["triple_drive_pA"]
        return out
    rule_names = [name for name, _ in st["rules"]]
    rule_steps: dict[str, int] = {}
    fixed_values: list[int] = []
    for name, delay_ms in st["rules"]:
        declared = declared_rounds.get(name)
        if declared is None:
            _rule_delay_steps(name, delay_ms, dt)
            rule_steps[name] = int(delay_steps_from_ms(float(delay_ms), dt))
        else:
            realized = _rule_delay_steps_allow_zero(name, delay_ms, dt)
            if realized >= 1:
                raise ValueError(
                    f"connections(name={name!r}) declares intra_step_rounds={declared} but "
                    f"delay_ms={delay_ms} at dt_ms={dt} realizes to {realized} steps (>= 1); "
                    "intra_step_rounds admits only a delay rounding to 0 steps on that "
                    "connection -- omit it or set a sub-step delay"
                )
            rule_steps[name] = 0
            fixed_values.append(int(declared))
    if len(set(fixed_values)) > 1:
        raise ValueError(
            f"connections declare distinct intra_step_rounds={sorted(set(fixed_values))}; "
            "the adex route's intra-step fixed point realizes one R -- declare one"
        )
    edge_rule = np.asarray(st["rule_id"], dtype=np.int64)
    steps = np.asarray([rule_steps[rule_names[int(r)]] for r in edge_rule], dtype=np.int64)
    zero = steps == 0
    n_zero = int(zero.sum())
    if fixed_values and not n_zero:
        raise ValueError(
            f"connections declare intra_step_rounds={fixed_values[0]} but realize no "
            f"delay-0 edge at dt_ms={dt}; a declared R with nothing to resolve is refused "
            "rather than silently unconsumed"
        )
    out = {
        "params": emitter,
        "edge_pre": np.asarray(st["pre"])[~zero],
        "edge_post": np.asarray(st["post"])[~zero],
        "edge_weight_nS": np.asarray(st["weight_nS"])[~zero],
        "edge_receptor": np.asarray(st["receptor"])[~zero],
        "edge_delay_steps": steps[~zero],
        "receptor_reversal_mV": st["receptor_reversal_mV"],
        "receptor_tau_ms": st["receptor_tau_ms"],
        "decay": st["decay"],
        "step_order": st["step_order"],
        "edge0_pre": np.asarray(st["pre"])[zero],
        "edge0_post": np.asarray(st["post"])[zero],
        "edge0_weight_nS": np.asarray(st["weight_nS"])[zero],
        "edge0_receptor": np.asarray(st["receptor"])[zero],
        "intra_step_rounds": int(fixed_values[0]),
    }
    if st.get("triple_drive_pA", None) is not None:
        out["I_ext_pA"] = st["triple_drive_pA"]
    return out


# ``Simulation(poisson_drive=...)`` on the adex route: the CAMINOS external drive, one Poisson
# count stream per target cell added to one receptor's conductance after a delay.
_DRIVE_KEYS = (
    "n_sources", "rate_hz", "weight_nS", "mechanism", "target_cell_types", "delay_ms", "seed",
)
# Optional drive-window bounds in ms (both declared or neither): the drive runs on steps
# [t_start, t_stop) only, realized with the plain route's rule at the run's dt.
_DRIVE_WINDOW_KEYS = ("t_start_ms", "t_stop_ms")
# Optional same-step flag: ``"same_step": True`` admits a delay rounding to 0 steps, so the
# count drawn for step t acts at step t (external drive has no same-step feedback, so no
# fixed point is needed). Any other value, or ``True`` with a delay of >= 1 step, is refused.
_DRIVE_OPT_KEYS = ("same_step",)


def _drive_error(detail: str) -> ValueError:
    return ValueError(
        f"Simulation(poisson_drive={detail}) is not realized for emitter family 'adex': "
        "the drive needs exactly the keys "
        f"{list(_DRIVE_KEYS)} (n_sources: int >= 1, rate_hz > 0, weight_nS >= 0, "
        "mechanism: a declared mechanism name, target_cell_types: a non-empty list of "
        "declared cell types, delay_ms, seed: int), plus optionally "
        f"{list(_DRIVE_WINDOW_KEYS)} (t_start_ms/t_stop_ms in ms, both or neither, "
        "with t_start_ms < t_stop_ms) and 'same_step' (same_step=True admits a delay "
        "rounding to 0 steps, so the count drawn for step t acts at step t)"
    )


def _checked_grouped_rates(drive: Mapping[str, Any]) -> list[float]:
    """Validate a listed ``rate_hz``: non-empty, finite, >= 0, distinct."""
    rates = list(drive["rate_hz"])
    if not rates:
        raise _drive_error(f"rate_hz={rates!r} must name at least one rate")
    seen: set[float] = set()
    for rate in rates:
        if not (_finite_real(rate) and float(rate) >= 0):
            raise _drive_error(f"rate_hz={rates!r} must hold rates >= 0")
        if float(rate) in seen:
            raise _drive_error(f"rate_hz={rates!r} must hold distinct rates")
        seen.add(float(rate))
    return [float(rate) for rate in rates]


def _grouped_drive_specs(
    drive: Mapping[str, Any],
    rates: list[float],
    targets: Any,
    dt: float,
    sim: Any,
) -> dict[str, Any]:
    """Grouped drive specs: per-group lambdas, a 2-d schedule and target groups.

    The declared ``n_sources`` channels split evenly across the groups; the
    target cells split evenly into contiguous subsets, one per group in the
    declared order. ``ext_lambda`` is the ``[n_steps, G]`` per-group schedule
    (``share * rate_g * dt_ms / 1000``) and ``ext_target_groups`` maps each
    target to its group column, so the kernel draws one stream per group.
    """
    total = int(drive["n_sources"])
    n_groups = len(rates)
    if total % n_groups:
        raise _drive_error(
            f"n_sources={total!r} must split evenly across {n_groups} rates"
        )
    n_targets = int(np.asarray(targets).size)
    if n_groups > n_targets:
        raise _drive_error(
            f"rate_hz={rates!r} names {n_groups} rates for {n_targets} target "
            "cells: each group needs at least one cell, so a group would hold "
            "an empty target subset -- declare at most one rate per target cell"
        )
    if n_targets % n_groups:
        raise _drive_error(
            f"{n_targets} target cells must split evenly across {n_groups} rates"
        )
    share = total // n_groups
    per_group = [float(share * rate * dt / 1000.0) for rate in rates]
    n_steps = int(sim.n_steps)
    return {
        "ext_targets": targets,
        "ext_lambda": np.tile(np.asarray(per_group)[None, :], (n_steps, 1)),
        "ext_target_groups": np.repeat(np.arange(n_groups), n_targets // n_groups),
    }


def adex_external_drive_kwargs(model: Model, sim: Any) -> dict[str, Any]:
    """Keyword arguments for the kernel's external drive on ``model`` (``ext_*`` plus ``key``).

    Empty when ``sim.poisson_drive`` is None. ``ext_lambda`` is
    ``n_sources * rate_hz * dt_ms / 1000`` (ms to s); a listed ``rate_hz``
    splits the channels evenly across the groups and rides a 2-d
    ``[n_steps, G]`` per-group schedule with ``ext_target_groups`` mapping
    each target to its group column (see ``_grouped_drive_specs``); the key is
    ``jax.random.PRNGKey(seed)``; targets are the cells whose type is listed in
    ``target_cell_types``, in index order; the receptor is the mechanism's declaration
    index; the delay is realized by the plain route's rule, except a delay rounding
    to 0 steps, which is admitted only with ``"same_step": True`` in the drive dict
    (the kernel's ``ext_delay_steps=0``: the count drawn for step t acts at step t;
    refused under ``step_order="brian2"``, which reads every arrival one step later).
    The optional
    ``t_start_ms``/``t_stop_ms`` window rides as the kernel's
    ``ext_t_start_steps``/``ext_t_stop_steps`` (realized by the same rule at
    the run's ``dt_ms``, refused unless ``0 <= t_start < t_stop <= n_steps``);
    undeclared, neither key is passed and the drive runs every step. Every
    refusal names ``poisson_drive`` and the offending key.
    """
    drive = sim.poisson_drive
    if drive is None:
        return {}
    if not isinstance(drive, Mapping):
        raise _drive_error(repr(drive))
    unknown = [
        k
        for k in drive
        if k not in _DRIVE_KEYS and k not in _DRIVE_WINDOW_KEYS and k not in _DRIVE_OPT_KEYS
    ]
    if unknown:
        raise _drive_error(f"{dict(drive)!r} has unknown keys {unknown}")
    missing = [k for k in _DRIVE_KEYS if k not in drive]
    if missing:
        raise _drive_error(f"{dict(drive)!r} is missing keys {missing}")
    st = model.static["adex"]
    receptor_names = list(st["receptor_names"])
    labels = tuple(model.params["emitter"].labels)
    n_sources = drive["n_sources"]
    if isinstance(n_sources, bool) or not isinstance(n_sources, (int, np.integer)):
        raise _drive_error(f"n_sources={n_sources!r} must be an int >= 1")
    if int(n_sources) < 1:
        raise _drive_error(f"n_sources={n_sources!r} must be an int >= 1")
    rate_hz = drive["rate_hz"]
    grouped_rates: list[float] | None = None
    if isinstance(rate_hz, (list, tuple)):
        grouped_rates = _checked_grouped_rates(drive)
    elif not (_finite_real(rate_hz) and float(rate_hz) > 0):
        raise _drive_error(f"rate_hz={rate_hz!r} must be > 0")
    weight_nS = drive["weight_nS"]
    if not (_finite_real(weight_nS) and float(weight_nS) >= 0):
        raise _drive_error(f"weight_nS={weight_nS!r} must be >= 0")
    mechanism = drive["mechanism"]
    if mechanism not in receptor_names:
        raise _drive_error(
            f"mechanism={mechanism!r} names an undeclared mechanism; declared: {receptor_names}"
        )
    wanted = drive["target_cell_types"]
    if (
        not isinstance(wanted, (list, tuple))
        or not wanted
        or any(not isinstance(t, str) for t in wanted)
        or any(t not in set(labels) for t in wanted)
    ):
        raise _drive_error(
            f"target_cell_types={wanted!r} must be a non-empty list of declared cell types "
            f"{sorted(set(labels))}"
        )
    delay_ms = drive["delay_ms"]
    if not _finite_real(delay_ms):
        raise _drive_error(f"delay_ms={delay_ms!r} must be a number")
    dt = float(sim.dt_ms)
    same_step = drive.get("same_step", False) is True
    if "same_step" in drive and not same_step:
        raise _drive_error(
            f"same_step={drive['same_step']!r} is not realized: admit a 0-step drive "
            "delay with same_step=True, or omit same_step"
        )
    if same_step and str(st.get("step_order", "default")) == "brian2":
        raise _drive_error(
            "same_step=True with step_order='brian2' is not realized: brian2 reads "
            "every arrival one step later, so a count drawn for step t cannot act at step t"
        )
    try:
        delay_steps = delay_steps_from_ms(float(delay_ms), dt)
    except ValueError as exc:
        if same_step and _finite_real(delay_ms) and float(delay_ms) >= 0.0:
            delay_steps = 0
        else:
            raise _drive_error(
                f"delay_ms={delay_ms!r} at dt_ms={dt} rounds to 0 steps; the drive needs >= 1"
            ) from exc
    if delay_steps < 1:
        if not same_step:
            raise _drive_error(
                f"delay_ms={delay_ms!r} at dt_ms={dt} is {delay_steps} steps; the drive needs >= 1"
            )
        delay_steps = 0
    elif same_step:
        raise _drive_error(
            f"same_step=True with delay_ms={delay_ms!r} at dt_ms={dt} realizes to "
            f"{delay_steps} steps (>= 1); same_step admits only a delay rounding to "
            "0 steps -- omit same_step or set a sub-step delay"
        )
    seed = drive["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise _drive_error(f"seed={seed!r} must be an int")
    targets = np.asarray(
        [i for i, lab in enumerate(labels) if lab in set(wanted)], dtype=np.int64
    )
    if grouped_rates is not None:
        out = _grouped_drive_specs(drive, grouped_rates, targets, dt, sim)
    else:
        out = {
            "ext_targets": targets,
            "ext_lambda": float(int(n_sources) * float(rate_hz) * dt / 1000.0),
        }
    out.update(
        {
            "ext_weight_nS": float(weight_nS),
            "ext_receptor": int(receptor_names.index(mechanism)),
            "ext_delay_steps": int(delay_steps),
            "key": jax.random.PRNGKey(int(seed)),
        }
    )
    t_start_ms = drive.get("t_start_ms")
    t_stop_ms = drive.get("t_stop_ms")
    if (t_start_ms is None) != (t_stop_ms is None):
        missing = "t_stop_ms" if t_start_ms is not None else "t_start_ms"
        raise _drive_error(
            f"{dict(drive)!r} declares one window bound without the other; "
            f"declare t_start_ms and t_stop_ms together ({missing} is missing)"
        )
    if t_start_ms is not None:
        for key_name, bound in (("t_start_ms", t_start_ms), ("t_stop_ms", t_stop_ms)):
            if not (_finite_real(bound) and float(bound) >= 0.0):
                raise _drive_error(f"{key_name}={bound!r} must be a finite number >= 0 in ms")
        if not float(t_start_ms) < float(t_stop_ms):
            raise _drive_error(
                f"t_start_ms={t_start_ms!r} must be below t_stop_ms={t_stop_ms!r}"
            )
        n_steps = int(sim.n_steps)
        try:
            t0 = int(delay_steps_from_ms(float(t_start_ms), dt))
            t1 = int(delay_steps_from_ms(float(t_stop_ms), dt))
        except ValueError as exc:
            raise _drive_error(
                f"t_start_ms={t_start_ms!r} and t_stop_ms={t_stop_ms!r} at dt_ms={dt}: "
                f"{exc}"
            ) from exc
        if not (0 <= t0 < t1 <= n_steps):
            raise _drive_error(
                f"t_start_ms={t_start_ms!r} and t_stop_ms={t_stop_ms!r} realize to steps "
                f"{t0} and {t1} at dt_ms={dt}, which must satisfy "
                f"0 <= t_start < t_stop <= n_steps ({n_steps})"
            )
        out["ext_t_start_steps"] = t0
        out["ext_t_stop_steps"] = t1
    return out
