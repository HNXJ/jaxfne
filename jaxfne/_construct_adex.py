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
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._config import Configuration, _counts_from_fractions, edge_seed_from_metadata
from ._model import Model
from .connectivity import compile_connection_rules, delay_steps_from_ms
from .emitters_adex import AdExParams, make_adex_params

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
    ("Configuration", "metadata", "circuit", "connections", "*", "allow_self_connections"),
    ("Configuration", "metadata", "circuit", "connections", "*", "sampling"),
    ("Configuration", "metadata", "circuit", "connections", "*", "status"),
    # Builder bookkeeping: mechanisms() and connections() write these; _BUILDER_FLAGS checks the value.
    ("Configuration", "metadata", "circuit_declared"),
    ("Configuration", "metadata", "circuit_compiled"),
    # Edge seed: connectivity(edge_seed=), else the runtime seed (edge_seed_from_metadata).
    ("Configuration", "metadata", "connectivity", "edge_seed"),
    ("Configuration", "metadata", "seed"),
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


def _cast_params(params: AdExParams, dtype) -> AdExParams:
    fields = {f: jnp.asarray(getattr(params, f), dtype=dtype) for f in params._fields if f != "labels"}
    return params._replace(**fields)


def construct_adex_model(
    cfg: Configuration,
    *,
    neuron_rows: "Sequence[dict] | None" = None,
    positions: Any = None,
    param_overrides: "Mapping[str, Mapping[str, float]] | None" = None,
) -> Model:
    """Build a :class:`Model` for the ``adex`` family (see the module docstring).

    The edges live in ``static["adex"]`` (``params["edge_list"]`` is None: the adex route runs
    its own kernel). ``static["adex"]`` keeps the declared per-rule delays; the delay in steps
    is realized at simulate time, when dt is known (:func:`adex_kernel_args`).

    The tensor path passes its harvested ``neuron_rows`` (real area/layer/cell
    types, so the selector rules compile against them), the posed
    ``positions``, and per-``NeuronType`` ``param_overrides`` (forwarded to
    :func:`jaxfne.emitters_adex.make_adex_params`); the pair comes together or
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
            make_adex_params(labels, overrides=param_overrides), jnp.dtype(dtype_name)
        )
        area_names = sorted({str(row["area"]) for row in rows})
        layer_names = sorted({str(row["layer"]) for row in rows})
        model_positions = jnp.asarray(np.asarray(positions, dtype=np.float64).reshape(n, 3))
    # The edge seed is the plain route's: connectivity(edge_seed=) or else the runtime seed.
    edge_seed = edge_seed_from_metadata(cfg.metadata)
    result = compile_connection_rules(rows, rules, mechanisms, seed=edge_seed, dtype=dtype_name)

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
        "network": None if network_name is None else str(network_name),
        "receptor_names": tuple(receptor_names),
        "receptor_reversal_mV": receptor_reversal,
        "receptor_tau_ms": receptor_tau,
        "decay": str(decay),
        "step_order": str(step_order),
        "edge_seed": int(edge_seed),
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
    """``round(delay_ms / dt_ms)`` (the plain route's rule); refuse anything below one step."""
    try:
        steps = delay_steps_from_ms(delay_ms, dt_ms)
    except ValueError as exc:
        raise ValueError(
            f"connections(name={name!r}) delay_ms={delay_ms} at dt_ms={dt_ms} rounds to 0 steps; "
            "the adex route needs >= 1"
        ) from exc
    if steps < 1:
        raise ValueError(
            f"connections(name={name!r}) delay_ms={delay_ms} at dt_ms={dt_ms} is {steps} steps; "
            "the adex route needs >= 1"
        )
    return steps


def adex_kernel_args(model: Model, dt_ms: float, dtype: str | None = None) -> dict[str, Any]:
    """Keyword arguments for :func:`jaxfne.emitters_adex.simulate_adex` on ``model``.

    Returns ``params``, the five ``edge_*`` arrays (delays in steps at ``dt_ms``), the
    receptor reversal/tau tuples, ``decay`` and ``step_order``. The caller adds ``n_steps``,
    ``dt_ms``, ``record_v`` and optionally ``I_ext_pA``/``V0_mV``. ``dtype`` casts the cell
    parameters (and so the kernel's dtype); None keeps them as built.
    """
    emitter = model.params["emitter"]
    if dtype is not None:
        emitter = _cast_params(emitter, jnp.dtype(dtype))
    st = model.static["adex"]
    dt = float(dt_ms)
    for name, delay_ms in st["rules"]:
        _rule_delay_steps(name, delay_ms, dt)
    steps = np.asarray(
        [delay_steps_from_ms(float(ms), dt) for ms in st["delay_ms"]], dtype=np.int64
    )
    return {
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


# ``Simulation(poisson_drive=...)`` on the adex route: the CAMINOS external drive, one Poisson
# count stream per target cell added to one receptor's conductance after a delay.
_DRIVE_KEYS = (
    "n_sources", "rate_hz", "weight_nS", "mechanism", "target_cell_types", "delay_ms", "seed",
)


def _drive_error(detail: str) -> ValueError:
    return ValueError(
        f"Simulation(poisson_drive={detail}) is not realized for emitter family 'adex': "
        "the drive needs exactly the keys "
        f"{list(_DRIVE_KEYS)} (n_sources: int >= 1, rate_hz > 0, weight_nS >= 0, "
        "mechanism: a declared mechanism name, target_cell_types: a non-empty list of "
        "declared cell types, delay_ms, seed: int)"
    )


def adex_external_drive_kwargs(model: Model, sim: Any) -> dict[str, Any]:
    """Keyword arguments for the kernel's external drive on ``model`` (``ext_*`` plus ``key``).

    Empty when ``sim.poisson_drive`` is None. ``ext_lambda`` is
    ``n_sources * rate_hz * dt_ms / 1000`` (ms to s); the key is
    ``jax.random.PRNGKey(seed)``; targets are the cells whose type is listed in
    ``target_cell_types``, in index order; the receptor is the mechanism's declaration
    index; the delay is realized by the plain route's rule. Every refusal names
    ``poisson_drive`` and the offending key.
    """
    drive = sim.poisson_drive
    if drive is None:
        return {}
    if not isinstance(drive, Mapping):
        raise _drive_error(repr(drive))
    unknown = [k for k in drive if k not in _DRIVE_KEYS]
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
    if not (_finite_real(rate_hz) and float(rate_hz) > 0):
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
    try:
        delay_steps = delay_steps_from_ms(float(delay_ms), dt)
    except ValueError as exc:
        raise _drive_error(
            f"delay_ms={delay_ms!r} at dt_ms={dt} rounds to 0 steps; the drive needs >= 1"
        ) from exc
    if delay_steps < 1:
        raise _drive_error(
            f"delay_ms={delay_ms!r} at dt_ms={dt} is {delay_steps} steps; the drive needs >= 1"
        )
    seed = drive["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise _drive_error(f"seed={seed!r} must be an int")
    targets = np.asarray(
        [i for i, lab in enumerate(labels) if lab in set(wanted)], dtype=np.int64
    )
    return {
        "ext_targets": targets,
        "ext_lambda": float(int(n_sources) * float(rate_hz) * dt / 1000.0),
        "ext_weight_nS": float(weight_nS),
        "ext_receptor": int(receptor_names.index(mechanism)),
        "ext_delay_steps": int(delay_steps),
        "key": jax.random.PRNGKey(int(seed)),
    }
