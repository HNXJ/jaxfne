"""Agent tool surface: six scientific operations over typed objects (agent-native step 5).

EXPERIMENTAL, not exported at the package root. An agent builds a standard
experiment from these six operations instead of the classified root symbols;
each composes existing canonical functions and adds no new mathematics::

    realize(TFNE text) -> Realization                  (parse -> resolve -> realize)
    simulate(Realization | Configuration | Model, duration_ms, dt_ms, seed) -> Run
    inspect(Run | Model) -> dict                       (counts, edges, recorded outputs)
    compare(Run) -> dict                               (realized vs executed, per class)
    observe(Run, quantity) -> Observation              (recorded readout or refusal)
    verify(Run, property) -> dict                      (named property -> PASS/FAIL + evidence)

Refusal rule: an operation that cannot establish its answer from the objects
raises (``ValueError``/``KeyError``) or reports ``NOT_APPLICABLE``; it never
substitutes a default, a proxy, or a synthesized readout.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "Observation",
    "PROPERTIES",
    "Run",
    "compare",
    "inspect",
    "observe",
    "realize",
    "simulate",
    "verify",
]

# Epistemic level per observable quantity. Membrane state and spikes are the
# emitter's native (uncalibrated) dynamics; sources and every laminar field
# readout are relative proxies (no calibration transform exists).
_LEVELS: dict[str, str] = {
    "V_m": "NATIVE_UNCALIBRATED",
    "spikes": "NATIVE_UNCALIBRATED",
    "sources": "RELATIVE_PROXY",
    "lfp_proxy": "RELATIVE_PROXY",
    "csd_proxy": "RELATIVE_PROXY",
    "phi_e_proxy": "RELATIVE_PROXY",
    "source_proxy": "RELATIVE_PROXY",
}


@dataclass(frozen=True)
class Run:
    """One executed simulation with the objects it came from.

    ``realization`` is present only when the run started from TFNE; without it
    the configured side of ``compare`` is unavailable (reported, not guessed).
    """

    model: Any
    signals: Any
    duration_ms: float
    dt_ms: float
    seed: int
    realization: Any = None
    configuration: Any = None


@dataclass(frozen=True)
class Observation:
    """A recorded readout with its epistemic level (never synthesized)."""

    quantity: str
    values: np.ndarray
    level: str
    shape: tuple[int, ...] = field(default=())


def realize(spec: str, *, seed: Optional[int] = None) -> Any:
    """TFNE text -> ``Realization`` (``s``, ``h0``, ``I``) via parse -> resolve -> realize."""
    from .tfne import parse, resolve
    from .tfne import realize as _realize

    program = parse(spec)
    return _realize(resolve(program), program, seed=seed)


def simulate(obj: Any, *, duration_ms: float, dt_ms: float, seed: int = 0) -> Run:
    """Construct when needed, then simulate; return the ``Run`` with its sources.

    ``obj`` is a ``Realization`` (compiled at ``dt_ms``), a ``Configuration``
    or a constructed ``Model``. Anything else is refused.
    """
    import jaxfne as J

    from .tfne import Realization, to_configuration

    realization = configuration = None
    if isinstance(obj, Realization):
        realization = obj
        configuration = to_configuration(obj, duration_ms=duration_ms, dt_ms=dt_ms)
        model = J.construct(configuration)
    elif isinstance(obj, J.Configuration):
        configuration = obj
        model = J.construct(obj)
    elif isinstance(obj, J.Model):
        model = obj
    else:
        raise TypeError(
            f"simulate takes a Realization, Configuration or Model, not {type(obj).__name__}"
        )
    signals = J.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
    return Run(
        model=model,
        signals=signals,
        duration_ms=float(duration_ms),
        dt_ms=float(dt_ms),
        seed=int(seed),
        realization=realization,
        configuration=configuration,
    )


def _recorded(signals: Any) -> list[str]:
    out = [k for k in ("V_m", "spikes") if getattr(signals, k, None) is not None]
    if getattr(signals, "sources", None) is not None:
        out.append("sources")
    fld = getattr(signals, "field", None)
    if fld is not None:
        out += [k for k in ("lfp_proxy", "csd_proxy", "phi_e_proxy", "source_proxy")
                if getattr(fld, k, None) is not None]
    return out


def inspect(obj: Any) -> dict[str, Any]:
    """Counts and recorded outputs of a ``Run`` (or a bare ``Model``), read from the objects."""
    run = obj if isinstance(obj, Run) else None
    model = run.model if run is not None else obj
    edges = model.params.get("edge_list")
    n_neurons = len(model.neuron_table())
    out: dict[str, Any] = {
        "n_neurons": n_neurons,
        "n_edges": None if edges is None else int(edges.n_edges),
        "edge_storage": None if edges is None else {
            "weight": edges.weight_storage,
            "delay": edges.delay_storage,
            "tau": edges.tau_storage,
        },
        "emitter": type(model.params.get("emitter")).__name__,
        "tfne": run is not None and run.realization is not None,
    }
    if run is not None:
        sig = run.signals
        out.update(
            duration_ms=run.duration_ms,
            dt_ms=run.dt_ms,
            seed=run.seed,
            n_steps=int(np.asarray(sig.time_ms).shape[0]),
            recorded=_recorded(sig),
        )
    return out


def _executed_edges(model: Any) -> tuple[Counter, Counter, Counter]:
    """Kernel-resolved (pre, post, weight), (pre, post, mechanism), (pre, post, delay_steps)."""
    from .emitters import _resolved_edge_weight, resolve_edge_delay_steps

    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre).astype(int)
    post = np.asarray(edges.post).astype(int)
    w = np.asarray(_resolved_edge_weight(edges, edges.weight.dtype, model.params["emitter"]))
    steps = np.asarray(resolve_edge_delay_steps(edges)).astype(int)
    weights = Counter((int(a), int(b), round(float(x), 6)) for a, b, x in zip(pre, post, w))
    delays = Counter((int(a), int(b), int(s)) for a, b, s in zip(pre, post, steps))
    mechs = Counter(
        (int(e["pre"]), int(e["post"]), str(e["receptor_type"]).split("__")[0])
        for e in model.edge_table()
    )
    return weights, mechs, delays


def _realized_edges(r: Any, dt_ms: float) -> tuple[Counter, Counter, Counter]:
    from .connectivity import delay_steps_from_ms

    s = r.s
    pre = np.asarray(s["edge_pre"]).astype(int)
    post = np.asarray(s["edge_post"]).astype(int)
    w = np.asarray(s["edge_weight"], dtype=float)
    names = [m["name"] for m in s["mechanism_table"]]
    mech = np.asarray(s["edge_mechanism"]).astype(int)
    d_ms = np.asarray(s["edge_delay_ms"], dtype=float)
    weights = Counter((int(a), int(b), round(float(x), 6)) for a, b, x in zip(pre, post, w))
    mechs = Counter((int(a), int(b), names[int(m)]) for a, b, m in zip(pre, post, mech))
    delays = Counter(
        (int(a), int(b), delay_steps_from_ms(float(d), dt_ms) if d > 0 else 0)
        for a, b, d in zip(pre, post, d_ms)
    )
    return weights, mechs, delays


def _class_verdict(realized: Counter, executed: Counter) -> dict[str, Any]:
    missing = realized - executed
    extra = executed - realized
    return {
        "verdict": "EQUAL" if not missing and not extra else "DIFFERENT",
        "n_realized": sum(realized.values()),
        "n_executed": sum(executed.values()),
        "missing_in_executed": [list(k) for k in list(missing)[:5]],
        "extra_in_executed": [list(k) for k in list(extra)[:5]],
    }


def compare(run: Run) -> dict[str, Any]:
    """Realized (TFNE) vs executed (kernel) per semantic class, plus the time grid.

    Edge classes compare multisets of (pre, post, value): weight through the
    kernel's own resolver, mechanism through ``edge_table()``, delay as
    ``round(delay_ms / dt_ms)`` against the kernel's delay steps. Without a
    realization the edge classes are ``NOT_APPLICABLE``.
    """
    sig = run.signals
    t = np.asarray(sig.time_ms, dtype=float)
    n_expected = int(round(run.duration_ms / run.dt_ms))
    dt_exec = float(t[1] - t[0]) if t.shape[0] > 1 else None
    time_ok = t.shape[0] == n_expected and dt_exec is not None and np.isclose(dt_exec, run.dt_ms)
    out: dict[str, Any] = {
        "time": {
            "verdict": "EQUAL" if time_ok else "DIFFERENT",
            "configured": {"duration_ms": run.duration_ms, "dt_ms": run.dt_ms, "n_steps": n_expected},
            "executed": {"dt_ms": dt_exec, "n_steps": int(t.shape[0])},
        }
    }
    if run.realization is None:
        for k in ("weight", "mechanism", "delay"):
            out[k] = {"verdict": "NOT_APPLICABLE", "reason": "run did not start from TFNE"}
        return out
    rw, rm, rd = _realized_edges(run.realization, run.dt_ms)
    ew, em, ed = _executed_edges(run.model)
    out["weight"] = _class_verdict(rw, ew)
    out["mechanism"] = _class_verdict(rm, em)
    out["delay"] = _class_verdict(rd, ed)
    return out


def observe(run: Run, quantity: str) -> Observation:
    """Return a recorded readout (aliases as ``Signals.get``); refuse what was not recorded."""
    from ._signals import _SIGNALS_GET_KEY_ALIASES

    key = _SIGNALS_GET_KEY_ALIASES.get(quantity)
    if key is None or key not in _LEVELS:
        raise KeyError(f"unknown observable {quantity!r}; want one of {sorted(_LEVELS)}")
    if key not in _recorded(run.signals):
        raise ValueError(f"{key} was not recorded in this run (recorded: {_recorded(run.signals)})")
    values = np.asarray(run.signals.get(key))
    return Observation(quantity=key, values=values, level=_LEVELS[key], shape=tuple(values.shape))


def _prop_finite(run: Run) -> tuple[bool, dict[str, Any]]:
    bad = {k: int(np.sum(~np.isfinite(np.asarray(observe(run, k).values, dtype=float))))
           for k in _recorded(run.signals)}
    return not any(bad.values()), {"non_finite_counts": bad}


def _prop_class(name: str) -> Callable[[Run], tuple[bool, dict[str, Any]]]:
    def check(run: Run) -> tuple[bool, dict[str, Any]]:
        c = compare(run)[name]
        if c["verdict"] == "NOT_APPLICABLE":
            raise ValueError(f"{name}_identity needs a TFNE run: {c['reason']}")
        return c["verdict"] == "EQUAL", c

    return check


PROPERTIES: dict[str, Callable[[Run], tuple[bool, dict[str, Any]]]] = {
    "finite": _prop_finite,
    "time_identity": _prop_class("time"),
    "weight_identity": _prop_class("weight"),
    "mechanism_identity": _prop_class("mechanism"),
    "delay_identity": _prop_class("delay"),
}


def verify(run: Run, prop: str) -> dict[str, Any]:
    """Check one named property; unknown properties are refused, never approximated."""
    if prop not in PROPERTIES:
        raise KeyError(f"unknown property {prop!r}; want one of {sorted(PROPERTIES)}")
    ok, evidence = PROPERTIES[prop](run)
    return {"property": prop, "verdict": "PASS" if ok else "FAIL", "evidence": evidence}
