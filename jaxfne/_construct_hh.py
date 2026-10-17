"""Standalone ``construct_hh_model()`` for the ``emitters_hh`` kernel (GSDR lane).

The HH route is deliberately NOT wired into :func:`jaxfne.core.construct`
dispatch (that would change an existing public function): the etude calls
:func:`construct_hh_model` directly, the CAMINOS route-(c) precedent
(``artifacts/etudes/caminos/caminos_api.py``).

``Configuration`` wires the circuit only: ``network(n=, cell_types=)`` sets
the population, ``mechanisms()`` declares the graded-synapse receptors, and
``connections()`` rules compile to edges via
:func:`jaxfne.connectivity.compile_connection_rules` with
``allow_self_connections=True`` (the reference includes self-edges: E->all
covers all 50 cells). The kernel has no delay ring, so any ``delay_ms`` set
away from None is refused. Edge weights are conductances (mS/cm2, >= 0);
the reversal sets the sign.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._config import _counts_from_fractions, edge_seed_from_metadata
from ._model import Model
from .connectivity import compile_connection_rules
from .emitters_hh import make_hh_params

# Receptor parameter keys the HH route reads from mechanisms(params=...).
# ``tau_ms``/``reversal_mV`` are the shared names the compiler reads;
# ``tau_r_ms``/``slope_mV``/``v_th_mV``/``s0`` are HH-graded-synapse extras.
_HH_RECEPTOR_DEFAULTS = {
    "tau_r_ms": 0.5,
    "slope_mV": 5.0,
    "v_th_mV": -20.0,
    "s0": 0.1,
}


def _finite_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _hh_population(cfg: Any) -> tuple[int, tuple[str, ...]]:
    """``(n, labels)``: labels in the cell_types dict's order."""
    net = cfg.networks[0] if cfg.networks else {}
    if net.get("n") is None:
        raise ValueError("network(n=...) is required for the hh route: it sets the population size")
    if net.get("cell_types") is None:
        raise ValueError(
            "network(cell_types=...) is required for the hh route: it maps each cell type to "
            "its fraction (types: E, Ig, Il)"
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


def _hh_mechanisms(cfg: Any) -> tuple[list[str], dict[str, tuple[float, ...]]]:
    """Receptor names and per-receptor parameter tuples, in declaration order."""
    names: list[str] = []
    cols: dict[str, list[float]] = {
        "tau_ms": [], "reversal_mV": [], "tau_r_ms": [], "slope_mV": [], "v_th_mV": [], "s0": [],
    }
    for mech in (cfg.metadata.get("circuit") or {}).get("mechanisms", []):
        name = mech.get("name")
        if mech.get("kind") != "receptor":
            raise ValueError(
                f"mechanism {name!r} has kind={mech.get('kind')!r}; the hh route realizes "
                "kind 'receptor' only"
            )
        params = dict(mech.get("params") or {})
        tau_ms = params.get("tau_ms")
        rev_mV = params.get("reversal_mV")
        if not (_finite_real(tau_ms) and tau_ms > 0) or not _finite_real(rev_mV):
            raise ValueError(
                f"mechanism {name!r} is not realized on the hh route: it needs params "
                f"tau_ms > 0 and a finite reversal_mV; got {params!r}"
            )
        tau_r = params.get("tau_r_ms", _HH_RECEPTOR_DEFAULTS["tau_r_ms"])
        slope = params.get("slope_mV", _HH_RECEPTOR_DEFAULTS["slope_mV"])
        v_th = params.get("v_th_mV", _HH_RECEPTOR_DEFAULTS["v_th_mV"])
        s0 = params.get("s0", _HH_RECEPTOR_DEFAULTS["s0"])
        if not (_finite_real(tau_r) and tau_r > 0):
            raise ValueError(
                f"mechanism {name!r} needs tau_r_ms > 0; got {tau_r!r}"
            )
        if not (_finite_real(slope) and slope > 0):
            raise ValueError(
                f"mechanism {name!r} needs slope_mV > 0; got {slope!r}"
            )
        if not _finite_real(v_th):
            raise ValueError(
                f"mechanism {name!r} needs a finite v_th_mV; got {v_th!r}"
            )
        if not _finite_real(s0):
            raise ValueError(
                f"mechanism {name!r} needs a finite s0; got {s0!r}"
            )
        names.append(str(name))
        cols["tau_ms"].append(float(tau_ms))
        cols["reversal_mV"].append(float(rev_mV))
        cols["tau_r_ms"].append(float(tau_r))
        cols["slope_mV"].append(float(slope))
        cols["v_th_mV"].append(float(v_th))
        cols["s0"].append(float(s0))
    if not names:
        raise ValueError("the hh route needs at least one mechanisms(kind='receptor') declaration")
    return names, {k: tuple(v) for k, v in cols.items()}


def _hh_rules(cfg: Any, receptor_names: list[str]) -> list[dict[str, Any]]:
    """The connections() rules, each checked for the fields the hh compiler needs."""
    rules = list((cfg.metadata.get("circuit") or {}).get("connections", []))
    if not rules:
        raise ValueError("the hh route needs at least one connections() rule")
    for rule in rules:
        rname = rule.get("name")
        if rule.get("mechanism") is None:
            raise ValueError(
                f"connections(name={rname!r}) has no mechanism: the hh route needs mechanism=... "
                "naming a declared mechanism"
            )
        if rule["mechanism"] not in receptor_names:
            raise ValueError(
                f"connections(name={rname!r}, mechanism={rule['mechanism']!r}) names an undeclared "
                f"mechanism; declared: {receptor_names}"
            )
        if rule.get("weight") is None:
            raise ValueError(f"connections(name={rname!r}) needs weight (mS/cm2) on the hh route")
        if float(rule["weight"]) < 0.0:
            raise ValueError(
                f"connections(name={rname!r}) weight must be >= 0 (the reversal sets the sign); "
                f"got {rule['weight']!r}"
            )
        if rule.get("delay_ms") is not None:
            raise ValueError(
                f"connections(name={rname!r}) delay_ms={rule['delay_ms']!r} is not realized on the "
                "hh route: the graded-synapse kernel has no delay ring"
            )
        if rule.get("max_in_degree") is not None:
            raise ValueError(
                f"connections(name={rname!r}) max_in_degree is not realized on the hh route"
            )
    return rules


def construct_hh_model(cfg: Any) -> Model:
    """Build a :class:`Model` for the HH graded-synapse kernel (see module docstring).

    The edges live in ``static["hh"]`` (``params["edge_list"]`` is None: the hh
    route runs its own kernel, :func:`jaxfne.emitters_hh.simulate_hh`).
    """
    from ._construct_core import _construct_build_static

    if getattr(cfg, "emitters", None):
        raise ValueError(
            "the hh route takes no emitter() declaration: the kernel is fixed to classic HH; "
            f"got {cfg.emitters!r}"
        )
    n, labels = _hh_population(cfg)
    dtype_name = str(cfg.metadata.get("dtype", "float32"))
    dtype = jnp.dtype(dtype_name)
    params = make_hh_params(labels)
    params = params._replace(**{f: jnp.asarray(getattr(params, f), dtype=dtype) for f in params._fields})
    mechanisms = list((cfg.metadata.get("circuit") or {}).get("mechanisms", []))
    receptor_names, receptor_params = _hh_mechanisms(cfg)
    rules = _hh_rules(cfg, receptor_names)
    network_name = cfg.networks[0].get("name") if cfg.networks else None

    rows = [
        {"neuron_id": i, "area": "hh", "layer": "hh", "cell_type": labels[i]} for i in range(n)
    ]
    edge_seed = edge_seed_from_metadata(cfg.metadata)
    result = compile_connection_rules(
        rows, rules, mechanisms, seed=edge_seed,
        allow_self_connections=True, dtype=dtype_name,
    )

    static = _construct_build_static(cfg, geometry_meta=None)
    static["neuron_metadata"] = rows
    static["neuron_metadata_summary"] = {
        "n_rows": n,
        "areas": ["hh"],
        "layers": ["hh"],
        "cell_types": sorted(set(labels)),
    }
    static["hh"] = {
        "pre": np.asarray(result.edge_pre, dtype=np.int64),
        "post": np.asarray(result.edge_post, dtype=np.int64),
        "weight": np.asarray(result.edge_weight, dtype=np.float64),
        "receptor": np.asarray(result.edge_mechanism, dtype=np.int64),
        "network": None if network_name is None else str(network_name),
        "receptor_names": tuple(receptor_names),
        "receptor_params": {k: tuple(v) for k, v in receptor_params.items()},
        "labels": tuple(labels),
        "edge_seed": int(edge_seed),
    }
    return Model(
        cfg=cfg,
        params={
            "emitter": params,
            "positions": jnp.zeros((n, 3), dtype=dtype),
            "edge_list": None,
        },
        static=static,
    )


def hh_kernel_args(model: Model, dt_ms: float) -> dict[str, Any]:
    """Keyword arguments for :func:`jaxfne.emitters_hh.simulate_hh` on ``model``.

    Returns ``params``, the four ``edge_*`` arrays and the ``receptor_*``
    tuples. The caller adds ``n_steps``, ``dt_ms``, ``ou_*``, ``I_ext``,
    ``V0_mV`` and ``record_v``.
    """
    dt = float(dt_ms)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt_ms must be finite and > 0; got {dt_ms!r}")
    st = model.static["hh"]
    rp = st["receptor_params"]
    if len(set(rp["s0"])) > 1:
        raise ValueError(
            f"receptors declare mixed s0={rp['s0']!r}; the kernel takes one scalar receptor_s0"
        )
    return {
        "params": model.params["emitter"],
        "edge_pre": st["pre"],
        "edge_post": st["post"],
        "edge_weight": st["weight"],
        "edge_receptor": st["receptor"],
        "receptor_e_rev": rp["reversal_mV"],
        "receptor_tau_r": rp["tau_r_ms"],
        "receptor_tau_d": rp["tau_ms"],
        "receptor_slope": rp["slope_mV"],
        "receptor_vth": rp["v_th_mV"],
        "receptor_s0": float(rp["s0"][0]),
    }
