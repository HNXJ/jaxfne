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
from .emitters_hh import (
    HH_GRADED_S0_DEFAULT,
    HH_GRADED_SLOPE_DEFAULT,
    HH_GRADED_VTH_DEFAULT,
    HHMorphology,
    _check_comp_range,
    _check_morphology,
    _comp_index_array,
    make_hh_params,
)

# Receptor parameter keys the HH route reads from mechanisms(params=...).
# ``tau_ms``/``reversal_mV`` are the shared names the compiler reads;
# ``tau_r_ms``/``slope_mV``/``v_th_mV``/``s0`` are HH-graded-synapse extras.
# Fallback values (unchanged): slope 5.0 mV and Vth -20.0 mV are the Fig. 6
# run values shared by every graded class, s0 0.1 the connected-class value
# (PORT_SPEC.md section 4); tauR 0.5 ms is the GABAa reference (AMPA 0.2,
# GABAb 10.0: receptors needing another tauR must declare tau_r_ms).
_HH_RECEPTOR_DEFAULTS = {
    "tau_r_ms": 0.5,
    "slope_mV": HH_GRADED_SLOPE_DEFAULT,
    "v_th_mV": HH_GRADED_VTH_DEFAULT,
    "s0": HH_GRADED_S0_DEFAULT,
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


def construct_hh_model(cfg: Any, morphology: Any = None, *,
                       edge_pre_comp=None, edge_post_comp=None,
                       stim_comp=None, record_comp=None,
                       weight_unit: str = "density", gate_init=None) -> Model:
    """Build a :class:`Model` for the HH graded-synapse kernel (see module docstring).

    The edges live in ``static["hh"]`` (``params["edge_list"]`` is None: the hh
    route runs its own kernel, :func:`jaxfne.emitters_hh.simulate_hh`).

    ``morphology`` is None (the default: one compartment per cell, the kernel
    runs its single-compartment path bit-identically to before) or an
    :class:`jaxfne.emitters_hh.HHMorphology` shared by every cell, with
    per-edge ``edge_pre_comp``/``edge_post_comp`` (presynaptic-V and synapse
    compartments, default 0) and per-cell ``stim_comp``/``record_comp``
    (stimulus and spike-readout compartments, default 0). Any compartment
    selection without ``morphology``, any unknown ``weight_unit``, or any
    out-of-range compartment is refused. ``weight_unit`` is ``"density"``
    (the default: edge weights are conductances in mS/cm2) or ``"uS"``
    (point-process conductances in uS, converted by the target compartment
    area at simulate time; needs ``morphology``). ``gate_init`` is None (the
    default: single-compartment gates start at their rate fixed point,
    multi-compartment gates at the reference default 0.2) or an explicit
    gating value forwarded as the kernel ``gate_init`` argument. The weight
    unit, gate init, morphology and compartment selections are stored in
    ``static["hh"]`` and forwarded by :func:`hh_kernel_args`; point-process
    inputs convert by the target compartment area at simulate time.
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
    if len(set(receptor_params["s0"])) > 1:
        raise ValueError(
            f"receptors declare mixed s0={receptor_params['s0']!r}; the kernel takes one scalar receptor_s0"
        )
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
    if weight_unit not in ("density", "uS"):
        raise ValueError(
            f"weight_unit must be 'density' (mS/cm2) or 'uS' (point process); "
            f"got {weight_unit!r}"
        )
    if weight_unit == "uS" and morphology is None:
        raise ValueError(
            "weight_unit='uS' needs morphology=... (an HHMorphology): "
            "point-process conversion needs compartment areas"
        )
    if gate_init is None:
        gate_init_f = None
    else:
        try:
            gate_init_f = float(gate_init)
        except (TypeError, ValueError):
            raise ValueError(
                f"gate_init must be a finite number or None; got {gate_init!r}"
            )
        if not math.isfinite(gate_init_f):
            raise ValueError(
                f"gate_init must be a finite number or None; got {gate_init!r}"
            )
    if morphology is None:
        for name, given in (("edge_pre_comp", edge_pre_comp),
                            ("edge_post_comp", edge_post_comp),
                            ("stim_comp", stim_comp),
                            ("record_comp", record_comp)):
            if given is not None:
                raise ValueError(
                    f"{name} needs morphology=... (an HHMorphology); got {given!r}"
                )
        morph_block: dict[str, Any] = {}
    else:
        mx = _check_morphology(morphology)
        n_comp = int(mx["n_comp"])
        n_e = int(np.asarray(result.edge_pre).shape[0])
        pre_c = _comp_index_array(edge_pre_comp, n_e, "edge_pre_comp")
        _check_comp_range(pre_c, n_comp, "edge_pre_comp")
        post_c = _comp_index_array(edge_post_comp, n_e, "edge_post_comp")
        _check_comp_range(post_c, n_comp, "edge_post_comp")
        stim_c = _comp_index_array(stim_comp, n, "stim_comp")
        _check_comp_range(stim_c, n_comp, "stim_comp")
        rec_c = _comp_index_array(record_comp, n, "record_comp")
        _check_comp_range(rec_c, n_comp, "record_comp")
        morph_block = {
            "morphology": {
                "parents": tuple(int(p) for p in morphology.parents),
                "length_um": tuple(float(v) for v in morphology.length_um),
                "radius_um": tuple(float(v) for v in morphology.radius_um),
                "ra_ohmcm": float(morphology.ra_ohmcm),
            },
            "edge_pre_comp": np.asarray(pre_c, dtype=np.int64),
            "edge_post_comp": np.asarray(post_c, dtype=np.int64),
            "stim_comp": np.asarray(stim_c, dtype=np.int64),
            "record_comp": np.asarray(rec_c, dtype=np.int64),
        }
    static["hh"] = {
        "pre": np.asarray(result.edge_pre, dtype=np.int64),
        "post": np.asarray(result.edge_post, dtype=np.int64),
        "weight": np.asarray(result.edge_weight, dtype=np.float64),
        "weight_unit": weight_unit,
        "gate_init": gate_init_f,
        "receptor": np.asarray(result.edge_mechanism, dtype=np.int64),
        "network": None if network_name is None else str(network_name),
        "receptor_names": tuple(receptor_names),
        "receptor_params": {k: tuple(v) for k, v in receptor_params.items()},
        "labels": tuple(labels),
        "edge_seed": int(edge_seed),
        **morph_block,
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
    ``V0_mV`` and ``record_v``. When the model was constructed with a
    morphology, the morphology and its compartment selections are forwarded
    (rebuilt as :class:`jaxfne.emitters_hh.HHMorphology`), with the weight
    array under ``edge_weight_uS`` for ``weight_unit="uS"`` and under
    ``edge_weight`` otherwise, plus ``gate_init`` when set; without a
    morphology the returned dict is the single-compartment form (plus
    ``gate_init`` only when set).
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
    wu = st.get("weight_unit", "density")
    if wu not in ("density", "uS"):
        raise ValueError(
            f"unknown stored weight_unit {wu!r}; expected 'density' or 'uS'"
        )
    out: dict[str, Any] = {
        "params": model.params["emitter"],
        "edge_pre": st["pre"],
        "edge_post": st["post"],
        "edge_receptor": st["receptor"],
        "receptor_e_rev": rp["reversal_mV"],
        "receptor_tau_r": rp["tau_r_ms"],
        "receptor_tau_d": rp["tau_ms"],
        "receptor_slope": rp["slope_mV"],
        "receptor_vth": rp["v_th_mV"],
        "receptor_s0": float(rp["s0"][0]),
    }
    out["edge_weight_uS" if wu == "uS" else "edge_weight"] = st["weight"]
    if st.get("gate_init") is not None:
        out["gate_init"] = float(st["gate_init"])
    if "morphology" in st:
        m = st["morphology"]
        out["morphology"] = HHMorphology(
            parents=tuple(m["parents"]), length_um=tuple(m["length_um"]),
            radius_um=tuple(m["radius_um"]), ra_ohmcm=float(m["ra_ohmcm"]),
        )
        out["edge_pre_comp"] = st["edge_pre_comp"]
        out["edge_post_comp"] = st["edge_post_comp"]
        out["stim_comp"] = st["stim_comp"]
        out["record_comp"] = st["record_comp"]
    return out
