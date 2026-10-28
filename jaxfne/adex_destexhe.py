"""Destexhe 2009 aeIF cell types for the AdEx conductance route.

Reference: Destexhe, J Comput Neurosci 27:493-506 (2009),
doi:10.1007/s10827-009-0164-4, Eqs. 1-2 and section 2.2; parameter values as
recorded in ``artifacts/etudes/papers/destexhe2009/SOURCE_P1.md`` section 1. Read as
data only; nothing is imported from the etude.

The paper's equations use specific units (``Cm = 1 uF/cm^2``,
``gL = 0.05 mS/cm^2``, ``S = 20000 um^2`` membrane area, ``a`` in uS,
``b`` in nA). This module carries the absolute-current reading used by the
port (SOURCE_P1.md section 1): divide Eq. 1 by ``S`` (``S = 2e-4 cm^2``), so
``C = Cm * S = 200 pF``, ``gL = 10 nS``, ``a[nS] = a[uS] * 1000``,
``b[pA] = b[nA] * 1000``, ``w`` in pA. Units are jaxfne's (mV, ms, pF, nS, pA).

``construct_adex_model`` (``jaxfne/_construct_adex.py``) builds
``AdExParams`` from the CAMINOS E/PV/SOM table with no overrides, so the
Destexhe RS/FS/LTS/TC/RE values are inexpressible through the
``Configuration`` alone. This module adds them without touching the existing
route: :func:`destexhe_overrides` builds the ``overrides`` mapping that
:func:`jaxfne.emitters_adex.make_adex_params` already accepts, and
:func:`apply_destexhe_params` returns a copy of an ``adex``-family
:class:`Model` whose emitter parameters carry the Destexhe values. Edges,
receptors, delays, drive and the kernel (``adex_kernel_args``,
``adex_external_drive_kwargs``, ``simulate_adex``) are untouched, so the
model still runs exactly as ``artifacts/etudes/caminos/caminos_api.py`` runs
it. The carrier labels (``E``/``PV``/``SOM``) stay as built; only the eleven
per-neuron parameter fields change. The time-gated kick rides the core's
``ext_t_start_steps``/``ext_t_stop_steps`` drive window
(``t_start_ms``/``t_stop_ms`` on ``Simulation(poisson_drive=...)``); the
wiring uses ``sampling="capped_bernoulli"`` (the authors' per-cell cap).

Assumptions carried from the source record: FS ``a`` is unstated in the paper
and taken as 0.001 uS (gap G1); RE ``a``/``b`` are read with the printed units
swapped (gap G2); ``Delta``/``VT`` are uniform across types (gap G7).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from .emitters_adex import AdExParams, make_adex_params

# Shared aeIF geometry (section 1): C = 200 pF, gL = 10 nS, EL = -60 mV,
# VT = -50 mV, Delta = 2.5 mV, Vreset = -60 mV, tauw = 600 ms, Tref = 2.5 ms,
# Vpeak = -50 mV (authors' detection VTR, demo_cx05_N500b_LTS.py:86,152-153).
_SHARED: dict[str, float] = {
    "C_pF": 200.0,
    "gL_nS": 10.0,
    "EL_mV": -60.0,
    "VT_mV": -50.0,
    "Delta_mV": 2.5,
    "Vreset_mV": -60.0,
    "tauw_ms": 600.0,
    "Tref_ms": 2.5,
    "Vpeak_mV": -50.0,
}

# Per-type (a, b) in (nS, pA): RS Fig. 1A, RSweak Fig. 1B, FS Fig. 1C (a
# assumed, gap G1), LTS Fig. 1D, TC Fig. 1E, RE Fig. 1F (gap G2).
_AB: dict[str, tuple[float, float]] = {
    "RS": (1.0, 40.0),
    "RSweak": (1.0, 5.0),
    "FS": (1.0, 0.0),
    "LTS": (20.0, 0.0),
    "TC": (40.0, 0.0),
    "RE": (80.0, 30.0),
}

DESTEXHE_CELL_DEFAULTS: dict[str, dict[str, float]] = {
    name: {**_SHARED, "a_nS": a, "b_pA": b} for name, (a, b) in _AB.items()
}


def destexhe_overrides(mapping: Mapping[str, str]) -> dict[str, dict[str, float]]:
    """Build the ``make_adex_params`` overrides for a carrier-to-type mapping.

    ``mapping`` maps each carrier label of the built model (``E``/``PV``/``SOM``)
    to a Destexhe cell type (``RS``, ``RSweak``, ``FS``, ``LTS``, ``TC``,
    ``RE``). Unknown Destexhe types and non-string entries raise
    ``ValueError``. Carriers absent from the model are refused later by
    :func:`apply_destexhe_params`, which sees the model's labels.
    """
    mapping = dict(mapping)
    if not mapping:
        raise ValueError("mapping must name at least one carrier cell type")
    overrides: dict[str, dict[str, float]] = {}
    for carrier, dtype in mapping.items():
        if not isinstance(carrier, str) or not isinstance(dtype, str):
            raise ValueError(f"mapping must hold str -> str, got {carrier!r} -> {dtype!r}")
        if dtype not in DESTEXHE_CELL_DEFAULTS:
            raise ValueError(
                f"unknown Destexhe cell type {dtype!r}; "
                f"expected one of {sorted(DESTEXHE_CELL_DEFAULTS)}"
            )
        overrides[carrier] = dict(DESTEXHE_CELL_DEFAULTS[dtype])
    return overrides


def apply_destexhe_params(model: Any, mapping: Mapping[str, str]) -> Any:
    """Return a copy of an ``adex``-family model with Destexhe cell parameters.

    ``model`` is built by ``construct`` on emitter ``family="adex"``; its
    ``params["emitter"]`` is an :class:`AdExParams` whose labels are the carrier
    types. ``mapping`` is passed through :func:`destexhe_overrides`, then every
    carrier it names must be a label of the model, else ``ValueError``. The
    returned model shares ``cfg`` and ``static`` (edges, receptors, delays) and
    carries the rebuilt emitter parameters at the model's own dtype, so
    ``adex_kernel_args`` and ``simulate`` run it unchanged.
    """
    emitter = model.params.get("emitter") if hasattr(model, "params") else None
    if not isinstance(emitter, AdExParams):
        raise ValueError(
            "apply_destexhe_params needs an adex-family model "
            f"(params['emitter'] is {type(emitter).__name__}, not AdExParams)"
        )
    overrides = destexhe_overrides(mapping)
    labels = tuple(emitter.labels)
    unknown = [c for c in overrides if c not in labels]
    if unknown:
        raise ValueError(
            f"mapping names carrier cell types {unknown} absent from the model "
            f"labels {sorted(set(labels))}"
        )
    present = {c: overrides[c] for c in labels if c in overrides}
    rebuilt = make_adex_params(labels, present)
    dtype = jnp.asarray(emitter.C_pF).dtype
    fields = {
        f: jnp.asarray(getattr(rebuilt, f), dtype=dtype) for f in rebuilt._fields if f != "labels"
    }
    params = dict(model.params)
    params["emitter"] = rebuilt._replace(**fields)
    return dataclasses.replace(model, params=params)


def subset_drive_kwargs(model: Any, sim: Any, fraction: float) -> dict[str, Any]:
    """External-drive kwargs for a seeded fraction of the drive's targets.

    The paper's initiation reaches 2-10% of neurons (SOURCE_P1.md section 3);
    the route's :func:`adex_external_drive_kwargs` addresses whole cell types,
    so the port's kick reaches every RS cell. This keeps that function's
    validation and rate/weight/receptor/delay/key exactly (plus its
    ``ext_t_*`` drive-window bounds, when the sim declares them), then keeps
    a seeded ``fraction`` (0 < fraction <= 1) of its target cells, drawn with
    ``numpy.random.default_rng(seed)`` on the drive's own seed. ``fraction``
    outside (0, 1] and a missing drive are refused.
    """
    import math

    from ._construct_adex import adex_external_drive_kwargs
    from .inputs import refuse_poisson_drive_list

    refuse_poisson_drive_list(
        getattr(sim, "poisson_drive", None), "subset_drive_kwargs (adex_destexhe, emitter family 'adex')")
    try:
        frac = float(fraction)
    except (TypeError, ValueError):
        raise ValueError(f"fraction must be a number in (0, 1]; got {fraction!r}")
    if isinstance(fraction, bool) or not math.isfinite(frac) or not 0.0 < frac <= 1.0:
        raise ValueError(f"fraction must be a number in (0, 1]; got {fraction!r}")
    base = adex_external_drive_kwargs(model, sim)
    if not base:
        raise ValueError(
            "fraction needs a Poisson drive on sim "
            f"(sim.poisson_drive={getattr(sim, 'poisson_drive', None)!r}); got none"
        )
    if "ext_target_groups" in base:
        raise ValueError(
            "fraction is not realized for a grouped Poisson drive "
            f"(sim.poisson_drive={dict(sim.poisson_drive)!r} names "
            f"{len(sim.poisson_drive['rate_hz'])} rates): subset_drive_kwargs keeps "
            "one seeded fraction of the drive's targets, which has no grouped meaning"
        )
    targets = np.asarray(base["ext_targets"])
    seed = sim.poisson_drive["seed"]
    rng = np.random.default_rng(int(seed))
    k = max(1, int(math.floor(frac * targets.size + 0.5)))
    chosen = np.sort(rng.choice(targets, size=k, replace=False).astype(np.int64))
    out = dict(base)
    out["ext_targets"] = chosen
    return out
