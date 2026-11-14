"""AdEx network kernel: conductance synapses and a per-edge delay ring.

Reference: Tahvili, Vinck, di Volo, PLoS Comput Biol 2026 (DOI 10.1371/journal.pcbi.1014378),
Methods Eqs. 15-18 and Tables 1-2. ``adex_step`` mirrors the CAMINOS etude's ``adex_step``
(``artifacts/etudes/papers/caminos/caminos_jax.py``, reference branch ``l2-caminos``); the cell table
mirrors ``PORT_SPEC.md`` Table 1. Nothing is imported from the etude.

Units are jaxfne's: mV, ms, pF, nS, pA (nS * mV = pA; pA / pF = mV / ms).

Function-level only. Configuration, construct and simulate wiring is not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_STEP_ORDERS = ("default", "brian2")
_DECAYS = ("exp", "euler")
# State keys `simulate_adex_hdp_registered` writes under `return_state=True`
# and reads back under `state=...`.
_HDP_STATE_KEYS = (
    "V_mV", "w_pA", "ref_steps", "g_nS", "delay_ring", "ext_ring", "step",
    "ext_key", "hdp_key", "H", "W_nS", "aux", "bias", "prev_spikes",
    "n_neurons", "n_edges", "n_receptors", "ring_length", "n_ext_targets",
)
# Same clip as the reference: the spike threshold binds at an argument of 5.
_EXP_ARG_MAX = 20.0

# Tahvili, Vinck, di Volo 2026, PLoS Comput Biol, Table 1 (values as in PORT_SPEC.md).
ADEX_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "E": {
        "C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -65.0, "VT_mV": -50.0, "Delta_mV": 2.0,
        "Vreset_mV": -65.0, "tauw_ms": 500.0, "a_nS": 4.0, "b_pA": 50.0, "Tref_ms": 5.0,
    },
    "PV": {
        "C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -65.0, "VT_mV": -50.0, "Delta_mV": 0.5,
        "Vreset_mV": -65.0, "tauw_ms": 500.0, "a_nS": 0.0, "b_pA": 0.0, "Tref_ms": 5.0,
    },
    "SOM": {
        "C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -55.0, "VT_mV": -50.0, "Delta_mV": 2.0,
        "Vreset_mV": -65.0, "tauw_ms": 500.0, "a_nS": 4.0, "b_pA": 80.0, "Tref_ms": 5.0,
    },
}

_PARAM_FIELDS = (
    "C_pF", "gL_nS", "EL_mV", "VT_mV", "Delta_mV", "Vreset_mV", "tauw_ms", "a_nS", "b_pA", "Tref_ms",
    "Vpeak_mV",
)


class AdExParams(NamedTuple):
    """Per-neuron AdEx parameters: float arrays of shape [N], jaxfne units."""

    C_pF: jax.Array
    gL_nS: jax.Array
    EL_mV: jax.Array
    VT_mV: jax.Array
    Delta_mV: jax.Array
    Vreset_mV: jax.Array
    tauw_ms: jax.Array
    a_nS: jax.Array
    b_pA: jax.Array
    Tref_ms: jax.Array
    Vpeak_mV: jax.Array
    labels: tuple[str, ...]


def make_adex_params(
    labels: Sequence[str],
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> AdExParams:
    """Build :class:`AdExParams` for one cell type per neuron.

    Defaults: Tahvili, Vinck & di Volo (2026), Table 1, doi:10.1371/journal.pcbi.1014378.

    ``overrides`` maps a cell type to a dict of field overrides (field names as in
    :class:`AdExParams`). Unknown cell types, unknown fields, and overrides for a cell
    type absent from ``labels`` raise ``ValueError``.

    ``Vpeak_mV`` is the per-type spike-detection threshold (NaN = absent = the
    reference ``VT + 5 * Delta``). An explicit value must be finite and above
    that type's ``EL`` (bools refused, never coerced); absent, every entry is
    NaN and the kernel runs the reference threshold bit-identically.

    Every override must also keep the kernel integrable: ``C_pF``, ``gL_nS``
    and ``tauw_ms`` above 0, ``Tref_ms`` and ``Delta_mV`` at or above 0
    (``Delta_mV`` 0 runs the conductance-LIF limit). An ``EL_mV`` above
    ``VT_mV`` is a valid tonically firing regime and is realized.
    """
    labels = tuple(str(lab) for lab in labels)
    for lab in labels:
        if lab not in ADEX_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"unknown AdEx cell type {lab!r}; expected one of {sorted(ADEX_CELL_TYPE_DEFAULTS)}"
            )
    overrides = {} if overrides is None else overrides
    for ctype, fields in overrides.items():
        if ctype not in ADEX_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"overrides name unknown AdEx cell type {ctype!r}; "
                f"expected one of {sorted(ADEX_CELL_TYPE_DEFAULTS)}"
            )
        if ctype not in labels:
            raise ValueError(f"overrides name cell type {ctype!r}, which is not in labels {labels}")
        for field in fields:
            if field not in _PARAM_FIELDS:
                raise ValueError(
                    f"unknown AdEx parameter field {field!r} for cell type {ctype!r}; "
                    f"expected one of {_PARAM_FIELDS}"
                )
        if "Vpeak_mV" in fields:
            _check_vpeak_override(ctype, fields["Vpeak_mV"], overrides.get(ctype, {}))
        _check_physical_adex_override(ctype, fields)
    columns = {}
    for field in _PARAM_FIELDS:
        if field == "Vpeak_mV":
            values = [
                float(overrides.get(lab, {}).get(field, math.nan)) for lab in labels
            ]
        else:
            values = [
                float(overrides.get(lab, {}).get(field, ADEX_CELL_TYPE_DEFAULTS[lab][field]))
                for lab in labels
            ]
        columns[field] = jnp.asarray(np.asarray(values, dtype=np.float64))
    return AdExParams(labels=labels, **columns)


def _check_vpeak_override(ctype: str, value: Any, type_fields: Mapping[str, float]) -> None:
    """Refuse a non-finite ``Vpeak_mV`` override or one at or below ``EL``.

    ``EL`` is the type's override when one is declared, else its reference
    value; NaN (or any other non-finite value, or a bool) is refused, absence
    is declared by omitting the key, never by an explicit NaN.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"overrides for cell type {ctype!r}: Vpeak_mV must be a finite float "
            f"above EL; got {value!r}"
        )
    try:
        vpeak = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"overrides for cell type {ctype!r}: Vpeak_mV must be a finite float "
            f"above EL; got {value!r}"
        )
    if not math.isfinite(vpeak):
        raise ValueError(
            f"overrides for cell type {ctype!r}: Vpeak_mV must be a finite float "
            f"above EL; got {value!r}"
        )
    el_raw = type_fields.get("EL_mV", ADEX_CELL_TYPE_DEFAULTS[ctype]["EL_mV"])
    if isinstance(el_raw, bool):
        raise ValueError(
            f"overrides for cell type {ctype!r}: Vpeak_mV={vpeak} cannot be checked "
            f"against a non-real EL_mV={el_raw!r}"
        )
    el = float(el_raw)
    if not math.isfinite(el) or vpeak <= el:
        raise ValueError(
            f"overrides for cell type {ctype!r}: Vpeak_mV={vpeak} must be above "
            f"EL_mV={el_raw!r}"
        )


def _check_physical_adex_override(ctype: str, type_fields: Mapping[str, float]) -> None:
    """Refuse an override the kernel cannot integrate.

    ``C_pF``, ``gL_nS`` and ``tauw_ms`` take values above 0; ``Tref_ms`` and
    ``Delta_mV`` take values at or above 0 (``Delta_mV`` 0 runs the
    conductance-LIF limit). An ``EL_mV`` above ``VT_mV`` is a valid
    tonically firing regime and is realized. Non-real values are refused
    here as ``ValueError`` (never coerced); finiteness rides the same refusal.
    """
    defaults = ADEX_CELL_TYPE_DEFAULTS[ctype]

    def _resolved(field: str) -> Any:
        raw = type_fields.get(field, defaults[field])
        if isinstance(raw, bool):
            raise ValueError(
                f"overrides for cell type {ctype!r}: {field}={raw!r} must be "
                "a finite real number"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"overrides for cell type {ctype!r}: {field}={raw!r} must be "
                "a finite real number"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"overrides for cell type {ctype!r}: {field}={raw!r} must be "
                "a finite real number"
            )
        return value

    positive = {"C_pF": "above 0", "gL_nS": "above 0", "tauw_ms": "above 0"}
    for field, bound in positive.items():
        value = _resolved(field)
        if not value > 0:
            raise ValueError(
                f"overrides for cell type {ctype!r}: {field}={value} must be "
                f"{bound} for the kernel to integrate it"
            )
    nonnegative = {"Tref_ms": "at or above 0", "Delta_mV": "at or above 0"}
    for field, bound in nonnegative.items():
        value = _resolved(field)
        if not value >= 0:
            raise ValueError(
                f"overrides for cell type {ctype!r}: {field}={value} must be "
                f"{bound} for the kernel to integrate it"
            )


def adex_step(state, I_pA, params: AdExParams, dt_ms: float, step_order: str = "default"):
    """One explicit-Euler step of the AdEx equations for a vector of cells.

    Cell model: Brette & Gerstner (2005), doi:10.1152/jn.00686.2005; the refractory
    clamp and the spike threshold ``VT + 5 * Delta`` follow Tahvili et al. (2026).

    ``state`` is ``(V, w, ref)``: membrane voltage (mV), adaptation (pA), and refractory
    steps remaining (int). ``I_pA`` is injected current. A cell with ``ref > 0`` is clamped
    at ``Vreset`` while ``w`` keeps evolving with ``V = Vreset``. An active cell spikes when
    its updated V reaches ``Vpeak`` where declared (per-cell ``params.Vpeak_mV``, NaN =
    absent), else the reference ``VT + 5 * Delta``; on spike V is reset, ``w += b``, and
    ``ref`` is set to ``round(Tref / dt)`` (one fewer under ``step_order="brian2"``).

    A per-cell ``Delta_mV`` of exactly ``0.0`` runs the conductance-LIF limit: the
    exponential term vanishes and the (undeclared-``Vpeak``) threshold binds at
    ``VT``. Positive ``Delta_mV`` steps the reference equations bit-identically.

    Returns ``((V, w, ref), spk)`` with boolean spike vector ``spk``. Pure ``jax.numpy``;
    dtype follows the inputs.
    """
    if step_order not in _STEP_ORDERS:
        raise ValueError(f"step_order must be 'default' or 'brian2', got {step_order!r}")
    V, w, ref = state
    p = params
    v_thr = jnp.where(
        jnp.isnan(p.Vpeak_mV), p.VT_mV + 5.0 * p.Delta_mV, p.Vpeak_mV
    )
    # LIF limit: Delta == 0 zeroes the exponential current exactly. The safe
    # denominator keeps the quotient finite under jit; the where() below then
    # selects the exact limit per cell, so mixed populations step each cell
    # under its own model.
    is_lif = p.Delta_mV == 0.0
    safe_delta = jnp.where(is_lif, jnp.ones_like(p.Delta_mV), p.Delta_mV)
    arg = jnp.minimum((V - p.VT_mV) / safe_delta, _EXP_ARG_MAX)
    exp_current = jnp.where(
        is_lif, jnp.zeros_like(arg), p.gL_nS * p.Delta_mV * jnp.exp(arg)
    )
    dV = (
        p.gL_nS * (p.EL_mV - V)
        + exp_current
        - w
        + I_pA
    ) / p.C_pF * dt_ms
    dw_act = (p.a_nS * (V - p.EL_mV) - w) / p.tauw_ms * dt_ms
    dw_ref = (p.a_nS * (p.Vreset_mV - p.EL_mV) - w) / p.tauw_ms * dt_ms
    n_ref = jnp.round(p.Tref_ms / dt_ms).astype(jnp.int32)
    if step_order == "brian2":
        n_ref = n_ref - jnp.int32(1)
    active = ref <= 0
    v_cand = V + dV
    spk = active & (v_cand >= v_thr)
    V_new = jnp.where(spk, p.Vreset_mV, jnp.where(active, v_cand, p.Vreset_mV))
    w_new = jnp.where(active, w + dw_act + jnp.where(spk, p.b_pA, 0.0), w + dw_ref)
    ref_new = jnp.where(spk, n_ref, jnp.maximum(ref - 1, 0))
    return (V_new, w_new, ref_new), spk


_EDGE_ARGS = (
    "edge_pre", "edge_post", "edge_weight_nS", "edge_receptor", "edge_delay_steps",
)

_EDGE0_ARGS = (
    "edge0_pre", "edge0_post", "edge0_weight_nS", "edge0_receptor",
)

_EXT_ARGS = (
    "ext_targets", "ext_lambda", "ext_weight_nS", "ext_receptor", "ext_delay_steps", "key",
)

# Multi-target external drive (one adex drive list entry per target subset):
# per-target mean counts, weights, receptors and delays, plus one int entry
# size per entry (entries hold contiguous target blocks summing to ``M``).
# ``extm_lambda`` is a scalar (shared by every target), a 1-d ``[M]``
# per-target vector, or a 2-d ``[n_steps, M]`` per-target schedule;
# ``extm_weight_nS`` is a scalar or ``[M]``; ``extm_receptor``/``extm_delay_steps``
# are ``[M]`` integer arrays. ``key`` holds one JAX PRNG key per entry with
# shape ``(n_entries, 2)`` (each entry's mapping-path seed through
# ``jax.random.PRNGKey``); entry ``e`` owns ``extm_entry_sizes[e]`` targets.
_EXTM_ARGS = (
    "extm_targets", "extm_lambda", "extm_weight_nS", "extm_receptor",
    "extm_delay_steps", "extm_entry_sizes", "key",
)


def _nonneg_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; got {value!r}")
    if isinstance(value, bool) or not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite float >= 0; got {value!r}")
    return out


def _checked_lambda(value):
    """Validate ``ext_lambda``: a scalar mean or a per-step schedule array.

    A scalar (or 0-d) value takes the historical path: a finite float >= 0.
    Anything else must be a 1-d (``[n_steps]``) or 2-d (``[n_steps, M]``)
    array of finite values >= 0 (bools refused, never coerced); its shape is
    checked against the run in :func:`simulate_adex` and
    :func:`simulate_adex_hdp_registered`, where ``n_steps`` and ``M`` are
    known. Returns ``("scalar", lam)`` or ``("schedule", array)``.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ext_lambda must be a finite float >= 0, a 1-d [n_steps] array, or a "
            f"2-d [n_steps, M] array; got {value!r}"
        ) from exc
    if arr.ndim == 0:
        return ("scalar", _nonneg_float("ext_lambda", value))
    if np.asarray(value).dtype == np.bool_ or arr.ndim not in (1, 2):
        raise ValueError(
            "ext_lambda must be a finite float >= 0, a 1-d [n_steps] array, or a "
            f"2-d [n_steps, M] array; got {value!r}"
        )
    if arr.shape[0] == 0 or (arr.ndim == 2 and arr.shape[1] == 0):
        raise ValueError(f"ext_lambda schedule must be non-empty; got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr < 0.0)):
        raise ValueError("ext_lambda schedule must hold finite values >= 0")
    return ("schedule", arr)


def _check_target_groups(groups, M, n_steps, lam):
    """Validate ``ext_target_groups`` against the drive; return a spec dict, or None."""
    if groups is None:
        return None
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape[0] != M:
        raise ValueError(
            "ext_target_groups must hold one int group per target "
            f"(shape ({M},)); got shape {g.shape}"
        )
    gf = g.astype(np.float64)
    if not np.all(np.isfinite(gf)) or np.any(gf != np.round(gf)) or np.any(gf < 0):
        raise ValueError("ext_target_groups must hold integers >= 0")
    kind, val = lam
    if kind != "schedule" or np.asarray(val).ndim != 2:
        raise ValueError(
            "ext_target_groups needs a 2-d [n_steps, G] ext_lambda schedule "
            f"(one rate per group per step); got {kind}"
        )
    sched = np.asarray(val, dtype=np.float64)
    if sched.shape[0] != n_steps:
        raise ValueError(
            "ext_lambda schedule with per-group rates must have shape "
            f"({n_steps}, G); got {sched.shape}"
        )
    G = int(sched.shape[1])
    gi = gf.astype(np.int64)
    if gi.max() >= G:
        raise ValueError(
            f"ext_target_groups names group {int(gi.max())} but the ext_lambda "
            f"schedule holds {G} groups"
        )
    sizes = [int(np.sum(gi == c)) for c in range(G)]
    if any(s == 0 for s in sizes):
        raise ValueError(
            f"ext_target_groups leaves a group empty (sizes {sizes}): "
            "every group needs at least one target cell"
        )
    order = np.argsort(gi, kind="stable")
    unorder = np.argsort(order, kind="stable")
    return {"G": G, "sizes": sizes, "order": order, "unorder": unorder}


def _check_drive_window(n_steps, t_start_steps, t_stop_steps) -> "tuple[int, int] | None":
    """Validate the kernel's optional drive-window step bounds.

    Both bounds come together or not at all; absent, the drive runs every
    step (the historical path). Present, they must be integers with
    ``0 <= t_start < t_stop <= n_steps``: the schedule carries the scalar
    mean on steps ``[t_start, t_stop)`` and 0 elsewhere, drawn under the same
    key stream, so windowed steps reproduce the scalar run bit-for-bit.
    """
    if t_start_steps is None and t_stop_steps is None:
        return None
    for name, bound in (("ext_t_start_steps", t_start_steps), ("ext_t_stop_steps", t_stop_steps)):
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, np.integer))
        ):
            raise ValueError(
                f"{name} must be an int step index or None (with its pair); got {bound!r}"
            )
    t0, t1 = int(t_start_steps), int(t_stop_steps)
    if not (0 <= t0 < t1 <= int(n_steps)):
        raise ValueError(
            f"ext_t_start_steps={t0} and ext_t_stop_steps={t1} must satisfy "
            f"0 <= t_start < t_stop <= n_steps ({int(n_steps)})"
        )
    return (t0, t1)


def _check_external_drive(N, R, given, allow_zero_delay: bool = False):
    """Validate the six external-drive arguments; return a dict, or None when all are absent."""
    missing = [k for k in _EXT_ARGS if given[k] is None]
    if missing and len(missing) < len(_EXT_ARGS):
        raise ValueError(
            "external drive arguments (ext_targets, ext_lambda, ext_weight_nS, ext_receptor, "
            f"ext_delay_steps, key) must be all given or all None; missing {missing}"
        )
    if missing:
        return None
    targets = np.asarray(given["ext_targets"])
    if targets.ndim != 1:
        raise ValueError(f"ext_targets must be one-dimensional, got shape {targets.shape}")
    t64 = targets.astype(np.float64)
    if not np.all(np.isfinite(t64)) or np.any(t64 != np.round(t64)):
        raise ValueError("ext_targets must hold integers")
    t64 = t64.astype(np.int64)
    if t64.size == 0:
        raise ValueError("ext_targets must hold at least one cell index")
    if t64.min() < 0 or t64.max() >= N:
        raise ValueError(
            f"ext_targets must index cells in [0, {N}); got min {t64.min()}, max {t64.max()}"
        )
    if np.unique(t64).size != t64.size:
        raise ValueError(f"ext_targets must hold unique cell indices; got {t64.tolist()}")

    def _nonneg_float(name, value):
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number; got {value!r}")
        if isinstance(value, bool) or not math.isfinite(out) or out < 0.0:
            raise ValueError(f"{name} must be a finite float >= 0; got {value!r}")
        return out

    lam = _checked_lambda(given["ext_lambda"])
    weight = _nonneg_float("ext_weight_nS", given["ext_weight_nS"])
    rec = given["ext_receptor"]
    if (
        isinstance(rec, bool)
        or not isinstance(rec, (int, np.integer, float, np.floating))
        or not math.isfinite(float(rec))
        or float(rec) != round(float(rec))
        or int(rec) < 0
        or int(rec) >= R
    ):
        raise ValueError(f"ext_receptor must index receptors in [0, {R}); got {rec!r}")
    delay = given["ext_delay_steps"]
    lo = 0 if allow_zero_delay else 1
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, np.integer, float, np.floating))
        or not math.isfinite(float(delay))
        or float(delay) != round(float(delay))
        or int(delay) < lo
    ):
        raise ValueError(f"ext_delay_steps must be an integer >= {lo}; got {delay!r}")
    try:
        key = given["key"]
        jax.random.split(key)
    except Exception:
        raise ValueError(f"key must be a JAX PRNG key; got {given['key']!r}")
    return {
        "targets": t64,
        "lambda": lam,
        "weight": weight,
        "receptor": int(rec),
        "delay": int(delay),
        "key": key,
    }


def _checked_lambda_multi(value, M):
    """Validate ``extm_lambda``: scalar, ``[M]`` or ``[n_steps, M]`` (shape-checked later)."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extm_lambda must be a finite float >= 0, a 1-d [M] array, or a "
            f"2-d [n_steps, M] array; got {value!r}"
        ) from exc
    if arr.ndim == 0:
        return ("scalar", _nonneg_float("extm_lambda", value))
    if np.asarray(value).dtype == np.bool_ or arr.ndim not in (1, 2):
        raise ValueError(
            "extm_lambda must be a finite float >= 0, a 1-d [M] array, or a "
            f"2-d [n_steps, M] array; got {value!r}"
        )
    if arr.ndim == 1 and arr.shape[0] != M:
        raise ValueError(
            f"extm_lambda with per-target rates must have shape ({M},); got {arr.shape}"
        )
    if arr.ndim == 1 and (arr.shape[0] == 0 or not bool(np.all(np.isfinite(arr)))
                           or bool(np.any(arr < 0.0))):
        raise ValueError("extm_lambda vector must hold finite values >= 0")
    if arr.ndim == 2 and (arr.shape[1] != M or arr.shape[0] == 0):
        raise ValueError(
            f"extm_lambda schedule must have shape (n_steps, {M}); got {arr.shape}"
        )
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr < 0.0)):
        raise ValueError("extm_lambda schedule must hold finite values >= 0")
    if arr.ndim == 1:
        return ("vector", arr)
    return ("schedule", arr)


def _checked_drive_key_stack(key):
    """One JAX PRNG key per drive entry, stacked with shape ``(n_entries, 2)``.

    A single key is refused (it names no entry count): stack one
    ``jax.random.PRNGKey(seed)`` per entry, in entry order.
    """
    try:
        arr = jnp.asarray(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extm key must hold one JAX PRNG key per entry with shape "
            f"(n_entries, 2); got {key!r}"
        ) from exc
    if arr.shape == (2,):
        raise ValueError(
            "extm key holds one JAX PRNG key per entry with shape (n_entries, 2); "
            "got a single key of shape (2,): stack one key per entry"
        )
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2:
        raise ValueError(
            "extm key must hold one JAX PRNG key per entry with shape "
            f"(n_entries, 2); got shape {tuple(arr.shape)}"
        )
    try:
        jax.vmap(jax.random.split)(arr)
    except Exception as exc:
        raise ValueError(f"extm key must hold valid JAX PRNG keys; got {key!r}") from exc
    return arr


def _check_external_drive_multi(N, R, given):
    """Validate the seven multi-target drive arguments; a dict, or None when all are absent."""
    missing = [k for k in _EXTM_ARGS if given[k] is None]
    if missing and len(missing) < len(_EXTM_ARGS):
        raise ValueError(
            "multi-target drive arguments (extm_targets, extm_lambda, extm_weight_nS, "
            f"extm_receptor, extm_delay_steps, extm_entry_sizes, key) must be all given or all None; missing {missing}"
        )
    if missing:
        return None
    targets = np.asarray(given["extm_targets"])
    if targets.ndim != 1:
        raise ValueError(f"extm_targets must be one-dimensional, got shape {targets.shape}")
    t64 = targets.astype(np.float64)
    if not np.all(np.isfinite(t64)) or np.any(t64 != np.round(t64)):
        raise ValueError("extm_targets must hold integers")
    t64 = t64.astype(np.int64)
    if t64.size == 0:
        raise ValueError("extm_targets must hold at least one cell index")
    if t64.min() < 0 or t64.max() >= N:
        raise ValueError(
            f"extm_targets must index cells in [0, {N}); got min {t64.min()}, max {t64.max()}"
        )
    if np.unique(t64).size != t64.size:
        raise ValueError(f"extm_targets must hold unique cell indices; got {t64.tolist()}")
    M = int(t64.size)
    lam = _checked_lambda_multi(given["extm_lambda"], M)
    try:
        w_arr = np.asarray(given["extm_weight_nS"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"extm_weight_nS must be a finite float >= 0 or a 1-d [{M}] array; "
            f"got {given['extm_weight_nS']!r}"
        ) from exc
    if w_arr.ndim == 0:
        if w_arr.dtype == np.bool_ or not math.isfinite(float(w_arr)) or float(w_arr) < 0.0:
            raise ValueError(
                f"extm_weight_nS must be a finite float >= 0 or a 1-d [{M}] array; "
                f"got {given['extm_weight_nS']!r}"
            )
        weight: Any = float(w_arr)
    elif w_arr.ndim == 1 and w_arr.shape == (M,) and bool(np.all(np.isfinite(w_arr))) \
            and bool(np.all(w_arr >= 0.0)):
        weight = w_arr
    else:
        raise ValueError(
            f"extm_weight_nS must be a finite float >= 0 or a 1-d [{M}] array; "
            f"got {given['extm_weight_nS']!r}"
        )
    rec = np.asarray(given["extm_receptor"])
    if rec.ndim != 1 or rec.shape != (M,):
        raise ValueError(f"extm_receptor must have shape ({M},); got {rec.shape}")
    recf = rec.astype(np.float64)
    if not np.all(np.isfinite(recf)) or np.any(recf != np.round(recf)) \
            or recf.min() < 0 or recf.max() >= R:
        raise ValueError(f"extm_receptor must index receptors in [0, {R}); got {rec.tolist()}")
    delay = np.asarray(given["extm_delay_steps"])
    if delay.ndim != 1 or delay.shape != (M,):
        raise ValueError(f"extm_delay_steps must have shape ({M},); got {delay.shape}")
    delayf = delay.astype(np.float64)
    if not np.all(np.isfinite(delayf)) or np.any(delayf != np.round(delayf)) \
            or delayf.min() < 1:
        raise ValueError(f"extm_delay_steps must hold integers >= 1; got {delay.tolist()}")
    keys = _checked_drive_key_stack(given["key"])
    n_entries = int(keys.shape[0])
    try:
        sizes_arr = np.asarray(given["extm_entry_sizes"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extm_entry_sizes must hold one int entry size per key row "
            f"(shape ({n_entries},)); got {given['extm_entry_sizes']!r}"
        ) from exc
    if sizes_arr.ndim != 1 or sizes_arr.shape[0] != n_entries:
        raise ValueError(
            "extm_entry_sizes must hold one int entry size per key row "
            f"(shape ({n_entries},)); got shape {sizes_arr.shape}"
        )
    sizesf = sizes_arr.astype(np.float64)
    if not np.all(np.isfinite(sizesf)) or np.any(sizesf != np.round(sizesf)) \
            or sizesf.min() < 1:
        raise ValueError(
            f"extm_entry_sizes must hold integers >= 1; got {sizes_arr.tolist()}"
        )
    sizes = sizes_arr.astype(np.int64)
    if int(sizes.sum()) != M:
        raise ValueError(
            f"extm_entry_sizes sums to {int(sizes.sum())} but extm_targets holds "
            f"{M} cells; entries hold contiguous target blocks covering every target"
        )
    return {
        "targets": t64,
        "lambda": lam,
        "weight": weight,
        "receptor": rec.astype(np.int64),
        "delay": delay.astype(np.int64),
        "keys": keys,
        "entry_sizes": tuple(int(s) for s in sizes),
    }


def _check_edges(N, R, given):
    """Validate the five edge arrays; return numpy arrays, or None when all are absent."""
    missing = [k for k in _EDGE_ARGS if given[k] is None]
    if missing and len(missing) < len(_EDGE_ARGS):
        raise ValueError(f"edge arrays must be all given or all None; missing {missing}")
    if missing:
        return None
    arrs = {}
    for k in _EDGE_ARGS:
        a = np.asarray(given[k])
        if a.ndim != 1:
            raise ValueError(f"{k} must be one-dimensional, got shape {a.shape}")
        arrs[k] = a
    lengths = {k: a.shape[0] for k, a in arrs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"edge arrays must have equal lengths; got {lengths}")

    def _ints(name):
        a = arrs[name].astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integers")
        return a.astype(np.int64)

    pre = _ints("edge_pre")
    post = _ints("edge_post")
    rec = _ints("edge_receptor")
    delay = _ints("edge_delay_steps")
    if pre.size and (pre.min() < 0 or pre.max() >= N):
        raise ValueError(f"edge_pre must index cells in [0, {N}); got min {pre.min()}, max {pre.max()}")
    if post.size and (post.min() < 0 or post.max() >= N):
        raise ValueError(f"edge_post must index cells in [0, {N}); got min {post.min()}, max {post.max()}")
    if delay.size and delay.min() < 1:
        raise ValueError(f"edge_delay_steps must be >= 1; got min {delay.min()}")
    if rec.size and (rec.min() < 0 or rec.max() >= R):
        raise ValueError(f"edge_receptor must index receptors in [0, {R}); got min {rec.min()}, max {rec.max()}")
    return {
        "pre": pre,
        "post": post,
        "weight": arrs["edge_weight_nS"].astype(np.float64),
        "receptor": rec,
        "delay": delay,
    }


def _check_zero_delay_edges(N, R, given):
    """Validate the four delay-0 edge arrays; return numpy arrays, or None when all are absent."""
    missing = [k for k in _EDGE0_ARGS if given[k] is None]
    if missing and len(missing) < len(_EDGE0_ARGS):
        raise ValueError(f"delay-0 edge arrays must be all given or all None; missing {missing}")
    if missing:
        return None
    arrs = {}
    for k in _EDGE0_ARGS:
        a = np.asarray(given[k])
        if a.ndim != 1:
            raise ValueError(f"{k} must be one-dimensional, got shape {a.shape}")
        arrs[k] = a
    lengths = {k: a.shape[0] for k, a in arrs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"delay-0 edge arrays must have equal lengths; got {lengths}")

    def _ints(name):
        a = arrs[name].astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integers")
        return a.astype(np.int64)

    pre = _ints("edge0_pre")
    post = _ints("edge0_post")
    rec = _ints("edge0_receptor")
    if pre.size and (pre.min() < 0 or pre.max() >= N):
        raise ValueError(f"edge0_pre must index cells in [0, {N}); got min {pre.min()}, max {pre.max()}")
    if post.size and (post.min() < 0 or post.max() >= N):
        raise ValueError(f"edge0_post must index cells in [0, {N}); got min {post.min()}, max {post.max()}")
    if rec.size and (rec.min() < 0 or rec.max() >= R):
        raise ValueError(f"edge0_receptor must index receptors in [0, {R}); got min {rec.min()}, max {rec.max()}")
    weight = arrs["edge0_weight_nS"].astype(np.float64)
    if weight.size and not bool(np.all(np.isfinite(weight))):
        raise ValueError("edge0_weight_nS must hold finite values")
    return {
        "pre": pre,
        "post": post,
        "weight": weight,
        "receptor": rec,
    }


EDGE_DELIVERIES = ("all_edges", "event")
_EVENT_TIER_GROWTH = 8


def _event_delivery(src, n_src, seg, absw, gain_mask, n_out, dtype):
    """Per-step synaptic sum over the out-edges of active sources only.

    Edge ``e`` leaves source ``src[e]`` (in ``[0, n_src)``) and lands on flat
    target ``seg[e]`` (in ``[0, n_out)``) with weight ``absw[e]``. Returns
    ``(deliver, info)``; ``deliver(active, g_t)`` equals the all-edges sum
    ``segment_sum(where(active[src], w, 0), seg, n_out)`` with
    ``w = absw * where(gain_mask, g_t, 1)`` (``absw`` when ``gain_mask`` is None).

    Edges are held in CSR order (stable sort by source). A step with ``n``
    active sources runs the smallest capacity tier ``k >= n`` of ``8, 64, 512,
    ...`` (tiers kept while ``k < n_src`` and ``2 k D <= E``, ``D`` the max out-degree):
    it gathers the out-edges of ``k`` sources, padded to ``D``, and
    scatter-adds them, costing ``k D`` instead of ``E``. A step with more
    active sources than the largest tier runs the all-edges sum, so no event is
    dropped. ``lax.switch`` runs one branch per step.

    Summation order: each target gets its terms in (source, edge index) order;
    the all-edges sum adds them in edge index order. ``info["order_exact"]`` is
    True when the two orders agree for every target (along edge index, the
    sources reaching one target never decrease); the sums are then
    bit-identical on CPU, where XLA scatter-adds in update order (unordered
    GPU atomics make neither path deterministic); otherwise they may differ by
    rounding.
    """
    src = np.asarray(src, dtype=np.int64)
    seg = np.asarray(seg, dtype=np.int64)
    absw = np.asarray(absw, dtype=np.float64)
    E = int(src.size)
    perm = np.argsort(src, kind="stable")
    deg = np.bincount(src, minlength=n_src)
    start = np.zeros(n_src + 1, np.int64)
    np.cumsum(deg, out=start[1:])
    D = int(deg.max())
    by_seg = np.argsort(seg, kind="stable")
    same = seg[by_seg][1:] == seg[by_seg][:-1]
    src_by_seg = src[by_seg]
    order_exact = bool(np.all(src_by_seg[1:][same] >= src_by_seg[:-1][same]))
    caps: list[int] = []
    k = _EVENT_TIER_GROWTH
    while k < n_src and 2 * k * D <= E:
        caps.append(k)
        k *= _EVENT_TIER_GROWTH

    one = jnp.asarray(1.0, dtype)
    src_d = jnp.asarray(src, jnp.int32)
    seg_d = jnp.asarray(seg, jnp.int32)
    w_d = jnp.asarray(absw, dtype)
    gm_d = None if gain_mask is None else jnp.asarray(gain_mask)
    # CSR arrays with one pad slot at index E: weight 0, target n_out (dropped).
    seg_c = jnp.asarray(np.append(seg[perm], n_out), jnp.int32)
    w_c = jnp.asarray(np.append(absw[perm], 0.0), dtype)
    gm_c = None if gain_mask is None else jnp.asarray(np.append(np.asarray(gain_mask)[perm], False))
    start_c = jnp.asarray(start, jnp.int32)  # start_c[n_src] = E: the fill source's slots pad
    deg_c = jnp.asarray(np.append(deg, 0), jnp.int32)
    off = jnp.arange(D, dtype=jnp.int32)

    def _all_edges(active, g_t):
        w = w_d if gm_d is None else w_d * jnp.where(gm_d, g_t, one)
        return jax.ops.segment_sum(jnp.where(active[src_d], w, 0.0), seg_d, num_segments=n_out)

    def _tier(cap):
        def run(active, g_t):
            ids = jnp.nonzero(active, size=cap, fill_value=n_src)[0]
            slot = start_c[ids][:, None] + off[None, :]
            slot = jnp.where(off[None, :] < deg_c[ids][:, None], slot, E).reshape(-1)
            w = w_c[slot]
            if gm_c is not None:
                w = w * jnp.where(gm_c[slot], g_t, one)
            return jnp.zeros((n_out,), dtype).at[seg_c[slot]].add(w, mode="drop")

        return run

    branches = [_tier(c) for c in caps] + [_all_edges]
    caps_j = jnp.asarray(caps, jnp.int32)

    def deliver(active, g_t=None):
        if not caps:
            return _all_edges(active, g_t)
        n = jnp.sum(active, dtype=jnp.int32)
        return jax.lax.switch(jnp.sum(n > caps_j, dtype=jnp.int32), branches, active, g_t)

    info = {"tiers": caps, "max_out_degree": D, "n_sources": int(n_src), "order_exact": order_exact}
    return deliver, info


def simulate_adex(
    params: AdExParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight_nS=None,
    edge_receptor=None,
    edge_delay_steps=None,
    edge0_pre=None,
    edge0_post=None,
    edge0_weight_nS=None,
    edge0_receptor=None,
    intra_step_rounds=None,
    receptor_reversal_mV=(),
    receptor_tau_ms=(),
    decay: str,
    step_order: str = "default",
    I_ext_pA=None,
    V0_mV=None,
    record_v: bool = False,
    ext_targets=None,
    ext_lambda=None,
    ext_weight_nS=None,
    ext_receptor=None,
    ext_delay_steps=None,
    key=None,
    extm_targets=None,
    extm_lambda=None,
    extm_weight_nS=None,
    extm_receptor=None,
    extm_delay_steps=None,
    extm_entry_sizes=None,
    ext_target_groups=None,
    ext_t_start_steps=None,
    ext_t_stop_steps=None,
    spike_out_mask=None,
    spike_out_onset_step=0,
    spike_replace_mask=None,
    spike_replace_lam=None,
    spike_replace_key=None,
    edge_gain_mask=None,
    edge_gain_schedule=None,
    edge0_gain_mask=None,
    edge_delivery: str = "all_edges",
) -> dict[str, Any]:
    """Simulate conductance-based AdEx cells on a sparse edge list.

    Cells: Brette & Gerstner (2005), doi:10.1152/jn.00686.2005. Network, synapses and
    external drive: Tahvili, Vinck & di Volo (2026), Methods Eqs. 15-18,
    doi:10.1371/journal.pcbi.1014378.

    Each edge ``e`` runs from ``edge_pre[e]`` to ``edge_post[e]`` through receptor
    ``edge_receptor[e]``, with weight ``edge_weight_nS[e]`` and delay ``edge_delay_steps[e]``.
    Negative weights are accepted and ``|w|`` is used; the receptor reversal sets the sign of
    the current.

    Per step ``t``, in order:

    1. the drive key is split once (``rkey, ks = split(rkey)``); with no external drive
       the key stays absent and no draw happens;
    2. arrivals: each edge reads the spike vector of step ``t - d_e`` (``d_e`` plus 1 under
       ``step_order="brian2"``) from a ring of length ``max(d_e)`` (plus 1 for brian2), and
       each external target reads the Poisson count drawn ``d_ext`` steps earlier (plus 1
       under ``step_order="brian2"``) from an int32 ring of the same length, except
       ``ext_delay_steps=0`` (same-step drive): the count for step ``t`` is drawn first
       and joins ``g`` before the current, under the same key stream;
    3. conductance ``g[N, R] = g * decay_r + scatter_add(post, receptor, |w| * arrival)``,
       with ``decay_r = exp(-dt / tau_r)`` for ``decay="exp"`` or ``1 - dt / tau_r`` for
       ``decay="euler"`` (the CAMINOS reference; required, no default, because the two differ
       by 0.951229 vs 0.95 per step at dt 0.1 ms, tau 2 ms); each external target adds
       ``ext_weight_nS * count`` at ``[target, ext_receptor]``, keeping multiplicities;
    4. current ``I = sum_r g[:, r] * (E_r - V) + I_ext[t]``;
    5. :func:`adex_step`;
    6. this step's spikes are written into the ring, and fresh counts
       ``poisson(ks, ext_lambda)`` per target are written into the external ring, so a
       count drawn at step ``t`` acts at step ``t + d_ext`` and the first ``d_ext`` steps
       carry no external drive (under same-step drive nothing is written: the count
       acts at once).

    The external drive is opt-in through six keyword arguments that are all given or all
    None: ``ext_targets`` (int ``[M]`` cell indices, unique, in range), ``ext_lambda``
    (mean count per step per target, >= 0, either a scalar float, or a per-step
    schedule array ``[n_steps]`` (shared by every target) or ``[n_steps, M]`` (per
    target); finite values >= 0, bools refused), ``ext_weight_nS`` (conductance per count, >= 0),
    ``ext_receptor`` (a valid receptor index), ``ext_delay_steps`` (int >= 0; 0 is the
    same-step drive above and is refused under ``step_order="brian2"``; a delay >= 1
    gains 1 under brian2, as for edges), and
    ``key`` (a JAX PRNG key). A scalar
    ``ext_lambda`` takes the historical path bit-identically; a schedule is drawn
    per step (``poisson(ks, lam[t])`` under the same one-split-per-step key
    stream) and validated against ``(n_steps, M)`` here, where both are known.

    ``spike_out_mask`` (bool ``[N]``) with ``spike_out_onset_step`` (int in
    ``[0, n_steps)``) silences the emitted spike of masked cells on steps
    ``t >= spike_out_onset_step``: the mask acts before the same-step fixed
    point, so from the onset on a masked cell never enters a step's spike
    accumulator and its spike reaches neither delay-0 targets, the delay
    ring, nor the recorded spikes, while ``V``, ``w`` and the refractory
    counters keep evolving, after the CAMINOS step-loop mask
    (``artifacts/etudes/papers/caminos/caminos_jax.py`` lines 812, 840-846).
    ``None`` (the default) takes the historical path bit-identically; an onset
    without a mask, a wrong shape or dtype, or an onset outside
    ``[0, n_steps)`` is refused.

    ``spike_replace_mask`` (bool ``[N]``) with ``spike_replace_lam`` (float
    ``[N]``, the mean Poisson count per step of each cell, every entry
    finite and >= 0 with 0 outside the mask) and ``spike_replace_key`` (a
    JAX PRNG key) replaces the emitted spike of masked cells by an
    independent per-step draw: ``where(mask, poisson(fold_in(key, t),
    lam_vec) > 0, spk)``. The draw folds the passed key
    per step and never touches the drive ``rkey`` stream, so the default
    draws are unchanged; the replacement acts wherever ``spike_out_mask``
    acts (round 0, fixed-point rounds, final spikes), before silencing, so a
    masked-and-silenced cell stays silent, after the CAMINOS step-loop
    replacement (``artifacts/etudes/papers/caminos/caminos_jax.py`` lines
    833-839). All three are given or all None (``None`` takes the historical
    path bit-identically); an all-False mask leaves the run unchanged
    whatever ``lam_vec`` is.

    ``edge_gain_mask`` (bool ``[E]``) with ``edge_gain_schedule`` (float
    ``[n_steps]``, every entry finite and >= 0) scales the conductance
    weight of masked edges by the per-step factor ``schedule[t]``: each
    masked edge adds ``|w| * schedule[t]`` where it arrives, each unmasked
    edge adds ``|w|``. ``edge0_gain_mask`` (bool ``[E0]``) does the same
    for the delay-0 edges; it is required with the pair when delay-0 edges
    exist and refused when none exist. All given or all None (``None``
    takes the historical path bit-identically); an all-ones schedule is
    bit-exact and an all-False mask leaves the run unchanged whatever the
    schedule is.

    ``ext_target_groups`` (int ``[M]``, one group per target, given with the six
    drive arguments or refused) selects the grouped path: ``ext_lambda`` must
    then be a 2-d ``[n_steps, G]`` per-group schedule, each group non-empty,
    and step ``t`` draws each group ``g`` separately as
    ``poisson(fold_in(fold_in(key, g), t), lam[t, g])`` over its member cells,
    so groups own independent streams while the scalar path above is
    byte-identical.

    The multi-target drive (``extm_targets``, ``extm_lambda``, ``extm_weight_nS``,
    ``extm_receptor``, ``extm_delay_steps``, ``extm_entry_sizes``, ``key``: all
    given or all None, never beside ``ext_*``) carries one Poisson stream per
    entry: ``extm_lambda`` is a scalar, a 1-d ``[M]`` vector, or a 2-d
    ``[n_steps, M]`` schedule; weights are a scalar or ``[M]``; receptors and
    delays (integers >= 1) are ``[M]``. ``extm_entry_sizes`` holds one int size
    per entry (entries hold contiguous target blocks covering every target);
    ``key`` holds one JAX PRNG key per entry with shape ``(n_entries, 2)``.
    Each step splits every entry key once and draws that entry's segment under
    its own stream, so one entry's seed leaves every other entry's counts
    unchanged; target ``m`` reads its count ``delay[m]`` steps later from its
    own ring column and adds ``weight[m] * count`` at ``[target[m],
    receptor[m]]``. The shared window above applies unchanged;
    ``ext_target_groups`` is refused with this path.

    ``ext_t_start_steps``/``ext_t_stop_steps`` (both given or both None) restrict
    the drive to steps ``[t_start, t_stop)``: they must be int step indices with
    ``0 <= t_start < t_stop <= n_steps``. Every step still splits the drive key,
    and the Poisson count is drawn and then zeroed outside the window, so steps
    inside the window reproduce the all-steps drive run bit-for-bit. Given
    without the six drive arguments, they are refused rather than silently
    ignored.

    ``I_ext_pA`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``. ``V0_mV`` is None
    (rest at ``EL``) or shape ``[N]``. ``edge_*`` arrays are all given or all None. Edge
    delays must be integers >= 1.

    The declared intra-step fixed point for zero-delay edges rides the
    ``edge0_*`` arrays (``edge0_pre``, ``edge0_post``, ``edge0_weight_nS``,
    ``edge0_receptor``: all given or all None, same shapes as ``edge_*``
    minus the delay) plus ``intra_step_rounds`` (a positive int R, the cap):
    within one step, round 0 is the step above giving ``spk``; then further
    rounds run while some cell that newly spiked in the last round owns a
    delay-0 out-edge and fewer than R rounds have run. Each such round adds
    the newly spiked cells' delay-0 weights to the post cells' ``g`` for this
    step, and every cell not yet spiked this step is re-stepped from the same
    pre-step ``(V, w, ref)`` with the updated ``g``; spikes are OR-ed, never
    undone. The loop stops at quiescence (a round whose newly spiked cells
    own no delay-0 out-edge adds no conductance, so stopping changes
    nothing); the cap stops it otherwise, and then the run is refused rather
    than returning dropped events. The final ``g`` includes all delay-0
    arrivals of the step, and the final ``spk`` is written to the ring.
    Absent ``edge0_*``/``intra_step_rounds`` (both None) the scan body
    is the step above byte-for-byte; one without the other is refused.

    ``edge_delivery`` selects how step 3 sums the delayed-edge arrivals:
    ``"all_edges"`` (default) reads every edge each step; ``"event"`` sums
    only the out-edges of the (cell, delay) sources that spiked ``delay``
    steps earlier, in capacity tiers with the all-edges sum as the overflow
    tier (:func:`_event_delivery`), and returns ``event_delivery`` (tiers,
    max out-degree, source count, ``order_exact``). With ``order_exact`` True
    the two are bit-identical on CPU; otherwise they may differ in rounding of the
    per-target sum. Delay-0 fixed-point rounds read all ``edge0_*`` edges
    under both.

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV`` (``[n_steps, N]``, after each
    step's update, only if ``record_v``: 4.6 GB in float32 at 9500 cells and 120000 steps),
    and the final ``V``, ``w``, ``ref`` and ``g``. With delay-0 edges, also
    ``fixed_point_unresolved`` (int32 ``[n_steps]``: per step, the number of
    cells that newly spiked in the final round and own delay-0 out-edges;
    nonzero only where the cap stopped the loop) and ``fixed_point_rounds``
    (int32 ``[n_steps]``: rounds run after round 0); absent otherwise. A nonzero
    unresolved count anywhere refuses with ``ValueError`` naming the cap, the
    first such step and the total, saying to raise ``intra_step_rounds``;
    nothing partial is returned.
    Implemented with ``jax.lax.scan``; no dense ``N x N`` matrix. dtype follows ``params``.

    Raises ``ValueError`` naming the argument for every refused input.
    """
    if step_order not in _STEP_ORDERS:
        raise ValueError(f"step_order must be 'default' or 'brian2', got {step_order!r}")
    if decay not in _DECAYS:
        raise ValueError(f"decay must be 'exp' or 'euler', got {decay!r}")
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not dt_ms > 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}")
    if edge_delivery not in EDGE_DELIVERIES:
        raise ValueError(f"edge_delivery must be one of {EDGE_DELIVERIES}, got {edge_delivery!r}")

    N = int(jnp.shape(params.C_pF)[0])
    if spike_out_mask is None:
        if spike_out_onset_step != 0:
            raise ValueError(
                "spike_out_onset_step without spike_out_mask is not realized: "
                f"got onset {spike_out_onset_step!r} with no mask -- pass both or neither"
            )
        mask_j = None
        onset_j = 0
    else:
        mask_arr = np.asarray(spike_out_mask)
        if mask_arr.dtype != np.bool_:
            raise ValueError(
                "spike_out_mask must be a bool array of shape "
                f"({N},); got dtype {mask_arr.dtype}"
            )
        if mask_arr.shape != (N,):
            raise ValueError(
                "spike_out_mask must be a bool array of shape "
                f"({N},); got shape {mask_arr.shape}"
            )
        if (
            isinstance(spike_out_onset_step, bool)
            or not isinstance(spike_out_onset_step, (int, np.integer))
            or not 0 <= int(spike_out_onset_step) < int(n_steps)
        ):
            raise ValueError(
                "spike_out_onset_step must be an int in [0, n_steps) "
                f"([0, {int(n_steps)})); got {spike_out_onset_step!r}"
            )
        mask_j = jnp.asarray(mask_arr)
        onset_j = int(spike_out_onset_step)
    _rep_given = (
        spike_replace_mask is not None,
        spike_replace_lam is not None,
        spike_replace_key is not None,
    )
    if not any(_rep_given):
        rep_j = None
        rep_lam_j = None
        rep_base = None
    elif not all(_rep_given):
        raise ValueError(
            "spike_replace_mask, spike_replace_lam and spike_replace_key must be "
            "all given or all None; got "
            f"mask={_rep_given[0]}, lam={_rep_given[1]}, key={_rep_given[2]}"
        )
    else:
        rep_arr = np.asarray(spike_replace_mask)
        if rep_arr.dtype != np.bool_:
            raise ValueError(
                "spike_replace_mask must be a bool array of shape "
                f"({N},); got dtype {rep_arr.dtype}"
            )
        if rep_arr.shape != (N,):
            raise ValueError(
                "spike_replace_mask must be a bool array of shape "
                f"({N},); got shape {rep_arr.shape}"
            )
        lam_arr = np.asarray(spike_replace_lam, dtype=np.float64)
        if lam_arr.shape != (N,):
            raise ValueError(
                "spike_replace_lam must be a float array of shape "
                f"({N},); got shape {lam_arr.shape}"
            )
        if lam_arr.dtype == np.bool_ or not np.all(np.isfinite(lam_arr)):
            raise ValueError(
                "spike_replace_lam must hold finite entries; "
                f"got {spike_replace_lam!r}"
            )
        if bool(np.any(lam_arr < 0.0)):
            raise ValueError(
                "spike_replace_lam must hold entries >= 0; "
                f"got {spike_replace_lam!r}"
            )
        try:
            jax.random.split(spike_replace_key)
        except Exception:
            raise ValueError(
                f"spike_replace_key must be a JAX PRNG key; got {spike_replace_key!r}"
            )
        rep_j = jnp.asarray(rep_arr)
        rep_lam_j = jnp.asarray(lam_arr)
        rep_base = spike_replace_key
    dtype = jnp.asarray(params.C_pF).dtype
    if N * max(len(np.reshape(receptor_tau_ms, -1)), 1) >= 2**31:
        raise ValueError(f"N * n_receptors must be < 2**31 (int32 segment ids); N = {N}")
    rev = np.asarray(receptor_reversal_mV, dtype=np.float64).reshape(-1)
    tau = np.asarray(receptor_tau_ms, dtype=np.float64).reshape(-1)
    if rev.shape != tau.shape:
        raise ValueError(
            "receptor_reversal_mV and receptor_tau_ms must have the same length; "
            f"got {rev.size} and {tau.size}"
        )
    if np.any(tau <= 0):
        raise ValueError(f"receptor_tau_ms must be > 0 for every receptor; got {tau.tolist()}")
    if decay == "euler" and np.any(tau <= dt_ms):
        raise ValueError(
            f"decay='euler' needs receptor_tau_ms > dt_ms ({dt_ms}); got {tau.tolist()}"
        )
    R = int(rev.size)

    edges = _check_edges(
        N,
        R,
        {
            "edge_pre": edge_pre,
            "edge_post": edge_post,
            "edge_weight_nS": edge_weight_nS,
            "edge_receptor": edge_receptor,
            "edge_delay_steps": edge_delay_steps,
        },
    )
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "receptor": np.zeros((0,), np.int64),
            "delay": np.zeros((0,), np.int64),
        }
    E = int(edges["pre"].shape[0])
    delay_eff = edges["delay"] + (1 if step_order == "brian2" else 0)
    ext_given = any(v is not None for v in (
        ext_targets, ext_lambda, ext_weight_nS, ext_receptor, ext_delay_steps,
    ))
    extm_given = any(v is not None for v in (
        extm_targets, extm_lambda, extm_weight_nS, extm_receptor, extm_delay_steps,
        extm_entry_sizes,
    ))
    if ext_given and extm_given:
        raise ValueError(
            "ext_* and extm_* drive arguments cannot combine: pass one drive path or neither"
        )
    if ext_given:
        ext = _check_external_drive(
            N,
            R,
            {
                "ext_targets": ext_targets,
                "ext_lambda": ext_lambda,
                "ext_weight_nS": ext_weight_nS,
                "ext_receptor": ext_receptor,
                "ext_delay_steps": ext_delay_steps,
                "key": key,
            },
            allow_zero_delay=True,
        )
        extm = None
    elif extm_given or key is not None:
        ext = None
        extm = _check_external_drive_multi(
            N,
            R,
            {
                "extm_targets": extm_targets,
                "extm_lambda": extm_lambda,
                "extm_weight_nS": extm_weight_nS,
                "extm_receptor": extm_receptor,
                "extm_delay_steps": extm_delay_steps,
                "extm_entry_sizes": extm_entry_sizes,
                "key": key,
            },
        )
    else:
        ext = extm = None
    L_edge = int(delay_eff.max()) if E else 0
    use_multi = extm is not None
    if ext is None and extm is None:
        if ext_target_groups is not None:
            raise ValueError(
                "ext_target_groups needs the external drive; got "
                f"{np.asarray(ext_target_groups).tolist()} with no ext_targets -- "
                "pass the six drive arguments with a 2-d [n_steps, G] schedule, or neither"
            )
        if ext_t_start_steps is not None or ext_t_stop_steps is not None:
            raise ValueError(
                "ext_t_start_steps/ext_t_stop_steps need the external drive; got "
                f"{ext_t_start_steps!r}/{ext_t_stop_steps!r} with no ext_targets — "
                "pass the six drive arguments or neither bound"
            )
        L = L_edge if E else 1
        d_ext = 0
        M = 0
        window = None
        grouped = None
    else:
        if use_multi:
            assert extm is not None
            if ext_target_groups is not None:
                raise ValueError(
                    "ext_target_groups needs the single-drive path; the multi-target "
                    "path carries one stream per target already"
                )
            d_j = np.asarray(extm["delay"], dtype=np.int64)
            d_ext_j = d_j + (1 if step_order == "brian2" else 0)
            L = int(max(L_edge, int(d_ext_j.max()), 1))
            M = int(extm["targets"].shape[0])
            window = _check_drive_window(n_steps, ext_t_start_steps, ext_t_stop_steps)
            grouped = None
            d_ext = 0
        else:
            assert ext is not None
            d_ext = int(ext["delay"]) + (1 if step_order == "brian2" else 0)
            L = max(L_edge, d_ext, 1)
            M = int(ext["targets"].shape[0])
            window = _check_drive_window(n_steps, ext_t_start_steps, ext_t_stop_steps)
            grouped = _check_target_groups(ext_target_groups, M, n_steps, ext["lambda"])
            if int(ext["delay"]) == 0 and step_order == "brian2":
                raise ValueError(
                    "ext_delay_steps=0 requests a same-step drive, which step_order='brian2' "
                    "does not realize (it reads every arrival one step later): use a delay >= 1"
                )

    zero = _check_zero_delay_edges(
        N,
        R,
        {
            "edge0_pre": edge0_pre,
            "edge0_post": edge0_post,
            "edge0_weight_nS": edge0_weight_nS,
            "edge0_receptor": edge0_receptor,
        },
    )
    if zero is None and intra_step_rounds is None:
        has_zero = False
        n_rounds = 0
    elif zero is None:
        raise ValueError(
            f"intra_step_rounds={intra_step_rounds!r} needs the delay-0 edge arrays "
            "(edge0_pre, edge0_post, edge0_weight_nS, edge0_receptor); pass both or neither"
        )
    elif intra_step_rounds is None:
        raise ValueError(
            "delay-0 edge arrays (edge0_pre, edge0_post, edge0_weight_nS, edge0_receptor) "
            "need intra_step_rounds; pass both or neither"
        )
    else:
        from .pcl_topology import validate_rounds

        n_rounds = validate_rounds(intra_step_rounds, label="intra_step_rounds")
        has_zero = True

    _gain_given = (
        edge_gain_mask is not None
        or edge_gain_schedule is not None
        or edge0_gain_mask is not None
    )
    if not _gain_given:
        gain_mask_j = None
        gain_sched_j = None
        gain0_j = None
    else:
        if edge_gain_mask is None or edge_gain_schedule is None:
            raise ValueError(
                "edge_gain_mask and edge_gain_schedule must be both given or both None; "
                f"got mask={edge_gain_mask is not None}, schedule={edge_gain_schedule is not None}"
            )
        mask_arr = np.asarray(edge_gain_mask)
        if mask_arr.dtype != np.bool_:
            raise ValueError(
                "edge_gain_mask must be a bool array of shape "
                f"({E},); got dtype {mask_arr.dtype}"
            )
        if mask_arr.shape != (E,):
            raise ValueError(
                "edge_gain_mask must be a bool array of shape "
                f"({E},); got shape {mask_arr.shape}"
            )
        sched_raw = np.asarray(edge_gain_schedule)
        if sched_raw.dtype == np.bool_:
            raise ValueError(
                "edge_gain_schedule must be a float array of shape "
                f"({n_steps},), not bools"
            )
        try:
            sched_arr = np.asarray(edge_gain_schedule, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "edge_gain_schedule must be a float array of shape "
                f"({n_steps},); got {edge_gain_schedule!r}"
            ) from exc
        if sched_arr.shape != (n_steps,):
            raise ValueError(
                "edge_gain_schedule must be a float array of shape "
                f"({n_steps},); got shape {sched_arr.shape}"
            )
        if not np.all(np.isfinite(sched_arr)):
            raise ValueError("edge_gain_schedule must hold finite entries")
        if bool(np.any(sched_arr < 0.0)):
            raise ValueError(
                "edge_gain_schedule must hold entries >= 0; "
                f"got min {float(sched_arr.min())}"
            )
        if has_zero:
            if edge0_gain_mask is None:
                raise ValueError(
                    "edge_gain_mask and edge_gain_schedule need edge0_gain_mask when "
                    "delay-0 edges exist; pass all three or all None"
                )
            mask0_arr = np.asarray(edge0_gain_mask)
            E0 = int(zero["pre"].shape[0]) if zero is not None else 0
            if mask0_arr.dtype != np.bool_:
                raise ValueError(
                    "edge0_gain_mask must be a bool array of shape "
                    f"({E0},); got dtype {mask0_arr.dtype}"
                )
            if mask0_arr.shape != (E0,):
                raise ValueError(
                    "edge0_gain_mask must be a bool array of shape "
                    f"({E0},); got shape {mask0_arr.shape}"
                )
            gain0_j = jnp.asarray(mask0_arr)
        else:
            if edge0_gain_mask is not None:
                raise ValueError(
                    "edge0_gain_mask is not realized with no delay-0 edges; "
                    "pass edge_gain_mask and edge_gain_schedule only, or all None"
                )
            gain0_j = None
        gain_mask_j = jnp.asarray(mask_arr)
        gain_sched_j = jnp.asarray(sched_arr, dtype)

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    seg_j = jnp.asarray(edges["post"] * R + edges["receptor"], jnp.int32)
    absw_j = jnp.asarray(np.abs(edges["weight"]), dtype)
    lag_j = jnp.asarray(delay_eff, jnp.int32)
    deliver = None
    event_info = None
    if edge_delivery == "event" and not E:
        event_info = {"tiers": [], "max_out_degree": 0, "n_sources": 0, "order_exact": True}
    elif edge_delivery == "event":
        # One source per distinct (pre, delay) pair, active at step t when its cell
        # spiked delay steps earlier. Pre-major keys keep a pre-sorted edge list
        # in its own per-target order (order_exact).
        src_key, src_of_edge = np.unique(
            edges["pre"].astype(np.int64) * (L + 1) + delay_eff.astype(np.int64), return_inverse=True
        )
        src_pre_j = jnp.asarray(src_key // (L + 1), jnp.int32)
        src_lag_j = jnp.asarray(src_key % (L + 1), jnp.int32)
        deliver, event_info = _event_delivery(
            src_of_edge.reshape(-1), int(src_key.size), edges["post"] * R + edges["receptor"],
            np.abs(edges["weight"]), None if gain_mask_j is None else np.asarray(gain_mask_j),
            N * R, dtype,
        )

    def _arrivals(ring, t):
        """Conductance increments ``[N, R]`` from the delayed edges arriving at step ``t``."""
        if not E:
            return jnp.zeros((N, R), dtype)
        if deliver is not None:
            g_t = None if gain_mask_j is None else gain_sched_j[t]
            return deliver(ring[(t - src_lag_j) % L, src_pre_j], g_t).reshape(N, R)
        arrived = ring[(t - lag_j) % L, pre_j]
        if gain_mask_j is None:
            return jax.ops.segment_sum(
                jnp.where(arrived, absw_j, 0.0), seg_j, num_segments=N * R
            ).reshape(N, R)
        g_t = gain_sched_j[t]
        w_eff = absw_j * jnp.where(gain_mask_j, g_t, jnp.asarray(1.0, dtype))
        return jax.ops.segment_sum(
            jnp.where(arrived, w_eff, 0.0), seg_j, num_segments=N * R
        ).reshape(N, R)

    if has_zero:
        assert zero is not None
        pre0_j = jnp.asarray(zero["pre"], jnp.int32)
        seg0_j = jnp.asarray(zero["post"] * R + zero["receptor"], jnp.int32)
        absw0_j = jnp.asarray(np.abs(zero["weight"]), dtype)
        has_out0_j = jnp.asarray(np.bincount(zero["pre"], minlength=N) > 0)
    decay_r = np.exp(-dt_ms / tau) if decay == "exp" else 1.0 - dt_ms / tau
    decay_j = jnp.asarray(decay_r, dtype)
    rev_j = jnp.asarray(rev, dtype)
    if ext is not None:
        targets_j = jnp.asarray(ext["targets"], jnp.int32)
        w_ext_j = jnp.asarray(ext["weight"], dtype)
        rec_ext = int(ext["receptor"])
        lam_kind, lam_val = ext["lambda"]
        grp_G = 0
        grp_sizes: list[int] = []
        grp_unorder_j = None
        grp_base = None
        if lam_kind == "scalar":
            lam_f: Any = float(lam_val)
            lam_1d = None
            lam_2d = None
        else:
            lam_f = None
            sched = np.asarray(lam_val, dtype=np.float64)
            if sched.ndim == 1:
                if sched.shape[0] != n_steps:
                    raise ValueError(
                        "ext_lambda schedule with one shared rate must have shape "
                        f"({n_steps},); got {sched.shape}"
                    )
                lam_1d = jnp.asarray(sched, dtype)
                lam_2d = None
            elif grouped is None:
                if sched.shape != (n_steps, M):
                    raise ValueError(
                        "ext_lambda schedule with per-target rates must have shape "
                        f"({n_steps}, {M}); got {sched.shape}"
                    )
                lam_1d = None
                lam_2d = jnp.asarray(sched, dtype)
            else:
                assert grouped is not None
                lam_1d = None
                lam_2d = jnp.asarray(sched, dtype)
                grp_G = int(grouped["G"])
                grp_sizes = [int(s) for s in grouped["sizes"]]
                grp_unorder_j = jnp.asarray(grouped["unorder"])
                grp_base = ext["key"]
    if extm is not None:
        assert ext is None
        mt_j = jnp.asarray(extm["targets"], jnp.int32)
        md_j = jnp.asarray(extm["delay"], jnp.int32) + (
            jnp.int32(1) if step_order == "brian2" else jnp.int32(0)
        )
        mr_j = jnp.asarray(extm["receptor"], jnp.int32)
        mw_raw = extm["weight"]
        mw_j = (
            jnp.asarray(float(mw_raw), dtype)
            if isinstance(mw_raw, float)
            else jnp.asarray(np.asarray(mw_raw, dtype=np.float64), dtype)
        )
        mlam_kind, mlam_val = extm["lambda"]
        if mlam_kind == "scalar":
            mlam_f: Any = float(mlam_val)
            mlam_v = None
            mlam_2d = None
        elif mlam_kind == "vector":
            mlam_f = None
            mlam_v = jnp.asarray(np.asarray(mlam_val, dtype=np.float64), dtype)
            mlam_2d = None
        else:
            mlam_f = None
            mlam_v = None
            sched_m = np.asarray(mlam_val, dtype=np.float64)
            if sched_m.shape != (n_steps, M):
                raise ValueError(
                    "extm_lambda schedule with per-target rates must have shape "
                    f"({n_steps}, {M}); got {sched_m.shape}"
                )
            mlam_2d = jnp.asarray(sched_m, dtype)
        mkey0 = extm["keys"]
        mentry_sizes = tuple(int(s) for s in extm["entry_sizes"])
        mentry_offs = (0,)
        for _size in mentry_sizes:
            mentry_offs = mentry_offs + (mentry_offs[-1] + _size,)

    # A constant current is closed over, so no [n_steps, N] array is built for it.
    I_const = jnp.zeros((N,), dtype)
    I_all = None
    if I_ext_pA is not None:
        I_arr = jnp.asarray(I_ext_pA, dtype)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext_pA must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )

    if V0_mV is None:
        V0 = jnp.asarray(params.EL_mV, dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape != (N,):
            raise ValueError(f"V0_mV must be None or shape ({N},); got {V0.shape}")

    def body(carry, I_t):
        V, w, ref, g, ring, ering, t, rkey = carry
        if ext is not None:
            rkey, ks = jax.random.split(rkey)
        if rep_j is not None:
            rep_po = jax.random.poisson(
                jax.random.fold_in(rep_base, t), rep_lam_j, shape=(N,)) > 0
        I_t = I_const if I_t is None else I_t
        g = g * decay_j[None, :] + _arrivals(ring, t)
        if ext is not None:
            earr = ering[(t - d_ext) % L]
            g = g.at[targets_j, rec_ext].add(w_ext_j * earr.astype(dtype))
        if not has_zero:
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V, w, ref), spk = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
        else:
            # Declared intra-step fixed point: round 0 is the step above, then
            # further rounds re-step every not-yet-spiked cell from the same
            # pre-step (V, w, ref) with the delay-0 arrivals folded into g, until
            # no newly spiked cell owns a delay-0 out-edge (quiescence) or R
            # rounds have run. A round with no such cell adds no conductance, so
            # stopping there changes nothing.
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V1, w1, ref1), spk0 = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
            if rep_j is not None:
                spk0 = jnp.where(rep_j, rep_po, spk0)
            if mask_j is not None:
                # Silencing acts before the fixed point: a masked cell never
                # enters spk_acc, so its spike reaches no delay-0 target.
                spk0 = spk0 & ~(mask_j & (t >= onset_j))
            V_pre, w_pre, ref_pre = V, w, ref

            def _fp_cond(fp_carry):
                _, _, _, _, _, prev_new, done = fp_carry
                return (done < n_rounds) & jnp.any(prev_new & has_out0_j)

            def _fp_round(fp_carry):
                spk_acc, Vc, wc, refc, gc, prev_new, done = fp_carry
                if gain0_j is None:
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], absw0_j, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                else:
                    g_t0 = gain_sched_j[t]
                    w0_eff = absw0_j * jnp.where(
                        gain0_j, g_t0, jnp.asarray(1.0, dtype))
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], w0_eff, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                gc = gc + add0
                I_2 = jnp.sum(gc * (rev_j[None, :] - V_pre[:, None]), axis=1) + I_t
                (Vn, wn, refn), spkn = adex_step(
                    (V_pre, w_pre, ref_pre), I_2, params, dt_ms, step_order
                )
                if rep_j is not None:
                    spkn = jnp.where(rep_j, rep_po, spkn)
                if mask_j is not None:
                    spkn = spkn & ~(mask_j & (t >= onset_j))
                take = ~spk_acc
                Vc = jnp.where(take, Vn, Vc)
                wc = jnp.where(take, wn, wc)
                refc = jnp.where(take, refn, refc)
                return (spk_acc | spkn, Vc, wc, refc, gc, spkn & ~spk_acc, done + 1)

            spk_acc, V, w, ref, g, _prev, n_done = jax.lax.while_loop(
                _fp_cond, _fp_round, (spk0, V1, w1, ref1, g, spk0, jnp.int32(0))
            )
            spk = spk_acc
            # Cells that newly spiked in the final round and own delay-0
            # out-edges: their same-step outputs were never delivered.
            n_unres = jnp.sum(_prev & has_out0_j).astype(jnp.int32)
        if rep_j is not None:
            spk = jnp.where(rep_j, rep_po, spk)
        if mask_j is not None:
            spk = spk & ~(mask_j & (t >= onset_j))
        ring = ring.at[t % L].set(spk)
        if ext is not None:
            if grouped is not None:
                assert grp_base is not None and grp_unorder_j is not None
                lam_g = lam_2d[t]
                segs = [
                    jax.random.poisson(
                        jax.random.fold_in(jax.random.fold_in(grp_base, g), t),
                        lam_g[g],
                        shape=(grp_sizes[g],),
                    )
                    for g in range(grp_G)
                ]
                extc = jnp.concatenate(segs)[grp_unorder_j].astype(jnp.int32)
            elif lam_1d is None and lam_2d is None:
                lam_t = lam_f
                extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
            elif lam_2d is None:
                lam_t = lam_1d[t]
                extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
            else:
                lam_t = lam_2d[t]
                extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
            if window is not None:
                t0, t1 = window
                extc = jnp.where((t >= t0) & (t < t1), extc, jnp.zeros_like(extc))
            ering = ering.at[t % L].set(extc)
        if has_zero:
            out = (spk, V, n_unres, n_done) if record_v else (spk, n_unres, n_done)
        else:
            out = (spk, V) if record_v else spk
        return (V, w, ref, g, ring, ering, t + 1, rkey), out

    def body0(carry, I_t):
        """Same-step external drive (``d_ext == 0``): the count drawn for step t acts at t.

        Same one-split-per-step key stream as :func:`body` above, so the drawn counts
        are the delayed run's counts shifted by one step; the draw happens first and
        the fresh counts join ``g`` before the current is computed. The external ring
        is unused (carried only so the scan signature matches).
        """
        V, w, ref, g, ring, ering, t, rkey = carry
        rkey, ks = jax.random.split(rkey)
        if rep_j is not None:
            rep_po = jax.random.poisson(
                jax.random.fold_in(rep_base, t), rep_lam_j, shape=(N,)) > 0
        I_t = I_const if I_t is None else I_t
        g = g * decay_j[None, :] + _arrivals(ring, t)
        if grouped is not None:
            assert grp_base is not None and grp_unorder_j is not None
            lam_g = lam_2d[t]
            segs = [
                jax.random.poisson(
                    jax.random.fold_in(jax.random.fold_in(grp_base, g), t),
                    lam_g[g],
                    shape=(grp_sizes[g],),
                )
                for g in range(grp_G)
            ]
            extc = jnp.concatenate(segs)[grp_unorder_j].astype(jnp.int32)
        elif lam_1d is None and lam_2d is None:
            lam_t = lam_f
            extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
        elif lam_2d is None:
            lam_t = lam_1d[t]
            extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
        else:
            lam_t = lam_2d[t]
            extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
        if window is not None:
            t0, t1 = window
            extc = jnp.where((t >= t0) & (t < t1), extc, jnp.zeros_like(extc))
        g = g.at[targets_j, rec_ext].add(w_ext_j * extc.astype(dtype))
        if not has_zero:
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V, w, ref), spk = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
        else:
            # Declared intra-step fixed point, round 0 as above (g already holds
            # the same-step arrivals); further rounds re-step as in body, until
            # quiescence or R rounds.
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V1, w1, ref1), spk0 = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
            if rep_j is not None:
                spk0 = jnp.where(rep_j, rep_po, spk0)
            if mask_j is not None:
                # Silencing acts before the fixed point: a masked cell never
                # enters spk_acc, so its spike reaches no delay-0 target.
                spk0 = spk0 & ~(mask_j & (t >= onset_j))
            V_pre, w_pre, ref_pre = V, w, ref

            def _fp_cond(fp_carry):
                _, _, _, _, _, prev_new, done = fp_carry
                return (done < n_rounds) & jnp.any(prev_new & has_out0_j)

            def _fp_round(fp_carry):
                spk_acc, Vc, wc, refc, gc, prev_new, done = fp_carry
                if gain0_j is None:
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], absw0_j, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                else:
                    g_t0 = gain_sched_j[t]
                    w0_eff = absw0_j * jnp.where(
                        gain0_j, g_t0, jnp.asarray(1.0, dtype))
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], w0_eff, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                gc = gc + add0
                I_2 = jnp.sum(gc * (rev_j[None, :] - V_pre[:, None]), axis=1) + I_t
                (Vn, wn, refn), spkn = adex_step(
                    (V_pre, w_pre, ref_pre), I_2, params, dt_ms, step_order
                )
                if rep_j is not None:
                    spkn = jnp.where(rep_j, rep_po, spkn)
                if mask_j is not None:
                    spkn = spkn & ~(mask_j & (t >= onset_j))
                take = ~spk_acc
                Vc = jnp.where(take, Vn, Vc)
                wc = jnp.where(take, wn, wc)
                refc = jnp.where(take, refn, refc)
                return (spk_acc | spkn, Vc, wc, refc, gc, spkn & ~spk_acc, done + 1)

            spk_acc, V, w, ref, g, _prev, n_done = jax.lax.while_loop(
                _fp_cond, _fp_round, (spk0, V1, w1, ref1, g, spk0, jnp.int32(0))
            )
            spk = spk_acc
            # As in body: final-round newly spiked cells with delay-0 out-edges.
            n_unres = jnp.sum(_prev & has_out0_j).astype(jnp.int32)
        if rep_j is not None:
            spk = jnp.where(rep_j, rep_po, spk)
        if mask_j is not None:
            spk = spk & ~(mask_j & (t >= onset_j))
        ring = ring.at[t % L].set(spk)
        if has_zero:
            out = (spk, V, n_unres, n_done) if record_v else (spk, n_unres, n_done)
        else:
            out = (spk, V) if record_v else spk
        return (V, w, ref, g, ring, ering, t + 1, rkey), out

    def bodyM(carry, I_t):
        """Delayed multi-target drive: one Poisson stream per entry.

        Each entry key splits once per step (the mapping path's one-split
        key stream, one per entry), and entry ``e`` draws its own segment
        under its own stream, so one entry's seed leaves every other
        entry's counts unchanged and entry order changes nothing. Each
        target ``m`` reads the count drawn ``md_j[m]`` steps earlier from
        the ``(L, M)`` external ring and adds ``mw_j[m] * count`` at
        ``[mt_j[m], mr_j[m]]``; per-target delays ride the same ring with
        one column per target. The intra-step fixed-point branches below
        match :func:`body` round for round.
        """
        V, w, ref, g, ring, ering, t, rkey = carry
        split_all = jax.vmap(jax.random.split)(rkey)
        if rep_j is not None:
            rep_po = jax.random.poisson(
                jax.random.fold_in(rep_base, t), rep_lam_j, shape=(N,)) > 0
        rkey, kss = split_all[:, 0], split_all[:, 1]
        I_t = I_const if I_t is None else I_t
        g = g * decay_j[None, :] + _arrivals(ring, t)
        rows_m = (t - md_j) % L
        earr = ering[rows_m, jnp.arange(M)]
        g = g.at[mt_j, mr_j].add(mw_j * earr.astype(dtype))
        if not has_zero:
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V, w, ref), spk = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
        else:
            I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
            (V1, w1, ref1), spk0 = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
            if rep_j is not None:
                spk0 = jnp.where(rep_j, rep_po, spk0)
            if mask_j is not None:
                # Silencing acts before the fixed point: a masked cell never
                # enters spk_acc, so its spike reaches no delay-0 target.
                spk0 = spk0 & ~(mask_j & (t >= onset_j))
            V_pre, w_pre, ref_pre = V, w, ref

            def _fp_cond(fp_carry):
                _, _, _, _, _, prev_new, done = fp_carry
                return (done < n_rounds) & jnp.any(prev_new & has_out0_j)

            def _fp_round(fp_carry):
                spk_acc, Vc, wc, refc, gc, prev_new, done = fp_carry
                if gain0_j is None:
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], absw0_j, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                else:
                    g_t0 = gain_sched_j[t]
                    w0_eff = absw0_j * jnp.where(
                        gain0_j, g_t0, jnp.asarray(1.0, dtype))
                    add0 = jax.ops.segment_sum(
                        jnp.where(prev_new[pre0_j], w0_eff, 0.0),
                        seg0_j,
                        num_segments=N * R,
                    ).reshape(N, R)
                gc = gc + add0
                I_2 = jnp.sum(gc * (rev_j[None, :] - V_pre[:, None]), axis=1) + I_t
                (Vn, wn, refn), spkn = adex_step(
                    (V_pre, w_pre, ref_pre), I_2, params, dt_ms, step_order
                )
                if rep_j is not None:
                    spkn = jnp.where(rep_j, rep_po, spkn)
                if mask_j is not None:
                    spkn = spkn & ~(mask_j & (t >= onset_j))
                take = ~spk_acc
                Vc = jnp.where(take, Vn, Vc)
                wc = jnp.where(take, wn, wc)
                refc = jnp.where(take, refn, refc)
                return (spk_acc | spkn, Vc, wc, refc, gc, spkn & ~spk_acc, done + 1)

            spk_acc, V, w, ref, g, _prev, n_done = jax.lax.while_loop(
                _fp_cond, _fp_round, (spk0, V1, w1, ref1, g, spk0, jnp.int32(0))
            )
            spk = spk_acc
            n_unres = jnp.sum(_prev & has_out0_j).astype(jnp.int32)
        if rep_j is not None:
            spk = jnp.where(rep_j, rep_po, spk)
        if mask_j is not None:
            spk = spk & ~(mask_j & (t >= onset_j))
        ring = ring.at[t % L].set(spk)
        segs = []
        for _e, _size in enumerate(mentry_sizes):
            _off = mentry_offs[_e]
            if mlam_f is not None:
                segs.append(
                    jax.random.poisson(kss[_e], mlam_f, shape=(_size,)).astype(jnp.int32)
                )
            elif mlam_v is not None:
                segs.append(jax.random.poisson(kss[_e], mlam_v[_off:_off + _size]))
            else:
                assert mlam_2d is not None
                segs.append(jax.random.poisson(kss[_e], mlam_2d[t, _off:_off + _size]))
        extc = jnp.concatenate(segs).astype(jnp.int32)
        if window is not None:
            t0, t1 = window
            extc = jnp.where((t >= t0) & (t < t1), extc, jnp.zeros_like(extc))
        ering = ering.at[t % L].set(extc)
        if has_zero:
            out = (spk, V, n_unres, n_done) if record_v else (spk, n_unres, n_done)
        else:
            out = (spk, V) if record_v else spk
        return (V, w, ref, g, ring, ering, t + 1, rkey), out

    if ext is not None:
        rkey0 = ext["key"]
        ering0 = jnp.zeros((L, M), jnp.int32)
    elif extm is not None:
        rkey0 = mkey0
        ering0 = jnp.zeros((L, M), jnp.int32)
    else:
        rkey0 = jnp.zeros((), jnp.int32)
        ering0 = jnp.zeros((0,), jnp.int32)
    carry0 = (
        V0,
        jnp.zeros((N,), dtype),
        jnp.zeros((N,), jnp.int32),
        jnp.zeros((N, R), dtype),
        jnp.zeros((L, N), bool),
        ering0,
        jnp.int32(0),
        rkey0,
    )
    scan_body = body0 if (ext is not None and d_ext == 0) else (bodyM if use_multi else body)
    (V, w, ref, g, _, _, _, _), ys = jax.lax.scan(scan_body, carry0, I_all, length=n_steps)
    result = {
        "spikes": ys[0] if (record_v or has_zero) else ys,
        "V": V,
        "w": w,
        "ref": ref,
        "g": g,
    }
    if record_v:
        result["V_mV"] = ys[1]
    if event_info is not None:
        result["event_delivery"] = event_info
    if has_zero:
        result["fixed_point_unresolved"] = ys[2] if record_v else ys[1]
        result["fixed_point_rounds"] = ys[3] if record_v else ys[2]
        _unres_h = np.asarray(
            result["fixed_point_unresolved"], dtype=np.int64
        ).reshape(-1)
        _n_unres = int(_unres_h.sum())
        if _n_unres:
            raise ValueError(
                f"intra-step fixed point reached its cap intra_step_rounds={n_rounds} "
                f"with {_n_unres} undelivered same-step spikes, "
                f"first at step {int(np.argmax(_unres_h > 0))}; raise intra_step_rounds"
            )
    return result


def simulate_adex_hdp_registered(
    params: AdExParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight_nS=None,
    edge_receptor=None,
    edge_delay_steps=None,
    receptor_reversal_mV=(),
    receptor_tau_ms=(),
    decay: str,
    step_order: str = "default",
    I_ext_pA=None,
    V0_mV=None,
    record_v: bool = False,
    ext_targets=None,
    ext_lambda=None,
    ext_weight_nS=None,
    ext_receptor=None,
    ext_delay_steps=None,
    key=None,
    ext_t_start_steps=None,
    ext_t_stop_steps=None,
    hdp_rule: str,
    hdp_rule_params: Mapping[str, Any] | None = None,
    plasticity_mask=None,
    hdp_key=None,
    seed: int | None = None,
    record_w_trace: bool = True,
    state: Mapping[str, Any] | None = None,
    return_state: bool = False,
    spike_out_mask=None,
    spike_out_onset_step=0,
    spike_replace_mask=None,
    spike_replace_lam=None,
    spike_replace_key=None,
    edge_gain_mask=None,
    edge_gain_schedule=None,
    edge0_gain_mask=None,
) -> dict[str, Any]:
    """AdEx conductance network with a registered HDP rule (lane 7, gap G3).

    Same conductance network as :func:`simulate_adex`; same validation,
    delays, external drive, and per-step order, except each edge's
    conductance weight is the plastic magnitude ``|w|`` (nS) carried under
    HDP: the HDP layer (``jaxfne._hdp_family_kernel``) supplies presynaptic
    arrivals, postsynaptic spikes, ``aux``, ``H`` and ``W`` to any rule in
    ``hdp_rule.list_registered_hdp_rules()``, while the AdEx family supplies
    its step ``(state, I, params, dt) -> (state, spikes, v)``.

    Per step ``t``, in order: arrivals are read from the delay ring (as in
    :func:`simulate_adex`); conductance ``g`` is built from the current
    plastic magnitudes; current and :func:`adex_step` give this step's
    spikes; the rule observes delayed arrivals as per-edge ``pre_sp`` (so
    the qualified chain ``event_{t-d} -> H_t`` holds) and current spikes,
    and its update integrates ``H``/``aux``/``|w|`` for step ``t + 1``.

    AdEx notes: the family has no recovery variable, so the rule context
    carries a zero ``u`` coordinate; the current sign lives in the receptor
    reversal, so every edge updates by magnitude and ``exc_mask`` is
    ``None``. Only ``edge_weight`` rules run here: a rule declaring any
    other target (e.g. ``drive_bias``) is refused naming the rule. Unknown
    ``hdp_rule`` names and unconsumed ``hdp_rule_params`` keys fail closed.
    Identity rule params (e.g. ``synthetic_presyn_gain`` defaults) leave the
    ``simulate_adex`` outputs bit-identical provided baseline ``|w|`` lies
    within the rule's ``w_bounds`` (the declared clip still applies).
    ``spike_out_mask``/``spike_out_onset_step`` are refused here (fail closed):
    the rule observes the emitted spikes, so masking would need a rule-gated
    form of the same :func:`simulate_adex` rule; silence on the HDP path is
    not realized. ``spike_replace_mask``/``spike_replace_lam``/
    ``spike_replace_key`` are refused for the same reason: replacement would
    need its own rule-gated form; pass the replacement to
    :func:`simulate_adex` only.

    ``plasticity_mask`` is ``None`` (every edge plastic) or a per-edge
    finite mask; ``hdp_key`` seeds the per-step rule-noise stream, else
    ``seed`` (an int seed) via a domain-separated fold of
    ``PRNGKey(seed)`` so an explicit ``seed=0`` never equals the
    all-None ``PRNGKey(0)`` below, else ``fold_in`` of the external key
    when present, else a fixed base; never drawn from, so
    the external stream stays bit-identical. This kernel has no model
    route (the adex ``simulate()`` path runs ``simulate_adex`` and
    refuses HDP inputs), so callers pass ``seed`` explicitly; a non-int
    seed is refused by name.
    ``record_w_trace`` records per-step ``|w|``. Returns the
    :func:`simulate_adex` dict plus ``H_final``, ``w_final_nS``,
    ``aux_final``, ``H_trace``, ``w_trace_nS`` (when recorded),
    ``hdp_rule`` and ``hdp_rule_registered``.

    ``state`` resumes a run from a state ``return_state=True`` returned:
    the full final carry (``V``, adaptation current, refractory counters,
    per-receptor conductances, the delay ring, plastic weights ``W``, rule
    ``aux`` traces, ``H``, the step counter, and the external-drive and HDP
    keys as they stood after the last step). A resumed call with the same
    network arguments continues the unsegmented run bit-for-bit: spikes,
    final ``V``/``W``/``aux``, and the Poisson drive (the key chain
    continues from the state, so ``key``/``seed`` only rebuild the same
    streams). A state from a different network size is refused, as are a
    per-step ``ext_lambda`` schedule, a drive window, and ``V0_mV`` with
    ``state`` (the carry restores ``V``). Defaults are unchanged: a call
    without these arguments is bit-identical to the baseline.
    """
    from ._hdp_family_kernel import (  # lazy: hdp_rule owns .emitters, so top-level would cycle
        apply_registered_rule_update,
        check_rule_family_support,
        hdp_init_carry,
        resolve_registered_rule,
    )
    # (registry access goes through resolve_registered_rule above)
    if step_order not in _STEP_ORDERS:
        raise ValueError(f"step_order must be 'default' or 'brian2', got {step_order!r}")
    if decay not in _DECAYS:
        raise ValueError(f"decay must be 'exp' or 'euler', got {decay!r}")
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not dt_ms > 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}")

    N = int(jnp.shape(params.C_pF)[0])
    if (spike_out_mask is not None or spike_out_onset_step != 0
            or spike_replace_mask is not None or spike_replace_lam is not None
            or spike_replace_key is not None):
        raise ValueError(
            "spike_out_mask/spike_replace_mask is not realized for the HDP kernel "
            "simulate_adex_hdp_registered: silence/replacement there would need its own "
            "rule-gated masking; pass the mask to simulate_adex only"
        )
    if (edge_gain_mask is not None or edge_gain_schedule is not None
            or edge0_gain_mask is not None):
        raise ValueError(
            "edge_gain_mask/edge_gain_schedule is not realized for the HDP kernel "
            "simulate_adex_hdp_registered: gain there would need its own "
            "rule-gated scaling; pass the gain to simulate_adex only"
        )
    dtype = jnp.asarray(params.C_pF).dtype
    if N * max(len(np.reshape(receptor_tau_ms, -1)), 1) >= 2**31:
        raise ValueError(f"N * n_receptors must be < 2**31 (int32 segment ids); N = {N}")
    rev = np.asarray(receptor_reversal_mV, dtype=np.float64).reshape(-1)
    tau = np.asarray(receptor_tau_ms, dtype=np.float64).reshape(-1)
    if rev.shape != tau.shape:
        raise ValueError(
            "receptor_reversal_mV and receptor_tau_ms must have the same length; "
            f"got {rev.size} and {tau.size}"
        )
    if np.any(tau <= 0):
        raise ValueError(f"receptor_tau_ms must be > 0 for every receptor; got {tau.tolist()}")
    if decay == "euler" and np.any(tau <= dt_ms):
        raise ValueError(
            f"decay='euler' needs receptor_tau_ms > dt_ms ({dt_ms}); got {tau.tolist()}"
        )
    R = int(rev.size)

    edges = _check_edges(
        N,
        R,
        {
            "edge_pre": edge_pre,
            "edge_post": edge_post,
            "edge_weight_nS": edge_weight_nS,
            "edge_receptor": edge_receptor,
            "edge_delay_steps": edge_delay_steps,
        },
    )
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "receptor": np.zeros((0,), np.int64),
            "delay": np.zeros((0,), np.int64),
        }
    E = int(edges["pre"].shape[0])
    delay_eff = edges["delay"] + (1 if step_order == "brian2" else 0)
    ext = _check_external_drive(
        N,
        R,
        {
            "ext_targets": ext_targets,
            "ext_lambda": ext_lambda,
            "ext_weight_nS": ext_weight_nS,
            "ext_receptor": ext_receptor,
            "ext_delay_steps": ext_delay_steps,
            "key": key,
        },
        allow_zero_delay=True,
    )
    L_edge = int(delay_eff.max()) if E else 0
    if ext is None:
        if ext_t_start_steps is not None or ext_t_stop_steps is not None:
            raise ValueError(
                "ext_t_start_steps/ext_t_stop_steps need the external drive; got "
                f"{ext_t_start_steps!r}/{ext_t_stop_steps!r} with no ext_targets — "
                "pass the six drive arguments or neither bound"
            )
        L = L_edge if E else 1
        d_ext = 0
        M = 0
        window = None
    else:
        d_ext = int(ext["delay"]) + (1 if step_order == "brian2" else 0)
        L = max(L_edge, d_ext, 1)
        M = int(ext["targets"].shape[0])
        window = _check_drive_window(n_steps, ext_t_start_steps, ext_t_stop_steps)

    if ext is not None and int(ext["delay"]) == 0:
        raise ValueError(
            "ext_delay_steps=0 requests a same-step drive, which the registered-HDP "
            "kernel does not realize: use simulate_adex, or a delay >= 1 step"
        )
    descriptor, rule_step, rule_params = resolve_registered_rule(hdp_rule, hdp_rule_params)
    check_rule_family_support(descriptor, family="adex", rule_name=hdp_rule)
    if plasticity_mask is not None:
        _mask_raw = np.asarray(plasticity_mask)
        if _mask_raw.ndim != 1 or _mask_raw.shape[0] != E:
            raise ValueError(
                f"plasticity_mask must have shape ({E},), got {_mask_raw.shape}"
            )
        if not bool(np.all(np.isfinite(_mask_raw))):
            raise ValueError("plasticity_mask must be finite (got NaN/inf)")
        plastic_edge = jnp.asarray(_mask_raw > 0.5)
    else:
        plastic_edge = None

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    post_j = jnp.asarray(edges["post"], jnp.int32)
    seg_j = jnp.asarray(edges["post"] * R + edges["receptor"], jnp.int32)
    wmag0 = jnp.asarray(np.abs(edges["weight"]), dtype)
    lag_j = jnp.asarray(delay_eff, jnp.int32)
    decay_r = np.exp(-dt_ms / tau) if decay == "exp" else 1.0 - dt_ms / tau
    decay_j = jnp.asarray(decay_r, dtype)
    rev_j = jnp.asarray(rev, dtype)
    if ext is not None:
        targets_j = jnp.asarray(ext["targets"], jnp.int32)
        w_ext_j = jnp.asarray(ext["weight"], dtype)
        rec_ext = int(ext["receptor"])
        lam_kind, lam_val = ext["lambda"]
        if lam_kind == "scalar":
            lam_f: Any = float(lam_val)
            lam_1d = None
            lam_2d = None
        else:
            lam_f = None
            sched = np.asarray(lam_val, dtype=np.float64)
            if sched.ndim == 1:
                if sched.shape[0] != n_steps:
                    raise ValueError(
                        "ext_lambda schedule with one shared rate must have shape "
                        f"({n_steps},); got {sched.shape}"
                    )
                lam_1d = jnp.asarray(sched, dtype)
                lam_2d = None
            else:
                if sched.shape != (n_steps, M):
                    raise ValueError(
                        "ext_lambda schedule with per-target rates must have shape "
                        f"({n_steps}, {M}); got {sched.shape}"
                    )
                lam_1d = None
                lam_2d = jnp.asarray(sched, dtype)

    # A constant current is closed over, so no [n_steps, N] array is built for it.
    I_const = jnp.zeros((N,), dtype)
    I_all = None
    if I_ext_pA is not None:
        I_arr = jnp.asarray(I_ext_pA, dtype)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext_pA must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )

    if V0_mV is None:
        V0 = jnp.asarray(params.EL_mV, dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape != (N,):
            raise ValueError(f"V0_mV must be None or shape ({N},); got {V0.shape}")

    H0, aux0, b0 = hdp_init_carry(descriptor, n_neurons=N, n_edges=E, jdtype=dtype)
    dt_arr = jnp.asarray(dt_ms, dtype=dtype)
    u_zero = jnp.zeros((N,), dtype)
    if hdp_key is not None:
        try:
            jax.random.split(hdp_key)
        except Exception:
            raise ValueError(f"hdp_key must be a JAX PRNG key; got {hdp_key!r}")
        rule_base = hdp_key
    elif seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an int or None; got {seed!r}")
        rule_base = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0xAD3E)
    elif ext is not None:
        rule_base = jax.random.fold_in(ext["key"], 9147)
    else:
        rule_base = jax.random.PRNGKey(0)

    if state is not None:
        if not isinstance(state, Mapping):
            raise ValueError(
                f"state must be a mapping from return_state=True; got {type(state).__name__}"
            )
        missing = [k for k in _HDP_STATE_KEYS if k not in state]
        if missing:
            raise ValueError(
                f"state is missing keys {missing}; pass back the full state mapping unmodified"
            )
        for name, want in (("n_neurons", N), ("n_edges", E),
                           ("n_receptors", R), ("ring_length", L),
                           ("n_ext_targets", M)):
            if int(state[name]) != int(want):
                raise ValueError(
                    f"state {name}={state[name]} does not match this network "
                    f"({name}={want}); a state resumes only the network that wrote it"
                )
        if V0_mV is not None:
            raise ValueError(
                "V0_mV does not combine with state; the state carry restores V"
            )
        if window is not None:
            raise ValueError(
                "ext_t_start_steps/ext_t_stop_steps do not combine with state; "
                "a window is per call, while a state continues the step counter"
            )
        if ext is not None and (lam_1d is not None or lam_2d is not None):
            raise ValueError(
                "a per-step ext_lambda schedule does not combine with state; "
                "it is indexed per call, while a state continues the step counter"
            )
        try:
            jax.random.split(state["hdp_key"])
        except Exception:
            raise ValueError("state['hdp_key'] must be a JAX PRNG key") from None
        rule_base = state["hdp_key"]
        t0 = state["step"]
        if isinstance(t0, bool) or not isinstance(t0, (int, np.integer)):
            raise ValueError(
                f"state['step'] must be an int step counter; got {t0!r}"
            )
        t0 = int(t0)
        if t0 < 0:
            raise ValueError(f"state['step'] must be >= 0; got {t0}")
        carry_from_state = (
            jnp.asarray(state["V_mV"], dtype),
            jnp.asarray(state["w_pA"], dtype),
            jnp.asarray(state["ref_steps"], np.int32),
            jnp.asarray(state["g_nS"], dtype),
            jnp.asarray(state["delay_ring"], bool),
            jnp.asarray(state["ext_ring"], np.int32),
            jnp.int32(t0),
            state["ext_key"],
            jnp.asarray(state["H"], dtype),
            jnp.asarray(state["W_nS"], dtype),
            jnp.asarray(state["aux"], dtype),
            jnp.asarray(state["bias"], dtype),
            jnp.asarray(state["prev_spikes"], dtype),
        )
        _fresh_shapes = (
            V0.shape, (N,), (N,), (N, R), (L, N),
            ((L, M) if ext is not None else (0,)),
            (),
            (tuple(jnp.shape(ext["key"])) if ext is not None else ()),
            H0.shape, wmag0.shape, aux0.shape,
            b0.shape, (N,),
        )
        for value, want_shape in zip(carry_from_state, _fresh_shapes):
            if tuple(jnp.shape(value)) != tuple(want_shape):
                raise ValueError(
                    f"state carry shape {tuple(jnp.shape(value))} does not match "
                    f"this network (expected {tuple(want_shape)})"
                )

    def body_hdp(carry, I_t):
        V, w, ref, g, ring, ering, t, rkey, H, wmag, aux, bias, prev_spk = carry
        if ext is not None:
            rkey, ks = jax.random.split(rkey)
        I_t = I_const if I_t is None else I_t
        if E:
            arrived = ring[(t - lag_j) % L, pre_j]
            arrived_f = arrived.astype(dtype)
            added = jax.ops.segment_sum(
                wmag * arrived_f, seg_j, num_segments=N * R
            ).reshape(N, R)
        else:
            arrived_f = jnp.zeros((0,), dtype)
            added = jnp.zeros((N, R), dtype)
        g = g * decay_j[None, :] + added
        if ext is not None:
            earr = ering[(t - d_ext) % L]
            g = g.at[targets_j, rec_ext].add(w_ext_j * earr.astype(dtype))
        I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
        (V, w, ref), spk = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
        spk_f = spk.astype(dtype)
        H_next, wmag_next, aux_next, _ = apply_registered_rule_update(
            descriptor=descriptor,
            rule_step=rule_step,
            H=H,
            aux=aux,
            bias=bias,
            v=V,
            u=u_zero,
            spikes=spk_f,
            prev_spikes=prev_spk,
            syn_state=wmag * arrived_f,
            w=wmag,
            pre=pre_j,
            post=post_j,
            exc_mask=None,
            dt=dt_arr,
            n_neurons=N,
            rule_params=rule_params,
            key=jax.random.fold_in(rule_base, t),
            pre_sp=arrived_f,
            rule_name=hdp_rule,
            plasticity_mask=plastic_edge,
        )
        ring = ring.at[t % L].set(spk)
        if ext is not None:
            if lam_1d is None and lam_2d is None:
                lam_t = lam_f
            elif lam_2d is None:
                lam_t = lam_1d[t]
            else:
                lam_t = lam_2d[t]
            extc = jax.random.poisson(ks, lam_t, shape=(M,)).astype(jnp.int32)
            if window is not None:
                t0, t1 = window
                extc = jnp.where((t >= t0) & (t < t1), extc, jnp.zeros_like(extc))
            ering = ering.at[t % L].set(extc)
        carry_out = (
            V, w, ref, g, ring, ering, t + 1, rkey,
            H_next, wmag_next, aux_next, bias, spk_f,
        )
        out = (spk, V, H_next, wmag_next) if record_w_trace else (spk, V, H_next)
        return carry_out, out

    if ext is not None:
        rkey0 = ext["key"]
        ering0 = jnp.zeros((L, M), jnp.int32)
    else:
        rkey0 = jnp.zeros((), jnp.int32)
        ering0 = jnp.zeros((0,), jnp.int32)
    carry0 = (
        V0,
        jnp.zeros((N,), dtype),
        jnp.zeros((N,), jnp.int32),
        jnp.zeros((N, R), dtype),
        jnp.zeros((L, N), bool),
        ering0,
        jnp.int32(0),
        rkey0,
        H0,
        wmag0,
        aux0,
        b0,
        jnp.zeros((N,), dtype),
    )
    if state is not None:
        carry0 = carry_from_state
    (Vf, wf, reff, gf, ringf, eringf, tf, rkeyf,
     Hf, wmagf, auxf, biasf, prevf), ys = jax.lax.scan(
        body_hdp, carry0, I_all, length=n_steps
    )
    result = {
        "spikes": ys[0],
        "V": Vf,
        "w": wf,
        "ref": reff,
        "g": gf,
        "H_final": Hf,
        "w_final_nS": wmagf,
        "aux_final": auxf,
        "H_trace": ys[2],
        "hdp_rule": hdp_rule,
        "hdp_rule_registered": True,
    }
    if return_state:
        result["state"] = {
            "V_mV": Vf,
            "w_pA": wf,
            "ref_steps": reff,
            "g_nS": gf,
            "delay_ring": ringf,
            "ext_ring": eringf,
            "step": int(tf),
            "ext_key": rkeyf,
            "hdp_key": rule_base,
            "H": Hf,
            "W_nS": wmagf,
            "aux": auxf,
            "bias": biasf,
            "prev_spikes": prevf,
            "n_neurons": N,
            "n_edges": E,
            "n_receptors": R,
            "ring_length": L,
            "n_ext_targets": M,
        }
    if record_v:
        result["V_mV"] = ys[1]
    if record_w_trace:
        result["w_trace_nS"] = ys[3]
    return result
