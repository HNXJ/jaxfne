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
    value; NaN (or any other non-finite value, or a bool) is refused — absence
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
    arg = jnp.minimum((V - p.VT_mV) / p.Delta_mV, _EXP_ARG_MAX)
    dV = (
        p.gL_nS * (p.EL_mV - V)
        + p.gL_nS * p.Delta_mV * jnp.exp(arg)
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
    ext_t_start_steps=None,
    ext_t_stop_steps=None,
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
    (mean count per step per target, >= 0 — either a scalar float, or a per-step
    schedule array ``[n_steps]`` (shared by every target) or ``[n_steps, M]`` (per
    target); finite values >= 0, bools refused), ``ext_weight_nS`` (conductance per count, >= 0),
    ``ext_receptor`` (a valid receptor index), ``ext_delay_steps`` (int >= 0; 0 is the
    same-step drive above and is refused under ``step_order="brian2"``; a delay >= 1
    gains 1 under brian2, as for edges), and
    ``key`` (a JAX PRNG key). A scalar
    ``ext_lambda`` takes the historical path bit-identically; a schedule is drawn
    per step (``poisson(ks, lam[t])`` under the same one-split-per-step key
    stream) and validated against ``(n_steps, M)`` here, where both are known.

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

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV`` (``[n_steps, N]``, after each
    step's update, only if ``record_v``: 4.6 GB in float32 at 9500 cells and 120000 steps),
    and the final ``V``, ``w``, ``ref`` and ``g``. With delay-0 edges, also
    ``fixed_point_unresolved`` (int32 ``[n_steps]``: per step, the number of
    cells that newly spiked in the final round and own delay-0 out-edges —
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

    N = int(jnp.shape(params.C_pF)[0])
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

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    seg_j = jnp.asarray(edges["post"] * R + edges["receptor"], jnp.int32)
    absw_j = jnp.asarray(np.abs(edges["weight"]), dtype)
    lag_j = jnp.asarray(delay_eff, jnp.int32)
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

    def body(carry, I_t):
        V, w, ref, g, ring, ering, t, rkey = carry
        if ext is not None:
            rkey, ks = jax.random.split(rkey)
        I_t = I_const if I_t is None else I_t
        if E:
            arrived = ring[(t - lag_j) % L, pre_j]
            added = jax.ops.segment_sum(
                jnp.where(arrived, absw_j, 0.0), seg_j, num_segments=N * R
            ).reshape(N, R)
        else:
            added = jnp.zeros((N, R), dtype)
        g = g * decay_j[None, :] + added
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
            V_pre, w_pre, ref_pre = V, w, ref

            def _fp_cond(fp_carry):
                _, _, _, _, _, prev_new, done = fp_carry
                return (done < n_rounds) & jnp.any(prev_new & has_out0_j)

            def _fp_round(fp_carry):
                spk_acc, Vc, wc, refc, gc, prev_new, done = fp_carry
                add0 = jax.ops.segment_sum(
                    jnp.where(prev_new[pre0_j], absw0_j, 0.0),
                    seg0_j,
                    num_segments=N * R,
                ).reshape(N, R)
                gc = gc + add0
                I_2 = jnp.sum(gc * (rev_j[None, :] - V_pre[:, None]), axis=1) + I_t
                (Vn, wn, refn), spkn = adex_step(
                    (V_pre, w_pre, ref_pre), I_2, params, dt_ms, step_order
                )
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
        I_t = I_const if I_t is None else I_t
        if E:
            arrived = ring[(t - lag_j) % L, pre_j]
            added = jax.ops.segment_sum(
                jnp.where(arrived, absw_j, 0.0), seg_j, num_segments=N * R
            ).reshape(N, R)
        else:
            added = jnp.zeros((N, R), dtype)
        g = g * decay_j[None, :] + added
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
            V_pre, w_pre, ref_pre = V, w, ref

            def _fp_cond(fp_carry):
                _, _, _, _, _, prev_new, done = fp_carry
                return (done < n_rounds) & jnp.any(prev_new & has_out0_j)

            def _fp_round(fp_carry):
                spk_acc, Vc, wc, refc, gc, prev_new, done = fp_carry
                add0 = jax.ops.segment_sum(
                    jnp.where(prev_new[pre0_j], absw0_j, 0.0),
                    seg0_j,
                    num_segments=N * R,
                ).reshape(N, R)
                gc = gc + add0
                I_2 = jnp.sum(gc * (rev_j[None, :] - V_pre[:, None]), axis=1) + I_t
                (Vn, wn, refn), spkn = adex_step(
                    (V_pre, w_pre, ref_pre), I_2, params, dt_ms, step_order
                )
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
        ring = ring.at[t % L].set(spk)
        if has_zero:
            out = (spk, V, n_unres, n_done) if record_v else (spk, n_unres, n_done)
        else:
            out = (spk, V) if record_v else spk
        return (V, w, ref, g, ring, ering, t + 1, rkey), out

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
    )
    scan_body = body0 if (ext is not None and d_ext == 0) else body
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
) -> dict[str, Any]:
    """AdEx conductance network with a registered HDP rule (lane 7, gap G3).

    Same conductance network as :func:`simulate_adex` — same validation,
    delays, external drive, and per-step order — except each edge's
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

    ``plasticity_mask`` is ``None`` (every edge plastic) or a per-edge
    finite mask; ``hdp_key`` seeds the per-step rule-noise stream, else
    ``seed`` (an int seed) via a domain-separated fold of
    ``PRNGKey(seed)`` so an explicit ``seed=0`` never equals the
    all-None ``PRNGKey(0)`` below, else ``fold_in`` of the external key
    when present, else a fixed base — never drawn from, so
    the external stream stays bit-identical. This kernel has no model
    route (the adex ``simulate()`` path runs ``simulate_adex`` and
    refuses HDP inputs), so callers pass ``seed`` explicitly; a non-int
    seed is refused by name.
    ``record_w_trace`` records per-step ``|w|``. Returns the
    :func:`simulate_adex` dict plus ``H_final``, ``w_final_nS``,
    ``aux_final``, ``H_trace``, ``w_trace_nS`` (when recorded),
    ``hdp_rule`` and ``hdp_rule_registered``.
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
    (V, w, ref, g, _, _, _, _, Hf, wmagf, auxf, _, _), ys = jax.lax.scan(
        body_hdp, carry0, I_all, length=n_steps
    )
    result = {
        "spikes": ys[0],
        "V": V,
        "w": w,
        "ref": ref,
        "g": g,
        "H_final": Hf,
        "w_final_nS": wmagf,
        "aux_final": auxf,
        "H_trace": ys[2],
        "hdp_rule": hdp_rule,
        "hdp_rule_registered": True,
    }
    if record_v:
        result["V_mV"] = ys[1]
    if record_w_trace:
        result["w_trace_nS"] = ys[3]
    return result
