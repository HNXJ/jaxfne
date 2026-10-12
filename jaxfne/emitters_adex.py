"""AdEx network kernel: conductance synapses and a per-edge delay ring.

Reference: Tahvili, Vinck, di Volo, PLoS Comput Biol 2026 (DOI 10.1371/journal.pcbi.1014378),
Methods Eqs. 15-18 and Tables 1-2. ``adex_step`` mirrors the CAMINOS etude's ``adex_step``
(``artifacts/etudes/caminos/caminos_jax.py``, reference branch ``l2-caminos``); the cell table
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
    columns = {}
    for field in _PARAM_FIELDS:
        values = [
            float(overrides.get(lab, {}).get(field, ADEX_CELL_TYPE_DEFAULTS[lab][field]))
            for lab in labels
        ]
        columns[field] = jnp.asarray(np.asarray(values, dtype=np.float64))
    return AdExParams(labels=labels, **columns)


def adex_step(state, I_pA, params: AdExParams, dt_ms: float, step_order: str = "default"):
    """One explicit-Euler step of the AdEx equations for a vector of cells.

    Cell model: Brette & Gerstner (2005), doi:10.1152/jn.00686.2005; the refractory
    clamp and the spike threshold ``VT + 5 * Delta`` follow Tahvili et al. (2026).

    ``state`` is ``(V, w, ref)``: membrane voltage (mV), adaptation (pA), and refractory
    steps remaining (int). ``I_pA`` is injected current. A cell with ``ref > 0`` is clamped
    at ``Vreset`` while ``w`` keeps evolving with ``V = Vreset``. An active cell spikes when
    its updated V reaches ``VT + 5 * Delta``; on spike V is reset, ``w += b``, and ``ref`` is
    set to ``round(Tref / dt)`` (one fewer under ``step_order="brian2"``).

    Returns ``((V, w, ref), spk)`` with boolean spike vector ``spk``. Pure ``jax.numpy``;
    dtype follows the inputs.
    """
    if step_order not in _STEP_ORDERS:
        raise ValueError(f"step_order must be 'default' or 'brian2', got {step_order!r}")
    V, w, ref = state
    p = params
    v_thr = p.VT_mV + 5.0 * p.Delta_mV
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

_EXT_ARGS = (
    "ext_targets", "ext_lambda", "ext_weight_nS", "ext_receptor", "ext_delay_steps", "key",
)


def _check_external_drive(N, R, given):
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

    lam = _nonneg_float("ext_lambda", given["ext_lambda"])
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
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, np.integer, float, np.floating))
        or not math.isfinite(float(delay))
        or float(delay) != round(float(delay))
        or int(delay) < 1
    ):
        raise ValueError(f"ext_delay_steps must be an integer >= 1; got {delay!r}")
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
       under ``step_order="brian2"``) from an int32 ring of the same length;
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
       carry no external drive.

    The external drive is opt-in through six keyword arguments that are all given or all
    None: ``ext_targets`` (int ``[M]`` cell indices, unique, in range), ``ext_lambda``
    (mean count per step per target, >= 0), ``ext_weight_nS`` (conductance per count, >= 0),
    ``ext_receptor`` (a valid receptor index), ``ext_delay_steps`` (int >= 1; plus 1 under
    ``step_order="brian2"``, as for edges), and ``key`` (a JAX PRNG key).

    ``I_ext_pA`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``. ``V0_mV`` is None
    (rest at ``EL``) or shape ``[N]``. ``edge_*`` arrays are all given or all None. Edge
    delays must be integers >= 1.

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV`` (``[n_steps, N]``, after each
    step's update, only if ``record_v``: 4.6 GB in float32 at 9500 cells and 120000 steps),
    and the final ``V``, ``w``, ``ref`` and ``g``.
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
    )
    L_edge = int(delay_eff.max()) if E else 0
    if ext is None:
        L = L_edge if E else 1
        d_ext = 0
        M = 0
    else:
        d_ext = int(ext["delay"]) + (1 if step_order == "brian2" else 0)
        L = max(L_edge, d_ext, 1)
        M = int(ext["targets"].shape[0])

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    seg_j = jnp.asarray(edges["post"] * R + edges["receptor"], jnp.int32)
    absw_j = jnp.asarray(np.abs(edges["weight"]), dtype)
    lag_j = jnp.asarray(delay_eff, jnp.int32)
    decay_r = np.exp(-dt_ms / tau) if decay == "exp" else 1.0 - dt_ms / tau
    decay_j = jnp.asarray(decay_r, dtype)
    rev_j = jnp.asarray(rev, dtype)
    if ext is not None:
        targets_j = jnp.asarray(ext["targets"], jnp.int32)
        w_ext_j = jnp.asarray(ext["weight"], dtype)
        lam_f = float(ext["lambda"])
        rec_ext = int(ext["receptor"])

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
        I_tot = jnp.sum(g * (rev_j[None, :] - V[:, None]), axis=1) + I_t
        (V, w, ref), spk = adex_step((V, w, ref), I_tot, params, dt_ms, step_order)
        ring = ring.at[t % L].set(spk)
        if ext is not None:
            extc = jax.random.poisson(ks, lam_f, shape=(M,)).astype(jnp.int32)
            ering = ering.at[t % L].set(extc)
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
    (V, w, ref, g, _, _, _, _), ys = jax.lax.scan(body, carry0, I_all, length=n_steps)
    result = {
        "spikes": ys[0] if record_v else ys,
        "V": V,
        "w": w,
        "ref": ref,
        "g": g,
    }
    if record_v:
        result["V_mV"] = ys[1]
    return result
