"""Hodgkin-Huxley network kernel: OU noise and graded tanh synapses on an edge list.

Reference: Nejat, Sherfey, Bastos 2026, PLoS One 21(8): e0354021 (GSDR),
Methods Eq. 1; the code that produced Fig. 6 (``HNXJ/GSDR`` ``misc/main.py``
run, read as data): stock Jaxley ``HH()`` plus one per-cell ``Inoise`` OU
channel, ``GradedAMPA``/``GradedGABAa`` with tanh activation. The HH kinetic
parameters below are the classic values assumed in
``artifacts/etudes/gsdr2026/PORT_SPEC.md`` A1 (the reference names ``C_m``
only); a fidelity fix can replace them.

Units are mV, ms, mS/cm2, uF/cm2, uA/cm2 (mS * mV = uA; uA / uF = mV / ms).
The paper's Eq. 1 sign convention holds for intrinsic and synaptic currents
(positive outward hyperpolarizes); the injected drive ``I_inj`` is
depolarizing-positive (inward), matching how the reference adds its stimulus.

Numerics follow the reference solver (jaxley 0.14.0, read as data):
gates advance by exact exponential Euler (``solver_gate``), voltages by
backward Euler with the ionic and synaptic currents linearized at frozen
gates (the reference ``bwd_euler`` split: channels stepped before voltage),
compartments init at ``v = -70`` mV with gates at their rate fixed point
(``HH.init_state``), and the ``exp`` arguments are clipped at 20
(``save_exp``). The HH rate shapes are the classic ones; ``eLeak = -54.3``
is the reference value (``channels/hh.py``).

Function-level only. Configuration, construct and simulate wiring is not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# PORT_SPEC.md A1: reference HH values (jaxley channels/hh.py, read as data).
HH_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "E": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
    "Ig": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
    "Il": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
}

_PARAM_FIELDS = (
    "gNa_mS_cm2", "gK_mS_cm2", "gL_mS_cm2",
    "ENa_mV", "EK_mV", "EL_mV", "Cm_uF_cm2",
)

# PORT_SPEC.md section 3, Fig. 6 run values (the builder defaults differ;
# the run values win): per-cell tau ~ U(10, 50), sigma = clip(U(-0.1, 0.1)).
HH_OU_TAU_LO = 10.0
HH_OU_TAU_HI = 50.0
HH_OU_SIGMA_CLIP = 0.05

# PORT_SPEC.md A5: spike readout threshold of the reference loss.
HH_SPIKE_THR_MV = -20.0

# Resting voltage used for V0 when the caller passes none (reference v_init).
HH_V_REST_MV = -70.0

# Clip of exp() arguments (the reference save_exp(max_value=20.0)).
_SAVE_EXP_MAX = 20.0


def _save_exp(x):
    """``exp(min(x, 20))``: the reference overflow guard (solver_gate)."""
    return jnp.exp(jnp.minimum(x, _SAVE_EXP_MAX))


def _hh_rate_coeffs(v):
    """Classic HH rate coefficients at absolute voltage ``v`` (elementwise).

    Same kinetics as the reference ``HH.m/h/n_gate`` (absolute-voltage form);
    ``vv = v + 65`` is the rest-relative frame.
    """
    vv = v + 65.0
    am = jnp.where(jnp.abs(25.0 - vv) < 1e-7, 1.0,
                   0.1 * (25.0 - vv) / (_save_exp((25.0 - vv) / 10.0) - 1.0))
    bm = 4.0 * _save_exp(-vv / 18.0)
    ah = 0.07 * _save_exp(-vv / 20.0)
    bh = 1.0 / (_save_exp((30.0 - vv) / 10.0) + 1.0)
    an = jnp.where(jnp.abs(10.0 - vv) < 1e-7, 0.1,
                   0.01 * (10.0 - vv) / (_save_exp((10.0 - vv) / 10.0) - 1.0))
    bn = 0.125 * _save_exp(-vv / 80.0)
    return am, bm, ah, bh, an, bn


def _exp_euler_gate(x, dt_ms, alpha, beta):
    """Exact exponential-Euler step for ``dx/dt = alpha(1-x) - beta x``."""
    tau = 1.0 / (alpha + beta)
    x_inf = alpha * tau
    e = _save_exp(-dt_ms / tau)
    return x * e + x_inf * (1.0 - e)


def hh_gate_inf(v):
    """Rate fixed point ``(m, h, n)`` at voltage ``v`` (the reference init)."""
    am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
    return (am / (am + bm), ah / (ah + bh), an / (an + bn))


class HHParams:
    """Per-neuron HH parameters: float arrays of shape [N], with cell labels."""

    def __init__(self, labels: tuple[str, ...], **columns: jax.Array) -> None:
        self.labels = labels
        for field in _PARAM_FIELDS:
            setattr(self, field, columns[field])

    _fields = _PARAM_FIELDS

    def _replace(self, **kwargs: jax.Array) -> "HHParams":
        columns = {f: kwargs.get(f, getattr(self, f)) for f in _PARAM_FIELDS}
        return HHParams(labels=self.labels, **columns)


def make_hh_params(
    labels: Sequence[str],
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> HHParams:
    """Build :class:`HHParams` for one cell type per neuron.

    Defaults: classic HH values (``HH_CELL_TYPE_DEFAULTS``; PORT_SPEC.md A1).

    ``overrides`` maps a cell type to a dict of field overrides (field names as
    in :class:`HHParams`). Unknown cell types, unknown fields, and overrides
    for a cell type absent from ``labels`` raise ``ValueError``.
    """
    labels = tuple(str(lab) for lab in labels)
    for lab in labels:
        if lab not in HH_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"unknown HH cell type {lab!r}; expected one of {sorted(HH_CELL_TYPE_DEFAULTS)}"
            )
    overrides = {} if overrides is None else overrides
    for ctype, fields in overrides.items():
        if ctype not in HH_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"overrides name unknown HH cell type {ctype!r}; "
                f"expected one of {sorted(HH_CELL_TYPE_DEFAULTS)}"
            )
        if ctype not in labels:
            raise ValueError(f"overrides name cell type {ctype!r}, which is not in labels {labels}")
        for field in fields:
            if field not in _PARAM_FIELDS:
                raise ValueError(
                    f"unknown HH parameter field {field!r} for cell type {ctype!r}; "
                    f"expected one of {_PARAM_FIELDS}"
                )
    columns = {}
    for field in _PARAM_FIELDS:
        values = [
            float(overrides.get(lab, {}).get(field, HH_CELL_TYPE_DEFAULTS[lab][field]))
            for lab in labels
        ]
        columns[field] = jnp.asarray(np.asarray(values, dtype=np.float64))
    return HHParams(labels=labels, **columns)


def hh_step(state, I_inj, params: HHParams, dt_ms: float):
    """One split step of the classic HH equations for a vector of cells.

    Gates advance by exact exponential Euler at frozen voltage; then voltage
    advances by backward Euler with the ionic current linearized at the new
    gates (``(v + dt*c/Cm) / (1 + dt*G/Cm)``) — the reference operator split
    (channels stepped before voltage). ``state`` is ``(v, m, h, n)``:
    membrane voltage (mV, absolute) and gating variables. ``I_inj`` is
    depolarizing-positive injected current (inward). Returns ``(v, m, h, n)``.
    Pure ``jax.numpy``; dtype follows the inputs.
    """
    v, m, h, n = state
    p = params
    am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
    m = _exp_euler_gate(m, dt_ms, am, bm)
    h = _exp_euler_gate(h, dt_ms, ah, bh)
    n = _exp_euler_gate(n, dt_ms, an, bn)
    g_na = p.gNa_mS_cm2 * m ** 3 * h
    g_k = p.gK_mS_cm2 * n ** 4
    g_l = p.gL_mS_cm2
    g_tot = g_na + g_k + g_l
    c_tot = g_na * p.ENa_mV + g_k * p.EK_mV + g_l * p.EL_mV + I_inj
    v = (v + c_tot / p.Cm_uF_cm2 * dt_ms) / (1.0 + g_tot / p.Cm_uF_cm2 * dt_ms)
    return (v, m, h, n)


def graded_activation(v_pre, v_th: float = -20.0, slope: float = 5.0):
    """Tanh activation of the graded synapses: ``0.5 * (1 + tanh((Vpre-Vth)/slope))``."""
    return 0.5 * (1.0 + jnp.tanh((v_pre - v_th) / slope))


def graded_syn_step(s, v_pre, tau_d: float, tau_r: float, slope: float = 5.0,
                    v_th: float = -20.0, dt: float = 0.1):
    """One graded-synapse gating step: ``ds/dt = -s/tauD + act*(1-s)/tauR``.

    A non-finite update keeps the old ``s`` (the reference float32 barrier).
    """
    act = graded_activation(v_pre, v_th, slope)
    new_s = s + ((-s / tau_d) + act * (1.0 - s) / tau_r) * dt
    return jnp.where(jnp.isfinite(new_s), new_s, s)


def graded_syn_current(g, s, v_post, e_rev):
    """``I_syn = g * s * (Vpost - E_rev)``, outward-positive (paper Eq. 1)."""
    return g * s * (v_post - e_rev)


def ou_step(n, key, mean: float = 0.0, sigma: float = 0.05, tau: float = 20.0,
            dt: float = 0.1):
    """One Ornstein-Uhlenbeck step (the reference ``Inoise.update_states``).

    ``n += (mean - n)/tau * dt + sigma * sqrt(2/tau) * xi * sqrt(dt)`` with
    ``xi ~ N(0, 1)`` drawn from ``key``; the noise current is ``-n``
    (outward-positive). Shape follows ``n``. A non-finite update keeps the
    old ``n`` (the reference float32 barrier). The reference noise channel
    current is in mA/cm2; callers working in uA/cm2 scale ``sigma`` by 1000.
    """
    xi = jax.random.normal(key, shape=jnp.shape(n), dtype=jnp.asarray(n).dtype)
    new_n = n + (mean - n) / tau * dt + sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt)
    return jnp.where(jnp.isfinite(new_n), new_n, n)


_EDGE_ARGS = ("edge_pre", "edge_post", "edge_weight", "edge_receptor")


def _check_edges(N, R, given):
    """Validate the four edge arrays; return numpy arrays, or None when all are absent."""
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
    if pre.size and (pre.min() < 0 or pre.max() >= N):
        raise ValueError(f"edge_pre must index cells in [0, {N}); got min {pre.min()}, max {pre.max()}")
    if post.size and (post.min() < 0 or post.max() >= N):
        raise ValueError(f"edge_post must index cells in [0, {N}); got min {post.min()}, max {post.max()}")
    if rec.size and (rec.min() < 0 or rec.max() >= R):
        raise ValueError(f"edge_receptor must index receptors in [0, {R}); got min {rec.min()}, max {rec.max()}")
    weight = arrs["edge_weight"].astype(np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError(f"edge_weight must be finite; got {weight}")
    if np.any(weight < 0.0):
        raise ValueError(f"edge_weight must be >= 0 (the reversal sets the sign); got min {weight.min()}")
    return {"pre": pre, "post": post, "weight": weight, "receptor": rec}


def simulate_hh(
    params: HHParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight=None,
    edge_receptor=None,
    receptor_e_rev=(),
    receptor_tau_r=(),
    receptor_tau_d=(),
    receptor_slope=(),
    receptor_vth=(),
    receptor_s0: float = 0.1,
    ou_tau=None,
    ou_sigma=None,
    ou_key=None,
    noise_gain: float = 1.0,
    I_ext=None,
    V0_mV=None,
    record_v: bool = True,
) -> dict[str, Any]:
    """Simulate HH cells with OU noise and graded tanh synapses on an edge list.

    Cells: classic Hodgkin-Huxley (:func:`hh_step`). Synapses: the reference
    graded form (PORT_SPEC.md section 4): each edge ``e`` runs from
    ``edge_pre[e]`` to ``edge_post[e]`` through receptor ``edge_receptor[e]``
    with conductance ``edge_weight[e]`` (mS/cm2, >= 0); the gating variable of
    edge ``e`` evolves as ``ds/dt = -s/tauD + act(Vpre)*(1-s)/tauR`` with
    ``act`` the tanh activation, and contributes
    ``g * s * (Vpost - E_rev)`` (outward-positive) to its post cell. The
    presynaptic voltage is read at the current step: the reference has no
    delay ring, so this kernel takes no ``edge_delay_steps`` argument.

    Noise: per-cell OU channel (:func:`ou_step`) with time constants
    ``ou_tau`` ([N], ms), amplitudes ``ou_sigma`` ([N]) and key ``ou_key``;
    the noise current is ``-ou``. With ``ou_sigma`` None or all zero no draw
    happens and ``ou_key`` may be None. ``noise_gain`` is a nonnegative scalar
    multiplier on ``ou_sigma`` (lane-4e residual X: the noise level; the
    default 1.0 reproduces the unscaled channel bit-exactly).

    ``I_ext`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``:
    depolarizing-positive injected current. ``V0_mV`` is None (rest at -70 mV)
    or shape ``[N]``.

    Per step, in order: synapse gates update from the current voltages
    (non-finite updates keep the old gating); synaptic conductance and offset
    terms accumulate per post cell; the OU channel updates; gates advance by
    exact exponential Euler and voltage by backward Euler at frozen gates
    (the :func:`hh_step` split); upward crossings of -20 mV are spikes.

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV``
    (``[n_steps, N]``, after each step's update, only if ``record_v``), and the
    final ``v``, ``m``, ``h``, ``n``, ``ou`` and ``s``. Implemented with
    ``jax.lax.scan``; no dense ``N x N`` matrix. dtype follows ``params``.

    Raises ``ValueError`` naming the argument for every refused input.
    """
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not dt_ms > 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}")

    N = int(jnp.shape(params.gNa_mS_cm2)[0])
    dtype = jnp.asarray(params.gNa_mS_cm2).dtype
    e_rev = np.asarray(receptor_e_rev, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(receptor_tau_r, dtype=np.float64).reshape(-1)
    tau_d = np.asarray(receptor_tau_d, dtype=np.float64).reshape(-1)
    slope = np.asarray(receptor_slope, dtype=np.float64).reshape(-1)
    vth = np.asarray(receptor_vth, dtype=np.float64).reshape(-1)
    if not (e_rev.shape == tau_r.shape == tau_d.shape == slope.shape == vth.shape):
        raise ValueError(
            "receptor_e_rev, receptor_tau_r, receptor_tau_d, receptor_slope and "
            f"receptor_vth must have the same length; got {[a.size for a in (e_rev, tau_r, tau_d, slope, vth)]}"
        )
    if np.any(tau_r <= 0):
        raise ValueError(f"receptor_tau_r must be > 0 for every receptor; got {tau_r.tolist()}")
    if np.any(tau_d <= 0):
        raise ValueError(f"receptor_tau_d must be > 0 for every receptor; got {tau_d.tolist()}")
    if np.any(slope <= 0):
        raise ValueError(f"receptor_slope must be > 0 for every receptor; got {slope.tolist()}")
    try:
        s0 = float(receptor_s0)
    except (TypeError, ValueError):
        raise ValueError(f"receptor_s0 must be a number; got {receptor_s0!r}")
    if not math.isfinite(s0):
        raise ValueError(f"receptor_s0 must be finite; got {receptor_s0!r}")
    R = int(e_rev.size)

    edges = _check_edges(
        N, R,
        {"edge_pre": edge_pre, "edge_post": edge_post,
         "edge_weight": edge_weight, "edge_receptor": edge_receptor},
    )
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "receptor": np.zeros((0,), np.int64),
        }
    n_edges = int(edges["pre"].shape[0])

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    post_j = jnp.asarray(edges["post"], jnp.int32)
    w_j = jnp.asarray(edges["weight"], dtype)
    rec_j = jnp.asarray(edges["receptor"], jnp.int32)
    erev_j = jnp.asarray(e_rev, dtype)
    taur_j = jnp.asarray(tau_r, dtype)
    taud_j = jnp.asarray(tau_d, dtype)
    slope_j = jnp.asarray(slope, dtype)
    vth_j = jnp.asarray(vth, dtype)

    if ou_sigma is None:
        sig_j = jnp.zeros((N,), dtype)
    else:
        sig_j = jnp.asarray(ou_sigma, dtype)
        if sig_j.shape != (N,):
            raise ValueError(f"ou_sigma must be None or shape ({N},); got {sig_j.shape}")
    try:
        gain = float(noise_gain)
    except (TypeError, ValueError):
        raise ValueError(f"noise_gain must be a number; got {noise_gain!r}")
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"noise_gain must be finite and >= 0; got {noise_gain!r}")
    sig_j = sig_j * jnp.asarray(gain, dtype)
    if ou_tau is None:
        tau_j = jnp.full((N,), 20.0, dtype=dtype)
    else:
        tau_j = jnp.asarray(ou_tau, dtype)
        if tau_j.shape != (N,):
            raise ValueError(f"ou_tau must be None or shape ({N},); got {tau_j.shape}")
    if ou_sigma is None:
        sigma_nonzero = False
    else:
        sigma_nonzero = bool(np.any(np.asarray(ou_sigma, dtype=np.float64) != 0.0))
        if sigma_nonzero and ou_key is None:
            raise ValueError("ou_key is required when ou_sigma is nonzero")
    noise_on = ou_key is not None
    if ou_key is not None:
        try:
            jax.random.split(ou_key)
        except Exception:
            raise ValueError(f"ou_key must be a JAX PRNG key; got {ou_key!r}")

    I_const = jnp.zeros((N,), dtype)
    I_all = None
    if I_ext is not None:
        I_arr = jnp.asarray(I_ext, dtype)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )

    if V0_mV is None:
        V0 = jnp.full((N,), HH_V_REST_MV, dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape != (N,):
            raise ValueError(f"V0_mV must be None or shape ({N},); got {V0.shape}")

    thr = jnp.asarray(HH_SPIKE_THR_MV, dtype)

    def body(carry, I_t):
        v, m, h, n, ou, s, prev_v, rkey = carry
        if noise_on:
            rkey, uk = jax.random.split(rkey)
        I_t = I_const if I_t is None else I_t
        g_syn = jnp.zeros((N,), dtype)
        c_syn = jnp.zeros((N,), dtype)
        if n_edges:
            v_pre = v[pre_j]
            act = 0.5 * (1.0 + jnp.tanh((v_pre - vth_j[rec_j]) / slope_j[rec_j]))
            new_s = s + ((-s / taud_j[rec_j]) + act * (1.0 - s) / taur_j[rec_j]) * dt_ms
            s = jnp.where(jnp.isfinite(new_s), new_s, s)
            gs = w_j * s
            g_syn = jax.ops.segment_sum(gs, post_j, num_segments=N)
            c_syn = jax.ops.segment_sum(gs * erev_j[rec_j], post_j, num_segments=N)
        if noise_on:
            xi = jax.random.normal(uk, shape=(N,), dtype=dtype)
            new_ou = ou + (0.0 - ou) / tau_j * dt_ms + sig_j * jnp.sqrt(2.0 / tau_j) * xi * jnp.sqrt(dt_ms)
            ou = jnp.where(jnp.isfinite(new_ou), new_ou, ou)
        i_noise = -ou
        am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
        m = _exp_euler_gate(m, dt_ms, am, bm)
        h = _exp_euler_gate(h, dt_ms, ah, bh)
        n = _exp_euler_gate(n, dt_ms, an, bn)
        g_na = jnp.asarray(params.gNa_mS_cm2, dtype) * m ** 3 * h
        g_k = jnp.asarray(params.gK_mS_cm2, dtype) * n ** 4
        g_l = jnp.asarray(params.gL_mS_cm2, dtype)
        g_tot = g_na + g_k + g_l + g_syn
        c_tot = (g_na * jnp.asarray(params.ENa_mV, dtype)
                 + g_k * jnp.asarray(params.EK_mV, dtype)
                 + g_l * jnp.asarray(params.EL_mV, dtype)
                 + c_syn - i_noise + I_t)
        cm = jnp.asarray(params.Cm_uF_cm2, dtype)
        v_new = (v + c_tot / cm * dt_ms) / (1.0 + g_tot / cm * dt_ms)
        spk = (prev_v < thr) & (v_new >= thr)
        out = (spk, v_new) if record_v else spk
        return (v_new, m, h, n, ou, s, v_new, rkey), out

    if ou_key is None:
        rkey0 = jnp.zeros((), jnp.int32)
    else:
        rkey0 = ou_key
    m0, h0, n0 = hh_gate_inf(V0)
    carry0 = (
        V0,
        jnp.asarray(m0, dtype),
        jnp.asarray(h0, dtype),
        jnp.asarray(n0, dtype),
        jnp.zeros((N,), dtype),
        jnp.full((n_edges,), s0, dtype),
        V0,
        rkey0,
    )
    (v, m, h, n, ou, s, _, _), ys = jax.lax.scan(body, carry0, I_all, length=n_steps)
    result = {
        "spikes": ys[0] if record_v else ys,
        "v": v,
        "m": m,
        "h": h,
        "n": n,
        "ou": ou,
        "s": s,
    }
    if record_v:
        result["V_mV"] = ys[1]
    return result
