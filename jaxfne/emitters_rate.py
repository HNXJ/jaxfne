"""Wilson-Cowan-type rate-population emitter (function-level only).

One nonlinear firing-rate unit per neuron (Mejias et al. 2016, Sci Adv
2(11):e1601335, Methods/Computational model: one excitatory plus one
inhibitory rate population per laminar module). Per unit, with step
``dt_ms`` and time constant ``tau_ms``::

    tau dr/dt = -r + phi(I_rec + I_ext) + noise,

``I_rec`` is the signed, delay-stepped presynaptic rate sum (excitatory
edges add, inhibitory edges subtract, as on the current-based routes);
``I_ext`` is the emitter's constant drive channel plus the declared input
(``Simulation(poisson_drive=...)``, same seed rule as the current-based
route). The noise is white with intensity sigma on the tau-scaled equation
(tau dr = (-r + phi) dt + sigma sqrt(tau) dW), so the Euler-Maruyama
increment per step is::

    noise = sigma * sqrt(dt_ms / tau_ms) * N(0, 1),

drawn independently per neuron and step from a seeded generator, so one
seed gives one trajectory. Rates floor at 0.0 (declared: the transfer
functions are non-negative, but the noise term can push a quiet unit
below zero).

Transfer functions (all normalized to ``phi(0) = 1``, the paper's
resting-rate scale):

- ``threshold_linear``: ``max(0, 1 + gain * x)``;
- ``sigmoid``: ``2 / (1 + exp(-gain * x))``;
- ``x_over_one_minus_exp``: ``gain * x / (1 - exp(-gain * x))`` with
  ``phi(0) = 1`` by continuity (the Wilson/Brunel ``x/(1-e^-x)`` form).

Function-level only. Configuration, construct and simulate wiring live in
:mod:`jaxfne._construct_rate` (population/edges/delays from the
current-based bridge, bit-exact) and :mod:`jaxfne._model_simulate`
(``_simulate_rate``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Transfer functions the rate route consumes (``NeuronType.emitter_params["transfer"]``).
RATE_TRANSFER_FUNCTIONS: tuple[str, ...] = (
    "threshold_linear",
    "sigmoid",
    "x_over_one_minus_exp",
)

#: Per-``NeuronType`` parameter keys the rate emitter consumes (checked by
#: :mod:`jaxfne.emitter_registry`; anything else is refused as unconsumed).
_RATE_PARAM_FIELDS: tuple[str, ...] = (
    "tau_ms",
    "transfer",
    "gain",
    "sigma",
    "r0",
)

#: Module defaults: one resting-rate unit per neuron, silent without drive.
RATE_DEFAULTS: dict[str, Any] = {
    "tau_ms": 10.0,
    "transfer": "threshold_linear",
    "gain": 1.0,
    "sigma": 0.0,
    "r0": 0.1,
}


def _checked_rate_scalar(name: str, value: Any) -> float:
    """Finite non-negative real for a rate numeric field, refusing the rest by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"rate emitter param {name}={value!r} is not realized: it takes "
            "a finite real number (bools refused, never coerced)"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"rate emitter param {name}={value!r} is not realized: it takes "
            "a finite real number"
        )
    if float(value) < 0.0:
        raise ValueError(
            f"rate emitter param {name}={value!r} is not realized: it takes "
            "a value >= 0"
        )
    return float(value)


def check_rate_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a per-triple rate params mapping against :data:`_RATE_PARAM_FIELDS`.

    Unknown keys are refused (mirroring :mod:`jaxfne.emitter_registry`);
    ``tau_ms`` must additionally be > 0, ``transfer`` one of
    :data:`RATE_TRANSFER_FUNCTIONS`, and every numeric field finite and
    >= 0. ``None`` reads as empty (module defaults apply).
    """
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError(
            f"rate emitter params {params!r} are not realized: they take a "
            f"mapping of {list(_RATE_PARAM_FIELDS)}"
        )
    unknown = [k for k in params if k not in _RATE_PARAM_FIELDS]
    if unknown:
        raise ValueError(
            f"rate emitter does not consume params {unknown!r}; "
            f"consumed: {list(_RATE_PARAM_FIELDS)}"
        )
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key == "transfer":
            if value not in RATE_TRANSFER_FUNCTIONS:
                raise ValueError(
                    f"rate emitter param transfer={value!r} is not realized: "
                    f"expected one of {list(RATE_TRANSFER_FUNCTIONS)}"
                )
            out[key] = str(value)
        elif key == "tau_ms":
            out[key] = _checked_rate_scalar(key, value)
            if out[key] <= 0.0:
                raise ValueError(
                    f"rate emitter param tau_ms={value!r} is not realized: "
                    "it takes a value > 0"
                )
        else:
            out[key] = _checked_rate_scalar(key, value)
    return out


def rate_transfer(
    x: np.ndarray, transfer: str, gain: np.ndarray | float,
) -> np.ndarray:
    """The rate transfer function ``phi`` with ``phi(0) = 1``.

    ``x`` is a float array; ``gain`` is a finite non-negative scalar or an
    array broadcastable onto ``x`` (checked here so direct kernel callers
    fail closed too). ``transfer`` names one of
    :data:`RATE_TRANSFER_FUNCTIONS`.
    """
    arr = np.asarray(x, dtype=np.float64)
    if transfer not in RATE_TRANSFER_FUNCTIONS:
        raise ValueError(
            f"rate transfer {transfer!r} is not realized: expected one of "
            f"{list(RATE_TRANSFER_FUNCTIONS)}"
        )
    g = np.asarray(gain, dtype=np.float64)
    if not bool(np.isfinite(g).all()) or bool((g < 0.0).any()):
        raise ValueError(
            f"rate gain {gain!r} is not realized: it takes finite values >= 0"
        )
    gx = g * arr
    if transfer == "threshold_linear":
        return np.maximum(0.0, 1.0 + gx)
    if transfer == "sigmoid":
        return 2.0 / (1.0 + np.exp(-gx))
    # x_over_one_minus_exp: g*x / (1 - exp(-g*x)), phi(0) = 1 by continuity.
    # -expm1(-gx) evaluates the denominator without the catastrophic
    # cancellation of 1 - exp(-gx) near zero (1e-12 reads 2.2e-5 high there).
    out = np.ones_like(gx)
    nz = gx != 0.0
    denom = -np.expm1(-gx[nz])
    # gx != 0 implies denom != 0 (exp is injective); guard anyway.
    tiny = denom == 0.0
    out[nz] = np.where(tiny, 1.0, gx[nz] / np.where(tiny, 1.0, denom))
    return out


@dataclass(frozen=True)
class RateNetworkParams:
    """Per-neuron rate-population parameters for a constructed population.

    ``tau_ms``/``gain``/``sigma``/``r0`` are float arrays of shape ``[N]``;
    ``transfer`` names one transfer function per neuron. ``drive`` is the
    constant external current per neuron (inherited from the current-based
    bridge channel, matching drive semantics); ``sign``/``W``/``labels``/
    ``layer_labels`` mirror the fields the shared construction machinery
    reads. ``W`` is a ``(0, 0)`` placeholder: rate edges live only in
    ``params["edge_list"]``, as on the lif route.
    """

    tau_ms: np.ndarray
    transfer: tuple[str, ...]
    gain: np.ndarray
    sigma: np.ndarray
    r0: np.ndarray
    drive: np.ndarray
    sign: np.ndarray
    W: np.ndarray
    labels: tuple[str, ...]
    layer_labels: tuple[str, ...] | None = None

    @property
    def n_neurons(self) -> int:
        """Population size (length of the drive channel)."""
        return int(np.asarray(self.drive).shape[0])


def make_rate_params(labels: Sequence[str]) -> RateNetworkParams:
    """Build :class:`RateNetworkParams` for one unit per neuron.

    Every neuron starts from :data:`RATE_DEFAULTS`; per-triple overrides
    apply afterwards via :func:`jaxfne._construct_rate.apply_rate_triple_param_overrides`
    (matching drive semantics, so two triples sharing a type name may
    declare different values without conflict). ``drive``/``sign`` start
    empty here and are filled by :mod:`jaxfne._construct_rate` from the
    bridge channel.
    """
    labels = tuple(str(lab) for lab in labels)
    n = len(labels)
    if not n:
        raise ValueError("make_rate_params needs at least one neuron label")
    return RateNetworkParams(
        tau_ms=np.full((n,), float(RATE_DEFAULTS["tau_ms"]), dtype=np.float64),
        transfer=tuple(str(RATE_DEFAULTS["transfer"]) for _ in range(n)),
        gain=np.full((n,), float(RATE_DEFAULTS["gain"]), dtype=np.float64),
        sigma=np.full((n,), float(RATE_DEFAULTS["sigma"]), dtype=np.float64),
        r0=np.full((n,), float(RATE_DEFAULTS["r0"]), dtype=np.float64),
        drive=np.zeros((n,), dtype=np.float64),
        sign=np.ones((n,), dtype=np.float64),
        W=np.zeros((0, 0), dtype=np.float64),
        labels=labels,
    )


def simulate_rate_edge(
    params: RateNetworkParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre: Any,
    edge_post: Any,
    edge_weight: Any,
    edge_delay_steps: Any,
    drive: Any,
    poisson: Any = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate Wilson-Cowan-type rate units on a sparse signed edge list.

    Per step ``t``, with ``r[t]`` the population rate vector:

    1. arrivals: each edge reads the presynaptic rate of step ``t - d_e``
       from a bool-free ring of length ``max(d_e)`` (delays are >= 1 step;
       :mod:`jaxfne._construct_rate` floors zero-step edges with a receipt);
    2. input ``I = segment_sum(w_e * arrival_e) + drive + poisson[t]``
       per neuron (signed weights: inhibitory edges subtract);
    3. Euler-Maruyama update ``r += (dt/tau) * (-r + phi(I))`` plus
       ``sigma * sqrt(dt/tau) * N(0, 1)`` per neuron, floored at 0.0
       (stationary variance of a linear unit sigma**2 / (2 - dt/tau), so
       sigma**2 / 2 as dt -> 0).

    ``poisson`` (None, or a ``(n_steps, N)`` array) is the declared input
    added to ``drive``. ``seed`` drives the noise generator, so one seed
    gives one trajectory. Returns ``{"rates": (n_steps, N) float64}``.
    """
    if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError(
            f"simulate_rate_edge needs n_steps >= 1; got {n_steps!r}"
        )
    n_steps = int(n_steps)
    dt = float(dt_ms)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"simulate_rate_edge needs a finite dt_ms > 0; got {dt_ms!r}"
        )
    n = int(params.n_neurons)
    tau = np.asarray(params.tau_ms, dtype=np.float64).ravel()
    gain = np.asarray(params.gain, dtype=np.float64).ravel()
    sigma = np.asarray(params.sigma, dtype=np.float64).ravel()
    r = np.asarray(params.r0, dtype=np.float64).ravel().copy()
    if not (tau.shape == gain.shape == sigma.shape == r.shape == (n,)):
        raise ValueError(
            "simulate_rate_edge rate columns disagree with the population: "
            f"{tau.shape}, {gain.shape}, {sigma.shape}, {r.shape} for N={n}"
        )
    if bool((tau <= 0.0).any()):
        raise ValueError("simulate_rate_edge needs tau_ms > 0 on every unit")
    drv = np.asarray(drive, dtype=np.float64).ravel()
    if drv.shape != (n,):
        raise ValueError(
            f"simulate_rate_edge drive has shape {drv.shape} for N={n}"
        )
    pre = np.asarray(edge_pre, dtype=np.int64).ravel()
    post = np.asarray(edge_post, dtype=np.int64).ravel()
    weight = np.asarray(edge_weight, dtype=np.float64).ravel()
    delay = np.asarray(edge_delay_steps, dtype=np.int64).ravel()
    if not (pre.shape == post.shape == weight.shape == delay.shape):
        raise ValueError(
            "simulate_rate_edge edge arrays disagree: "
            f"{pre.shape} vs {post.shape} vs {weight.shape} vs {delay.shape}"
        )
    if pre.size and (bool((pre < 0).any()) or bool((pre >= n).any())
                     or bool((post < 0).any()) or bool((post >= n).any())):
        raise ValueError("simulate_rate_edge edge endpoints leave [0, N)")
    if bool((delay < 1).any()):
        raise ValueError(
            "simulate_rate_edge needs edge_delay_steps >= 1 on every edge; "
            "zero-step edges floor to 1 in jaxfne._construct_rate"
        )
    if poisson is None:
        ext = np.zeros((n_steps, n), dtype=np.float64)
    else:
        ext = np.asarray(poisson, dtype=np.float64)
        if ext.shape != (n_steps, n):
            raise ValueError(
                f"simulate_rate_edge poisson has shape {ext.shape}; "
                f"expected ({n_steps}, {n})"
            )
        if not bool(np.isfinite(ext).all()):
            raise ValueError("simulate_rate_edge poisson must stay finite")
    if not bool(np.isfinite(drv).all()) or not bool(np.isfinite(r).all()):
        raise ValueError("simulate_rate_edge needs finite drive and r0")
    transfers = tuple(params.transfer)
    if len(transfers) != n or any(t not in RATE_TRANSFER_FUNCTIONS for t in transfers):
        raise ValueError(
            "simulate_rate_edge needs one transfer per neuron from "
            f"{list(RATE_TRANSFER_FUNCTIONS)}"
        )
    rng = np.random.default_rng(int(seed))
    max_d = int(delay.max()) if delay.size else 1
    ring = np.zeros((max_d, n), dtype=np.float64)
    ring[:] = r[None, :]
    rates = np.zeros((n_steps, n), dtype=np.float64)
    alpha = dt / tau
    noise_scale = sigma * np.sqrt(dt / tau)
    # One transfer group per distinct function: vectorized phi per group.
    groups: dict[str, np.ndarray] = {}
    for i, t in enumerate(transfers):
        groups.setdefault(t, []).append(i)
    group_idx = {t: np.asarray(ix, dtype=np.int64) for t, ix in groups.items()}
    for t in range(n_steps):
        rec = np.zeros((n,), dtype=np.float64)
        if pre.size:
            # Delayed presynaptic rates: step t - d reads the ring slot that
            # held it (the ring always holds the last max_d rate vectors).
            for d in np.unique(delay):
                m = delay == d
                if t - int(d) >= 0:
                    past = ring[(t - int(d)) % max_d]
                else:
                    # Before the first arrival the presynaptic rate reads as
                    # the initial rate r0 (resting state, not zero).
                    past = np.asarray(params.r0, dtype=np.float64).ravel()
                np.add.at(rec, post[m], weight[m] * past[pre[m]])
        phi = np.empty((n,), dtype=np.float64)
        cur = rec + drv + ext[t]
        for tname, ix in group_idx.items():
            phi[ix] = rate_transfer(cur[ix], tname, gain[ix])
        r = r + alpha * (-r + phi) + noise_scale * rng.standard_normal((n,))
        r = np.maximum(r, 0.0)
        ring[t % max_d] = r
        rates[t] = r
        if not bool(np.isfinite(r).all()):
            raise ValueError(
                f"simulate_rate_edge went non-finite at step {t}: drive, gain "
                "and sigma stay finite on the tensor route, so this names a "
                "caller-side overflow"
            )
    return {"rates": rates}
