"""Genetic Stochastic Delta Rule (GSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp

from .base import BaseSDRState


@dataclass(frozen=True)
class GSDRState(BaseSDRState):
    """Genetic Stochastic Delta Rule optimizer state.

    Similar to SDRState, with additional tracking for genetic deselection.
    """
    deselection_counter: int = 0


@dataclass(frozen=True)
class GSDRMCDPState(BaseSDRState):
    """GSDR-with-MCDP optimizer state (module-local, not exported).

    Extends the shared SDR-family state with the stagnation count for the
    ``checkpoint_n`` reset and the dynamic mixing weight ``a``.
    """

    stagnant: int = 0
    a: float = 0.0
    a_proposed: float = 0.0


def mcdp_factors(v_traces, pre, post, shift_steps, smooth_k=None):
    """Per-edge MCDP factors: squared Pearson pre/post correlation, z-scored.

    JAX port of the Fig. 6 lane oracle (``run_gsdr.mcdp_factors``, read as
    text): pre voltages against post voltages shifted forward by
    ``shift_steps``, squared Pearson correlation per edge, averaged over the
    batch when ``v_traces`` is (batch, time, cells), then z-scored as
    ``(c - mean) / (std + 1e-6)``. The oracle z-scores each conductance
    group separately (gAMPA, gGABAa), so call once per group as the oracle
    does; one joint call over both groups z-scores a different vector. The
    oracle match holds on non-degenerate edges: edges with no pre or post
    variance read 1.0 here, following the paper Methods (R is 1 where
    undefined), while the oracle has no such branch. With ``smooth_k``
    given, a normalized 1-D Gaussian (sigma ``smooth_k / 2``, taps
    ``-smooth_k..smooth_k``, edge-padded) smooths the per-edge squared
    correlations in edge order before z-scoring (the trainer-side
    restoration noted in the oracle); that kernel shape is this port's
    choice -- the lane records name only ``k = int(sqrt(N))``.

    ``v_traces`` is (time, cells) or (batch, time, cells); ``pre``/``post``
    are int edge-index arrays of equal length.
    """
    shift = int(shift_steps)
    if shift < 1:
        raise ValueError("shift_steps must be >= 1, got %r" % (shift_steps,))
    v = jnp.asarray(v_traces)
    if v.ndim not in (2, 3):
        raise ValueError(
            "v_traces must be (time, cells) or (batch, time, cells), got ndim %d"
            % v.ndim
        )
    n_time = v.shape[-2]
    if shift >= n_time:
        raise ValueError(
            "shift_steps %d leaves no overlap on %d time steps" % (shift, n_time)
        )
    pre_idx = jnp.asarray(pre)
    post_idx = jnp.asarray(post)
    if pre_idx.shape != post_idx.shape:
        raise ValueError(
            "pre and post must have equal length, got %s vs %s"
            % (tuple(pre_idx.shape), tuple(post_idx.shape))
        )
    if smooth_k is not None:
        k = int(smooth_k)
        if k < 1 or k != smooth_k:
            raise ValueError("smooth_k must be a positive int, got %r" % (smooth_k,))

    if v.ndim == 3:
        pv = v[:, :-shift][:, :, pre_idx]
        qv = v[:, shift:][:, :, post_idx]
    else:
        pv = v[:-shift, pre_idx]
        qv = v[shift:, post_idx]
    # No-variance edges read 1.0 below. Peak-to-peak on the raw slices
    # detects them exactly; the demeaned sums of squares carry float
    # reduction rounding (~1e-12 on constants), so `== 0` on them misses.
    undefined_raw = (jnp.ptp(pv, axis=-2) == 0) | (jnp.ptp(qv, axis=-2) == 0)
    pv = pv - pv.mean(axis=-2, keepdims=True)
    qv = qv - qv.mean(axis=-2, keepdims=True)
    pre_ss = (pv ** 2).sum(axis=-2)
    post_ss = (qv ** 2).sum(axis=-2)
    num = (pv * qv).sum(axis=-2)
    den = jnp.sqrt(pre_ss * post_ss) + 1e-6
    corr = (num / den) ** 2
    if corr.ndim == 2:
        undefined = undefined_raw.any(axis=0)
        corr = corr.mean(axis=0)
    else:
        undefined = undefined_raw

    if smooth_k is not None:
        taps = jnp.arange(-k, k + 1, dtype=corr.dtype)
        weights = jnp.exp(-(taps**2) / (2.0 * (k / 2.0) ** 2))
        weights = weights / weights.sum()
        corr = jnp.convolve(jnp.pad(corr, k, mode="edge"), weights, mode="valid")

    z = (corr - corr.mean()) / (corr.std() + 1e-6)
    return jnp.where(undefined, jnp.asarray(1.0, dtype=z.dtype), z)


def step_gsdr_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Plain gradient-descent step: ``u_t - eta * grad_l``.

    Despite the ``GSDRState``/``GSDR`` naming and ``deselection_counter``
    field, this does NOT implement genetic deselection or any behavior
    beyond ``state``/``hyperparams`` plumbing -- ``state``
    (including ``deselection_counter``) is passed through unchanged, never
    read or updated. A separate, more elaborate Genetic SDR implementation
    exists at :func:`jaxfne.optim.core.gsdr_transform` -- but it is NOT what
    ``Model.tune(optimizer="GSDR", ...)`` calls either (confirmed
    2026-07-18: that string dispatches through ``_resolve_optimizer`` to a
    blackbox candidate-proposal path, ``propose_blackbox_candidates``, which
    never references ``gsdr_transform``; passing the ``gsdr_transform()``
    object itself as ``optimizer=`` also does not work, since
    ``_resolve_optimizer`` has no branch for a raw ``GradientTransformation``
    and silently falls through to the same generic blackbox path instead of
    calling its ``init``/``update``). ``gsdr_transform`` is exercised only
    by ``tests/test_optim_tune.py`` directly, not by any confirmed
    production call path. Kept for backward
    compatibility with any caller using the lower-level
    ``step(u_t, grad_l, state, hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state


def gsdr_mcdp_transform(
    inner_optimizer: Optional[Any] = None,
    stochastic_scale: float = 1.0,
    deselection_threshold: float = 3.0,
    a_init: float = 0.0,
    a_dynamic: bool = True,
    a_step: float = 0.1,
    lambda_d: float = 1.0,
    checkpoint_n: int = 5,
    w_max: Optional[float] = None,
) -> Any:
    """Return an Optax-compatible GradientTransformation for GSDR with MCDP.

    Fig. 6 lane port (settings observed in the authors' run: inner SDR
    ``learning_rate=0.1, momentum=0.4``, ``deselection_threshold=3.0`` (the
    paper's C), ``a_init=0.0`` dynamic, ``lambda_d=1.0``,
    ``checkpoint_n=5``, ``mcdp=True``; PORT_SPEC section 10). Each update
    combines the inner gradient step with an MCDP-gated stochastic delta
    ``noise * params * mcdp`` mixed by the weight ``a``::

        update = lambda_d * (a * delta + (1 - a) * inner)

    ``lambda_d`` is a constant multiplier here: the lane records name an
    ``effective_lambda`` schedule but give no functional form, so none is
    implemented. ``stochastic_scale`` scales the stochastic delta term only
    (the lane's exploration-magnitude knob: 1.0 keeps the oracle-matched
    delta, 0.0 reduces the update to the scaled inner step); it exists so a
    caller can shrink or grow exploration without touching the inner step.
    ``a`` starts at ``a_init`` and, when ``a_dynamic``, takes a uniform
    ``±a_step`` random walk per update clipped to ``[0, 1]`` (the walk draws
    from the update key, so updates stay deterministic in key). The walked
    ``a`` is kept only when the step improved: state carries the accepted
    ``a`` and the proposed ``a``; at each update with a loss, the proposal
    is accepted when the loss improves on the previous best loss and
    otherwise the accepted ``a`` is kept; then a new proposal is drawn.
    ``w_max`` clamps the update so ``params + update`` lies in ``[0, w_max]``
    (``None`` means no clamp). Selection tracks the best loss and
    parameters: a C-threshold revert
    (``loss > best_loss * deselection_threshold``), a stagnation reset
    (``checkpoint_n`` updates without improvement) and a non-finite-loss
    revert each return ``best_params - params`` deltas, mirroring
    :func:`jaxfne.optim.core.gsdr_transform`. The C-threshold comparison is
    ``>`` (the authors' code); the paper text says ``>=`` -- the sources
    differ and the code follows the authors' code. ``mcdp=None`` voids the
    epoch
    (the NaN-batch case in the lane): zero updates, state unchanged.
    ``update`` requires an explicit ``key``. Like
    :func:`jaxfne.optim.core.gsdr_transform`, ``update`` runs host-side
    (Python control flow on the loss) and is not ``jax.jit``-compatible.
    """
    from .core import require_optax

    optax = require_optax()

    if deselection_threshold < 1:
        raise ValueError(
            "deselection_threshold must be >= 1, got %r" % (deselection_threshold,)
        )
    n_stagnant = int(checkpoint_n)
    if n_stagnant < 1 or n_stagnant != checkpoint_n:
        raise ValueError("checkpoint_n must be a positive int, got %r" % (checkpoint_n,))
    if not 0.0 <= a_init <= 1.0:
        raise ValueError("a_init must lie in [0, 1], got %r" % (a_init,))
    if lambda_d <= 0:
        raise ValueError("lambda_d must be positive, got %r" % (lambda_d,))
    if stochastic_scale < 0:
        raise ValueError("stochastic_scale must be >= 0, got %r" % (stochastic_scale,))

    if not 0.0 <= a_step <= 1.0:
        raise ValueError("a_step must lie in [0, 1], got %r" % (a_step,))
    if w_max is not None and float(w_max) <= 0:
        raise ValueError("w_max must be positive, got %r" % (w_max,))

    if inner_optimizer is None:
        inner_optimizer = optax.sgd(learning_rate=0.1, momentum=0.4)

    def init(params: Any) -> tuple[GSDRMCDPState, Any]:
        inner_state = inner_optimizer.init(params)
        return GSDRMCDPState(step=0, a=float(a_init), a_proposed=float(a_init)), inner_state

    def _broadcast_mcdp(template: Any, mcdp: Any) -> Any:
        factors = jnp.asarray(mcdp)

        def _one(leaf: Any) -> Any:
            if not hasattr(leaf, "shape"):
                return leaf
            try:
                full = jnp.broadcast_to(
                    factors.astype(leaf.dtype) if hasattr(leaf, "dtype") else factors,
                    leaf.shape,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "mcdp factors shape %s do not broadcast to leaf shape %s"
                    % (tuple(factors.shape), tuple(leaf.shape))
                ) from exc
            return full

        return jax.tree_util.tree_map(_one, template)

    def update(
        grads: Any,
        state: tuple[GSDRMCDPState, Any],
        params: Optional[Any] = None,
        *,
        loss: Optional[float] = None,
        key: Optional[Any] = None,
        mcdp: Optional[Any] = None,
    ) -> tuple[Any, tuple[GSDRMCDPState, Any]]:
        if key is None:
            raise ValueError("GSDR-MCDP transform requires explicit PRNG key")

        gsdr_state, inner_state = state
        template = params if params is not None else grads
        if mcdp is None:
            return jax.tree_util.tree_map(jnp.zeros_like, template), state

        inner_updates, inner_state = inner_optimizer.update(grads, inner_state, params)

        key_delta, key_walk = jax.random.split(key)
        noise = jax.tree_util.tree_map(
            lambda u: jax.random.normal(key_delta, u.shape, dtype=u.dtype)
            if hasattr(u, "shape")
            else u,
            inner_updates,
        )
        if params is not None:
            scaled = jax.tree_util.tree_map(
                lambda n, p: n * p if hasattr(n, "dtype") else n, noise, params
            )
        else:
            scaled = noise
        factors = _broadcast_mcdp(inner_updates, mcdp)
        delta = jax.tree_util.tree_map(
            lambda s, f: stochastic_scale * s * f if hasattr(s, "dtype") else s,
            scaled,
            factors,
        )

        finite = loss is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(float(loss))))
        )
        improves = finite and (loss < gsdr_state.best_loss)
        accepted = gsdr_state.a
        if a_dynamic and loss is not None:
            # Keep the walked `a` only when the step improved: accept the
            # previously proposed `a` on improvement, else revert to the
            # accepted one; then propose a new walk step.
            accepted = gsdr_state.a_proposed if improves else gsdr_state.a
            step_walk = jax.random.uniform(
                key_walk, shape=(), minval=-a_step, maxval=a_step
            )
            proposed = float(jnp.clip(accepted + step_walk, 0.0, 1.0))
        else:
            proposed = accepted

        combined = jax.tree_util.tree_map(
            lambda d, u: lambda_d * (accepted * d + (1.0 - accepted) * u)
            if hasattr(u, "dtype")
            else u,
            delta,
            inner_updates,
        )

        new_best_loss = gsdr_state.best_loss
        new_best_param = gsdr_state.best_param
        stagnant = gsdr_state.stagnant
        reset_counter = gsdr_state.reset_counter
        if loss is not None and params is not None:
            new_best_loss = loss if improves else gsdr_state.best_loss
            new_best_param = params if improves else gsdr_state.best_param
            stagnant = 0 if improves else gsdr_state.stagnant + 1
            deselect = (not finite) or (
                finite
                and float(loss) > float(gsdr_state.best_loss) * float(deselection_threshold)
            )
            should_reset = deselect or stagnant >= n_stagnant
            if should_reset:
                combined = jax.tree_util.tree_map(
                    lambda p_use, p: p_use - p, new_best_param, params
                )
                stagnant = 0
            reset_counter = gsdr_state.reset_counter + (1 if should_reset else 0)
        if w_max is not None and params is not None:
            cap = float(w_max)
            combined = jax.tree_util.tree_map(
                lambda u, p: jnp.clip(p + u, 0.0, cap) - p
                if hasattr(u, "dtype")
                else u,
                combined,
                params,
            )
        new_state = GSDRMCDPState(
            step=gsdr_state.step + 1,
            best_loss=float(new_best_loss),
            best_param=new_best_param,
            stagnant=int(stagnant),
            reset_counter=reset_counter,
            a=accepted,
            a_proposed=proposed,
            var_sup_ema=gsdr_state.var_sup_ema,
            var_unsup_ema=gsdr_state.var_unsup_ema,
            ema_decay=gsdr_state.ema_decay,
        )

        return combined, (new_state, inner_state)

    @dataclass(frozen=True)
    class GradientTransformation:
        init: Callable[[Any], Any]
        update: Callable[..., Any]

    return GradientTransformation(init=init, update=update)
