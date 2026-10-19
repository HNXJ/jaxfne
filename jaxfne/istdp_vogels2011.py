"""Inhibitory STDP rule, Vogels et al. 2011 (lane P2 core home).

Recurrent-network rule (authors' ``isp_brian.py``, read as text only, never
run): traces ``dA/dt = -A/tau_stdp`` with ``tau_stdp = 20 ms``; on a
presynaptic spike ``A_pre += 1`` and ``w += (A_post - alpha)*eta``; on a
postsynaptic spike ``A_post += 1`` and ``w += A_pre*eta``; cap ``gmax = 100``;
``eta = 1e-2``; ``alpha = 2*rho0*tau_stdp`` with the coded target rate
``rho0 = 3`` Hz. I-to-E edges only: excitatory weights and I-to-I weights
are fixed in the paper.

Registered as ``istdp_vogels2011`` for the conductance (AdEx) family kernel
:func:`jaxfne.emitters_adex.simulate_adex_hdp_registered`, which integrates
the returned drives as ``aux + dt*d_aux`` and ``|w| + dt*dw`` clipped to the
descriptor ``w_bounds``. Event jumps of ``+1`` on the traces and the
pairing updates on the weights are therefore returned in rate form
(jump divided by ``dt``); trace decay is Euler (``dA = -A/tau``), matching
the kernel's integration. ``alpha`` is always derived from the target rate,
``alpha = 2*rho0_hz*(tau_stdp_ms/1000)``: there is no ``alpha`` parameter to
mistype, and an unknown key (including ``alpha``) is refused by
:func:`jaxfne.hdp_rule.check_hdp_rule_params`.

The rule context carries no cell-type labels, so which edges learn is
declared at the call site with :func:`istdp_plasticity_mask` (I-to-E True,
everything else False); the kernel's ``plasticity_mask`` then freezes the
rest exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .hdp_rule import HDPRuleContext, HDPRuleDescriptor, HDPRuleUpdate, register_hdp_rule

#: Registry name of the rule.
ISTDP_RULE_NAME = "istdp_vogels2011"

#: Paper defaults (recurrent network parameter block, ``isp_brian.py``).
ETA_DEFAULT = 1e-2
TAU_STDP_MS_DEFAULT = 20.0
RHO0_HZ_DEFAULT = 3.0

#: Weight clip: floored at 0, capped at gmax = 100 (paper values).
W_MIN_NS = 0.0
W_MAX_NS = 100.0


def istdp_alpha(*, rho0_hz: float = RHO0_HZ_DEFAULT,
                tau_stdp_ms: float = TAU_STDP_MS_DEFAULT) -> float:
    """Depression constant ``alpha = 2*rho0*tau_stdp`` (tau in seconds).

    At the paper values (3 Hz, 20 ms) this is 0.12: the homogeneous-rate
    fixed point where mean drift zeroes.
    """
    return 2.0 * float(rho0_hz) * (float(tau_stdp_ms) / 1000.0)


def resolve_istdp_params(params: Mapping[str, Any]) -> dict[str, float]:
    """Fail-closed Python resolution of the rule parameters.

    Accepts the merged ``{**default_params, **rule_params}`` mapping (or a
    partial one; missing keys take paper defaults) and returns concrete
    floats with ``alpha`` derived. Refuses non-finite values and non-positive
    ``eta``/``tau_stdp_ms`` by name; the jitted step selects the same
    ``alpha`` through :func:`istdp_alpha`.
    """
    merged = {"eta": ETA_DEFAULT, "tau_stdp_ms": TAU_STDP_MS_DEFAULT,
              "rho0_hz": RHO0_HZ_DEFAULT}
    merged.update(dict(params))
    eta = float(merged["eta"])
    tau = float(merged["tau_stdp_ms"])
    rho0 = float(merged["rho0_hz"])
    if not np.isfinite(eta) or not eta > 0.0:
        raise ValueError(f"istdp_vogels2011 eta must be a finite float > 0; got {merged['eta']!r}")
    if not np.isfinite(tau) or not tau > 0.0:
        raise ValueError(
            f"istdp_vogels2011 tau_stdp_ms must be a finite float > 0; got {merged['tau_stdp_ms']!r}"
        )
    if not np.isfinite(rho0) or not rho0 >= 0.0:
        raise ValueError(
            f"istdp_vogels2011 rho0_hz must be a finite float >= 0; got {merged['rho0_hz']!r}"
        )
    return {"eta": eta, "tau_stdp_ms": tau, "rho0_hz": rho0,
            "alpha": istdp_alpha(rho0_hz=rho0, tau_stdp_ms=tau)}


def istdp_plasticity_mask(pre_is_inhibitory: Any, post_is_excitatory: Any) -> jax.Array:
    """Per-edge plasticity declaration: True exactly on I-to-E edges.

    The rule context carries no cell-type labels, so the caller declares
    them here from the neuron table (paper: I-to-E plastic; E-to-all and
    I-to-I fixed). Inputs are per-edge array-likes aligned with the kernel's
    edge order. An empty selection (no True) is refused: plasticity over no
    edge would run silently static.
    """
    pre = np.asarray(pre_is_inhibitory, dtype=bool)
    post = np.asarray(post_is_excitatory, dtype=bool)
    if pre.ndim != 1 or post.ndim != 1 or pre.shape != post.shape:
        raise ValueError(
            "istdp_plasticity_mask needs two 1-D per-edge boolean arrays of equal length; "
            f"got shapes {pre.shape} and {post.shape}"
        )
    mask = pre & post
    if not bool(mask.any()):
        raise ValueError(
            f"istdp_plasticity_mask selected no edge ({int(mask.size)} edges, 0 plastic): "
            "declare the I-to-E edges or drop the plasticity run"
        )
    return jnp.asarray(mask)


def _istdp_vogels2011_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    """One registered-rule step of Vogels 2011 inhibitory STDP."""
    cfg = resolve_istdp_params(ctx.rule_params)
    dtype = ctx.H.dtype
    eta = jnp.asarray(cfg["eta"], dtype=dtype)
    tau = jnp.asarray(cfg["tau_stdp_ms"], dtype=dtype)
    alpha = jnp.asarray(cfg["alpha"], dtype=dtype)
    dt = ctx.dt
    # Delayed presynaptic arrivals drive the pre trace at time t (the
    # qualified chain event_{t-d} -> H_t holds); the zero-delay path passes
    # pre_sp=None and falls back to spikes[pre].
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    a_pre = ctx.aux[:, 0]
    a_post = ctx.aux[:, 1]
    pre_events = (
        jax.ops.segment_sum(pre_sp, ctx.pre, num_segments=ctx.n_neurons) / dt
    )
    post_events = ctx.spikes / dt
    d_aux = jnp.stack(
        (a_pre / -tau + pre_events, a_post / -tau + post_events), axis=1
    )
    dw = (
        eta
        * (
            pre_sp * (a_post[ctx.post] - alpha)
            + post_sp * a_pre[ctx.pre]
        )
        / dt
    )
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H), d_aux=d_aux, d_theta={"edge_weight": dw}
    )


register_hdp_rule(
    HDPRuleDescriptor(
        name=ISTDP_RULE_NAME,
        h_coords=("H",),
        theta_targets=("edge_weight",),
        aux_coords=("A_pre", "A_post"),
        aux_layout="per_neuron",
        scope="node",
        w_bounds=(W_MIN_NS, W_MAX_NS),
        default_params={
            "eta": ETA_DEFAULT,
            "tau_stdp_ms": TAU_STDP_MS_DEFAULT,
            "rho0_hz": RHO0_HZ_DEFAULT,
        },
    ),
    _istdp_vogels2011_step,
)
