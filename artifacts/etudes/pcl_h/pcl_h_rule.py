"""PCL causal STDP as a registered jaxfne HDP rule (``pcl_stdp_h``), Izhikevich kernel.

Same update as ``pcl_stdp`` (paper Eqs. 4-11), but the per-neuron traces live
in the H state instead of per-edge aux:

H layout (h_shape=(2,), H[n] = (x[n], Y[n])):
  x[n]   presynaptic trace: x <- x*decay + spikes[n] (never reset).
  Y[n]   postsynaptic spike trace: Y <- Y*decay, set to 1 where n spikes.
aux layout (per_edge, two coordinates):
  S[e]   snapshot of x[pre[e]] at post[e]'s last spike: S <- S*decay, then set
         to x1[pre[e]] (after this step's update) where post[e] spikes.
  B[e]   deferred LTD sum: B <- B + pre_sp[e]*Y[post[e]]*decay (old Y).
The LTP term of edge e is A[e] = x1[pre[e]] - S_decayed[e]: the presynaptic
events since post[e]'s last spike, including this step's. At a postsynaptic
spike: dw = (w_max - w) f eta A - w f eta B, w >= 0, then L1-normalization of
the group to lam where ``norm`` = 1; S/B reset. Updates are returned as rates
so the kernel's x + dt*dx lands on x_next.

Uniform-tau refusal: x decays per neuron but ``tau`` arrives per edge, so a
non-uniform ``tau`` (max > min) raises ValueError instead of averaging.
Tie case (pre and post spike in the same step): A uses the decayed snapshot
taken *before* this step's update while x1 already includes this step's pre
spike, exactly as ``pcl_stdp`` A1 = A*decay + pre_sp includes it; S is then
set to x1, so A reads 0 afterwards, matching the reset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jaxfne.emitters import _segment_sum  # noqa: E402
from jaxfne.hdp_rule import (HDPRuleDescriptor, HDPRuleUpdate, list_registered_hdp_rules,  # noqa: E402
                             register_hdp_rule)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

NAME = "pcl_stdp_h"
PARAM_KEYS = ("tau", "w_max", "eta", "f", "lam", "norm", "group", "n_groups")


def _step(ctx):
    """PCL update with x/Y traces in H and snapshot/deferred-LTD per edge."""
    p = ctx.rule_params
    dt = ctx.dt
    tau = np.asarray(p["tau"])
    if tau.max() > tau.min():
        raise ValueError(
            "pcl_stdp_h requires uniform tau (one scalar for all edges); "
            f"got min {tau.min()} max {tau.max()}"
        )
    dec = jnp.exp(-dt / jnp.asarray(float(tau.min()), dtype=dt.dtype))
    x, Y = ctx.H[:, 0], ctx.H[:, 1]
    S, B = ctx.aux.reshape(-1, 2)[:, 0], ctx.aux.reshape(-1, 2)[:, 1]
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    x1 = x * dec + ctx.spikes
    Y1 = jnp.where(ctx.spikes > 0, 1.0, Y * dec)
    S_dec = S * dec
    A = x1[ctx.pre] - S_dec
    B1 = B + pre_sp * Y[ctx.post] * dec
    w = jnp.abs(ctx.w)
    dW = post_sp * ((p["w_max"] - w) * p["f"] * p["eta"] * A - w * p["f"] * p["eta"] * B1)
    w1 = jnp.maximum(w + dW, 0.0)
    sums = _segment_sum(w1, p["group"], int(p["n_groups"]))[p["group"]]
    w_norm = jnp.where(p["norm"] > 0, w1 * p["lam"] / jnp.maximum(sums, 1e-12), w1)
    w2 = jnp.where(post_sp > 0, w_norm, w)
    S_next = jnp.where(post_sp > 0, x1[ctx.pre], S_dec)
    B_next = B1 * (1.0 - post_sp)
    H_next = jnp.stack([x1, Y1], axis=-1)
    aux_next = jnp.stack([S_next, B_next], axis=-1)
    return HDPRuleUpdate(dH=(H_next - ctx.H) / dt, d_aux=(aux_next - ctx.aux) / dt,
                         d_theta={"edge_weight": (w2 - w) / dt})


def register():
    if NAME in list_registered_hdp_rules():
        return
    register_hdp_rule(
        HDPRuleDescriptor(
            name=NAME, h_coords=("pre_trace", "post_trace"), h_shape=(2,),
            theta_targets=("edge_weight",), aux_coords=("pre_snapshot", "ltd_sum"),
            aux_layout="per_edge", scope="node",
            h_bounds=(0.0, 1e9), w_bounds=(0.0, 1e9),
            default_params={k: 0.0 for k in PARAM_KEYS},
        ),
        _step,
    )


def rule_params(tau, w_max, eta, f, lam, norm, group, n_groups):
    """Per-edge arrays (float32) plus integer groups for ``hdp_rule_params``."""
    def as_f(x):
        return jnp.asarray(x, jnp.float32)

    return dict(tau=as_f(tau), w_max=as_f(w_max), eta=as_f(eta), f=as_f(f), lam=as_f(lam),
                norm=as_f(norm), group=jnp.asarray(group, jnp.int32), n_groups=int(n_groups))
