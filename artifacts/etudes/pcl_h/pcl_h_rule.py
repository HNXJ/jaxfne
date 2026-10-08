"""PCL causal STDP as a registered jaxfne HDP rule (``pcl_stdp_h``), Izhikevich kernel.

Same update as ``pcl_stdp`` (paper Eqs. 4-11), but the per-neuron traces live
in the H state instead of per-edge aux:

H layout (h_shape=(K+1,), H[n] = (x_1[n], ..., x_K[n], Y[n])):
  x_k[n] presynaptic trace for tau class k: x_k <- x_k*exp(-dt/tau_k) + spikes[n]
         (never reset). A neuron projecting through edges of K distinct tau
         values needs K traces; ``pcl_stdp_h`` has K = 1, ``pcl_stdp_h2`` K = 2.
  Y[n]   postsynaptic spike trace with the tau of n's plastic inputs: Y <- Y*decay,
         set to 1 where n spikes.
aux layout (per_edge, two coordinates):
  S[e]   snapshot of x[pre[e]] at post[e]'s last spike: S <- S*decay, then set
         to x1[pre[e]] (after this step's update) where post[e] spikes.
  B[e]   deferred LTD sum: B <- B + pre_sp[e]*Y[post[e]]*decay (old Y).
The LTP term of edge e is A[e] = x1[pre[e]] - S_decayed[e]: the presynaptic
events since post[e]'s last spike, including this step's. At a postsynaptic
spike: dw = (w_max - w) f eta A - w f eta B, w >= 0, then L1-normalization of
the group to lam where ``norm`` = 1; S/B reset. Updates are returned as rates
so the kernel's x + dt*dx lands on x_next.

Refusals: Y decays per post neuron, so tau must be uniform over the plastic
inputs of each post neuron; the number of distinct tau values must equal K.
Either violation raises ValueError instead of averaging. Pass H_final zeros:
the kernel's default H is ones.
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


def _tau_classes(tau, post, n, n_classes):
    """Distinct tau values, each edge's class and each neuron's post tau (numpy, trace time)."""
    taus = np.unique(tau)
    if len(taus) != n_classes:
        raise ValueError(f"rule holds {n_classes} tau classes in H; edges carry {len(taus)}: {taus}")
    hi = np.full(n, -np.inf)
    lo = np.full(n, np.inf)
    np.maximum.at(hi, post, tau)
    np.minimum.at(lo, post, tau)
    has = np.isfinite(hi)
    if np.any(hi[has] > lo[has]):
        raise ValueError("pcl_stdp_h requires uniform tau over the plastic inputs of each post neuron")
    return taus, np.searchsorted(taus, tau), np.where(has, hi, taus[0])


def _step(ctx):
    """PCL update with x/Y traces in H and snapshot/deferred-LTD per edge."""
    p = ctx.rule_params
    dt = ctx.dt
    n_classes = int(ctx.H.shape[1]) - 1
    taus, k, tau_post = _tau_classes(np.asarray(p["tau"]), np.asarray(ctx.post), int(ctx.n_neurons), n_classes)
    dec_k = jnp.exp(-dt / jnp.asarray(taus, dtype=dt.dtype))  # (K,)
    dec_e = dec_k[jnp.asarray(k)]  # per edge
    dec_n = jnp.exp(-dt / jnp.asarray(tau_post, dtype=dt.dtype))  # per neuron, post trace
    x, Y = ctx.H[:, :n_classes], ctx.H[:, n_classes]
    S, B = ctx.aux.reshape(-1, 2)[:, 0], ctx.aux.reshape(-1, 2)[:, 1]
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    x1 = x * dec_k[None, :] + ctx.spikes[:, None]
    Y1 = jnp.where(ctx.spikes > 0, 1.0, Y * dec_n)
    x1_e = x1[ctx.pre, jnp.asarray(k)]
    S_dec = S * dec_e
    A = x1_e - S_dec
    B1 = B + pre_sp * Y[ctx.post] * dec_e
    w = jnp.abs(ctx.w)
    dW = post_sp * ((p["w_max"] - w) * p["f"] * p["eta"] * A - w * p["f"] * p["eta"] * B1)
    w1 = jnp.maximum(w + dW, 0.0)
    sums = _segment_sum(w1, p["group"], int(p["n_groups"]))[p["group"]]
    w_norm = jnp.where(p["norm"] > 0, w1 * p["lam"] / jnp.maximum(sums, 1e-12), w1)
    w2 = jnp.where(post_sp > 0, w_norm, w)
    S_next = jnp.where(post_sp > 0, x1_e, S_dec)
    B_next = B1 * (1.0 - post_sp)
    H_next = jnp.concatenate([x1, Y1[:, None]], axis=-1)
    aux_next = jnp.stack([S_next, B_next], axis=-1)
    return HDPRuleUpdate(dH=(H_next - ctx.H) / dt, d_aux=(aux_next - ctx.aux) / dt,
                         d_theta={"edge_weight": (w2 - w) / dt})


NAME2 = "pcl_stdp_h2"  # two tau classes (the column: 7 ms onto simple, 40 ms onto complex cells)


def register():
    for name, h_coords in ((NAME, ("pre_trace", "post_trace")),
                           (NAME2, ("pre_trace_tau1", "pre_trace_tau2", "post_trace"))):
        if name in list_registered_hdp_rules():
            continue
        register_hdp_rule(
            HDPRuleDescriptor(
                name=name, h_coords=h_coords, h_shape=(len(h_coords),),
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
