"""PCL causal STDP as a registered jaxfne HDP rule (``pcl_stdp``), Izhikevich kernel.

Per-edge aux coordinates (aux_layout="per_edge", shape (n_edges, 3)):
  ltp   sum of exp(-(t - t_i)/tau) over presynaptic events since the last postsynaptic spike
  ltd   sum of exp(-(t_i - t_{s-1})/tau) over the same events (deferred LTD)
  ypost exp(-(t - t_{s-1})/tau), the postsynaptic spike trace copied onto each edge
At a postsynaptic spike (paper Eqs. 4-11):
  dw = (w_max - w) f eta ltp - w f eta ltd,  w >= 0,
  then the post neuron's incoming weights of the same group are L1-normalized to lam
  (only where ``norm`` = 1), and ltp/ltd reset. Updates are returned as rates
  (x_next - x)/dt so the kernel's x + dt*dx lands on x_next.
Per-edge parameters are arrays of length n_edges; ``n_groups`` is the number of
(post neuron, connection type) groups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jaxfne.emitters import _segment_sum  # noqa: E402
from jaxfne.hdp_rule import (HDPRuleDescriptor, HDPRuleUpdate, list_registered_hdp_rules,  # noqa: E402
                             register_hdp_rule)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

NAME = "pcl_stdp"
PARAM_KEYS = ("tau", "w_max", "eta", "f", "lam", "norm", "group", "n_groups")


def _step(ctx):
    p = ctx.rule_params
    dt = ctx.dt
    A, B, Y = ctx.aux[:, 0], ctx.aux[:, 1], ctx.aux[:, 2]
    decay = jnp.exp(-dt / p["tau"])
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    A1 = A * decay + pre_sp
    B1 = B + pre_sp * Y * decay
    w = jnp.abs(ctx.w)
    dW = post_sp * ((p["w_max"] - w) * p["f"] * p["eta"] * A1 - w * p["f"] * p["eta"] * B1)
    w1 = jnp.maximum(w + dW, 0.0)
    sums = _segment_sum(w1, p["group"], int(p["n_groups"]))[p["group"]]
    w_norm = jnp.where(p["norm"] > 0, w1 * p["lam"] / jnp.maximum(sums, 1e-12), w1)
    w2 = jnp.where(post_sp > 0, w_norm, w)
    keep = 1.0 - post_sp
    aux_next = jnp.stack([A1 * keep, B1 * keep, jnp.where(post_sp > 0, 1.0, Y * decay)], axis=-1)
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=(aux_next - ctx.aux) / dt,
                         d_theta={"edge_weight": (w2 - w) / dt})


def register():
    if NAME in list_registered_hdp_rules():
        return
    register_hdp_rule(
        HDPRuleDescriptor(
            name=NAME, h_coords=("H",), theta_targets=("edge_weight",),
            aux_coords=("ltp", "ltd", "ypost"), aux_layout="per_edge", scope="node",
            w_bounds=(0.0, 1e9), default_params={k: 0.0 for k in PARAM_KEYS},
        ),
        _step,
    )


def rule_params(tau, w_max, eta, f, lam, norm, group, n_groups):
    """Per-edge arrays (float32) plus integer groups for ``hdp_rule_params``."""
    def as_f(x):
        return jnp.asarray(x, jnp.float32)

    return dict(tau=as_f(tau), w_max=as_f(w_max), eta=as_f(eta), f=as_f(f), lam=as_f(lam),
                norm=as_f(norm), group=jnp.asarray(group, jnp.int32), n_groups=int(n_groups))
