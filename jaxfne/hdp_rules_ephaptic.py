"""Ephaptic lateral voltage coupling as a registered HDP rule (``ephaptic_field``).

The registry name is historical. The rule uses the HDP registry as a carrier only: it is
neither RBD nor HDP dynamics and is not derived from the source-field chain.

Each neuron receives a coupling term built from its neighbours' membrane potentials:

    Phi_i = alpha * sum_j M_ij (v_j - v_rest)

``M`` is a sparse spatial kernel (``src``, ``dst``, ``weight`` arrays, one entry per
ordered pair j -> i), so memory and work scale with the number of neighbour pairs,
never with N^2. ``Phi`` enters the membrane through the kernel's ``drive_bias``
coordinate, which the Izhikevich and LIF routes add to the step input current;
``alpha`` therefore carries that route's input unit per mV: on the Izhikevich route
the input enters dv/dt, so ``alpha`` is in 1/ms; on the PCL LIF route the input is a
voltage increment added once per step, so ``alpha`` is per step and the coupling
it sets scales with ``dt`` (rate equivalent ``alpha/dt`` in 1/ms). ``v_rest`` has no default: it is the
route's resting potential (0 mV for the PCL LIF, near -65 mV for Izhikevich).
``alpha = 0`` gives ``Phi = 0`` and leaves ``drive_bias`` at 0. The run then equals the
rule-free run on the same backend: a noise-free LIF kernel run, and through ``simulate()``
the rule-free run with ``recurrent_backend="edge_list"`` (tested; an engaged rule always
runs the edge-list kernel, ISSUE_LOG P-057 and P-062).

Timing: the coupling term set during step t is applied at step t + 1. The voltages it reads
differ by route: the LIF kernel presents the voltages after step t (one-step lag);
the Izhikevich registrable kernel presents the voltages entering step t, i.e. after
step t - 1 (two-step lag).

The rule's own contribution to ``drive_bias`` is tracked in the per-neuron aux
coordinate ``phi`` and changed by jumps ``Phi_new - phi``, so superposed rules that
also write ``drive_bias`` add to it. ``drive_bias`` is unbounded for this rule (a clip
would change ``Phi``).

An exogenous (imposed) field is a known time series, not a function of ``v``; supply
it through the route's drive schedule. It does not use this rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .hdp_rule import HDPRuleContext, HDPRuleDescriptor, HDPRuleUpdate, register_hdp_rule

#: Registry name of the rule.
EPHAPTIC_RULE_NAME = "ephaptic_field"

_KERNEL_KEYS = ("src", "dst", "weight")


def ephaptic_kernel(positions, *, sigma, cutoff, normalize="row"):
    """Sparse Gaussian kernel ``M_ij = exp(-d_ij^2 / (2 sigma^2))`` for ``0 < d_ij <= cutoff``.

    ``positions`` is ``(N, D)``; ``sigma`` and ``cutoff`` are in the unit of
    ``positions``. Self pairs and coincident positions (``d_ij^2 = 0`` in float64) are excluded.
    ``normalize="row"`` scales each row to sum 1; a row without neighbours, or whose
    weights all underflow to 0, stays 0. ``"none"`` keeps the raw Gaussian.
    Returns ``{"src", "dst", "weight"}`` as numpy arrays; entry k adds
    ``weight[k] * (v[src[k]] - v_rest)`` to neuron ``dst[k]``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError(f"positions must be (N, D), got shape {pos.shape}")
    if not (sigma > 0 and cutoff > 0):
        raise ValueError(f"sigma and cutoff must be > 0, got sigma={sigma}, cutoff={cutoff}")
    if normalize not in ("row", "none"):
        raise ValueError(f"normalize must be 'row' or 'none', got {normalize!r}")
    pairs = cKDTree(pos).query_pairs(r=float(cutoff), output_type="ndarray")
    src = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)
    d2 = np.sum((pos[src] - pos[dst]) ** 2, axis=1)
    keep = d2 > 0.0  # coincident positions (d = 0) give no pair
    src, dst, d2 = src[keep], dst[keep], d2[keep]
    weight = np.exp(-d2 / (2.0 * float(sigma) ** 2))
    if normalize == "row":
        row = np.bincount(dst, weights=weight, minlength=pos.shape[0])[dst]
        # A row whose weights all underflow to 0 stays 0 instead of 0/0.
        weight = np.divide(weight, row, out=np.zeros_like(weight), where=row > 0.0)
    return {"src": src, "dst": dst, "weight": weight}


def ephaptic_rule_params(kernel, *, alpha, v_rest):
    """``hdp_rule_params`` for ``ephaptic_field`` from a kernel and the two scalars."""
    missing = [k for k in _KERNEL_KEYS if k not in kernel]
    if missing:
        raise ValueError(f"kernel lacks {missing}; build it with ephaptic_kernel")
    return {
        "alpha": float(alpha),
        "v_rest": float(v_rest),
        "src": jnp.asarray(kernel["src"], jnp.int32),
        "dst": jnp.asarray(kernel["dst"], jnp.int32),
        "weight": jnp.asarray(kernel["weight"]),
    }


def ephaptic_field(v, *, src, dst, weight, alpha, v_rest):
    """``Phi = alpha * M (v - v_rest)`` by segment sum over the kernel pairs."""
    contrib = jnp.asarray(weight, v.dtype) * (v[src] - jnp.asarray(v_rest, v.dtype))
    return jnp.asarray(alpha, v.dtype) * jax.ops.segment_sum(contrib, dst, num_segments=v.shape[0])


def _step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    p = ctx.rule_params
    if any(p[k] is None for k in (*_KERNEL_KEYS, "v_rest")):
        raise ValueError("ephaptic_field needs src, dst, weight and v_rest; use ephaptic_rule_params")
    phi = ephaptic_field(ctx.v, src=p["src"], dst=p["dst"], weight=p["weight"],
                         alpha=p["alpha"], v_rest=p["v_rest"])
    jump = phi - ctx.aux
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), j_aux=jump, j_theta={"drive_bias": jump})


register_hdp_rule(
    HDPRuleDescriptor(
        name=EPHAPTIC_RULE_NAME,
        h_coords=("H",),
        theta_targets=("drive_bias",),
        aux_coords=("phi",),
        aux_layout="per_neuron",
        scope="node",
        b_bounds=(-float("inf"), float("inf")),
        default_params={"alpha": 0.0, "v_rest": None, "src": None, "dst": None, "weight": None},
    ),
    _step,
)
