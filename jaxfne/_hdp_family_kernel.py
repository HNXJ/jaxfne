"""Family-generic layer over registered HDP rules (lane 7, gap G3).

Any model is ``(h, y) = f(x, h')`` with ``x : A : y`` on a NeuronalTensor ``A``
(Hamm 2026-10-13): ``h`` carries RBS/H, plasticity traces and weights under
HDP for every neuron model. This module factors the HDP machinery: rule
resolution, aux-state layout, weight update, H state and recording budgets;
so it runs with any emitter family's step function.

Family step interface: a family supplies a per-neuron step

    ``(state, I, params, dt) -> (state, spikes_bool, v)``

(``state`` is the family's own carry, ``I`` injected current, ``spikes_bool``
a boolean per-neuron vector, ``v`` membrane voltage for the rule context).
The HDP layer supplies presynaptic/postsynaptic spikes, ``aux``, ``H`` and
``W`` through :class:`HDPRuleContext` and integrates the returned
:class:`HDPRuleUpdate` with :func:`apply_registered_rule_update`.

Families and their plastic targets:

- ``"izhikevich"``: current-clamp edge list with signed weights; supports
  ``edge_weight`` (sign-preserving magnitude) and ``drive_bias``. The
  existing :func:`simulate_edge_recurrent_izhikevich_hdp_registered` is the
  reference implementation and is intentionally left as is.
- ``"adex"``: conductance synapses with plastic ``|w|`` (weight magnitude
  in nS; the current sign lives in the receptor reversal, not the weight);
  supports ``edge_weight`` only. A rule whose descriptor needs anything else
  (e.g. ``drive_bias``) is refused by name via
  :func:`check_rule_family_support`.
- ``"lif"``: PCL current-based LIF on a signed edge list; supports
  ``edge_weight`` (sign-preserving magnitude, as on ``"izhikevich"``) and
  ``drive_bias`` (the rule's per-neuron ``bias`` adds to the step input in
  :func:`jaxfne.emitters_lif.simulate_lif_edge`).

``u`` (Izhikevich recovery) has no AdEx counterpart: AdEx contexts carry a
zero ``u`` coordinate, documented at the call site. No descriptor channel can
declare a need for ``u``, so there is nothing to refuse there; the refusable
surface is the declared ``theta_targets``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Tuple

import jax
import jax.numpy as jnp

from .hdp_rule import (
    HDPRuleContext,
    HDPRuleDescriptor,
    apply_registered_rule_update as apply_registered_rule_update,
    check_hdp_rule_params,
    expected_aux_shape,
    get_hdp_rule,
)

#: Family identifiers understood by :func:`check_rule_family_support`.
FAMILY_IZHIKEVICH = "izhikevich"
FAMILY_ADEX = "adex"
FAMILY_LIF = "lif"

#: Plastic targets each family integrates. AdEx keeps magnitude semantics on
#: conductance weights; intrinsic-excitability bias is an Izhikevich-only
#: coordinate, so a ``drive_bias`` rule on AdEx is refused by name.
FAMILY_THETA_SUPPORT: dict[str, tuple[str, ...]] = {
    FAMILY_IZHIKEVICH: ("edge_weight", "drive_bias"),
    FAMILY_ADEX: ("edge_weight",),
    FAMILY_LIF: ("edge_weight", "drive_bias"),
}

#: ``FamilyStepFn``: ``(state, I, params, dt) -> (state, spikes_bool, v)``.
FamilyStepFn = Callable[..., Tuple[Any, jax.Array, jax.Array]]


def resolve_registered_rule(
    hdp_rule: str, hdp_rule_params: Mapping[str, Any] | None
) -> tuple[HDPRuleDescriptor, Callable[[HDPRuleContext], Any], dict[str, Any]]:
    """Resolve a registered rule name and fail closed on bad params.

    Unknown rule names raise via :func:`get_hdp_rule`; unconsumed param keys
    raise via :func:`check_hdp_rule_params`. Returns
    ``(descriptor, rule_step, merged_params)``.
    """
    descriptor, rule_step = get_hdp_rule(hdp_rule)
    rule_params = check_hdp_rule_params(hdp_rule, hdp_rule_params)
    return descriptor, rule_step, rule_params


def check_rule_family_support(
    descriptor: HDPRuleDescriptor, *, family: str, rule_name: str
) -> None:
    """Refuse a rule whose descriptor needs something ``family`` lacks.

    Raises ``ValueError`` naming the rule and the unsupported targets;
    returns ``None`` when the family integrates every declared target.
    """
    try:
        supported = FAMILY_THETA_SUPPORT[family]
    except KeyError:
        raise ValueError(
            f"unknown HDP family {family!r}; expected one of "
            f"{sorted(FAMILY_THETA_SUPPORT)}"
        ) from None
    missing = [t for t in descriptor.theta_targets if t not in supported]
    if missing:
        raise ValueError(
            f"hdp_rule {rule_name!r} declares theta_targets {missing!r} needing "
            f"state the {family!r} family lacks; {family!r} supports "
            f"{list(supported)}"
        )


def hdp_init_carry(
    descriptor: HDPRuleDescriptor,
    *,
    n_neurons: int,
    n_edges: int,
    jdtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Initial ``(H, aux, bias)`` carry for a descriptor (RBS reference: H=1)."""
    h_shape = (int(n_neurons),) + tuple(int(d) for d in descriptor.h_shape)
    aux_shape = expected_aux_shape(
        descriptor, n_neurons=int(n_neurons), n_edges=int(n_edges)
    )
    return (
        jnp.ones(h_shape, dtype=jdtype),
        jnp.zeros(aux_shape, dtype=jdtype),
        jnp.zeros((int(n_neurons),), dtype=jdtype),
    )


# ``apply_registered_rule_update`` is defined on jaxfne.hdp_rule (D32 public
# name) and imported above; this keeps the private path a bound alias.


def assert_freeze_allclose(actual: Any, expected: Any, *, rtol: float = 1e-12) -> None:
    """Baseline-freeze comparator: allclose on float64, exact on spikes."""
    import numpy as np

    a = np.asarray(actual)
    e = np.asarray(expected)
    assert a.shape == e.shape, f"freeze shape drift: {a.shape} vs {e.shape}"
    if a.dtype == bool or e.dtype == bool:
        np.testing.assert_array_equal(a, e)
    else:
        np.testing.assert_allclose(a, e, rtol=rtol, atol=0.0)
