"""P-057 closure: through the public ``simulate()``, a registered HDP rule at zero gain
reproduces the rule-free run on the same backend.

An engaged HDP rule always runs the edge-list kernel, so the rule-free reference uses
``recurrent_backend="edge_list"``; against the default dense backend the runs differ for
backend reasons, not rule reasons (ISSUE_LOG P-057, P-062).
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.hdp_rules_ephaptic import EPHAPTIC_RULE_NAME, ephaptic_kernel, ephaptic_rule_params

N = 100
DURATION_MS = 100.0
DT_MS = 0.5
SEED = 0


def _model():
    layer = nt.Layer(
        name="L4",
        n_neurons=N,
        neuron_types=[nt.NeuronType.make("E", fraction=0.9), nt.NeuronType.make("PV", fraction=0.1)],
    )
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])], name="p057_identity")
    return jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=SEED, duration_ms=DURATION_MS, dt_ms=DT_MS))


def _zero_rule_params(rule):
    if rule == "synthetic_presyn_gain":
        return {"hdp_rule": rule}  # every gain defaults to 0
    pos = np.random.default_rng(1).uniform(0.0, 100.0, size=(N, 2))
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return {"hdp_rule": rule, "hdp_rule_params": ephaptic_rule_params(k, alpha=0.0, v_rest=-65.0)}


@pytest.mark.parametrize("rule", ["synthetic_presyn_gain", EPHAPTIC_RULE_NAME])
def test_zero_gain_rule_equals_rule_free_on_edge_list(rule):
    model = _model()
    free = jtfne.simulate(
        model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
    )
    zero = jtfne.simulate(
        model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=_zero_rule_params(rule), recurrent_backend="edge_list"),
    )
    s_free, s_zero = np.asarray(free.spikes), np.asarray(zero.spikes)
    assert (s_free > 0.5).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(s_zero, s_free)
    np.testing.assert_array_equal(np.asarray(zero.V_m), np.asarray(free.V_m))
