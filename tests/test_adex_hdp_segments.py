"""Segmented runs of the registered AdEx HDP kernel (`state` / `return_state`).

A 1000-step run must equal 400 + 600 steps chained through the returned
full final state bit-for-bit (spikes, final V, final W, aux), with the
registered `istdp_vogels2011` rule engaged on a plastic mask and a Poisson
external drive keyed by `key` (the key chain continues, so the drive
matches too). A state from a different network size is refused.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

import jaxfne as jtfne
assert "site-packages" not in jtfne.__file__, (
    "expected the repo jaxfne on sys.path, not site-packages"
)

from jaxfne.emitters_adex import make_adex_params, simulate_adex_hdp_registered
from jaxfne.istdp_vogels2011 import ISTDP_RULE_NAME, istdp_plasticity_mask

DT_MS = 0.1
N_STEPS = 1000
SPLIT = 400

# Cells 0-3 E, 4-5 PV: PV->E plastic, E->E and PV->PV frozen.
_EDGE_PRE = [4, 5, 0, 1, 4]
_EDGE_POST = [0, 1, 2, 3, 5]
_IS_PRE_PV = [True, True, False, False, True]
_IS_POST_E = [True, True, False, False, False]


def _kw(n_steps, seed=0):
    return dict(
        n_steps=int(n_steps),
        dt_ms=DT_MS,
        edge_pre=list(_EDGE_PRE),
        edge_post=list(_EDGE_POST),
        edge_weight_nS=[3.0] * 5,
        edge_receptor=[1, 1, 0, 0, 1],
        edge_delay_steps=[1, 1, 1, 1, 1],
        receptor_reversal_mV=(0.0, -80.0),
        receptor_tau_ms=(5.0, 10.0),
        decay="exp",
        I_ext_pA=np.full(6, 300.0),
        ext_targets=[0],
        ext_lambda=1.0,
        ext_weight_nS=1.0,
        ext_receptor=0,
        ext_delay_steps=1,
        key=jax.random.PRNGKey(int(seed)),
        hdp_rule=ISTDP_RULE_NAME,
        plasticity_mask=istdp_plasticity_mask(_IS_PRE_PV, _IS_POST_E),
        seed=int(seed),
        record_w_trace=False,
    )


def _params():
    return make_adex_params(("E", "E", "E", "E", "PV", "PV"))


def test_chained_segments_match_one_run_bit_for_bit():
    """400 + 600 chained steps equal the 1000-step run exactly."""
    params = _params()
    full = simulate_adex_hdp_registered(params, **_kw(N_STEPS))
    assert "state" not in full, "the default call returns no state"
    first = simulate_adex_hdp_registered(
        params, **dict(_kw(SPLIT), return_state=True))
    np.testing.assert_array_equal(
        np.asarray(first["spikes"]), np.asarray(full["spikes"])[:SPLIT],
        err_msg="the first segment must match the run prefix")
    second = simulate_adex_hdp_registered(
        params, **dict(_kw(N_STEPS - SPLIT), state=first["state"]))
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(first["spikes"]),
                       np.asarray(second["spikes"])]),
        np.asarray(full["spikes"]),
        err_msg="chained segments must reproduce the run bit-for-bit")
    for name in ("V", "w_final_nS", "aux_final", "H_final"):
        np.testing.assert_array_equal(
            np.asarray(second[name]), np.asarray(full[name]), err_msg=name)


def test_chained_run_spikes_and_moves_plastic_weights():
    """The fixture spikes (so the handoff covers spikes) and I-to-E moves."""
    params = _params()
    full = simulate_adex_hdp_registered(params, **_kw(N_STEPS))
    assert int(np.asarray(full["spikes"]).sum()) > 0, "fixture must spike"
    w_final = np.asarray(full["w_final_nS"], dtype=float)
    assert bool((w_final[:2] != 3.0).any()), \
        "plastic I-to-E weights must move under the registered rule"
    assert bool((w_final[2:] == 3.0).all()), \
        "frozen E and I-to-I edges must stay bit-identical"


def test_state_from_another_network_size_is_refused():
    """A state resumes only the network that wrote it (N and E checked)."""
    first = simulate_adex_hdp_registered(
        _params(), **dict(_kw(SPLIT), return_state=True))
    other = make_adex_params(("E", "E", "PV", "PV"))
    other_kw = dict(
        n_steps=10,
        dt_ms=DT_MS,
        edge_pre=[2],
        edge_post=[0],
        edge_weight_nS=[3.0],
        edge_receptor=[0],
        edge_delay_steps=[1],
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(2.0,),
        decay="exp",
        hdp_rule=ISTDP_RULE_NAME,
        seed=0,
    )
    with pytest.raises(ValueError, match="n_neurons"):
        simulate_adex_hdp_registered(other, **dict(other_kw, state=first["state"]))
