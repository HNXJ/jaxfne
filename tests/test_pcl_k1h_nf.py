"""Gate K1h-nf: the one-ulp nudge of the column driver and the gate verdict logic."""
from __future__ import annotations

import copy
import sys

import numpy as np

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.pcl.k1h_nf_gate as G  # noqa: E402
import artifacts.etudes.pcl.pcl_column_hdp as D  # noqa: E402


def test_nudge_moves_nonzero_weights_one_ulp_and_zero_is_identity():
    w = np.array([0.0, 1.0, 55.6, 419.53656], np.float64)
    w32 = w.astype(np.float32)
    assert np.array_equal(D.nudge_ulp(w, 0), w32)
    w1 = D.nudge_ulp(w, 1)
    assert w1[0] == 0.0
    assert np.array_equal(w1[1:] - w32[1:], np.spacing(w32[1:]))


def _runs(h_shift, nudge_shift, h_osi=0.05):
    acc = {g: {c: 0.5 for c in ("simple", "complex", "both")} for g in ("pcl", "no_inh")}
    acc["random"] = {"simple": 0.3}
    base = {"acc": acc, "osi_trained_median": 0.05, "osi_untrained_median": 0.06,
            "spikes": {"pcl": [100, 10], "no_inh": [300, 10]}}
    out = {}
    for s in G.SEEDS:
        nudge, h = copy.deepcopy(base), copy.deepcopy(base)
        nudge["acc"]["pcl"]["simple"] += nudge_shift
        h["acc"]["no_inh"]["both"] += h_shift
        h["osi_trained_median"] = h_osi
        out[s] = {"base": base, "nudge": nudge, "h": h}
    return out


def test_gate_verdicts():
    assert G.gate(_runs(0.02, 0.03))[0] == "PASS"
    assert G.gate(_runs(0.04, 0.03))[0] == "FAIL"
    assert G.gate(_runs(0.0, 0.0))[0] == "ERROR"
    assert G.gate(_runs(0.0, 0.03, h_osi=0.5))[0] == "FAIL"  # A1 verdict flips under H
