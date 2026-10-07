"""PCL étude: the Fig. 2 inhibitory-STDP gate orders inhibition by predictability."""
from __future__ import annotations

import numpy as np

import artifacts.etudes.pcl.pcl_fig2 as F


def test_inputs_copy_the_declared_fraction():
    t0, *rest = F.make_sample(np.random.default_rng(0))
    for p, tk in zip(F.P_REPLACE, rest):
        copied = np.isin(np.round(tk - F.LAG_MS, 9), np.round(t0, 9)).mean()
        assert abs(copied - p) < 0.15


def test_inhibition_ordered_by_predictability():
    r = F.run_network(0, w0=1.0, n_train=20, n_test=10, epochs=2)
    w, s = r["w_hist"][-1], r["suppression"]
    assert w[0] > w[1] > w[2] and s[0] > s[1] > s[2]


def test_no_learning_keeps_weights():
    net = F.Net(3.0)
    net.run(F.make_sample(np.random.default_rng(1)), learn=False)
    assert np.array_equal(net.w, np.full(3, 3.0))
