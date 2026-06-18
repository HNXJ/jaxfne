"""Correctness lock for the vectorized ``kappa_synchrony``.

``jaxfne.tutorial_utils.kappa_synchrony`` was reimplemented from an O(N^2)
pairwise Python loop to a single normalized correlation-matrix multiply. These
tests pin the vectorized result to a reference pairwise implementation across
ordinary and degenerate (silent / constant) spike trains, and check the
documented boundary returns.
"""

import numpy as np
import pytest

from jaxfne.tutorial_utils import kappa_synchrony


def _reference_kappa(spikes: np.ndarray) -> float:
    """Original O(N^2) pairwise-loop definition, used only as the oracle."""
    spikes = np.asarray(spikes, dtype=float)
    if spikes.ndim != 2:
        return 0.0
    n_timesteps, n_neurons = spikes.shape
    if n_neurons < 2 or n_timesteps < 1:
        return 0.0
    correlations = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            a, b = spikes[:, i], spikes[:, j]
            if np.sum(a) == 0 or np.sum(b) == 0:
                continue
            sa, sb = np.std(a), np.std(b)
            if sa > 0 and sb > 0:
                correlations.append(
                    np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)
                )
    return float(np.mean(correlations)) if correlations else 0.0


@pytest.mark.parametrize(
    "T,N,p,seed",
    [(200, 50, 0.05, 0), (500, 80, 0.1, 1), (300, 40, 1.0, 2), (150, 60, 0.02, 3)],
)
def test_matches_reference_pairwise(T, N, p, seed):
    rng = np.random.default_rng(seed)
    spikes = (rng.random((T, N)) < p).astype(float)
    assert np.isclose(kappa_synchrony(spikes), _reference_kappa(spikes), atol=1e-9)


def test_matches_reference_with_degenerate_columns():
    rng = np.random.default_rng(7)
    spikes = (rng.random((200, 40)) < 0.1).astype(float)
    spikes[:, 0] = 0.0   # silent neuron
    spikes[:, 1] = 1.0   # always-spiking (zero variance)
    spikes[:, 2] = 0.0   # silent neuron
    assert np.isclose(kappa_synchrony(spikes), _reference_kappa(spikes), atol=1e-9)


def test_boundary_returns():
    assert kappa_synchrony(np.zeros((0, 0))) == 0.0
    assert kappa_synchrony(np.ones((10,))) == 0.0          # 1-D
    assert kappa_synchrony(np.ones((5, 1))) == 0.0         # single neuron
    assert kappa_synchrony(np.zeros((5, 4))) == 0.0        # all silent -> no pairs
