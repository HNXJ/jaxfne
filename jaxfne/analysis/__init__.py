"""Analysis metrics for population-level spike and field diagnostics.

This module provides population-level metrics for characterizing neural population
activity from simulations. All functions accept JAX arrays or NumPy arrays and return
JSON-safe Python floats.

Functions
---------
mean_pairwise_spike_correlation
    Compute mean pairwise Pearson correlation of spike trains.

burst_index
    Compute fraction of time bins where population is active (bursting).

fano_factor
    Compute fraction of spike counts (Poisson-ness metric).

fleiss_kappa_binary
    Compute Fleiss-kappa-like agreement metric for population synchrony.

wang_buzsaki_kappa
    Compute mean pairwise coincidence of binary binned spike trains.

activity_regime
    Classify one population activity trace into a standard regime.

activity_regimes
    Classify named neuron subsets of one trace.
"""

from .metrics import (
    ACTIVITY_REGIME_DEFAULTS,
    activity_regime,
    activity_regimes,
    burst_index,
    fano_factor,
    fleiss_kappa_binary,
    mean_pairwise_spike_correlation,
    wang_buzsaki_kappa,
)
from .granger import (
    directionality_index,
    granger_spectral,
)
from .spectral import (
    spectrolaminar_psd_jax,
    bandpower_jax,
    spectrolaminar_readout_kernel_jax,
    spectrolaminar_similarity_kernel_jax,
    spectrolaminar_similarity_candidates_jax,
    spectrolaminar_similarity_candidates_seeds_jax,
)

__all__ = [
    "mean_pairwise_spike_correlation",
    "burst_index",
    "fano_factor",
    "fleiss_kappa_binary",
    "wang_buzsaki_kappa",
    "activity_regime",
    "activity_regimes",
    "ACTIVITY_REGIME_DEFAULTS",
    "granger_spectral",
    "directionality_index",
    "spectrolaminar_psd_jax",
    "bandpower_jax",
    "spectrolaminar_readout_kernel_jax",
    "spectrolaminar_similarity_kernel_jax",
    "spectrolaminar_similarity_candidates_jax",
    "spectrolaminar_similarity_candidates_seeds_jax",
]
