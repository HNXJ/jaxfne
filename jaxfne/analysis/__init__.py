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
"""

from .metrics import (
    burst_index,
    fano_factor,
    fleiss_kappa_binary,
    mean_pairwise_spike_correlation,
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
    "spectrolaminar_psd_jax",
    "bandpower_jax",
    "spectrolaminar_readout_kernel_jax",
    "spectrolaminar_similarity_kernel_jax",
    "spectrolaminar_similarity_candidates_jax",
    "spectrolaminar_similarity_candidates_seeds_jax",
]
