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

cv_isi
    Mean ISI coefficient of variation over spiking cells (Destexhe Eq. 3).

pairwise_cc_binned_disjoint
    Mean Pearson correlation over disjoint pairs in binned counts (Eq. 4).

mean_rate_hz
    Population mean firing rate (Hz) over the whole raster.

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
    cv_isi,
    fano_factor,
    fleiss_kappa_binary,
    mean_pairwise_spike_correlation,
    mean_rate_hz,
    pairwise_cc_binned_disjoint,
    population_rates,
    wang_buzsaki_kappa,
    xcorr_peak_lag,
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
    welch_numpy,
    welch_spectrum,
    aperiodic_log10,
    fit_aperiodic,
    rate_welch_psd,
    aperiodic_spectral_peak,
    oscillation_call,
    ref42_lfp,
    ref42_phase_delay,
    pairwise_phase_consistency,
)

__all__ = [
    "mean_pairwise_spike_correlation",
    "burst_index",
    "fano_factor",
    "fleiss_kappa_binary",
    "wang_buzsaki_kappa",
    "cv_isi",
    "pairwise_cc_binned_disjoint",
    "mean_rate_hz",
    "population_rates",
    "xcorr_peak_lag",
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
    "welch_numpy",
    "welch_spectrum",
    "aperiodic_log10",
    "fit_aperiodic",
    "rate_welch_psd",
    "aperiodic_spectral_peak",
    "oscillation_call",
    "ref42_lfp",
    "ref42_phase_delay",
    "pairwise_phase_consistency",
]
