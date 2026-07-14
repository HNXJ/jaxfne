"""Population-level spike and field metrics for jaxfne simulations.

Provides dimensionless, JSON-safe metrics for characterizing population activity
from neural simulations. All metrics accept JAX arrays or NumPy arrays and return
Python floats.

This module is part of the computational scaffold. Metrics are dimensionless
descriptors, not biological validation. No biological truth claims are supported.
"""

import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import jax


def mean_pairwise_spike_correlation(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    min_neurons: int = 2,
    min_events: int = 2,
) -> float:
    r"""Compute mean pairwise Pearson correlation of spike trains.

    Measures population coherence by averaging pairwise correlations across
    spike train pairs. Returns 0 if population lacks sufficient variability
    for correlation.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Values: binary indicators (0/1) or continuous spike proxies [0,1].
        Must have at least 2 neurons and 2 time steps.
    min_neurons : int, default=2
        Minimum number of neurons with nonzero variance required for correlation.
    min_events : int, default=2
        Minimum number of time steps with spike activity required.

    Returns
    -------
    float
        Mean pairwise Pearson correlation in [-1, 1].
        Returns 0.0 if population has insufficient variance or activity.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import mean_pairwise_spike_correlation
    >>> spikes = np.random.rand(10, 100) > 0.8  # 10 neurons, 100 timesteps
    >>> corr = mean_pairwise_spike_correlation(spikes)
    >>> print(f"Population correlation: {corr:.3f}")

    Notes
    -----
    - Metric is dimensionless (no units).
    - Uses Pearson correlation; requires variance in each spike train.
    - NaN/Inf values are filtered before mean; metric is robust to edge cases.
    - Claim level: computational_scaffold (no biological interpretation).
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    n_neurons, n_time = spikes_array.shape

    if n_neurons < 2 or n_time < 2:
        return 0.0

    # Filter neurons with nonzero variance
    has_variance = np.std(spikes_array, axis=1) > 0
    n_active = np.sum(has_variance)

    if n_active < min_neurons:
        return 0.0

    # Filter time steps with any activity
    has_activity = np.sum(spikes_array, axis=0) > 0
    n_active_steps = np.sum(has_activity)

    if n_active_steps < min_events:
        return 0.0

    # Compute pairwise correlations
    active_spikes = spikes_array[has_variance, :]
    corr_matrix = np.corrcoef(active_spikes)

    # Extract upper triangle (off-diagonal pairs only)
    n_active = active_spikes.shape[0]
    if n_active < 2:
        return 0.0

    upper_idx = np.triu_indices(n_active, k=1)
    pairwise_corrs = corr_matrix[upper_idx]

    # Remove NaN/Inf values
    finite_corrs = pairwise_corrs[np.isfinite(pairwise_corrs)]

    if len(finite_corrs) == 0:
        return 0.0

    return float(np.mean(finite_corrs))


def burst_index(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    bin_ms: float,
    dt_ms: float,
    active_fraction_threshold: float = 0.5,
) -> float:
    r"""Compute burst index: fraction of time with population activity.

    Measures temporal clustering of spike activity. High values indicate
    bursting behavior; low values indicate sparse or distributed spiking.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Values: binary (0/1) or continuous [0,1].
    bin_ms : float
        Binning window in milliseconds (used for conceptual framing only;
        actual binning is determined by dt_ms scaling).
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    active_fraction_threshold : float, default=0.5
        Fraction of neurons firing required to mark bin as "active" (burst).
        Range: [0, 1]. Default 0.5 = "50% of neurons firing".

    Returns
    -------
    float
        Burst index in [0, 1]. Fraction of time steps where ≥ active_fraction_threshold
        of neurons are spiking.
        0 = no bursting (distributed); 1 = continuous bursting.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import burst_index
    >>> spikes = np.random.rand(20, 1000) > 0.8  # sparse spiking
    >>> bi = burst_index(spikes, bin_ms=10, dt_ms=0.1)
    >>> print(f"Burst index: {bi:.3f}")
    >>> if bi > 0.3:
    ...     print("Network exhibits strong bursting")

    Notes
    -----
    - Metric is dimensionless.
    - dt_ms parameter is provided for user framing; computation is invariant to dt.
    - NaN/Inf spikes are treated as 0 (not firing).
    - Claim level: computational_scaffold (no oscillatory/biological claim).
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    if not (0 <= active_fraction_threshold <= 1):
        raise ValueError(
            f"active_fraction_threshold must be in [0,1], got {active_fraction_threshold}"
        )

    n_neurons, n_time = spikes_array.shape

    if n_neurons == 0 or n_time == 0:
        return 0.0

    # Handle NaN/Inf by treating as 0 (not firing)
    safe_spikes = np.nan_to_num(spikes_array, nan=0.0, posinf=0.0, neginf=0.0)
    safe_spikes = np.clip(safe_spikes, 0, 1)  # Clamp to [0,1]

    # Compute active fraction per time step
    active_fractions = np.mean(safe_spikes, axis=0)

    # Count time steps exceeding threshold
    n_bursting = np.sum(active_fractions >= active_fraction_threshold)

    return float(n_bursting / n_time)


def fano_factor(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    bin_size_ms: float = 10.0,
    dt_ms: float = 0.1,
) -> float:
    r"""Compute Fano factor of spike counts: variance / mean.

    Measures spike train irregularity. Fano = 1 (Poisson), < 1 (regular),
    > 1 (bursty). Computed per neuron, then averaged.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array.
    bin_size_ms : float, default=10.0
        Binning window in milliseconds. Population spikes are aggregated
        into bins of this size before computing variance/mean.
    dt_ms : float, default=0.1
        Simulation timestep in milliseconds. Used to compute bin size in samples.

    Returns
    -------
    float
        Fano factor (variance / mean of binned spike counts).
        Returns 0.0 if mean count is zero (no spiking).

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import fano_factor
    >>> spikes = np.random.rand(20, 10000) > 0.9  # moderately sparse
    >>> ff = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
    >>> print(f"Fano factor: {ff:.3f}")
    >>> if ff > 1.5:
    ...     print("Irregular (Poisson-like) spiking")
    >>> elif ff < 0.5:
    ...     print("Regular (oscillatory) spiking")

    Notes
    -----
    - Metric is dimensionless.
    - Bins are computed from dt_ms: bin_samples = int(bin_size_ms / dt_ms).
    - Computes spike count per bin, then var/mean across all bins.
    - Handles zero-mean case by returning 0.
    - NaN/Inf spike counts are removed before var/mean.
    - Claim level: computational_scaffold (no biological interpretation).
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be > 0, got {bin_size_ms}")

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    n_neurons, n_time = spikes_array.shape

    if n_neurons == 0 or n_time == 0:
        return 0.0

    # Compute bin size in samples
    bin_samples = max(1, int(np.round(bin_size_ms / dt_ms)))

    # Sum spikes across neurons (population count per timestep)
    pop_spikes = np.sum(spikes_array, axis=0)

    # Bin the population counts
    n_bins = n_time // bin_samples
    if n_bins == 0:
        n_bins = 1
        bin_samples = n_time

    # Truncate to exact bin count
    binned_spikes = pop_spikes[: n_bins * bin_samples].reshape(n_bins, bin_samples)
    bin_counts = np.sum(binned_spikes, axis=1)

    # Compute variance and mean
    bin_var = np.var(bin_counts)
    bin_mean = np.mean(bin_counts)

    if bin_mean <= 0:
        return 0.0

    ff = bin_var / bin_mean

    # Handle NaN/Inf
    if not np.isfinite(ff):
        return 0.0

    return float(ff)


def fleiss_kappa_binary(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    bin_ms: float,
    dt_ms: float,
) -> float:
    r"""Compute Fleiss-kappa-like agreement metric for population synchrony.

    Measures inter-rater agreement (neurons as "raters") of whether each
    time bin is "active" or "silent". Range: [-1, 1].
    - κ = 1: perfect agreement (all neurons fire in sync)
    - κ = 0: chance agreement (independent)
    - κ < 0: disagreement (anti-correlated)

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Binary (0/1) or continuous [0,1].
    bin_ms : float
        Binning window in milliseconds.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Fleiss-kappa-like metric in [-1, 1].
        Returns 0.0 if no sufficient variance for agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import fleiss_kappa_binary
    >>> spikes = np.random.rand(30, 10000) > 0.8
    >>> kappa = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
    >>> print(f"Population agreement: {kappa:.3f}")
    >>> if 0.4 <= kappa <= 0.75:
    ...     print("Good temporal coordination")

    Notes
    -----
    - Adapted from Fleiss' kappa (Cohen's kappa for multiple raters).
    - Computes: κ = (P_o - P_e) / (1 - P_e)
      - P_o: observed agreement (fraction of neurons agreeing per bin)
      - P_e: expected agreement under independence
    - Metric is dimensionless.
    - Claim level: computational_scaffold (not a formal statistical test).

    References
    ----------
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
    Psychological Bulletin, 76(5), 378–382.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    n_neurons, n_time = spikes_array.shape

    if n_neurons < 2 or n_time < 1:
        return 0.0

    # Compute bin size
    bin_samples = max(1, int(np.round(bin_ms / dt_ms)))
    n_bins = n_time // bin_samples

    if n_bins == 0:
        n_bins = 1
        bin_samples = n_time

    # Bin the spikes (binarize to "active" or "silent")
    binned_raw = spikes_array[:, : n_bins * bin_samples].reshape(
        n_neurons, n_bins, bin_samples
    )
    # Each neuron is "active" in a bin if it fires at least once in that bin
    binned = np.any(binned_raw > 0, axis=2).astype(float)  # (n_neurons, n_bins)

    # Compute observed agreement per bin
    # For each bin, count neurons agreeing (all 0 or all 1)
    n_agree_per_bin = np.sum(
        (np.sum(binned, axis=0) == 0) | (np.sum(binned, axis=0) == n_neurons)
    )
    p_o = n_agree_per_bin / n_bins  # observed proportion

    # Compute expected agreement (under independence)
    # For each neuron, estimate P(fires) and P(silent)
    p_fires = np.mean(binned, axis=1)
    p_silent = 1.0 - p_fires

    # P(random neuron fires | random neuron also fires)
    p_both_fire = np.mean(p_fires**2)
    # P(random neuron silent | random neuron also silent)
    p_both_silent = np.mean(p_silent**2)

    p_e = p_both_fire + p_both_silent

    # Fleiss-kappa
    if p_e >= 1.0:
        # No variance; return 0
        return 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)

    # Handle NaN/Inf
    if not np.isfinite(kappa):
        return 0.0

    # Clamp to [-1, 1] (can exceed due to binning)
    kappa = float(np.clip(kappa, -1.0, 1.0))

    return kappa
