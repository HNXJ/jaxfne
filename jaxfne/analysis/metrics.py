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


def _bin_counts(
    spikes: Union[np.ndarray, "jax.Array"],
    bin_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Bin spikes into non-overlapping count windows.

    Shared helper for the declared metric definitions. ``b = max(1,
    round(bin_ms / dt_ms))`` steps per bin; bins are non-overlapping over
    axis 1; the trailing partial bin is dropped. Returns a
    ``(n_neurons, n_bins)`` float array, or ``(n_neurons, 0)`` when
    ``n_time // b == 0``.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if bin_ms <= 0:
        raise ValueError(f"bin_ms must be > 0, got {bin_ms}")

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    n_neurons, n_time = spikes_array.shape
    bin_samples = max(1, int(np.round(bin_ms / dt_ms)))
    n_bins = n_time // bin_samples

    if n_bins == 0:
        return np.zeros((n_neurons, 0), dtype=float)

    trimmed = spikes_array[:, : n_bins * bin_samples]
    return trimmed.reshape(n_neurons, n_bins, bin_samples).sum(axis=2).astype(float)


def mean_pairwise_spike_correlation(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    bin_ms: float,
    dt_ms: float,
    min_neurons: int = 2,
    min_events: int = 2,
) -> float:
    r"""Compute mean pairwise Pearson correlation of binned spike counts.

    Spikes are grouped into non-overlapping bins of ``max(1,
    round(bin_ms / dt_ms))`` steps (trailing partial bin dropped), then the
    pairwise Pearson r is averaged over the upper triangle of the
    correlation matrix of the binned count rows.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Values: binary indicators (0/1) or continuous spike proxies [0,1].
        Must have at least 2 neurons and enough steps for 2 bins.
    bin_ms : float
        Binning window in milliseconds. Required keyword argument.
    dt_ms : float
        Simulation timestep in milliseconds. Required keyword argument.
    min_neurons : int, default=2
        Minimum number of binned-count rows with nonzero variance required.
    min_events : int, default=2
        Minimum number of bins with any spike activity required.

    Returns
    -------
    float
        Mean pairwise Pearson correlation in [-1, 1].
        Returns 0.0 if the binned population has insufficient variance
        or activity, or when fewer than 2 bins result.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import mean_pairwise_spike_correlation
    >>> spikes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    >>> mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
    -1.0

    Notes
    -----
    Definition declared 2026-10-13 (T-13); before it, the mean pairwise
    Pearson r of raw per-step spike rows with no binning.

    See also
    --------
    ``jnwb.spike_count_correlation`` on per-unit spike times with an
    explicit ``window_s`` and ``bin_ms``, reached via
    ``jaxfne.jnwb_view.to_jnwb(...).spike_times_s``.
    """
    binned = _bin_counts(spikes, bin_ms, dt_ms)

    if binned.shape[1] == 0:
        return 0.0

    n_neurons, n_bins = binned.shape

    if n_neurons < 2 or n_bins < 2:
        return 0.0

    # Filter neurons with nonzero variance
    has_variance = np.std(binned, axis=1) > 0
    n_active = np.sum(has_variance)

    if n_active < min_neurons:
        return 0.0

    # Filter bins with any activity
    has_activity = np.sum(binned, axis=0) > 0
    n_active_bins = np.sum(has_activity)

    if n_active_bins < min_events:
        return 0.0

    # Compute pairwise correlations
    active_binned = binned[has_variance, :]
    corr_matrix = np.corrcoef(active_binned)

    # Extract upper triangle (off-diagonal pairs only)
    n_active = active_binned.shape[0]
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
    r"""Compute the fraction of binned windows reaching population activity.

    Spikes are grouped into non-overlapping bins of ``max(1,
    round(bin_ms / dt_ms))`` steps (trailing partial bin dropped). Per bin,
    the fraction of neurons with at least one spike (count > 0) is computed;
    the metric is the fraction of bins where that fraction is at least
    ``active_fraction_threshold``.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Values: binary (0/1) or continuous [0,1].
    bin_ms : float
        Binning window in milliseconds. Must be > 0.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    active_fraction_threshold : float, default=0.5
        A bin counts when the fraction of neurons with count > 0 in that
        bin is at least this value. Range [0, 1].

    Returns
    -------
    float
        Fraction of bins whose active-neuron fraction is at least
        ``active_fraction_threshold``. In [0, 1]. Returns 0.0 when fewer
        than 1 bin results.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import burst_index
    >>> spikes = np.zeros((2, 4))
    >>> spikes[:, 0] = 1.0
    >>> burst_index(spikes, bin_ms=1, dt_ms=1)
    0.25

    Notes
    -----
    Definition declared 2026-10-13 (T-13); before it, the fraction of raw
    time steps reaching threshold with bin_ms accepted but unused.

    See also
    --------
    ``jnwb.network_burst_index`` on per-unit spike times with ``window_s``,
    ``bin_ms``, ``threshold_hz`` and ``min_duration_ms``, reached via
    ``jaxfne.jnwb_view.to_jnwb(...).spike_times_s``.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if not (0 <= active_fraction_threshold <= 1):
        raise ValueError(
            f"active_fraction_threshold must be in [0,1], got {active_fraction_threshold}"
        )

    n_neurons = spikes_array.shape[0]
    if n_neurons == 0:
        return 0.0

    binned = _bin_counts(spikes_array, bin_ms, dt_ms)

    if binned.shape[1] == 0:
        return 0.0

    # Fraction of neurons with at least one spike per bin
    active_fractions = np.mean(binned > 0, axis=0)

    # Count bins reaching threshold
    n_bursting = np.sum(active_fractions >= active_fraction_threshold)

    return float(n_bursting / binned.shape[1])


def fano_factor(
    spikes: Union[np.ndarray, "jax.Array"],
    *,
    bin_size_ms: float = 10.0,
    dt_ms: float = 0.1,
) -> float:
    r"""Compute the per-neuron Fano factor averaged over neurons.

    Spikes are grouped into non-overlapping bins of ``max(1,
    round(bin_size_ms / dt_ms))`` steps (trailing partial bin dropped). For
    each neuron with a positive mean binned count, ``F_i = var(c_i) /
    mean(c_i)`` (population variance, ddof 0); the metric is the mean of
    ``F_i`` over those neurons.

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array.
    bin_size_ms : float, default=10.0
        Binning window in milliseconds. Must be > 0.
    dt_ms : float, default=0.1
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Mean of the per-neuron variance/mean ratios over neurons with
        positive mean counts. Returns 0.0 when no neuron has a positive
        mean, when fewer than 1 bin results, or when the mean ratio is
        not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import fano_factor
    >>> spikes = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    >>> fano_factor(spikes, bin_size_ms=1, dt_ms=1)
    0.5

    Notes
    -----
    Definition declared 2026-10-13 (T-13); before it, the variance/mean of
    the population-summed binned count.

    See also
    --------
    ``jnwb.fano_factor`` on per-unit trial counts with ``onsets_s``,
    ``window_s`` and ``summary``, reached via
    ``jaxfne.jnwb_view.to_jnwb_trials(...)``.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be > 0, got {bin_size_ms}")

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    binned = _bin_counts(spikes_array, bin_size_ms, dt_ms)

    if binned.shape[1] == 0:
        return 0.0

    # Per-neuron variance/mean over neurons with positive mean counts
    means = np.mean(binned, axis=1)
    variances = np.var(binned, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = variances / means
    valid = (means > 0) & np.isfinite(ratios)

    if not np.any(valid):
        return 0.0

    ff = float(np.mean(ratios[valid]))

    # Handle NaN/Inf
    if not np.isfinite(ff):
        return 0.0

    return ff


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
