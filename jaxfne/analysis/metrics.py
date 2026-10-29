"""Population-level spike and field metrics for jaxfne simulations.

Provides dimensionless, JSON-safe metrics for characterizing population activity
from neural simulations. All metrics accept JAX arrays or NumPy arrays and return
Python floats.

This module is part of the computational scaffold. Metrics are dimensionless
descriptors, not biological validation. No biological truth claims are supported.
"""

import numpy as np
import operator
import warnings
from typing import Union, TYPE_CHECKING

from scipy.signal import welch

if TYPE_CHECKING:
    import jax


def _bin_counts(
    spikes: Union[np.ndarray, "jax.Array"],
    bin_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Bin spikes into non-overlapping count windows.

    Shared helper for the declared metric definitions. ``b = bin_ms / dt_ms``
    steps per bin, which must be a whole number (else ValueError); bins are
    non-overlapping over axis 1; the trailing partial bin is dropped. Returns
    a ``(n_neurons, n_bins)`` float array, or ``(n_neurons, 0)`` when
    ``n_time // b == 0``.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(f"spikes must be 2D array, got {spikes_array.ndim}D")

    n_neurons, n_time = spikes_array.shape
    bin_samples = _validate_bin_steps(bin_ms, dt_ms)
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
        Returns NaN when the correlation is undefined (fewer than 2 bins,
        fewer than ``min_neurons`` rows with variance, fewer than
        ``min_events`` active bins, or no finite pair).

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
        return float("nan")

    n_neurons, n_bins = binned.shape

    if n_neurons < 2 or n_bins < 2:
        return float("nan")

    # Filter neurons with nonzero variance
    has_variance = np.std(binned, axis=1) > 0
    n_active = np.sum(has_variance)

    if n_active < min_neurons:
        return float("nan")

    # Filter bins with any activity
    has_activity = np.sum(binned, axis=0) > 0
    n_active_bins = np.sum(has_activity)

    if n_active_bins < min_events:
        return float("nan")

    # Compute pairwise correlations
    active_binned = binned[has_variance, :]
    corr_matrix = np.corrcoef(active_binned)

    # Extract upper triangle (off-diagonal pairs only)
    n_active = active_binned.shape[0]
    if n_active < 2:
        return float("nan")

    upper_idx = np.triu_indices(n_active, k=1)
    pairwise_corrs = corr_matrix[upper_idx]

    # Remove NaN/Inf values
    finite_corrs = pairwise_corrs[np.isfinite(pairwise_corrs)]

    if len(finite_corrs) == 0:
        return float("nan")

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
        ``active_fraction_threshold``. In [0, 1]. Returns NaN when undefined
        (zero neurons or fewer than 1 bin results).

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
        return float("nan")

    binned = _bin_counts(spikes_array, bin_ms, dt_ms)

    if binned.shape[1] == 0:
        return float("nan")

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
        Binning window in milliseconds. Must be a positive multiple of
        ``dt_ms``, else ValueError naming ``bin_size_ms`` and the nearest
        valid width.
    dt_ms : float, default=0.1
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Mean of the per-neuron variance/mean ratios over neurons with
        positive mean counts. Returns NaN when undefined (fewer than 1 bin
        results, no neuron with a positive mean count, or a non-finite
        mean ratio).

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

    _validate_bin_steps(bin_size_ms, dt_ms, name="bin_size_ms")
    binned = _bin_counts(spikes_array, bin_size_ms, dt_ms)

    if binned.shape[1] == 0:
        return float("nan")

    # Per-neuron variance/mean over neurons with positive mean counts
    means = np.mean(binned, axis=1)
    variances = np.var(binned, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = variances / means
    valid = (means > 0) & np.isfinite(ratios)

    if not np.any(valid):
        return float("nan")

    ff = float(np.mean(ratios[valid]))

    # Handle NaN/Inf
    if not np.isfinite(ff):
        return float("nan")

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
    - κ ≈ 0: independent neurons give chance agreement
    - κ < 0: disagreement (anti-correlated)

    Parameters
    ----------
    spikes : (n_neurons, n_time) ndarray
        Spike indicator array. Shape: (n_neurons, n_time_steps).
        Binary (0/1) or continuous [0,1].
    bin_ms : float
        Binning window in milliseconds. Must be a positive multiple of
        ``dt_ms``, else ValueError naming ``bin_ms`` and the nearest valid
        width.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Fleiss-kappa in [-1, 1].
        Returns NaN when kappa is mathematically undefined (fewer than 2
        neurons, zero time steps, zero full bins (``n_bins == 0``), expected
        agreement ``p_e == 1``, or a non-finite kappa) (undefined).

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
    - Textbook Fleiss (1971) kappa for two categories: each bin is a
      subject, each neuron a rater, and a neuron is "active" in a bin when
      it fires at least once there.
    - With n neurons and n_i1 active and n_i0 silent in bin i:
      P_i = (n_i1 (n_i1 - 1) + n_i0 (n_i0 - 1)) / (n (n - 1)),
      P_bar = mean over bins of P_i, p_1 = total active / (n_bins n),
      p_0 = 1 - p_1, P_e = p_1^2 + p_0^2, κ = (P_bar - P_e) / (1 - P_e).
    - NaN when P_e = 1, which holds only when the whole raster is all
      active or all silent.
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
    bin_samples = _validate_bin_steps(bin_ms, dt_ms)

    if n_neurons < 2 or n_time < 1:
        return float("nan")

    n_bins = n_time // bin_samples

    if n_bins == 0:
        return float("nan")

    # Bin the spikes (binarize to "active" or "silent")
    binned_raw = spikes_array[:, : n_bins * bin_samples].reshape(
        n_neurons, n_bins, bin_samples
    )
    # Each neuron is "active" in a bin if it fires at least once in that bin
    binned = np.any(binned_raw > 0, axis=2).astype(float)  # (n_neurons, n_bins)

    # Textbook Fleiss (1971), two categories: bins are subjects, neurons raters.
    n_active = np.sum(binned, axis=0)  # active raters per bin
    n_silent = n_neurons - n_active
    p_i = (n_active * (n_active - 1) + n_silent * (n_silent - 1)) / (
        n_neurons * (n_neurons - 1)
    )
    p_bar = float(np.mean(p_i))  # mean observed agreement over bins

    # Chance agreement from the pooled active and silent proportions
    p_1 = float(np.sum(binned)) / (n_bins * n_neurons)
    p_0 = 1.0 - p_1
    p_e = p_1**2 + p_0**2

    # Fleiss-kappa
    if p_e >= 1.0:
        # No variance: expected agreement is 1, so kappa is undefined
        return float("nan")

    kappa = (p_bar - p_e) / (1.0 - p_e)

    # Handle NaN/Inf
    if not np.isfinite(kappa):
        return float("nan")

    # Clamp to [-1, 1] (can exceed due to binning)
    kappa = float(np.clip(kappa, -1.0, 1.0))

    return kappa


# --- T-23 activity-regime classifier (Hamm 2026-10-14) -------------------------

#: Default decision boundaries for :func:`activity_regime`. Every threshold the
#: classifier applies lives in this dict; per-study overrides pass through the
#: ``thresholds`` argument, which may replace any named entry here.
ACTIVITY_REGIME_DEFAULTS = {
    "silent_rate_hz": 0.1,
    "silent_active_fraction": 0.05,
    "saturated_active_fraction": 0.95,
    "saturated_kappa": 0.5,
    "saturated_rate_hz": 100.0,
    "runaway_ratio": 2.0,
    "oscillation_floor": 5.0,
    "async_kappa": 0.1,
    "async_cv_lo": 0.7,
    "async_cv_hi": 1.5,
    "bin_ms": 5.0,
    "welch_nperseg": 256,
}


def _validate_bin_steps(bin_ms: float, dt_ms: float, name: str = "bin_ms", rel_tol: float = 1e-9) -> int:
    """Steps per bin, refusing a bin width that misses the time grid.

    The message names ``name`` (the caller's bin argument) and the nearest
    valid bin width. ``rel_tol`` is the relative tolerance on ``bin_ms / dt_ms``.
    """
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")
    if bin_ms <= 0:
        raise ValueError(f"{name} must be > 0, got {bin_ms}")
    ratio = bin_ms / dt_ms
    steps = int(round(ratio))
    if steps < 1 or abs(ratio - steps) > rel_tol * abs(ratio):
        nearest = max(1, steps) * dt_ms
        raise ValueError(
            f"{name}={bin_ms!r} must be a positive multiple of dt_ms={dt_ms!r} "
            f"(bin / dt = {ratio!r}, tolerance {rel_tol:g} relative); "
            f"nearest valid bin width is {nearest:.12g} ms"
        )
    return steps


def wang_buzsaki_kappa(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
    bin_ms: float = 5.0,
) -> float:
    r"""Mean pairwise coincidence of binary binned spike trains.

    Each cell is reduced to a binary train over non-overlapping bins of
    ``bin_ms`` (1 when the cell fires at least once in the bin, else 0; the
    trailing partial bin is dropped). For cells ``i`` and ``j`` with binned
    trains ``X_i`` and ``X_j``::

        kappa_ij = sum(X_i * X_j) / sqrt(sum(X_i) * sum(X_j)).

    The metric is the mean of ``kappa_ij`` over pairs ``i != j`` whose trains
    are both non-empty.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Values: binary indicators (0/1); any positive value counts as
        a spike, non-positive values (including NaN) count as silence.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    bin_ms : float, default=5.0
        Binning window in milliseconds. Must be a positive multiple of
        ``dt_ms`` within 1e-9 relative, else ValueError naming ``bin_ms``.

    Returns
    -------
    float
        Mean pairwise coincidence in [0, 1]. Returns NaN when undefined
        (fewer than 2 cells with a non-empty binned train, or zero full
        bins).

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import wang_buzsaki_kappa
    >>> spikes = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=bool)
    >>> wang_buzsaki_kappa(spikes, 1.0, bin_ms=2.0)
    0.0

    Notes
    -----
    Declared 2026-10-14 (T-23, Hamm): the Wang-Buzsaki coherence on binary
    spike counts in declared bins. It measures coincidence of active bins,
    while ``tutorial_utils.kappa_synchrony`` is a Pearson correlation at dt
    resolution and ``fleiss_kappa_binary`` is an agreement-against-chance
    score; the three kappas answer different questions.

    References
    ----------
    Wang, X.-J., & Buzsáki, G. (1996). Gamma oscillation by synaptic
    inhibition in a hippocampal interneuronal network model. *The Journal
    of Neuroscience* 16(20), 6402-6413.
    doi:10.1523/JNEUROSCI.16-20-06402.1996
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(
            f"spikes must be 2D [n_steps, n_neurons], got {spikes_array.ndim}D"
        )

    _validate_bin_steps(bin_ms, dt_ms)

    # Shared binning with the sibling metrics: _bin_counts takes
    # (n_neurons, n_time) count windows, so the input goes in transposed.
    # Binarize per step first: a non-positive value is silence and must not
    # cancel a spike in the same bin.
    counts = _bin_counts((spikes_array > 0).T.astype(float), bin_ms, dt_ms)

    if counts.shape[1] == 0:
        return float("nan")

    binary = (counts.T > 0).astype(float)  # (n_bins, n_neurons)
    totals = binary.sum(axis=0)
    has_train = totals > 0

    if int(np.sum(has_train)) < 2:
        return float("nan")

    active = binary[:, has_train]
    sums = totals[has_train]
    gram = active.T @ active
    scale = np.sqrt(np.outer(sums, sums))
    with np.errstate(divide="ignore", invalid="ignore"):
        pairwise = gram / scale
    n_active = active.shape[1]
    return float((pairwise.sum() - n_active) / (n_active * (n_active - 1)))


def _resolve_thresholds(thresholds: Union[dict, None]) -> dict:
    """Merge a per-call override over :data:`ACTIVITY_REGIME_DEFAULTS`."""
    merged = dict(ACTIVITY_REGIME_DEFAULTS)
    if thresholds is None:
        return merged
    unknown = [key for key in thresholds if key not in merged]
    if unknown:
        raise ValueError(
            f"unknown activity_regime threshold(s): {sorted(unknown)}; "
            f"known: {sorted(merged)}"
        )
    merged.update(thresholds)
    return merged


def _spectral_peak_excess(
    population_rate_hz: np.ndarray,
    dt_ms: float,
    nperseg: int,
) -> float:
    """Peak-over-median Welch power of the population rate in 5-200 Hz."""
    n_points = int(population_rate_hz.size)
    try:
        segments = int(nperseg)
    except (TypeError, ValueError):
        return float("nan")
    segments = min(segments, n_points)
    if segments < 8:
        return float("nan")
    freqs, power = welch(
        np.asarray(population_rate_hz, dtype=float),
        fs=1000.0 / dt_ms,
        nperseg=segments,
        detrend="constant",
    )
    band = power[(freqs >= 5.0) & (freqs <= 200.0)]
    if band.size == 0:
        return float("nan")
    median = float(np.median(band))
    peak = float(band.max())
    if median <= 0.0:
        return float("inf") if peak > 0.0 else 1.0
    return float(peak / median)


def _spectral_peak_hz(
    population_rate_hz: np.ndarray,
    dt_ms: float,
    nperseg: int,
) -> float:
    """Frequency of the peak Welch power of the population rate in 5-200 Hz."""
    n_points = int(population_rate_hz.size)
    try:
        segments = int(nperseg)
    except (TypeError, ValueError):
        return float("nan")
    segments = min(segments, n_points)
    if segments < 8:
        return float("nan")
    freqs, power = welch(
        np.asarray(population_rate_hz, dtype=float),
        fs=1000.0 / dt_ms,
        nperseg=segments,
        detrend="constant",
    )
    in_band = (freqs >= 5.0) & (freqs <= 200.0)
    if not in_band.any() or not np.any(power[in_band] > 0.0):
        return float("nan")
    return float(freqs[in_band][np.argmax(power[in_band])])


def activity_regime(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
    *,
    kept_window: Union[tuple, list, None] = None,
    thresholds: Union[dict, None] = None,
) -> dict:
    r"""Classify one population activity trace into a standard regime.

    The classifier aggregates the kept window into scalar descriptors and
    applies the decision boundaries in order, returning the first match:

    silent
        Mean rate below ``silent_rate_hz`` or active fraction below
        ``silent_active_fraction``.
    saturated
        Active fraction above ``saturated_active_fraction`` with kappa above
        ``saturated_kappa``, or mean rate above ``saturated_rate_hz``.
    runaway
        Second-half rate above ``runaway_ratio`` times the first-half rate
        (first-half rate above zero).
    oscillatory
        Welch spectral peak excess of the per-step population rate
        (peak over median power in 5-200 Hz) at or above
        ``oscillation_floor``.
    asynchronous
        Kappa below ``async_kappa`` with CV_ISI inside
        [``async_cv_lo``, ``async_cv_hi``].
    irregular
        None of the above.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Any positive value counts as a spike.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    kept_window : (start_step, end_step) or None, default=None
        Half-open step window the labels are computed over; None keeps the
        whole trace. Must satisfy ``0 <= start < end <= n_steps``, else
        ValueError naming ``kept_window``.
    thresholds : dict or None, default=None
        Per-call overrides of named entries of
        :data:`ACTIVITY_REGIME_DEFAULTS`. An unknown key raises ValueError
        naming it.

    Returns
    -------
    dict
        Per-call dict with ``rate_hz`` (mean spikes per neuron per second),
        ``active_fraction`` (cells with at least one spike), ``kappa``
        (:func:`wang_buzsaki_kappa` at ``bin_ms``), ``cv_isi`` (mean
        ISI coefficient of variation over cells with at least 3 spikes, NaN
        when no cell qualifies), ``rate_ratio`` (second-half over first-half
        rate; inf when only the second half fires, NaN when neither does),
        ``spectral_peak_excess``, ``spectral_peak_hz`` (frequency of that
        Welch peak in 5-200 Hz; NaN when the band holds no power),
        ``label`` (one of ``"silent"``,
        ``"saturated"``, ``"runaway"``, ``"asynchronous"``,
        ``"oscillatory"``, ``"irregular"``), and ``thresholds`` (the merged
        boundaries this call applied).

    Notes
    -----
    Declared 2026-10-14 (T-23, Hamm): one standard classifier on the executed
    population activity, applied per population over the kept window with
    study-overridable defaults. It is distinct from
    ``jaxfne.w3b_parameter_domain.classify_regime``, which classifies a
    parameter point from synthesis diagnostics rather than executed spikes.
    Its ``kappa`` descriptor is :func:`wang_buzsaki_kappa`.

    References
    ----------
    Wang, X.-J., & Buzsáki, G. (1996). Gamma oscillation by synaptic
    inhibition in a hippocampal interneuronal network model. *The Journal
    of Neuroscience* 16(20), 6402-6413.
    doi:10.1523/JNEUROSCI.16-20-06402.1996
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(
            f"spikes must be 2D [n_steps, n_neurons], got {spikes_array.ndim}D"
        )

    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    n_steps, n_cells = spikes_array.shape

    if kept_window is None:
        start, end = 0, n_steps
    else:
        expect = (
            "kept_window must be (start_step, end_step) with "
            f"0 <= start_step < end_step <= {n_steps}, got {kept_window!r}"
        )
        try:
            first, last = kept_window
            start, end = operator.index(first), operator.index(last)
        except (TypeError, ValueError):
            raise ValueError(expect) from None
        if not (0 <= start < end <= n_steps):
            raise ValueError(expect)

    limits = _resolve_thresholds(thresholds)

    kept = spikes_array[start:end]
    fired = kept > 0
    window_steps = end - start
    duration_s = window_steps * dt_ms / 1000.0
    per_cell = fired.sum(axis=0)

    if n_cells and duration_s > 0:
        rate_hz = float(fired.sum() / (n_cells * duration_s))
        active_fraction = float(np.sum(per_cell > 0) / n_cells)
    else:
        rate_hz = float("nan")
        active_fraction = float("nan")

    kappa = wang_buzsaki_kappa(kept, dt_ms, bin_ms=limits["bin_ms"])

    cvs = []
    for j in range(n_cells):
        onsets = np.flatnonzero(fired[:, j])
        if onsets.size >= 3:
            isi_ms = np.diff(onsets).astype(float) * dt_ms
            cvs.append(float(isi_ms.std() / isi_ms.mean()))
    cv_isi = float(np.mean(cvs)) if cvs else float("nan")

    half = window_steps // 2
    if half == 0 or n_cells == 0:
        first_rate = second_rate = float("nan")
        rate_ratio = float("nan")
    else:
        first_rate = float(fired[:half].sum() / (n_cells * half * dt_ms / 1000.0))
        rest = window_steps - half
        second_rate = float(fired[half:].sum() / (n_cells * rest * dt_ms / 1000.0))
        if first_rate > 0.0:
            rate_ratio = float(second_rate / first_rate)
        elif second_rate > 0.0:
            rate_ratio = float("inf")
        else:
            rate_ratio = float("nan")

    if n_cells:
        population_rate_hz = fired.sum(axis=1).astype(float) / (
            n_cells * dt_ms / 1000.0
        )
    else:
        population_rate_hz = np.zeros(window_steps)
    excess = _spectral_peak_excess(
        population_rate_hz, dt_ms, limits["welch_nperseg"]
    )

    if (
        not np.isfinite(rate_hz)
        or rate_hz < limits["silent_rate_hz"]
        or active_fraction < limits["silent_active_fraction"]
    ):
        label = "silent"
    elif (
        active_fraction > limits["saturated_active_fraction"]
        and kappa > limits["saturated_kappa"]
    ) or rate_hz > limits["saturated_rate_hz"]:
        label = "saturated"
    elif (
        np.isfinite(first_rate)
        and first_rate > 0.0
        and second_rate > limits["runaway_ratio"] * first_rate
    ):
        label = "runaway"
    elif not np.isnan(excess) and excess >= limits["oscillation_floor"]:
        label = "oscillatory"
    elif (
        kappa < limits["async_kappa"]
        and limits["async_cv_lo"] <= cv_isi <= limits["async_cv_hi"]
    ):
        label = "asynchronous"
    else:
        label = "irregular"

    return {
        "rate_hz": rate_hz,
        "active_fraction": active_fraction,
        "kappa": kappa,
        "cv_isi": cv_isi,
        "rate_ratio": rate_ratio,
        "spectral_peak_excess": excess,
        "spectral_peak_hz": _spectral_peak_hz(
            population_rate_hz, dt_ms, limits["welch_nperseg"]
        ),
        "label": label,
        "thresholds": dict(limits),
    }


def activity_regimes(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
    populations: dict,
    *,
    kept_window: Union[tuple, list, None] = None,
    thresholds: Union[dict, None] = None,
) -> dict:
    r"""Classify named neuron subsets of one trace with :func:`activity_regime`.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Any positive value counts as a spike.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    populations : dict[str, index array]
        Mapping of population name to the neuron columns it covers. Each
        entry is classified independently over the same kept window.
    kept_window : (start_step, end_step) or None, default=None
        Half-open step window shared by every population.
    thresholds : dict or None, default=None
        Per-call overrides shared by every population.

    Returns
    -------
    dict
        Mapping of population name to its :func:`activity_regime` dict.
    """
    spikes_array = np.asarray(spikes)

    if spikes_array.ndim != 2:
        raise ValueError(
            f"spikes must be 2D [n_steps, n_neurons], got {spikes_array.ndim}D"
        )

    if not isinstance(populations, dict) or len(populations) == 0:
        raise ValueError(
            "populations must be a non-empty dict of name -> neuron index "
            f"array, got {populations!r}"
        )

    return {
        name: activity_regime(
            spikes_array[:, np.asarray(index)],
            dt_ms,
            kept_window=kept_window,
            thresholds=thresholds,
        )
        for name, index in populations.items()
    }


# --- Destexhe 2009 lane estimators (T-32d stage 1) ---------------------------


def cv_isi(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
) -> float:
    r"""Mean over spiking cells of sigma_ISI / mean_ISI (paper Eq. 3).

    For each cell with at least 3 spikes, the spike times give the
    interspike intervals ``ISI``; the cell contributes
    ``std(ISI) / mean(ISI)`` (population std, ddof 0). The metric is the
    mean over the contributing cells.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Any nonzero value counts as a spike.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Mean ISI coefficient of variation over cells with at least 3 spikes.
        Returns NaN when no cell qualifies.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import cv_isi
    >>> spikes = np.zeros((10, 2), dtype=bool)
    >>> spikes[[1, 3, 5, 7], 0] = True
    >>> cv_isi(spikes, 1.0)
    0.0

    Notes
    -----
    Moved verbatim from the Destexhe 2009 lane
    (``artifacts/etudes/papers/destexhe2009/destexhe_api.py``), which applies
    it to the free-run raster. Cells with fewer than 3 spikes carry no ISI
    variance and are excluded.
    """
    arr = np.asarray(spikes, dtype=bool)
    cvs = []
    for col in range(arr.shape[1]):
        t = np.flatnonzero(arr[:, col]).astype(float) * dt_ms
        if t.size < 3:
            continue
        isi = np.diff(t)
        if isi.mean() <= 0:
            continue
        cvs.append(float(isi.std() / isi.mean()))
    if not cvs:
        return float("nan")
    return float(np.mean(cvs))


def pairwise_cc_binned_disjoint(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
    seed: int = 0,
    bin_ms: float = 5.0,
    n_pairs: int = 500,
) -> float:
    r"""Mean Pearson correlation over disjoint pairs in binned spike counts.

    Spikes are grouped into non-overlapping bins of ``bin_ms / dt_ms`` steps,
    which must be a whole number (else ValueError naming ``bin_ms``); the
    trailing partial bin is dropped. Counts (not binarized) fill a
    ``(n_bins, n_neurons)`` table. A seeded
    permutation of the cells is cut into disjoint pairs (at most
    ``n_pairs``); the metric is the mean Pearson r over pairs where both
    binned trains have nonzero variance.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Any nonzero value counts as a spike.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.
    seed : int, default=0
        Seed driving the disjoint pairing permutation.
    bin_ms : float, default=5.0
        Binning window in milliseconds.
    n_pairs : int, default=500
        Maximum number of disjoint pairs sampled.

    Returns
    -------
    float
        Mean pairwise Pearson correlation in [-1, 1]. Returns NaN when no
        sampled pair has variance on both trains.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import pairwise_cc_binned_disjoint
    >>> rng = np.random.default_rng(0)
    >>> spikes = rng.random((1000, 6)) < 0.01
    >>> cc = pairwise_cc_binned_disjoint(spikes, 0.5, seed=0)

    Notes
    -----
    Moved verbatim from the Destexhe 2009 lane
    (``artifacts/etudes/papers/destexhe2009/destexhe_api.py``), which scores
    it in 5 ms bins over 500 disjoint pairs (paper Eq. 4). It differs from
    :func:`mean_pairwise_spike_correlation`: that function takes
    ``(n_neurons, n_time)`` input, averages over all upper-triangle pairs
    with no seed, and filters low-activity bins, while this one takes
    ``(n_steps, n_neurons)`` input and averages over a seeded disjoint
    pairing with no bin filtering.
    """
    arr = np.asarray(spikes, dtype=bool)
    width = _validate_bin_steps(bin_ms, dt_ms)
    cut = (arr.shape[0] // width) * width
    binned = arr[:cut].reshape(cut // width, width, arr.shape[1]).sum(axis=1)
    n = arr.shape[1]
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    order = order[: (len(order) // 2) * 2].reshape(-1, 2)
    order = order[:n_pairs]
    rs = []
    for i, j in order:
        x = binned[:, i].astype(float)
        y = binned[:, j].astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    if not rs:
        return float("nan")
    return float(np.mean(rs))


def mean_rate_hz(
    spikes: Union[np.ndarray, "jax.Array"],
    dt_ms: float,
) -> float:
    r"""Population mean firing rate (Hz) over the whole raster.

    Total spikes divided by the number of cells times the raster duration
    in seconds.

    Parameters
    ----------
    spikes : (n_steps, n_neurons) ndarray
        Spike indicator array with one row per time step and one column per
        neuron. Any nonzero value counts as a spike.
    dt_ms : float
        Simulation timestep in milliseconds. Must be > 0.

    Returns
    -------
    float
        Mean spikes per neuron per second. Returns 0.0 for a silent raster.

    Examples
    --------
    >>> import numpy as np
    >>> from jaxfne.analysis import mean_rate_hz
    >>> spikes = np.zeros((1000, 2), dtype=bool)
    >>> spikes[::100, 0] = True
    >>> mean_rate_hz(spikes, 1.0)
    5.0

    Notes
    -----
    Moved verbatim from the Destexhe 2009 lane
    (``artifacts/etudes/papers/destexhe2009/destexhe_api.py``), which scores
    it per carrier over the free-run raster.
    """
    arr = np.asarray(spikes, dtype=bool)
    seconds = arr.shape[0] * dt_ms / 1000.0
    return float(arr.sum() / (arr.shape[1] * seconds))


# --- CAMINOS lane estimators (T-32d stage 2: generic core homes) ----------------

_POP_DEFAULTS = ("E", "PV", "SOM")
_DISCARD_S_DEFAULT = 2.0


def _seconds_from_ms_or_s(name_s, value_s, name_ms, value_ms, default_s=None):
    """Seconds value of one quantity given as ``name_ms`` or the deprecated ``name_s``."""
    if value_s is not None and value_ms is not None:
        raise ValueError(f"pass only one of {name_ms} and {name_s}")
    if value_ms is not None:
        return float(value_ms) / 1000.0
    if value_s is not None:
        warnings.warn(
            f"{name_s} is deprecated and will be removed; pass {name_ms} (ms) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return float(value_s)
    return default_s


def population_rates(counts, dt=None, sizes=None, pops=_POP_DEFAULTS, discard_s=None,
                     *, dt_ms=None, discard_ms=None):
    """Mean rate (Hz) per population over the kept window.

    ``counts`` is a per-step population count array ``[steps, n_pops]`` in
    ``pops`` order; ``dt_ms`` is the step in milliseconds; ``sizes`` maps each
    name in ``pops`` to its cell count. Rows before ``discard_ms`` are dropped
    (default 2000 ms: prereg Runs discard the first 2 s). Returns a dict of
    Python floats.

    The seconds forms ``dt`` and ``discard_s`` are deprecated for one release
    and emit DeprecationWarning. Passing both forms of one quantity raises
    ValueError.
    """
    if sizes is None:
        raise TypeError("population_rates() missing required argument: 'sizes'")
    dt_s = _seconds_from_ms_or_s("dt", dt, "dt_ms", dt_ms)
    if dt_s is None:
        raise TypeError("population_rates() needs dt_ms")
    discard = _seconds_from_ms_or_s(
        "discard_s", discard_s, "discard_ms", discard_ms, _DISCARD_S_DEFAULT
    )
    arr = np.asarray(counts)
    if arr.ndim != 2 or arr.shape[1] != len(pops):
        raise ValueError("counts must be [steps, %d], got shape %r" % (len(pops), arr.shape))
    cut = int(round(discard / dt_s))
    if cut >= arr.shape[0]:
        raise ValueError("run shorter than the %.g s discard: %d steps at dt=%g"
                         % (discard, arr.shape[0], dt_s))
    keep = arr[cut:]
    dur = keep.shape[0] * dt_s
    return {p: float(keep[:, i].sum() / dur / sizes[p]) for i, p in enumerate(pops)}


def xcorr_peak_lag(a, b, dt_ms, window_ms=14.0):
    """Peak lag of the cross-correlation of two rate traces, in steps.

    ``a`` and ``b`` are 1D smoothed rate traces (demeaned here); the
    correlation is searched over ``-window_ms..+window_ms``. Returns an int:
    positive means ``a`` after ``b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = a - a.mean()
    y = b - b.mean()
    n = len(x)
    max_lag = int(round(float(window_ms) / float(dt_ms)))
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.empty(lags.size)
    for j, lag in enumerate(lags):
        if lag >= 0:
            cc[j] = np.dot(x[lag:], y[:n - lag])
        else:
            cc[j] = np.dot(x[:n + lag], y[-lag:])
    return int(lags[int(np.argmax(cc))])
