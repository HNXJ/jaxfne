"""
Spectrolaminar objective and null distribution functions.

This module implements spectrolaminar profile scoring with null distributions
and synchrony gates for multi-area neural circuits.

Scope: Scoring objectives for relative spectrolaminar profiles. Evidence: finite
output arrays, valid JSON serialization. Interpretation: simulated spectrolaminar
readouts (alpha/beta and gamma profiles) matched to targets under null distributions
and synchrony constraints. Physical amplitude claims NOT allowed.

truth_mode: truth_safe_unverified
"""

import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import warnings


def spectrolaminar_profile_score(
    readout: Dict[str, Any],
    target_alpha_beta: np.ndarray,
    target_gamma: np.ndarray,
    similarity_metric: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Compute spectrolaminar profile score without null distribution.

    Parameters
    ----------
    readout : dict
        Readout dict from spectrolaminar_readout() with keys:
        'alpha_beta', 'gamma', 'pos_from_l4', etc.
    target_alpha_beta : np.ndarray
        Target alpha/beta profile [n_contacts,]
    target_gamma : np.ndarray
        Target gamma profile [n_contacts,]
    similarity_metric : callable, optional
        Similarity function (readout, targets) -> float [0, 100].
        Default: MSE-based similarity.

    Returns
    -------
    dict
        Report with:
        - score_type: "profile_score_no_null"
        - profile_score_percent: float [0, 100]
        - motif_gate_percent: null
        - S_lam: null (only with null distribution)
        - ... metadata ...
    """
    if similarity_metric is None:
        # Default MSE-based similarity
        mse_ab = np.mean((readout["alpha_beta"] - target_alpha_beta) ** 2)
        mse_gamma = np.mean((readout["gamma"] - target_gamma) ** 2)
        mse_total = mse_ab + mse_gamma
        similarity = 100.0 * np.exp(-3.0 * mse_total)
    else:
        similarity = similarity_metric(readout, (target_alpha_beta, target_gamma))

    # Ensure finite
    profile_score = float(np.clip(similarity, 0.0, 100.0))
    if not np.isfinite(profile_score):
        profile_score = 0.0

    report = {
        "objective_kind": "spectrolaminar_profile",
        "score_type": "profile_score_no_null",
        "profile_score_percent": profile_score,
        "motif_gate_percent": None,
        "S_lam": None,
        "nulls_run": False,
        "null_distribution_n": 0,
        "null_normalization_method": None,
        "synchrony_checked": False,
        "synchrony_metric": None,
        "synchrony_value": None,
        "synchrony_threshold": None,
        "synchrony_rejection": False,
        "rejection_reasons": [],
        "uses_teaching_control_source": readout.get("metadata", {}).get(
            "teaching_control_source", False
        ),
        "default_evidence_path": readout.get("metadata", {}).get(
            "default_evidence_path", True
        ),
        "physical_amplitude_claim_allowed": False,
        "truth_mode": "truth_safe_unverified",
        "bands": {
            "alpha_beta": [8.0, 25.0],
            "gamma": [40.0, 150.0],
        },
    }

    return report


def layer_shuffle_null(readout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Layer shuffle null: permute profiles across L4-relative positions.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma', 'pos_from_l4'

    Returns
    -------
    dict
        Shuffled profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()
    pos = readout["pos_from_l4"]

    # Shuffle by depth (pos_from_l4)
    perm = np.random.permutation(len(ab))

    return {
        "alpha_beta_shuffled": ab[perm],
        "gamma_shuffled": gamma[perm],
        "null_type": "layer_shuffle",
    }


def band_label_shuffle_null(readout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Band label shuffle null: swap alpha/beta and gamma profiles.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma'

    Returns
    -------
    dict
        Swapped profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    # Swap the two band profiles
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()

    # Swap alpha/beta with gamma
    ab_shuffled = gamma.copy()
    gamma_shuffled = ab.copy()

    return {
        "alpha_beta_shuffled": ab_shuffled,
        "gamma_shuffled": gamma_shuffled,
        "null_type": "band_label_shuffle",
    }


def uniform_gain_null(readout: Dict[str, Any], gain_min: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Uniform gain null: scale profiles by random uniform gain.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma'
    gain_min : float
        Minimum gain factor (default 0.1)

    Returns
    -------
    dict
        Scaled profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()

    # Random gain [gain_min, 1.0]
    gain = np.random.uniform(gain_min, 1.0)

    return {
        "alpha_beta_shuffled": ab * gain,
        "gamma_shuffled": gamma * gain,
        "null_type": "uniform_gain",
    }


def no_field_projection_null(readout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    No field projection null: flatten profiles to mean.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma'

    Returns
    -------
    dict
        Flattened profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()

    # Flatten to mean + small noise
    ab_mean = np.mean(ab)
    gamma_mean = np.mean(gamma)

    ab_shuffled = np.ones_like(ab) * ab_mean + np.random.normal(0, 0.01, size=ab.shape)
    gamma_shuffled = np.ones_like(gamma) * gamma_mean + np.random.normal(
        0, 0.01, size=gamma.shape
    )

    return {
        "alpha_beta_shuffled": ab_shuffled,
        "gamma_shuffled": gamma_shuffled,
        "null_type": "no_field_projection",
    }


def phase_randomized_null(readout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Phase randomized null: randomize phase relationships.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma'

    Returns
    -------
    dict
        Phase-randomized profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()

    # Add random phase offsets
    phase_ab = np.random.uniform(0, 2 * np.pi, size=ab.shape)
    phase_gamma = np.random.uniform(0, 2 * np.pi, size=gamma.shape)

    ab_shuffled = ab * np.cos(phase_ab)
    gamma_shuffled = gamma * np.cos(phase_gamma)

    return {
        "alpha_beta_shuffled": ab_shuffled,
        "gamma_shuffled": gamma_shuffled,
        "null_type": "phase_randomized",
    }


def source_polarity_flip_null(readout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Source polarity flip null: invert profiles.

    Parameters
    ----------
    readout : dict
        Readout dict with 'alpha_beta', 'gamma'

    Returns
    -------
    dict
        Inverted profiles: 'alpha_beta_shuffled', 'gamma_shuffled'
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()

    return {
        "alpha_beta_shuffled": -ab,
        "gamma_shuffled": -gamma,
        "null_type": "source_polarity_flip",
    }


def compute_synchrony_metric(
    spikes: np.ndarray,
    bin_ms: float = 5.0,
    dt_ms: float = 0.1,
    method: str = "mean_pairwise_correlation",
) -> float:
    """
    Compute synchrony metric across neurons.

    Parameters
    ----------
    spikes : np.ndarray
        Spike matrix [T, N] (T time steps, N neurons)
    bin_ms : float
        Binning window in ms (default 5.0 ms)
    dt_ms : float
        Simulation dt in ms (default 0.1 ms)
    method : str
        Synchrony method: "mean_pairwise_correlation" or "variance"

    Returns
    -------
    float
        Synchrony metric [0, 1]
    """
    if spikes.shape[0] == 0 or spikes.shape[1] < 2:
        return 0.0

    # Bin spikes
    bin_steps = int(np.ceil(bin_ms / dt_ms))
    n_bins = spikes.shape[0] // bin_steps

    if n_bins < 2:
        return 0.0

    binned = np.zeros((n_bins, spikes.shape[1]))
    for i in range(n_bins):
        start_idx = i * bin_steps
        end_idx = min((i + 1) * bin_steps, spikes.shape[0])
        binned[i, :] = np.sum(spikes[start_idx:end_idx, :], axis=0)

    if method == "mean_pairwise_correlation":
        # Compute pairwise correlations
        correlations = []
        for i in range(spikes.shape[1]):
            for j in range(i + 1, spikes.shape[1]):
                corr = np.corrcoef(binned[:, i], binned[:, j])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)

        if len(correlations) == 0:
            return 0.0

        sync = float(np.mean(correlations))
        return float(np.clip(sync, 0.0, 1.0))

    elif method == "variance":
        # Spike count variance across population
        pop_rate = np.mean(binned, axis=1)
        if np.std(pop_rate) == 0:
            return 0.0

        variance_coeff = np.std(pop_rate) / (np.mean(pop_rate) + 1e-8)
        return float(np.clip(variance_coeff, 0.0, 1.0))

    else:
        raise ValueError(f"Unknown synchrony method: {method}")


def spectrolaminar_objective(
    readout: Dict[str, Any],
    target_alpha_beta: np.ndarray,
    target_gamma: np.ndarray,
    nulls: Optional[List[str]] = None,
    null_n_samples: int = 10,
    synchrony_metric: Optional[str] = None,
    synchrony_spikes: Optional[np.ndarray] = None,
    synchrony_threshold: float = 0.7,
    similarity_metric: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Full spectrolaminar objective with null distributions and synchrony gates.

    Parameters
    ----------
    readout : dict
        Readout dict from spectrolaminar_readout()
    target_alpha_beta : np.ndarray
        Target alpha/beta profile [n_contacts,]
    target_gamma : np.ndarray
        Target gamma profile [n_contacts,]
    nulls : list of str, optional
        Null types to run: ["layer_shuffle", "band_label_shuffle", "uniform_gain",
        "no_field_projection", "phase_randomized", "source_polarity_flip"]
    null_n_samples : int
        Number of null samples per type (default 10)
    synchrony_metric : str, optional
        Synchrony method: "mean_pairwise_correlation" or "variance"
    synchrony_spikes : np.ndarray, optional
        Spike matrix [T, N] for synchrony computation
    synchrony_threshold : float
        Threshold for synchrony rejection (default 0.7)
    similarity_metric : callable, optional
        Custom similarity function

    Returns
    -------
    dict
        Objective report with all required fields
    """
    rejection_reasons = []

    # Check teaching/control source
    if readout.get("metadata", {}).get("teaching_control_source", False):
        rejection_reasons.append("teaching_control_source_detected")

    # Run nulls if specified
    null_distribution_n = 0
    null_normalization_method = None
    S_lam = None

    if nulls and len(nulls) > 0:
        null_functions = {
            "layer_shuffle": layer_shuffle_null,
            "band_label_shuffle": band_label_shuffle_null,
            "uniform_gain": uniform_gain_null,
            "no_field_projection": no_field_projection_null,
            "phase_randomized": phase_randomized_null,
            "source_polarity_flip": source_polarity_flip_null,
        }

        null_scores = []

        for null_type in nulls:
            if null_type not in null_functions:
                continue

            null_func = null_functions[null_type]

            for _ in range(null_n_samples):
                try:
                    null_readout = null_func(readout)

                    # Score null
                    if similarity_metric is None:
                        mse_ab = np.mean(
                            (null_readout["alpha_beta_shuffled"] - target_alpha_beta) ** 2
                        )
                        mse_gamma = np.mean(
                            (null_readout["gamma_shuffled"] - target_gamma) ** 2
                        )
                        mse_total = mse_ab + mse_gamma
                        null_score = 100.0 * np.exp(-3.0 * mse_total)
                    else:
                        null_score = similarity_metric(
                            null_readout, (target_alpha_beta, target_gamma)
                        )

                    if np.isfinite(null_score):
                        null_scores.append(float(null_score))
                except Exception:
                    pass

        if len(null_scores) > 0:
            null_distribution_n = len(null_scores)
            null_mean = np.mean(null_scores)
            null_std = np.std(null_scores)

            # Compute actual profile score
            actual_report = spectrolaminar_profile_score(
                readout, target_alpha_beta, target_gamma, similarity_metric
            )
            actual_score = actual_report["profile_score_percent"]

            # Z-score normalization
            if null_std > 0:
                S_lam = float((actual_score - null_mean) / null_std)
            else:
                S_lam = 0.0

            null_normalization_method = "z_score"

    # Compute synchrony if spikes provided
    synchrony_value = None
    synchrony_rejection = False

    if synchrony_spikes is not None and synchrony_metric is not None:
        try:
            synchrony_value = compute_synchrony_metric(
                synchrony_spikes, method=synchrony_metric
            )

            if synchrony_value > synchrony_threshold:
                synchrony_rejection = True
                rejection_reasons.append(
                    f"synchrony_exceeded_threshold_{synchrony_metric}_{synchrony_value:.2f}>"
                    f"{synchrony_threshold}"
                )
        except Exception:
            synchrony_value = None

    # Determine score type and profile score
    if S_lam is not None and null_distribution_n > 0:
        score_type = "null_normalized_similarity"
        profile_score = None
    else:
        score_type = "profile_score_no_null"
        actual_report = spectrolaminar_profile_score(
            readout, target_alpha_beta, target_gamma, similarity_metric
        )
        profile_score = actual_report["profile_score_percent"]
        S_lam = None

    # Build final report
    report = {
        "objective_kind": "spectrolaminar_profile",
        "score_type": score_type,
        "profile_score_percent": profile_score,
        "motif_gate_percent": None,
        "S_lam": S_lam,
        "nulls_run": len(nulls) > 0 if nulls else False,
        "null_distribution_n": null_distribution_n,
        "null_normalization_method": null_normalization_method,
        "synchrony_checked": synchrony_spikes is not None,
        "synchrony_metric": synchrony_metric,
        "synchrony_value": synchrony_value,
        "synchrony_threshold": synchrony_threshold,
        "synchrony_rejection": synchrony_rejection,
        "rejection_reasons": rejection_reasons,
        "uses_teaching_control_source": readout.get("metadata", {}).get(
            "teaching_control_source", False
        ),
        "default_evidence_path": readout.get("metadata", {}).get(
            "default_evidence_path", True
        ),
        "physical_amplitude_claim_allowed": False,
        "truth_mode": "truth_safe_unverified",
        "bands": {
            "alpha_beta": [8.0, 25.0],
            "gamma": [40.0, 150.0],
        },
    }

    return report


def spectrolaminar_objective_factory(
    target_alpha_beta: np.ndarray,
    target_gamma: np.ndarray,
    nulls: Optional[List[str]] = None,
    null_n_samples: int = 10,
    synchrony_metric: Optional[str] = None,
    synchrony_threshold: float = 0.7,
) -> callable:
    """
    Factory for spectrolaminar objective (jaxfne pattern).

    Parameters
    ----------
    target_alpha_beta : np.ndarray
        Target alpha/beta profile
    target_gamma : np.ndarray
        Target gamma profile
    nulls : list of str, optional
        Null types to run
    null_n_samples : int
        Null samples per type
    synchrony_metric : str, optional
        Synchrony method
    synchrony_threshold : float
        Synchrony threshold

    Returns
    -------
    callable
        Objective function that takes (readout, spikes) and returns report
    """

    def objective_fn(readout, synchrony_spikes=None):
        return spectrolaminar_objective(
            readout,
            target_alpha_beta,
            target_gamma,
            nulls=nulls,
            null_n_samples=null_n_samples,
            synchrony_metric=synchrony_metric,
            synchrony_spikes=synchrony_spikes,
            synchrony_threshold=synchrony_threshold,
        )

    return objective_fn
