"""
Spectrolaminar objective and null distribution functions.

Docs: ``docs/api/objectives.md`` (https://jaxfne.readthedocs.io/en/latest/api/objectives/) —
update that page when this module's public API changes.

This module implements spectrolaminar profile scoring with null distributions
and synchrony gates for multi-area neural circuits.

Scope: Scoring objectives for relative spectrolaminar profiles. Evidence: finite
output arrays, valid JSON serialization. Interpretation: simulated spectrolaminar
readouts (alpha/beta and gamma profiles) matched to targets under null distributions
and synchrony constraints. Physical amplitude claims NOT allowed.

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
        "physical_amplitude_calibrated": False,
        "bands": {
            "alpha_beta": [8.0, 25.0],
            "gamma": [40.0, 150.0],
        },
    }

    return report



def _make_null(readout: Dict[str, Any], mutate_fn: callable, null_type: str) -> Dict[str, np.ndarray]:
    """Higher-order null distribution generator.
    
    Applies a mutation function to copies of the readout profiles.
    """
    ab = readout["alpha_beta"].copy()
    gamma = readout["gamma"].copy()
    ab_shuffled, gamma_shuffled = mutate_fn(ab, gamma, readout)
    return {
        "alpha_beta_shuffled": ab_shuffled,
        "gamma_shuffled": gamma_shuffled,
        "null_type": null_type,
    }


def layer_shuffle_null(
    readout: Dict[str, Any], *, rng: "np.random.Generator | None" = None
) -> Dict[str, np.ndarray]:
    """Layer shuffle null: permute profiles across L4-relative positions.

    Pass an explicit ``rng`` (a ``np.random.Generator``) for reproducible
    nulls. When ``rng is None`` an unseeded generator is used, which is
    nondeterministic by design (preserves legacy behavior).
    """
    gen = rng if rng is not None else np.random.default_rng()

    def _mutate(ab, gamma, r):
        perm = gen.permutation(len(ab))
        return ab[perm], gamma[perm]
    return _make_null(readout, _mutate, "layer_shuffle")


def band_label_shuffle_null(
    readout: Dict[str, Any], *, rng: "np.random.Generator | None" = None
) -> Dict[str, np.ndarray]:
    """Band label shuffle null: swap alpha/beta and gamma profiles.

    Deterministic; ``rng`` is accepted for a uniform dispatcher signature
    and intentionally ignored.
    """
    del rng

    def _mutate(ab, gamma, r):
        return gamma.copy(), ab.copy()
    return _make_null(readout, _mutate, "band_label_shuffle")


def uniform_gain_null(
    readout: Dict[str, Any],
    gain_min: float = 0.1,
    *,
    rng: "np.random.Generator | None" = None,
) -> Dict[str, np.ndarray]:
    """Uniform gain null: scale profiles by random uniform gain.

    Pass an explicit ``rng`` for reproducible nulls; ``None`` is unseeded.
    """
    gen = rng if rng is not None else np.random.default_rng()

    def _mutate(ab, gamma, r):
        gain = gen.uniform(gain_min, 1.0)
        return ab * gain, gamma * gain
    return _make_null(readout, _mutate, "uniform_gain")


def no_field_projection_null(
    readout: Dict[str, Any], *, rng: "np.random.Generator | None" = None
) -> Dict[str, np.ndarray]:
    """No field projection null: flatten profiles to mean.

    Pass an explicit ``rng`` for reproducible nulls; ``None`` is unseeded.
    """
    gen = rng if rng is not None else np.random.default_rng()

    def _mutate(ab, gamma, r):
        return (np.ones_like(ab) * np.mean(ab) + gen.normal(0, 0.01, size=ab.shape),
                np.ones_like(gamma) * np.mean(gamma) + gen.normal(0, 0.01, size=gamma.shape))
    return _make_null(readout, _mutate, "no_field_projection")


def phase_randomized_null(
    readout: Dict[str, Any], *, rng: "np.random.Generator | None" = None
) -> Dict[str, np.ndarray]:
    """Phase randomized null: randomize phase relationships.

    Pass an explicit ``rng`` for reproducible nulls; ``None`` is unseeded.
    """
    gen = rng if rng is not None else np.random.default_rng()

    def _mutate(ab, gamma, r):
        return ab * np.cos(gen.uniform(0, 2*np.pi, size=ab.shape)), gamma * np.cos(gen.uniform(0, 2*np.pi, size=gamma.shape))
    return _make_null(readout, _mutate, "phase_randomized")


def source_polarity_flip_null(
    readout: Dict[str, Any], *, rng: "np.random.Generator | None" = None
) -> Dict[str, np.ndarray]:
    """Source polarity flip null: invert profiles.

    Deterministic; ``rng`` is accepted for a uniform dispatcher signature
    and intentionally ignored.
    """
    del rng

    def _mutate(ab, gamma, r):
        return -ab, -gamma
    return _make_null(readout, _mutate, "source_polarity_flip")


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
        # Vectorized mean pairwise correlation: one correlation-matrix solve
        # instead of an O(N^2) Python loop of np.corrcoef per neuron pair.
        # Identical result (mean of finite off-diagonal upper-triangle entries;
        # constant/silent neurons yield NaN and are excluded, as before).
        corr_matrix = np.corrcoef(binned.T)
        iu = np.triu_indices(corr_matrix.shape[0], k=1)
        vals = corr_matrix[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        sync = float(np.mean(vals))
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
    null_seed: Optional[int] = None,
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
    null_seed : int, optional
        Seed for the null-distribution generator. When provided, the null
        sample sequence is reproducible across runs while still drawing
        distinct samples within a single run. When ``None`` (default), an
        unseeded generator is used — nondeterministic by design, preserving
        legacy behavior.

        Note: a single generator is shared across all null types in the order
        given by ``nulls``, so reproducibility holds for a fixed ``nulls`` list
        and ordering. Adding, removing, or reordering entries shifts the RNG
        state seen by later null types. For per-type isolation, call this
        function once per null type with the same ``null_seed``.

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

        # One generator threaded through every null draw: distinct samples
        # within a run, reproducible across runs when null_seed is set.
        null_gen = np.random.default_rng(null_seed)

        for null_type in nulls:
            if null_type not in null_functions:
                continue

            null_func = null_functions[null_type]

            for _ in range(null_n_samples):
                try:
                    null_readout = null_func(readout, rng=null_gen)

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
        "physical_amplitude_calibrated": False,
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
    null_seed: Optional[int] = None,
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
    null_seed : int, optional
        Seed forwarded to ``spectrolaminar_objective`` for reproducible null
        distributions. ``None`` (default) preserves the legacy unseeded,
        nondeterministic behavior.

    Returns
    -------
    callable
        Objective function that takes (readout, spikes) and returns report
    """

    def objective_fn(readout, synchrony_spikes=None):
        """Documented public function `objective_fn`."""
        return spectrolaminar_objective(
            readout,
            target_alpha_beta,
            target_gamma,
            nulls=nulls,
            null_n_samples=null_n_samples,
            synchrony_metric=synchrony_metric,
            synchrony_spikes=synchrony_spikes,
            synchrony_threshold=synchrony_threshold,
            null_seed=null_seed,
        )

    return objective_fn
