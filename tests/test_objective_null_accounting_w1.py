"""W1: an invalid null draw and a broken implementation are different events.

Previously every exception inside the null loop was swallowed by `except Exception:
pass`, so a configuration or programming error silently shrank the null sample count
and the surviving nulls were used as if the requested distribution had been drawn.
A degenerate null distribution additionally reported a fabricated `S_lam = 0.0`.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxfne.objectives import (
    _NULL_MIN_SAMPLES_FOR_SIGMA,
    null_samples_for_sigma_precision,
    spectrolaminar_objective,
)

TARGET_AB = np.array([0.2, 0.3, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02])
TARGET_GAMMA = np.array([0.05, 0.03, 0.08, 0.15, 0.2, 0.18, 0.12, 0.08])


def _readout():
    return {"alpha_beta": TARGET_AB.copy(), "gamma": TARGET_GAMMA.copy()}


def _run(**kwargs):
    return spectrolaminar_objective(
        _readout(), TARGET_AB, TARGET_GAMMA,
        nulls=["layer_shuffle"], null_n_samples=12, null_seed=0, **kwargs,
    )


def test_report_exposes_requested_accepted_and_rejected_counts():
    report = _run()
    assert report["null_samples_requested"] == 12
    assert report["null_samples_accepted"] == report["null_distribution_n"]
    assert (
        report["null_samples_rejected"]
        == report["null_samples_requested"] - report["null_samples_accepted"]
    )
    assert isinstance(report["null_rejection_classes"], dict)


def test_unexpected_exception_propagates_instead_of_shrinking_the_null_count():
    """A broken similarity metric is a defect, not an invalid null draw."""

    def broken_metric(null_readout, targets):
        raise KeyError("similarity_metric is misconfigured")

    with pytest.raises(KeyError, match="misconfigured"):
        _run(similarity_metric=broken_metric)


def test_expected_invalid_draw_is_recorded_as_data_not_silently_dropped():
    """A ValueError from scoring is an invalid sample: counted and classified."""
    calls = {"n": 0}

    def flaky_metric(null_readout, targets):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ValueError("degenerate null draw")
        return 50.0 + calls["n"]

    report = _run(similarity_metric=flaky_metric)
    assert report["null_samples_rejected"] == 6
    assert report["null_rejection_classes"] == {"expected_exception:ValueError": 6}
    assert "degenerate null draw" in report["null_rejection_examples"][
        "expected_exception:ValueError"
    ]


def test_non_finite_scores_are_classified_separately_from_exceptions():
    def half_nan(null_readout, targets):
        half_nan.n = getattr(half_nan, "n", 0) + 1
        return np.nan if half_nan.n % 2 == 0 else 40.0 + half_nan.n

    report = _run(similarity_metric=half_nan)
    assert report["null_rejection_classes"].get("non_finite_score") == 6


def test_degenerate_null_distribution_raises_instead_of_fabricating_zero():
    """sigma == 0 makes the z-score undefined; it previously reported S_lam = 0.0."""
    with pytest.raises(ValueError, match="degenerate"):
        _run(similarity_metric=lambda null_readout, targets: 42.0)


def test_caller_declared_minimum_valid_samples_is_enforced():
    with pytest.raises(ValueError, match="caller-declared minimum"):
        _run(null_min_valid_samples=1000)


def test_accepted_below_requested_is_legitimate_when_above_the_floor():
    """Rejection can be legitimate: no exception merely because accepted < requested."""

    def occasionally_invalid(null_readout, targets):
        occasionally_invalid.n = getattr(occasionally_invalid, "n", 0) + 1
        if occasionally_invalid.n % 4 == 0:
            raise ValueError("invalid draw")
        return 30.0 + occasionally_invalid.n

    report = _run(similarity_metric=occasionally_invalid)
    assert report["null_samples_accepted"] == 9 < report["null_samples_requested"]
    assert report["score_type"] == "null_normalized_similarity"


# ── the advisory precision figure ────────────────────────────────────────────
def test_sigma_precision_requirement_matches_its_closed_form():
    # RSE(sigma_hat) ~= 1/sqrt(2(n-1))  ->  n >= 1 + 1/(2 rse^2)
    assert null_samples_for_sigma_precision(0.5) == 3
    assert null_samples_for_sigma_precision(0.25) == 9
    assert null_samples_for_sigma_precision(0.1) == 51
    assert null_samples_for_sigma_precision(0.99) >= _NULL_MIN_SAMPLES_FOR_SIGMA


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_sigma_precision_rejects_an_out_of_range_target(bad):
    with pytest.raises(ValueError, match="relative_standard_error"):
        null_samples_for_sigma_precision(bad)


def test_report_carries_the_advisory_precision_figure():
    assert _run()["null_samples_for_sigma_rse_25pct"] == 9
