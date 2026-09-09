"""W3 adversarial tests for consequential broad exception handlers.

Negative tests demonstrate defects fixed in 0.4.22 W3 closure. Each test would
have failed (or masked failure) on the prior implementation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from jaxfne.objectives import spectrolaminar_objective
from jaxfne.validation import validate_field_arrays_finite


def _minimal_readout():
    return {"alpha_beta": np.ones(5), "gamma": np.ones(5), "metadata": {}}


def test_synchrony_implementation_error_propagates_not_false_pass():
    """Prior: broad except swallowed RuntimeError -> synchrony_rejection stayed False."""
    readout = _minimal_readout()
    spikes = np.ones((100, 10))

    with patch(
        "jaxfne.objectives.compute_synchrony_metric",
        side_effect=RuntimeError("implementation bug"),
    ):
        with pytest.raises(RuntimeError, match="implementation bug"):
            spectrolaminar_objective(
                readout,
                np.ones(5),
                np.ones(5),
                synchrony_spikes=spikes,
                synchrony_metric="mean_pairwise_correlation",
                synchrony_threshold=0.1,
            )


def test_synchrony_unknown_method_propagates_value_error():
    """Invalid synchrony method must fail, not appear as an unchecked gate."""
    readout = _minimal_readout()
    spikes = np.ones((100, 10))

    with pytest.raises(ValueError, match="Unknown synchrony method"):
        spectrolaminar_objective(
            readout,
            np.ones(5),
            np.ones(5),
            synchrony_spikes=spikes,
            synchrony_metric="not_a_real_method",
            synchrony_threshold=0.5,
        )


def test_validate_field_arrays_conversion_error_fails_not_vacuous_pass():
    """Prior: conversion error left phi_e_finite=None and all_finite=True vacuously."""

    class _Unconvertible:
        def tolist(self):
            raise RuntimeError("cannot convert")

    result = validate_field_arrays_finite(phi_e=_Unconvertible())

    assert result["phi_e_finite"] is False
    assert result["all_finite"] is False
    assert result["evidence"] is not None
    assert any("phi_e validation error" in item for item in result["evidence"])


def test_synchrony_degenerate_inputs_still_return_zero_without_exception():
    """Degenerate statistics remain explicit 0.0 — not confused with implementation failure."""
    from jaxfne.objectives import compute_synchrony_metric

    # Single neuron: defined degenerate path
    assert compute_synchrony_metric(np.ones((10, 1))) == 0.0
