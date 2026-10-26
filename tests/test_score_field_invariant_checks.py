"""The field-invariant scorer writes one {error, tolerance, pass, reference} row per check."""

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_field_invariant_checks.py"
SCORE = REPO / "artifacts" / "etudes" / "core_checks" / "fields" / "score.json"
README = SCORE.parent / "README.md"

CHECK_IDS = (
    "operator_superposition",
    "csd_flat_potential_zero",
    "csd_quadratic_potential",
    "cable_filter_transfer",
    "poisson_1d_flux_magnitude",
    "poisson_1d_flux_span",
)


def load_score():
    return json.loads(SCORE.read_text(encoding="utf-8"))


def test_scorer_script_exists_with_repo_root_guard():
    text = SCORER.read_text(encoding="utf-8")
    assert "sys.path.insert" in text and "site-packages" in text, (
        "the scorer must carry the P-025 repo-root guard like scripts/score_head_forward_checks.py"
    )


def test_score_file_holds_six_checks_in_score_form():
    doc = load_score()
    assert set(doc["checks"]) == set(CHECK_IDS), f"checks are {sorted(doc['checks'])}"
    assert isinstance(doc["pass"], bool), "top-level pass is a boolean"
    for cid in CHECK_IDS:
        row = doc["checks"][cid]
        assert {"error", "tolerance", "pass", "reference"} <= set(row), cid
        assert row["reference"], f"{cid} names its independent reference"
        assert isinstance(row["pass"], bool), cid
        assert row["error"] <= row["tolerance"], f"{cid}: error above tolerance"


def test_every_computed_check_passes():
    doc = load_score()
    bad = sorted(c for c, r in doc["checks"].items() if not r["pass"])
    assert not bad, f"failing field checks: {bad}"


def test_score_readme_states_references_seed_and_rerun():
    text = README.read_text(encoding="utf-8")
    for cid in CHECK_IDS:
        assert cid in text, f"README names {cid}"
    assert "score_field_invariant_checks.py --check" in text
    assert "seed 100" in text, "the stochastic superposition fixture records its seed"


def test_recomputed_pass_is_error_within_tolerance():
    from scripts import score_field_invariant_checks as scorer

    rows = scorer.compute_all()["checks"]
    assert set(rows) == set(CHECK_IDS), f"checks are {sorted(rows)}"
    for cid, row in rows.items():
        assert row["pass"] == (row["error"] <= row["tolerance"]), cid
        assert row["tolerance"] > 0, cid


def test_tightened_tolerance_flips_a_pass_to_false(monkeypatch):
    from scripts import score_field_invariant_checks as scorer

    row = scorer.check_cable_filter_transfer()
    assert row["error"] > 0, "fixture must carry a non-zero residual"
    assert row["pass"], "the cable-filter check must pass before tightening"
    monkeypatch.setitem(scorer.TOLERANCES, "cable_filter_transfer",
                        tuple(v * 1e-3 for v in scorer.TOLERANCES["cable_filter_transfer"]))
    assert scorer.check_cable_filter_transfer()["pass"] is False


def test_recompute_matches_within_rtol_1e_9():
    from scripts import score_field_invariant_checks as scorer

    assert scorer.compare(load_score(), scorer.compute_all(), rtol=1e-9) == []


def test_cable_reference_has_unit_dc_gain():
    """The one-pole cascade has unit DC gain: alpha / (1 - (1 - alpha)) = 1 for any order."""
    from scripts import score_field_invariant_checks as scorer

    assert scorer.discrete_cascade_gain(0.0, 0.02, 0.5, order=2) == pytest.approx(1.0)


def test_cable_reference_matches_the_complex_transfer_on_the_unit_circle():
    """The closed-form |H| equals |alpha / (1 - (1 - alpha) e^{-j 2 pi f dt})|**order evaluated directly."""
    from scripts import score_field_invariant_checks as scorer

    tau, dt_ms = 0.02, 0.5
    alpha = 1.0 - np.exp(-(dt_ms / 1000.0) / tau)
    for f in (5.0, 20.0, 100.0):
        z = np.exp(-1j * 2.0 * np.pi * f * dt_ms / 1000.0)
        direct = abs(alpha / (1.0 - (1.0 - alpha) * z)) ** 2
        assert scorer.discrete_cascade_gain(f, tau, dt_ms, order=2) == pytest.approx(direct, rel=1e-12)


def test_csd_quadratic_reference_is_minus_second_derivative():
    """csd_tensor is -(second difference) with no conductivity: a quadratic a z^2 gives -2a."""
    from scripts import score_field_invariant_checks as scorer

    assert scorer.csd_quadratic_reference(2.5) == pytest.approx(-5.0)
    assert scorer.csd_quadratic_reference(-0.5) == pytest.approx(1.0)
