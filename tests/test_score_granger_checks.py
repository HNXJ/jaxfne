"""The granger scorer writes one {error, tolerance, pass, reference} row per check."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_granger_checks.py"
SCORE = REPO / "artifacts" / "etudes" / "core_checks" / "probes" / "granger" / "score.json"
README = SCORE.parent / "README.md"
PINNED_TEST = REPO / "tests" / "test_granger_spectral.py"

CHECK_IDS = (
    "spectral_gc_var1",
    "reverse_direction_near_zero",
)


def load_score():
    return json.loads(SCORE.read_text(encoding="utf-8"))


def load_pinned_test_module():
    spec = importlib.util.spec_from_file_location("pinned_granger_fixture", PINNED_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scorer_script_exists_with_repo_root_guard():
    text = SCORER.read_text(encoding="utf-8")
    assert "sys.path.insert" in text and "site-packages" in text, (
        "the scorer must carry the P-025 repo-root guard like scripts/score_field_invariant_checks.py"
    )


def test_score_file_holds_two_checks_in_score_form():
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
    assert not bad, f"failing granger checks: {bad}"


def test_score_readme_states_references_seed_and_rerun():
    text = README.read_text(encoding="utf-8")
    for cid in CHECK_IDS:
        assert cid in text, f"README names {cid}"
    assert "score_granger_checks.py --check" in text
    assert "seed 11" in text, "the stochastic fixture records its seed"
    assert "time-domain" in text, "the README states why Geweke's time-domain identity is not scored"


def test_recomputed_pass_is_error_within_tolerance():
    from scripts import score_granger_checks as scorer

    rows = scorer.compute_all()["checks"]
    assert set(rows) == set(CHECK_IDS), f"checks are {sorted(rows)}"
    for cid, row in rows.items():
        assert row["pass"] == (row["error"] <= row["tolerance"]), cid
        assert row["tolerance"] > 0, cid


@pytest.mark.parametrize("cid", CHECK_IDS)
def test_tightened_tolerance_flips_a_pass_to_false(monkeypatch, cid):
    from scripts import score_granger_checks as scorer

    fn = dict(scorer.CHECKS)[cid]
    row = fn()
    assert row["error"] > 0, "fixture must carry a non-zero residual"
    assert row["pass"], f"{cid} must pass before tightening"
    monkeypatch.setitem(scorer.TOLERANCES, cid, row["error"] / 2.0)
    assert fn()["pass"] is False, f"{cid} must fail once its tolerance sits below its error"


def test_recompute_matches_within_rtol_1e_9():
    from scripts import score_granger_checks as scorer

    assert scorer.compare(load_score(), scorer.compute_all(), rtol=1e-9) == []


def test_scorer_fixture_is_the_pinned_test_fixture():
    """The scorer's drive equals the test's own _drive (x_to_y) at a short length: the copy is the pinned one."""
    from scripts import score_granger_checks as scorer

    pinned = load_pinned_test_module()
    x_t, y_t = pinned._drive(n=500, seed=11, coupling=0.6, direction="x_to_y")
    x_s, y_s = scorer.drive_x_to_y(500, 11, 0.6)
    np.testing.assert_array_equal(x_s, x_t)
    np.testing.assert_array_equal(y_s, y_t)


def test_pinned_fixture_drives_x_to_y():
    """The forward spectrum dominates the reverse, so the reverse check is not vacuous on a dead fixture."""
    from scripts import score_granger_checks as scorer

    spectra = scorer._pinned_spectra()
    assert bool((spectra["gc_xy"] > spectra["gc_yx"]).all()), "x -> y must dominate on the pinned fixture"


def test_analytic_reference_is_zero_without_coupling():
    """With no coupling H_yx = 0, so the Geweke ratio is ln(1) = 0 at every frequency."""
    from scripts import score_granger_checks as scorer

    np.testing.assert_allclose(scorer.analytic_gc_xy(scorer.FREQS, scorer.DT_MS, 0.0), 0.0, atol=1e-15)
