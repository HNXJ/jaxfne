"""The head-forward scorer writes one {error, tolerance, pass, reference} row per analytic check."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_head_forward_checks.py"
SCORE = REPO / "artifacts" / "etudes" / "core_checks" / "head_forward" / "score.json"
README = SCORE.parent / "README.md"

CHECK_IDS = (
    "meg_radial",
    "meg_full_vector",
    "meg_radial_silence",
    "eeg_neumann",
    "eeg_mne",
    "eeg_scaling",
    "eeg_reference",
)


def load_score():
    return json.loads(SCORE.read_text(encoding="utf-8"))


def test_scorer_script_exists_with_repo_root_guard():
    text = SCORER.read_text(encoding="utf-8")
    assert "sys.path.insert" in text and "site-packages" in text, (
        "the scorer must carry the P-025 repo-root guard like scripts/benchmark_jaxfne.py"
    )


def test_score_file_holds_seven_checks_in_score_form():
    doc = load_score()
    assert set(doc["checks"]) == set(CHECK_IDS), f"checks are {sorted(doc['checks'])}"
    assert isinstance(doc["pass"], bool), "top-level pass is a boolean"
    for cid in CHECK_IDS:
        row = doc["checks"][cid]
        assert {"error", "tolerance", "pass", "reference"} <= set(row), cid
        assert row["reference"], f"{cid} names its independent reference"
        if "skipped" in row:
            assert row["error"] is None and row["tolerance"] is None and row["pass"] is None, cid
            assert row["skipped"], f"{cid} states why it skipped"
        else:
            assert isinstance(row["pass"], bool), cid
            assert row["error"] <= row["tolerance"], f"{cid}: error above tolerance"


def test_every_computed_check_passes():
    doc = load_score()
    live = {c: r for c, r in doc["checks"].items() if "skipped" not in r}
    assert live, "at least one check must compute"
    bad = sorted(c for c, r in live.items() if not r["pass"])
    assert not bad, f"failing analytic checks: {bad}"


def test_score_readme_states_references_and_rerun():
    text = README.read_text(encoding="utf-8")
    for cid in CHECK_IDS:
        assert cid in text, f"README names {cid}"
    assert "score_head_forward_checks.py --check" in text


def test_recomputed_pass_is_error_within_tolerance():
    from scripts import score_head_forward_checks as scorer

    rows = scorer.compute_all()["checks"]
    live = {c: r for c, r in rows.items() if "skipped" not in r}
    assert live, "fixture must compute at least one check"
    for cid, row in live.items():
        assert row["pass"] == (row["error"] <= row["tolerance"]), cid
        assert row["tolerance"] > 0, cid


def _doc(**rows):
    return {"checks": rows, "pass": True, "skipped": []}


def _row(error, tolerance=1.0):
    return {"error": error, "tolerance": tolerance, "pass": error <= tolerance, "reference": "r"}


def test_compare_accepts_rounding_residuals_and_refuses_real_changes():
    from scripts import score_head_forward_checks as scorer

    # Both residuals below a tenth of the tolerance: a platform difference, no verdict at stake.
    assert scorer.compare(_doc(a=_row(4e-6, 2e-4)), _doc(a=_row(3e-6, 2e-4))) == []
    # A residual near the tolerance must reproduce.
    assert scorer.compare(_doc(a=_row(0.5)), _doc(a=_row(0.6))) != []
    # The tolerance itself always reproduces.
    assert scorer.compare(_doc(a=_row(1e-6, 1.0)), _doc(a=_row(1e-6, 1.1))) != []


def test_compare_lets_a_missing_optional_dependency_stand_on_the_stored_row():
    from scripts import score_head_forward_checks as scorer

    skipped = {"error": None, "tolerance": None, "pass": None, "reference": "r", "skipped": "mne not installed"}
    assert scorer.compare(_doc(a=_row(1e-5)), _doc(a=skipped)) == []
    # The reverse, a stored skip that now computes, is a change to record.
    assert scorer.compare(_doc(a=skipped), _doc(a=_row(1e-5))) == ["a: skipped True != False"]


def test_recompute_matches_within_rtol_1e_9():
    from scripts import score_head_forward_checks as scorer

    assert scorer.compare(load_score(), scorer.compute_all(), rtol=1e-9) == []
