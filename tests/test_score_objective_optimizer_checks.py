"""The objective and optimizer scorer writes one {error, tolerance, pass, reference, stage} row per core-method check."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_objective_optimizer_checks.py"
SCORE = REPO / "artifacts" / "etudes" / "core_checks" / "objectives" / "score.json"
README = SCORE.parent / "README.md"

STAGES = {
    "polarity_null_flips_sign": "objective",
    "layer_shuffle_null_permutes": "objective",
    "uniform_gain_null_scales": "objective",
    "synchrony_metric_limits": "objective",
    "null_samples_closed_form": "objective",
    "spectrolaminar_score_known_profile": "objective",
    "box_projection_is_clip": "optimizer",
    "sdr_quadratic_minimum": "optimizer",
    "gsdr_quadratic_minimum": "optimizer",
    "agsdr_quadratic_minimum": "optimizer",
    "gsgd_quadratic_minimum": "optimizer",
}
CHECK_IDS = tuple(STAGES)
DROPPED_IDS = ("phase_null_keeps_power", "optax_adam_quadratic_minimum")


def load_score():
    return json.loads(SCORE.read_text(encoding="utf-8"))


def test_scorer_script_exists_with_repo_root_guard():
    text = SCORER.read_text(encoding="utf-8")
    assert "sys.path.insert" in text and "site-packages" in text, (
        "the scorer must carry the P-025 repo-root guard like scripts/score_head_forward_checks.py"
    )


def test_score_file_holds_eleven_checks_in_score_form():
    doc = load_score()
    assert set(doc["checks"]) == set(CHECK_IDS), f"checks are {sorted(doc['checks'])}"
    assert isinstance(doc["pass"], bool), "top-level pass is a boolean"
    for cid in CHECK_IDS:
        row = doc["checks"][cid]
        assert {"error", "tolerance", "pass", "reference", "stage"} <= set(row), cid
        assert row["reference"], f"{cid} names its independent reference"
        assert row["stage"] == STAGES[cid], f"{cid} stage is {row['stage']!r}"
        assert isinstance(row["pass"], bool), cid
        assert row["error"] <= row["tolerance"], f"{cid}: error above tolerance"


def test_dropped_checks_are_recorded_with_reasons():
    doc = load_score()
    assert set(doc["dropped"]) == set(DROPPED_IDS), f"dropped are {sorted(doc['dropped'])}"
    for cid in DROPPED_IDS:
        assert doc["dropped"][cid], f"{cid} states why it was dropped"
        assert cid not in doc["checks"], f"{cid} is dropped, so it has no row"


def test_every_check_passes():
    doc = load_score()
    bad = sorted(c for c, r in doc["checks"].items() if not r["pass"])
    assert not bad, f"failing checks: {bad}"
    assert doc["pass"] is True


def test_score_readme_states_references_stages_and_rerun():
    text = README.read_text(encoding="utf-8")
    for cid in CHECK_IDS + DROPPED_IDS:
        assert cid in text, f"README names {cid}"
    assert "score_objective_optimizer_checks.py --check" in text
    assert "objective" in text and "optimizer" in text


@pytest.mark.parametrize("cid", CHECK_IDS)
def test_each_check_recomputes_to_pass(cid):
    from scripts import score_objective_optimizer_checks as scorer

    row = dict(scorer.CHECKS)[cid]()
    assert row["stage"] == STAGES[cid], cid
    assert row["tolerance"] > 0, cid
    assert row["error"] <= row["tolerance"], f"{cid}: error {row['error']!r} above {row['tolerance']!r}"
    assert row["pass"] is True, cid


def test_recompute_matches_within_rtol_1e_9():
    from scripts import score_objective_optimizer_checks as scorer

    assert scorer.compare(load_score(), scorer.compute_all(), rtol=1e-9) == []


def test_tightened_tolerance_flips_a_pass_to_false(monkeypatch):
    from scripts import score_objective_optimizer_checks as scorer

    row = scorer.check_gsgd_quadratic_minimum()
    assert row["error"] > 0, "fixture must carry a non-zero residual"
    assert row["pass"], "the pinned check must pass before tightening"
    monkeypatch.setitem(scorer.TOLERANCES, "gsgd_quadratic_minimum",
                        tuple(v * 1e-6 for v in scorer.TOLERANCES["gsgd_quadratic_minimum"]))
    assert scorer.check_gsgd_quadratic_minimum()["pass"] is False
