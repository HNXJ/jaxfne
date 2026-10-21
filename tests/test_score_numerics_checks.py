"""The numerics scorer writes one {error, tolerance, pass, reference} row per core-method check."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_numerics_checks.py"
SCORE = REPO / "artifacts" / "etudes" / "core_checks" / "numerics" / "score.json"
README = SCORE.parent / "README.md"

CHECK_IDS = (
    "euler_scan_exponential_decay",
    "euler_solve_ode_decay",
    "source_scale_gradient_fd",
)


def load_score():
    return json.loads(SCORE.read_text(encoding="utf-8"))


def test_scorer_script_exists_with_repo_root_guard():
    text = SCORER.read_text(encoding="utf-8")
    assert "sys.path.insert" in text and "site-packages" in text, (
        "the scorer must carry the P-025 repo-root guard like scripts/benchmark_jaxfne.py"
    )


def test_score_file_holds_three_checks_in_score_form():
    doc = load_score()
    assert set(doc["checks"]) == set(CHECK_IDS), f"checks are {sorted(doc['checks'])}"
    assert isinstance(doc["pass"], bool), "top-level pass is a boolean"
    for cid in CHECK_IDS:
        row = doc["checks"][cid]
        assert {"error", "tolerance", "pass", "reference"} <= set(row), cid
        assert row["reference"], f"{cid} names its independent reference"
        assert isinstance(row["pass"], bool), cid
        assert row["error"] <= row["tolerance"], f"{cid}: error above tolerance"
    assert doc["checks"]["source_scale_gradient_fd"].get("seed") == 3, \
        "the gradient check records its simulation seed"


def test_every_computed_check_passes():
    doc = load_score()
    bad = sorted(c for c, r in doc["checks"].items() if not r["pass"])
    assert not bad, f"failing numerics checks: {bad}"


def test_score_readme_states_references_and_rerun():
    text = README.read_text(encoding="utf-8")
    for cid in CHECK_IDS:
        assert cid in text, f"README names {cid}"
    assert "score_numerics_checks.py --check" in text


def test_recomputed_pass_is_error_within_tolerance():
    from scripts import score_numerics_checks as scorer

    rows = scorer.compute_all()["checks"]
    assert set(rows) == set(CHECK_IDS)
    for cid, row in rows.items():
        assert row["pass"] == (row["error"] <= row["tolerance"]), cid
        assert row["tolerance"] > 0, cid


def test_tightened_tolerance_flips_a_pass_to_false(monkeypatch):
    from scripts import score_numerics_checks as scorer

    row = scorer.check_euler_scan_exponential_decay()
    assert row["error"] > 0, "fixture must carry a non-zero residual"
    assert row["pass"], "the pinned check must pass before tightening"
    monkeypatch.setitem(scorer.TOLERANCES, "euler_scan_exponential_decay",
                        tuple(v * 1e-6 for v in scorer.TOLERANCES["euler_scan_exponential_decay"]))
    assert scorer.check_euler_scan_exponential_decay()["pass"] is False


def _doc(**rows):
    return {"checks": rows, "pass": True, "skipped": []}


def _row(error, tolerance=1.0):
    return {"error": error, "tolerance": tolerance, "pass": error <= tolerance, "reference": "r"}


def test_compare_accepts_rounding_residuals_and_refuses_real_changes():
    from scripts import score_numerics_checks as scorer

    # Both residuals below a tenth of the tolerance: a platform difference, no verdict at stake.
    assert scorer.compare(_doc(a=_row(4e-6, 2e-4)), _doc(a=_row(3e-6, 2e-4))) == []
    # A residual near the tolerance must reproduce.
    assert scorer.compare(_doc(a=_row(0.5)), _doc(a=_row(0.6))) != []
    # The tolerance itself always reproduces.
    assert scorer.compare(_doc(a=_row(1e-6, 1.0)), _doc(a=_row(1e-6, 1.1))) != []


def test_recompute_matches_within_rtol_1e_9():
    from scripts import score_numerics_checks as scorer

    assert scorer.compare(load_score(), scorer.compute_all(), rtol=1e-9) == []
