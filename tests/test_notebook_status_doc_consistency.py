"""Cheap, fast-lane guard against drift between the notebook-execution test
suite and artifacts/tutorials/NOTEBOOK_STATUS.md.

Deliberately has no nbclient/nbformat/slow/notebook markers -- this runs on
every push, unlike the real notebook executions in test_notebook_execution_suite.py
/ test_suite_no1_notebook_execution.py / test_suite_no4_notebook_execution.py.
"""

from __future__ import annotations

from pathlib import Path

from test_notebook_execution_suite import RELEASE_FACING_NOTEBOOKS

TUTORIALS_DIR = Path(__file__).parent.parent / "artifacts" / "tutorials"

EXECUTED_ELSEWHERE = {
    "jaxfne_suite_no_1_computational_biophysics.ipynb",
    "jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb",
}


def test_release_facing_notebook_list_matches_status_doc():
    """Every execution-tested notebook must be named in NOTEBOOK_STATUS.md,
    and every notebook on disk must be classified there (release-facing,
    archived, or template) so an unclassified notebook can't silently
    appear."""
    status_text = (TUTORIALS_DIR / "NOTEBOOK_STATUS.md").read_text(encoding="utf-8")

    execution_tested = {
        nb.rsplit("/", 1)[-1] for nb in RELEASE_FACING_NOTEBOOKS
    } | EXECUTED_ELSEWHERE

    for basename in execution_tested:
        assert basename in status_text, (
            f"{basename} is execution-tested but missing from NOTEBOOK_STATUS.md"
        )

    for nb_path in TUTORIALS_DIR.rglob("*.ipynb"):
        if "outputs" in nb_path.parts or "executed" in nb_path.parts:
            continue
        if nb_path.relative_to(TUTORIALS_DIR).as_posix().startswith("templates/"):
            continue
        basename = nb_path.name
        if basename not in execution_tested:
            # Archived notebooks are intentionally excluded from execution
            # coverage; just confirm NOTEBOOK_STATUS.md still lists them
            # somewhere so an unclassified notebook can't silently appear.
            assert basename in status_text, (
                f"{basename} is neither execution-tested nor listed in "
                f"NOTEBOOK_STATUS.md -- classify it as release-facing, "
                f"archived, or template"
            )
