"""Broad-lane notebook smoke: four release notebooks execute cleanly in fresh kernels.

The broad gate (``-m "not slow"``) ran no notebook, so 0.5.5 refusals broke three
release notebooks unseen (P-021). The full execution sweep is slow-marked and
runs in the release lane (``test_notebook_execution_suite.py``); this is its
fast subset, about 40 s, chosen to span the surfaces the refusals touch: a
single neuron, a two-neuron E/I pair, a small recurrent E/I network and an
evoked-L4 ``Paradigm``. One test, run in sequence, so xdist does not start
several kernels at once (kernel and zmq stability on Windows).

Skipped, with the reason in ``-rs``, when nbclient is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nbclient")
pytest.importorskip("nbformat")

from _notebook_exec_helpers import execute_notebook_via_nbclient, format_cell_errors

TUTORIALS_DIR = Path(__file__).parent.parent / "artifacts" / "tutorials"

# Subset of RELEASE_FACING_NOTEBOOKS in test_notebook_execution_suite.py.
FAST_NOTEBOOKS = [
    "jaxfne_v031_single_neuron.ipynb",
    "jaxfne_v033_two_neuron_ei.ipynb",
    "jaxfne_v035_small_recurrent_ei.ipynb",
    "jaxfne_suite_no_2_evoked_l4_drive.ipynb",
]


def test_fast_release_notebooks_execute_cleanly(tmp_path):
    failures = {}
    for name in FAST_NOTEBOOKS:
        path = TUTORIALS_DIR / name
        assert path.exists(), f"notebook missing: {path}"
        try:
            errors = execute_notebook_via_nbclient(path, tmp_path, timeout=300)
        except Exception as exc:  # nbclient raises CellExecutionError on the first failing cell
            errors = [(-1, type(exc).__name__, str(exc)[-600:])]
        if errors:
            failures[name] = format_cell_errors(errors)
    assert not failures, "\n\n".join(f"{n}:\n{m}" for n, m in failures.items())
