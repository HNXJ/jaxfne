"""Real nbclient execution coverage for the release-facing tutorial suite.

Suite No. 1 and Suite No. 4 already have dedicated execution tests
(test_suite_no1_notebook_execution.py, test_suite_no4_notebook_execution.py).
The other 26 release-facing notebooks (see tutorials/NOTEBOOK_STATUS.md)
previously had zero execution coverage -- only structural/grammar checks.

Each notebook below is executed end-to-end in a fresh kernel via nbclient;
the test fails if any cell raises. Marked both ``slow`` and ``notebook`` so a
dedicated CI workflow can select them precisely (``pytest -m notebook``)
without slowing down the push-to-push fast lane (``pytest -m "not slow"``).

Requires nbclient/nbformat/jupyter_client (the whole module is skipped if
unavailable) and can take from under a minute to several minutes per
notebook depending on simulation size -- this is why it lives in a separate
nightly/manual CI lane rather than the push-triggered lane.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")

from _notebook_exec_helpers import execute_notebook_via_nbclient, format_cell_errors

TUTORIALS_DIR = Path(__file__).parent.parent / "tutorials"

# Release-facing notebooks not already covered by a dedicated execution test.
# Keep in sync with tutorials/NOTEBOOK_STATUS.md's release-facing list.
RELEASE_FACING_NOTEBOOKS = [
    "jaxfne-sanity-checker-notebook-01.ipynb",
    "jaxfne-sanity-delta-test-hierarchical-global-local-oddball.ipynb",
    "etudes/jaxfne_etude_no_1_base.ipynb",
    "etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb",
    "etudes/jaxfne_etude_no_4_homeostatic_V1_column.ipynb",
    "etudes/jaxfne_etude_no_5_enhanced_biophysical_column.ipynb",
    "etudes/jaxfne_etude_no_6_multi_area_network.ipynb",
    "etudes/jaxfne_etude_no_7_multitrial_spectrolaminar.ipynb",
    "etudes/jaxfne_etude_no_8_continuous_adaptation.ipynb",
    "etudes/jaxfne_etude_no_9_local_oddball.ipynb",
    "etudes/jaxfne_etude_no_10_global_local_oddball.ipynb",
    "etudes/jaxfne_etude_no_11_omission_local.ipynb",
    "etudes/jaxfne_etude_no_12_omission_global_coop.ipynb",
    "etudes/jaxfne_etude_tcm_v1_6pop.ipynb",
    "jaxfne_suite_no_2_evoked_l4_drive.ipynb",
    "jaxfne_suite_no_2_spectrolaminar_motif.ipynb",
    "jaxfne_suite_no_3_low_frequency_scaling.ipynb",
    "jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb",
    "jaxfne_v0313_omission_oddball.ipynb",
    "jaxfne_v031_single_neuron.ipynb",
    "jaxfne_v032_parameter_sweep.ipynb",
    "jaxfne_v033_two_neuron_ei.ipynb",
    "jaxfne_v035_small_recurrent_ei.ipynb",
    "jaxfne_v036_100_neuron_ei_population.ipynb",
    "jaxfne_v038_lfp_csd_readout.ipynb",
    "jaxfne_v040_continuous_omission_oddball.ipynb",
    "jaxfne_v040_homeostasis_plasticity_dc_noise_sweep.ipynb",
]

pytestmark = [pytest.mark.slow, pytest.mark.notebook]

# No placeholder notebooks remain: all release-facing études (1, 3-12, TCM)
# are real, implemented, and executed by the gate as of 0.4.5. Kept as an
# empty set so the xfail machinery below stays wired for any future
# planned-but-unbuilt notebook without reworking the test.
PLACEHOLDER_NOTEBOOKS = set()


@pytest.mark.parametrize("notebook_rel_path", RELEASE_FACING_NOTEBOOKS)
def test_release_notebook_executes_cleanly(notebook_rel_path, tmp_path):
    """Execute a release-facing notebook end-to-end and assert no cell errors."""
    if os.environ.get("SKIP_NOTEBOOK_EXECUTION", "0") == "1":
        pytest.skip("SKIP_NOTEBOOK_EXECUTION=1 set in environment")

    nb_path = TUTORIALS_DIR / notebook_rel_path
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    if notebook_rel_path in PLACEHOLDER_NOTEBOOKS:
        # execute_notebook_via_nbclient's NotebookClient.execute() raises
        # CellExecutionError on the first failing cell (nbclient's default
        # error-stops-execution behavior) rather than returning it in the
        # error list -- catch it directly here instead of relying on the
        # helper's "empty list = clean" contract, which only holds for
        # notebooks that genuinely execute with zero errors.
        from nbclient.exceptions import CellExecutionError

        try:
            execute_notebook_via_nbclient(nb_path, tmp_path, timeout=900)
        except CellExecutionError as exc:
            assert "NotImplementedError" in str(exc), (
                "Placeholder notebook failed for an unexpected reason "
                f"(expected NotImplementedError): {exc}"
            )
            pytest.xfail("Skeleton-only Etude: implementation pending (see notebook plan cell).")
        else:
            pytest.fail(
                "Placeholder notebook executed with zero errors -- it has been "
                "implemented; remove it from PLACEHOLDER_NOTEBOOKS."
            )

    errors = execute_notebook_via_nbclient(nb_path, tmp_path, timeout=900)
    assert errors == [], format_cell_errors(errors)
