"""Static-text thinness gate for Etude No. 1.

Asserts the notebook delegates all reusable work to the jaxfne package
(`jtfne.tutorial_utils.*` / `jtfne.vis.*`) and contains no local simulator,
builder, readout, or legacy low-level scaffolding.
"""
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "tutorials"
    / "etudes"
    / "jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
)

REQUIRED_TOKENS = (
    "import jaxfne as jtfne",
    "tutorial_utils",
    "activity_trace_suite",
    "spectrolaminar_suite_3panel",
)

FORBIDDEN_TOKENS = (
    "jbiophysic",
    "git clone",
    "sys.path.insert",
    "def simulate_emitters",
    "def build_tfne_izhikevich_model",
    "def activity_suit",
    "def spectrolaminar_from_trials",
    "Configuration(",
)


def _notebook_source() -> str:
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    return "\n".join("".join(cell["source"]) for cell in nb.cells)


def test_notebook_contains_required_tokens():
    src = _notebook_source()
    missing = [t for t in REQUIRED_TOKENS if t not in src]
    assert not missing, f"Notebook missing required tokens: {missing}"


def test_notebook_lacks_forbidden_tokens():
    src = _notebook_source()
    present = [t for t in FORBIDDEN_TOKENS if t in src]
    assert not present, f"Notebook contains forbidden tokens: {present}"
