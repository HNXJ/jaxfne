"""Static-text thinness gate for Etude No. 1.

This test intentionally uses only stdlib JSON so collection stays fast and does
not require notebook dependencies in core/fast CI.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "artifacts"
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
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def test_notebook_contains_required_tokens():
    src = _notebook_source()
    missing = [token for token in REQUIRED_TOKENS if token not in src]
    assert not missing, f"Notebook missing required tokens: {missing}"


def test_notebook_lacks_forbidden_tokens():
    src = _notebook_source()
    present = [token for token in FORBIDDEN_TOKENS if token in src]
    assert not present, f"Notebook contains forbidden tokens: {present}"
