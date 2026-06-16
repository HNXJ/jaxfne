"""Receipt / portability gate for the v0.3.31 delta-test sanity notebook.

These checks guard the *release* properties of
``tutorials/jaxfne-sanity-checker-notebook-01.ipynb`` independent of execution:

- no user-specific absolute path is baked into the notebook,
- the portable bootstrap and required structure are present,
- observed-vs-regularized metric language is documented,
- all nine required gate PNG names are referenced,
- the canonical import alias is used,
- no overclaiming wording (real EEG/MEG, calibrated amplitude, mechanism proof,
  biological validation/learning rule, physical CSD) appears anywhere.

They run fast (pure text inspection of the notebook JSON), so they belong in the
default suite as a standing regression guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "tutorials" / "jaxfne-sanity-checker-notebook-01.ipynb"

REQUIRED_PNG_NAMES = [
    "raster.png",
    "eeg_proxy_16ch.png",
    "meg_proxy_16ch.png",
    "agsdr_rate_tuning.png",
    "spectrolaminar_proxy_V1.png",
    "spectrolaminar_proxy_V4.png",
    "spectrolaminar_proxy_MT.png",
    "spectrolaminar_proxy_FEF.png",
    "spectrolaminar_proxy_PFC.png",
]

# Wording that would overclaim beyond the computational-scaffold truth gates.
FORBIDDEN_PHRASES = [
    "real eeg",
    "real meg",
    "calibrated amplitude",
    "physical csd",
    "mechanism proof",
    "biological validation",
    "biological learning rule",
]


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"notebook not found: {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text())


@pytest.fixture(scope="module")
def all_source(notebook) -> str:
    """Concatenated source of every cell (code + markdown)."""
    chunks = []
    for cell in notebook.get("cells", []):
        src = cell.get("source", "")
        chunks.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def code_source(notebook) -> str:
    chunks = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        chunks.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(chunks)


def test_no_user_specific_absolute_path(all_source):
    # A public release notebook must not embed a developer's home/workspace path.
    for needle in ("/Users/hamednejat", "/workspace/main/jaxfne"):
        assert needle not in all_source, f"hardcoded path {needle!r} present in notebook"


def test_portable_bootstrap_present(code_source):
    # The bootstrap resolves the repo via env var, cwd/parents, or installed package.
    assert "JAXFNE_REPO_ROOT" in code_source
    assert "repo_root_mode" in code_source
    assert "pyproject.toml" in code_source
    assert "installed_package" in code_source
    # Stale-module purge before import.
    assert "sys.modules" in code_source


def test_required_apis_verified(code_source):
    for api in (
        "laminar_cortex_config",
        "construct",
        "simulate",
        "euler_scan",
        "validation_report",
        "probe_report",
        "asset_hashes",
    ):
        assert api in code_source, f"required API {api!r} not referenced/verified in notebook"


def test_agsdr_section_present(code_source):
    assert "AGSDR" in code_source
    assert "connectivity_gain" in code_source
    assert "candidate_gain" in code_source


def test_observed_vs_regularized_language(all_source):
    lowered = all_source.lower()
    assert "observed" in lowered
    assert "regularized" in lowered
    assert "min_rate_gate_basis" in all_source
    assert "observed_spike_rate" in all_source


def test_all_required_png_names_referenced(code_source):
    for name in REQUIRED_PNG_NAMES:
        assert name in code_source, f"required figure {name!r} not referenced in notebook"


def test_key_json_artifacts_referenced(code_source):
    assert "optimizer_report.json" in code_source
    assert "metrics.json" in code_source


def test_canonical_import_alias(code_source):
    assert "import jaxfne as jtfne" in code_source


def test_truth_gates_present(all_source):
    for gate in (
        "",
        "computational_scaffold",
        "linear_solver",
        "physical_amplitude_calibrated",
        "biological_learning_claim",
        "mechanism_claim_status",
    ):
        assert gate in all_source, f"truth gate {gate!r} missing from notebook"


def test_no_forbidden_overclaiming_wording(all_source):
    lowered = all_source.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in lowered]
    assert not hits, f"forbidden overclaiming wording present: {hits}"
