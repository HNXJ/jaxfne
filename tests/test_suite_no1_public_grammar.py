from __future__ import annotations

import json
from pathlib import Path

import nbformat

NOTEBOOK = Path("tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb")
DOC = Path("docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md")


def _nb_text() -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join("".join(cell.get("source", "")) for cell in nb.cells)


def test_suite_no1_uses_public_configuration_grammar():
    text = _nb_text()
    required = [
        "import jaxfne as jtfne",
        "jtfne.Configuration()",
        "cfg_single",
        "cfg_pop",
        "cfg_column",
        "jtfne.construct",
        "jtfne.simulate",
        ".probes(",
        "model_single.probe",
        "model_pop.probe",
        "model_column.probe",
        "jtfne.vis.spectrolaminar",
        "model_column.tune",
    ]
    for token in required:
        assert token in text


def test_suite_no1_avoids_legacy_public_path_calls():
    text = _nb_text()
    forbidden = [
        "simulate_eig_izhikevich(",
        "project_laminar_sources(",
        "jtfne.IzhikevichParams(",
        "jtfne.emitters.izhikevich_eig_params(",
        "def compute_metrics(",
        "def compute_loss(",
        "def run_with_params(",
        "cfg.set_probes(",
    ]
    for token in forbidden:
        assert token not in text


def test_suite_no1_markdown_cells_do_not_contain_executable_python_blocks():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    suspicious = []
    executable_markers = [
        "from pathlib import Path",
        "import shutil",
        "shutil.make_archive",
        "files.download(",
        "subprocess.run(",
        "FIG_DIR =",
        "cfg_single =",
        "model_single =",
    ]
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        src = "".join(cell.get("source", ""))
        if any(marker in src for marker in executable_markers):
            suspicious.append((idx, src[:120]))
    assert not suspicious


def test_suite_no1_public_wording_clean():
    text = _nb_text() + "\n" + DOC.read_text(encoding="utf-8")
    forbidden = [
        "claim",
        "real EEG",
        "real MEG",
        "validated EEG",
        "validated MEG",
        "biological metabolism",
        "mechanism proof",
        "calibrated amplitude",
        "solved PDE",
        "Maxwell solver",
        "Poisson solver",
    ]
    low = text.lower()
    for phrase in forbidden:
        assert phrase.lower() not in low


def test_suite_no1_notebook_json_is_valid():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 15
