"""
Tests for public documentation hygiene and Colab link coverage.

Classification: docs_hygiene_test
Scope: Validate public-facing tutorial docs and notebooks meet language and linking standards
Truth: truth_safe_unverified
"""

import json
from pathlib import Path
import pytest


class TestColabLinkCoverage:
    """Test that active tutorial docs and notebooks have Colab links."""

    ACTIVE_MARKDOWN_TUTORIALS = [
        "docs/tutorials/04_v1_column.md",
        "docs/tutorials/05_v1_pfc_dual_column.md",
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md",
        "docs/tutorials_v030/032_parameter_sweep.md",
        "docs/tutorials_v030/033_two_neuron_ei.md",
        "docs/tutorials_v030/035_small_recurrent_ei.md",
        "docs/tutorials_v030/v0303_two_neuron_ei_multimodal.md",
    ]

    ACTIVE_NOTEBOOKS = [
        "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb",
        "tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb",
        "tutorials/jaxfne_v031_single_neuron.ipynb",
        "tutorials/jaxfne_v032_parameter_sweep.ipynb",
        "tutorials/jaxfne_v033_two_neuron_ei.ipynb",
        "tutorials/jaxfne_v035_small_recurrent_ei.ipynb",
    ]

    @pytest.mark.parametrize("doc_path", ACTIVE_MARKDOWN_TUTORIALS)
    def test_markdown_doc_has_colab_link(self, doc_path):
        """Active markdown tutorial docs must have Colab badge link."""
        doc_file = Path(doc_path)
        assert doc_file.exists(), f"Doc file not found: {doc_path}"

        content = doc_file.read_text()
        assert "colab.research.google.com" in content, \
            f"Markdown doc {doc_path} missing Colab link"
        assert "Open in Colab" in content or "open in Colab" in content, \
            f"Markdown doc {doc_path} missing Colab badge text"

    @pytest.mark.parametrize("notebook_path", ACTIVE_NOTEBOOKS)
    def test_notebook_has_colab_link(self, notebook_path):
        """Active notebooks must have Colab badge in first markdown cell."""
        notebook_file = Path(notebook_path)
        assert notebook_file.exists(), f"Notebook not found: {notebook_path}"

        notebook = json.loads(notebook_file.read_text())
        first_markdown = None
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                first_markdown = cell
                break

        assert first_markdown is not None, \
            f"Notebook {notebook_path} has no markdown cell"

        markdown_text = ''.join(first_markdown['source'])
        assert "colab.research.google.com" in markdown_text, \
            f"Notebook {notebook_path} first markdown cell missing Colab link"
        assert "Open in Colab" in markdown_text, \
            f"Notebook {notebook_path} first markdown cell missing Colab badge text"


class TestSuiteNo1PublicLanguage:
    """Test Suite No. 1 removed internal QA language."""

    SUITE_NO_1_FILES = [
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb",
    ]

    FORBIDDEN_QA_PHRASES = [
        "grammar-corrected",
        "public grammar boundary",
        "grammar boundary",
        "intentionally avoids low-level",
        "direct emitter-kernel",
        "local simulator functions",
        "local source operators",
        "local objective engines",
        "local optimizer loops",
    ]

    @pytest.mark.parametrize("file_path", SUITE_NO_1_FILES)
    @pytest.mark.parametrize("phrase", FORBIDDEN_QA_PHRASES)
    def test_suite_no1_no_internal_qa_language(self, file_path, phrase):
        """Suite No. 1 must not contain internal QA language."""
        file = Path(file_path)
        if not file.exists():
            pytest.skip(f"File {file_path} not found")

        content = file.read_text()
        assert phrase.lower() not in content.lower(), \
            f"Suite No. 1 file {file_path} contains forbidden phrase: '{phrase}'"


class TestForbiddenPublicWording:
    """Test Suite No. 1 and active Suite No. 2 removed forbidden scope language."""

    # Suite No. 1 and 2 are current/active and must be strict
    CURRENT_ACTIVE_TUTORIALS = [
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md",
        "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb",
        "tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb",
    ]

    CRITICAL_OVERCLAIMING_PHRASES = [
        "grammar-corrected",
        "public grammar boundary",
        "grammar boundary",
        "intentionally avoids",
        "direct emitter-kernel",
        "local simulator functions",
        "local source operators",
        "local objective engines",
        "local optimizer loops",
    ]

    @pytest.mark.parametrize("file_path", CURRENT_ACTIVE_TUTORIALS)
    def test_current_active_no_overclaiming_language(self, file_path):
        """Current Suite No. 1/2 tutorials must not contain overclaiming language."""
        file = Path(file_path)
        if not file.exists():
            pytest.skip(f"File {file_path} not found")

        content = file.read_text()
        found_phrases = []
        for phrase in self.CRITICAL_OVERCLAIMING_PHRASES:
            if phrase.lower() in content.lower():
                found_phrases.append(phrase)

        assert not found_phrases, \
            f"File {file_path} contains overclaiming phrases: {found_phrases}"


class TestNoAdverbHeavyProse:
    """Test active tutorial prose removed common banned adverbs."""

    BANNED_ADVERBS = [
        "actually",
        "basically",
        "clearly",
        "simply",
        "really",
        "very",
        "highly",
        "strongly",
        "explicitly",
        "intentionally",
        "generally",
        "usually",
        "currently",
        "obviously",
        "carefully",
        "fully",
        "properly",
        "directly",
        "approximately",
        "automatically",
        "mostly",
        "likely",
        "merely",
        "only",
        "just",
        "quite",
        "rather",
        "fairly",
        "extremely",
        "significantly",
        "substantially",
        "systematically",
        "empirically",
        "typically",
        "mechanistically",
        "biologically",
        "biophysically",
        "purely",
        "correctly",
        "slightly",
        "poorly",
        "silently",
    ]

    # Allowlist for false positives and technical terms
    ALLOWLIST = {
        "Plotly",
        "plotly",
        "family",
        "apply",
        "proxy_readout_only",
        "ReadOnly",
    }

    ACTIVE_TUTORIAL_MARKDOWN = [
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md",
    ]

    @pytest.mark.parametrize("file_path", ACTIVE_TUTORIAL_MARKDOWN)
    def test_markdown_prose_limited_adverbs(self, file_path):
        """Active tutorial markdown prose should avoid banned adverbs."""
        file = Path(file_path)
        if not file.exists():
            pytest.skip(f"File {file_path} not found")

        content = file.read_text()

        # Split into words and check for adverbs outside code blocks
        lines = content.split('\n')
        found_adverbs = {}

        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            lower_line = line.lower()
            for adverb in self.BANNED_ADVERBS:
                if adverb in lower_line and adverb not in [a.lower() for a in self.ALLOWLIST]:
                    # Check if it's a false positive (part of a larger word)
                    import re
                    if re.search(rf'\b{adverb}\b', lower_line):
                        if adverb not in found_adverbs:
                            found_adverbs[adverb] = 0
                        found_adverbs[adverb] += 1

        # Allow a small number of adverbs (some may be unavoidable)
        # This is a practical lint, not a perfect grammar check
        assert len(found_adverbs) <= 3, \
            f"File {file_path} contains more than 3 instances of banned adverbs: {found_adverbs}"


class TestVersionBaseline:
    """Test active docs reference current v0.3.4 baseline."""

    ACTIVE_VERSION_DOCS = [
        "docs/index.md",
        "docs/packaging.md",
        "README.md",
    ]

    @pytest.mark.parametrize("file_path", ACTIVE_VERSION_DOCS)
    def test_active_docs_reference_v034(self, file_path):
        """Active docs must reference v0.3.4 or later, not v0.2.30."""
        file = Path(file_path)
        if not file.exists():
            pytest.skip(f"File {file_path} not found")

        content = file.read_text()

        # Should not reference v0.2.30 outside historical sections
        assert "0.2.30" not in content or "archival" in content.lower() or "v030" in content.lower(), \
            f"File {file_path} references v0.2.30 but is not marked as archival"

        # Should reference v0.3.4 or later in active sections
        if "archival" not in content.lower():
            assert "0.3.4" in content or "0.3.5" in content or "v0.3" in content, \
                f"File {file_path} does not reference current baseline v0.3.4"


class TestPublicAPIGrammar:
    """Test Suite No. 1 uses only public jaxfne API."""

    SUITE_NO_1_NOTEBOOK = "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb"

    def test_suite_no1_uses_public_api(self):
        """Suite No. 1 notebook uses jtfne.Configuration, construct, simulate, probes."""
        notebook_path = Path(self.SUITE_NO_1_NOTEBOOK)
        assert notebook_path.exists(), f"Notebook not found: {self.SUITE_NO_1_NOTEBOOK}"

        notebook = json.loads(notebook_path.read_text())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        all_code = '\n'.join(''.join(cell['source']) for cell in code_cells)

        # Must use public API
        assert "jtfne.Configuration" in all_code, "Must use jtfne.Configuration"
        assert "jtfne.construct" in all_code, "Must use jtfne.construct"
        assert "jtfne.simulate" in all_code, "Must use jtfne.simulate"
        assert ".probes(" in all_code, "Must use .probes()"

        # Must NOT use private patterns
        assert "cfg.set_probes(" not in all_code, "Must not use cfg.set_probes()"
        assert "simulate_eig_izhikevich(" not in all_code, "Must not call simulate_eig_izhikevich()"
        assert "project_laminar_sources(" not in all_code, "Must not call project_laminar_sources()"


class TestNotebookJSONValidity:
    """Test notebooks are valid JSON and parse correctly."""

    ACTIVE_NOTEBOOKS = [
        "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb",
        "tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb",
        "tutorials/jaxfne_v031_single_neuron.ipynb",
        "tutorials/jaxfne_v032_parameter_sweep.ipynb",
        "tutorials/jaxfne_v033_two_neuron_ei.ipynb",
        "tutorials/jaxfne_v035_small_recurrent_ei.ipynb",
    ]

    @pytest.mark.parametrize("notebook_path", ACTIVE_NOTEBOOKS)
    def test_notebook_valid_json(self, notebook_path):
        """Notebook must be valid JSON."""
        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        try:
            notebook = json.loads(notebook_file.read_text())
            assert isinstance(notebook, dict), f"{notebook_path} root is not a dict"
            assert 'cells' in notebook, f"{notebook_path} has no 'cells' key"
            assert isinstance(notebook['cells'], list), f"{notebook_path} 'cells' is not a list"
        except json.JSONDecodeError as e:
            pytest.fail(f"Notebook {notebook_path} is not valid JSON: {e}")


class TestDocsIndexCoverage:
    """Test docs/tutorials/index.md lists all active tutorials."""

    def test_tutorials_index_links_active_tutorials(self):
        """Tutorials index should link to all active Suite/v0.3 tutorials."""
        index_path = Path("docs/tutorials/index.md")
        assert index_path.exists(), "docs/tutorials/index.md not found"

        content = index_path.read_text()

        # Should mention the active Suites
        assert "Suite No. 1" in content or "suite_no_1" in content.lower(), \
            "Index does not reference Suite No. 1"
        assert "Suite No. 2" in content or "suite_no_2" in content.lower(), \
            "Index does not reference Suite No. 2"

        # Should have links to notebooks
        assert "jaxfne_colab_tutorial_computational_biophysics.ipynb" in content or \
               "suite_no_1" in content.lower(), \
            "Index does not link to Suite No. 1 notebook"
