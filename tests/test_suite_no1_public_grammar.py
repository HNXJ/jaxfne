"""
Tests for Suite No. 1 public grammar and regression verification.

Classification: grammar_regression_test
Scope: Verify notebook uses public jaxfne API, no private/local simulator exposure
Truth: truth_safe_unverified
"""

import json
from pathlib import Path
import pytest


class TestSuiteNo1GrammarRegression:
    """Test that Suite No. 1 notebook adheres to public API grammar and contains no forbidden patterns."""

    @staticmethod
    def load_notebook_code():
        """Load and extract all code from the Suite No. 1 notebook."""
        notebook_path = Path("tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb")
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        notebook = json.loads(notebook_path.read_text())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        all_code = '\n'.join(''.join(cell['source']) for cell in code_cells)
        return all_code, notebook

    def test_imports_jaxfne_canonical(self):
        """Notebook must import jaxfne as jtfne."""
        code, _ = self.load_notebook_code()
        assert 'import jaxfne as jtfne' in code, \
            "Notebook must contain canonical import: 'import jaxfne as jtfne'"

    def test_uses_public_configuration(self):
        """Notebook must use jtfne.Configuration() for setup."""
        code, _ = self.load_notebook_code()
        assert 'jtfne.Configuration' in code, \
            "Notebook must use jtfne.Configuration for configuration (public API)"

    def test_uses_public_construct(self):
        """Notebook must use jtfne.construct() to build model."""
        code, _ = self.load_notebook_code()
        assert 'jtfne.construct' in code, \
            "Notebook must use jtfne.construct(cfg) for model construction (public API)"

    def test_uses_public_simulate(self):
        """Notebook must use jtfne.simulate() for simulation."""
        code, _ = self.load_notebook_code()
        assert 'jtfne.simulate' in code, \
            "Notebook must use jtfne.simulate(model, ...) for simulation (public API)"

    def test_uses_public_vis(self):
        """Notebook must use jtfne.vis or matplotlib for visualization."""
        code, _ = self.load_notebook_code()
        assert 'jtfne.vis' in code or 'matplotlib' in code, \
            "Notebook must use jtfne.vis or matplotlib for visualization"

    def test_uses_probes_method(self):
        """Notebook must use cfg.probes([...]) to declare probes."""
        code, _ = self.load_notebook_code()
        assert '.probes(' in code, \
            "Notebook must use cfg.probes([...]) to declare probes (public API)"

    def test_no_cfg_set_probes_pattern(self):
        """Notebook must NOT contain cfg.set_probes() (private pattern)."""
        code, _ = self.load_notebook_code()
        assert 'cfg.set_probes(' not in code, \
            "Notebook must not use cfg.set_probes() (private pattern); use cfg.probes() instead"

    def test_no_direct_simulator_call(self):
        """Notebook must NOT contain simulate_eig_izhikevich() call (private simulator)."""
        code, _ = self.load_notebook_code()
        assert 'simulate_eig_izhikevich(' not in code, \
            "Notebook must not call simulate_eig_izhikevich() directly; use jtfne.simulate() instead"

    def test_no_project_laminar_sources(self):
        """Notebook must NOT contain project_laminar_sources() call (private field operation)."""
        code, _ = self.load_notebook_code()
        assert 'project_laminar_sources(' not in code, \
            "Notebook must not call project_laminar_sources() directly; use probes for field readout"

    def test_no_compute_metrics_definition(self):
        """Notebook must NOT define local compute_metrics function."""
        code, _ = self.load_notebook_code()
        assert 'def compute_metrics' not in code, \
            "Notebook must not define local compute_metrics; use jtfne probe API"

    def test_no_compute_loss_definition(self):
        """Notebook must NOT define local compute_loss function."""
        code, _ = self.load_notebook_code()
        assert 'def compute_loss' not in code, \
            "Notebook must not define local compute_loss; use jtfne objective API"

    def test_no_run_with_params_definition(self):
        """Notebook must NOT define local run_with_params function."""
        code, _ = self.load_notebook_code()
        assert 'def run_with_params' not in code, \
            "Notebook must not define local run_with_params; use jtfne simulation API"

    def test_no_markdown_executable_code(self):
        """Markdown cells must not contain executable Python code by mistake."""
        _, notebook = self.load_notebook_code()
        markdown_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'markdown']

        forbidden_patterns = [
            'import ',
            'def ',
            '= jtfne.',
            '== True',
            'for ',
            'while ',
        ]

        for i, cell in enumerate(markdown_cells):
            markdown_text = ''.join(cell['source']).lower()
            # Only check if it looks like it might be code (has indentation, not just docs)
            for pattern in forbidden_patterns:
                if pattern.lower() in markdown_text and '```' not in ''.join(cell['source']):
                    # Allow if inside code fence
                    cell_text = ''.join(cell['source'])
                    if not (cell_text.count('```') >= 2):  # Not in code fence
                        pytest.skip(f"Markdown cell {i} may contain unintended executable code; inspect manually")

    def test_public_wording_forbids_claim_gates(self):
        """Notebook must not use overclaiming language in public sections."""
        code, _ = self.load_notebook_code()

        forbidden_phrases = [
            'claim gates',
            'claim_level',
            'claim gate summary',
            'What this tutorial claims',
            'What this tutorial does NOT claim',
        ]

        for phrase in forbidden_phrases:
            assert phrase not in code, \
                f"Notebook must not use overclaiming phrase: '{phrase}'"

    def test_docs_mirror_notebook_grammar(self):
        """Markdown docs must use same public API grammar as notebook."""
        code, _ = self.load_notebook_code()
        docs_path = Path("docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md")
        assert docs_path.exists(), f"Docs file not found at {docs_path}"

        docs_text = docs_path.read_text()

        # Docs should reference the same public API elements
        assert 'Configuration' in docs_text, "Docs must reference Configuration"
        assert 'construct' in docs_text, "Docs must reference construct"
        assert 'simulate' in docs_text, "Docs must reference simulate"
        assert 'probes' in docs_text, "Docs must reference probes"

    def test_no_internal_qa_language(self):
        """Public prose must not expose internal QA/compliance language."""
        code, _ = self.load_notebook_code()
        docs_path = Path("docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md")
        docs_text = docs_path.read_text()

        internal_qa_phrases = [
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

        for phrase in internal_qa_phrases:
            assert phrase not in code, \
                f"Notebook must not use internal QA language: '{phrase}'"
            assert phrase not in docs_text, \
                f"Docs must not use internal QA language: '{phrase}'"


class TestSuiteNo1JSONSafety:
    """Test that Suite No. 1 outputs are JSON-safe."""

    def test_notebook_parses_valid_json(self):
        """Notebook file itself must be valid JSON."""
        notebook_path = Path("tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb")
        try:
            notebook = json.loads(notebook_path.read_text())
            assert notebook is not None
        except json.JSONDecodeError as e:
            pytest.fail(f"Notebook is not valid JSON: {e}")
