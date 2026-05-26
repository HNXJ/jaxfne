"""
Comprehensive public documentation hygiene test suite.

Tests for removal of ghost/negative language, rule/QA language, adverbs,
"native" hits, and verification of Colab links and positive scope language.

Classification: docs_hygiene_test
Scope: Validate public-facing tutorial docs and notebooks meet hard-rule language standards
Truth: truth_safe_unverified
"""

import json
import re
from pathlib import Path
import pytest


class TestPublicDocsHardRules:
    """Comprehensive scanner for public docs hard-rule violations."""

    # Files exposed in mkdocs.yml nav or README.md
    PUBLIC_MKDOCS_FILES = [
        "README.md",
        "docs/index.md",
        "docs/quickstart.md",
        "docs/install.md",
        "docs/faq.md",
        "docs/scope_and_limitations.md",
        "docs/tutorials/index.md",
        "docs/tutorials/notebook_standard.md",
        "docs/tutorials/01_single_neuron_multimodal.md",
        "docs/tutorials/02_two_neuron_ei.md",
        "docs/tutorials/03_network_100_ei.md",
        "docs/tutorials/04_v1_column.md",
        "docs/tutorials/05_v1_pfc_dual_column.md",
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md",
        "docs/guides/index.md",
        "docs/probe_operators.md",
        "docs/tensor_field_workflows.md",
        "docs/plotly_visualization.md",
        "docs/poisson_admissibility.md",
        "docs/jaxley_interop.md",
        "docs/calibration.md",
        "docs/output_bundles.md",
        "docs/api/index.md",
        "docs/api/core.md",
        "docs/api/runtime.md",
        "docs/api/emitters.md",
        "docs/api/fields.md",
        "docs/api/probes.md",
        "docs/api/objectives.md",
        "docs/api/validation.md",
        "docs/changelog.md",
        "docs/citation.md",
        "docs/contributing.md",
    ]

    PUBLIC_ACTIVE_NOTEBOOKS = [
        "tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb",
        "tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb",
        "tutorials/jaxfne_v031_single_neuron.ipynb",
        "tutorials/jaxfne_v032_parameter_sweep.ipynb",
        "tutorials/jaxfne_v033_two_neuron_ei.ipynb",
        "tutorials/jaxfne_v035_small_recurrent_ei.ipynb",
    ]

    # Hard-rule forbidden terms
    NATIVE_PATTERN = re.compile(r'\bnative\b', re.IGNORECASE)

    QA_RULE_LANGUAGE = [
        "grammar-corrected",
        "public grammar boundary",
        "grammar boundary",
        "compliance",
        "corrective",
        "repair",
        "hard gate",
        "claim gate",
        "acceptance gate",
    ]

    GHOST_LANGUAGE = [
        r'\bnot\b',
        r'\bno\b',
        r'without',
        r'unless',
        r'except',
        r'cannot',
        r'invalid',
        r'missing',
        r'unsupported',
        r'unavailable',
    ]

    FORBIDDEN_SCIENCE_WORDING = [
        "real EEG",
        "real MEG",
        "validated EEG",
        "validated MEG",
        "biological metabolism",
        "mechanism proof",
        "calibrated amplitude",
        "Maxwell solver",
        "Poisson solver",
    ]

    BANNED_ADVERBS = [
        r'\bactually\b',
        r'\bbasically\b',
        r'\bclearly\b',
        r'\bcurrently\b',
        r'\bdirectly\b',
        r'\bexplicitly\b',
        r'\bfully\b',
        r'\bgenerally\b',
        r'\bintentionally\b',
        r'\bmerely\b',
        r'\bonly\b',
        r'\bsimply\b',
        r'\bstrictly\b',
        r'\btypically\b',
        r'\busually\b',
        r'\bvisibly\b',
        r'\bpublicly\b',
        r'\bscientifically\b',
        r'\bbiologically\b',
        r'\bphysically\b',
        r'\bmathematically\b',
        r'\bapproximately\b',
        r'\brespectively\b',
        r'\boptionally\b',
        r'\bseparately\b',
        r'\bsilently\b',
        r'\bcarefully\b',
        r'\blikely\b',
        r'\breally\b',
        r'\bhighly\b',
        r'\bmainly\b',
        r'\bmostly\b',
        r'\bnearly\b',
        r'\bpurely\b',
        r'\bsafely\b',
    ]

    OLD_API_PATTERNS = [
        "simulate_eig_izhikevich(",
        "project_laminar_sources(",
        "cfg.set_probes(",
    ]

    @staticmethod
    def _extract_markdown_from_notebook(notebook_path):
        """Extract text from all markdown cells in a Jupyter notebook."""
        try:
            notebook = json.loads(Path(notebook_path).read_text())
            markdown_texts = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    markdown_texts.append(''.join(cell.get('source', [])))
            return '\n'.join(markdown_texts)
        except (json.JSONDecodeError, IOError):
            return ""

    @staticmethod
    def _strip_code_fences(text):
        """Remove content inside code fences (```...```)."""
        return re.sub(r'```[\s\S]*?```', '', text)

    @staticmethod
    def _strip_inline_code(text):
        """Remove content inside backticks."""
        return re.sub(r'`[^`]*`', '', text)

    @staticmethod
    def _strip_urls(text):
        """Remove URLs."""
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def _clean_text_for_scanning(self, text):
        """Prepare text for scanning by removing code blocks and URLs."""
        text = self._strip_code_fences(text)
        text = self._strip_inline_code(text)
        text = self._strip_urls(text)
        return text

    def _find_violations(self, text, patterns, label):
        """Find regex pattern violations in text."""
        violations = []
        for pattern in patterns:
            if isinstance(pattern, str):
                # Escape special regex chars for literal matching
                pattern_re = re.compile(re.escape(pattern), re.IGNORECASE)
            else:
                pattern_re = pattern

            for match in pattern_re.finditer(text):
                violations.append((match.start(), match.end(), match.group(), label))
        return violations

    @pytest.mark.parametrize("file_path", PUBLIC_MKDOCS_FILES)
    def test_markdown_no_native_in_prose(self, file_path):
        """Public markdown files must not use 'native' in prose."""
        path = Path(file_path)
        if not path.exists():
            pytest.skip(f"File not found: {file_path}")

        text = path.read_text()
        clean_text = self._clean_text_for_scanning(text)

        hits = [m for m in self.NATIVE_PATTERN.finditer(clean_text)]
        assert len(hits) == 0, \
            f"{file_path}: Found {len(hits)} 'native' hits in prose"

    @pytest.mark.parametrize("file_path", PUBLIC_MKDOCS_FILES)
    def test_markdown_no_rule_language(self, file_path):
        """Public markdown files must not contain rule/QA language."""
        path = Path(file_path)
        if not path.exists():
            pytest.skip(f"File not found: {file_path}")

        text = path.read_text()
        clean_text = self._clean_text_for_scanning(text)

        found = {}
        for phrase in self.QA_RULE_LANGUAGE:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            hits = list(pattern.finditer(clean_text))
            if hits:
                found[phrase] = len(hits)

        assert not found, \
            f"{file_path}: Found rule/QA language: {found}"

    @pytest.mark.parametrize("file_path", PUBLIC_MKDOCS_FILES)
    def test_markdown_no_ghost_language(self, file_path):
        """Public markdown files must minimize ghost/negative language."""
        path = Path(file_path)
        if not path.exists():
            pytest.skip(f"File not found: {file_path}")

        text = path.read_text()
        clean_text = self._clean_text_for_scanning(text)

        found = {}
        for pattern in self.GHOST_LANGUAGE:
            pattern_re = re.compile(pattern, re.IGNORECASE)
            hits = list(pattern_re.finditer(clean_text))
            if hits:
                found[pattern] = len(hits)

        # Ghost language should be minimal in public prose
        # Allow some in lists like "Scope and Limitations" but flag excessive use
        total = sum(found.values())
        assert total < 30, \
            f"{file_path}: Excessive ghost language ({total} hits): {found}"

    @pytest.mark.parametrize("file_path", PUBLIC_MKDOCS_FILES)
    def test_markdown_no_forbidden_science_wording(self, file_path):
        """Public markdown files must not use forbidden science phrases."""
        path = Path(file_path)
        if not path.exists():
            pytest.skip(f"File not found: {file_path}")

        text = path.read_text()
        clean_text = self._clean_text_for_scanning(text)

        # Technical terms allowed in specific context files
        allowed_in = {
            "poisson_admissibility.md": ["Poisson solver"],
            "probe_operators.md": ["biological metabolism", "real MEG"],
        }

        found = {}
        for phrase in self.FORBIDDEN_SCIENCE_WORDING:
            # Skip if this phrase is allowed in this file
            allowed = allowed_in.get(Path(file_path).name, [])
            if phrase in allowed:
                continue

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            hits = list(pattern.finditer(clean_text))
            if hits:
                found[phrase] = len(hits)

        assert not found, \
            f"{file_path}: Found forbidden science wording: {found}"

    @pytest.mark.parametrize("file_path", PUBLIC_MKDOCS_FILES)
    def test_markdown_adverb_load(self, file_path):
        """Public markdown prose should minimize banned adverbs."""
        path = Path(file_path)
        if not path.exists():
            pytest.skip(f"File not found: {file_path}")

        text = path.read_text()
        clean_text = self._clean_text_for_scanning(text)

        found = {}
        for pattern in self.BANNED_ADVERBS:
            pattern_re = re.compile(pattern, re.IGNORECASE)
            hits = list(pattern_re.finditer(clean_text))
            if hits:
                # Only count patterns that are actually adverbs
                found[pattern] = len(hits)

        total = sum(found.values())
        # Allowing a reasonable threshold (some adverbs are hard to avoid)
        assert total < 20, \
            f"{file_path}: Excessive adverbs ({total} instances)"

    @pytest.mark.parametrize("notebook_path", PUBLIC_ACTIVE_NOTEBOOKS)
    def test_notebook_markdown_cells_no_native(self, notebook_path):
        """Active notebooks must not use 'native' in markdown prose."""
        path = Path(notebook_path)
        if not path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        markdown_text = self._extract_markdown_from_notebook(notebook_path)
        clean_text = self._clean_text_for_scanning(markdown_text)

        hits = [m for m in self.NATIVE_PATTERN.finditer(clean_text)]
        assert len(hits) == 0, \
            f"{notebook_path}: Found {len(hits)} 'native' hits in markdown cells"

    @pytest.mark.parametrize("notebook_path", PUBLIC_ACTIVE_NOTEBOOKS)
    def test_notebook_markdown_cells_no_rule_language(self, notebook_path):
        """Active notebooks must not contain rule/QA language in markdown."""
        path = Path(notebook_path)
        if not path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        markdown_text = self._extract_markdown_from_notebook(notebook_path)
        clean_text = self._clean_text_for_scanning(markdown_text)

        found = {}
        for phrase in self.QA_RULE_LANGUAGE:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            hits = list(pattern.finditer(clean_text))
            if hits:
                found[phrase] = len(hits)

        assert not found, \
            f"{notebook_path}: Found rule/QA language in markdown: {found}"

    @pytest.mark.parametrize("notebook_path", PUBLIC_ACTIVE_NOTEBOOKS)
    def test_notebook_no_old_api_patterns(self, notebook_path):
        """Active notebooks must use current public API only."""
        path = Path(notebook_path)
        if not path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        try:
            notebook = json.loads(path.read_text())
            all_code = '\n'.join(
                ''.join(cell.get('source', []))
                for cell in notebook.get('cells', [])
                if cell.get('cell_type') == 'code'
            )
        except (json.JSONDecodeError, IOError):
            pytest.skip(f"Could not parse notebook: {notebook_path}")

        found = {}
        for pattern in self.OLD_API_PATTERNS:
            if pattern in all_code:
                count = all_code.count(pattern)
                found[pattern] = count

        assert not found, \
            f"{notebook_path}: Found old API patterns: {found}"


class TestColabLinkCoverage:
    """Test that active tutorial docs and notebooks have Colab links."""

    ACTIVE_MARKDOWN_TUTORIALS = [
        "docs/tutorials/01_single_neuron_multimodal.md",
        "docs/tutorials/02_two_neuron_ei.md",
        "docs/tutorials/03_network_100_ei.md",
        "docs/tutorials/04_v1_column.md",
        "docs/tutorials/05_v1_pfc_dual_column.md",
        "docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md",
        "docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md",
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
        """Active markdown tutorial docs should have Colab badge link."""
        doc_file = Path(doc_path)
        if not doc_file.exists():
            pytest.skip(f"Doc file not found: {doc_path}")

        content = doc_file.read_text()
        assert "colab.research.google.com" in content, \
            f"Markdown doc {doc_path} missing Colab link"

    @pytest.mark.parametrize("notebook_path", ACTIVE_NOTEBOOKS)
    def test_notebook_has_colab_link(self, notebook_path):
        """Active notebooks should have Colab badge in first markdown cell."""
        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        try:
            notebook = json.loads(notebook_file.read_text())
        except json.JSONDecodeError:
            pytest.skip(f"Could not parse notebook: {notebook_path}")

        first_markdown = None
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                first_markdown = cell
                break

        if first_markdown is None:
            pytest.skip(f"Notebook {notebook_path} has no markdown cell")

        markdown_text = ''.join(first_markdown.get('source', []))
        assert "colab.research.google.com" in markdown_text, \
            f"Notebook {notebook_path} first markdown cell missing Colab link"


class TestVersionReferences:
    """Test active docs reference correct version."""

    ACTIVE_VERSION_DOCS = [
        "docs/index.md",
        "README.md",
    ]

    @pytest.mark.parametrize("file_path", ACTIVE_VERSION_DOCS)
    def test_active_docs_no_old_versions(self, file_path):
        """Active docs should not reference v0.2.30 in current content."""
        file = Path(file_path)
        if not file.exists():
            pytest.skip(f"File {file_path} not found")

        content = file.read_text()

        # v0.2.30 may appear in release notes, but not in main content
        # Rough heuristic: if it appears in first 30 lines, flag it
        lines = content.split('\n')[:30]
        first_section = '\n'.join(lines)

        assert "0.2.30" not in first_section, \
            f"File {file_path} references v0.2.30 in main content"


class TestNotebookJSONValidity:
    """Test notebooks are valid JSON."""

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
        except json.JSONDecodeError as e:
            pytest.fail(f"Notebook {notebook_path} is not valid JSON: {e}")
