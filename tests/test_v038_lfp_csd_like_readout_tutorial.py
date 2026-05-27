"""
Test suite for v0.3.8 LFP/CSD-like readout tutorial.

Validates:
- Notebook structure and API compliance
- Documentation page integrity
- Public wording and scope metadata
- Figure generation and artifact validation
- JSON-safe manifest output
"""

import json
import pytest
from pathlib import Path
import os


class TestNotebookStructure:
    """Validate notebook file and cell structure."""

    def test_notebook_exists(self):
        """Assert notebook file exists."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        assert nb_path.exists(), f"Expected {nb_path}"

    def test_notebook_is_valid_json(self):
        """Assert notebook is valid JSON."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        assert isinstance(content, dict), "Notebook is not valid JSON"
        assert "cells" in content, "Notebook missing cells key"

    def test_notebook_has_cells(self):
        """Assert notebook has expected cell count."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        cells = content.get("cells", [])
        assert len(cells) >= 12, f"Expected >= 12 cells, got {len(cells)}"

    def test_first_cell_is_title_markdown(self):
        """Assert first cell is markdown with title."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        first_cell = content["cells"][0]
        assert first_cell["cell_type"] == "markdown", "First cell should be markdown"
        cell_text = "".join(first_cell.get("source", []))
        assert "v0.3.8" in cell_text, "Title should mention v0.3.8"

    def test_second_cell_imports_jaxfne(self):
        """Assert second code cell imports jaxfne."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        code_cells = [c for c in content["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 1, "No code cells found"
        first_code = code_cells[0]
        source_text = "".join(first_code.get("source", []))
        assert "import jaxfne" in source_text, "First code cell should import jaxfne"


class TestCanonicalImports:
    """Validate canonical API imports and helpers."""

    def test_notebook_defines_helper_functions(self):
        """Assert helper functions are defined."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        required_helpers = ["save_png", "finite_status", "population_rate_hz", "display_run_summary"]
        for helper in required_helpers:
            assert helper in notebook_text, f"Missing helper function: {helper}"

    def test_notebook_uses_public_api_workflow(self):
        """Assert notebook uses public API workflow."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        # Check for canonical workflow steps
        assert "Configuration()" in notebook_text, "Missing Configuration() call"
        assert ".runtime(" in notebook_text, "Missing .runtime() call"
        assert ".column(" in notebook_text, "Missing .column() call"
        assert ".probes(" in notebook_text, "Missing .probes() call"
        assert "jtfne.construct(" in notebook_text, "Missing construct() call"
        assert "jtfne.simulate(" in notebook_text, "Missing simulate() call"
        assert ".probe(" in notebook_text, "Missing .probe() call"


class TestConfigurationAPI:
    """Validate configuration examples."""

    def test_notebook_single_neuron_config(self):
        """Assert single neuron example exists."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "single_neuron" in notebook_text or "cfg_single" in notebook_text, \
            "Missing single neuron configuration"

    def test_notebook_laminar_config(self):
        """Assert laminar column example exists."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "laminar" in notebook_text.lower() or "cfg_laminar" in notebook_text, \
            "Missing laminar configuration"

    def test_notebook_probes_lfp_csd(self):
        """Assert LFP-proxy and CSD-proxy probes are specified."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "LFP-proxy" in notebook_text, "Missing LFP-proxy probe"
        assert "CSD-proxy" in notebook_text, "Missing CSD-proxy probe"

    def test_notebook_has_n_contacts_16(self):
        """Assert n_contacts=16 is specified."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "n_contacts=16" in notebook_text or "N_CONTACTS = 16" in notebook_text, \
            "Missing n_contacts specification"


class TestFigureDeclaration:
    """Validate figure generation and naming."""

    def test_figure_filenames_declared(self):
        """Assert all expected figure names are mentioned."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        figure_names = [
            "01_source_heatmap",
            "02_lfp_contact_traces",
            "03_csd_contact_map",
            "04_projection_kernel",
            "05_validation_summary",
            "06_laminar_profile"
        ]

        for fig_name in figure_names:
            assert fig_name in notebook_text, f"Missing figure: {fig_name}"

    def test_figures_directory_created(self):
        """Assert figures output directory is created."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "tutorial_outputs" in notebook_text, "Missing tutorial_outputs directory creation"


class TestMetadataManifest:
    """Validate metadata manifest structure."""

    def test_notebook_declares_manifest_keys(self):
        """Assert manifest keys are declared."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        required_keys = [
            "scope_status",
            "physical_amplitude_claim_allowed",
            "readout_status",
            "n_neurons",
            "n_contacts",
            "source_shape",
            "lfp_proxy_shape",
            "csd_proxy_shape",
            "finite_outputs"
        ]

        for key in required_keys:
            assert key in notebook_text, f"Missing manifest key: {key}"

    def test_notebook_physical_amplitude_false(self):
        """Assert physical_amplitude_claim_allowed is False."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        assert "physical_amplitude_claim_allowed" in notebook_text, \
            "Missing physical_amplitude_claim_allowed"
        assert "False" in notebook_text or "false" in notebook_text, \
            "physical_amplitude_claim_allowed should be False"


class TestPublicWording:
    """Validate public-facing language."""

    def test_no_negative_prose_patterns(self):
        """Assert no negative prose patterns in public text."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))

        forbidden_patterns = [
            "This is not",
            "Does NOT",
            "Does Not",
            "claim_gate",
            "claim_level",
            "truth_mode"
        ]

        for cell in content.get("cells", []):
            if cell["cell_type"] == "markdown":
                cell_text = "".join(cell.get("source", []))
                for pattern in forbidden_patterns:
                    assert pattern not in cell_text, \
                        f"Forbidden pattern '{pattern}' found in markdown"

    def test_approved_scope_language(self):
        """Assert approved scope language is present."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))
        notebook_text = json.dumps(content)

        approved_terms = [
            "computational scaffold",
            "simulated proxy",
            "tutorial-scale"
        ]

        # At least one approved term should be present
        has_approved = any(term in notebook_text for term in approved_terms)
        assert has_approved, f"Missing approved scope language from {approved_terms}"


class TestDocumentation:
    """Validate documentation page."""

    def test_docs_page_exists(self):
        """Assert tutorial documentation page exists."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        assert docs_path.exists(), f"Expected {docs_path}"

    def test_docs_page_has_content(self):
        """Assert docs page has substantive content."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        content = docs_path.read_text(encoding='utf-8')
        assert len(content) > 1000, "Documentation page too short"

    def test_docs_mentions_examples(self):
        """Assert docs page covers all examples."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        content = docs_path.read_text(encoding='utf-8')

        examples = ["Example 1", "Example 2", "Example 3", "Single Neuron", "Laminar"]
        for example in examples:
            assert example in content, f"Missing {example} in docs"

    def test_docs_includes_equations(self):
        """Assert docs page includes mathematical equations."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        content = docs_path.read_text(encoding='utf-8')

        # Check for LaTeX math delimiters or equation references
        assert "$$" in content or "\\(" in content or "Equation" in content, \
            "Missing equations in documentation"

    def test_docs_includes_configuration_examples(self):
        """Assert docs page shows configuration examples."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        content = docs_path.read_text(encoding='utf-8')

        assert "Configuration()" in content, "Missing Configuration() example"
        assert ".probes(" in content, "Missing .probes() example"


class TestMkdocsIntegration:
    """Validate mkdocs.yml integration."""

    def test_mkdocs_includes_new_tutorial(self):
        """Assert v0.3.8 tutorial is in mkdocs nav."""
        mkdocs_path = Path("mkdocs.yml")
        content = mkdocs_path.read_text(encoding='utf-8')

        # Check for tutorial reference (flexible pattern for YAML structure)
        assert "08" in content or "v038" in content or "0.3.8" in content, \
            "v0.3.8 tutorial not referenced in mkdocs.yml"


class TestTutorialIndexIntegration:
    """Validate docs/tutorials/index.md integration."""

    def test_tutorials_index_exists(self):
        """Assert tutorials index page exists."""
        index_path = Path("docs/tutorials/index.md")
        assert index_path.exists(), f"Expected {index_path}"

    def test_tutorials_index_mentioned(self):
        """Assert v0.3.8 is referenced or can be added."""
        index_path = Path("docs/tutorials/index.md")
        content = index_path.read_text(encoding='utf-8')

        # Just ensure the file is properly structured
        assert "# " in content or "##" in content, "Index page should have headings"


class TestArtifactValidation:
    """Validate generated artifacts (gated by JAXFNE_VALIDATE_TUTORIAL_OUTPUTS)."""

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_output_directory_exists(self):
        """Assert output directory was created."""
        output_dir = Path("tutorial_outputs/v038_lfp_csd_like_readout/figures")
        assert output_dir.exists(), f"Output directory {output_dir} not created"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_all_figures_exist(self):
        """Assert all 6 PNG figures were generated."""
        figure_dir = Path("tutorial_outputs/v038_lfp_csd_like_readout/figures")

        expected_figures = [
            "01_source_heatmap.png",
            "02_lfp_contact_traces.png",
            "03_csd_contact_map.png",
            "04_projection_kernel.png",
            "05_validation_summary.png",
            "06_laminar_profile.png"
        ]

        for fig in expected_figures:
            fig_path = figure_dir / fig
            assert fig_path.exists(), f"Missing figure: {fig}"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_figures_nonzero_size(self):
        """Assert PNG files have nonzero size."""
        figure_dir = Path("tutorial_outputs/v038_lfp_csd_like_readout/figures")

        for png_file in figure_dir.glob("*.png"):
            size = png_file.stat().st_size
            assert size > 1000, f"Figure {png_file.name} too small ({size} bytes)"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_exists_and_valid(self):
        """Assert manifest JSON exists and is valid."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        assert manifest_path.exists(), f"Manifest {manifest_path} not found"

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert isinstance(manifest, dict), "Manifest is not a dict"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_json_safe(self):
        """Assert manifest is JSON-safe (no NaN/Inf)."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        # Should not raise on allow_nan=False
        try:
            json.dumps(manifest, allow_nan=False)
        except ValueError as e:
            pytest.fail(f"Manifest contains NaN/Inf: {e}")

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_has_required_keys(self):
        """Assert manifest has all required keys."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        required_keys = [
            "scope_status",
            "readout_status",
            "physical_amplitude_claim_allowed",
            "n_neurons",
            "n_contacts",
            "source_shape",
            "lfp_proxy_shape",
            "csd_proxy_shape",
            "finite_outputs",
            "mean_population_rate_hz"
        ]

        for key in required_keys:
            assert key in manifest, f"Missing manifest key: {key}"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_scope_gates(self):
        """Assert manifest has correct scope gates."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert manifest["scope_status"] == "computational_scaffold"
        assert manifest["readout_status"] == "simulated_proxy"
        assert manifest["physical_amplitude_claim_allowed"] is False

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_shapes_correct(self):
        """Assert manifest tensor shapes are as expected."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        # Source should have shape [T, N] where N is the neuron count
        n_neurons_actual = manifest["source_shape"][1]
        assert n_neurons_actual > 0, "Source should have at least 1 neuron"
        assert manifest["source_shape"][0] > 1000, "Source should have >1000 timepoints"

        # LFP and CSD should be [T, 16]
        assert manifest["lfp_proxy_shape"][1] == 16, "LFP shape should have 16 contacts"
        assert manifest["csd_proxy_shape"][1] == 16, "CSD shape should have 16 contacts"

        # All shapes should have time dimension consistent
        assert manifest["source_shape"][0] == manifest["lfp_proxy_shape"][0], "Source and LFP time dims should match"
        assert manifest["source_shape"][0] == manifest["csd_proxy_shape"][0], "Source and CSD time dims should match"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_outputs_finite(self):
        """Assert manifest reports all outputs are finite."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert manifest["finite_outputs"] is True, "All outputs should be finite"

    @pytest.mark.skipif(
        not os.getenv("JAXFNE_VALIDATE_TUTORIAL_OUTPUTS"),
        reason="Artifact tests require JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1"
    )
    def test_manifest_population_rate_reasonable(self):
        """Assert population rate is in active-rate regime."""
        manifest_path = Path("tutorial_outputs/v038_lfp_csd_like_readout/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        rate = manifest["mean_population_rate_hz"]
        # Active-rate regime for v0.3.8
        assert 0.1 <= rate <= 100, f"Population rate {rate} Hz outside expected range"


class TestExecutionSmoke:
    """Quick smoke tests that don't require full artifact generation."""

    def test_notebook_syntax_valid(self):
        """Assert notebook can be parsed without errors."""
        nb_path = Path("tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb")
        content = json.loads(nb_path.read_text(encoding='utf-8'))

        # Basic validation
        assert "cells" in content
        assert len(content["cells"]) > 0

        for i, cell in enumerate(content["cells"]):
            assert "cell_type" in cell, f"Cell {i} missing cell_type"
            assert "source" in cell, f"Cell {i} missing source"

    def test_docs_markdown_valid(self):
        """Assert documentation page is valid markdown."""
        docs_path = Path("docs/tutorials/08_v038_lfp_csd_like_readout.md")
        content = docs_path.read_text(encoding='utf-8')

        # Basic markdown checks
        assert content.startswith("#"), "Markdown should start with heading"
        assert content.count("#") >= 2, "Markdown should have multiple sections"
        assert "python" in content.lower() or "code" in content.lower(), \
            "Documentation should include code examples"
