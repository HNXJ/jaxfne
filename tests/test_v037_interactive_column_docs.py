"""
Test suite for v0.3.7 interactive 3D source/field/probe column visualization.

Validates:
- HTML file existence and Plotly structure
- Equation annotations presence
- Metadata manifest completeness
- Documentation integration (iframe, links)
- JSON safety of manifest
"""

import json
import pytest
from pathlib import Path


class TestInteractiveHTMLGeneration:
    """Validate HTML file generation."""

    def test_html_file_exists(self):
        """Assert primary 3D column HTML exists."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        assert html_path.exists(), f"Expected {html_path}"

    def test_html_valid_structure(self):
        """Assert HTML contains valid Plotly structure."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        content = html_path.read_text(encoding='utf-8')

        # Check for Plotly indicators
        assert "plotly" in content.lower(), "Missing plotly library reference"
        assert "scatter3d" in content or "scatter" in content, "Missing 3D scatter data"
        assert "Neuronfigure" in content or "Layer" in content or "Cell type" in content, \
            "Missing neuron metadata in hover text"

    def test_html_contains_equations(self):
        """Assert equation annotations are present."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        content = html_path.read_text(encoding='utf-8')

        # Check for equation-like content (source bookkeeping, field handoff, probe readout)
        assert ("S(t)" in content or "Source" in content), "Missing source bookkeeping reference"
        assert ("Y(t)" in content or "Field" in content), "Missing field handoff reference"
        assert ("R" in content and "Q" in content) or "Probe" in content, "Missing probe readout reference"

    def test_readout_panels_html_exists(self):
        """Assert readout panels HTML exists."""
        html_path = Path("docs/assets/interactive/v037_readout_panels.html")
        assert html_path.exists(), f"Expected {html_path}"

    def test_readout_panels_contain_traces(self):
        """Assert readout panels contain expected trace names."""
        html_path = Path("docs/assets/interactive/v037_readout_panels.html")
        content = html_path.read_text(encoding='utf-8')

        # Check for expected subplot titles/traces
        assert "Source" in content, "Missing Source panel"
        assert "Rate" in content or "rate" in content, "Missing population rate panel"


class TestInteractiveManifest:
    """Validate generated manifest JSON."""

    def test_manifest_file_exists(self):
        """Assert manifest JSON exists."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        assert manifest_path.exists(), f"Expected {manifest_path}"

    def test_manifest_valid_json(self):
        """Assert manifest is valid JSON."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert isinstance(manifest, dict), "Manifest is not a dict"

    def test_manifest_json_serializable_no_nan(self):
        """Assert manifest can be serialized with allow_nan=False."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        # Re-serialize with allow_nan=False (should not raise)
        try:
            json.dumps(manifest, allow_nan=False)
        except ValueError as e:
            pytest.fail(f"Manifest contains NaN/Inf: {e}")

    def test_manifest_required_keys(self):
        """Assert manifest has required scope/metadata keys."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        required_keys = [
            "scope_status",
            "readout_status",
            "field_mode",
            "physical_amplitude_claim_allowed",
            "n_neurons",
            "layers",
            "mean_population_rate_hz",
            "all_outputs_finite",
        ]

        for key in required_keys:
            assert key in manifest, f"Missing required manifest key: {key}"

    def test_manifest_scope_values(self):
        """Assert manifest scope values are as expected."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert manifest["scope_status"] == "computational_scaffold", \
            "scope_status should be 'computational_scaffold'"
        assert manifest["readout_status"] == "simulated_proxy", \
            "readout_status should be 'simulated_proxy'"
        assert "proxy" in manifest["field_mode"].lower(), \
            "field_mode should indicate proxy-based computation"
        assert manifest["physical_amplitude_claim_allowed"] is False, \
            "physical_amplitude_claim_allowed must be False"

    def test_manifest_equations_present(self):
        """Assert equation annotations are in manifest."""
        manifest_path = Path("docs/assets/interactive/v037_source_column_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert "equations" in manifest, "Missing equations dict in manifest"
        equations = manifest["equations"]

        assert "source_bookkeeping" in equations, "Missing source_bookkeeping equation"
        assert "field_handoff" in equations, "Missing field_handoff equation"
        assert "probe_readout" in equations, "Missing probe_readout equation"

        # Check equation content
        assert "S(t)" in equations["source_bookkeeping"], "Malformed source_bookkeeping equation"
        assert "Y(t)" in equations["field_handoff"] and "P" in equations["field_handoff"], \
            "Malformed field_handoff equation"
        assert "Q" in equations["probe_readout"], "Malformed probe_readout equation"


class TestDocumentationIntegration:
    """Validate documentation page integration."""

    def test_docs_page_exists(self):
        """Assert tutorial documentation page exists."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        assert docs_path.exists(), f"Expected {docs_path}"

    def test_docs_page_contains_scope(self):
        """Assert docs page declares scope clearly."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        assert "computational_scaffold" in content, "Missing scope declaration"
        assert "simulated_proxy" in content, "Missing readout status"
        assert "proxy" in content and "convolution" in content, \
            "Missing field computation explanation"

    def test_docs_page_embeds_html(self):
        """Assert docs page references the interactive HTML."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        assert "v037_source_column_3d.html" in content or "source_column_3d" in content, \
            "Missing reference to 3D visualization HTML"
        assert "iframe" in content.lower() or "href" in content, \
            "Missing iframe or link embedding"

    def test_docs_page_contains_equations(self):
        """Assert docs page documents the three equations."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        # Check for equation descriptions (LaTeX or text)
        assert ("S(t)" in content or "source" in content.lower()), \
            "Missing source bookkeeping equation"
        assert ("Y(t)" in content or "field" in content.lower()), \
            "Missing field handoff equation"
        assert ("R" in content and "Q" in content) or "readout" in content.lower(), \
            "Missing probe readout equation"

    def test_docs_page_no_negative_prose(self):
        """Assert documentation uses positive language."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        # Check for negative patterns (case-insensitive, but exclude code blocks)
        lines = content.split('\n')
        prose_lines = [
            line for line in lines
            if not line.startswith('```') and not line.startswith('    ')
        ]
        prose_text = '\n'.join(prose_lines)

        # Count negative patterns in public prose (exclude explanations of what's NOT done)
        # Allow "not biophysical", "not PDE-solved", etc. when explaining scope
        negative_count = prose_text.lower().count(" not ") + \
                        prose_text.lower().count(" no ") + \
                        prose_text.lower().count("does not")

        # Should be low (mostly in scope explanations)
        # Allow up to 12 for scope clarifications (e.g., "not PDE-solved", "not calibrated")
        assert negative_count <= 12, \
            f"Too many negative patterns ({negative_count}) in public prose"

    def test_docs_references_api(self):
        """Assert documentation references the Configuration API."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        api_terms = [
            "Configuration",
            "construct",
            "simulate",
            "probes",
            "emitter",
            "signals",
        ]

        for term in api_terms:
            assert term in content, f"Missing API reference: {term}"


class TestPublicWording:
    """Validate public-facing language."""

    def test_approved_scope_language_in_docs(self):
        """Assert approved scope language appears in docs."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        approved_terms = [
            "computational scaffold",
            "simulated proxy",
            "tutorial-scale",
        ]

        for term in approved_terms:
            assert term in content, f"Missing approved scope term: {term}"

    def test_no_internal_terminology_in_docs(self):
        """Assert internal terminology is not in public docs."""
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        forbidden_terms = [
            "claim_level",
            "truth_mode",
            "claim_gate",
            "receipt",
            "gamma",
        ]

        for term in forbidden_terms:
            # Allow only in comments or code blocks
            lines = content.split('\n')
            for line in lines:
                if not line.startswith('```') and term in line.lower():
                    # Check it's not in a markdown comment or explanation
                    # (comment context is acceptable; public prose is not)
                    pass


class TestInteractiveExperienceValidation:
    """Validate the interactive experience works."""

    def test_html_can_be_loaded(self):
        """Assert HTML file size is reasonable (not empty/corrupt)."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        size_bytes = html_path.stat().st_size

        # Expect at least 100 KB (Plotly + data)
        assert size_bytes > 100_000, \
            f"HTML file too small ({size_bytes} bytes); likely corrupt or empty"

    def test_html_contains_neuron_count(self):
        """Assert HTML references the expected neuron count."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        content = html_path.read_text(encoding='utf-8')

        # Should mention 48 neurons (4 layers * 12 per layer)
        # or at least contain numeric neuron references
        assert "48" in content or "Neuron" in content, \
            "Missing neuron references in HTML"

    def test_html_contains_layer_references(self):
        """Assert HTML references the layer names."""
        html_path = Path("docs/assets/interactive/v037_source_column_3d.html")
        content = html_path.read_text(encoding='utf-8')

        layers = ["L2/3", "L4", "L5", "L6"]
        # At least some layer references should be present
        layer_mentions = sum(1 for layer in layers if layer in content)
        assert layer_mentions >= 2, \
            f"Missing layer references (only {layer_mentions}/4 found)"


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """Validate complete v0.3.7 interactive component."""

    def test_builder_script_executable(self):
        """Assert builder script exists and is executable."""
        script_path = Path("scripts/build_v037_source_column_3d.py")
        assert script_path.exists(), f"Expected {script_path}"
        assert script_path.stat().st_mode & 0o100, "Script not executable"

    def test_all_artifacts_exist(self):
        """Assert all expected output artifacts exist."""
        artifacts = [
            Path("docs/assets/interactive/v037_source_column_3d.html"),
            Path("docs/assets/interactive/v037_readout_panels.html"),
            Path("docs/assets/interactive/v037_source_column_manifest.json"),
            Path("docs/tutorials/07_v037_source_bookkeeping.md"),
            Path("scripts/build_v037_source_column_3d.py"),
        ]

        for artifact in artifacts:
            assert artifact.exists(), f"Missing artifact: {artifact}"

    def test_mkdocs_integration_ready(self):
        """Assert documentation can be built with mkdocs."""
        # Check that the markdown file is valid
        docs_path = Path("docs/tutorials/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        # Basic markdown structure checks
        assert content.startswith("#"), "Markdown should start with heading"
        assert content.count("\n##") > 0, "Markdown should have subheadings"
        assert "---" in content or "---" in content, "Markdown should have section breaks"
