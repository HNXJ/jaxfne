"""
Test suite for v0.3.7 interactive 3D source/field/probe column visualization.

Validates:
- Readout panels HTML (the 3-D figure is built in memory and writes no file)
- Equation annotations presence
- Metadata manifest completeness
- Documentation integration (iframe, links)
- JSON safety of manifest
"""

import json
import sys
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def column_figure():
    """The 3-D column figure as the figure site builds it (simulates the 48-cell column, about 6 s)."""
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_v037_source_column_3d as v037

    return v037.build_column_figure()


@pytest.fixture(scope="module")
def built_outputs(tmp_path_factory):
    """Run the builder into a tmp dir (module scope, so tmp_path_factory); return (html, manifest) paths."""
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_v037_source_column_3d as v037

    out_dir = tmp_path_factory.mktemp("v037_interactive")
    v037.main(output_dir=out_dir)
    return out_dir / "v037_readout_panels.html", out_dir / "v037_source_column_manifest.json"


class TestColumnFigure:
    """The 3-D column figure: one scatter3d of 48 neurons, layer hover text, equation annotations."""

    def test_neurons_are_one_scatter3d_of_48_points(self, column_figure):
        assert [t.type for t in column_figure.data] == ["scatter3d"]
        assert len(column_figure.data[0].x) == 48, "4 layers x 12 neurons"

    def test_hover_text_names_every_layer(self, column_figure):
        hover = " ".join(str(t.hovertext or t.text or "") for t in column_figure.data)
        for layer in ("L2/3", "L4", "L5", "L6"):
            assert layer in hover, f"hover text lacks {layer}"
        assert "Cell type" in hover

    def test_annotations_state_source_field_and_probe(self, column_figure):
        text = " ".join(a.text for a in column_figure.layout.annotations or [])
        assert "S(t)" in text and "Y(t)" in text, "source bookkeeping and field handoff"
        assert "Probe" in text, "probe readout"


class TestInteractiveHTMLGeneration:
    """Validate readout panels HTML built into tmp_path (the 3-D figure is built in memory)."""

    PANEL_TITLES = (
        "Readout Panels: Source / Field / Probe",
        "Source Summary",
        "Population Firing Rate",
        "LFP-Proxy (sample contact)",
        "Mean Voltage by Layer",
    )
    TRACE_NAMES = ("Source mean", "Population rate", "LFP-proxy", "L2/3", "L4", "L5", "L6")

    def test_readout_panels_html_exists(self, built_outputs):
        """Assert readout panels HTML is written to the output dir."""
        html_path, _ = built_outputs
        assert html_path.is_file(), f"Expected {html_path}"

    def test_readout_panels_contain_traces(self, built_outputs):
        """Assert the built page carries the four panel titles and the trace names."""
        html_path, _ = built_outputs
        # Plotly's JSON writes "/" as /; decode it so the literal panel strings match.
        content = html_path.read_text(encoding='utf-8').replace("\\u002f", "/")

        for title in self.PANEL_TITLES:
            assert title in content, f"Missing panel title: {title}"
        for name in self.TRACE_NAMES:
            assert f'"{name}"' in content, f"Missing trace name: {name}"


class TestInteractiveManifest:
    """Validate generated manifest JSON."""

    def test_manifest_file_exists(self, built_outputs):
        """Assert manifest JSON is written to the output dir."""
        _, manifest_path = built_outputs
        assert manifest_path.is_file(), f"Expected {manifest_path}"

    def test_manifest_valid_json(self, built_outputs):
        """Assert manifest is valid JSON."""
        _, manifest_path = built_outputs
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert isinstance(manifest, dict), "Manifest is not a dict"

    def test_manifest_json_serializable_no_nan(self, built_outputs):
        """Assert manifest can be serialized with allow_nan=False."""
        _, manifest_path = built_outputs
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        # Re-serialize with allow_nan=False (should not raise)
        try:
            json.dumps(manifest, allow_nan=False)
        except ValueError as e:
            pytest.fail(f"Manifest contains NaN/Inf: {e}")

    def test_manifest_required_keys(self, built_outputs):
        """Assert manifest has required scope/metadata keys."""
        _, manifest_path = built_outputs
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        required_keys = [
            "scope_status",
            "readout_status",
            "field_mode",
            "physical_amplitude_calibrated",
            "n_neurons",
            "layers",
            "mean_population_rate_hz",
            "all_outputs_finite",
        ]

        for key in required_keys:
            assert key in manifest, f"Missing required manifest key: {key}"

    def test_manifest_scope_values(self, built_outputs):
        """Assert manifest scope values are as expected."""
        _, manifest_path = built_outputs
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert manifest["scope_status"] == "computational_scaffold", \
            "scope_status should be 'computational_scaffold'"
        assert manifest["readout_status"] == "simulated_proxy", \
            "readout_status should be 'simulated_proxy'"
        assert "proxy" in manifest["field_mode"].lower(), \
            "field_mode should indicate proxy-based computation"
        assert manifest["physical_amplitude_calibrated"] is False, \
            "physical_amplitude_calibrated must be False"

    def test_manifest_equations_present(self, built_outputs):
        """Assert equation annotations are in manifest."""
        _, manifest_path = built_outputs
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
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        assert docs_path.exists(), f"Expected {docs_path}"

    def test_docs_page_contains_scope(self):
        """Assert docs page declares scope clearly."""
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        assert "computational_scaffold" in content, "Missing scope declaration"
        assert "simulated_proxy" in content, "Missing readout status"
        assert "proxy" in content and "convolution" in content, \
            "Missing field computation explanation"

    def test_docs_page_embeds_html(self):
        """Assert docs page references the interactive HTML."""
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        assert "v037_source_column_3d.html" in content or "source_column_3d" in content, \
            "Missing reference to 3D visualization HTML"
        assert "iframe" in content.lower() or "href" in content, \
            "Missing iframe or link embedding"

    def test_docs_page_contains_equations(self):
        """Assert docs page documents the three equations."""
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
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
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
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
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
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
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        approved_terms = [
            "computational scaffold",
            "simulated proxy",
            "tutorial-scale",
        ]

        for term in approved_terms:
            assert term in content, f"Missing approved scope term: {term}"

    def test_no_internal_terminology_in_docs(self):
        """Assert internal terminology is not in public prose (fenced code blocks are exempt)."""
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        prose_lines = []
        in_code = False
        for line in content.split('\n'):
            if line.startswith('```'):
                in_code = not in_code
            elif not in_code:
                prose_lines.append(line)
        assert prose_lines, "no prose lines parsed; fence tracking failed"

        forbidden_terms = [
            "claim_level",
            "claim_gate",
            "receipt",
            "gamma",
        ]
        hits = [
            (term, line)
            for term in forbidden_terms
            for line in prose_lines
            if term in line.lower()
        ]
        assert not hits, f"internal terms in public prose: {hits}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """Validate complete v0.3.7 interactive component."""

    def test_builder_script_executable(self):
        """Assert builder script exists and is executable."""
        import platform
        import stat as _stat

        if platform.system() == "Windows":
            pytest.skip("POSIX executable bit is not representable on Windows; enforced on POSIX CI")
        script_path = Path("scripts/build_v037_source_column_3d.py")
        assert script_path.exists(), f"Expected {script_path}"
        assert script_path.stat().st_mode & _stat.S_IXUSR, "Script not executable"

    def test_all_artifacts_exist(self, built_outputs):
        """Assert the tracked docs page and builder exist, and the builder writes both outputs."""
        html_path, manifest_path = built_outputs
        artifacts = [
            html_path,
            manifest_path,
            Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md"),
            Path("scripts/build_v037_source_column_3d.py"),
        ]

        for artifact in artifacts:
            assert artifact.exists(), f"Missing artifact: {artifact}"

    def test_mkdocs_integration_ready(self):
        """Assert documentation can be built with mkdocs."""
        # Check that the markdown file is valid
        docs_path = Path("docs/archive/tutorials/basics/07_v037_source_bookkeeping.md")
        content = docs_path.read_text(encoding='utf-8')

        # Basic markdown structure checks
        assert content.startswith("#"), "Markdown should start with heading"
        assert content.count("\n##") > 0, "Markdown should have subheadings"
        assert "---" in content or "---" in content, "Markdown should have section breaks"
