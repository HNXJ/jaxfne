"""
v0.3.4 Tutorial Figure Manifest Tests

Validates:
1. Manifest exists and is JSON-safe
2. Figure count meets minimum (>= 10 real data)
3. All figure paths exist
4. All files are PNGs with nonzero size
5. All figures are visually confirmed
6. Status fields are scaffold/proxy
7. Blocked phrase audit
8. Metadata integrity
"""

import json
from pathlib import Path

import pytest


MANIFEST_PATH = Path(__file__).parent.parent / "docs" / "_static" / "tutorial_figures" / "figure_manifest.json"
FIGURES_DIR = MANIFEST_PATH.parent


@pytest.fixture
def manifest():
    """Load and parse the figure manifest."""
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return data


class TestManifestStructure:
    """Tests for manifest schema and structure."""

    def test_manifest_exists(self):
        """Manifest file must exist."""
        assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"

    def test_manifest_json_safe(self, manifest):
        """Manifest must be JSON-safe (no NaN/Inf)."""
        # If we loaded it successfully, it's already JSON-safe
        # Double-check by re-dumping
        json_str = json.dumps(manifest, allow_nan=False)
        assert json.loads(json_str) == manifest

    def test_manifest_has_required_keys(self, manifest):
        """Manifest must have all required top-level keys."""
        required_keys = [
            "figure_count",
            "real_data_figure_count",
            "min_required",
            "jaxfne_version",
            "run_status",
            "model_status",
            "field_solver_status",
            "amplitude_status",
            "metabolism_status",
            "source_script",
            "visual_confirmation_method",
            "figures",
        ]
        for key in required_keys:
            assert key in manifest, f"Missing required key: {key}"

    def test_manifest_status_fields(self, manifest):
        """Status fields must be correct."""
        assert manifest["run_status"] == "tutorial_scaffold"
        assert manifest["model_status"] == "computational_scaffold"
        assert manifest["field_solver_status"] == "linear_solver"
        assert manifest["amplitude_status"] is False
        assert manifest["metabolism_status"] is False


class TestFigureCount:
    """Tests for figure count requirements."""

    def test_figure_count_total(self, manifest):
        """Must have exactly 12 figures."""
        assert manifest["figure_count"] == 12, f"Expected 12 figures, got {manifest['figure_count']}"

    def test_real_data_figure_count(self, manifest):
        """Must have >= 10 real-data figures."""
        real_count = manifest["real_data_figure_count"]
        min_required = manifest["min_required"]
        assert real_count >= min_required, f"Real data figures ({real_count}) < minimum ({min_required})"

    def test_figures_list_length(self, manifest):
        """Figures list must match figure_count."""
        assert len(manifest["figures"]) == manifest["figure_count"]


class TestFigurePaths:
    """Tests for figure file paths and existence."""

    def test_all_figure_paths_exist(self, manifest):
        """Every figure path in manifest must exist."""
        for fig in manifest["figures"]:
            path = Path(fig["path"])
            assert path.exists(), f"Figure file not found: {path}"

    def test_all_figures_are_png(self, manifest):
        """All figure files must be PNG."""
        for fig in manifest["figures"]:
            path = Path(fig["path"])
            assert path.suffix.lower() == ".png", f"Not a PNG: {path}"

    def test_all_figures_nonzero_size(self, manifest):
        """All figure files must be nonzero size."""
        for fig in manifest["figures"]:
            path = Path(fig["path"])
            size = path.stat().st_size
            assert size > 0, f"Zero-size figure: {path}"

    def test_all_figures_have_minimum_size(self, manifest):
        """All figure files should be > 1 KB."""
        for fig in manifest["figures"]:
            path = Path(fig["path"])
            size = path.stat().st_size
            assert size > 1024, f"Figure too small ({size} bytes): {path}"


class TestFigureMetadata:
    """Tests for per-figure metadata."""

    def test_all_figures_have_required_fields(self, manifest):
        """Each figure must have required metadata fields."""
        required_fields = [
            "filename",
            "title",
            "type",
            "uses_real_data",
            "path",
            "visually_confirmed",
            "visual_status",
            "readout_status",
        ]
        for fig in manifest["figures"]:
            for field in required_fields:
                assert field in fig, f"Missing field '{field}' in figure {fig.get('filename', 'unknown')}"

    def test_all_figures_visually_confirmed(self, manifest):
        """All figures must be marked as visually confirmed."""
        for fig in manifest["figures"]:
            assert fig["visually_confirmed"] is True, f"Not confirmed: {fig['filename']}"

    def test_all_figures_visual_status_pass(self, manifest):
        """All figures must have visual_status='pass'."""
        for fig in manifest["figures"]:
            assert fig["visual_status"] == "pass", f"Visual status not 'pass': {fig['filename']}"

    def test_readout_status_proxy(self, manifest):
        """All figures must have readout_status containing proxy/simulated."""
        for fig in manifest["figures"]:
            status = fig.get("readout_status", "")
            if fig.get("uses_real_data", False):
                assert "proxy" in status.lower() or "simulated" in status.lower(), \
                    f"Readout status not proxy/simulated: {fig['filename']}"


class TestBlockedPhrases:
    """Tests for blocked public wording."""

    FORBIDDEN_PHRASES = [
        "real EEG",
        "real MEG",
        "validated EEG",
        "validated MEG",
        "biological metabolism",
        "proof of mechanism",
        "sensor-level",
        "full Maxwell",
        "stress-energy tensor",
        "Maxwell solver",
        "Poisson solver",
    ]

    def test_no_forbidden_phrases_in_descriptions(self, manifest):
        """Figure titles and descriptions must not contain forbidden phrases."""
        for fig in manifest["figures"]:
            title = fig.get("title", "").lower()
            for phrase in self.FORBIDDEN_PHRASES:
                assert phrase.lower() not in title, \
                    f"Forbidden phrase '{phrase}' in title: {fig['filename']}"

    def test_no_forbidden_phrases_in_global_fields(self, manifest):
        """Global manifest fields must not contain forbidden phrases."""
        forbidden_fields = [
            "run_status",
            "model_status",
            "field_solver_status",
            "source_script",
            "visual_confirmation_method",
        ]
        for field in forbidden_fields:
            value = str(manifest.get(field, "")).lower()
            for phrase in self.FORBIDDEN_PHRASES:
                assert phrase.lower() not in value, \
                    f"Forbidden phrase '{phrase}' in field '{field}'"


class TestStatusFields:
    """Tests for immutable status fields."""

    def test_status_fields_immutable(self, manifest):
        """Status fields must be in allowed states."""
        fields = {
            "run_status": "tutorial_scaffold",
            "model_status": "computational_scaffold",
            "field_solver_status": "linear_solver",
            "amplitude_status": False,
            "metabolism_status": False,
        }
        for field_name, field_value in fields.items():
            actual = manifest.get(field_name)
            assert actual == field_value,                 f"Field '{field_name}' has unexpected value: {actual} (expected {field_value})"

    def test_jaxfne_version_matches_frozen_v034_manifest(self, manifest):
        """The manifest's embedded jaxfne_version is a frozen historical value
        (this manifest was generated at v0.3.4 and is not regenerated on
        release, same as docs/releases/v0.3.4.md) -- 0.3.4 is the correct,
        permanent expectation here, not a staleness bug relative to the
        current package version."""
        version = manifest.get("jaxfne_version", "")
        assert version == "0.3.4", f"jaxfne_version: {version} (expected 0.3.4)"


class TestDataIntegrity:
    """Tests for data integrity in manifest."""

    def test_figure_filenames_match_paths(self, manifest):
        """Figure filenames must match their paths."""
        for fig in manifest["figures"]:
            filename = fig["filename"]
            path = fig["path"]
            assert path.endswith(filename), f"Path/filename mismatch: {path} vs {filename}"

    def test_no_duplicate_filenames(self, manifest):
        """All filenames must be unique."""
        filenames = [fig["filename"] for fig in manifest["figures"]]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames found"

    def test_real_data_count_consistency(self, manifest):
        """Real-data count must match figures marked with uses_real_data=True."""
        counted = sum(1 for fig in manifest["figures"] if fig.get("uses_real_data", False))
        reported = manifest.get("real_data_figure_count", 0)
        assert counted == reported, \
            f"Real data count mismatch: {counted} figures vs {reported} reported"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
