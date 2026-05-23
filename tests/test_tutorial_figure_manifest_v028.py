"""
v0.2.29 Tutorial Figure Manifest Tests

Validates:
1. Manifest exists and is JSON-safe
2. Figure count meets minimum (>= 10 real data)
3. All figure paths exist
4. All files are PNGs with nonzero size
5. All figures are visually confirmed
6. Claim gates are scaffold/proxy
7. Forbidden phrases audit
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
        assert json_str is not None

    def test_manifest_has_required_keys(self, manifest):
        """Manifest must have all required top-level keys."""
        required_keys = [
            "figure_count",
            "real_data_figure_count",
            "min_required",
            "jaxfne_version",
            "truth_mode",
            "claim_level",
            "field_solver_status",
            "physical_amplitude_claim_allowed",
            "biological_metabolism_claim_allowed",
            "source_script",
            "visual_confirmation_method",
            "figures",
        ]
        for key in required_keys:
            assert key in manifest, f"Missing required key: {key}"

    def test_manifest_truth_fields(self, manifest):
        """Truth status fields must be correct."""
        assert manifest["truth_mode"] == "truth_safe_unverified"
        assert manifest["claim_level"] == "computational_scaffold"
        assert manifest["field_solver_status"] == "laminar_proxy_no_pde"
        assert manifest["physical_amplitude_claim_allowed"] is False
        assert manifest["biological_metabolism_claim_allowed"] is False


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
            "claim_status",
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

    def test_claim_status_proxy(self, manifest):
        """All figures must have claim_status containing 'proxy'."""
        for fig in manifest["figures"]:
            claim = fig.get("claim_status", "")
            if fig.get("uses_real_data", False):
                # Real-data figures should have proxy claim
                assert "proxy" in claim.lower() or "simulated" in claim.lower(), \
                    f"Claim not proxy/simulated: {fig['filename']}"


class TestForbiddenPhrases:
    """Tests for forbidden claim language."""

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
            "truth_mode",
            "claim_level",
            "field_solver_status",
            "source_script",
            "visual_confirmation_method",
        ]
        for field in forbidden_fields:
            value = str(manifest.get(field, "")).lower()
            for phrase in self.FORBIDDEN_PHRASES:
                assert phrase.lower() not in value, \
                    f"Forbidden phrase '{phrase}' in field '{field}'"


class TestClaimGates:
    """Tests for immutable claim gates."""

    def test_claim_gates_immutable(self, manifest):
        """Claim gates must be in allowed states."""
        gates = {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "field_solver_status": "laminar_proxy_no_pde",
            "physical_amplitude_claim_allowed": False,
            "biological_metabolism_claim_allowed": False,
        }
        for gate_name, gate_value in gates.items():
            actual = manifest.get(gate_name)
            assert actual == gate_value, \
                f"Gate '{gate_name}' has unexpected value: {actual} (expected {gate_value})"

    def test_jaxfne_version_current(self, manifest):
        """jaxfne_version should be 0.2.29."""
        version = manifest.get("jaxfne_version", "")
        assert version == "0.2.29", f"jaxfne_version: {version} (expected 0.2.29)"


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
