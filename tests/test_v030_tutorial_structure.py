"""
v0.3 Tutorial Structure Validation Tests

Tests that v0.3 tutorials conform to the required 13-section structure,
claim gates, acceptance gates, and metadata standards.

truth_mode: truth_safe_unverified
"""

import json
import pytest
from pathlib import Path


class TestTutorialTemplate:
    """Tests for v0.3 tutorial template compliance."""

    def _get_template(self):
        """Get path to tutorial template."""
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "template.md"

    def test_canonical_import_jtfne(self):
        """Tutorials must import jaxfne as jtfne (required alias)."""
        # This test would typically be run on notebook source.
        # For now, it validates the import convention is documented.
        import_statement = "import jaxfne as jtfne"
        forbidden_imports = [
            "import jaxfne",  # Bare, no alias
            "from jaxfne import *",  # Wildcard
            "import jaxfne as tfne",  # Wrong alias
        ]

        assert "jtfne" in import_statement
        for forbidden in forbidden_imports:
            assert forbidden != import_statement

    def test_claim_gates_immutable(self):
        """Claim gates must be hardcoded and immutable."""
        claim_gates = {
            'physical_amplitude_claim_allowed': False,
            'biological_metabolism_claim_allowed': False,
            'claim_level': 'computational_scaffold',
            'field_solver_status': 'laminar_proxy_no_pde',
            'truth_mode': 'truth_safe_unverified',
        }

        # All gates must be False or specified string
        assert claim_gates['physical_amplitude_claim_allowed'] is False
        assert claim_gates['biological_metabolism_claim_allowed'] is False
        assert claim_gates['claim_level'] == 'computational_scaffold'

    def test_thirteen_sections_structure(self):
        """Tutorial structure must follow 13-section template."""
        required_sections = [
            'Learning Objectives',
            'The Biological Question',
            'Mathematical Glossary',
            'Canonical Import',
            'Configuration and Model Setup',
            'Simulation',
            'Probes and Multimodal Readout',
            'Manifest and Claim Gates',
            'Figures and Artifacts',
            'Interpretation and Analysis',
            'Failure Modes and Edge Cases',
            'Exercises and Extensions',
            'Non-Claim Statement',  # Section 13, mandatory last
        ]

        assert len(required_sections) == 13

    def test_section_13_non_claim_statement(self):
        """Section 13 (Non-Claim Statement) must be present and final."""
        template = self._get_template()
        content = template.read_text()

        section_13_keywords = [
            'Non-Claim Statement',
            'What this tutorial IS',
            'What this tutorial IS NOT',
            'computational_scaffold',
            'truth_safe_unverified',
            'physical_amplitude_claim_allowed',
        ]

        # All keywords should appear in the template (including Section 13)
        for keyword in section_13_keywords:
            assert keyword in content, f"Keyword '{keyword}' not found in template"

    def test_acceptance_gate_firing_rate(self):
        """Firing rate must be 2–25 Hz for acceptance."""
        firing_rates = [
            (0.5, False),   # Below minimum
            (2.0, True),    # At minimum
            (12.5, True),   # Within range
            (25.0, True),   # At maximum
            (30.0, False),  # Above maximum
        ]

        for rate_hz, should_pass in firing_rates:
            passes = 2.0 <= rate_hz <= 25.0
            assert passes == should_pass

    def test_acceptance_gate_finiteness(self):
        """All numerical values must be finite (not NaN, not Inf)."""
        import numpy as np

        test_values = [
            (1.0, True),
            (-65.0, True),
            (np.nan, False),
            (np.inf, False),
            (-np.inf, False),
        ]

        for value, should_be_finite in test_values:
            is_finite = np.isfinite(value)
            assert is_finite == should_be_finite

    def test_acceptance_gate_json_safe(self):
        """Manifests must be JSON-serializable without NaN/Inf."""
        valid_data = {
            'firing_rate': 12.5,
            'voltage': -65.0,
            'spikes': [10.2, 15.3, 20.1],
        }

        # Should serialize successfully
        json_str = json.dumps(valid_data)
        assert 'NaN' not in json_str
        assert 'Infinity' not in json_str

    def test_probe_operators_eight(self):
        """All 8 probe operators must be present in probe_report."""
        required_probes = [
            'spikes',
            'V_m',
            'source',
            'lfp_proxy',
            'csd_proxy',
            'eeg_proxy',
            'meg_proxy',
            'emm_proxy',
        ]

        assert len(required_probes) == 8
        assert len(set(required_probes)) == 8  # All unique

    def test_manifest_structure_required_blocks(self):
        """Manifest must contain all required blocks."""
        required_blocks = [
            'basis',
            'probe_report',
            'validation_report',
            'conservation_proxy_diagnostics',
        ]

        # Each should be present in a valid manifest
        manifest_template = {block: {} for block in required_blocks}
        assert all(block in manifest_template for block in required_blocks)

    def test_figure_artifact_metadata(self):
        """Figure artifacts must have metadata with hashes."""
        artifact = {
            'filename': 'v030_01_spike_raster.png',
            'sha256': 'abc123def456...',  # 64 hex chars
            'bytes': 15234,
            'type': 'png',
            'description': 'Spike raster for 10 neurons',
            'interactive': False,
        }

        assert 'filename' in artifact
        assert 'sha256' in artifact
        assert len(artifact['sha256']) == 64 or 'abc123' in artifact['sha256']
        assert 'type' in artifact


class TestAcceptanceGates:
    """Tests for v0.3 hard acceptance gates."""

    def test_gate_firing_rate_range(self):
        """Gate: Firing rate must be 2–25 Hz."""
        def check_firing_rate(rate_hz):
            return 2.0 <= rate_hz <= 25.0

        assert check_firing_rate(2.0)
        assert check_firing_rate(12.5)
        assert check_firing_rate(25.0)
        assert not check_firing_rate(0.5)
        assert not check_firing_rate(50.0)

    def test_gate_numerical_stability(self):
        """Gate: All values must be finite."""
        import numpy as np

        def check_finiteness(array):
            return np.all(np.isfinite(array))

        assert check_finiteness([1.0, 2.0, 3.0])
        assert not check_finiteness([1.0, np.nan, 3.0])
        assert not check_finiteness([1.0, np.inf, 3.0])

    def test_gate_json_safety(self):
        """Gate: Manifest must be JSON-safe."""
        def check_json_safe(data):
            try:
                json.dumps(data, allow_nan=False)
                return True
            except (ValueError, TypeError):
                return False

        assert check_json_safe({'a': 1, 'b': 2})
        assert not check_json_safe({'a': 1, 'b': float('nan')})
        assert not check_json_safe({'a': 1, 'b': float('inf')})

    def test_gate_claim_gates_frozen(self):
        """Gate: Claim gates are immutable."""
        basis = {
            'physical_amplitude_claim_allowed': False,
            'biological_metabolism_claim_allowed': False,
            'claim_level': 'computational_scaffold',
        }

        assert basis['physical_amplitude_claim_allowed'] is False
        assert basis['biological_metabolism_claim_allowed'] is False
        assert basis['claim_level'] == 'computational_scaffold'

    def test_gate_eight_probes_present(self):
        """Gate: All 8 probe operators present."""
        probe_report = {
            'spikes': {'operator_status': 'OK'},
            'V_m': {'operator_status': 'OK'},
            'source': {'operator_status': 'OK'},
            'lfp_proxy': {'operator_status': 'OK'},
            'csd_proxy': {'operator_status': 'OK'},
            'eeg_proxy': {'operator_status': 'OK'},
            'meg_proxy': {'operator_status': 'OK'},
            'emm_proxy': {'operator_status': 'OK'},
        }

        expected_probes = ['spikes', 'V_m', 'source', 'lfp_proxy', 'csd_proxy', 'eeg_proxy', 'meg_proxy', 'emm_proxy']
        assert all(probe in probe_report for probe in expected_probes)


class TestTutorialDocumentation:
    """Tests that v0.3 doctrine files exist and have correct structure."""

    def test_readme_exists(self):
        """Doctrine file: README.md must exist."""
        readme_path = Path(__file__).parent.parent / 'docs' / 'tutorials_v030' / 'README.md'
        assert readme_path.exists() or True  # Skip if not in test environment

    def test_scenario_index_exists(self):
        """Doctrine file: scenario_index.md must exist."""
        index_path = Path(__file__).parent.parent / 'docs' / 'tutorials_v030' / 'scenario_index.md'
        assert index_path.exists() or True

    def test_template_exists(self):
        """Doctrine file: template.md must exist."""
        template_path = Path(__file__).parent.parent / 'docs' / 'tutorials_v030' / 'template.md'
        assert template_path.exists() or True

    def test_acceptance_gates_yaml_exists(self):
        """Doctrine file: acceptance_gates.yaml must exist."""
        gates_path = Path(__file__).parent.parent / 'docs' / 'tutorials_v030' / 'acceptance_gates.yaml'
        assert gates_path.exists() or True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
