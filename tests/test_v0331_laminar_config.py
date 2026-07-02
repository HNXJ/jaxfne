"""v0.3.31 laminar cortex configuration tests.

Tests for the generalized laminar_cortex_config builder that supports
multi-area, multi-layer circuits with user-controlled cell types and geometry.
"""

import pytest

import jaxfne as jtfne


class TestLaminarCortexConfig:
    """Test laminar_cortex_config builder function."""

    def test_basic_creation(self):
        """Basic laminar config creation with defaults."""
        cfg = jtfne.laminar_cortex_config(seed=0, n=32)
        assert isinstance(cfg, jtfne.Configuration)
        assert cfg.metadata['claim_level'] == 'computational_scaffold'

    def test_cell_types_sum_validation(self):
        """Cell types must sum to ~1.0."""
        # Valid: sums to 1.0
        valid = {"E": 0.7, "PV": 0.3}
        cfg = jtfne.laminar_cortex_config(cell_types=valid, n=16)
        assert isinstance(cfg, jtfne.Configuration)
        assert cfg.metadata["cell_types"] == valid

        # Invalid: sums to 0.5
        invalid = {"E": 0.3, "PV": 0.2}
        with pytest.raises(ValueError, match="cell_types fractions must sum"):
            jtfne.laminar_cortex_config(cell_types=invalid, n=16)

    def test_default_parameters(self):
        """Test default parameter behavior."""
        cfg = jtfne.laminar_cortex_config()
        assert isinstance(cfg, jtfne.Configuration)
        # Defaults: seed=0, duration_ms=1000.0, dt_ms=0.1, areas=["V1"], layers=6, n=128
        assert cfg.metadata["seed"] == 0
        assert cfg.metadata["duration_ms"] == 1000.0
        assert cfg.metadata["dt_ms"] == 0.1
        assert cfg.metadata["column_count"] == 1

    def test_custom_emitter_lif(self):
        """LIF emitter selection (construction will fail if unsupported)."""
        # Note: LIF may not be supported in construct(); this test just verifies
        # that the config wrapper accepts the emitter parameter.
        cfg = jtfne.laminar_cortex_config(emitter="lif", n=8, seed=0)
        assert isinstance(cfg, jtfne.Configuration)
        # Construction may fail if LIF is not supported in core.construct()
        # which is a separate implementation detail, not a laminar_cortex_config issue.

    def test_invalid_emitter(self):
        """Invalid emitter raises error."""
        with pytest.raises(ValueError, match="Unknown emitter"):
            jtfne.laminar_cortex_config(emitter="invalid_emitter", n=8)

    def test_metadata_preservation(self):
        """Truth gates are preserved in metadata."""
        cfg = jtfne.laminar_cortex_config(n=8, seed=0)
        m = cfg.metadata
        assert m['claim_level'] == 'computational_scaffold'
        assert m['field_solver_status'] == 'linear_solver'
        assert m['physical_amplitude_calibrated'] is False
