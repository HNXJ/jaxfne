"""Tests for task schedule compilation and validation."""

import pytest
import jaxfne as jtfne


class TestHierarchicalOddballParadigm:
    """Test paradigm and schedule generation."""

    def test_paradigm_properties(self):
        """Test paradigm has required properties."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm(name="custom_name", sequence=("A", "A", "A", "B"))

        assert paradigm.name == "custom_name"
        assert paradigm.sequence == ("A", "A", "A", "B")
        assert paradigm.prefix_ms == 500.0
        assert paradigm.fixation_required_ms == 500.0

    def test_make_fixation_gate(self):
        """Test fixation gate creation."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        gate = paradigm.make_fixation_gate()

        assert isinstance(gate, jtfne.BehaviorGate)
        assert gate.area == "PFC"
        assert gate.layer_group == "superficial"
        assert gate.target_rate_hz == 10.0
        assert gate.tolerance_hz == 1.0

    def test_schedule_compilation(self):
        """Test task schedule compiles correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        schedule = paradigm.to_schedule()

        assert "segments" in schedule
        assert len(schedule["segments"]) > 0
        assert schedule["sequence"] == ["A", "A", "A", "B"]
        assert "truth_gates" in schedule

    def test_schedule_segments_structure(self):
        """Test schedule segments have required fields."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        schedule = paradigm.to_schedule()

        for seg in schedule["segments"]:
            assert "segment_id" in seg
            assert "segment_type" in seg
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "duration_ms" in seg
            assert "fixation_required" in seg
            assert "reward_eligible" in seg

    def test_reward_only_after_d4(self):
        """Test reward is eligible only after d4."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        schedule = paradigm.to_schedule()

        d4_segment = [s for s in schedule["segments"] if s["segment_id"] == "d4"][0]
        assert d4_segment["reward_eligible"] is True

        other_delays = [s for s in schedule["segments"] if s["segment_id"] in ["d1", "d2", "d3"]]
        for seg in other_delays:
            assert seg["reward_eligible"] is False

    def test_total_duration_calculation(self):
        """Test total duration is calculated correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm(
            prefix_ms=500,
            fixation_required_ms=500,
            presentation_ms=100,
            delay_ms=400,
            review_ms=500,
        )
        schedule = paradigm.to_schedule()

        # Prefix + 4*(pres + delay) + review = 500 + 4*(100+400) + 500 = 2500
        expected = 500 + 4 * (100 + 400) + 500
        assert schedule["total_duration_ms"] == expected


class TestBehaviorGate:
    """Test fixation gate logic."""

    def test_gate_initialization(self):
        """Test gate initializes correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        gate = paradigm.make_fixation_gate(target_rate_hz=12.0, tolerance_hz=2.0)

        assert gate.target_rate_hz == 12.0
        assert gate.tolerance_hz == 2.0

    def test_gate_evaluate_in_range(self):
        """Test gate evaluation when rate is in range."""
        import jax.numpy as jnp

        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        gate = paradigm.make_fixation_gate(window_ms=100.0)  # Smaller window for testing

        # Mock spike array with rate close to 10 Hz
        # window=100ms, dt=0.1ms -> need 1000 steps
        # 10 Hz = 0.01 mean spike rate (10 spikes per 1000 ms)
        spikes = jnp.ones((1000, 500)) * 0.01
        gate_pass, report = gate.evaluate(spikes, dt_ms=0.1)

        assert isinstance(gate_pass, bool)
        assert "status" in report or "observed_rate_hz" in report
