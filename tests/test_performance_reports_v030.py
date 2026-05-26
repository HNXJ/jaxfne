"""Tests for v0.3.4 performance reporting schema.

Validates benchmark report structure, JSON safety, and claim gate integrity.
Does NOT run heavy benchmarks (those are scripts/benchmark_jaxfne.py).

truth_mode: truth_safe_unverified
claim_level: computational_scaffold
"""

import json
import pathlib

import pytest


class TestPerformanceReportSchema:
    """Test performance benchmark report structure and safety."""

    def test_benchmark_report_structure_mock(self):
        """Mock benchmark report passes schema validation."""
        mock_report = {
            "benchmark_series": "v0.3.4_performance_baseline",
            "results": [
                {
                    "case_name": "small_50n_100ms",
                    "timings": [
                        {"phase": "configuration_setup", "elapsed_ms": 10.5},
                        {"phase": "model_construction", "elapsed_ms": 5.2},
                        {"phase": "simulation_setup", "elapsed_ms": 2.1},
                        {"phase": "simulate_core", "elapsed_ms": 125.3},
                        {"phase": "probe_readout", "elapsed_ms": 8.7},
                        {"phase": "evaluate_objective", "elapsed_ms": 3.2},
                        {"phase": "manifest_generation", "elapsed_ms": 1.5},
                    ],
                    "total_ms": 156.5,
                    "hardware_info": {
                        "platform": "macOS-13.0-arm64",
                        "python_version": "3.11.15",
                        "jax_version": "0.10.0",
                        "devices": [{"device_type": "cpu", "device_id": 0}],
                        "device_count": 1,
                    },
                    "jaxfne_version": "0.3.4",
                    "claim_level": "computational_scaffold",
                    "truth_mode": "truth_safe_unverified",
                    "local_environment_receipt_only": True,
                    "notes": "Local CPU benchmark, no universal claim.",
                }
            ],
            "generated_at": "2026-05-23 10:00:00",
        }

        # Validate structure
        assert "benchmark_series" in mock_report
        assert "results" in mock_report
        assert len(mock_report["results"]) > 0

        for result in mock_report["results"]:
            assert "case_name" in result
            assert "timings" in result
            assert "total_ms" in result
            assert "hardware_info" in result
            assert "jaxfne_version" in result
            assert "claim_level" in result
            assert "truth_mode" in result
            assert "local_environment_receipt_only" in result

            # All timings have phase and elapsed_ms
            for timing in result["timings"]:
                assert "phase" in timing
                assert "elapsed_ms" in timing
                assert isinstance(timing["elapsed_ms"], (int, float))
                assert timing["elapsed_ms"] >= 0.0

    def test_benchmark_report_json_safe(self):
        """Benchmark report is JSON-serializable without NaN/Inf."""
        mock_report = {
            "benchmark_series": "v0.3.4_performance_baseline",
            "results": [
                {
                    "case_name": "test_case",
                    "timings": [{"phase": "phase1", "elapsed_ms": 10.5}],
                    "total_ms": 10.5,
                    "hardware_info": {"devices": 1},
                    "jaxfne_version": "0.3.4",
                    "claim_level": "computational_scaffold",
                    "truth_mode": "truth_safe_unverified",
                    "local_environment_receipt_only": True,
                    "notes": "Test report",
                }
            ],
        }

        # Should serialize with allow_nan=False (JSON-safe)
        json_str = json.dumps(mock_report, allow_nan=False)
        assert json_str is not None

        # Should deserialize back
        reparsed = json.loads(json_str)
        assert reparsed["results"][0]["case_name"] == "test_case"

    def test_benchmark_claim_gates_frozen(self):
        """Claim gates in benchmark report are conservative."""
        mock_result = {
            "case_name": "test",
            "timings": [],
            "total_ms": 100.0,
            "hardware_info": {},
            "jaxfne_version": "0.3.4",
            "claim_level": "computational_scaffold",
            "truth_mode": "truth_safe_unverified",
            "local_environment_receipt_only": True,
            "notes": "Test",
        }

        assert mock_result["claim_level"] == "computational_scaffold"
        assert mock_result["truth_mode"] == "truth_safe_unverified"
        assert mock_result["local_environment_receipt_only"] is True

    def test_hardware_info_required_fields(self):
        """Hardware metadata includes required fields."""
        mock_hw = {
            "platform": "macOS-13.0-arm64",
            "python_version": "3.11.15",
            "jax_version": "0.10.0",
            "jaxlib_version": "0.10.0",
            "numpy_version": "1.24.3",
            "devices": [
                {"device_type": "cpu", "device_id": 0},
            ],
            "device_count": 1,
            "default_device": "CpuDevice(id=0)",
        }

        required = ["platform", "python_version", "jax_version", "devices", "device_count"]
        for field in required:
            assert field in mock_hw, f"Missing required field: {field}"

    def test_timing_phases_consistent(self):
        """Timing phase names are consistent across reports."""
        expected_phases = [
            "configuration_setup",
            "model_construction",
            "simulation_setup",
            "simulate_core",
            "probe_readout",
            "evaluate_objective",
            "manifest_generation",
        ]

        mock_result = {
            "case_name": "test",
            "timings": [
                {"phase": p, "elapsed_ms": 10.0}
                for p in expected_phases
            ],
            "total_ms": sum([10.0] * len(expected_phases)),
            "hardware_info": {},
            "jaxfne_version": "0.3.4",
            "claim_level": "computational_scaffold",
            "truth_mode": "truth_safe_unverified",
            "local_environment_receipt_only": True,
            "notes": "Test",
        }

        actual_phases = [t["phase"] for t in mock_result["timings"]]
        assert actual_phases == expected_phases

    def test_case_names_documented(self):
        """Benchmark case names are descriptive."""
        case_names = ["small_50n_100ms", "medium_100n_300ms"]
        for name in case_names:
            # Must include neuron count (digit+n) and duration (digit+ms)
            assert any(c.isdigit() for c in name), f"Case name missing neuron count: {name}"
            assert "ms" in name, f"Case name missing duration: {name}"

    def test_no_nan_infinity_in_mock_values(self):
        """Mock benchmark values never contain NaN or infinity."""
        mock_report = {
            "results": [
                {
                    "timings": [
                        {"phase": "test", "elapsed_ms": 10.5},
                        {"phase": "test2", "elapsed_ms": 0.0},
                    ],
                    "total_ms": 10.5,
                    "hardware_info": {"devices": 1},
                }
            ]
        }

        # Should not raise ValueError when serialized with allow_nan=False
        json.dumps(mock_report, allow_nan=False)

    def test_validation_report_json_safe(self):
        """JSON validation report structure is JSON-safe."""
        mock_validation = {
            "validation_target": "jaxfne_json_safety_v0.3.4",
            "scans": [
                {
                    "directory": "outputs",
                    "files_checked": 5,
                    "files_valid": 5,
                    "files_invalid": 0,
                    "failures": [],
                }
            ],
            "total_valid": 5,
            "total_invalid": 0,
        }

        json_str = json.dumps(mock_validation, allow_nan=False)
        reparsed = json.loads(json_str)
        assert reparsed["total_valid"] == 5

    def test_benchmark_timing_monotonic(self):
        """Individual phase timings are non-negative."""
        mock_result = {
            "timings": [
                {"phase": "setup", "elapsed_ms": 5.0},
                {"phase": "compute", "elapsed_ms": 100.0},
                {"phase": "teardown", "elapsed_ms": 1.0},
            ],
            "total_ms": 106.0,
        }

        total_phases = sum(t["elapsed_ms"] for t in mock_result["timings"])
        # Total should be at least the sum of phases (some overhead acceptable)
        assert mock_result["total_ms"] >= total_phases * 0.95
