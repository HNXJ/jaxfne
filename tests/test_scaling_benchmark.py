"""Tests for benchmarks/scaling_benchmark.py (N=100/1,000/10,000 scaling evidence).

Runs only a tiny N so this stays in the fast lane; the full N=100/1,000/10,000
sweep is a deliberate manual/CI-optional run (~20-30s on Apple Silicon CPU,
dominated by the N=10,000 simulate phase) via the script's __main__ entry point.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "benchmarks" / "scaling_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scaling_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_imports():
    assert SCRIPT_PATH.exists()
    module = _load_module()
    assert hasattr(module, "run_case_isolated")
    assert hasattr(module, "main")


def test_run_case_isolated_tiny_n():
    """A tiny-N case runs end to end in an isolated subprocess and returns real timings."""
    module = _load_module()
    result = module.run_case_isolated("n_tiny", n_neurons=8, duration_ms=5.0, dt_ms=0.5)

    assert result["case_name"] == "n_tiny"
    assert result["n_neurons"] == 8
    for phase in ("configuration_setup_ms", "construct_ms", "simulate_ms", "probe_ms"):
        assert phase in result["timings"]
        assert result["timings"][phase] >= 0.0
    assert result["total_ms"] >= 0.0
    assert result["peak_rss_mb"] > 0.0


def test_report_is_json_safe(tmp_path, monkeypatch):
    """The report dict written by main()'s logic round-trips through strict JSON."""
    module = _load_module()
    r1 = module.run_case_isolated("a", n_neurons=8, duration_ms=5.0, dt_ms=0.5)
    r2 = module.run_case_isolated("b", n_neurons=16, duration_ms=5.0, dt_ms=0.5)

    growth = [{
        "from": r1["case_name"],
        "to": r2["case_name"],
        "n_ratio": r2["n_neurons"] / r1["n_neurons"],
        "construct_ratio": r2["timings"]["construct_ms"] / max(r1["timings"]["construct_ms"], 1e-6),
        "simulate_ratio": r2["timings"]["simulate_ms"] / max(r1["timings"]["simulate_ms"], 1e-6),
        "rss_ratio": r2["peak_rss_mb"] / max(r1["peak_rss_mb"], 1e-6),
    }]

    report = {
        "benchmark_series": "test_series",
        "claim_level": "local_environment_receipt_only",
        "results": [r1, r2],
        "growth_ratios": growth,
    }
    json_str = json.dumps(report, allow_nan=False)
    assert json.loads(json_str) == report
