"""Focused, non-timing tests for scripts/benchmark_protocol_v1.py (P-06).

Deliberately no timing thresholds and no large benchmarks: the script's real
workloads (N=1000) stay out of the test lane. These tests cover the
deterministic contract: arg parsing/defaults, structured GPU skip, JSON
schema, and a tiny CPU-safe measurement path.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_protocol_v1.py"

REQUIRED_KEYS = {
    "schema_version",
    "claim_level",
    "status",
    "backend_requested",
    "backend_resolved",
    "backend_note",
    "device",
    "jax_version",
    "jaxlib_version",
    "dtype",
    "jit_policy_resolved",
    "workload",
    "flags",
    "modes",
}


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_protocol_v1", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_imports():
    assert SCRIPT_PATH.exists()
    module = _load_module()
    for name in ("parse_args", "resolve_backend", "measure_mode",
                 "gpu_skip_report", "build_report", "main"):
        assert hasattr(module, name)


def test_cli_defaults():
    """The documented flag contract: conservative defaults, no surprises."""
    module = _load_module()
    ns = module.parse_args([])
    assert ns.backend == "auto"
    assert ns.runs == 3
    assert ns.warmup == 1
    assert ns.seed == 0
    assert ns.mode == "both"
    assert ns.record_fields == "off"
    assert ns.record_weight_trace == "off"
    assert ns.preset == "default"
    assert ns.json_out is None


def test_cli_rejects_invalid_backend_mode():
    module = _load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--backend", "tpu"])
    with pytest.raises(SystemExit):
        module.parse_args(["--mode", "both_hdp"])


def test_gpu_skip_report_is_structured_and_json_safe():
    module = _load_module()
    report = module.gpu_skip_report()
    assert report["status"] == "skipped"
    assert report["backend_requested"] == "gpu"
    assert report["backend_resolved"] is None
    assert "no CPU fallback" in report["skip_reason"]
    assert report["schema_version"] == module.SCHEMA_VERSION
    assert not report["modes"]
    json.dumps(report, allow_nan=False)


def test_gpu_skip_when_no_gpu(monkeypatch, tmp_path, capsys):
    """main() with --backend gpu on a GPU-less machine: structured skip, exit 0."""
    module = _load_module()

    def _no_gpu(platform_=None):
        raise RuntimeError("Unknown backend: 'gpu' requested ... Platforms present are: cpu")

    monkeypatch.setattr(module.jax, "devices", _no_gpu)
    out_path = tmp_path / "gpu_skip.json"
    exit_code = module.main(["--backend", "gpu", "--json-out", str(out_path)])
    assert exit_code == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["status"] == "skipped"
    assert "SKIP" in capsys.readouterr().out


def test_report_schema_and_json_safe(tmp_path):
    """build_report produces the documented schema on a mocked path."""
    module = _load_module()
    fake_args = module.parse_args(["--seed", "7", "--preset", "large"])
    fake_modes = {
        "standard": {
            "mode": "standard",
            "warmup_runs": 1,
            "measured_runs": 2,
            "timings_seconds": [0.5, 0.6],
            "median_seconds": 0.55,
            "min_seconds": 0.5,
            "max_seconds": 0.6,
            "shape_summary": {"V_m": [100, 8]},
            "finite_summary": {"V_m": True},
        }
    }
    report = module.build_report(
        args=fake_args,
        backend_resolved="cpu",
        backend_note="forced",
        mode_results=fake_modes,
        n_steps=100,
    )
    assert REQUIRED_KEYS <= set(report.keys())
    assert report["status"] == "ok"
    assert report["workload"]["seed"] == 7
    assert report["workload"]["preset"] == "large"
    json.dumps(report, allow_nan=False)  # strict JSON-safe
    out = tmp_path / "r.json"
    module._write_report(report, str(out))
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "ok"


def _tiny_standard_run(module):
    cfg = (
        module.jtfne.configuration()
        .network(name="t", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = module.jtfne.construct(cfg)
    return module.measure_mode(
        mode="standard",
        model=model,
        n_neurons=8,
        duration_ms=20.0,
        dt_ms=2.0,
        seed=0,
        backend="cpu",
        warmup=0,
        runs=1,
        record_fields=False,
        record_weight_trace=False,
    )


def test_measure_mode_tiny_end_to_end():
    """A tiny CPU-safe run returns real structure with valid shape/finite fields."""
    module = _load_module()
    result = _tiny_standard_run(module)
    assert result["mode"] == "standard"
    assert result["measured_runs"] == 1
    assert result["timings_seconds"][0] >= 0.0
    assert result["median_seconds"] == result["timings_seconds"][0]
    assert result["shape_summary"]["V_m"][0] == 10  # 20ms / 2ms
    assert result["finite_summary"]["V_m"] is True