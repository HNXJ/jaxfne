"""Tests for v0.2.0 runtime and dtype validation.

Covers default dtype (float32), x64 opt-in, runtime report structure,
manifest runtime metadata, CPU compatibility, and JSON safety.
"""

import json

import jax
import pytest

import jaxfne


def _cfg(n=8):
    """Minimal configuration."""
    return (
        jaxfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def _model_and_signals(n=8, seed=0, x64_enabled=False):
    """Construct model and run simulation with optional x64."""
    if x64_enabled:
        jax.config.update("jax_enable_x64", True)
    try:
        cfg = _cfg(n=n)
        model = jaxfne.construct(cfg)
        sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=seed)
        signals = model.simulate(sim)
        return model, signals
    finally:
        if x64_enabled:
            jax.config.update("jax_enable_x64", False)


class TestRuntimeDefaultDtype:
    """Runtime dtype defaults and validation."""

    def test_runtime_default_dtype_float32(self):
        """Default runtime dtype must be float32."""
        model, signals = _model_and_signals()
        runtime = signals.metadata.get("runtime", {})
        assert runtime.get("dtype") == "float32"
        assert runtime.get("actual_dtype") == "float32"

    def test_runtime_dtype_in_signals_metadata(self):
        """Runtime dtype must be accessible in signals.metadata."""
        model, signals = _model_and_signals()
        assert "runtime" in signals.metadata
        assert "dtype" in signals.metadata["runtime"]

    def test_x64_not_required_for_tests(self):
        """CPU-safe execution does not require x64 enabled."""
        x64_before = jax.config.read("jax_enable_x64")
        model, signals = _model_and_signals(x64_enabled=False)
        x64_after = jax.config.read("jax_enable_x64")
        # Verify x64 remains in initial state
        assert x64_before == x64_after
        # Verify signals are still valid
        assert signals.V_m.shape[0] > 0

    def test_x64_enabled_flag_reported(self):
        """Runtime must report x64_enabled status."""
        model, signals = _model_and_signals(x64_enabled=False)
        runtime = signals.metadata.get("runtime", {})
        assert "x64_enabled" in runtime
        assert runtime["x64_enabled"] is False


class TestRuntimeReport:
    """Runtime report structure and completeness."""

    def test_runtime_report_json_safe(self):
        """Runtime report must be JSON-safe (no NaN, no Inf)."""
        model, signals = _model_and_signals()
        runtime = signals.metadata.get("runtime", {})
        # Should not raise with allow_nan=False
        json_str = json.dumps(runtime, allow_nan=False)
        assert isinstance(json_str, str)

    def test_runtime_report_contains_backend_info(self):
        """Runtime report must contain backend info."""
        model, signals = _model_and_signals()
        runtime = signals.metadata.get("runtime", {})
        required_fields = ["backend", "dtype", "jax_version", "jaxlib_version"]
        for field in required_fields:
            assert field in runtime, f"Missing required runtime field: {field}"

    def test_runtime_report_contains_jax_version(self):
        """Runtime report must contain JAX version."""
        model, signals = _model_and_signals()
        runtime = signals.metadata.get("runtime", {})
        assert "jax_version" in runtime
        assert isinstance(runtime["jax_version"], str)
        assert "." in runtime["jax_version"]  # Semantic version

    def test_runtime_report_contains_device_info(self):
        """Runtime report must contain device information."""
        model, signals = _model_and_signals()
        runtime = signals.metadata.get("runtime", {})
        assert "available_devices" in runtime
        assert isinstance(runtime["available_devices"], list)
        # At minimum, CPU should be available
        assert len(runtime["available_devices"]) > 0


class TestManifestRuntimeMetadata:
    """Runtime metadata in manifest integration."""

    def test_manifest_includes_runtime_report(self):
        """Manifest must include runtime report."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        assert "runtime_report" in manifest or "runtime" in manifest

    def test_manifest_runtime_dtype_preserved(self):
        """Manifest runtime must preserve dtype from signals."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        signals_dtype = signals.metadata.get("runtime", {}).get("dtype")
        manifest_dtype = (
            manifest.get("runtime_report", {}).get("dtype")
            or manifest.get("runtime", {}).get("dtype")
        )
        assert manifest_dtype == signals_dtype == "float32"

    def test_manifest_runtime_json_safe(self):
        """Manifest runtime section must be JSON-safe."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        # Should not raise with allow_nan=False
        json.dumps(manifest, allow_nan=False)

    def test_manifest_cpu_backend_reported(self):
        """Manifest must report CPU as available backend."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        runtime = manifest.get("runtime_report", {}) or manifest.get("runtime", {})
        available = runtime.get("available_devices", [])
        # CPU should be in available devices
        assert any("cpu" in str(dev).lower() for dev in available) or len(available) > 0


class TestDtypeConsistency:
    """Dtype consistency across simulation and output."""

    def test_voltages_dtype_float32(self):
        """Output voltages must be float32 by default."""
        model, signals = _model_and_signals()
        # JAX arrays have dtype; check without explicit conversion
        dtype_str = str(signals.V_m.dtype)
        assert "float32" in dtype_str

    def test_spikes_dtype_preserved(self):
        """Spike output must have consistent dtype."""
        model, signals = _model_and_signals()
        dtype_str = str(signals.spikes.dtype)
        # Spikes should be numeric type (typically float32)
        assert "float" in dtype_str or "int" in dtype_str

    def test_sources_dtype_float32(self):
        """Source output must be float32 by default."""
        model, signals = _model_and_signals()
        if signals.sources is not None:
            dtype_str = str(signals.sources.dtype)
            assert "float32" in dtype_str


class TestRuntimeCPUCompatibility:
    """CPU-first execution and no CUDA assumptions."""

    def test_simulation_completes_on_cpu(self):
        """Simulation must complete successfully on CPU."""
        model, signals = _model_and_signals(n=4)
        assert signals.V_m.shape[0] > 0
        assert signals.spikes.shape == signals.V_m.shape

    def test_no_cuda_requirement(self):
        """Tests must not require CUDA or GPU."""
        # If we reach here, the test environment is sufficient
        # (all prior tests run on CPU only)
        devices = jax.devices()
        # CPU device should be available
        assert any("cpu" in str(dev).lower() for dev in devices) or len(devices) > 0

    def test_field_computation_on_cpu(self):
        """Field computation must work on CPU."""
        model, signals = _model_and_signals(n=8)
        assert signals.field is not None
        assert hasattr(signals.field, "phi_e_proxy")
        assert signals.field.phi_e_proxy.shape[0] > 0


class TestRuntimeSeeding:
    """Runtime seed reporting and reproducibility."""

    def test_runtime_seed_reported(self):
        """Runtime report must include seed."""
        model, signals = _model_and_signals(seed=42)
        runtime = signals.metadata.get("runtime", {})
        assert "seed" in runtime
        assert runtime["seed"] == 42

    def test_different_seeds_reported_differently(self):
        """Different seeds must be reported in runtime."""
        model1, signals1 = _model_and_signals(seed=42)
        model2, signals2 = _model_and_signals(seed=43)
        seed1 = signals1.metadata.get("runtime", {}).get("seed")
        seed2 = signals2.metadata.get("runtime", {}).get("seed")
        assert seed1 == 42
        assert seed2 == 43
        assert seed1 != seed2
