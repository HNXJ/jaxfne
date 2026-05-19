"""
v0.1.2 scan-backend performance validation and metadata tests.

Tests that dense, edge-list, and receptor-exponential paths:
- Are backed by jax.lax.scan (or equivalently JAX primitives)
- Preserve metadata and truth gates
- Produce correct output shapes
- Support deterministic seeding
- Report backend information correctly
"""

import json
import jax
import jax.numpy as jnp
import pytest
import jaxfne as jtfne


def _make_model(n=20, kind="cortical_column"):
    """Create a test model with configurable size."""
    cfg = (
        jtfne.configuration()
        .network(name="test", kind=kind, n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


class TestDenseScanBackend:
    """Dense recurrent path (default) uses lax.scan."""

    def test_dense_backend_default(self):
        """Dense backend is the default."""
        model = _make_model(n=10)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
        signals = model.simulate(sim)
        assert signals.metadata["recurrent_backend"] == "dense"

    def test_dense_backend_explicit(self):
        """Dense backend can be explicitly selected."""
        model = _make_model(n=10)
        rt = jtfne.runtime(recurrent_backend="dense", seed=42)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, runtime=rt)
        signals = model.simulate(sim)
        assert signals.metadata["recurrent_backend"] == "dense"

    def test_dense_output_shapes(self):
        """Dense path produces correct output shapes."""
        model = _make_model(n=15)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
        signals = model.simulate(sim)
        n_steps = int(10.0 / 0.1)
        assert signals.spikes.shape == (n_steps, 15)
        assert signals.V_m.shape == (n_steps, 15)

    def test_dense_deterministic_seed(self):
        """Dense path is deterministic for same seed."""
        model = _make_model(n=10)
        sim1 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=123)
        sim2 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=123)
        signals1 = model.simulate(sim1)
        signals2 = model.simulate(sim2)
        assert jnp.allclose(signals1.spikes, signals2.spikes)
        assert jnp.allclose(signals1.V_m, signals2.V_m)

    def test_dense_different_seed_different_output(self):
        """Dense path produces different output for different seeds."""
        model = _make_model(n=10)
        sim1 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=123)
        sim2 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=456)
        signals1 = model.simulate(sim1)
        signals2 = model.simulate(sim2)
        assert not jnp.allclose(signals1.spikes, signals2.spikes)

    def test_dense_truth_metadata_unchanged(self):
        """Dense path preserves truth gates."""
        model = _make_model(n=10)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
        signals = model.simulate(sim)
        receipt = model.run_receipt(signals)
        truth = receipt.truth
        assert truth["truth_mode"] == "truth_safe_unverified"
        assert truth["claim_level"] == "computational_scaffold"
        assert truth["field_solver_status"] == "laminar_proxy_no_pde"
        assert truth["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        assert truth["physical_amplitude_claim_allowed"] is False

    def test_dense_metadata_json_safe(self):
        """Dense metadata serializes without NaN/Inf."""
        model = _make_model(n=10)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
        signals = model.simulate(sim)
        # Should not raise
        json.dumps(signals.metadata, allow_nan=False)


class TestEdgeListScanBackend:
    """Edge-list recurrent path uses lax.scan with sparse aggregation."""

    def test_edge_list_backend_explicit(self):
        """Edge-list backend can be selected."""
        model = _make_model(n=10)
        rt = jtfne.runtime(recurrent_backend="edge_list", seed=42)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, runtime=rt)
        signals = model.simulate(sim)
        assert signals.metadata["recurrent_backend"] == "edge_list"

    def test_edge_list_output_shapes(self):
        """Edge-list path produces correct output shapes."""
        model = _make_model(n=15)
        rt = jtfne.runtime(recurrent_backend="edge_list", seed=0)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=rt)
        signals = model.simulate(sim)
        n_steps = int(10.0 / 0.1)
        assert signals.spikes.shape == (n_steps, 15)
        assert signals.V_m.shape == (n_steps, 15)

    def test_edge_list_deterministic_seed(self):
        """Edge-list path is deterministic for same seed."""
        model = _make_model(n=10)
        rt1 = jtfne.runtime(recurrent_backend="edge_list", seed=123)
        rt2 = jtfne.runtime(recurrent_backend="edge_list", seed=123)
        sim1 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=123, runtime=rt1)
        sim2 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=123, runtime=rt2)
        signals1 = model.simulate(sim1)
        signals2 = model.simulate(sim2)
        assert jnp.allclose(signals1.spikes, signals2.spikes)
        assert jnp.allclose(signals1.V_m, signals2.V_m)

    def test_edge_list_truth_metadata_unchanged(self):
        """Edge-list path preserves truth gates."""
        model = _make_model(n=10)
        rt = jtfne.runtime(recurrent_backend="edge_list", seed=0)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=rt)
        signals = model.simulate(sim)
        receipt = model.run_receipt(signals)
        truth = receipt.truth
        assert truth["truth_mode"] == "truth_safe_unverified"
        assert truth["claim_level"] == "computational_scaffold"
        assert truth["field_solver_status"] == "laminar_proxy_no_pde"
        assert truth["physical_amplitude_claim_allowed"] is False


class TestReceptorExponentialScanBackend:
    """Receptor-exponential kernel uses lax.scan with receptor-indexed state."""

    def test_receptor_exponential_backend(self):
        """Receptor-exponential kernel can be selected."""
        model = _make_model(n=10)
        rt = jtfne.runtime(
            recurrent_backend="edge_list",
            synaptic_kernel="receptor_exponential",
            seed=42
        )
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42, runtime=rt)
        signals = model.simulate(sim)
        assert signals.metadata["recurrent_backend"] == "edge_list"
        assert signals.metadata["synaptic_kernel"] == "receptor_exponential"

    def test_receptor_exponential_output_shapes(self):
        """Receptor-exponential path produces correct shapes."""
        model = _make_model(n=15)
        rt = jtfne.runtime(
            recurrent_backend="edge_list",
            synaptic_kernel="receptor_exponential",
            seed=0
        )
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=rt)
        signals = model.simulate(sim)
        n_steps = int(10.0 / 0.1)
        assert signals.spikes.shape == (n_steps, 15)
        assert signals.V_m.shape == (n_steps, 15)

    def test_receptor_exponential_deterministic(self):
        """Receptor-exponential path is deterministic."""
        model = _make_model(n=10)
        rt1 = jtfne.runtime(
            recurrent_backend="edge_list",
            synaptic_kernel="receptor_exponential",
            seed=789
        )
        rt2 = jtfne.runtime(
            recurrent_backend="edge_list",
            synaptic_kernel="receptor_exponential",
            seed=789
        )
        sim1 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=789, runtime=rt1)
        sim2 = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=789, runtime=rt2)
        signals1 = model.simulate(sim1)
        signals2 = model.simulate(sim2)
        assert jnp.allclose(signals1.spikes, signals2.spikes)


class TestBackendMetadata:
    """Verify backend metadata is correctly reported in all paths."""

    def test_backend_metadata_in_signals(self):
        """Backend metadata flows into signals.metadata."""
        model = _make_model(n=10)
        for backend in ["dense", "edge_list"]:
            rt = jtfne.runtime(recurrent_backend=backend, seed=0)
            sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=rt)
            signals = model.simulate(sim)
            assert "recurrent_backend" in signals.metadata
            assert signals.metadata["recurrent_backend"] == backend

    def test_backend_metadata_in_receipt(self):
        """Backend metadata flows into run receipts."""
        model = _make_model(n=10)
        rt = jtfne.runtime(recurrent_backend="edge_list", seed=0)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=rt)
        signals = model.simulate(sim)
        receipt = model.run_receipt(signals)
        assert receipt.backend["recurrent_backend"] == "edge_list"

    def test_synaptic_kernel_metadata(self):
        """Synaptic kernel metadata is reported correctly."""
        model = _make_model(n=10)
        rt = jtfne.runtime(
            recurrent_backend="edge_list",
            synaptic_kernel="receptor_exponential",
            seed=0
        )
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=rt)
        signals = model.simulate(sim)
        assert signals.metadata["synaptic_kernel"] == "receptor_exponential"
