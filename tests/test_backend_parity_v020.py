"""Tests for v0.2.0 dense and edge-list backend parity validation.

This phase validates deterministic computational consistency and metadata
preservation across dense and edge-list recurrent backends under controlled
matched-graph conditions. Parity means same neurons, seed, adjacency/weights,
external drive/noise policy, then compare outputs within declared tolerances.

Parity is computational correctness evidence only; no biological calibration.
"""

import json

import jax
import jax.numpy as jnp
import pytest

import jaxfne
from jaxfne.emitters import make_edge_list_from_dense


def _cfg(n=8):
    """Minimal configuration."""
    return (
        jaxfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def _model_and_signals(n=8, seed=0, backend="dense"):
    """Construct model with specified backend and run simulation."""
    cfg = _cfg(n=n)
    model = jaxfne.construct(cfg)
    # Pass backend through RuntimeConfig via Simulation
    sim = jaxfne.Simulation(
        duration_ms=10.0,
        dt_ms=0.5,
        seed=seed,
        runtime=jaxfne.runtime(recurrent_backend=backend)
    )
    signals = model.simulate(sim)
    return model, signals


class TestDenseBackendDeterminism:
    """Dense backend deterministic behavior validation."""

    def test_dense_deterministic_same_seed(self):
        """Dense backend with same seed must produce identical trajectories."""
        model1, signals1 = _model_and_signals(n=4, seed=42, backend="dense")
        model2, signals2 = _model_and_signals(n=4, seed=42, backend="dense")

        # Voltages and spikes must be bitwise identical
        assert jnp.allclose(signals1.V_m, signals2.V_m, rtol=0, atol=0)
        assert jnp.allclose(signals1.spikes, signals2.spikes, rtol=0, atol=0)

    def test_dense_stochastic_different_seeds(self):
        """Dense backend with different seeds must produce measurable differences."""
        model1, signals1 = _model_and_signals(n=4, seed=42, backend="dense")
        model2, signals2 = _model_and_signals(n=4, seed=43, backend="dense")

        # Trajectories should differ
        diff = jnp.max(jnp.abs(signals1.V_m - signals2.V_m))
        assert float(diff) > 0.0

    def test_dense_reported_in_metadata(self):
        """Dense backend must be reported in signals metadata."""
        model, signals = _model_and_signals(n=4, backend="dense")
        assert signals.metadata.get("recurrent_backend") == "dense"

    def test_dense_output_shapes_consistent(self):
        """Dense backend output shapes must be consistent."""
        model, signals = _model_and_signals(n=8, backend="dense")
        assert signals.V_m.shape == signals.spikes.shape
        if signals.sources is not None:
            assert signals.sources.shape[0] == signals.V_m.shape[0]


class TestEdgeListBackendDeterminism:
    """Edge-list backend deterministic behavior validation."""

    def test_edge_list_deterministic_same_seed(self):
        """Edge-list backend with same seed must produce identical trajectories."""
        model1, signals1 = _model_and_signals(n=4, seed=42, backend="edge_list")
        model2, signals2 = _model_and_signals(n=4, seed=42, backend="edge_list")

        # Voltages and spikes must be bitwise identical
        assert jnp.allclose(signals1.V_m, signals2.V_m, rtol=0, atol=0)
        assert jnp.allclose(signals1.spikes, signals2.spikes, rtol=0, atol=0)

    def test_edge_list_stochastic_different_seeds(self):
        """Edge-list backend with different seeds must produce measurable differences."""
        model1, signals1 = _model_and_signals(n=4, seed=42, backend="edge_list")
        model2, signals2 = _model_and_signals(n=4, seed=43, backend="edge_list")

        # Trajectories should differ
        diff = jnp.max(jnp.abs(signals1.V_m - signals2.V_m))
        assert float(diff) > 0.0

    def test_edge_list_reported_in_metadata(self):
        """Edge-list backend must be reported in signals metadata."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        assert signals.metadata.get("recurrent_backend") == "edge_list"

    def test_edge_list_output_shapes_consistent(self):
        """Edge-list backend output shapes must be consistent."""
        model, signals = _model_and_signals(n=8, backend="edge_list")
        assert signals.V_m.shape == signals.spikes.shape
        if signals.sources is not None:
            assert signals.sources.shape[0] == signals.V_m.shape[0]


class TestBackendMetadataPreservation:
    """Metadata preservation across backends."""

    def test_dense_source_calibration_status_preserved(self):
        """Dense backend must preserve uncalibrated source status."""
        model, signals = _model_and_signals(n=4, backend="dense")
        assert signals.metadata["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        sb = signals.metadata.get("source_bookkeeping", {})
        assert sb.get("source_calibration_status") == "uncalibrated_izhikevich_native_current"

    def test_edge_list_source_calibration_status_preserved(self):
        """Edge-list backend must preserve uncalibrated source status."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        assert signals.metadata["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        sb = signals.metadata.get("source_bookkeeping", {})
        assert sb.get("source_calibration_status") == "uncalibrated_izhikevich_native_current"

    def test_dense_physical_amplitude_claim_false(self):
        """Dense backend must report physical_amplitude_claim_allowed=False."""
        model, signals = _model_and_signals(n=4, backend="dense")
        sb = signals.metadata.get("source_bookkeeping", {})
        assert sb.get("physical_amplitude_claim_allowed") is False

    def test_edge_list_physical_amplitude_claim_false(self):
        """Edge-list backend must report physical_amplitude_claim_allowed=False."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        sb = signals.metadata.get("source_bookkeeping", {})
        assert sb.get("physical_amplitude_claim_allowed") is False

    def test_dense_field_claim_level_preserved(self):
        """Dense backend must preserve field_claim_level=proxy_readout_only."""
        model, signals = _model_and_signals(n=4, backend="dense")
        assert signals.metadata.get("field_claim_level") == "proxy_readout_only"

    def test_edge_list_field_claim_level_preserved(self):
        """Edge-list backend must preserve field_claim_level=proxy_readout_only."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        assert signals.metadata.get("field_claim_level") == "proxy_readout_only"


class TestBackendTruthGatesPreserved:
    """Truth gates must remain frozen across both backends."""

    def test_dense_truth_gates_frozen(self):
        """Dense backend must preserve all 6 frozen truth gates."""
        model, signals = _model_and_signals(n=4, backend="dense")
        manifest = model.manifest(signals=signals)

        truth_gates = {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
        }
        for key, expected_value in truth_gates.items():
            assert manifest.get(key) == expected_value, f"Gate {key} mismatch in dense backend"

    def test_edge_list_truth_gates_frozen(self):
        """Edge-list backend must preserve all 6 frozen truth gates."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        manifest = model.manifest(signals=signals)

        truth_gates = {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
        }
        for key, expected_value in truth_gates.items():
            assert manifest.get(key) == expected_value, f"Gate {key} mismatch in edge_list backend"


class TestBackendManifestJSONSafety:
    """Manifest JSON safety for both backends."""

    def test_dense_manifest_json_safe(self):
        """Dense backend manifest must be JSON-safe."""
        model, signals = _model_and_signals(n=4, backend="dense")
        manifest = model.manifest(signals=signals)
        json_str = json.dumps(manifest, allow_nan=False)
        assert isinstance(json_str, str)

    def test_edge_list_manifest_json_safe(self):
        """Edge-list backend manifest must be JSON-safe."""
        model, signals = _model_and_signals(n=4, backend="edge_list")
        manifest = model.manifest(signals=signals)
        json_str = json.dumps(manifest, allow_nan=False)
        assert isinstance(json_str, str)


class TestMatchedGraphParity:
    """Backend parity under matched-graph conditions."""

    def test_matched_graph_shapes_equal(self):
        """Dense and edge-list with matched graph must have equal output shapes."""
        model_dense, signals_dense = _model_and_signals(n=8, seed=42, backend="dense")
        model_edge, signals_edge = _model_and_signals(n=8, seed=42, backend="edge_list")

        # Shapes must match
        assert signals_dense.V_m.shape == signals_edge.V_m.shape
        assert signals_dense.spikes.shape == signals_edge.spikes.shape

    def test_matched_graph_outputs_finite(self):
        """Dense and edge-list outputs must both be finite."""
        model_dense, signals_dense = _model_and_signals(n=8, seed=42, backend="dense")
        model_edge, signals_edge = _model_and_signals(n=8, seed=42, backend="edge_list")

        assert bool(jnp.all(jnp.isfinite(signals_dense.V_m)))
        assert bool(jnp.all(jnp.isfinite(signals_edge.V_m)))
        assert bool(jnp.all(jnp.isfinite(signals_dense.spikes)))
        assert bool(jnp.all(jnp.isfinite(signals_edge.spikes)))

    def test_matched_graph_spike_count_parity(self):
        """Dense and edge-list with matched graph should have similar spike counts."""
        model_dense, signals_dense = _model_and_signals(n=8, seed=42, backend="dense")
        model_edge, signals_edge = _model_and_signals(n=8, seed=42, backend="edge_list")

        # Total spike count should be close (may differ due to exponential decay in edge-list)
        dense_spike_count = float(jnp.sum(signals_dense.spikes))
        edge_spike_count = float(jnp.sum(signals_edge.spikes))

        # Allow 10% difference due to synaptic kernel differences
        tolerance = 0.1 * max(dense_spike_count, edge_spike_count)
        assert abs(dense_spike_count - edge_spike_count) <= tolerance

    def test_matched_graph_mean_voltage_parity(self):
        """Dense and edge-list with matched graph should have similar mean voltages."""
        model_dense, signals_dense = _model_and_signals(n=8, seed=42, backend="dense")
        model_edge, signals_edge = _model_and_signals(n=8, seed=42, backend="edge_list")

        mean_voltage_dense = float(jnp.mean(signals_dense.V_m))
        mean_voltage_edge = float(jnp.mean(signals_edge.V_m))

        # Voltages should be close (within a few mV)
        assert abs(mean_voltage_dense - mean_voltage_edge) < 10.0


class TestEdgeListConstruction:
    """Edge-list construction from dense weight matrix."""

    def test_edge_list_from_dense_preserves_nonzeros(self):
        """Converting dense matrix to edge-list must preserve nonzero entries."""
        W = jnp.array([[1.0, 0.0, -2.0], [0.0, 1.5, 0.0], [-0.5, 0.0, 1.0]])
        edges = make_edge_list_from_dense(W)

        # Should have 5 nonzero weights (1.0, -2.0, 1.5, -0.5, 1.0)
        assert edges.n_edges == 5

    def test_edge_list_receptor_index_sign_consistency(self):
        """Edge-list receptor_index must match weight sign (positive=AMPA, negative=GABA)."""
        W = jnp.array([[1.0, -2.0], [3.0, -1.0]])
        edges = make_edge_list_from_dense(W)

        # Positive weights should have receptor_index=0 (AMPA-like)
        # Negative weights should have receptor_index=1 (GABA-like)
        for i, weight in enumerate(edges.weight):
            if float(weight) > 0:
                assert int(edges.receptor_index[i]) == 0
            else:
                assert int(edges.receptor_index[i]) == 1

    def test_edge_list_tau_assignment(self):
        """Edge-list must assign tau_ms based on receptor type."""
        W = jnp.array([[1.0, -2.0], [3.0, -1.0]])
        edges = make_edge_list_from_dense(W)

        # AMPA-like (receptor_index=0) should have tau_ms≈2.0
        # GABA-like (receptor_index=1) should have tau_ms≈5.0
        for i, receptor_idx in enumerate(edges.receptor_index):
            tau = float(edges.tau_ms[i])
            if int(receptor_idx) == 0:
                assert abs(tau - 2.0) < 0.1
            else:
                assert abs(tau - 5.0) < 0.1


class TestFieldAdmissibilityBothBackends:
    """Field admissibility metadata must be preserved in both backends."""

    def test_dense_field_admissibility_in_manifest(self):
        """Dense backend manifest must include field_admissibility."""
        model, signals = _model_and_signals(n=8, backend="dense")
        manifest = model.manifest(signals=signals)
        assert "field_admissibility" in manifest["backend_metadata"]
        field_adm = manifest["backend_metadata"]["field_admissibility"]
        assert field_adm["physical_amplitude_claim_allowed"] is False

    def test_edge_list_field_admissibility_in_manifest(self):
        """Edge-list backend manifest must include field_admissibility."""
        model, signals = _model_and_signals(n=8, backend="edge_list")
        manifest = model.manifest(signals=signals)
        assert "field_admissibility" in manifest["backend_metadata"]
        field_adm = manifest["backend_metadata"]["field_admissibility"]
        assert field_adm["physical_amplitude_claim_allowed"] is False
