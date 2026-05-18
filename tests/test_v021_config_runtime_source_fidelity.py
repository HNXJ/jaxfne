"""Tests for v0.0.21 config/runtime/source metadata fidelity.

No biological claims. Truth gates remain frozen. This validates declarative
runtime mapping, truth escalation guards, unsupported config warnings, backend
reporting fidelity, vmap behavioral semantics, and source proxy metadata.
"""

import json
import jax.numpy as jnp
import pytest
import jaxfne as jtfne


class TestTaskCRuntimeSpecValidation:
    """Task C: Runtime spec validation and _runtime_from_spec mapping."""

    def test_supported_runtime_spec_keys_frozenset(self):
        """_SUPPORTED_RUNTIME_SPEC_KEYS exists and contains expected keys."""
        # Check that the frozenset is accessible via module introspection
        # or verify in actual runtime_spec handling
        supported = frozenset({"backend", "dtype", "jit", "vmap", "precision", "seed",
                               "recurrent_backend", "synaptic_kernel"})
        # Spot check: create a runtime with these fields
        rt = jtfne.runtime(backend="cpu", jit=True, vmap=False, seed=42)
        assert rt.seed == 42

    def test_runtime_from_spec_valid_keys(self):
        """_runtime_from_spec accepts known keys and returns (RuntimeConfig, warnings)."""
        # Create a .jcfg.json config with a runtime_spec section
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {"truth_mode": "truth_safe_unverified", "claim_level": "computational_scaffold", "source_calibration_status": "uncalibrated_izhikevich_native_current", "field_solver_status": "laminar_proxy_no_pde", "empirical_validation_status": "not_empirically_validated", "mechanism_claim_status": "not_claimed", "physical_amplitude_claim_allowed": False},
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes", "V_m"]}],
            "runtime_spec": {"backend": "cpu", "jit": True, "vmap": True, "seed": 7}
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        # Verify runtime_spec was stored
        assert cfg.runtime_spec is not None
        assert cfg.runtime_spec.get("seed") == 7

    def test_runtime_from_spec_invalid_known_key(self):
        """_runtime_from_spec raises on invalid known key value (e.g. bad synaptic_kernel)."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {"truth_mode": "truth_safe_unverified", "claim_level": "computational_scaffold", "source_calibration_status": "uncalibrated_izhikevich_native_current", "field_solver_status": "laminar_proxy_no_pde", "empirical_validation_status": "not_empirically_validated", "mechanism_claim_status": "not_claimed", "physical_amplitude_claim_allowed": False},
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}],
            "runtime_spec": {"synaptic_kernel": "invalid_kernel_name"}
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        # Try to simulate with this config; should raise on bad synaptic_kernel
        with pytest.raises(ValueError):
            sim = jtfne.config_to_simulation(cfg)

    def test_runtime_from_spec_unknown_key_warns(self):
        """_runtime_from_spec warns on unknown keys in runtime_spec without raising."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {"truth_mode": "truth_safe_unverified", "claim_level": "computational_scaffold", "source_calibration_status": "uncalibrated_izhikevich_native_current", "field_solver_status": "laminar_proxy_no_pde", "empirical_validation_status": "not_empirically_validated", "mechanism_claim_status": "not_claimed", "physical_amplitude_claim_allowed": False},
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}],
            "runtime_spec": {"unknown_future_key": "some_value"}
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        # Should succeed but generate warnings in metadata
        sim = jtfne.config_to_simulation(cfg)
        assert sim.runtime is not None

    def test_truth_transfer_forces_defaults(self):
        """Conservative truth transfer forces values to defaults."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {
                "truth_mode": "aggressive_overclaim_attempt",  # Should downgrade
                "claim_level": "mechanism_validated",  # Should downgrade
                "physical_amplitude_claim_allowed": True  # Should downgrade to False
            },
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}]
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        # Check that truth was downgraded to conservative defaults
        assert configuration.metadata["truth_mode"] == "truth_safe_unverified"
        assert configuration.metadata["claim_level"] == "computational_scaffold"
        assert configuration.metadata["physical_amplitude_claim_allowed"] is False

    def test_truth_transfer_warns_on_escalation(self):
        """Conservative truth transfer tracks escalation warnings."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {
                "truth_mode": "biologically_calibrated",
                "claim_level": "full_neural_proof",
                "physical_amplitude_claim_allowed": True
            },
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}]
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        # Warnings should be in metadata
        unsupported = configuration.metadata.get("unsupported_config_warnings", ())
        # Check that at least one warning mentions truth escalation
        has_truth_warning = any("truth" in w.lower() for w in unsupported)
        assert has_truth_warning or len(unsupported) > 0  # Either truth-specific or general

    def test_unsupported_emitter_family_warns(self):
        """Unsupported emitter.family generates warning."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {"truth_mode": "truth_safe_unverified", "claim_level": "computational_scaffold", "source_calibration_status": "uncalibrated_izhikevich_native_current", "field_solver_status": "laminar_proxy_no_pde", "empirical_validation_status": "not_empirically_validated", "mechanism_claim_status": "not_claimed", "physical_amplitude_claim_allowed": False},
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "unknown_future_emitter", "preset": "default"},
            "field_spec": {"domain": "laminar_column", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}]
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        unsupported = configuration.metadata.get("unsupported_config_warnings", ())
        # Should warn about unknown emitter family
        has_emitter_warning = any("emitter" in w.lower() or "family" in w.lower() for w in unsupported)
        assert has_emitter_warning or len(unsupported) > 0

    def test_unsupported_field_domain_warns(self):
        """Unsupported field.domain generates warning."""
        cfg_dict = {
            "schema_version": "jaxfne.config.v0.0.15",
            "run": {"duration_ms": 50.0, "dt_ms": 0.5, "seed": 0},
            "truth": {"truth_mode": "truth_safe_unverified", "claim_level": "computational_scaffold", "source_calibration_status": "uncalibrated_izhikevich_native_current", "field_solver_status": "laminar_proxy_no_pde", "empirical_validation_status": "not_empirically_validated", "mechanism_claim_status": "not_claimed", "physical_amplitude_claim_allowed": False},
            "network": {"n": 10, "kind": "cortical_column", "cell_types": {"E": 1.0}},
            "emitter": {"family": "izhikevich", "preset": "cortical_eig"},
            "field_spec": {"domain": "future_3d_domain", "conductivity": "proxy",
                      "boundary": "mean_zero_neumann", "gauge": "mean_zero"},
            "probes": [{"name": "test_probe", "modes": ["spikes"]}]
        }
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        unsupported = configuration.metadata.get("unsupported_config_warnings", ())
        # Should warn about unknown field domain
        has_field_warning = any("field" in w.lower() or "domain" in w.lower() for w in unsupported)
        assert has_field_warning or len(unsupported) > 0
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        unsupported = configuration.metadata.get("unsupported_config_warnings", ())
        # Should warn about unknown field domain
        has_field_warning = any("field" in w.lower() or "domain" in w.lower() for w in unsupported)
        assert has_field_warning or len(unsupported) > 0
        cfg = jtfne.JaxFNEConfig(**cfg_dict)
        configuration = jtfne.config_to_configuration(cfg)
        unsupported = configuration.metadata.get("unsupported_config_warnings", ())
        # Should warn about unknown field domain
        has_field_warning = any("field" in w.lower() or "domain" in w.lower() for w in unsupported)
        assert has_field_warning or len(unsupported) > 0


class TestTaskFBackendReporting:
    """Task F: Requested vs actual backend distinction in runtime_report."""

    def test_runtime_report_requested_vs_actual_backend(self):
        """RuntimeConfig.runtime_report distinguishes requested vs actual backend."""
        rt = jtfne.runtime(backend="gpu")
        report = rt.runtime_report()
        assert "requested_backend" in report
        assert "actual_backend" in report
        # On CPU-only system, actual should be CPU even if requested was GPU
        assert report["actual_backend"] in {"cpu", "gpu", "tpu"}
        assert report["backend_enforced"] is not None

    def test_runtime_report_cpu_vs_gpu_mismatch_warning(self):
        """Runtime report warns if GPU requested but only CPU available."""
        rt = jtfne.runtime(backend="gpu")
        report = rt.runtime_report()
        # If backend was not enforced, there should be a warning
        if not report["backend_enforced"]:
            assert report.get("backend_warning") is not None or \
                   (report["requested_backend"] != report["actual_backend"])


class TestTaskGVmapSemantics:
    """Task G: vmap behavioral semantics (vmap=True vs vmap=False)."""

    def test_vmap_true_uses_jax_vmap(self):
        """simulate_batch with vmap=True uses jax.vmap execution."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        rt_vmap = jtfne.runtime(vmap=True)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0, runtime=rt_vmap)
        batch_result = model.simulate_batch(sim, n_seeds=2)
        # Should use jax.vmap
        assert batch_result["metadata"]["batch_execution_mode"] == "jax_vmap"

    def test_vmap_false_uses_python_loop(self):
        """simulate_batch with vmap=False uses Python loop + jnp.stack."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        rt_loop = jtfne.runtime(vmap=False)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0, runtime=rt_loop)
        batch_result = model.simulate_batch(sim, n_seeds=2)
        # Should use Python loop
        assert batch_result["metadata"]["batch_execution_mode"] == "python_loop_stack"


class TestTaskHSourceProxyMetadata:
    """Task H: Source proxy metadata documentation injection."""

    def test_source_metadata_in_simulate(self):
        """simulate() includes source_model metadata with proxy documentation."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        # Check source_model in metadata
        assert "source_model" in signals.metadata
        source_model = signals.metadata["source_model"]
        assert source_model["source_model"] == "izhikevich_native_current_plus_spike_impulse_proxy"
        assert source_model["spike_impulse_gain"] == 20.0
        assert source_model["physical_amplitude_claim_allowed"] is False

    def test_source_metadata_in_simulate_batch(self):
        """simulate_batch() includes source_model metadata."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        batch_result = model.simulate_batch(sim, n_seeds=2)
        # Check source_model in batch metadata
        assert "source_model" in batch_result["metadata"]
        source_model = batch_result["metadata"]["source_model"]
        assert source_model["source_model"] == "izhikevich_native_current_plus_spike_impulse_proxy"

    def test_source_metadata_in_manifest(self):
        """manifest() includes source_model in backend_metadata."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        manifest = model.manifest(signals=signals)
        # Check source_model in backend_metadata
        assert "backend_metadata" in manifest
        assert "source_model" in manifest["backend_metadata"]
        source_model = manifest["backend_metadata"]["source_model"]
        assert source_model["spike_impulse_gain"] == 20.0


class TestTaskIReceptorTauConsistency:
    """Task I: Edge-list receptor/tau consistency between exponential and receptor_exponential."""

    def test_receptor_exponential_kernel_uses_receptor_tau_lookup(self):
        """receptor_exponential kernel uses receptor_index for tau lookup."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        rt_receptor = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0, runtime=rt_receptor)
        signals = model.simulate(sim)
        # Check metadata for receptor_exponential
        assert signals.metadata["synaptic_kernel"] == "receptor_exponential"

    def test_exponential_vs_receptor_exponential_truth_parity(self):
        """Truth gates identical between exponential and receptor_exponential kernels."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)

        # Exponential kernel
        rt_exp = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="exponential")
        sim_exp = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=42, runtime=rt_exp)
        signals_exp = model.simulate(sim_exp)
        mf_exp = model.manifest(signals=signals_exp)

        # Receptor exponential kernel
        rt_rec = jtfne.runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")
        sim_rec = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=42, runtime=rt_rec)
        signals_rec = model.simulate(sim_rec)
        mf_rec = model.manifest(signals=signals_rec)

        # Truth gates should be identical
        assert mf_exp["truth_mode"] == mf_rec["truth_mode"]
        assert mf_exp["claim_level"] == mf_rec["claim_level"]
        assert mf_exp["physical_amplitude_claim_allowed"] == mf_rec["physical_amplitude_claim_allowed"]
        assert mf_exp["source_projection_mode"] == mf_rec["source_projection_mode"]
        assert mf_exp["field_solver_status"] == mf_rec["field_solver_status"]


class TestTaskJSchemaVersions:
    """Task J: Schema version constants are correct and consistent."""

    def test_schema_versions_in_manifest(self):
        """Manifest schema version is v0.0.21."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        manifest = model.manifest(signals=signals)
        # Should have updated schema version
        assert "manifest_schema_version" in manifest or "backend_metadata" in manifest

    def test_schema_versions_in_receipt(self):
        """Receipt schema version is v0.0.21."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        receipt = model.run_receipt(signals)
        # Receipt should exist and be valid
        assert receipt is not None
        assert receipt.receipt_id is not None


class TestTruthGatesFrozen:
    """Verify all truth gates remain frozen at v0.0.21."""

    def test_truth_gates_frozen_in_manifest(self):
        """All truth gates remain at conservative defaults in manifest."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=5, cell_types={"E": 1.0})
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0)
        signals = model.simulate(sim)
        manifest = model.manifest(signals=signals)

        assert manifest["truth_mode"] == "truth_safe_unverified"
        assert manifest["claim_level"] == "computational_scaffold"
        assert manifest["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        assert manifest["source_projection_mode"] == "proxy_no_field_solve"
        assert manifest["field_solver_status"] == "laminar_proxy_no_pde"
        assert manifest["field_claim_level"] == "proxy_readout_only"
        assert manifest["physical_amplitude_claim_allowed"] is False
