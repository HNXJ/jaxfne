import json
import math

import jaxfne as jtfne
from jaxfne.bridges import BridgeSpec, JaxleyEmitterBridge
from jaxfne.optim import propose_blackbox_candidates


def _cfg(n=8):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_version_008_exports_dataset_spec():
    parts = [int(p) for p in jtfne.__version__.split(".")]
    assert parts >= [0, 0, 8]
    spec = jtfne.dataset_spec(name="nwb_spk", modality="SPK", source_format="npy")
    assert spec.validate()["valid"] is True
    assert spec.to_dict()["empirical_validation_status"] == "not_empirically_validated"
    json.dumps(spec.to_dict(), allow_nan=False)


def test_dataset_condition_map_and_quality_gates_json_safe():
    paradigm = jtfne.standard_visual_omission()
    condition_map = {c.name: list(c.condition_numbers) for c in paradigm.conditions}
    spec = (
        jtfne.dataset_spec(name="monkeylogic_bhv", modality="behavior", source_format="mat")
        .with_condition_map(condition_map)
        .with_quality_gate("correct_trials_only", {"TrialError": 0})
    )
    payload = spec.to_dict()
    assert payload["condition_map"]["AAAB"] == [1, 2]
    assert payload["quality_gates"]["correct_trials_only"]["TrialError"] == 0
    json.dumps(payload, allow_nan=False)


def test_blackbox_candidate_generation_is_deterministic_and_bounded():
    spec = jtfne.gsdr(alpha=0.5, exploration=0.01)
    a = propose_blackbox_candidates(spec, n_steps=5, seed=7, bounds=(0.25, 2.0))
    b = propose_blackbox_candidates(spec, n_steps=5, seed=7, bounds=(0.25, 2.0))
    assert a == b
    assert len(a) == 5
    assert all(0.25 <= x <= 2.0 for x in a)


def test_model_tune_runs_blackbox_loop_and_returns_json_safe_report():
    model = jtfne.construct(_cfg())
    obj = jtfne.objective().loss("rate", target=10.0, metric="spike_rate_hz_mean")
    tuned, report = model.tune(
        obj,
        optimizer=jtfne.random_search(),
        steps=3,
        seed=2,
        simulation=jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=2),
        parameter="source_scale",
        bounds=(0.5, 1.5),
    )
    assert report["tuning_status"] == "blackbox_loop_v0.0.6"
    assert len(report["candidate_history"]) == 3
    assert report["physical_amplitude_claim_allowed"] is False
    assert report["empirical_validation_status"] == "not_empirically_validated"
    assert isinstance(report["same_model_unchanged"], bool)
    assert tuned is not None
    json.dumps(report, allow_nan=False)


def test_model_tune_zero_steps_stays_metadata_only():
    model = jtfne.construct(_cfg())
    obj = jtfne.objective().gate("rate", threshold=1000.0, metric="spike_rate_hz_mean")
    same, report = model.tune(obj, optimizer="GSDR", steps=0)
    assert same is model
    assert report["tuning_status"] == "metadata_only_no_steps_requested"
    assert report["same_model_unchanged"] is True
    assert report["acceptance_decision"] == "REVISE"


def test_optax_path_remains_guarded_and_truth_safe():
    model = jtfne.construct(_cfg())
    obj = jtfne.objective().loss("rate", target=10.0, metric="spike_rate_hz_mean")
    same, report = model.tune(obj, optimizer=jtfne.optax_adam(), steps=2, strict=False)
    assert same is model
    assert report["tuning_status"] == "blocked_non_differentiable_path"
    assert "spiking_reset_not_differentiable_without_surrogate" in report["warnings"]


def test_bridge_specs_are_json_safe_and_do_not_allow_amplitude_claims():
    spec = BridgeSpec(name="test", backend="jaxley").to_dict()
    assert spec["physical_amplitude_claim_allowed"] is False
    bridge = JaxleyEmitterBridge(morphology="toy", mechanisms=("hh",)).to_spec().to_dict()
    assert bridge["backend"] == "jaxley"
    assert bridge["source_calibration_status"] == "uncalibrated_jaxley_bridge"
    json.dumps(bridge, allow_nan=False)


def test_manifest_accepts_dataset_and_tuning_loop_report():
    model = jtfne.construct(_cfg())
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=1)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    obj = jtfne.objective().loss("rate", target=10.0, metric="spike_rate_hz_mean")
    tuned, report = model.tune(obj, optimizer="AGSDR", steps=2, seed=1, simulation=sim)
    dataset = jtfne.dataset_spec(name="schema_only", modality="SPK", source_format="npy")
    manifest = model.manifest(
        signals=signals,
        readout=readout,
        objective={"name": obj.name, "losses": obj.losses},
        tuning=report,
        dataset=dataset.to_dict(),
    )
    assert manifest["dataset_claim_labels"]["dataset_status"] == "schema_only_no_data_loaded"
    assert manifest["v005_claim_labels"]["mechanism_claim_status"] == "not_claimed"
    assert manifest["source_field_status"]["physical_amplitude_claim_allowed"] is False
    json.dumps(manifest, allow_nan=False)


def test_model_and_signal_summary_json_safe():
    model = jtfne.construct(_cfg(n=6))
    signals = model.simulate(jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=3))
    ms = model.summary()
    ss = signals.summary()
    assert ms["field_claim_level"] == "proxy_readout_only"
    assert ss["field_claim_level"] == "proxy_readout_only"
    assert ss["n_steps"] == 40
    json.dumps(ms, allow_nan=False)
    json.dumps(ss, allow_nan=False)


def test_jit_opt_in_simulation_matches_shape_and_truth_gates():
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(jit=True, dtype="float32", seed=4)
    signals = model.simulate(jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=4, runtime=rt))
    assert signals.V_m.shape == (30, 5)
    assert signals.metadata["runtime"]["jit"] is True
    assert signals.metadata["field_claim_level"] == "proxy_readout_only"


def test_simulate_batch_uses_vmap_and_is_json_safe_metadata():
    model = jtfne.construct(_cfg(n=4))
    rt = jtfne.runtime(jit=True, vmap=True, dtype="float32", seed=5)
    batch = model.simulate_batch(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=5, runtime=rt), n_seeds=3)
    assert batch["V_m"].shape == (3, 20, 4)
    assert batch["spikes"].shape == (3, 20, 4)
    assert batch["metadata"]["batch_status"] == "vmap_seed_batch_v0.0.8"
    assert batch["metadata"]["physical_amplitude_claim_allowed"] is False
    json.dumps(batch["metadata"], allow_nan=False)


def test_surrogate_config_is_declaration_only_and_json_safe():
    s = jtfne.surrogate_config(method="straight_through", beta=8.0)
    payload = s.to_dict()
    assert payload["gradient_path_status"] == "declared_surrogate"
    assert payload["status"] == "declaration_only_v0.0.8"
    assert payload["mechanism_claim_status"] == "not_claimed"
    json.dumps(payload, allow_nan=False)


def test_runtime_report_backend_and_x64_fields_present():
    report = jtfne.runtime(backend="gpu", dtype="float32", jit=True, vmap=True).runtime_report()
    assert report["backend"] == "gpu"
    assert report["jit"] is True
    assert report["vmap"] is True
    assert "x64_enabled" in report
