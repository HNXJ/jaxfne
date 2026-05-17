import json

import jax
import jax.numpy as jnp

import jaxfne as jtfne


def _cfg(n=12):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_minimal_api_smoke():
    cfg = _cfg(12)
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    assert signals.V_m.shape[1] == 12
    assert "spikes" in readout
    assert "CSD" in readout
    assert "csd_proxy" in readout
    manifest = model.manifest(signals, readout)
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["source_field_status"]["physical_amplitude_claim_allowed"] is False


def test_signals_vs_signal_alias():
    assert hasattr(jtfne, "Signals")
    assert hasattr(jtfne, "Signal")
    assert jtfne.Signal is jtfne.Signals


def test_probe_and_record_equivalence():
    model = jtfne.construct(_cfg(8))
    signals = model.simulate(jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=42))
    modes = ["spikes", "V_m"]
    probe_out = model.probe(signals, modes=modes)
    record_out = model.record(signals, modes=modes)
    assert set(probe_out.keys()) == set(record_out.keys())
    for key in probe_out:
        if hasattr(probe_out[key], "shape"):
            assert jnp.all(probe_out[key] == record_out[key])
        else:
            assert probe_out[key] == record_out[key]


def test_metadata_gates_defaults():
    meta = jtfne.configuration().metadata
    assert meta["truth_mode"] == "truth_safe_unverified"
    assert meta["claim_level"] == "computational_scaffold"
    assert meta["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    assert meta["source_projection_mode"] == "proxy_no_field_solve"
    assert meta["boundary_condition"] == "mean_zero_neumann"
    assert meta["gauge"] == "mean_zero"
    assert meta["csd_sign_convention"] == "proxy_positive_equals_extracellular_source_like"
    assert meta["field_solver_status"] == "laminar_proxy_no_pde"
    assert meta["manifest_schema_version"] == "0.0.4"
    assert isinstance(meta["operator_status"], dict)


def test_runtime_config_and_dtype_float32():
    rt = jtfne.runtime(dtype="float32", backend="auto", seed=42, n_steps=100)
    report = rt.runtime_report()
    assert report["requested_dtype"] == "float32"
    assert report["actual_dtype"] == "float32"
    assert "jax_version" in report
    model = jtfne.construct(_cfg(6))
    signals = model.simulate(jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0, runtime=rt))
    assert signals.V_m.dtype == jnp.float32
    assert signals.time_ms.dtype == jnp.float32
    assert signals.sources.dtype == jnp.float32


def test_runtime_float64_truthful_behavior():
    rt = jtfne.runtime(dtype="float64")
    model = jtfne.construct(_cfg(5))
    signals = model.simulate(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=0, runtime=rt))
    if bool(jax.config.read("jax_enable_x64")):
        assert signals.V_m.dtype == jnp.float64
    else:
        assert signals.V_m.dtype == jnp.float32
    manifest = model.manifest(signals)
    assert manifest["runtime"]["requested_dtype"] == "float64"
    assert manifest["runtime"]["actual_dtype"] in {"float32", "float64"}


def test_source_probe_invariants():
    model = jtfne.construct(_cfg(10))
    signals = model.simulate(jtfne.simulation(duration_ms=6.0, dt_ms=0.1, seed=1))
    field = signals.field
    assert field is not None
    diag = field.diagnostics
    assert diag["source_shape"] == (60, 10)
    assert diag["positions_shape"] == (10, 3)
    assert diag["source_proxy_shape"] == (60, 16)
    assert diag["phi_e_proxy_shape"] == (60, 16)
    assert diag["csd_proxy_shape"] == (60, 16)
    assert diag["lfp_proxy_shape"] == (60, 16)
    assert diag["finite_source_proxy"] is True
    assert diag["finite_csd_proxy"] is True
    assert diag["kernel_row_sum_max_abs_error"] < 1e-5


def test_source_field_truth_contract():
    model = jtfne.construct(_cfg(8))
    signals = model.simulate(jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=42))
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP", "J_e"])
    manifest = model.manifest(signals, readout)
    status = manifest["source_field_status"]
    assert status["field_solver_status"] == "laminar_proxy_no_pde"
    assert status["field_claim_level"] == "proxy_readout_only"
    assert status["physical_amplitude_claim_allowed"] is False
    assert status["is_proxy"] is True
    assert "J_e_status" in readout
    assert "J_e" not in readout
    assert any("J_e_not_computed" in w for w in status["warnings"])


def test_manifest_json_safe():
    model = jtfne.construct(_cfg(6))
    signals = model.simulate(jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=0))
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD"])
    manifest = model.manifest(signals, readout)
    json_str = json.dumps(manifest, allow_nan=False)
    assert isinstance(json_str, str)
    assert manifest["manifest_schema_version"] == "0.0.4"
    assert "operator_status" in manifest
    assert "source_field_status" in manifest
    assert "runtime" in manifest


def test_optional_dependency_guards():
    from jaxfne.bridges import require_jaxley
    from jaxfne.optim import require_optax

    try:
        require_jaxley()
    except ImportError as e:
        assert "optional dependency 'jaxley'" in str(e)
        assert "pip install" in str(e)

    try:
        require_optax()
    except ImportError as e:
        assert "optional dependency 'optax'" in str(e)
        assert "pip install" in str(e)


def test_version():
    assert jtfne.__version__ >= "0.0.7"
