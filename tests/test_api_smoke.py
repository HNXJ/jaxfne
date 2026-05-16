import json

import jaxfne as jtfne


def test_minimal_api_smoke():
    """Baseline smoke test: configuration -> construct -> simulate -> probe -> manifest."""
    cfg = jtfne.configuration()
    cfg = cfg.network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
    cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
    cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    assert signals.V_m.shape[1] == 12
    assert "spikes" in readout
    assert "CSD" in readout
    assert model.manifest(signals)["truth_mode"] == "truth_safe_unverified"


def test_signals_vs_signal_alias():
    """Test that Signals is canonical and Signal is backwards-compat alias."""
    assert hasattr(jtfne, "Signals")
    assert hasattr(jtfne, "Signal")
    # Signal should be the same class as Signals
    assert jtfne.Signal is jtfne.Signals


def test_probe_and_record_equivalence():
    """Test that probe() and record() return identical results."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column")
        .probe(name="test_probe", modes=["spikes", "V_m", "CSD"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=42)
    signals = model.simulate(sim)

    modes = ["spikes", "V_m"]
    probe_out = model.probe(signals, modes=modes)
    record_out = model.record(signals, modes=modes)

    assert set(probe_out.keys()) == set(record_out.keys())
    for key in probe_out.keys():
        assert (probe_out[key] == record_out[key]).all(), f"Mismatch in key {key}"


def test_metadata_gates_defaults():
    """Test that Configuration has required metadata gates with correct defaults."""
    cfg = jtfne.configuration()
    meta = cfg.metadata

    # Required fields in v0.0.2
    assert meta.get("truth_mode") == "truth_safe_unverified"
    assert meta.get("claim_level") == "computational_scaffold"
    assert meta.get("source_calibration_status") == "uncalibrated_izhikevich_native_current"
    assert meta.get("source_projection_mode") == "proxy_no_field_solve"
    assert meta.get("boundary_condition") == "mean_zero_neumann"
    assert meta.get("gauge") == "mean_zero"
    assert meta.get("csd_sign_convention") == "proxy_positive_equals_extracellular_source_like"
    assert meta.get("field_solver_status") == "laminar_proxy_no_pde"
    assert meta.get("manifest_schema_version") == "0.0.2"
    assert isinstance(meta.get("operator_status"), dict)


def test_manifest_json_safe():
    """Test that manifest output is strictly JSON-safe with allow_nan=False."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=6)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column")
        .probe(name="test_probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    manifest = model.manifest(signals)

    # Must serialize with allow_nan=False (strict JSON)
    json_str = json.dumps(manifest, allow_nan=False)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Verify key fields are present
    assert "truth_mode" in manifest
    assert "operator_status" in manifest
    assert "manifest_schema_version" in manifest


def test_optional_dependency_guards():
    """Test that optional dependencies raise informative ImportError when not installed."""
    # Test Jaxley guard
    from jaxfne.bridges import require_jaxley

    try:
        require_jaxley()
        # If we get here, jaxley is installed (ok, skip assertion)
    except ImportError as e:
        assert "optional dependency 'jaxley'" in str(e)
        assert "pip install" in str(e)

    # Test Optax guard
    from jaxfne.optim import require_optax

    try:
        require_optax()
        # If we get here, optax is installed (ok, skip assertion)
    except ImportError as e:
        assert "optional dependency 'optax'" in str(e)
        assert "pip install" in str(e)


def test_version():
    """Test that package version is 0.0.2."""
    assert jtfne.__version__ == "0.0.2"
