"""
Test suite for v0.3.3 Two-Neuron E/I Multimodal Tutorial.

26 tests organized across 5 groups:
  Group 0: Existing 7 tests (tutorial import, run, manifest, JSON, figures, claims, metrics, network)
  Group A: Dynamic coupling function tests (5 new tests)
  Group B: Tutorial output validation tests (6 new tests)
  Group C: Claim gate immutability tests (3 new tests)
  Group D: Asset integrity tests (3 new tests)
  Group E: 8-figure completeness tests (2 new tests)
"""

import json
import pathlib
import subprocess
import sys

import pytest
import numpy as np


# ============================================================================
# Group 0: Existing tests (7 tests — preserved)
# ============================================================================

def test_v033_tutorial_import():
    """Test that the v0.3.3 tutorial script can be imported."""
    examples_path = pathlib.Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    try:
        import v033_two_neuron_ei_multimodal  # noqa: F401
    finally:
        sys.path.pop(0)


def test_v033_tutorial_runs():
    """Test that the v0.3.3 tutorial runs without exceptions."""
    examples_path = pathlib.Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    try:
        from v033_two_neuron_ei_multimodal import main

        result = main()

        assert isinstance(result, dict)
        assert "atlas_manifest" in result
        assert "manifest_path" in result
        assert "e_firing_rate_hz" in result
        assert "i_firing_rate_hz" in result
        assert "e_firing_rate_gate_pass" in result
        assert "i_firing_rate_gate_pass" in result
        assert "figures" in result

    finally:
        sys.path.pop(0)


def test_v033_manifest_json_valid():
    """Test that the generated manifest.json is valid and complete."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    if not manifest_path.exists():
        pytest.skip("Manifest not generated; tutorial may not have been run")

    with open(manifest_path) as f:
        manifest = json.load(f)

    required_keys = [
        "run_id",
        "tutorial_id",
        "jaxfne_version",
        "schema_version",
        "basis",
        "probe_report",
        "validation_report",
        "conservation_proxy_diagnostics",
    ]
    for key in required_keys:
        assert key in manifest, f"Missing required key: {key}"

    basis = manifest["basis"]
    assert basis["claim_level"] == "computational_scaffold"
    assert basis["field_claim_level"] == "proxy_readout_only"
    assert basis["physical_amplitude_claim_allowed"] is False
    assert basis["field_solver_status"] == "laminar_proxy_no_pde"

    probe_keys = set(manifest["probe_report"].keys())
    required_probes = {"spikes", "V_m", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy"}
    assert probe_keys == required_probes, f"Probe keys mismatch: {probe_keys} vs {required_probes}"

    val_report = manifest["validation_report"]
    assert "e_firing_rate_gate_2_25_hz" in val_report
    assert "i_firing_rate_gate_2_25_hz" in val_report
    assert "e_voltage_finite" in val_report
    assert "i_voltage_finite" in val_report
    assert "status" in val_report


def test_v033_json_files_allow_nan_false():
    """Test that all JSON files are strict (no NaN/Inf)."""
    json_files = [
        "outputs/v030_03_two_neuron_ei_multimodal/manifest.json",
        "outputs/v030_03_two_neuron_ei_multimodal/probe_report.json",
        "outputs/v030_03_two_neuron_ei_multimodal/validation_report.json",
        "outputs/v030_03_two_neuron_ei_multimodal/metrics.json",
        "outputs/v030_03_two_neuron_ei_multimodal/asset_hashes.json",
    ]

    for fpath_str in json_files:
        fpath = pathlib.Path(fpath_str)
        if not fpath.exists():
            pytest.skip(f"Output {fpath} not generated; tutorial may not have been run")

        with open(fpath) as f:
            data = json.load(f)

        strict_json = json.dumps(data, allow_nan=False)
        assert isinstance(strict_json, str)


def test_v033_figures_exist():
    """Test that expected figure files exist."""
    figures_dir = pathlib.Path("docs/tutorials_v030/_static/figures")

    required_figures = [
        "v0303_two_neuron_ei_voltage.png",
        "v0303_two_neuron_ei_raster.png",
    ]

    for fig_name in required_figures:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            pytest.skip(f"Figure {fig_name} not found; tutorial may not have been run")

        assert fig_path.stat().st_size > 0, f"Figure {fig_name} is empty"


def test_v033_no_forbidden_claims():
    """Test that the manifest does not contain forbidden claim phrases."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    if not manifest_path.exists():
        pytest.skip("Manifest not generated")

    with open(manifest_path) as f:
        manifest = json.load(f)

    forbidden_phrases = [
        "real EEG",
        "real MEG",
        "biological validation",
        "empirically validated",
        "proven mechanism",
        "biophysical accuracy",
        "solved PDE",
        "measured neuroscience",
    ]

    basis_str = json.dumps(manifest.get("basis", {}))
    for phrase in forbidden_phrases:
        assert phrase.lower() not in basis_str.lower(), f"Forbidden phrase '{phrase}' in basis claims"


def test_v033_metrics_are_finite():
    """Test that all numerical metrics are finite (no NaN/Inf)."""
    metrics_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/metrics.json")

    if not metrics_path.exists():
        pytest.skip("Metrics not generated")

    with open(metrics_path) as f:
        metrics = json.load(f)

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"


def test_v033_network_configuration():
    """Test that network configuration is correctly reflected in manifest."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    if not manifest_path.exists():
        pytest.skip("Manifest not generated")

    with open(manifest_path) as f:
        manifest = json.load(f)

    network = manifest.get("network", {})
    assert network.get("n_neurons") == 2
    assert network.get("e_fraction") == 0.5
    assert network.get("i_fraction") == 0.5
    assert network.get("e_index") == 0
    assert network.get("i_index") == 1

    assert "e_neuron" in manifest
    assert "i_neuron" in manifest

    e_neuron = manifest["e_neuron"]
    assert e_neuron["cell_type"] == "excitatory"
    assert e_neuron["index"] == 0

    i_neuron = manifest["i_neuron"]
    assert "inhibitory" in i_neuron["cell_type"]
    assert i_neuron["index"] == 1


# ============================================================================
# Group A: Dynamic coupling function tests (5 new tests)
# ============================================================================

def test_simulate_dynamic_ei_coupling_returns_four_arrays():
    """simulate_dynamic_ei_coupling returns a 4-tuple."""
    import jax
    import jax.numpy as jnp
    from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params

    params = izhikevich_eig_params(2, {"E": 0.5, "PV": 0.5})
    key = jax.random.PRNGKey(0)
    result = simulate_dynamic_ei_coupling(params, n_steps=100, dt_ms=0.1, key=key)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"


def test_simulate_dynamic_ei_coupling_shapes():
    """Output shapes are (n_steps, 2) for all four arrays."""
    import jax
    import jax.numpy as jnp
    from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params

    params = izhikevich_eig_params(2, {"E": 0.5, "PV": 0.5})
    key = jax.random.PRNGKey(1)
    n_steps = 200
    v, s, c, src = simulate_dynamic_ei_coupling(params, n_steps=n_steps, dt_ms=0.1, key=key)
    assert v.shape == (n_steps, 2)
    assert s.shape == (n_steps, 2)
    assert c.shape == (n_steps, 2)
    assert src.shape == (n_steps, 2)


def test_simulate_dynamic_ei_coupling_voltages_finite():
    """Voltages must be finite."""
    import jax
    import jax.numpy as jnp
    from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params

    params = izhikevich_eig_params(2, {"E": 0.5, "PV": 0.5})
    key = jax.random.PRNGKey(2)
    v, _, _, _ = simulate_dynamic_ei_coupling(params, n_steps=500, dt_ms=0.1, key=key)
    assert bool(jnp.all(jnp.isfinite(v))), "Voltages contain NaN/Inf"


def test_simulate_dynamic_ei_coupling_syn_currents_nonzero_with_spikes():
    """Synaptic currents become nonzero after E spikes (carry state is active)."""
    import jax
    import jax.numpy as jnp
    from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params

    params = izhikevich_eig_params(2, {"E": 0.5, "PV": 0.5})
    key = jax.random.PRNGKey(3)
    v, spikes, syn_c, src = simulate_dynamic_ei_coupling(
        params, n_steps=2000, dt_ms=0.1, key=key, g_ei=10.0, g_ie=3.0
    )
    assert float(jnp.max(jnp.abs(syn_c))) > 0.0, \
        "syn_currents all zero — likely carry state bug (syn_traces not in carry)"


def test_simulate_dynamic_ei_coupling_deterministic():
    """Same PRNG key produces identical outputs."""
    import jax
    import jax.numpy as jnp
    from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params

    params = izhikevich_eig_params(2, {"E": 0.5, "PV": 0.5})
    key = jax.random.PRNGKey(42)
    v1, s1, c1, src1 = simulate_dynamic_ei_coupling(params, n_steps=100, dt_ms=0.1, key=key)
    v2, s2, c2, src2 = simulate_dynamic_ei_coupling(params, n_steps=100, dt_ms=0.1, key=key)
    assert jnp.allclose(v1, v2), "Outputs not deterministic for same key"
    assert jnp.allclose(s1, s2), "Spikes not deterministic for same key"


# ============================================================================
# Group B: Tutorial output validation tests (6 new tests)
# ============================================================================

def test_v033_manifest_coupling_is_dynamic():
    """manifest.json must declare coupling as dynamic injection, not post-hoc."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    with open(manifest_path) as f:
        manifest = json.load(f)
    coupling = manifest.get("coupling", {})
    impl = coupling.get("implementation_method", "")
    assert "dynamic" in impl.lower(), \
        f"Expected dynamic coupling implementation, got: '{impl}'"


def test_v033_i_neuron_fires():
    """I/PV neuron must have nonzero firing rate in metrics."""
    metrics_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/metrics.json")
    if not metrics_path.exists():
        pytest.skip("Metrics not generated")
    with open(metrics_path) as f:
        metrics = json.load(f)
    i_rate = metrics.get("i_firing_rate_hz", 0.0)
    assert i_rate > 0.0, f"I/PV neuron is silent: {i_rate} Hz"


def test_v033_i_neuron_rate_in_gate():
    """I/PV neuron firing rate must be in 2-25 Hz range."""
    metrics_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/metrics.json")
    if not metrics_path.exists():
        pytest.skip("Metrics not generated")
    with open(metrics_path) as f:
        metrics = json.load(f)
    i_rate = metrics.get("i_firing_rate_hz", 0.0)
    assert 2.0 <= i_rate <= 25.0, f"I firing rate {i_rate} Hz out of 2-25 Hz range"


def test_v033_e_neuron_rate_in_gate():
    """E neuron firing rate must be in 2-25 Hz range."""
    metrics_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/metrics.json")
    if not metrics_path.exists():
        pytest.skip("Metrics not generated")
    with open(metrics_path) as f:
        metrics = json.load(f)
    e_rate = metrics.get("e_firing_rate_hz", 0.0)
    assert 2.0 <= e_rate <= 25.0, f"E firing rate {e_rate} Hz out of 2-25 Hz range"


def test_v033_coupling_currents_figure_exists():
    """Coupling currents figure must exist (generated from dynamic synaptic currents)."""
    fig_path = pathlib.Path(
        "docs/tutorials_v030/_static/figures/v0303_two_neuron_ei_coupling_currents.png"
    )
    if not fig_path.exists():
        pytest.skip("Coupling currents figure not generated")
    assert fig_path.stat().st_size > 0, "Coupling currents figure is empty"


def test_v033_validation_report_i_gate_passes():
    """validation_report must show i_firing_rate_gate_2_25_hz as True."""
    val_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/validation_report.json")
    if not val_path.exists():
        pytest.skip("Validation report not generated")
    with open(val_path) as f:
        report = json.load(f)
    assert report.get("i_firing_rate_gate_2_25_hz") is True, \
        f"I/PV neuron firing rate gate failed; got: {report.get('i_firing_rate_gate_2_25_hz')}"


# ============================================================================
# Group C: Claim gate immutability tests (3 new tests)
# ============================================================================

def test_v033_basis_physical_amplitude_claim_false():
    """basis.physical_amplitude_claim_allowed must be exactly False."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["basis"]["physical_amplitude_claim_allowed"] is False, \
        "physical_amplitude_claim_allowed must be False"


def test_v033_basis_claim_level_computational_scaffold():
    """basis.claim_level must be exactly 'computational_scaffold'."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["basis"]["claim_level"] == "computational_scaffold", \
        f"claim_level must be 'computational_scaffold', got: {manifest['basis']['claim_level']}"


def test_v033_basis_truth_mode_correct():
    """basis.truth_mode must be 'truth_safe_unverified'."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["basis"]["truth_mode"] == "truth_safe_unverified", \
        f"truth_mode must be 'truth_safe_unverified', got: {manifest['basis']['truth_mode']}"


# ============================================================================
# Group D: Asset integrity tests (3 new tests)
# ============================================================================

def test_v033_asset_hashes_json_exists_and_nonempty():
    """asset_hashes.json must exist and contain hash entries."""
    hash_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/asset_hashes.json")
    if not hash_path.exists():
        pytest.skip("Asset hashes not generated")
    with open(hash_path) as f:
        hashes = json.load(f)
    assert isinstance(hashes, dict), "asset_hashes.json must be a dict"
    assert len(hashes) > 0, "asset_hashes.json must have at least one entry"


def test_v033_all_json_files_parseable():
    """All JSON files in outputs must parse without error as dicts."""
    out_dir = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal")
    if not out_dir.exists():
        pytest.skip("Output directory not generated")
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) > 0, "No JSON files found in output dir"
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        assert isinstance(data, dict), f"{jf.name} did not parse as dict"


def test_v033_syntax_check():
    """Tutorial script must pass py_compile (no syntax errors)."""
    script_path = str(
        pathlib.Path(__file__).parent.parent / "examples" / "v033_two_neuron_ei_multimodal.py"
    )
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", script_path],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, \
        f"py_compile failed:\n{result.stderr}"


# ============================================================================
# Group E: 8-figure completeness tests (2 new tests)
# ============================================================================

def test_v033_all_8_figures_exist_in_static():
    """All 8 required figures must exist in docs-stable _static/figures."""
    figures_dir = pathlib.Path("docs/tutorials_v030/_static/figures")
    required_figures = [
        "v0303_two_neuron_ei_voltage.png",
        "v0303_two_neuron_ei_raster.png",
        "v0303_two_neuron_ei_coupling_currents.png",
        "v0303_two_neuron_ei_source.png",
        "v0303_two_neuron_ei_lfp_proxy.png",
        "v0303_two_neuron_ei_csd_proxy.png",
        "v0303_two_neuron_ei_coupled_vs_uncoupled.png",
        "v0303_two_neuron_ei_circuit_schematic.png",
    ]
    for fig_name in required_figures:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            pytest.skip(f"Figure not found (tutorial not run): {fig_name}")
        assert fig_path.stat().st_size > 0, f"Figure is empty: {fig_name}"


def test_v033_manifest_has_8_figure_entries():
    """atlas_manifest figures dict must contain all 8 figure keys."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    with open(manifest_path) as f:
        manifest = json.load(f)
    figures = manifest.get("figures", {})
    required_keys = {
        "voltage_traces", "spike_raster", "coupling_currents", "source_aggregation",
        "lfp_proxy", "csd_proxy", "coupled_vs_uncoupled", "circuit_schematic",
    }
    missing = required_keys - set(figures.keys())
    assert not missing, f"Missing figure keys in manifest: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
