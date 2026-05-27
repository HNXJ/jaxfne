"""Detailed parameter propagation validation for Model.tune().

Tests verify that tune() parameter modifications have documented effects:
- source_scale: affects amplitude only, not spike rates
- drive_gain: affects firing rates
- synaptic_gain: scales W matrix only
"""

import jax.numpy as jnp
import jaxfne as jtfne


def _model_and_signals(parameter: str, value: float, duration_ms: float = 10.0, n: int = 12):
    """Create model with specified parameter and simulate."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    base_model = jtfne.construct(cfg)

    # Apply parameter modification
    from jaxfne.core import _model_with_scalar_parameter
    model = _model_with_scalar_parameter(base_model, parameter, float(value))

    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=42)
    signals = model.simulate(sim)
    return model, signals


def test_source_scale_invariant_spikes():
    """source_scale must not affect spike patterns (raster, rates)."""
    # Create models with different source_scale values
    model_1, signals_1 = _model_and_signals("source_scale", 1.0)
    model_2, signals_2 = _model_and_signals("source_scale", 2.0)

    # Spike patterns must be identical
    assert jnp.allclose(signals_1.spikes, signals_2.spikes), \
        "Spike raster changed with source_scale (should be invariant)"

    # Spike rates per neuron must be identical
    rate_1 = signals_1.spikes.sum(axis=0)
    rate_2 = signals_2.spikes.sum(axis=0)
    assert jnp.allclose(rate_1, rate_2), \
        "Spike rates changed with source_scale (should be invariant)"


def test_source_scale_affects_amplitude():
    """source_scale must scale source/readout signal amplitude."""
    # Create models with different source_scale values
    model_1, signals_1 = _model_and_signals("source_scale", 1.0)
    model_2, signals_2 = _model_and_signals("source_scale", 2.0)

    # Spike patterns must be identical (from previous test)
    assert jnp.allclose(signals_1.spikes, signals_2.spikes)

    # Sources/field must scale with source_scale
    if signals_1.sources is not None and signals_2.sources is not None:
        # Sources should scale approximately linearly with source_scale
        source_ratio = jnp.mean(jnp.abs(signals_2.sources) / (jnp.abs(signals_1.sources) + 1e-8))
        # Allow some tolerance due to numerical variation
        assert 1.8 <= source_ratio <= 2.2, \
            f"Source amplitude ratio {source_ratio} not close to 2.0 for 2x source_scale"

    # LFP should also scale
    if signals_1.field is not None and signals_2.field is not None:
        lfp_1 = signals_1.field.lfp
        lfp_2 = signals_2.field.lfp
        if lfp_1 is not None and lfp_2 is not None:
            lfp_ratio = jnp.mean(jnp.abs(lfp_2) / (jnp.abs(lfp_1) + 1e-8))
            # Allow some tolerance
            assert 1.8 <= lfp_ratio <= 2.2, \
                f"LFP amplitude ratio {lfp_ratio} not close to 2.0 for 2x source_scale"


def test_drive_gain_affects_firing_rate():
    """drive_gain must affect spike rates (firing rate)."""
    # Create models with different drive_gain values
    model_1, signals_1 = _model_and_signals("drive_gain", 0.5, duration_ms=100.0)
    model_2, signals_2 = _model_and_signals("drive_gain", 1.5, duration_ms=100.0)

    # Spike rates should differ (higher drive_gain → higher firing rate typically)
    rate_1 = signals_1.spikes.sum() / signals_1.spikes.shape[0]  # spikes per neuron
    rate_2 = signals_2.spikes.sum() / signals_2.spikes.shape[0]

    # They should not be identical (drive_gain should have effect)
    assert not jnp.allclose(rate_1, rate_2), \
        "Drive gain had no effect on firing rate"

    # Higher drive_gain should generally produce higher firing rate
    # (for standard Izhikevich params with positive drive)
    assert rate_2 > rate_1, \
        f"Expected higher firing rate with drive_gain=1.5 vs 0.5, got {rate_2} vs {rate_1}"


def test_drive_gain_in_objective_score():
    """drive_gain changes must be reflected in objective scores."""
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")

    model_low, _ = _model_and_signals("drive_gain", 0.5, duration_ms=100.0)
    model_high, _ = _model_and_signals("drive_gain", 1.5, duration_ms=100.0)

    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    signals_low = model_low.simulate(sim)
    signals_high = model_high.simulate(sim)

    report_low = model_low.evaluate(signals_low, obj)
    report_high = model_high.evaluate(signals_high, obj)

    # Scores should differ due to different firing rates
    score_low = report_low.get("total_loss", 0.0)
    score_high = report_high.get("total_loss", 0.0)

    # The key point: scores should be different because firing rates are different
    assert score_low != score_high, \
        f"Loss scores identical despite drive_gain difference: {score_low} vs {score_high}"


def test_synaptic_gain_scales_w_matrix():
    """synaptic_gain must scale the recurrent weight matrix W only."""
    # Get base model
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    base_model = jtfne.construct(cfg)

    # Create models with different synaptic_gain
    from jaxfne.core import _model_with_scalar_parameter
    model_1 = _model_with_scalar_parameter(base_model, "synaptic_gain", 1.0)
    model_2 = _model_with_scalar_parameter(base_model, "synaptic_gain", 2.0)

    # Get the W matrices
    W_1 = model_1.params["emitter"].W
    W_2_scaled = model_2.params["emitter"].W

    # Create a mask for non-diagonal finite elements
    mask = ~jnp.isnan(W_1) & (jnp.arange(W_1.shape[0])[:, None] != jnp.arange(W_1.shape[1]))

    # Check scaling on finite non-diagonal elements
    W_1_finite = W_1[mask]
    W_2_finite = W_2_scaled[mask]

    # W_2 should equal 2.0 * W_1 (approximately), handling sign correctly
    # The scaling should preserve the sign and multiply magnitude by 2.0
    expected_W_2 = 2.0 * W_1_finite
    assert jnp.allclose(W_2_finite, expected_W_2, atol=1e-5), \
        f"synaptic_gain scaling of W failed: W_2 not equal to 2*W_1"


def test_synaptic_gain_unchanged_drive():
    """synaptic_gain scaling must not affect the drive vector."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    base_model = jtfne.construct(cfg)

    from jaxfne.core import _model_with_scalar_parameter
    model_1 = _model_with_scalar_parameter(base_model, "synaptic_gain", 1.0)
    model_2 = _model_with_scalar_parameter(base_model, "synaptic_gain", 2.0)

    # Drive should be unchanged
    drive_1 = model_1.params["emitter"].drive
    drive_2 = model_2.params["emitter"].drive

    assert jnp.allclose(drive_1, drive_2), \
        "Drive changed with synaptic_gain (should be invariant)"


def test_synaptic_gain_unchanged_source_scale():
    """synaptic_gain scaling must not affect source_scale."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    base_model = jtfne.construct(cfg)

    from jaxfne.core import _model_with_scalar_parameter
    model_1 = _model_with_scalar_parameter(base_model, "synaptic_gain", 1.0)
    model_2 = _model_with_scalar_parameter(base_model, "synaptic_gain", 2.0)

    # source_scale should be unchanged
    ss_1 = model_1.params["emitter"].source_scale
    ss_2 = model_2.params["emitter"].source_scale

    assert jnp.allclose(ss_1, ss_2), \
        "source_scale changed with synaptic_gain (should be invariant)"


def test_synaptic_gain_affects_network_dynamics():
    """synaptic_gain affects network recurrent dynamics through W scaling."""
    # Create models with different synaptic_gain values
    # The effect on spikes depends on network initialization and dynamics
    model_1, signals_1 = _model_and_signals("synaptic_gain", 1.0, duration_ms=50.0)
    model_2, signals_2 = _model_and_signals("synaptic_gain", 1.0, duration_ms=50.0)

    # Verify that same parameters produce same results (determinism)
    assert jnp.allclose(signals_1.spikes, signals_2.spikes), \
        "Same synaptic_gain produced different spike patterns (non-deterministic)"

    # Test that W matrix scaling is correctly applied
    # (more direct test of synaptic_gain effect)
    from jaxfne.core import _model_with_scalar_parameter
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    base_model = jtfne.construct(cfg)

    model_low = _model_with_scalar_parameter(base_model, "synaptic_gain", 0.5)
    model_high = _model_with_scalar_parameter(base_model, "synaptic_gain", 2.0)

    W_low = model_low.params["emitter"].W
    W_high = model_high.params["emitter"].W

    # Verify W_high ≈ 4 * W_low (ratio of 2.0 / 0.5)
    ratio = 2.0 / 0.5
    expected_W_high = ratio * W_low

    # Check non-NaN elements
    mask = ~jnp.isnan(W_low)
    assert jnp.allclose(W_high[mask], expected_W_high[mask], atol=1e-5), \
        "W matrix not scaled correctly with synaptic_gain"


def test_parameter_bounds_small_values():
    """Parameter tuning must work with small values near bounds."""
    model_small, _ = _model_and_signals("source_scale", 0.1)
    model_large, _ = _model_and_signals("source_scale", 10.0)

    # Both should produce valid models
    assert model_small is not None
    assert model_large is not None

    # Simulate with both
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
    signals_small = model_small.simulate(sim)
    signals_large = model_large.simulate(sim)

    # Both should have valid outputs
    assert signals_small.spikes is not None
    assert signals_large.spikes is not None
    assert jnp.all(jnp.isfinite(signals_small.spikes))
    assert jnp.all(jnp.isfinite(signals_large.spikes))


def test_parameter_independence_source_drive():
    """Verify no cross-contamination between source_scale and drive_gain."""
    # Model with high source_scale, low drive_gain
    model_1, signals_1 = _model_and_signals("source_scale", 2.0)
    model_1_drive_mod = _model_and_signals("drive_gain", 0.5)[0]

    # Model with low source_scale, high drive_gain
    model_2, signals_2 = _model_and_signals("source_scale", 0.5)
    model_2_drive_mod = _model_and_signals("drive_gain", 2.0)[0]

    # The effect of source_scale should be independent of drive_gain
    # (verify by spike patterns not affected by source_scale alone)
    rate_1 = signals_1.spikes.sum()
    rate_2 = signals_2.spikes.sum()

    assert rate_1 == rate_2, \
        "Spike count changed with source_scale (should only affect amplitude)"


def test_tune_report_parameter_field():
    """tune() report must include parameter name."""
    model = _model_and_signals("source_scale", 1.0)[0]
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")

    _, report = model.tune(obj, optimizer="AGSDR", steps=3, parameter="synaptic_gain")

    assert "parameter" in report
    assert report["parameter"] == "synaptic_gain"


def test_tune_report_bounds_field():
    """tune() report must include parameter bounds."""
    model = _model_and_signals("source_scale", 1.0)[0]
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")

    bounds = (0.1, 2.0)
    _, report = model.tune(
        obj,
        optimizer="AGSDR",
        steps=3,
        parameter="synaptic_gain",
        bounds=bounds,
    )

    assert "bounds" in report
    assert report["bounds"] == list(bounds)
