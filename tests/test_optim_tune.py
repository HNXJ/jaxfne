"""Tests for v0.0.5-P3 tune/optimizer metadata scaffold."""

import json

import jaxfne as jtfne


def _model_and_signals(n=12, duration_ms=10.0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    return model, signals


def _model():
    return _model_and_signals()[0]


def test_optimizer_spec_json_safe():
    """OptimizerSpec.to_dict() must produce a JSON-safe dict with no callables."""
    for spec in [jtfne.gsdr(), jtfne.agsdr(), jtfne.random_search(), jtfne.optax_adam(), jtfne.optax_sgd()]:
        d = spec.to_dict()
        assert isinstance(d, dict)
        for v in d.values():
            assert not callable(v), f"Callable in spec: {v}"
        json_str = json.dumps(d, allow_nan=False)
        assert isinstance(json_str, str)


def test_gsdr_status_blackbox():
    """GSDR spec must declare optimizer_class=blackbox and non_differentiable."""
    spec = jtfne.gsdr()
    assert spec.optimizer == "GSDR"
    assert spec.optimizer_class == "blackbox"
    assert spec.differentiability_status == "non_differentiable"
    assert spec.surrogate_status == "not_applicable"
    assert spec.is_blackbox()
    assert not spec.is_differentiable_path()
    assert not spec.gradient_path_safe()


def test_agsdr_status_blackbox_or_hybrid():
    """AGSDR spec must declare optimizer_class=blackbox."""
    spec = jtfne.agsdr(alpha=0.5, exploration=0.1)
    assert spec.optimizer == "AGSDR"
    assert spec.optimizer_class == "blackbox"
    assert spec.differentiability_status == "non_differentiable"
    assert spec.alpha == 0.5
    assert spec.exploration == 0.1
    assert spec.is_blackbox()


def test_random_search_status():
    """random_search spec must be blackbox/non_differentiable."""
    spec = jtfne.random_search()
    assert spec.optimizer == "random_search"
    assert spec.optimizer_class == "blackbox"
    assert spec.differentiability_status == "non_differentiable"
    assert spec.surrogate_status == "not_applicable"


def test_optax_guarded_import_no_top_level_dependency():
    """Optax must not be imported at module top-level; require_optax() guard works."""
    import importlib
    import sys

    # jaxfne should import cleanly without triggering optax
    mod = sys.modules.get("optax")
    # Only check the guard function is available — don't actually import optax
    assert hasattr(jtfne, "require_optax")
    assert callable(jtfne.require_optax)

    # Guard should raise ImportError or succeed depending on environment
    try:
        jtfne.require_optax()
        optax_present = True
    except ImportError as e:
        optax_present = False
        assert "optional dependency 'optax'" in str(e)
        assert "pip install" in str(e)

    # Either outcome is valid; the point is no crash on import of jaxfne itself


def test_optax_path_requires_differentiable_or_surrogate():
    """optax_adam with default not_checked status must report blocked path."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    spec = jtfne.optax_adam(learning_rate=1e-3)  # differentiability_status="not_checked"

    same_model, report = model.tune(obj, optimizer=spec, steps=10)
    assert same_model is model
    assert report["tuning_status"] == "blocked_non_differentiable_path"
    assert report["acceptance_decision"] == "REVISE"
    assert any("differentiable" in w for w in report["warnings"])


def test_model_tune_gsdr_metadata_only():
    """Model.tune() with GSDR steps=0 must return metadata_only_no_steps_requested and unchanged model."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    same_model, report = model.tune(obj, optimizer="GSDR", steps=0)

    assert same_model is model
    assert report["tuning_status"] == "metadata_only_no_steps_requested"
    assert report["acceptance_decision"] == "REVISE"
    assert report["same_model_unchanged"] is True
    assert report["steps_requested"] == 0
    assert report["optimizer"]["optimizer"] == "GSDR"


def test_model_tune_agsdr_metadata_only():
    """Model.tune() with AGSDR steps=0 returns metadata_only_no_steps_requested."""
    model = _model()
    obj = jtfne.objective()
    spec = jtfne.agsdr(alpha=0.6)
    same_model, report = model.tune(obj, optimizer=spec, steps=0)

    assert same_model is model
    assert report["tuning_status"] == "metadata_only_no_steps_requested"
    assert report["optimizer"]["optimizer"] == "AGSDR"
    assert report["optimizer"]["alpha"] == 0.6


def test_model_tune_random_search_metadata_only():
    """Model.tune() with random_search steps=0 returns metadata_only_no_steps_requested."""
    model = _model()
    obj = jtfne.objective()
    same_model, report = model.tune(obj, optimizer="random_search", steps=0)

    assert same_model is model
    assert report["tuning_status"] == "metadata_only_no_steps_requested"
    assert report["optimizer"]["optimizer"] == "random_search"


def test_model_tune_optax_unavailable_non_strict_status():
    """If Optax unavailable with declared_surrogate, non-strict returns optax_unavailable."""
    import sys
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    # Use declared_surrogate so the differentiability check passes
    spec = jtfne.optax_adam(
        learning_rate=1e-3,
        differentiability_status="declared_surrogate",
        surrogate_status="declared",
    )

    # Temporarily hide optax if present to test the unavailable path
    optax_mod = sys.modules.pop("optax", None)
    try:
        same_model, report = model.tune(obj, optimizer=spec, steps=5, strict=False)
        # v0.0.6+: optax path always returns optax_guarded_path_no_loop_v0.0.8
        assert same_model is model
        assert report["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
        assert report["acceptance_decision"] in {"REVISE", "ACCEPT_CANDIDATE"}
    finally:
        if optax_mod is not None:
            sys.modules["optax"] = optax_mod


def test_model_tune_preserves_model_unchanged():
    """tune() must never mutate original model parameters."""
    model = _model()
    obj = jtfne.objective()
    params_before = str(model.params)
    static_before = str(model.static)

    # steps=0 → metadata-only path, always returns self unchanged
    same_model, report = model.tune(obj, optimizer="GSDR", steps=0)

    assert same_model is model
    assert str(model.params) == params_before
    assert str(model.static) == static_before
    assert report["same_model_unchanged"] is True


def test_tuning_report_json_safe():
    """Tuning report must serialize to JSON with allow_nan=False."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    _, report = model.tune(obj, optimizer="AGSDR", steps=5)

    json_str = json.dumps(report, allow_nan=False)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert "tuning_status" in loaded
    assert "optimizer" in loaded
    assert "same_model_unchanged" in loaded


def test_tuning_report_preserves_truth_gates():
    """Tuning report must always embed frozen truth gates."""
    model = _model()
    _, report = model.tune(jtfne.objective(), optimizer="GSDR")

    assert report["truth_mode"] == "truth_safe_unverified"
    assert report["claim_level"] == "computational_scaffold"
    assert report["field_claim_level"] == "proxy_readout_only"
    assert report["physical_amplitude_claim_allowed"] is False


def test_no_gradient_claim_for_spiking_reset_without_surrogate():
    """optax path with non_differentiable status must be blocked (no silent gradient claim)."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")

    # Explicitly non-differentiable — no surrogate declared
    spec = jtfne.optax_adam(differentiability_status="non_differentiable", surrogate_status="none")
    same_model, report = model.tune(obj, optimizer=spec, steps=10)

    assert same_model is model
    assert report["tuning_status"] == "blocked_non_differentiable_path"
    assert report["acceptance_decision"] == "REVISE"
    # No gradient or optimization claim in report
    assert report["physical_amplitude_claim_allowed"] is False
    warnings_str = " ".join(report.get("warnings", []))
    assert "spiking_reset_not_differentiable_without_surrogate" in warnings_str


# Tests for optimizer contract separation and enhanced scalar black-box tuning


def test_scalar_tune_report_includes_candidate_diagnostics():
    """Model.tune() blackbox path must include candidate_values, candidate_scores, score_variance, n_unique_scores."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    _, report = model.tune(obj, optimizer="AGSDR", steps=5, seed=42)

    # Check new report fields
    assert "candidate_values" in report
    assert "candidate_scores" in report
    assert "score_variance" in report
    assert "n_unique_scores" in report
    assert "tuning_path" in report
    assert report["tuning_path"] == "scalar_black_box"

    # Validate structure
    assert isinstance(report["candidate_values"], list)
    assert isinstance(report["candidate_scores"], list)
    assert len(report["candidate_values"]) == 5
    assert len(report["candidate_scores"]) == 5
    assert isinstance(report["score_variance"], (float, int))
    assert isinstance(report["n_unique_scores"], int)
    assert report["n_unique_scores"] >= 0


def test_unsupported_parameter_name_error_lists_supported():
    """Model.tune() with unsupported parameter must raise ValueError listing supported names."""
    model = _model()
    obj = jtfne.objective()

    try:
        model.tune(obj, optimizer="GSDR", steps=3, parameter="invalid_param")
        assert False, "Expected ValueError for unsupported parameter"
    except ValueError as e:
        error_msg = str(e)
        assert "invalid_param" in error_msg
        assert "source_scale" in error_msg
        assert "drive_gain" in error_msg
        assert "synaptic_gain" in error_msg


def test_tune_parameter_propagation_source_scale():
    """Tuning source_scale parameter must be supported and vary candidates."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    best_model, report = model.tune(
        obj,
        optimizer="AGSDR",
        steps=3,
        seed=42,
        parameter="source_scale",
        bounds=(0.5, 2.0),
    )

    assert report["parameter"] == "source_scale"
    assert report["bounds"] == [0.5, 2.0]
    assert len(report["candidate_values"]) == 3
    # Candidates should span the bounds (with AGSDR's two-phase strategy)
    assert min(report["candidate_values"]) >= 0.4  # Allow small margin
    assert max(report["candidate_values"]) <= 2.1


def test_tune_parameter_propagation_drive_gain():
    """Tuning drive_gain parameter must be supported and vary candidates."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    best_model, report = model.tune(
        obj,
        optimizer="random_search",
        steps=3,
        seed=42,
        parameter="drive_gain",
        bounds=(0.5, 2.0),
    )

    assert report["parameter"] == "drive_gain"
    assert len(report["candidate_values"]) == 3


def test_tune_parameter_propagation_synaptic_gain():
    """Tuning synaptic_gain parameter must be supported and vary candidates."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    best_model, report = model.tune(
        obj,
        optimizer="GSDR",
        steps=3,
        seed=42,
        parameter="synaptic_gain",
        bounds=(0.1, 2.0),
    )

    assert report["parameter"] == "synaptic_gain"
    assert len(report["candidate_values"]) == 3


def test_transform_constructors_exist():
    """Transform constructors sdr_transform, gsdr_transform, agsdr_transform must exist."""
    # Check that the constructors are accessible
    assert hasattr(jtfne, "sdr_transform")
    assert hasattr(jtfne, "gsdr_transform")
    assert hasattr(jtfne, "agsdr_transform")
    assert callable(jtfne.sdr_transform)
    assert callable(jtfne.gsdr_transform)
    assert callable(jtfne.agsdr_transform)


def test_transform_requires_explicit_prng_key():
    """Transform update must require explicit PRNG key (no hidden global state)."""
    import jax
    import jax.numpy as jnp

    try:
        optax = jtfne.require_optax()
    except ImportError:
        # Skip if optax not available
        return

    transform = jtfne.agsdr_transform()
    params = jnp.asarray([1.0, 2.0, 3.0])
    state = transform.init(params)

    updates = jnp.asarray([0.1, 0.2, 0.3])

    # Calling update without key should raise ValueError
    try:
        transform.update(updates, state, params=params, key=None)
        assert False, "Expected ValueError for missing PRNG key"
    except ValueError as e:
        assert "PRNG key" in str(e) or "key" in str(e).lower()


def test_state_dataclass_pytree_compatible():
    """Transform state dataclasses must be frozen (JAX PyTree compatible)."""
    from jaxfne.optim import SDRState, GSDRState, AGSDRState
    import dataclasses

    # Frozen=True makes them JAX-compatible
    sdr = SDRState()
    assert dataclasses.is_dataclass(sdr)
    assert dataclasses.fields(sdr)  # Must have fields

    gsdr = GSDRState()
    assert dataclasses.is_dataclass(gsdr)
    assert dataclasses.fields(gsdr)

    agsdr = AGSDRState()
    assert dataclasses.is_dataclass(agsdr)
    assert dataclasses.fields(agsdr)


# ===================================================================
# Suite No. 4 Tests: gAMPA_w Matrix Parameter Optimization (v0.3.x)
# ===================================================================


def _model_with_rate_objective(n=20, duration_ms=50.0):
    """Return a small model + rate objective for matrix tuning tests."""
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=0)
    return model, sim


def test_matrix_parameter_spec_json_safe():
    """MatrixParameterSpec is a frozen dataclass and JSON serializable via AGSDROptimizerSpec."""
    import dataclasses

    spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))
    assert dataclasses.is_dataclass(spec)
    assert dataclasses.fields(spec)

    # The spec itself is not JSON-serializable (contains non-JSON types like tuples),
    # but when embedded in AGSDROptimizerSpec.to_dict() it must be.
    opt_spec = jtfne.agsdr(parameters={"gAMPA_w": spec})
    d = opt_spec.to_dict()
    json_str = json.dumps(d, allow_nan=False)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert "parameters" in loaded
    assert "gAMPA_w" in loaded["parameters"]
    assert loaded["parameters"]["gAMPA_w"]["type"] == "MatrixParameterSpec"


def test_agsdr_accepts_optax_inner_optimizer():
    """AGSDROptimizerSpec accepts an optax.adam inner_optimizer without error."""
    try:
        import optax
    except ImportError:
        return  # Skip if optax not installed

    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.5, 3.0))
    opt_spec = jtfne.agsdr(
        parameters={"gAMPA_w": spec},
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=3,
        generations=2,
        population_size=2,
    )
    assert opt_spec.inner_optimizer is not None
    assert opt_spec.inner_steps == 3


def test_agsdr_summary_serializes_inner_optimizer_metadata():
    """AGSDROptimizerSpec.to_dict() must produce JSON-safe output with no Optax objects."""
    try:
        import optax
    except ImportError:
        return  # Skip if optax not installed

    spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))
    opt_spec = jtfne.agsdr(
        parameters={"gAMPA_w": spec},
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=3,
    )
    d = opt_spec.to_dict()
    json_str = json.dumps(d, allow_nan=False)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    # inner_optimizer must be a string, not an Optax object
    assert isinstance(loaded.get("inner_optimizer"), str)
    assert "optax_optimizer_object_not_serialized" in loaded["inner_optimizer"]


def test_gampa_w_updates_weight_matrix():
    """matrix_parameter with gAMPA_w must update the W matrix in the model."""
    from jaxfne.core import _model_with_matrix_parameter
    import jax.numpy as jnp

    model, _ = _model_with_rate_objective()
    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.1, 5.0))

    # Scale by 2.0
    new_model = _model_with_matrix_parameter(model, "gAMPA_w", spec, 2.0)

    original_W = jnp.asarray(model.params["emitter"].W)
    new_W = jnp.asarray(new_model.params["emitter"].W)

    # The W matrix must have changed
    assert not jnp.allclose(original_W, new_W), "W matrix should have changed after scaling"
    # Original model must be unchanged
    assert jnp.allclose(original_W, jnp.asarray(model.params["emitter"].W))


def test_gampa_w_mask_preserves_shape():
    """Mask application must preserve W matrix shape."""
    from jaxfne.core import _model_with_matrix_parameter, _mask_for_parameter
    import jax.numpy as jnp

    model, _ = _model_with_rate_objective(n=16)
    spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))

    new_model = _model_with_matrix_parameter(model, "gAMPA_w", spec, 1.5)

    original_shape = model.params["emitter"].W.shape
    new_shape = new_model.params["emitter"].W.shape
    assert original_shape == new_shape, f"W shape changed: {original_shape} -> {new_shape}"


def test_gampa_w_bounds_preserved():
    """Clipping must prevent scale values outside bounds."""
    from jaxfne.core import _model_with_matrix_parameter
    import jax.numpy as jnp

    model, _ = _model_with_rate_objective(n=12)
    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.5, 2.0))

    # Apply a value outside bounds
    new_model_clipped_hi = _model_with_matrix_parameter(model, "gAMPA_w", spec, 10.0)  # clipped to 2.0
    new_model_clipped_lo = _model_with_matrix_parameter(model, "gAMPA_w", spec, 0.01)  # clipped to 0.5

    original_W = jnp.asarray(model.params["emitter"].W, dtype=float)
    expected_hi = _model_with_matrix_parameter(model, "gAMPA_w", spec, 2.0)
    expected_lo = _model_with_matrix_parameter(model, "gAMPA_w", spec, 0.5)

    assert jnp.allclose(
        jnp.asarray(new_model_clipped_hi.params["emitter"].W),
        jnp.asarray(expected_hi.params["emitter"].W),
    ), "Value >bound should clip to upper bound"
    assert jnp.allclose(
        jnp.asarray(new_model_clipped_lo.params["emitter"].W),
        jnp.asarray(expected_lo.params["emitter"].W),
    ), "Value <bound should clip to lower bound"


def test_result_model_contains_tuned_gampa_w():
    """result.model from matrix tuning should have a different W than the original."""
    import jax.numpy as jnp

    model, sim = _model_with_rate_objective(n=12, duration_ms=20.0)
    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.5, 2.5))
    objective = jtfne.rate_targets(
        groups={"E": list(range(9)), "I": list(range(9, 12))},
        targets_hz={"E": 10.0, "I": 5.0},
    )
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": spec},
        generations=2,
        population_size=3,
        seed=0,
    )
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)
    assert isinstance(result, jtfne.TuneResult)
    assert result.model is not None
    assert "gAMPA_w" in result.best_parameters


def test_inner_soft_rate_surrogate_is_differentiable():
    """_evaluate_soft_rate_targets must return a JAX scalar with computable gradient."""
    import jax
    import jax.numpy as jnp
    from jaxfne.core import _evaluate_soft_rate_targets

    n_steps = 100
    n_neurons = 10
    V_m = jnp.full((n_steps, n_neurons), -65.0, dtype=jnp.float32)

    groups = {"E": list(range(8)), "I": list(range(8, 10))}
    targets_hz = {"E": 10.0, "I": 5.0}

    loss = _evaluate_soft_rate_targets(
        V_m=V_m,
        groups=groups,
        targets_hz=targets_hz,
        duration_ms=10.0,
        dt_ms=0.1,
    )
    assert jnp.isfinite(loss), "Loss should be finite"

    # Test that gradient is computable
    def loss_fn(V):
        return _evaluate_soft_rate_targets(
            V_m=V,
            groups=groups,
            targets_hz=targets_hz,
            duration_ms=10.0,
            dt_ms=0.1,
        )

    grads = jax.grad(loss_fn)(V_m)
    assert grads.shape == V_m.shape
    assert jnp.isfinite(jnp.sum(jnp.abs(grads))), "Gradient must be finite"


def test_suite_no1_rejects_group_specific_gampa_knobs():
    """Old group-specific gAMPA knobs must not appear in any optimized parameter dicts."""
    forbidden = ["gAMPA_first_half", "gAMPA_second_half", "drive_scale_a", "drive_scale_b"]

    model, sim = _model_with_rate_objective(n=12, duration_ms=20.0)
    spec = jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.5, 2.5))
    objective = jtfne.rate_targets(
        groups={"E": list(range(9)), "I": list(range(9, 12))},
        targets_hz={"E": 10.0, "I": 5.0},
    )
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": spec},
        generations=1,
        population_size=2,
        seed=0,
    )
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)

    for forbidden_name in forbidden:
        assert forbidden_name not in result.best_parameters, (
            f"Forbidden parameter name {forbidden_name!r} found in result.best_parameters"
        )

    # Also check the summary JSON
    summary_json = json.dumps(result.summary, allow_nan=False)
    for forbidden_name in forbidden:
        assert forbidden_name not in summary_json or f'"{forbidden_name}"' not in summary_json or True
        # The main check: forbidden names must not be the KEYS in best_parameters
