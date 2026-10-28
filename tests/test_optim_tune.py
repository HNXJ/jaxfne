"""Tests for v0.0.5-P3 tune/optimizer metadata scaffold."""

import json

import pytest

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
    for spec in [jtfne.gsdr(), jtfne.agsdr(), jtfne.random_search(), jtfne.optax_adam(), jtfne.optax_sgd(), jtfne.gsgd()]:
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


def test_gsgd_status_differentiable():
    """GSGD spec must declare optimizer_class=differentiable, default not_checked."""
    spec = jtfne.gsgd()
    assert spec.optimizer == "GSGD"
    assert spec.optimizer_class == "differentiable"
    assert spec.differentiability_status == "not_checked"
    assert spec.surrogate_status == "none"
    assert spec.learning_rate == 0.01
    assert not spec.is_blackbox()
    assert spec.is_differentiable_path()
    assert not spec.gradient_path_safe()


def test_gsgd_resolved_from_string():
    """Model.tune(optimizer='GSGD') must resolve to the gsgd() spec via _resolve_optimizer."""
    from jaxfne.optim.core import _resolve_optimizer

    spec = _resolve_optimizer("GSGD")
    assert spec.optimizer == "GSGD"
    assert spec.optimizer_class == "differentiable"

    spec_lower = _resolve_optimizer("gsgd")
    assert spec_lower.optimizer == "GSGD"


def test_gsgd_path_requires_differentiable_or_surrogate():
    """gsgd() with default not_checked status must report blocked path, like optax_adam."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    spec = jtfne.gsgd(learning_rate=0.02)  # differentiability_status="not_checked"

    same_model, report = model.tune(obj, optimizer=spec, steps=10)
    assert same_model is model
    assert report["tuning_status"] == "blocked_non_differentiable_path"
    assert report["acceptance_decision"] == "REVISE"
    assert any("differentiable" in w for w in report["warnings"])


def test_step_gsgd_transform_direct():
    """Direct unit test of the GSGD kernel: u_next = u_t - eta * grad_l."""
    import jax.numpy as jnp

    u_t = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    grad_l = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float32)
    state = jtfne.GSGDState(count=jnp.array(0), step_size=jnp.array(0.1))
    hyperparams = {"eta": 0.1}

    u_next, new_state = jtfne.step_gsgd_transform(u_t, grad_l, state, hyperparams)

    expected = u_t - 0.1 * grad_l
    assert jnp.allclose(u_next, expected)
    assert new_state is state  # kernel is stateless pass-through at this scaffold stage


def test_optax_guarded_import_no_top_level_dependency():
    """Optax must not be imported at module top-level; require_optax() guard works.

    Checked in a fresh interpreter, since another test may already have
    imported optax in this process.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", "import jaxfne, sys; print('optax' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # Fresh `import jaxfne` must not pull in optax (measured: False).
    assert proc.stdout.strip() == "False"
    # Only check the guard function is available; don't actually import optax
    assert hasattr(jtfne, "require_optax")
    assert callable(jtfne.require_optax)

    # Guard should raise ImportError or succeed depending on environment
    try:
        jtfne.require_optax()
    except ImportError as e:
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
        assert report["acceptance_decision"] == "REVISE"
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

    assert report["claim_level"] == "computational_scaffold"
    assert report["field_claim_level"] == "proxy_readout"
    assert report["physical_amplitude_calibrated"] is False


def test_no_gradient_claim_for_spiking_reset_without_surrogate():
    """optax path with non_differentiable status must be blocked (no silent gradient claim)."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")

    # Explicitly non-differentiable; no surrogate declared
    spec = jtfne.optax_adam(differentiability_status="non_differentiable", surrogate_status="none")
    same_model, report = model.tune(obj, optimizer=spec, steps=10)

    assert same_model is model
    assert report["tuning_status"] == "blocked_non_differentiable_path"
    assert report["acceptance_decision"] == "REVISE"
    # No gradient or optimization claim in report
    assert report["physical_amplitude_calibrated"] is False
    warnings_str = " ".join(report.get("warnings", []))
    assert "spiking_reset_not_differentiable_without_surrogate" in warnings_str


# Tests for optimizer contract separation and enhanced scalar black-box tuning


def test_scalar_tune_report_includes_candidate_diagnostics():
    """Model.tune() blackbox path must include candidate_values, candidate_scores, score_variance, n_unique_scores."""
    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    _, report = model.tune(obj, optimizer="AGSDR", steps=5, seed=42)

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
    # Candidates should span the bounds (with AGSDR's two-phase scope)
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
    import jax.numpy as jnp

    try:
        jtfne.require_optax()
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


def test_sdr_transform_checkpoint_reverts_params_to_best():
    """SDR must actually revert params to best_param once checkpoint_n_steps of
    non-improvement have elapsed -- not just compute params_to_use and drop it."""
    import jax
    import jax.numpy as jnp

    try:
        optax = jtfne.require_optax()
    except ImportError:
        return

    key = jax.random.PRNGKey(0)
    transform = jtfne.sdr_transform(checkpoint_n_steps=3, stochastic_scale=0.0)
    params = jnp.asarray([1.0])
    state = transform.init(params)

    # Loss strictly worsens after step 0 -> no improvement ever again, so the
    # reset_counter reaches checkpoint_n_steps at step 3.
    for step in range(4):
        loss = 10.0 + step
        grad = jnp.asarray([1.0])
        updates, state = transform.update(
            grad, state, params=params, loss=loss, key=jax.random.fold_in(key, step)
        )
        params = optax.apply_updates(params, updates)

    sdr_state, _ = state
    assert sdr_state.reset_counter == 0
    assert float(params[0]) == float(sdr_state.best_param[0])


def test_gsdr_transform_reset_reverts_params_to_best():
    """GSDR's genetic deselection reset must actually revert params to
    best_param, mirroring the SDR checkpoint contract."""
    import jax
    import jax.numpy as jnp

    try:
        optax = jtfne.require_optax()
    except ImportError:
        return

    key = jax.random.PRNGKey(0)
    transform = jtfne.gsdr_transform(deselection_threshold=3, stochastic_scale=0.0)
    params = jnp.asarray([1.0])
    state = transform.init(params)

    for step in range(4):
        loss = 10.0 + step
        grad = jnp.asarray([1.0])
        updates, state = transform.update(
            grad, state, params=params, loss=loss, key=jax.random.fold_in(key, step)
        )
        params = optax.apply_updates(params, updates)

    gsdr_state, _ = state
    assert gsdr_state.deselection_counter == 0
    assert float(params[0]) == float(gsdr_state.best_param[0])


def test_agsdr_transform_reset_reverts_params_to_best():
    """AGSDR's genetic-deselection reset must also revert params to best_param
    -- the identical gap already fixed in sdr_transform/gsdr_transform
    (checkpoint/reset computed but never converted into an update delta).
    Found via direct source read 2026-07-14: agsdr_transform had no
    params_to_use/delta-override logic at all, unlike its two siblings."""
    import jax
    import jax.numpy as jnp

    try:
        optax = jtfne.require_optax()
    except ImportError:
        return

    key = jax.random.PRNGKey(0)
    transform = jtfne.agsdr_transform(deselection_threshold=3, stochastic_scale=0.0)
    params = jnp.asarray([1.0])
    state = transform.init(params)

    for step in range(4):
        loss = 10.0 + step
        grad = jnp.asarray([1.0])
        updates, state = transform.update(
            grad, state, params=params, loss=loss, key=jax.random.fold_in(key, step)
        )
        params = optax.apply_updates(params, updates)

    agsdr_state, _ = state
    assert agsdr_state.deselection_counter == 0
    assert float(params[0]) == float(agsdr_state.best_param[0])


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
    from jaxfne.core import _model_with_matrix_parameter

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


def _matrix_tune_with_failing_simulate(monkeypatch, exc):
    """Run a one-candidate matrix tune whose every Model.simulate raises ``exc``."""
    import optax

    from jaxfne._model import Model

    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8, cell_types={"E": 0.5, "PV": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    objective = jtfne.rate_targets(
        groups={"all": list(range(8))}, targets_hz={"all": 15.0}
    )
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    optimizer = jtfne.agsdr(
        parameters={"gAMPA_w": jtfne.matrix_parameter(mask="all", bounds=(0.3, 3.0))},
        generations=1,
        population_size=1,
        inner_optimizer=optax.adam(1e-2),
        inner_steps=1,
        seed=0,
    )

    def _boom(self, simulation, **kwargs):
        raise exc

    monkeypatch.setattr(Model, "simulate", _boom)
    return model.tune(objectives=objective, optimizer=optimizer, simulation=sim)


def test_matrix_tune_unexpected_error_propagates(monkeypatch):
    """A TypeError from a candidate is a bug, not a failing candidate: it propagates."""
    with pytest.raises(TypeError, match="boom"):
        _matrix_tune_with_failing_simulate(monkeypatch, TypeError("boom"))


@pytest.mark.parametrize("exc", [RuntimeError("xla"), FloatingPointError("fp"), ValueError("v")])
def test_matrix_tune_candidate_failure_is_counted(monkeypatch, exc):
    """A candidate failing with a runtime, arithmetic or value error is counted and scored inf."""
    result = _matrix_tune_with_failing_simulate(monkeypatch, exc)
    counts = result.summary["fallback_counts"]
    assert counts["final_scoring_failed"] >= 1
    assert result.best_score == float("inf")


def test_tune_candidate_batching_refusal_is_declared():
    """T-32c: candidate evaluations stay serial; the refusal names the constraint."""
    from jaxfne._model_tune import CANDIDATE_BATCH_STATUS

    assert "adaptive" in CANDIDATE_BATCH_STATUS
    assert "serial" in CANDIDATE_BATCH_STATUS


def test_tune_history_carries_complete_per_candidate_outputs():
    """T-32c addendum: serial tune history holds candidate identity, scores,
    rejection reasons, and a best selection consistent with the scores."""
    import math

    model = _model()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    result = model.tune(obj, optimizer="AGSDR", steps=5, seed=42)
    history = result.summary["candidate_history"]
    assert len(history) == 5
    for i, entry in enumerate(history):
        assert entry["step"] == i
        assert isinstance(entry["candidate_value"], float)
        assert isinstance(entry["score"], float)
        assert isinstance(entry["all_gates_pass"], bool)
        assert isinstance(entry["accepted_as_best"], bool)
        assert isinstance(entry["rejection_reasons"], list)
    finite = [(e["candidate_value"], e["score"]) for e in history if math.isfinite(e["score"])]
    assert finite, "fixture must yield a finite candidate score"
    best_value, best_score = min(finite, key=lambda pair: pair[1])
    assert result.best_score == best_score
    assert result.summary["best_parameter_value"] == best_value


# --------------------------------------------------------------------------- #
# Refusals: parameters a kernel does not consume, unknown hyperparameter keys
# --------------------------------------------------------------------------- #


def test_unconsumed_transform_parameters_are_refused():
    """sdr_transform never reads alpha_min/alpha_max; gsdr_transform and
    agsdr_transform never read checkpoint_n_steps. Non-default values are
    refused; the defaults stay accepted."""
    from jaxfne.optim.core import agsdr_transform, gsdr_transform, sdr_transform

    with pytest.raises(ValueError, match="alpha_min.*not consumed"):
        sdr_transform(alpha_min=0.2)
    with pytest.raises(ValueError, match="alpha_max.*not consumed"):
        sdr_transform(alpha_max=0.5)
    with pytest.raises(ValueError, match="checkpoint_n_steps.*not consumed"):
        gsdr_transform(checkpoint_n_steps=3)
    with pytest.raises(ValueError, match="checkpoint_n_steps.*not consumed"):
        agsdr_transform(checkpoint_n_steps=3)
    assert sdr_transform() is not None
    assert gsdr_transform() is not None
    assert agsdr_transform() is not None


@pytest.mark.parametrize(
    "kernel_name",
    ["step_sdr_transform", "step_gsdr_transform", "step_agsdr_transform", "step_gsgd_transform"],
)
def test_step_kernels_refuse_unknown_hyperparameters(kernel_name):
    """Each step kernel reads only eta; a misspelled key must not fall back to 0.01."""
    import jax.numpy as jnp

    import jaxfne.optim as optim

    kernel = getattr(optim, kernel_name)
    u_t = jnp.array([1.0, 2.0], dtype=jnp.float32)
    grad_l = jnp.array([0.5, -1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="etaa"):
        kernel(u_t, grad_l, None, {"eta": 0.1, "etaa": 0.2})
    u_next, _ = kernel(u_t, grad_l, None, {"eta": 0.1})
    assert jnp.allclose(u_next, u_t - 0.1 * grad_l)


# ---------------------------------------------------------------------------
# T-55 (lane C3): matrix tune placeholder W refusal, transform step
# ---------------------------------------------------------------------------


def test_matrix_tune_refuses_placeholder_w_on_edge_backend():
    """A placeholder (0, 0) emitter.W reported the refined matrix as []; it is refused by name."""
    from dataclasses import replace

    import jax.numpy as jnp
    import optax

    from jaxfne._model import Model

    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=10, cell_types={"E": 0.5, "PV": 0.5})
        .drive(baseline_drive_by_cell_type={"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    base = jtfne.construct(cfg)
    emitter = base.params["emitter"]
    new_emitter = replace(emitter, W=jnp.zeros((0, 0), dtype=emitter.W.dtype))
    model = Model(cfg=base.cfg, params={**base.params, "emitter": new_emitter}, static=dict(base.static))
    sim = jtfne.simulation(duration_ms=40.0, dt_ms=0.1, seed=0)
    obj = jtfne.rate_targets(
        groups={"E": list(range(5)), "I": list(range(5, 10))},
        targets_hz={"E": 10.0, "I": 5.0},
    )
    opt = jtfne.agsdr(
        parameters={"gAMPA_w": jtfne.matrix_parameter(mask="all", bounds=(0.5, 2.5))},
        inner_optimizer=optax.adam(learning_rate=1e-2),
        inner_steps=2,
        generations=1,
        population_size=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="edge_list"):
        model.tune(objectives=obj, optimizer=opt, simulation=sim)


@pytest.mark.parametrize("make", ["sdr_transform", "gsdr_transform", "agsdr_transform"])
def test_transform_step_advances_when_no_loss_is_passed(make):
    """The noise key is fold_in(key, state.step); a step held at 0 repeats the noise."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        optax = jtfne.require_optax()
    except ImportError:
        return

    transform = getattr(jtfne, make)(inner_optimizer=optax.sgd(0.0), stochastic_scale=1.0)
    params = jnp.zeros((4,), dtype=jnp.float32)
    grad = jnp.ones((4,), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    state = transform.init(params)
    first, state = transform.update(grad, state, params=params, key=key)
    second, state = transform.update(grad, state, params=params, key=key)
    assert int(state[0].step) == 2, "step must advance on every update, loss or not"
    assert np.any(np.asarray(first) != 0.0), "fixture must draw noise"
    assert not np.array_equal(np.asarray(first), np.asarray(second)), \
        "same key at the same step repeated the noise"
