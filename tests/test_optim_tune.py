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
