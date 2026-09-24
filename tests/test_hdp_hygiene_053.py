"""0.5.3 item 7b (ENGINE W2): HDP parameter hygiene, adversarial (H5/H7).

Fail-closed policy:
- ``validate_hdp_params_semantics``: non-dict input reports an issue in
  BOTH modes (no silent pass in non-strict); strict raises.
- ``compile_step_fn``: unknown ``**hdp_kwargs`` keys raise at compile time;
  ``hdp_rule_params`` is checked per-descriptor and refused alongside a
  legacy (non-registered) rule.
- ``hdp_rule_params`` and registered rule names are KNOWN-good: the
  validator must not false-positive on the configs item-5 controls rely on.
"""

from __future__ import annotations

import pytest

import jaxfne as jtfne
from jaxfne._pipeline import compile_step_fn
from jaxfne.hdp_rule import check_hdp_rule_params
from jaxfne.public_surface import validate_hdp_params_semantics as validate


@pytest.fixture(scope="module")
def model():
    return jtfne.construct(jtfne.suite2_net1_config(seed=11, n=8))


# --- validate_hdp_params_semantics: non-dict fails closed --------------------


@pytest.mark.parametrize("bad", [["K_HDP"], "K_HDP", None, 42, {"x"}.__iter__()])
def test_non_dict_fails_closed_non_strict(bad):
    issues = validate(bad)
    assert len(issues) == 1
    assert "must be a dict" in issues[0]


@pytest.mark.parametrize("bad", [["K_HDP"], "K_HDP", None, 42])
def test_non_dict_strict_raises(bad):
    with pytest.raises(ValueError, match="must be a dict"):
        validate(bad, strict=True)


# --- unknown keys ------------------------------------------------------------


def test_unknown_keys_flagged_non_strict():
    issues = validate({"K_HDP": 0.0, "K_HDP_typo": 0.1, "k_h": 0.5})
    assert any("unrecognized keys" in m and "K_HDP_typo" in m for m in issues)
    assert any("k_h" in m for m in issues)


def test_unknown_keys_strict_raises():
    with pytest.raises(ValueError, match="unrecognized keys"):
        validate({"K_HDP_typo": 0.1}, strict=True)


# --- malformed values --------------------------------------------------------


def test_bad_locality_flagged():
    issues = validate({"h_state_locality": "per_neuron"})
    assert any("h_state_locality must be one of" in m for m in issues)
    with pytest.raises(ValueError, match="h_state_locality must be one of"):
        validate({"h_state_locality": "per_neuron"}, strict=True)


def test_unknown_rule_name_flagged():
    issues = validate({"hdp_rule": "stdp_classic"})
    assert any("registered rule" in m and "stdp_classic" in m for m in issues)
    with pytest.raises(ValueError, match="stdp_classic"):
        validate({"hdp_rule": "stdp_classic"}, strict=True)


def test_internal_dispatch_id_still_refused():
    issues = validate(
        {
            "hdp_rule": "population_vector_restoring",
            "h_state_locality": "population",
            "controller_B": [[1.0, 0.0], [0.0, 1.0]],
            "m_ei_edge_mask": [True],
        }
    )
    assert any("internal dispatch identifier" in m for m in issues)


def test_population_missing_theta_keys_flagged():
    issues = validate({"h_state_locality": "population"})
    assert any("theta-adaptation keys" in m for m in issues)


# --- known-good surface (no false positives for item-5 controls) -------------


def test_registered_rule_and_params_are_known():
    assert validate({"hdp_rule": "synthetic_presyn_gain"}) == []
    assert (
        validate(
            {
                "hdp_rule": "synthetic_presyn_gain",
                "hdp_rule_params": {"k_h": 0.1, "k_w": 0.0, "gamma": 0.0},
            }
        )
        == []
    )


def test_builtin_rules_and_defaults_are_clean():
    assert validate({}) == []
    assert validate({"hdp_rule": "signed_linear", "K_HDP": 0.0}) == []


# --- check_hdp_rule_params ---------------------------------------------------


def test_rule_params_typo_raises_not_silent_default():
    with pytest.raises(ValueError, match="unrecognized keys"):
        check_hdp_rule_params("synthetic_presyn_gain", {"k_h": 0.1, "k_ww": 0.2})


def test_rule_params_non_mapping_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        check_hdp_rule_params("synthetic_presyn_gain", ["k_h"])


def test_rule_params_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown registered hdp_rule"):
        check_hdp_rule_params("no_such_rule", {})


# --- compile_step_fn boundary ------------------------------------------------


def test_compile_hdp_rejects_unknown_key(model):
    with pytest.raises(ValueError, match="unrecognized hdp_kwargs keys"):
        compile_step_fn(model, dt_ms=0.5, kernel="hdp", K_HDP_typo=0.1)


def test_compile_hdp_typo_never_simulates_as_default(model):
    """Adversarial H5: a typo'd gain must raise, not run with the default."""
    with pytest.raises(ValueError, match="K_HDP_typo"):
        compile_step_fn(model, dt_ms=0.5, kernel="hdp", K_HDP=0.0, K_HDP_typo=0.5)


def test_compile_baseline_rejects_plasticity_keys(model):
    with pytest.raises(ValueError, match="unrecognized hdp_kwargs keys"):
        compile_step_fn(model, dt_ms=0.5, kernel="baseline", K_HDP=0.0)


def test_compile_rejects_call_site_owned_keys(model):
    with pytest.raises(ValueError, match="unrecognized hdp_kwargs keys"):
        compile_step_fn(model, dt_ms=0.5, kernel="hdp", step_indices=0)
    with pytest.raises(ValueError, match="unrecognized hdp_kwargs keys"):
        compile_step_fn(model, dt_ms=0.5, kernel="hdp", dtype="float32")


def test_compile_rejects_rule_params_with_legacy_rule(model):
    with pytest.raises(ValueError, match="requires a registered hdp_rule"):
        compile_step_fn(
            model,
            dt_ms=0.5,
            kernel="hdp",
            hdp_rule="signed_linear",
            hdp_rule_params={"k_h": 0.1},
        )


def test_compile_rejects_unknown_rule_params(model):
    with pytest.raises(ValueError, match="unrecognized keys"):
        compile_step_fn(
            model,
            dt_ms=0.5,
            kernel="hdp",
            hdp_rule="synthetic_presyn_gain",
            hdp_rule_params={"k_h": 0.1, "k_ww": 0.2},
        )


def test_compile_accepts_known_hdp_surface(model):
    step_fn, init = compile_step_fn(
        model,
        dt_ms=0.5,
        kernel="hdp",
        K_HDP=0.0,
        hdp_rule="synthetic_presyn_gain",
        hdp_rule_params={"k_h": 0.1, "k_w": 0.0, "gamma": 0.0},
    )
    assert step_fn is not None and init is not None


def test_compile_accepts_baseline_noise_only(model):
    step_fn, init = compile_step_fn(model, dt_ms=0.5, kernel="baseline", noise_scale=0.0)
    assert step_fn is not None and init is not None
