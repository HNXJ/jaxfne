"""0.4.13 Pass 1: public semantic / API contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import jaxfne as jtfne
from jaxfne.public_surface import (
    ADVANCED_NAMESPACE,
    COMPATIBILITY_DEPRECATIONS,
    INTERNAL_HDP_RULE_IDS,
    PUBLIC_EXPORTS,
    PUBLIC_H_STATE_LOCALITIES,
    public_surface_summary,
    symbol_tier,
    validate_hdp_params_semantics,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "artifacts/public_surface_contract_v0413.json"


def test_public_exports_match_contract_module():
    assert set(jtfne.__all__) == set(PUBLIC_EXPORTS)


def test_contract_artifact_matches_module():
    assert CONTRACT_PATH.exists(), "artifacts/public_surface_contract_v0413.json required"
    artifact = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    summary = public_surface_summary()
    assert artifact["public_exports"] == summary["public_exports"]
    assert artifact["counts"]["public_exports"] == len(PUBLIC_EXPORTS)


def test_advanced_symbols_remain_importable_but_not_public():
    """Root discoverability contraction must not delete functionality."""
    assert "simulate_edge_recurrent_izhikevich" not in jtfne.__all__
    assert hasattr(jtfne, "simulate_edge_recurrent_izhikevich")
    assert symbol_tier("simulate_edge_recurrent_izhikevich") == "ADVANCED"
    assert ADVANCED_NAMESPACE["simulate_edge_recurrent_izhikevich"] == "jaxfne.emitters"

    assert "SanityDeltaConfig" not in jtfne.__all__
    assert hasattr(jtfne, "SanityDeltaConfig")
    assert symbol_tier("SanityDeltaConfig") == "EXPERIMENTAL_INTERNAL"

    assert "write_nwb" not in jtfne.__all__
    assert hasattr(jtfne, "write_nwb")


def test_glif_lif_emitters_are_experimental_not_compatibility():
    assert "GLIFEmitter" not in jtfne.__all__
    assert "LIFEmitter" not in jtfne.__all__
    assert hasattr(jtfne, "GLIFEmitter")
    assert symbol_tier("GLIFEmitter") == "EXPERIMENTAL_INTERNAL"


def test_compatibility_aliases_retained_in_public_exports():
    for name in ("Net", "Config", "AGSDR", "construct_neuronal_tensor", "load_neuronal_tensor"):
        assert name in jtfne.__all__
        assert name in COMPATIBILITY_DEPRECATIONS


def test_population_vector_restoring_is_internal_dispatch_only():
    assert "population_vector_restoring" in INTERNAL_HDP_RULE_IDS
    assert "population_vector_restoring" not in jtfne.__all__
    issues = validate_hdp_params_semantics(
        {
            "hdp_rule": "population_vector_restoring",
            "h_state_locality": "population",
            "controller_B": [[0.0, 0.0], [0.0, 0.0]],
            "m_ei_edge_mask": [True],
        }
    )
    assert any("internal dispatch identifier" in msg for msg in issues)


def test_public_h_state_localities():
    assert PUBLIC_H_STATE_LOCALITIES == frozenset({"node", "population"})
    issues = validate_hdp_params_semantics({"h_state_locality": "per_neuron"})
    assert any("h_state_locality must be one of" in msg for msg in issues)


def test_public_symbol_count_contraction_from_baseline():
    """259 baseline → 189 public exports (+5 JDNA additive surface 0.4.17;
    −2 SurrogateConfig pair re-classified EXPERIMENTAL_INTERNAL on 2026-08-22
    W4: declaration-only dormant metadata, zero manuscript/example/doc usage)."""
    summary = public_surface_summary()
    assert summary["counts"]["baseline_all"] == 265
    assert summary["counts"]["public_exports"] == 189
    assert summary["counts"]["compatibility"] == 13
    assert summary["counts"]["experimental_internal"] == 18


def test_surrogate_config_pair_is_experimental_not_public():
    """W4 reclassification: declaration-only surrogate metadata is fenced at
    its earned tier while remaining importable (contraction idiom)."""
    assert "SurrogateConfig" not in jtfne.__all__
    assert "surrogate_config" not in jtfne.__all__
    assert hasattr(jtfne, "SurrogateConfig")
    assert hasattr(jtfne, "surrogate_config")
    assert symbol_tier("SurrogateConfig") == "EXPERIMENTAL_INTERNAL"
    assert symbol_tier("surrogate_config") == "EXPERIMENTAL_INTERNAL"
    # construction semantics unchanged
    s = jtfne.surrogate_config(method="straight_through", beta=8.0)
    assert s is not None
