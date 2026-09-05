"""0.4.13 Pass 1: public semantic / API contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import jaxfne as jtfne
from jaxfne.public_surface import (
    _ADVANCED,
    _ALL_CLASSIFIED,
    _CANONICAL,
    _COMPATIBILITY,
    _EXPERIMENTAL_INTERNAL,
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
    assert summary["counts"]["baseline_all"] == 266
    assert summary["counts"]["public_exports"] == 190
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


# --- Contract artifact drift gates -------------------------------------------
#
# The artifact previously drifted from the module in two places at once
# (counts.baseline_all recorded 265 against a live 266, and hdp_param_groups
# lost the H-boundary-stabilization keys) because only `public_exports` was
# ever compared. These gates hold every field, so the artifact cannot go stale
# silently again. Regenerate with scripts/generate_public_surface_contract.py.

def _artifact() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_artifact_every_count_field_matches_live():
    """Every count field -- not just public_exports -- must match the module."""
    live = public_surface_summary()["counts"]
    stored = _artifact()["counts"]
    assert set(stored) == set(live), (
        f"count field set drift: artifact-only={sorted(set(stored) - set(live))}, "
        f"live-only={sorted(set(live) - set(stored))}"
    )
    mismatched = {k: (stored[k], live[k]) for k in live if stored[k] != live[k]}
    assert not mismatched, f"count drift (artifact, live): {mismatched}"


def test_contract_artifact_matches_generator_output_exactly():
    """The tracked artifact must be exactly what the generator emits."""
    from scripts.generate_public_surface_contract import render

    assert CONTRACT_PATH.read_text(encoding="utf-8") == render(), (
        "artifacts/public_surface_contract_v0413.json is stale; regenerate with "
        "python scripts/generate_public_surface_contract.py"
    )


def test_contract_artifact_non_count_fields_match_live():
    """Payload fields beyond counts must also track the module."""
    live = public_surface_summary()
    stored = _artifact()
    for field in (
        "schema",
        "version",
        "public_exports",
        "compatibility_deprecations",
        "hdp_param_groups",
        "internal_hdp_rule_ids",
        "public_h_state_localities",
    ):
        assert stored[field] == live[field], f"artifact field {field!r} drifted from the module"


def test_counts_are_internally_arithmetically_consistent():
    """baseline_all must equal the sum of the four tier counts, in both places.

    The stale artifact was self-inconsistent: 177+58+13+18 = 266 while it
    recorded baseline_all = 265.
    """
    for label, counts in (("live", public_surface_summary()["counts"]), ("artifact", _artifact()["counts"])):
        tier_sum = (
            counts["canonical"]
            + counts["advanced"]
            + counts["compatibility"]
            + counts["experimental_internal"]
        )
        assert counts["baseline_all"] == tier_sum, (
            f"{label} counts inconsistent: tiers sum to {tier_sum} but "
            f"baseline_all = {counts['baseline_all']}"
        )


def test_tier_sets_are_disjoint():
    """The four tiers partition the classified baseline; no symbol has two tiers.

    Disjointness is what makes the arithmetic check above meaningful -- an
    overlap would make |union| < sum without either number being wrong.
    """
    tiers = {
        "CANONICAL": _CANONICAL,
        "ADVANCED": _ADVANCED,
        "COMPATIBILITY": _COMPATIBILITY,
        "EXPERIMENTAL_INTERNAL": _EXPERIMENTAL_INTERNAL,
    }
    names = sorted(tiers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = tiers[a] & tiers[b]
            assert not overlap, f"symbols classified in both {a} and {b}: {sorted(overlap)}"

    assert len(_ALL_CLASSIFIED) == sum(len(t) for t in tiers.values())
    assert _ALL_CLASSIFIED == set().union(*tiers.values())


def test_public_exports_partition_matches_canonical_plus_compatibility():
    """PUBLIC_EXPORTS is exactly CANONICAL | COMPATIBILITY, with no leakage."""
    assert set(PUBLIC_EXPORTS) == set(_CANONICAL) | set(_COMPATIBILITY)
    assert not set(PUBLIC_EXPORTS) & set(_ADVANCED)
    assert not set(PUBLIC_EXPORTS) & set(_EXPERIMENTAL_INTERNAL)
