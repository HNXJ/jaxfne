"""Small, fast tests for the "hebbian_pairwise" conductance_rule (jaxfne/
emitters_homeostatic_ei.py): independent gains per population pair
(E-E/E-I/I-E/I-I), extending this session's E<->I cross-population coupling
work one level down -- from the homeostasis_rule (H) level
(cubic_penalty_coupled) to the conductance_rule (G) level.

User-confirmed scope (AskUserQuestion): 4 fixed, independently-settable gain
constants via a factory (`make_hebbian_pairwise_rule`), not a self-adjusting
controller.
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import (
    make_minimal_ei_params,
    make_hebbian_pairwise_rule,
    simulate_homeostatic_ei,
)

N_STEPS = 300
DT_MS = 0.5


def test_hebbian_pairwise_default_gains_matches_plain_hebbian():
    """All 4 gains at their default of 1.0 must be mathematically identical
    to plain 'hebbian' -- the gains are the only thing this rule adds."""
    params = make_minimal_ei_params(4)
    common = dict(
        params=params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", homeostasis_rule="linear",
    )
    _, _, _, G_hebbian, _, diag_a = simulate_homeostatic_ei(conductance_rule="hebbian", **common)
    _, _, _, G_pairwise, _, diag_b = simulate_homeostatic_ei(conductance_rule="hebbian_pairwise", **common)
    assert not bool(diag_a["error"]) and not bool(diag_b["error"])
    assert bool(jnp.allclose(G_hebbian, G_pairwise, atol=1e-6)), (
        f"default-gain hebbian_pairwise must match plain hebbian, "
        f"max diff={float(jnp.max(jnp.abs(G_hebbian - G_pairwise)))}"
    )


def test_distinct_gains_produce_a_measurably_different_trajectory():
    """A custom-gain closure (k_ei != k_ie, both != default) must produce a
    real, measurable difference from the default-gain rule -- confirms the
    gains actually take effect, not just that the factory constructs
    without error."""
    params = make_minimal_ei_params(4)  # 3 E (idx 0-2), 1 I (idx 3)
    common = dict(
        params=params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", homeostasis_rule="linear",
    )
    _, _, _, G_default, _, diag_a = simulate_homeostatic_ei(conductance_rule="hebbian_pairwise", **common)
    custom_rule = make_hebbian_pairwise_rule(k_ee=1.0, k_ei=5.0, k_ie=0.2, k_ii=1.0)
    _, _, _, G_custom, _, diag_b = simulate_homeostatic_ei(conductance_rule=custom_rule, **common)
    assert not bool(diag_a["error"]) and not bool(diag_b["error"])
    max_diff = float(jnp.max(jnp.abs(G_default[-1] - G_custom[-1])))
    assert max_diff > 0.05, f"distinct pairwise gains should produce a real difference, got max_diff={max_diff}"


def test_hebbian_pairwise_gain_mask_matches_pre_post_convention():
    """Direct, single-step check of the gain mask itself (no simulation):
    G[i,j] is the edge from presynaptic j to postsynaptic i (this module's
    existing convention -- G @ x sums over j in every activation_rule). With
    is_e=[True, False] (neuron 0=E, neuron 1=I) and x=H=ones, dG_ij (before
    the -G decay term) should equal exactly the gain for that (pre,post)
    pair, since H_i*x_i*x_j=1 for all i,j here."""
    is_e = jnp.array([True, False])
    x = jnp.ones(2)
    H = jnp.ones(2)
    G = jnp.zeros((2, 2))
    rule = make_hebbian_pairwise_rule(k_ee=2.0, k_ei=3.0, k_ie=5.0, k_ii=7.0)
    dG = rule(x, H, G, is_e)
    # dG[i,j] = gain(pre=j,post=i)*1 - 0
    assert float(dG[0, 0]) == 2.0   # pre=E(0) -> post=E(0): k_ee
    assert float(dG[1, 0]) == 3.0   # pre=E(0) -> post=I(1): k_ei
    assert float(dG[0, 1]) == 5.0   # pre=I(1) -> post=E(0): k_ie
    assert float(dG[1, 1]) == 7.0   # pre=I(1) -> post=I(1): k_ii
