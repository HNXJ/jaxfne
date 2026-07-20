"""Small, fast tests for `_soft_bound` and `bound_mode="stable"` (jaxfne/
emitters_homeostatic_ei.py).

Motivation: found while rendering N=8/N=16 rasters for the user -- the
canonical circuit's `x` state has no bound of any kind, and explicit Euler on
the cubic activation term overshoots once `|x|` exceeds a real, derived
numerical-stability radius (`|x| <~ 2.58` at this circuit's dt_x), producing
a genuine, reproducible-across-seeds divergence to NaN at N=16 with the
shipped flat G_max=5.0 default. A hard `jnp.clip` (bound_mode="minimal",
already used for G/H) is a *force*, not a *bound* -- it can still be outrun
by a single large enough step. `_soft_bound` (a tanh remap) is a bounded
*codomain*: mathematically incapable of producing a non-finite value from
any finite input, regardless of step size, N, or gain.
"""
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.emitters_homeostatic_ei import (
    _soft_bound,
    make_minimal_ei_params,
    simulate_homeostatic_ei,
)


def test_soft_bound_stays_within_range_for_extreme_inputs():
    """The guarantee is finite-and-within-[lo,hi] (a closed interval), not a
    strict open interval: at large enough |v|, tanh saturates to exactly
    +-1.0 in float32, so soft_bound correctly returns exactly lo/hi rather
    than something strictly interior -- verified empirically (soft_bound
    saturates to exactly 10.0 at v=1e6 already), not assumed."""
    lo, hi = jnp.array(0.1), jnp.array(10.0)
    for v in (jnp.array(-1e30), jnp.array(-1.0), jnp.array(5.0), jnp.array(1e6), jnp.array(1e30)):
        out = _soft_bound(v, lo, hi)
        assert bool(jnp.isfinite(out)), f"soft_bound({float(v)}) must be finite, got {out}"
        assert bool((out >= lo) & (out <= hi)), f"soft_bound({float(v)}) = {out} left [{lo}, {hi}]"


def test_soft_bound_is_near_linear_close_to_the_midpoint():
    lo, hi = jnp.array(0.1), jnp.array(10.0)
    mid = (lo + hi) / 2.0
    small_offset = jnp.array(0.05)  # tiny relative to the (0.1,10.0) range
    out = _soft_bound(mid + small_offset, lo, hi)
    # tanh(z) ~= z for small z, so soft_bound(mid+eps) ~= mid+eps near the middle.
    assert abs(float(out - (mid + small_offset))) < 1e-3


def test_bound_mode_stable_keeps_n16_finite_with_the_flat_g_max_that_previously_diverged():
    """Reproduces the exact failure condition found this session: flat
    G_max=5.0 (jaxfne/_construct.py's shipped canonical default, NOT the
    N-scaled 10.0/n this module's own make_minimal_ei_params defaults to) at
    N=16 reliably diverges to NaN around 210-221ms under bound_mode="minimal"
    -- verified reproducible across 3 seeds before this fix landed. Confirms
    bound_mode="stable" resolves it with the SAME G_max, not by getting the
    N-scaling right."""
    params = make_minimal_ei_params(16, G_max=5.0, x_min=-10.0, x_max=10.0)
    for seed in (0, 1, 2):
        V_m, _, _, _, H_hist, diag = simulate_homeostatic_ei(
            params, n_steps=2000, dt_ms=0.5, key=jax.random.PRNGKey(seed),
            activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
            bound_mode="stable",
        )
        assert not bool(diag["error"]), f"seed={seed}: bound_mode='stable' should not diverge"
        assert bool(jnp.all(jnp.isfinite(V_m)))
        assert bool(jnp.all(jnp.isfinite(H_hist)))


def test_bound_mode_minimal_still_reproduces_the_n16_divergence():
    """Negative control: confirms the test above is actually discriminating
    (bound_mode="minimal" with the same flat G_max=5.0 still fails), not
    passing vacuously because N=16 never diverges under any configuration."""
    params = make_minimal_ei_params(16, G_max=5.0, x_min=-10.0, x_max=10.0)
    V_m, _, _, _, _, diag = simulate_homeostatic_ei(
        params, n_steps=2000, dt_ms=0.5, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
        bound_mode="minimal",
    )
    assert bool(diag["error"]), "expected bound_mode='minimal' to still diverge with flat G_max=5.0 at N=16"


def test_bound_mode_default_is_minimal_and_unchanged_for_existing_callers():
    """Backward-compatibility check: omitting bound_mode entirely must give
    bit-identical results to explicitly passing bound_mode="minimal"."""
    params = make_minimal_ei_params(8)
    args = dict(
        params=params, n_steps=500, dt_ms=0.5, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
    )
    out_default = simulate_homeostatic_ei(**args)
    out_explicit = simulate_homeostatic_ei(**args, bound_mode="minimal")
    for a, b in zip(out_default[:5], out_explicit[:5]):
        assert bool(jnp.array_equal(a, b))


def test_bound_mode_reachable_via_configuration_construct_model_simulate():
    """bound_mode must be reachable through the full jaxfne-objective-grammar
    chain (Configuration -> construct() -> Model.simulate() -> Signals), not
    just via a direct simulate_homeostatic_ei() call -- found missing
    (2026-07-16) when checking whether this session's new homeostatic_ei
    features are actually wired into the grammar; fixed in the same pass
    (Configuration.set_emitter's new bound_mode kwarg, HomeostaticEIParams's
    new bound_mode field, Model._simulate_homeostatic_ei forwarding it)."""
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=200.0, dt_ms=0.5)
        .network(name="ei16", n=16)
        .set_emitter("homeostatic_ei", bound_mode="stable")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    model = jtfne.construct(cfg)
    assert model.params["emitter"].bound_mode == "stable"

    sim = jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=0)
    signals = model.simulate(sim)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(signals.V_m))))
    assert signals.metadata["hdp"]["rules"]["bound_mode"] == "stable"

    # JSON-safety: Configuration must stay JSON-safe with the new kwarg.
    jtfne.io.json_safe(cfg.metadata)
    jtfne.io.config_hash(cfg)
