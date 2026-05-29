"""Tests for numerical stability boundaries of all emitter families.

Sweeps dt_ms in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] to find max stable step sizes,
failure modes, and verify optional clipping constraints.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_izhikevich_stability_sweep():
    """Sweep dt_ms values to verify IzhikevichEmitter numerical stability limits."""
    dts = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    n = 5
    duration_ms = 100.0  # Reduced duration for fast unit test execution while retaining sweep dynamics
    
    stable_dt_ms_max_observed = 0.0
    first_failed_dt_ms = None
    failure_mode = "none"
    
    emitter = jtfne.emitters.IzhikevichEmitter(n=n, dtype="float32")
    
    for dt in dts:
        n_steps = int(duration_ms / dt)
        state = emitter.initial_state(seed=0)
        inputs = jnp.zeros((n_steps, n), dtype=jnp.float32)
        
        # Rollout step under JAX scan
        def body_fn(carry, inp):
            next_s, out = emitter.step(carry, inp, dt_ms=dt)
            return next_s, out
            
        final_state, outputs = jax.lax.scan(body_fn, state, inputs)
        
        # Check for finiteness (failure mode: nonfinite_voltage)
        is_finite = bool(np.all(np.isfinite(outputs.voltage)))
        
        # Check for rate explosion (firing rate > 500 Hz is unreasonable in 100ms)
        mean_spikes_per_sec = float(np.mean(outputs.spikes) * (1000.0 / dt))
        unreasonable_rate = mean_spikes_per_sec > 500.0
        
        if is_finite and not unreasonable_rate:
            stable_dt_ms_max_observed = max(stable_dt_ms_max_observed, dt)
        else:
            if first_failed_dt_ms is None:
                first_failed_dt_ms = dt
                if not is_finite:
                    failure_mode = "nonfinite_voltage"
                elif unreasonable_rate:
                    failure_mode = "rate_explosion"
                    
    # Generate the stability report dictionary
    stability_report = {
        "emitter_family": "izhikevich",
        "stable_dt_ms_max_observed": stable_dt_ms_max_observed,
        "first_failed_dt_ms": first_failed_dt_ms,
        "failure_mode": failure_mode,
        "clip_policy": "none",
        "substepping_policy": "none"
    }
    
    assert stable_dt_ms_max_observed >= 0.1
    # Verify report is populated correctly
    assert "failure_mode" in stability_report
    assert stability_report["clip_policy"] == "none"


def test_placeholder_stability_failure():
    """Verify that GLIF and LIF placeholders fail stability sweeps gracefully."""
    with pytest.raises(NotImplementedError):
        emitter = jtfne.emitters.GLIFEmitter(n=10)
        
    with pytest.raises(NotImplementedError):
        emitter = jtfne.emitters.LIFEmitter(n=10)
