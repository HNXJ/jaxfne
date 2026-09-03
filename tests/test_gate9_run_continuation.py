"""Gate 9: Canonical run_continuation export and behavioral invariance test.

Verifies that:
1. jaxfne.run_continuation is exported at root.
2. jaxfne.run_continuation is the canonical object from jaxfne._pipeline, not a divergent wrapper.
3. Continuation behavior and output match existing Model.simulate(..., continuation=...) bit-exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import (
    ContinuationState,
    compile_step_fn,
    dynamic_state_from_model,
    run_continuation,
)
from jaxfne._pipeline import run_continuation as _internal_run_continuation


def test_gate9_run_continuation_is_canonical_identity():
    """Verify jaxfne.run_continuation is strictly identical to _pipeline.run_continuation."""
    assert hasattr(jtfne, "run_continuation")
    assert jtfne.run_continuation is _internal_run_continuation
    assert "run_continuation" in jtfne.__all__


def test_gate9_run_continuation_execution_matches_model_simulate():
    """Verify run_continuation execution step-by-step matches Model.simulate continuation."""
    cfg = jtfne.suite2_net1_config(seed=42, n=4, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(
        duration_ms=10.0,
        dt_ms=0.5,
        seed=101,
        record_sources=True,
        record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
    )

    # 1. Run baseline via Model.simulate with return_state=True
    sig_base, state_out = jtfne.simulate(model, sim, return_state=True)

    # 2. Re-run manually using exported primitives
    dyn = dynamic_state_from_model(model)
    init_state = ContinuationState(dynamic=dyn, prng_key=jax.random.PRNGKey(101), step_index=0)
    step_fn, _ = compile_step_fn(model, dt_ms=0.5, kernel="baseline")
    drive = jnp.zeros((sim.n_steps, 4), dtype=jnp.float32)

    next_state, outputs = run_continuation(step_fn, init_state, drive)

    # Invariance check
    np.testing.assert_array_equal(np.asarray(sig_base.V_m), np.asarray(outputs[0]))
    np.testing.assert_array_equal(np.asarray(sig_base.spikes), np.asarray(outputs[1]))
    np.testing.assert_array_equal(np.asarray(sig_base.sources), np.asarray(outputs[2]))
    assert next_state.step_index == state_out.step_index
