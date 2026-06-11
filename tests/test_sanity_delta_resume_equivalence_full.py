"""Tests for the resume and restore equivalence in full simulation mode."""

import jax.numpy as jnp
import jaxfne as jtfne

def test_resume_equivalence_full_mode():
    """Verify that stopping and resuming from backup produces equivalent trajectory."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(runtime_mode="full")
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    backup = model.initialize_backup(paradigm)
    
    # Run continuous
    episode_continuous = model.run_task(paradigm, paradigm.make_fixation_gate(), backup)
    
    # Resume at d4 segment
    episode_resumed = episode_continuous.resume(from_segment="d4")
    
    # Find start index of d4 segment
    schedule = paradigm.to_schedule()
    start_ms = 0.0
    for seg in schedule["segments"]:
        if seg["segment_id"] == "d4":
            start_ms = seg["start_ms"]
            break
            
    step_idx = int(start_ms / cfg.dt_ms)
    
    # Check that after resume point, the state is equivalent
    diff_vm = jnp.max(jnp.abs(episode_continuous.vm[step_idx:] - episode_resumed.vm[step_idx:]))
    diff_spk = jnp.sum(jnp.abs(episode_continuous.spikes[step_idx:] - episode_resumed.spikes[step_idx:]))
    
    # The trajectory should be identical
    assert diff_vm < 1e-9, f"Voltage mismatch: {diff_vm}"
    assert diff_spk == 0, f"Spike mismatch count: {diff_spk}"
