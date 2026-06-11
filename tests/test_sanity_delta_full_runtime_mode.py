"""Tests verifying the behavior of the full runtime simulation mode."""

import jax.numpy as jnp
import jaxfne as jtfne

def test_full_runtime_mode_outputs():
    """Verify that full runtime mode produces finite non-zero outputs and correct shapes."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(runtime_mode="full")
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    backup = model.initialize_backup(paradigm)
    
    episode = model.run_task(paradigm, paradigm.make_fixation_gate(), backup)
    
    assert episode.runtime_mode == "full"
    
    # Check shapes
    assert episode.vm.shape == (episode.model.n_steps, episode.model.n_neurons)
    assert episode.spikes.shape == (episode.model.n_steps, episode.model.n_neurons)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(episode.vm))
    assert jnp.all(jnp.isfinite(episode.spikes))
    
    # Check that vm has dynamic, non-default values (it evolved)
    assert not jnp.all(episode.vm == -65.0)
    
    # Check that spikes occurred
    assert jnp.sum(episode.spikes) > 0
    
    # Check proxy readouts
    assert "lfp_proxy" in episode.source_terms
    assert "csd_proxy" in episode.source_terms
    assert "eeg_proxy" in episode.source_terms
    assert "meg_proxy" in episode.source_terms
    
    assert episode.source_terms["lfp_proxy"].shape == (episode.model.n_steps, 20)
    assert episode.source_terms["csd_proxy"].shape == (episode.model.n_steps, 20)
    assert episode.source_terms["eeg_proxy"].shape == (episode.model.n_steps, 16)
    assert episode.source_terms["meg_proxy"].shape == (episode.model.n_steps, 16)
