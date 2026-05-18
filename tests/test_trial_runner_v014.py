"""Tests for v0.0.14 Sequential Trial Runner.

Tests A-I from ALPHA plan:
A. trial_batch factory creates expected number of trials with deterministic IDs.
B. run_trials produces TrialBatchResult with expected count.
C. Each TrialResult carries unique seed.
D. TrialResult correctly maps ParadigmCondition name.
E. serialization: to_dict() returns JSON-safe dict.
F. serialization: to_dict() excludes large arrays (V_m, spikes, sources, field).
G. error handling: exception in simulate() is captured in TrialResult(success=False).
H. metadata persistence: TrialSpec.metadata flows to TrialResult.metadata.
I. consistency: run_trials results are identical to manual simulate() calls with same seeds.
"""

import jaxfne as jtfne
import pytest
import jax.numpy as jnp
from dataclasses import replace
from jaxfne.core import ParadigmCondition


def _bad_condition() -> ParadigmCondition:
    """Return a condition whose events=None will raise inside simulate(), not at construction."""
    return ParadigmCondition(
        name="bad_test_condition",
        sequence=(),
        omission_position=None,
        probability=None,
        condition_numbers=(99,),
        events=None,
    )

def test_a_trial_batch_factory():
    paradigm = jtfne.standard_visual_omission()
    conditions = paradigm.conditions[:3]  # AAAB, AXAB, AAXB
    n_reps = 2
    batch = jtfne.trial_batch(conditions, n_reps=n_reps, seed=100)
    
    assert len(batch.trials) == 6
    assert batch.trials[0].trial_id == "trial_0000_AAAB"
    assert batch.trials[1].trial_id == "trial_0001_AXAB"
    assert batch.trials[3].trial_id == "trial_0003_AAAB"
    assert batch.trials[0].seed == 100
    # paired_by_replicate default: rep 1 → base_seed + 1 = 101
    assert batch.trials[5].seed == 101
    assert batch.trials[0].metadata["rep"] == 0
    assert batch.trials[3].metadata["rep"] == 1

def test_b_run_trials_basic():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    paradigm = jtfne.standard_visual_omission()
    conditions = paradigm.conditions[:2]
    batch = jtfne.trial_batch(conditions, n_reps=1, seed=42)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)
    
    batch_res = model.run_trials(batch, sim)
    assert isinstance(batch_res, jtfne.TrialBatchResult)
    assert len(batch_res.results) == 2
    assert batch_res.results[0].success is True
    assert batch_res.results[0].signals is not None

def test_c_unique_seeds():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    paradigm = jtfne.standard_visual_omission()
    conditions = [paradigm.conditions[0]] * 2
    # Use unique_per_trial to guarantee different seeds per trial
    batch = jtfne.trial_batch(conditions, n_reps=1, seed=42, seed_policy="unique_per_trial")
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)

    batch_res = model.run_trials(batch, sim)
    # Seeds 42 and 43 should produce different results
    res1 = batch_res.results[0].signals.V_m
    res2 = batch_res.results[1].signals.V_m
    assert not jnp.allclose(res1, res2)

def test_d_condition_mapping():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    paradigm = jtfne.standard_visual_omission()
    batch = jtfne.trial_batch(paradigm.conditions[:2], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)
    
    batch_res = model.run_trials(batch, sim)
    assert batch_res.results[0].condition_label == "AAAB"
    assert batch_res.results[1].condition_label == "AXAB"

def test_e_serialization_json_safe():
    paradigm = jtfne.standard_visual_omission()
    batch = jtfne.trial_batch(paradigm.conditions[:1], n_reps=1)
    d = batch.to_dict()
    assert isinstance(d, dict)
    assert d["batch_id"] == batch.batch_id
    assert len(d["trials"]) == 1

def test_f_serialization_array_exclusion():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    paradigm = jtfne.standard_visual_omission()
    batch = jtfne.trial_batch(paradigm.conditions[:1], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)
    
    batch_res = model.run_trials(batch, sim)
    d = batch_res.to_dict()
    res0 = d["results"][0]
    
    assert "signals" in res0
    # signals in TrialResult.to_dict() should be the summary dict
    assert isinstance(res0["signals"], dict)
    assert "V_m" not in res0["signals"]
    assert "spikes" not in res0["signals"]
    assert "n_steps" in res0["signals"]

def test_g_error_handling():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    # Use a condition with events=None to trigger an error inside simulate(), not at construction.
    batch = jtfne.trial_batch([_bad_condition()], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)

    # collect_errors=True records the failure
    batch_res = model.run_trials(batch, sim, collect_errors=True)
    assert batch_res.results[0].success is False
    assert batch_res.results[0].error_message is not None
    assert batch_res.results[0].signals is None

def test_h_metadata_persistence():
    cond = jtfne.standard_visual_omission().conditions[0]
    trial = jtfne.TrialSpec(trial_id="T1", condition=cond, seed=1, metadata={"user_tag": "test"})
    batch = jtfne.TrialBatch(trials=(trial,), batch_id="B1", metadata={"batch_tag": "test_batch"})
    
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)
    
    batch_res = model.run_trials(batch, sim)
    assert batch_res.metadata["batch_tag"] == "test_batch"
    assert batch_res.results[0].metadata["user_tag"] == "test"

def test_i_consistency():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    cond = jtfne.standard_visual_omission().conditions[0]
    seed = 123
    
    # Manual simulation
    sim_manual = jtfne.simulation(duration_ms=20.0, dt_ms=0.5, seed=seed)
    signals_manual = model.simulate(sim_manual, paradigm=cond)
    
    # Trial runner
    batch = jtfne.trial_batch([cond], n_reps=1, seed=seed)
    sim_batch = jtfne.simulation(duration_ms=20.0, dt_ms=0.5)
    batch_res = model.run_trials(batch, sim_batch)
    signals_batch = batch_res.results[0].signals
    
    assert jnp.allclose(signals_manual.V_m, signals_batch.V_m)
    assert jnp.allclose(signals_manual.spikes, signals_batch.spikes)

def test_j_module_level_run_trials():
    cfg = jtfne.configuration().network(n=5).emitter().field().probe()
    model = jtfne.construct(cfg)
    cond = jtfne.standard_visual_omission().conditions[0]

    # Test module-level run_trials function
    batch = jtfne.trial_batch([cond], n_reps=1, seed=42)
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.5)

    # Call module-level function
    result = jtfne.run_trials(model, batch, sim)

    # Verify it returns TrialBatchResult
    assert isinstance(result, jtfne.TrialBatchResult)
    assert len(result.results) == 1
    assert result.results[0].success is True

def test_k_collect_errors_true():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    # Use bad condition (events=None) to trigger error inside simulate, not at construction.
    batch = jtfne.trial_batch([_bad_condition()], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)

    # With collect_errors=True, should return TrialBatchResult with failed trial
    result = model.run_trials(batch, sim, collect_errors=True)
    assert isinstance(result, jtfne.TrialBatchResult)
    assert len(result.results) == 1
    assert result.results[0].success is False
    assert result.results[0].error_message is not None

def test_l_collect_errors_false():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    batch = jtfne.trial_batch([_bad_condition()], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)

    # With collect_errors=False (default), should raise immediately
    with pytest.raises(Exception):
        model.run_trials(batch, sim, collect_errors=False)

def test_m_module_level_run_trials_collect_errors():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    batch = jtfne.trial_batch([_bad_condition()], n_reps=1)
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.5)

    # Module-level with collect_errors=True should capture the error
    result = jtfne.run_trials(model, batch, sim, collect_errors=True)
    assert result.results[0].success is False

    # Module-level with collect_errors=False should raise
    with pytest.raises(Exception):
        jtfne.run_trials(model, batch, sim, collect_errors=False)

def test_n_seed_policy_paired_by_replicate():
    paradigm = jtfne.standard_visual_omission()
    conditions = paradigm.conditions[:2]  # AAAB, AXAB
    n_reps = 2
    # paired_by_replicate: seed = base_seed + rep_idx
    batch = jtfne.trial_batch(conditions, n_reps=n_reps, seed=100, seed_policy="paired_by_replicate")
    
    # rep 0
    assert batch.trials[0].seed == 100  # cond 0
    assert batch.trials[1].seed == 100  # cond 1
    # rep 1
    assert batch.trials[2].seed == 101  # cond 0
    assert batch.trials[3].seed == 101  # cond 1

def test_o_seed_policy_unique_per_trial():
    paradigm = jtfne.standard_visual_omission()
    conditions = paradigm.conditions[:2]
    n_reps = 2
    # unique_per_trial (default): seed = base_seed + trial_idx
    batch = jtfne.trial_batch(conditions, n_reps=n_reps, seed=100, seed_policy="unique_per_trial")
    
    assert batch.trials[0].seed == 100
    assert batch.trials[1].seed == 101
    assert batch.trials[2].seed == 102
    assert batch.trials[3].seed == 103

def test_p_seed_policy_invalid():
    paradigm = jtfne.standard_visual_omission()
    with pytest.raises(ValueError, match="invalid_seed_policy"):
        jtfne.trial_batch(paradigm.conditions[:1], seed_policy="invalid_policy")

def test_q_manifest_trials_integration():
    cfg = jtfne.configuration().network(n=10).emitter().field().probe()
    model = jtfne.construct(cfg)
    
    # Without trials metadata
    manifest_clean = model.manifest()
    assert "trials" not in manifest_clean
    
    # With trials metadata
    trials_meta = {"batch_id": "test_batch", "n_trials": 4}
    manifest_trials = model.manifest(trials=trials_meta)
    assert "trials" in manifest_trials
    assert manifest_trials["trials"] == trials_meta
