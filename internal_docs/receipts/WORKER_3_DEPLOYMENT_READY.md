[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][20260529-0420]

# WORKER 3 DEPLOYMENT — PATCH F OBJECTIVE/NULL/SYNCHRONY

**Status**: READY FOR IMMEDIATE DEPLOYMENT

**Gate Status**: 
- ✓ Fresh-worker behavior check: PASS
- ✓ A–C + D–E infrastructure: Stable (1509 tests pass, 0 regressions)
- ✓ Phase 2 validator: Implemented and tested
- ✓ All instruction documents: Complete

---

## WORKER 3 EXACT DISPATCH PROMPT

Use this prompt verbatim to deploy Worker 3:

```
Role: Worker 3 — Patch F Objective/Null/Synchrony Correction

Mission:
Implement spectrolaminar objective scoring with null distributions and synchrony gates 
in new module jaxfne/objectives.py.

Base Repository State:
- Branch: dev
- Latest commit: 8926afd
- Status: Stable (A–C + D–E merged, 1509 tests pass, 0 regressions)
- Infrastructure: Multi-area config, connectivity, emitter, source (dynamics-derived + teaching/control quarantined), readout (no scoring)

Critical Operating Rules (Inherited from fresh-worker check):
1. Use the available jaxfne skills (don't invent code locally)
2. Do not claim success without exact command receipts
3. Separate verified facts from pending checks
4. Every statement needs a receipt: exact command + exact output
5. Follow header/footer protocol: [model][path][yyyymmdd-hhmm]
6. Provide one next safe action only

New Branch:
git checkout dev && git pull
git checkout -b feat/spectrolaminar-objective-nulls-synchrony

Required Implementation:

File: jaxfne/objectives.py (new module)

Functions:
- spectrolaminar_profile_score(readout, target_alpha_beta, target_gamma, ...) → dict
- spectrolaminar_objective(readout, target_alpha_beta, target_gamma, nulls=None, synchrony_metric=None) → dict

Null distributions (4 required, 2 optional):
- layer_shuffle_null(readout) → dict
- band_label_shuffle_null(readout) → dict
- uniform_gain_null(readout) → dict
- no_field_projection_null(readout) → dict
- [optional] phase_randomized_null(readout) → dict
- [optional] source_polarity_flip_null(readout) → dict

Synchrony metric:
- compute_synchrony_metric(spikes, bin_ms=5.0, method="mean_pairwise_correlation") → float

Score Grammar (CRITICAL - use exactly ONE type per report):
```yaml
score_type: profile_score_no_null
  # When no null distribution exists
  # Export: profile_score_percent
  # Do NOT export: S_lam

score_type: null_normalized_similarity
  # When null distribution computed
  # Export: S_lam (null-normalized)
  # Also export: null_distribution_n, null_normalization_method

score_type: internal_motif_gate
  # Hard pass/fail gate
  # Export: motif_gate_percent
  # Do NOT export: S_lam unless nulls exist
```

Objective Report Schema (must match exactly):
```yaml
objective_kind: spectrolaminar_profile
score_type: [profile_score_no_null | null_normalized_similarity | internal_motif_gate]
profile_score_percent: float_or_null
motif_gate_percent: float_or_null
S_lam: float_or_null
nulls_run: bool
null_distribution_n: int
null_normalization_method: z_score | percentile | declared_other
synchrony_checked: bool
synchrony_metric: mean_pairwise_correlation | variance | null
synchrony_value: float_or_null
synchrony_threshold: float_or_null
synchrony_rejection: bool
rejection_reasons: [list of strings]
uses_teaching_control_source: bool
default_evidence_path: bool
physical_amplitude_claim_allowed: false
truth_mode: truth_safe_unverified
bands:
  alpha_beta: [8.0, 25.0]
  gamma: [40.0, 150.0]
```

Tests Required:
Create tests/test_spectrolaminar_objectives.py with minimum 10 tests:
1. test_profile_score_returns_bounded_percent()
2. test_profile_score_exports_no_s_lam_without_nulls()
3. test_null_layer_shuffle_changes_score()
4. test_null_band_shuffle_changes_score()
5. test_null_uniform_gain_reduces_profile_info()
6. test_null_no_field_projection_handled()
7. test_null_distribution_z_score_normalized()
8. test_teaching_control_source_rejected_from_default()
9. test_synchrony_metric_computes_finite()
10. test_high_synchrony_adds_rejection_reason()
11. test_objective_report_json_serializable()
12. test_objective_metadata_says_physical_amplitude_false()
13. test_s_lam_absent_without_nulls()
14. test_s_lam_present_with_null_distribution()
15. test_score_type_grammar_correct()

Boundaries (MUST NOT violate):
- NO scoring in readout layer (readout stays in fields.py, does NOT score)
- NO export S_lam without null distribution
- NO teaching/control source in default evidence path
- NO physical amplitude claims without calibration
- NO NaN/Inf in serialized reports
- NO hardcoded target profiles (use function parameters)

Validation Checklist (before submission):
```bash
# 1. Compile
python -m compileall -q jaxfne tests examples

# 2. Patch F tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_spectrolaminar_objectives.py -q --tb=line

# 3. A–C + D–E regression (ensure 0 new failures)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
  tests/test_spectrolaminar_sources.py \
  tests/test_spectrolaminar_readout.py \
  tests/test_multi_area_config.py \
  tests/test_synapse_connectivity.py \
  tests/test_multi_area_emitter_runtime.py \
  -q --tb=line

# 4. Run Phase 2 validator (REQUIRED FINAL GATE)
python ~/.claude/skills/jaxfne-theta-tutorial-validator/validator.py --mode repo_smoke
```

Acceptance Criteria (ALL must be TRUE):
☐ Patch F implemented in jaxfne/objectives.py
☐ 4+ null distributions implemented
☐ Synchrony metric + rejection gates implemented
☐ Score grammar (profile_score_no_null | null_normalized_similarity | motif_gate) used
☐ S_lam exported only with null distribution
☐ Teaching/control source rejected from default
☐ 10+ tests implemented and passing
☐ A–C + D–E regression: 0 new failures
☐ Phase 2 validator: PASS status required
☐ Metadata: truth_mode, physical_amplitude_claim_allowed, proxy-safe labels
☐ JSON serialization safe (allow_nan=False)
☐ Final report includes Phase 2 validator PASS receipt

Final Report Format:
[verified_model_or_unknown][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]

1. Repository state (branch, commit, working tree)
2. Skills invoked / checks used
3. Commands run and exact results (all 4 validation commands)
4. Verified facts
5. Pending or skipped checks
6. Blockers (if any)
7. One next safe action

[verified_model_or_unknown][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]

CRITICAL GATE:
Phase 2 validator must return PASS. No exceptions. If validator returns FAIL, report the blockers and do not submit until PASS is achieved.

Key References:
- WORKER_3_PATCH_F_INSTRUCTION.md (detailed scope)
- FRESH_WORKER_BEHAVIOR_CHECK_REPORT.md (discipline baseline)
- ~/.claude/skills/jaxfne-theta-tutorial-validator/ (validator implementation)
- FRESH_WORKER_BEHAVIOR_CHECK.md (fresh-worker gate spec)
```

---

## Pre-Deployment Checklist

✓ Fresh-worker behavior check completed and PASS  
✓ A–C + D–E infrastructure verified stable  
✓ Phase 2 validator ready (repo_smoke mode)  
✓ Worker 3 instructions documented  
✓ Score grammar rules documented  
✓ Null distributions specified  
✓ Synchrony gate specifications provided  
✓ Test requirements documented  
✓ Validation checklist provided  
✓ Acceptance criteria listed  
✓ Final report format specified  

---

## Deployment Status

```
WORKER 3 READY: YES
GATE: Phase 2 validator PASS required before merge
BRANCH: feat/spectrolaminar-objective-nulls-synchrony from dev
BASE: commit 8926afd (stable)
DISCIPLINE: Read-first CLAUDE.md rules apply
```

---

[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][20260529-0420]
