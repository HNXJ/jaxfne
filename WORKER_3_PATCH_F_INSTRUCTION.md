[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][20260529-0300]

# Worker 3 Instruction — Patch F Objective/Null/Synchrony Correction

**Status**: Ready to begin  
**Base**: dev (commit 8926afd, A–C + D–E corrections merged)  
**Decision**: Proceed with Patch F corrections in jaxfne/objectives.py

---

## Mission

Implement spectrolaminar objective scoring with null distributions and synchrony gates in new module `jaxfne/objectives.py`.

**Critical requirement**: Use Phase 2 validator as final acceptance gate before submitting Worker 3 output.

---

## Locked Starting Conditions

✓ A–C infrastructure stable on dev (36 tests pass)
✓ Patch D source projector corrected (17 tests pass)
  - dynamics_derived path explicit
  - teaching/control source quarantined
  - dt_ms parameterized
✓ Patch E readout complete (29 tests pass)
  - score_computed = False enforced
  - metadata dict explicit
  - readout does NOT score

**No changes to A–E**. Only build Patch F in clean module.

---

## Implementation Scope

### Required (Patch F acceptance)

**File: jaxfne/objectives.py** (new module)

```python
def spectrolaminar_profile_score(
    readout: dict,
    target_alpha_beta: jax.Array,
    target_gamma: jax.Array,
    ...
) -> dict

def spectrolaminar_objective(
    readout: dict,
    target_alpha_beta: jax.Array,
    target_gamma: jax.Array,
    nulls: dict | None = None,
    synchrony_metric: str | None = None,
) -> dict

# Null distribution generators:
def layer_shuffle_null(readout: dict) -> dict
def band_label_shuffle_null(readout: dict) -> dict
def uniform_gain_null(readout: dict) -> dict
def no_field_projection_null(readout: dict) -> dict

# Optional (preferred):
def phase_randomized_null(readout: dict) -> dict
def source_polarity_flip_null(readout: dict) -> dict

# Synchrony metric:
def compute_synchrony_metric(
    spikes: jax.Array,
    bin_ms: float = 5.0,
    method: str = "mean_pairwise_correlation",
) -> float
```

### Score Grammar (CRITICAL)

Use **only one** of these per objective report:

```yaml
score_type: profile_score_no_null  
  # No null distribution exists
  # Export: profile_score_percent
  # Do NOT export: S_lam

score_type: null_normalized_similarity
  # Null distribution computed
  # Export: S_lam (null-normalized)
  # Also export: null_distribution_n, null_normalization_method

score_type: internal_motif_gate
  # Hard gate (pass/fail judgment)
  # Export: motif_gate_percent
  # Do NOT export: S_lam unless nulls exist
```

### Objective Report Schema

```yaml
objective_kind: spectrolaminar_profile
score_type: [one of above]
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

### Required Tests

Create `tests/test_spectrolaminar_objectives.py` (minimum 10 tests):

```python
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
```

---

## Boundary Enforcement

### MUST NOT do:

- [ ] Score in readout layer (readout lives in fields.py)
- [ ] Export S_lam without null distribution
- [ ] Allow teaching/control source in default evidence path
- [ ] Claim physical amplitude without calibration
- [ ] Use NaN/Inf in serialized reports
- [ ] Hardcode target profiles (use function parameters)

### MUST do:

- [x] Implement score_type grammar (only one type per report)
- [x] Implement 4+ null distributions
- [x] Implement synchrony metric + rejection gate
- [x] Metadata: truth_mode, physical_amplitude_claim_allowed, proxy-safe labels
- [x] Tests: nulls, synchrony, report schema, JSON safety
- [x] Export to `jaxfne/objectives.py` (new module)
- [x] Use Phase 2 validator as final gate

---

## Validation Checklist (Before Submitting)

```bash
# 1. Compile
python -m compileall -q jaxfne tests examples

# 2. Patch F tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_spectrolaminar_objectives.py -q --tb=line

# 3. A–E regression (ensure no new failures)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
  tests/test_spectrolaminar_sources.py \
  tests/test_spectrolaminar_readout.py \
  tests/test_multi_area_config.py \
  tests/test_synapse_connectivity.py \
  tests/test_multi_area_emitter_runtime.py \
  -q --tb=line

# 4. Run Phase 2 validator
python ~/.claude/skills/jaxfne-theta-tutorial-validator/validator.py --mode repo_smoke

# 5. Check Patch F in repo
git status --short
git diff --stat dev...HEAD
```

---

## Phase 2 Validator Gate (REQUIRED FINAL STEP)

After implementing Patch F, run:

```bash
python ~/.claude/skills/jaxfne-theta-tutorial-validator/validator.py --mode repo_smoke
```

**You MUST NOT submit Worker 3 output unless the validator returns PASS.**

If FAIL:
- Fix the blockers listed in the receipt
- Re-run validator
- Report the corrected receipt

---

## Worker 3 Final Report Template

```
[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]

Worker 3 Completion Report — Patch F Objective/Null/Synchrony Corrections

Mission: Implement spectrolaminar objective with null distributions and synchrony gates
Location: jaxfne/objectives.py (new module)
Status: ✓ COMPLETE (or BLOCKED)

Changes:
  - Patch F functions: spectrolaminar_profile_score, spectrolaminar_objective
  - Null generators: layer_shuffle, band_swap, uniform_gain, no_field (4 + 2 optional)
  - Synchrony metric: mean_pairwise_correlation
  - Tests: [N] test cases in test_spectrolaminar_objectives.py

Validation:
  - Compilation: PASS
  - Patch F tests: [N] passed
  - A–E regression: [N] passed (0 new failures)
  - Phase 2 validator: PASS ← REQUIRED GATE

Blocker check:
  - score_type grammar: [grammar used]
  - S_lam behavior: exported only with nulls
  - teaching/control rejection: confirmed
  - proxy-safe wording: confirmed

Repo state:
  - Branch: feat/spectrolaminar-objective-nulls-synchrony
  - Commits: [#]
  - Files changed: [list]
  - Working tree: [clean/modified]

Validator receipt:
  [JSON output from Phase 2 validator — PASS status required]

Truth status:
  truth_mode: truth_safe_unverified
  evidence_honest: true
  evidence_finite: true
  evidence_reproducible: true

Blockers: [list or "none"]

Next safe action:
  Merge Patch F to dev and prepare for full suite validation.

[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][yyyymmdd-hhmm]
```

---

## Acceptance Criteria (Must ALL be TRUE)

- [ ] Patch F implemented in jaxfne/objectives.py
- [ ] 4+ null distributions implemented
- [ ] Synchrony metric + rejection gates implemented
- [ ] Score grammar (profile_score_no_null | null_normalized_similarity | motif_gate) used
- [ ] S_lam exported only with null distribution
- [ ] Teaching/control source rejected from default
- [ ] 10+ tests implemented and passing
- [ ] A–C regression: 0 new failures
- [ ] Phase 2 validator: PASS status required
- [ ] Metadata: truth_mode, physical_amplitude_claim_allowed, proxy-safe labels
- [ ] JSON serialization safe (allow_nan=False)
- [ ] Final report includes validator receipt

---

## Branch Management

Create from dev:

```bash
git checkout dev
git pull
git checkout -b feat/spectrolaminar-objective-nulls-synchrony
```

Do NOT merge to dev until:
1. All tests pass
2. Phase 2 validator returns PASS
3. Worker 3 final report submitted with validator receipt

---

## Known Constraints

- A–C and D–E are LOCKED (read-only for Patch F work)
- No notebook integration yet (Suite No. 5 integration waits)
- release HOLD until Patch F accepted
- main HOLD until dev is stable

---

**You are Worker 3. Begin implementation when ready.**

**Requirement: Phase 2 validator gate before submission.**

[claude-haiku-4-5][/Users/hamednejat/workspace/main/jaxfne][20260529-0300]
