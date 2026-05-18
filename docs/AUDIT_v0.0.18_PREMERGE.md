# jaxfne v0.0.18 Pre-merge Audit

**Date:** 2026-05-18  
**Branch:** dev-v0.0.18-objective-report  
**HEAD before hardening:** 07d2119  
**truth_mode:** truth_safe_unverified

---

## Repo state at BETA

- Working tree: clean (no uncommitted changes)
- origin/dev-v0.0.18-objective-report: in sync
- origin/main: 5 commits behind HEAD
- Tests: 178 passed, 0 failed
- Compileall: clean (jaxfne, tests, examples)
- Examples sampled (4/7): all pass
- `jaxfne.__version__`: 0.0.18
- pyproject version: 0.0.18

## Commits ahead of main

| SHA | Message |
|---|---|
| 07d2119 | fix(config): add truth_mode to _CONSERVATIVE_TRUTH_DEFAULTS and validate_config |
| 54ba7ba | feat(report): add ObjectiveReport structured evaluation result |
| 407f698 | feat(readout): add ReadoutSpec/ReadoutResult declarative feature extraction |
| 1f49424 | feat(receipt): add RunReceipt deterministic run-capture object |
| fa282ef | feat(config): add jcfg config standard foundation |

## BETA audit findings (resolved in this hardening pass)

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| F7 | **Blocking** (resolved) | `truth_mode` absent from `_CONSERVATIVE_TRUTH_DEFAULTS`; validate_config never checked it | Fixed at 07d2119; 3 new tests (P/Q/R) |
| README | Non-blocking (resolved) | Status header stale at v0.0.5; test count 55; package structure missing new types; version section ordering reversed | Patched in this hardening pass |
| venv_push | Non-blocking (resolved) | 475MB `venv_push/` not covered by .gitignore | Added to .gitignore |
| pip metadata | Non-blocking (resolved) | Installed editable wheel reported v0.0.3 | Resolved by `pip install -e .` |

## Remaining non-blocking items (post-merge cleanup)

- `config_truth_boundary()` docstring: should state it is a passthrough, not an enforcer — add "call validate_config first"
- `receipt_id` docstring: document that upgrading `_JAXFNE_VERSION` changes receipt IDs for the same config/seed
- `config_hash` docstring: document that unknown keys (from `load_config`) enter the hash
- Optional-section absent warnings (geometry/paradigm/trials): consider a flag to suppress on minimal valid configs
- Remote branch sprawl: `dev-v0.0.{5,8,9,10,11-*,12,13,14,15,16,17}` are candidates for archival once main is updated

## Merge blockers

None. All blocking defects resolved.

## Scientific / truth status

```
truth_mode:                    truth_safe_unverified
claim_level:                   computational_scaffold
source_calibration_status:     uncalibrated_izhikevich_native_current
field_solver_status:           laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
empirical_validation_status:   not_empirically_validated
mechanism_claim_status:        not_claimed
```

No biological/scientific claim is made. Receipts and reports are computational
validation artifacts, not empirical evidence.

## Next safe action

Merge `dev-v0.0.18-objective-report` → `main` via PR or `--ff-only`.
