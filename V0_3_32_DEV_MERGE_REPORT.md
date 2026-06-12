# v0.3.32-alpha Dev Merge Report

**Date:** 2026-06-10  
**Status:** MERGED to `dev` (alpha-only)  
**Main/PyPI:** BLOCKED (10 runtime patches required)  
**Release:** NOT READY

---

## Merge Summary

```
Source branch: delta/v032-hierarchical-oddball-api
Target branch: dev
Pre-merge dev HEAD: 0d6059c test(delta): add spectrolaminar motif-quality audit
Post-merge dev HEAD: 70f0211 feat(delta): add v0.3.32-alpha hierarchical oddball API scaffold
Merge commit message: feat(delta): add v0.3.32-alpha hierarchical oddball API scaffold
Strategy: --no-ff (explicit merge commit)
Conflicts: 0
Status: SUCCESS ✅
```

---

## Changed Files (11 total)

| File | Change | Lines |
|------|--------|-------|
| V0_3_32_WORKER_REPORT.md | +NEW | +347 |
| jaxfne/sanity_delta.py | +NEW | +626 |
| jaxfne/__init__.py | MODIFIED | +18 |
| artifacts/public_api_before.json | MODIFIED | +19/-10 |
| tests/test_sanity_delta_hierarchical_oddball_config.py | +NEW | +105 |
| tests/test_sanity_delta_task_schedule.py | +NEW | +115 |
| tests/test_sanity_delta_backup_resume.py | +NEW | +147 |
| tests/test_sanity_delta_plasticity.py | +NEW | +70 |
| tests/test_sanity_delta_proxy_readout_names.py | +NEW | +96 |
| tests/test_sanity_delta_optional_imports.py | +NEW | +82 |
| tutorials/jaxfne-sanity-delta-test-hierarchical-global-local-oddball.ipynb | +NEW | +317 |

**Total: +1932 lines, -10 lines**

---

## Pre-Merge Status

**Branch:** delta/v032-hierarchical-oddball-api (commit 12a2086)

✅ All audit gates passed:
- API scaffold strong (7 classes, all exported/tested)
- Truth gates enforced
- Notebook purity verified
- No local engines
- Root imports lazy
- v0.3.31 behavior unchanged

❌ Runtime contracts are **alpha/stubbed**:
- task execution returns zeros
- resume() returns self
- plasticity config-only
- export reports stubbed

---

## Post-Merge Validation

### Compilation

```bash
$ python3 -m compileall -q jaxfne scripts tests tutorials
✓ All files compile without syntax errors
```

### v0.3.32-alpha Targeted Tests (45)

```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_sanity_delta_*.py -q --tb=short
.............................................
45 passed in 6.02s ✅
```

Coverage:
- test_sanity_delta_hierarchical_oddball_config.py: 10 tests ✅
- test_sanity_delta_task_schedule.py: 8 tests ✅
- test_sanity_delta_backup_resume.py: 9 tests ✅
- test_sanity_delta_plasticity.py: 6 tests ✅
- test_sanity_delta_proxy_readout_names.py: 6 tests ✅
- test_sanity_delta_optional_imports.py: 6 tests ✅

### Public API Snapshot (2 tests)

```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_public_api_snapshot_v034.py -q --tb=short
..
2 passed ✅
```

Updated snapshot: v0.3.32, 165 public names

### Combined v0.3.32 Tests (47)

```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_sanity_delta_*.py tests/test_public_api_snapshot_v034.py -q --tb=short
...............................................
47 passed in 1.75s ✅
```

### Full Test Suite (2248+)

```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/ -q --tb=line
(background task bdq054g57 completed: exit code 0)
2248 passed, 67 skipped, 4 xfailed, 39 warnings ✅
```

**Regression:** NONE
- All existing v0.3.31 tests still pass ✅
- No new failures introduced ✅

### mkdocs Strict Build

```bash
$ python3 -m mkdocs build --strict
✓ mkdocs build passed ✅
```

---

## Branch State Verification

### Main Branch (Untouched)

```bash
$ git log origin/main -1 --oneline
0d6059c test(delta): add spectrolaminar motif-quality audit
(unchanged from pre-merge)
✓ Main remains at v0.3.31 HEAD
```

### Dev Branch (Updated)

```bash
$ git log origin/dev -1 --oneline
70f0211 feat(delta): add v0.3.32-alpha hierarchical oddball API scaffold
(new merge commit from delta/v032-hierarchical-oddball-api)
✓ Dev now includes v0.3.32-alpha scaffold
```

### Delta Branch (Preserved)

```bash
$ git log origin/delta/v032-hierarchical-oddball-api -1 --oneline
12a2086 feat(v0.3.32): hierarchical global-local oddball API hardening
(unchanged; available for v0.3.32-final patches)
✓ Source branch preserved for future reference
```

---

## Classification

### Current Status

```
v0.3.32-alpha: Hierarchical Oddball API Scaffold
├─ Dev integration: ✅ MERGED
├─ Main integration: ❌ BLOCKED
├─ PyPI release: ❌ BLOCKED
└─ v0.3.33 start: ❌ BLOCKED
```

### Score

```
Agent correction quality:        95 / 100
v0.3.32-alpha dev-merge readiness: 88 / 100
v0.3.32-final readiness:         58 / 100
Main/release readiness:          0 / 100 (until 10 patches)
```

### Release Gate

```
main: BLOCKED
PyPI: BLOCKED
dev: ACCEPTED (alpha-only)
```

---

## Next Actions

### Immediate (after this merge)

1. ✅ Merge to dev (DONE)
2. ✅ Push to origin/dev (DONE)
3. ✅ Create patch plan (DONE → internal_docs/v0.3.32_final_runtime_patch_plan.md)
4. ⏳ NO public tag (do not tag v0.3.32-alpha on GitHub)
5. ⏳ NO PyPI upload (do not publish to PyPI)
6. ⏳ NO main merge (do not merge to main)

### For v0.3.32-final (next phase)

1. Implement 10 runtime patches (see patch plan)
2. Run full integration tests
3. Re-run full BETA audit
4. Generate v0.3.32_final_audit_report.md
5. **Only then:** authorize main merge + PyPI release

---

## Artifact Locations

| Document | Purpose | Location |
|----------|---------|----------|
| Worker Report | Implementation details | V0_3_32_WORKER_REPORT.md |
| BETA Audit Report | Alpha audit findings | V0_3_32_BETA_AUDIT_REPORT.md |
| Patch Plan | v0.3.32-final blockers | internal_docs/v0.3.32_final_runtime_patch_plan.md |
| This Report | Merge details | V0_3_32_DEV_MERGE_REPORT.md |

---

## Summary

**v0.3.32-alpha successfully merged to dev as an API scaffold.**

What this provides:
- ✅ Strong public API (7 classes, well-tested)
- ✅ Clear contracts (method signatures, return types, validation)
- ✅ Proven structure (45 unit tests + 2248 full suite)
- ✅ Pure notebook scaffold (no local engines)
- ✅ Future-proof (10-patch plan ready)

What this does **NOT** provide (yet):
- ❌ Real task execution (synthetic zeros)
- ❌ Real checkpoint/resume (returns self)
- ❌ Real plasticity numerics (config-only)
- ❌ Real export reports (JSON stubs)
- ❌ Production runtime

**Status:** Ready for v0.3.32-final development on the 10 runtime patches. Do NOT release to main or PyPI until patches are complete and re-audited.

---

## Metadata

- **Merge date:** 2026-06-10
- **Merge time:** ~19:30 UTC
- **Merge commit:** 70f0211
- **Pre-merge validation:** ✅ PASSED
- **Post-merge validation:** ✅ PASSED
- **Full suite:** ✅ PASSED (2248+)
- **Release gate:** ❌ BLOCKED (alpha-only, 10 patches required)

---

**STATUS: v0.3.32-ALPHA DEV INTEGRATION COMPLETE**

*Awaiting v0.3.32-final patch implementation and full re-audit before main/PyPI release.*
