# 24-PRO-01 receipt — independent professionalism audit + findings worked

**Branch:** `dev`. Also satisfies the external half of AUDIT-01 per
assessment authorization (scope covered technical correctness +
professional/minimization quality).

## Provenance qualification
Critics ran as two separate read-only subagent sessions (no write access),
both reporting as `Muse Spark` @ `62b6fb4`. Independent sessions, same
model family — recorded honestly so a human can judge whether a
genuinely external party is still required for AUDIT-01 closure.

## Triage (11 accepted/worked, 5 TRADEOFF surfaced, 9 REJECTED with evidence)
Accepted DEFECT (fixed):
- H7 silent-drop: `homeostatic_ei` discarded paradigm/poisson/ablation →
  explicit `ValueError` (`_model_simulate.py`) + adversarial test
  (`test_phaseD_source_schema.py`). No tested caller affected.
- H7 metadata-only `Simulation.plasticity` documented as provenance-only
  (`_signals.py`) + bit-exact pinning test (`test_api_smoke.py`).
- Graphics-overhead test never imported top-level `jaxfne` → extended
  (`test_v0321_migration_boundaries.py`); guard verified holding.
- Public docs: doctrine governance/SHA block stripped (`rbs_rbd_hdp.md`),
  H7 tag + verification date + `weight state` + duplicated RBS def fixed
  (`guides/hdp.md`), `truth gates` → `status fields` (`scope_and_status.md`).
Accepted STRICTLY_SUPERIOR_CHANGE (all behavior/numerics-identical):
- 8 drive-schedule dual scan paths → single zero-filled path
  (`emitters.py`, net ≈ −200 lines incl. `dataclass_replace` below).
- 23-line manual `dataclass_replace` → stdlib `dataclasses.replace`
  (single caller, valid fields only).
Rejected with evidence: B1 README agent line (owned by
  `test_agent_context_hygiene.py`), B2 frozen 0.4.13 contract page
  (intentional; current contract at `docs/api/index.md`, 177+13 verified),
  test-path references in guides (11-file repo convention), A3 cross-kernel
  RNG streams (no parity claim exists; per-use-site splitting is correct),
  A5 `recovery_h_k` wrapper (named D1 entry point, 8 sites + test import),
  emitters-vs-homeostatic "duplicates" (distinct dynamical systems),
  continuation double-advance, HDP cache-key name, dead `prev_spikes`
  carry, null-HDP test asymmetry, mkdocs nav, RuntimeConfig duality,
  compat aliases, dead-code hunt (all referenced).
TRADEOFF surfaced for human decision (no action taken):
- Global `jnp.clip` monkey-patch in `bridges.py` (optional Jaxley compat).
- Host sync per `run_continuation` segment (`_pipeline.py:726`).
- Trace-time `warnings.warn` in jitted homeostatic_ei path.
- Null-HDP test pair (speed vs strength; both cheap — keep).
- `build_laminar_column` vs `build_tutorial_laminar_column` naming;
  EXPERIMENTAL_INTERNAL surface shipping in wheel (PKG follow-up).

## Incident + recovery
First broad run caught a real fall-through bug in my own unification
(plastic branch reached the unconditional tail scan, 5 failures). Fixed by
restoring the if/else exclusion; reran full broad gate.

## Gates
Broad: **3855 passed** (3850 + 5 recovered), 75 skipped, 37 deselected,
4 xfailed; vocab check pass; no orphan docs; ruff `jaxfne/` clean
(test-file F401/F811 pre-existing at HEAD, gate covers `jaxfne/` only).
Reconciles exactly with the pre-fix run (3850 + 5).
