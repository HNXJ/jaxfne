# VERIFY repair batch 1 — broad-gate failures (all pre-existing at 88fa347)

**Branch:** `dev`. Attribution proven per-file in a clean `88fa347`
worktree unless noted. No failure traces to the v0.4.23 programme delta.

## Production repairs
- Seeded-HDP engagement (`hdp_is_engaged`, shared by bulk +
  continuation + preflight estimator): `with_hdp_initial_state` was
  silently dropped whenever params classified identity — yet seeded
  non-uniform H drives real `dw ~ ΔH`. Fixes
  `test_h_override_seeds_hdp_initial_state`. Seeded-but-disabled stays
  inert; unseeded identity routing unchanged (HDP-01 invariant intact).
- `dynamic_state_from_model` `w0` resolved via `_resolved_edge_weight`
  (compact placeholder crashed continuation cold start; sphere20).
- `edge_list_with_delay_ms` touches delay fields only (full
  materialization crashed on compact weight modes needing a sign).
- `EdgeList.to_dict` dtype key `receptor_index` → `receptor_index_arr`
  (matches `from_dict`; fixes uint8→int32 drift).
- E2 `attach_provenance_class_delays` sets `delay_storage="per_edge"`
  (all 6 E2/E3 occupancy/gate failures).
- Ruff F401 unused import (`_model_simulate.py`) blocking the gate.

## Test-side repairs (intent preserved, now checks executed values)
- Delay-storage flag added: c2 (6), `delay_boundary` (6),
  `delay_preserves_forward_causality` (same HP-05 class — delays now
  genuinely engaged).
- Resolve instead of raw reads: `test_mcc` topology signs,
  connectivity tau/weight sets, mcc_412 sign preservation
  (now compares executed weights — stronger).
- Null-HDP diagnostics tests engage real HDP via `alpha` (weight ODE
  still null; stronger than asserting on the baseline route): grammar
  `hdp_off_matches` + `null_recovery`, homeostasis metadata shape test.
- `test_context_is_canonical_router`: human commit 751d0fe renamed
  "## TODO stack" → "## Project control"; test updated to the
  canonical anchor (doc authority wins).
- Checkpoint skip-set: weight compact metadata lives in the JSON
  sidecar by design (exact float roundtrip, like the tau table).

## Left open (not repaired here)
- `test_release_manifest_is_not_committed`: fails on the gitignored
  local v0.4.22 `release-dist-*` output present since Sep 10 (rglob
  covers ignored dirs). Environmental; release evidence not deleted.
  Needs a human call (clean workspace vs gate scope).
- SlowManual: etude scalar (23-HDP-02, human protocol pick) +
  disconnected-null semantic (human decision).
