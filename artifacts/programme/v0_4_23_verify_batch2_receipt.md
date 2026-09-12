# VERIFY repair batch 2 — seeded engagement, compact resolves, storage flags

**Branch:** `dev` (on top of `98be315`).

## Production repairs
- `hdp_is_engaged(hp, params, enable_hdp)` — single routing predicate
  (bulk + continuation + preflight estimator): seeded HDP state
  (`with_hdp_initial_state`) engages the kernel even when params classify
  identity, since seeded non-uniform H drives real `dw ~ ΔH`. Seeded-but-
  disabled stays inert; unseeded identity unchanged. Fixes
  `test_h_override_seeds_hdp_initial_state`.
- `dynamic_state_from_model` `w0` resolved via `_resolved_edge_weight`
  (compact placeholder crashed continuation cold start; sphere20).
- `edge_list_with_delay_ms` delay-only (no full materialization; crashed
  on compact weight modes). Preserves all other compact storage.
- `to_dict` dtype key `receptor_index_arr` (matches `from_dict`;
  uint8→int32 drift gone).
- E2 `attach_provenance_class_delays`: `delay_storage="per_edge"`.
- `collect_column_viewer_data`: weight/tau/delay resolved (compact crash).
- Ruff F401 dead import removed (gate-blocking).

## Test-side repairs (same storage/identity classes, intent preserved)
- Delay-storage flag: c2 (6), delay_boundary (6), grammar causality (1).
- Resolve instead of raw reads: mcc topology, connectivity tau/weight,
  mcc_412 signs (stronger: checks executed values).
- Null-HDP diagnostics tests engage via `alpha` (weight ODE still null;
  stronger); homeostasis metadata test likewise.
- Router test anchor updated to human-renamed "## Project control".
- Checkpoint skip-set: weight compact metadata is JSON-sidecar by design.
- Closure floor test: explicit engaged runtime (also exposes that
  `cfg.hdp(enable_hdp=..., hdp_params={...})` nests params where kernels
  never read them — cfg-level kwargs must be top-level).

## Verification so far
- Dev gate green (137). Broad re-run: 32 → 2 failures, then closure
  repair → 1 remaining + environmental manifest case.
- Slow non-notebook: sdist + laminar-1000n PASS; etude scalar fails
  exactly per HDP-02 diagnosis (owned human pick).
