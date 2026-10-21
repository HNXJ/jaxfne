# 23-SIMP-02 receipt — NO_CHANGE

**Branch:** `dev` (on top of `1be1ff2`)
**Verdict:** do not consolidate. No code changed.

## Inventory (5 dispatch sites, `jaxfne/_model_simulate.py`)
- A. homeostatic `_simulate_arrays` (~276–309): guard name varies
  (`simulate_homeostatic_plastic` vs `simulate_homeostatic`); key carries
  `_plastic_active` + fingerprint; result unpacked downstream.
- B. HDP `_simulate_arrays` (~425–456): guard `simulate_hdp`; key carries
  ablation/backend/fingerprint; result unpacked downstream.
- C. edge-list recurrent (~494–530) and D. dense (~537–573): inline
  `target_fn(k, s)`, guard `simulate`, terminal `return` inside branch.
- E. batch vmap (~1232–1259): single-arg mapped fn, guard
  `simulate_batch`, plus a *second* non-JIT fallback split
  (vmap vs python-loop) selected by `effective_vmap`.

## Why NO_CHANGE
A shared `_dispatch_jit_cached` needs per call: `cache_key`,
`guard_name`, `guard_mode`, `B/Z/C/T`, `backend`, jit-fn factory,
run args, eager fallback — ~11 parameters. Each site's key, name, arity
(`(k, s)` vs `(keys,)`), terminal-return vs unpack, and E's dual
fallback are load-bearing, so the helper moves the complexity into its
signature instead of removing it (est. net ~0 LOC after the helper's own
~20 lines). Cache-key and guard-name identity is pinned by
`test_cache_key_isolated_across_hdp_params_on_reused_model` and the
compile-once test; a wrong merge fails silently via cache collision.
The blocks are stable and explicit; indirection across 5 hot paths buys
nothing. R01 already flagged this as "mechanical but touches cache keys /
guard names" — the stricter bar confirms: revert (never applied), close.

## Gates (unchanged tree, sanity)
- `test_hdp_dispatch` (10) PASS incl. cache-key isolation and
  compile-once; `test_compat_jom01_regressions` (4) PASS (rerun this
  increment against the unmodified dispatcher).
