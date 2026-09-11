# 23-SIMP-05 receipt — NO_CHANGE

**Branch:** `dev` (on top of `4fc0d03`)
**Verdict:** do not unify. No code changed.

## Kernel map (scope: HDP family + baseline edge)
- Baseline edge: `simulate_edge_recurrent_izhikevich` (~1390–1604; 4 step
  closures: sched × record-bundle) + `_simulate_edge_recurrent_izhikevich_delayed`
  (~1122–1389; 4 step closures).
- Legacy HDP: `simulate_edge_recurrent_izhikevich_hdp` (3361–4399 = 1039
  lines; ~250 docstring; setup ~370; ONE `step` 3988–4282 = 295 lines
  with 3 internal branches; scan+diagnostics 4283–4399 = 117).
- Registrable HDP: `_hdp_registrable_kernel.py` (352 lines; zero-delay
  `step` ~55 + delayed `step_delayed` ~70 sharing `_apply_rule`).

## SHARED (~40 lines per instance, boilerplate)
Izhikevich integrate/spike/reset (~10), edge current/syn (~4), syn decay
update (~2), ring slot/set (~3), source proxy (~3), noise/sched/bulk
prologue (~15), dtype prologue (~10). Already partially factored
(`_izhikevich_dv_du`, `_segment_sum`, `_source_proxy_from_components`,
`_delayed_presynaptic_spikes`, `_apply_rule` within registrable).

## UNIQUE (semantic, dominant)
H state: node income/spending/tau/barrier (~40) vs boundary
r_bar/S_L/S_H/B' (~45) vs population controller (~15) vs registered
`step_fn` dispatch (~20). Theta: sign-split multiplicative rules (~25)
vs `upd.d_theta` (~10). Ordering differs load-bearingly: legacy
syn→H→w→integrate→spikes→drain vs registrable
syn→integrate→spikes→rule. Carry/diagnostics differ (theta_S/r_bar vs
aux; 4 record-flag arities).

## Quantification
- Duplicated equations: none — every H/Theta update equation is
  family-specific; only integration/current/ring plumbing repeats (~40).
- Merge design needs `hdp_mode` ∈ {node, boundary, population,
  registered} × delay × record arities plus per-mode params (~15 new
  args): a parameter-heavy universal kernel by definition, or N static
  variants (compile explosion per SIMP-03). Extracting the ~40 shared
  lines touches tuned numerics across 3 HDP families for ~50 lines saved
  with full preset re-verification owed.
- Step 7 applies: common code too small, semantic branches dominate →
  NO_CHANGE. Step 5 equivalence matrix therefore not run for a merge
  (no merge exists); all 8 listed behaviors remain covered by passing
  gates on the unmodified tree (HDP-01, delayed-registrable ×7,
  finite-delay, continuation, dispatch, audit01).
- No STDP/STP/Jomission content used.
