# Receipt — 0.5.3 ENGINE item 3: continuation + P-010 (w1-53)

Parent: `a0b848d` (item 2) + `e348ebf` (harness note).

## P-010 diagnosis (no code changed until located)

Decomposition C_t = (X, H, W, B, K, A): X = v/u/prev_spikes/syn_state
(+V/spikes/sources), B = delay_state ring, K = membrane/rule RNG draws,
A = aux/b/theta_S. Plain `simulate()` vs chained continuation compared
step by step on the worktree (pre-fix, via `run_w1.py` runner; plain
`python script.py` imports site-packages, not the worktree).

| Exp | Config | First V diff | First spike diff | Verdict |
|---|---|---|---|---|
| A | baseline fixed-W, default noise | step 0 | step 9 | K diverges |
| A0 | baseline, noise_scale=0.0 | step 0 | step 9 | K (dropped noise_scale, see b) |
| C/C0 | legacy HDP, noise 0.2/0.0 | none | none | X/H/W/B/A carry correct |
| D/D0 | registered rule, noise 0.2/0.0 | step 0 | step 9 | K (+ dropped noise_scale, see c) |
| E | baseline + true delays in flight, default noise | step 0 | step 13 | K (B carry correct: delay_state (4,8) threads) |
| F | legacy HDP + delays, noise 0 | none | none | B carry correct, finals exact |

(Harness correction during diagnosis: hand-set `delay_steps` without
`delay_storage="per_edge"` is silently resolved to zero delay by
`resolve_edge_delay_steps`; the sanctioned form is
`replace(edges, delay_steps=ds, delay_storage="per_edge")`.)

LOCATED CAUSE (K component; X/H/W/B/A carries were correct throughout):
plain `_simulate_arrays` drew membrane noise as one bulk
`normal(split(master)[1], (T,N))`, while chained continuation draws
per-step `normal(split(key_t)[1])` from the carried `_advance_prng_key`
chain — different draw sequences, so V diverged at step 0 and spikes
downstream (the 173v174 class). B8's "identical seed per segment"
suspicion is ruled out as mechanism: continuation carries `prng_key`
and ignores segment Simulation seeds. Contributing drops in the same
family: (a) `_simulate_continuation_arrays` baseline branch dropped
`noise_scale` (hp={} when HDP off) while plain honored it; (b) the
registered-HDP plain path dropped `noise_scale` (always 0.5) and both
registered paths lacked it; (c) registered rule-noise stream used
`fold_in(split(master)[0], t)` plain vs `fold_in(split(key_t)[0], t)`
chained. Never a tolerance issue: state/continuation defect, fixed as
such.

## Fix (minimal, chain contract as canonical)

- `emitters.py`: baseline zero-delay + delayed kernels take optional
  `noise_schedule` (None = legacy bulk, direct callers unchanged);
  Model plain path passes `continuation_noise_schedule` (chain).
  Missing schedule forwarding in the zero-delay→delayed dispatcher
  added (found by Exp E still diverging after the first pass).
- `_hdp_registrable_kernel.py`: optional `noise_schedule` (membrane);
  with a schedule, per-step rule keys re-derive from the same chain
  (`split(step_keys[i])[0]`, `fold_in(..., t_global)`) and ride the
  scan xs; without, legacy bulk behavior bit-identical. `_apply_rule`
  takes the per-step `rkey`.
- `_model_simulate.py`: plain baseline passes chain schedule + honors
  declared `noise_scale` (previously silently ignored); plain
  registered passes both; continuation registered `hdp_kwargs` gains
  `noise_scale`.
- `_pipeline.py`: `compile_step_fn` forwards `noise_scale` to the
  registered call (read-only lookup; baseline/legacy keep `**`
  forwarding — a first-pass `.pop` broke both and was repaired).
- Untouched: legacy HDP kernel (already chain-consistent), dense /
  receptor_exponential / homeostatic (no continuation path),
  `hdp_rule.py`, validators, fields/vis/benchmarks. `simulate_batch`
  keeps bulk (seed-replicate utility, no continuation claim).

Post-fix, ALL diagnosis exps are bit-exact (None/None), incl. delays
in flight with live noise.

## Behavior-change disclosure (human-authorized 2026-09-24)

Decisions 1+2 below were STOP-gated for human authorization and both
approved as recommended; the original strict-gate readings remain recorded
(item-10 precedent: immutable FAIL + recorded correction, not relabeling).

Human decision 2026-09-24: point 1 APPROVED (changelog [Unreleased]);
point 2 ACCEPTED.

1. Model-level stochastic baseline/registered runs draw a NEW
   (chain-consistent) noise stream; deterministic runs (noise 0) and
   direct-kernel bulk defaults are bit-preserved. No frozen trajectory
   checksums exist (`baseline_050.json` is an env receipt; equivalence
   gate figures regenerate bit-identical — verified).
2. `test_equiv01_table.py::test_equiv_baseline_exact`: baseline
   jit-vs-eager V/sources move from bit-exact to d<=EPS_V (measured
   max 3.9e-5; spikes stay exact). Cause located: identical schedule
   values eager==jit, but the scan fuses differently when noise
   arrives as a traced argument (bulk-None still eager==jit). Stable
   jit-vs-eager exactness for stochastic runs is not a contract
   (C5–C7); the table's own HDP cells already use EPS_V.
3. Cross-kernel engaged-frozen (HDP kernel, K_HDP=0, live H) vs
   baseline kernel: spikes bit-exact, weights bit-untouched, H live,
   V/sources differ (observed max 1.8e-4/24 steps; seeded 1-2 ulp
   kernel-body arithmetic, amplified by v^2). Pre-existing structure,
   characterized (not gated) in
   `test_plasticity_off_engaged_frozen_equals_fixed_w` (< 1e-3 bound).
   Same-kernel plasticity-off identity (identity params -> baseline)
   stays strict bit-exact incl. live noise.

## Tests

- `tests/test_phaseC_H_carry_resume.py` extended (kernel level):
  4-chunk + 8-chunk all-mutable-state resume, delayed 3-chunk resume
  in flight with delay_state/step-offset threading — 5 passed.
- New `tests/test_continuation_all_state_053.py` (19 passed):
  P-010 baseline stochastic 4-chunk; chunk sweep 2/3/4/6/8/12;
  HDP stochastic/deterministic 8-chunk all-state incl. aux/b/theta;
  delayed baseline/HDP/registered in flight with live noise; local
  key-reading rule (`gen053_stochastic_gain`) plain-vs-chained with
  rule noise reaching W; engaged-frozen characterization;
  identity-routing live-noise bit-exact; direct-kernel bulk preserved
  + chain opt-in == chained single steps; key/step-index advance.
- Fallout fixes: item-1 identity test now matches declared noise on
  both sides (declaration honored post-fix); equiv01 cell above.
- Regressions: continuation_contract, ownership, recording_budgets,
  phaseC, HDP kernel/registrable/delayed/closure/checkpoint/equiv01/
  rec01/C2-delay/boundary (all green); equivalence gate 6 passed +
  1 skipped; fig06 + backend/api/surface/utils (84); mcc/grammar/
  audit/law01/H-boundary (67).

## Invariants held

- Chunked == continuous bit-exact for every mutable state incl.
  delays (no tolerance anywhere on this claim).
- Full recording default untouched; K_HDP=0 null unchanged
  (weights bit-untouched, H null-pinned per kernel null test).
- H != HDP separation preserved and tested.
