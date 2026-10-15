# Receipt — NEXT 3 (P-020): bisect of the mcc3 condition-C timing drift

Question: which commit moved mcc3 condition C (θ̂, HDP off) spike timing
(κ 0.0228792 → 0.0182926, ISI CV 0.0126997 → 0.0122834; rates/counts kept)?

Method: worktree `bis020`, fixed checkpoint script per era (repaired
`98bbcce` version post-class-storage, guarded-import variant pre-storage;
era-own script first), run as `python scripts/...` with cwd = worktree
and `PYTHONPATH` = worktree (driver `_temp_bis020.py`; the search caught
and voided one full round that imported site-packages). Each run logs
`VERDICT` plus `identity_sha` (from the written metrics), kappa_C and
CV_C. GOOD: |κ − 0.0228792| ≤ 1e-3 with identity match. SIM_SEED=17,
OPT_SEED=42; Windows CPU float32.

Scope: 82 commits `c9e12f8..b0817dc` touching the HDP-off path
(emitters, stimulus, model/simulate, pipeline, config, runtime,
connectivity, construct, solvers, core). Verdicts (identity-proven,
exact values):

| Commit | Verdict | kappa_C | CV_C |
|---|---|---|---|
| `c9e12f8` | GOOD | 0.0228792 | 0.0126997 |
| `91291fe` (idx20) | GOOD | 0.0228792 | 0.0126997 |
| `0e347e7` (idx30) | GOOD | 0.0228792 | 0.0126997 |
| `4bc64c9` (idx37) | GOOD | 0.0228792 | 0.0126997 |
| `da5a911` (idx48) | GOOD | 0.0228792 | 0.0126997 |
| `477cb79` (idx52) | GOOD | 0.0228792 | 0.0126997 |
| `d22b05e` (idx53) | GOOD | 0.0228792 | 0.0126997 |
| `6b9595a` (idx55) | GOOD | 0.0228792 | 0.0126997 |
| `51f8c9b` (idx57) | GOOD | 0.0228792 | 0.0126997 |
| `e348ebf` | GOOD | 0.0228792 | 0.0126997 |
| `865e74b` | BAD | 0.0182926 | 0.0122834 |
| `b0817dc` | BAD | 0.0182926 | 0.0122834 | (log `bis020_good3.log` — misnamed, predates driver fix)

Mover `865e74b` is first-parent-adjacent to GOOD `e348ebf`
(ancestor exits 0; the `51f8c9b`→`865e74b` pair is cross-branch and was
discarded). BAD values identical at mover and range tip: nothing after
moves it further.

Instrument notes: post-sim line-628 signs crash (`edge_list.weight`
empty under class storage — `98bbcce` message, commit `1430219`)
bypassed reporting-only (try/except → None) from r8a-era (idx55+) up
(idx48 ran direct); sims had completed in every bypassed run. Tune-path
IndexError SKIP band idx38–44 (script/model era mismatch,
no bypass permitted): flanked by exact GOODs at idx37/idx48, so a
mover+revert inside is formally possible but judged unlikely (would need
a 7dp-identical revert of κ and CV).

LOCATED CAUSE (hypothesis, corroborated cross-lane): same `865e74b`
noise-schedule change as P-016; the HDP-off route mechanism is
a hypothesis, not verified here.

Gaps: single seed pair; Windows CPU float32 only; SKIP band formally
unattributed (see above); bypass runs share log files with their SKIP
predecessors.

Sign-off (unsigned): re-freeze condition C (κ 0.0182926, CV 0.0122834)
with cause stated above. Human signs here.
