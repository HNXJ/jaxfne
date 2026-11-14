# PCL R5 remote-only recovery inventory (2026-11-12)

Baseline: `jaxfne dev@342279db791a005e556cb52c6894970b0ed9c940`. Local `F:/warehouse/pcl/r5/` and detached workers are treated as **lost/unavailable**, not assumed completed. This inventory is not a new scored result.

## Observed tracked evidence

- `artifacts/etudes/papers/pcl/prereg/P3_fig3_r5_exact_result.md` exists; `scores/fig3_r5_exact.json` exists. The signed R4b criterion and current state record R5-exact **FAIL: 4 of 16 files exact**. Preserve these frozen outcomes.
- `artifacts/etudes/papers/pcl/prereg/` contains R4/R4b/R5-exact and older v4/v5 score records; **no R5-approximate result Markdown** appears in the directory listing.
- `artifacts/etudes/papers/pcl/prereg/scores/` contains `fig3_r5_exact.json` and v4/v5 JSON; **no R5-approximate score JSON** appears in the directory listing.
- `artifacts/todo_stack.md` and `artifacts/state.md` last record 2860/3168 tasks done at 2026-11-07 22:18Z, an estimate of finish time, and a shell logging bug that can make `rc` appear 0. These are stale progress reports, not proof of completion.
- P-055: first divergence occurs in the complex layer on each diverging file; suspected floating-point update ordering or equal-time tie ordering are hypotheses, not attributed causes.

## Next bounded actions

1. **Reconstruction gate**: from a fresh tracked checkout, resolve the signed R4b input manifest, driver, oracle, dataset license and external data availability. Count all 3168 task keys and check for committed raw outputs. A missing local dataset or raw output is a blocker for *re-scoring*, not a reason to fabricate it.
2. **Minimal differential diagnostic**: select one existing divergent test file only if the source data are legitimately available; replay core and oracle from a common event state to the first differing complex update. Log time, event order, weights, threshold margin, precision, and RNG. Distinguish tie ordering from rounding by an explicit perturbation. No changes to the signed criterion.
3. **Re-run strategy**: if raw data are unavailable remotely, record UNKNOWN and prepare a deterministic synthetic fixture for debugging **without** calling it R5 scoring. If data are recovered, smoke one task then run the signed protocol with checkpointing and correct shell return-code capture; verify terminal receipts, not progress estimates.
4. **Close** only when R5-approximate score JSON/result Markdown and complete source manifests are pinned, and P-055 has a demonstrated cause or remains explicitly unresolved with a minimal reproducer. Update the dashboard from actual scored outputs only.

Scope: evidence inventory and execution packet; no code changed, no run performed, no result fabricated.
