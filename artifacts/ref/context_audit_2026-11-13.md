# Context audit and near-term plan — 2026-11-13

**Evidence scope:** live GitHub remote `dev@24a95963f13a771b0d4727aadf163e3453e09dde`; no local checkout, full suite, simulation, or host validation. This is a navigation index, not a second authoritative task stack or a change to scientific acceptance.

## Baseline and authority

- Read Jarness `rules/start.md`, `AGENTS.md`, `artifacts/AGENTS.md`, `artifacts/state.md`, `artifacts/open_stack.md`, `artifacts/goal_stack.md`, `artifacts/todo_stack.md`, and `artifacts/assurance/issue_log/ISSUE_LOG.md`. Owner facts in `artifacts/fact_stack.md` are unchanged.
- GitHub: `main@e0160dfc` (published 0.5.9 baseline), `dev@24a95963` at inspection; these are intentionally different during 0.5.10. The latest inspected dev Actions list contains both a success and a failure at this SHA; identify workflow/job before calling dev green or red overall.
- `state.md` is stamped `d2507956`, behind the observed dev ref. Its 40/100 0.5.10 checklist score and broad-gate receipt are **historical**, not a current rerun. Restamp only after a checked state/gate result.
- Several temporary and audit branches, five open PRs (#124–#128), and a publication `gh-pages` branch exist. Preserve concurrent claims; do not delete or force reconcile.
- `O-R5SIZE` is an owner decision: immutable 3168-row prereg evidence conflicts with tree-size budget. Preserve both gates; do not trim evidence or relax budget without a ruling.

## Short-term: existing work, dependency order

1. **T-93 / O-R5SIZE:** obtain owner's route decision, then verify both `check_tree_size.py --check` and `check_record_integrity.py --check` on the same SHA. No unilateral acceptance change.
2. **CI/state reconciliation:** inspect the failing dev workflow/job and the successful run at `24a95963`; compare to `state.md` stamp, distinguish stale narrative from real defect. Check exact-SHA gate receipts.
3. **Independent scientific audit:** preserve PCL R5 exact and approximate FAIL results and the no-local inhibition attribution; address missing membrane-potential replay evidence without replacing the frozen preregistered result. P-053 sparse-weight gap and P-056 deep-layer kernel LFP remain open until their original falsifiers run.
4. **B1 benchmark:** review PRs #126/#128, reconcile signed M3 grid and matched HH baselines before T-87 performance comparison. Do not promote a speed or accuracy claim from a smoke run.
5. **0.5.10 remaining packages:** T-86 CAMINOS panels, T-F5 batching, T-B5 GPU parity, T-C1 ensemble, and T-D3 prereg-dependent Mejias. Preserve frozen task criteria and owner gates.
6. **Release T-92:** broad tests, CI, replication audit, dashboard, version/changelog and independent review on one candidate SHA; tag, publication and main push are reserved for the owner.

## Midterm

- **0.5.10:** close scored replication, benchmark, provenance and release-gate coverage without hiding null/FAIL outcomes. Score only from the actual checklist and exact receipts.
- **0.5.11:** execute already selected Lee 2025 / Mazzoni 2015 preregistration lanes and deferred scientific tasks after signed criteria; no retroactive tuning.
- **Paper G-0:** advance the arXiv paper's 8 main/16 supplementary figures and C1–C12 acceptance, separating generated figures from validated empirical claims.
- **Cross-repository:** jnwb G-3 needs a known-lag/spectrum field; Jomission uses a frozen JaxFNE engine, and OCMA audits JaxFNE literature claims. Do not infer validation of one from another.

## Documentation alignment

The public README is the scientific/product entrypoint. Keep internal status, work packets and planning under `artifacts/`. Do not change owner goals, facts, scientific definitions, pinned evidence, release metadata or gate code in this pass.

## Closing check

Each future closure needs its original todo/issue acceptance, non-vacuous evidence, exact SHA, independent review and CI where applicable. UNKNOWN is not PASS. This audit changes no implementation and asserts no fresh test success.
