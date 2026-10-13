## 0.5.5 stack

Legend: [A] an agent can run it now · [H] needs a human decision · [B] blocked.

- [A] Next release owned by the cursor agent (hub main #335); quiet-matrix timing run waits for an idle machine.
- [A] Seal notes for the receipt: H1 (e) 9958a72 (0 → 173 inter-area edges at defaults); P-017 runs whose drive changed (f8960a0); `build_laminar_column` geometry keys now written; AT-02…AT-06 records carry the pre-correction spec digest (regenerate; `tests/test_agent_bench_055.py`).
- [A] ATLAS 8 `artifacts/perf/matrix_055.json`: quiet-machine rerun before the seal; add `T_test` and per-cell drive provenance; null-drive `output_checksum` equals `matrix_051.json` (only `mech_hdp` may differ). Items 9–14 take deltas from it.
- [A] AT-10-R6 RSS growth (+80–98 MB over 4 chunks): resolve leak vs XLA cache before any run over 100 s.
- [A] `_meta.jaxfne_version` reads 0.5.0 in `artifacts/atlas/results/at01_055.json`, `at05_055.json`, `at07_055.json`, `at_reduction_055.json`. Blocked by: the version bump.
- Coverage (checked 2026-10-12): 48 rows, 42 VALIDATED, 6 OUT_OF_SCOPE. Seal not met; no tag. PR #101 (dev → main, Hamm 2026-10-07) not re-checked.

## MANUSCRIPT (ends 0.5.5), open

- [A] Item 9 coverage matrix 100% mapped or marked out of scope. Blocked by: human scope marks.
- [A] Item 10 figures via `scripts/publication_figures/` from generated Atlas data. Blocked by: item 9.
- [H] Item 11 claim ledger `artifacts/programme/claim_ledger_055.md` (48 rows): Hamm confirms each `kind` and fills 4 PLANNED rows.
- [A] Item 12 methods quoted from manifests. Blocked by: item 10.
- [A] Item 13 citations primary-verified; NMDA, GABA_B and AMPA owed.
- [H] Item 14 text consistent with the ledger; submission is Hamm's. Blocked by: items 9–13.

## ACCEPTANCE (0.5.5 seal = end of programme), open

- [H] Canonical configurations bit-identical to the frozen baseline except authorized repairs (P-014), listed in the seal receipt; `python scripts/run_test_gate.py release`, then `rc`. Tag, main merge and release are Hamm's.
- [A] AT-01…AT-10 and AT-10-N20 regenerate from manifests; coverage 100%; deltas vs v0.4.25 measured (T_simulation, M_peak, T_test, T_agent_task, E_semantic = 0); every claim traces to a PASS ledger row, negatives reported.

