# scripts/archive — retained, not executed

Receipt-only scripts moved out of the live tree by the T-33 tooling triage.
Each file below once produced a tracked receipt under `artifacts/`; the
receipts stay where they are and these files are kept only so the provenance
chain stays readable. No gate, workflow, test, or skill runs anything in this
directory. Do not import from here and do not revive a file into `scripts/`
without a packet that re-gates it.

Path caveat: two of these files resolve repository paths relative to their
own location (`parents[3]` idiom, `artifacts/` receipt paths). Run from this
directory those resolutions land elsewhere; that is acceptable because these
copies never run. Read them as text only.

| Archived file | Receipt it produced |
|---|---|
| `audit_methods_draft.py` | `artifacts/publication/results_reconstruction/methods_audit_receipt.json` |
| `benchmark_054_scaling.py` | `artifacts/assurance/issue_log/ISSUE_LOG.md`, `artifacts/programme/receipts/engine_053_054/composition_054.md` |
| `generate_readme_geometry_figure.py` | `artifacts/release/v0_4_17_release_receipt.json` |
| `test_harness_adversarial.py` | `artifacts/assurance/issue_log/ISSUE_LOG.md` |
| `fig08_e2_ping.py` | `artifacts/publication/final/e2_fig08_ping_receipt.json` |
| `fig09_e2_ssa.py` | `artifacts/publication/final/e2_fig09_ssa_receipt.json` |

Removed outright in the same triage (no tracked receipt, no live reference
in the searched scope): `scripts/emit_atlas_results_055.py`. Its only
in-scope hit was a `dashboard/models/hh_at01.md` reproduce line; nothing in
`scripts/`, `tests/`, `docs/`, or the workflows imports or executes it.

## Added 2026-10-20: `unreferenced/` (Hamm's D1 (a), release plan 0.5.7 section 7; P-042)

Scripts with no live reference (no gate, workflow, test, doc, skill or script hit), cited only by
the record listed. Moved with `git mv`; nothing deleted. Read as text only.

| Archived file | Record that cites it |
|---|---|
| `audit_w3_broad_handlers.py` | `artifacts/audit/reviews/sweep_2026-10-04_code.md` (its baseline `artifacts/audit/baselines/w3_broad_handler_tally.json` stays) |
| `p2v_a4_claim_language.py` | `artifacts/todo_stack.md` (T-17) |
| `sync_docs_version.py` | `artifacts/.lab/bundle.json` |

Kept in place, with the reason: `benchmark_055_matrix.py`, `gen_methods_055.py`,
`regenerate_hdp_population_restoring_metrics.py`, `run_protocol_h_h4_matrix.py` import jaxfne at
module level without a repo-root guard, which `tests/test_script_repo_root_guard.py` (P-025)
refuses at the new path; `publication_figures/polish/run_polish.py` is named by the frozen
`artifacts/publication/REVIEW_NAVIGATION.md`; the other ungated audit scripts are referenced from
`tests/` or `docs/`. Of P-042's 16 + 8, six were archived before this pass (table above).
