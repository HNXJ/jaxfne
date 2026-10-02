# Handout for opencode, 2026-09-29

Repo `E:\repos\jaxfne`, branch `dev`, upstream `origin/dev`, head `a97ba45`, tree has untracked handout, 0/0 vs origin/dev.
Read first: `artifacts/todo_stack.md`, then `artifacts/fact_stack.md` (read-only),
`artifacts/AGENTS.md`. Facts are human-owned; do not edit them.

## Done this session (all pushed)

| Area | Result |
|---|---|
| Fact stack | rebuilt as six two-row tables, 73 facts |
| `artifacts/` | roadmap, science, visualize_bundle moved to `archive/` |
| Diagram | `network_hspice(..., style="columns")` (`jaxfne/vis/network_glow.py`), 3 tests |
| Geometry | `build_laminar_column(radius_mm, height_mm)` consumed or refused; 3 tests |
| Notebook check | `tests/test_notebook_smoke_broad.py` (~40 s, 4 notebooks) |
| Études | plan, P0 table, P1 verification in `artifacts/etudes/literature_*.md` |
| Structural HDP | design note `artifacts/programme/structural_hdp_design.md` (not built) |
| Todo stack | reorganized; new items 0 (literature), 0b (structural HDP), 0c (primate mode) |

## Next, in order

| # | Item | Note |
|---|---|---|
| 1 | Label `p_feedforward=0.3`, `p_feedback=0.2` and weight ranges in `jaxfne/builders.py` as uncalibrated (docs/metadata only) | item 0c immediate task |
| 2 | Row 10 PDF (van Vreeswijk & Farkhooi, bioRxiv 10.1101/2025.09.04.674182); no code repo found | human supplies PDF |
| 3 | P2 GAP list for row 3 (Tahvili): jaxfne has no AdEx emitter | parameters already in `literature_p1_verification.md` |
| 4 | P3 preregistration | needs human sign-off (band definitions 50-150/10-30 Hz vs 75-150/10-19 Hz, detection rule) |
| 5 | NEXT 1-8 in the todo stack (P-012, P-019, P-016/P-020 bisect, P-013, D1b, D2, entrances check, D0b) | 0.5.5 seal work |

Item 0c step 1 (macaque evidence table per source/target layer and E/I) is the next
non-code deliverable after the label.

## Human decisions still open

D0 review list; re-freeze sign-offs P-016 and P-020; `arm_definitions` review; stale
v0.2.x notes; `units.py`/`pynwb_compat` wire-or-remove; artifacts reorg step 2;
agy install; band definition for the spectrolaminar observable.

## Gates and rules

- Push only on `dev`, upstream confirmed, exact paths staged (never `git add -A`).
- Code push: `python scripts/run_test_gate.py broad` (last: 4728 passed, 0 failed,
  74 skipped, ~5-6 min), `ruff` on `jaxfne/`. Doc-only push: `python
  scripts/audit_vocabulary.py --check`, `python scripts/audit_doc_code_integrity.py
  --check`, `tests/test_artifacts_root_layout.py`.
- Before pushing, count non-completed CI runs on `dev` (must be 0):
  `gh run list --branch dev --limit 30 --json status --jq '[.[]|select(.status!="completed")]|length'`.
- Hooks block heredoc source, `python -c`/`python -`, and `git add -A`; write scripts
  with a file tool and run by path. The Edit tool can flip LF to CRLF in mixed files;
  check with the eol guard.
- Citations must be primary-verified; WebFetch summaries and chat-derived numbers are
  unverified until checked against the paper.
- opencode write packets: set `PWD`, work in a clone, snapshot the source repo, rerun
  the worker's tests yourself; delegated output is evidence to verify.
- bioRxiv full text returns 429/503; Crossref, PMC, PLOS and `gh` work.
