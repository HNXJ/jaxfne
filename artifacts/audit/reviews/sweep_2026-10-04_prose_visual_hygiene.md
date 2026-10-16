# Deep review sweep 2026-10-04 — prose/visual/hygiene lane (quick)

Task: read-only review of docs/, README, committed figures, artifacts/ stacks against match-writing-style (audit), match-visual-style (static only), match-workspace-hygiene.
Worker: general subagent (quick). No writes made by worker.
Branch: dev @ d463e574. Deduplicated against D0-cleared items (spot-checked 3, correct).

## Verbatim report

| VERDICT | REVISE |
|---|---|
| Scope | branch dev @ d463e574; read-only; no writes, commits, sub-agents |

## PROSE (match-writing-style, audit)

| # | Sev | file:line | Quote (≤15w) | Status | Fix |
|---|---|---|---|---|---|
| P1 | major | docs/tutorial_figures.md:3 | "current development tree: 0.4.8" | observed | Update parenthetical to v0.5.x or delete it |
| P2 | major | docs/tutorials/05_v1_pfc_dual_column.md:131 | "shows stable, long-term homeostatic adaptation" | observed | Soften to settling-consistent; cite failing-possible gate or drop mechanism word |
| P3 | minor | docs/guides/plotly_visualization.md:24 | "development tree tested at 0.4.8" | observed | Re-test and label v0.5.x, or mark historical |
| P4 | minor | docs/tutorials/04_simulate_tensor.md:16 | "~215k (48 rules × p=1.0 bipartite; 215785…" | observed | Link generating receipt/run, or label observed values with date |
| P5 | minor | docs/api/probes.md:316 | "CSD source is located at 400 µm depth" | observed | Change example to relative fraction; note µm needs calibration |
| P6 | minor | docs/ci_policy.md:57 | "now proves exhaustiveness mechanically" | observed | Replace "proves" with "checks" + test name |
| P7 | minor | docs/tutorials/05_v1_pfc_dual_column.md:122 | "held at 12.53-12.55Hz across all 100" | observed | Values trace to receipt path at :144-145; OK, keep — listed for trace confirmation |
| S1 | minor | docs/tutorials/05_v1_pfc_dual_column.md:131 | "stable, long-term homeostatic adaptation" | observed (slop: intensifier stack + mechanism label) | Cut to one adjective; name the gate |
| S2 | minor | docs/ci_policy.md:57 | "proves exhaustiveness mechanically" | observed (slop: proves) | "checks exhaustiveness (test_release_gate_hierarchy)" |
| S3 | minor | docs/doctrine/protocol_w_hdp_parameter_memory.md:453 | "not robustly stable" | observed (slop-adjacent; margin-policy term) | Keep if defined term; else "margin <0.01" alone |

D0 spot-checks (assumed fixed, verified correct): objective_grammar.md:44-45 timing tied to `artifacts/perf/w11_atlas_10k.jsonl` ✓; install.md:9-10 PyPI 0.5.0 current, 0.4.25 previous ✓; 07_v037:200-201 relative-fraction units ✓. No µm leak in sampled guides/tutorials beyond P5. 0.4.x mentions elsewhere are provenance/frozen-contract labels, not current-version claims.

## VISUAL (match-visual-style, full-size PNG reads)

| # | Sev | File | Problem | Status | Fix |
|---|---|---|---|---|---|
| V1 | major | artifacts/figures/publication/fig01_tfne_grammar.png | Panel A title collides with brown "semantic nesting preserved" annotation; bottom legend strip overlaps panel F x-axis labels | observed | Move annotation above frame; lift legend clear of axis |
| V2 | major | artifacts/figures/publication/fig06_rbs_hdp_ladder.png | Panel B title/legend text overplots y=1.0 data line; panel C brown subtitle overprints panel title; panel E inset text truncated ("dot omega = 0 thro…") | observed | Offset B legend outside axes; separate C titles; enlarge E inset or drop micro-text |
| V3 | major | artifacts/figures/publication/fig03_local_observation.png | Lower panels y-labels "probe readout" / "readout" carry no units though colorbar reads "rel. units" | observed | Append "(rel. units)" to both y-labels |
| V4 | major | artifacts/figures/publication/final/e2_fig08_ping_combined.png | Panel B rep labels stack on identical coords (illegible); panel F callouts cross bar tops | observed | Jitter B labels or legend them; move E inset outside axes; shorten F callouts |
| V5 | minor | fig01 / fig06 / e2_fig08 | Small caption/annotation type (<6pt at print width) in fig01-F footer, fig06-G strip, e2_fig08-G table | observed | Raise to ≥6pt or move to caption text |
| V6 | — | fig03 colormap | RdBu_r diverging, labeled; no red/green-only encoding in sampled figures | observed | None |

NOT-RENDERED (HTML atlas, not judged): `docs/_static/atlas/objective_60/*`, `probe_32/*`, `source_column_48/*`, `assets/interactive/v037_source_column_3d.html`, tutorial iframe panels.

## HYGIENE (match-workspace-hygiene)

| # | Sev | File | Problem | Status | Fix |
|---|---|---|---|---|---|
| H1 | major | artifacts/issue_log/ISSUE_LOG.md:14 | Problem stack reads "(none)" while lens 1-2 defects exist (P1-P6, V1-V5) | observed | Log P-IDs for accepted findings; "(none)" is inaccurate |
| H2 | major | artifacts/fact_stack.md + git log | Last 3 touches (4196aca5, 9cb579b1, 24f724ba) include agent-style docs commits; human authorization for post-quiz fact edits not visible read-only | observed | Confirm human authorization per header rule; record it |
| H3 | minor | artifacts/todo_stack.md:118-132 | Completed/decided entries retained (D0 DONE, reorg DROPPED, AT-10-R4 OUT_OF_SCOPE) though role is remaining-work-only | observed | Move to seal receipt/state; keep decisions out of remaining list |
| H4 | minor | artifacts/ (30+ entries) | Extras beyond 7-file set (programme/, archive/, protocol_*, etc.); roles mapped in memory.md §7 + context.md | observed | Keep; mapping satisfies skill — no move |
| H5 | — | MEMORY.md (root) | Spot-checked ≤40/139 lines: lessons-only {trigger,cause,repair,evidence,scope}, no state/secrets | observed | None |
| H6 | — | scratch/CURRENT_TASK.md | Exists (Gate 0 mode source present) | observed | None |

Counts: PROSE 0/2/8 · VISUAL 0/4/1 · HYGIENE 0/2/2. Top-5: H1, H2, V1/V2, P1, P2.

## Integrator triage [I] (2026-10-04, dev)
- H2 DISMISSED: fact_stack log shows quiz-authorized rounds (0.5.5-facts rounds 12/13-14/15 + structural-HDP); authorization is recorded in the file header itself.
- H3 DISMISSED: Decisions-in-force is standing pre-existing convention, not drift.
- V1–V3 caution: fig01/fig03/fig06 are the FROZEN Figure 1–7 snapshot — files stay untouched; fixes go to the figure generators + future figures only. → P-034 records this constraint.
- Minted P-033 (prose batch: P1–P6 + slop) and P-034 (figure rendering, generator-side).
