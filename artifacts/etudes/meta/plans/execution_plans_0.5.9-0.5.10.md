# Execution plans: open items and todos through 0.5.10

Written 2026-10-31 from `artifacts/open_stack.md`, `artifacts/todo_stack.md` (open rows), `release_plan_0.5.9-0.6.0.md`
(packages 9A to 9G and the 0.5.10 list) and the inventory of the pre-rewrite todo (`git show 6be045f7:artifacts/todo_stack.md`,
98 old items: 47 carried, 36 landed, 14 unmapped, 1 dropped). Four plans cover every open row and every open stack
item. Rows that wait on Hamm join no lane.

## Rules for every lane

- A lane is one executor: a Haiku 5.5 `worker` for sweeps, gates and mechanical edits, or an opencode Muse `code`
  packet (`delegate-cli`) for test-first code. Each lane writes in its own clone at the dev head it names.
- At most 3 lanes run at once across all plans. One writer per file set. Lanes that touch the same folder run in
  series (Plan 1 1e after 1a; Plan 4 4e last).
- Heavy GPU work: one job at a time in WSL (`~/venv_jaxfne`), claimed in jchat `p-jaxfne`.
- The orchestrator reads every diff, reruns each acceptance command, opens every number a lane cites, and checks
  every landing claim before it commits. Science verdicts stay with the orchestrator; a reviewer checks any
  science diff.
- Commit locally with exact paths. No push, tag, merge or branch: not authorised.
- Every lane report ends with a section `OPEN AND CONFUSIONS`: each open item, inconsistency or unclear claim the
  lane saw that is worth checking, with `path:line` and the one-line reason. The orchestrator turns each one into
  an open row or drops it with a reason. A lane does not fix a confusion outside its scope.

## Owner-gated rows (no lane)

| Row | Open item | Decision | Source |
|---|---|---|---|
| T-O73 | O-73 | rule private vs shared sources for the S1 drive (G3) | `open_stack.md`, `SOURCE_P1.md` line 18 |
| T-62 | Fig 1D | raw Onorato data: the paper's data statement names Martin Vinck, lead contact, on request | Onorato et al. 2025, data and code availability |
| T-46 close test | | close section 7 at 0 gaps or narrow to the 0.5.9 gaps | todo section 2 |
| Signatures | O-70 | Izhikevich P6, Potjans P3 v2, Vogels P3 v2, then Destexhe and GSDR after pilots, PCL R4 prereg | todo section 4 |
| P-052 | | surrogate fitness for training | `state.md` |
| T-O74 to T-O79 | O-74 to O-79 | keep retired selfcheck names as history, or edit the records | `open_stack.md` |
| T-54 | | awesome-jax PR, gated on the 0.5.10 release and 7 papers | todo section 4 |
| T-11, T-O10 | O-10 | external: jnwb path; jynx keeper takes or refuses each proposal (conference #29) | todo section 5 |
| T-RG | | external: komposer `run_guard.py stop` fallback | todo section 5 |

## Plan 1: Cell Reports replication (CAMINOS, Tahvili et al. 2025)

Rows: T-59, T-60, T-61, the Fig 1 part of T-62, the CAMINOS part of T-30. Release: 0.5.9 package 9D.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 1a | Muse code | Figs 2 bottom and 6B runners, packet `pk_fig2b6b_code.md`, baseline `98e7aefc` (launched 2026-10-31) | packet acceptance 0 failed; new tests fail on baseline |
| 1b | orchestrator, GPU | paper-scale Figs 3, 4, S3 (FIG-A runners), after 1a | receipts named in the report; verdicts with tolerances stated before the run |
| 1c | orchestrator, GPU | paper-scale Figs 2, 5, 6 (S1 waits on O-73) | same |
| 1d | orchestrator | Fig 1 panels B, C, D checked against the figure in the supplied paper (figure-level, not raw data); add the paper to `docs/reference/references.md` with its DOI | the references entry lands in the same commit |
| 1e | Muse code | CAMINOS estimators into `jaxfne.analysis` and the P3 core-route equivalence rerun, after 1a | `test_caminos_*` 0 failed; equivalence within the frozen between-seed range |

Acceptance for each lane: its pytest command 0 failed, and `python scripts/run_test_gate.py report` reads clean on
`replication_audit`.

## Plan 2: Replication core and conformance (PCL, pilots, GSDR, Izhikevich)

Rows: T-3 (R2 to R5), T-22 and T-25 (after R2), T-O70 (pilots only), T-32 (one replication a day), the non-CAMINOS
part of T-30, O-69 (PCL R2). Release: 0.5.9 packages 9C and 9E.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 2a | orchestrator, WSL | T-3 R2: the port's replay of the trained network's files through the chained scan, against the C++ reference (`~/pcl_cpp`, `data/events/shapes.h5`); then R3 residual attribution | R2 receipt in `artifacts/etudes/papers/pcl/`; R4 waits on its signature |
| 2b | Muse code | T-32 current open rows, rewritten first (see Plan 4 4f): core homes for ISI ratio and CV, aperiodic fit, tuning fit (T-32d); GSDR fit on `simulate_hh_batch` with `gsdr_mcdp_transform`, test first | each lands with its test or record |
| 2c | Haiku worker, high | T-O70 pilots: Potjans 1% and 2% at in-degree held, Vogels estimands, GSDR and Destexhe pilots on seeds 0 to 3 (scoring seeds 9 to 13 reserved); pilot receipts only, no signature | pilot receipts exist; each signature stays with Hamm |
| 2d | Haiku worker, high | T-22, T-25: PCL B3 on the core intra-step operator; trained weights load as A' (`with_trained_weights`), after 2a | both run on the core with tests |
| 2e | Haiku worker, high | T-30 non-CAMINOS entries: Izhikevich net.m order through `construct`; GSDR HH dispatch; Destexhe estimators, LEDGER, RESULT; pcl_h and Q-learning phase records and `sys.path` imports | `python scripts/check_replications.py --check` lists no violation without Hamm's exception |

## Plan 3: Protocols, paper items and dashboard

Rows: T-R28 (the executable parts, open stack O-72), T-45, T-46 (the 8 gaps; close test gated), T-10b, T-8, O-68
(local copies, rescue then delete, see Plan 4 4a). Release: 0.5.9 packages 9A, 9B and 9G.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 3a | Haiku worker, medium | O-72: dashboard P1 then P2 (Hamm's ruling 2026-10-28); regenerate. BLOCKED 2026-10-31: the P1 and P2 text is not in the repo (`todo_stack.md:42`, `open_stack.md:16`, `state.md:12` cite it; `dashboard/ARCHITECTURE_v2.md` predates the rulings). Needs Hamm or the keeper to commit or paste the text | `python scripts/build_dashboard_data.py --check` passes |
| 3b | Haiku worker, high (test-first; launched on Haiku, not Muse, 2026-10-31) | T-45: map D17, D21, D26, D28, D31, D32, D34, D35 to public symbols; object specializer `PV[model=IZH]` (test first) | each row names a public symbol with a test; `supplement.py` regenerates S1, S2, S5 |
| 3c | orchestrator | P-052 beta scan against the central finite difference, output-changing, with a changelog fragment | scan receipt; reviewer reads the diff |
| 3d | Muse code | C4 Euclidean radius cutoff (rotation-invariant candidates; arbitrary-angle S3 test); C6 `geometry="laminar"` with Brian2 and Jaxley baselines declared | tests pass; the arbitrary-angle test present |
| 3e | Haiku worker, high | T-46: the 8 gaps in `programme/paper/reproducibility_audit.md` section 7 that the 0.5.9 scope holds | section 7 lists the remaining gaps, each named |
| 3f | Haiku worker, high | T-10b: jnwb proxy-claim wording gate, with its test | gate lands with its test |

T-8 runs after each push that changes dev: dashboard refresh (skill `jaxfne-dashboard`).

## Plan 4: Fields, records, verification and the 0.5.10 backlog

Rows: O-68 and T-O68, T-47, T-48, T-49 (after Plans 1 and 2 close their lanes), T-42 (waits on the first live CI run,
which needs a push: not authorised, so no lane), the unmapped and stale items below. Release: 0.5.9 package 9F and 0.5.10.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 4a | Haiku worker, medium | O-68: rescue each cited copy (`cl-reader-cursor`, `jaxfne-wt-p022`, the WSL copies, the non-git leftovers) into the repo or `F:/warehouse` with a `state.md` pointer; list each deletion | each rescue verified; the orchestrator confirms each delete one at a time |
| 4b | Muse code | T-47: cherry-pick `6eb3c6b8`, `44cd0111`, `ff45364b` one per command; then the iSTDP out-degree check; then `coherence_radius_um` with its two limit tests | on dev with tests |
| 4c | Muse code | T-48: Jansen-Rit neural-mass emitter in the core, then P1, after 4b | P1 recorded |
| 4d | Haiku worker, high | 9F: JUnit parity classifier, figure lock file, clean-checkout figure job | rc gate end to end on the candidate |
| 4e | Haiku worker, medium | T-49: split closed lane folders by kind (`etudes/papers/caminos/`, `izhikevich2003/`, `brunel2000/`), after Plans 1 and 2 close | their exceptions leave `tests/test_artifacts_tree_degree.py` |
| 4f | Haiku worker, high | Rewrite the stale T-32 rows: landed per the inventory (Brunel S2 and S3 `c4406032`; Potjans G-TAUSYN `5a8d4ad4`, G-POPDRIVE `47c8076b`, G-WDIST and G-DDIST `9dc6afd3`; Vogels G-SC `94e62ef0`; Mejias phi(0)=1 `91d9fb24`); Potjans G-MULT (`b4d4dc5c`) landed, no row needed | T-32 lists only what is still open |
| 4g | Haiku worker, high | Verify each UNMAPPED old item (list below) with a `path:line` or a sha, then open a row, or drop it with its reason | each item has a row or a dropped reason |

Verified 2026-10-31 (workflow `wf_60293058-ded`, 6 read-only agents, adjudicated). Landed, no row: T-O69 E1
delayed-edge route; GSDR HH notes (held_steps, int32, full-kernel grad test); GSDR multi-compartment route; HH route
hardening; T-PR (Dependabot #115 and #116 closed unmerged 2026-10-28, named branches gone); T-OWN examples 03/05
fallback, testpypi checklist, protocol docstring, dev-gate double compile; T-33 records, api refusals, foil sweep;
T-19 design items, drive defaults, release-plan A5 reference; T-22 rulings 1 to 4; CAMINOS lag ruling. T-33.tooling
REFUTED (not landed; reopen as a row).

Still open, each a row for 4g to open or hand to the owner:
- Mejias G6 and the LIF edge kernel on GPU (kept in T-32).
- T-32e F3 rule: not in `scripts/check_replications.py`; F3 landed in `scripts/check_fidelity_fix.py` (9a1efe20). Owner's call whether it moves.
- GSDR dt: `jaxfne/_construct_hh.py:335` `hh_kernel_args` still coerces `dt="0.1"` via `float()`; no test at that entry.
- T-30 records: SOURCE_P1 drafts, stage_artifacts paths, hh-at01, GSDR O-25 estimand ruling.
- T-16 Atlas test-file merge; T-17 P3 audit owner calls; T-33.tree-degree; T-19 Izhikevich state clamp (Hamm's call).
- T-OWN: `statuss` typo in 8 lines (`git grep statuss`); `examples/_tutorial_runtime.py:51` `or DEFAULT_OUT_ROOT`;
  `scripts/consolidate_hdp_controllability_etude.py:38` names an absent script; `except ImportError: pass` handlers in examples 02 to 05.
- UNCERTAIN, needs a look: T-OWN/examples-v031-v032 JSON targets; T-OWN/tests-comment-paths.

Deferred items with a trigger (todo section 5) keep no lane.
