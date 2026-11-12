# todo stack

Ordered smallest to largest by effort, end to end. `→` is the next step, `⇒` a branch on the outcome.
`[A]` agent, `[H]` Hamm. Each item names the command that closes it. A `∥` item runs beside the path in its own
lane (at most 3 lanes, one writer per file set, clones at `E:/repos/_wt/jaxfne/<lane>`).
Done items are deleted; history is in git.

Problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`.

Audit 2026-11-12: 0.5.10 checklist holds 40 of 100 (packages 40 of 70, gates 0 of 30).
- 10E PCL Fig 3 is fully scored: E1 R5-approximate scored FAIL under signed R4b
  (`pcl/prereg/P3_fig3_r5_approx_result.md`, `scores/fig3_r5_approx.json`); oracle arm compared (1246 equal,
  1922 differing; `scores/fig3_r5_approx_armcheck.json`); E2 P-055 attributed in `ISSUE_LOG.md`. 10E 10/10.
- 10B B4 `build_config` repair (2 pts) done (`scripts/cortical_column_localized_workflow.py`, `tests/test_build_config_declared_layers.py`).
- 10D D4 S16 Destexhe 2009 panel (2 pts) done (`scripts/publication_figures/supplement.py::write_s16`, `tests/test_supplement_generators.py`).
- 10F F4 GSDR log-scale parameters (1 pt) done (`artifacts/etudes/papers/gsdr2026/PORT_SPEC.md` §10.1).
- 10G G3 Reproducibility audit §7 (1 pt) done (`artifacts/programme/paper/reproducibility_audit.md` §7).
- Dashboard R5 row linked under `pcl-fig3` (T-73); `replications.json` and figures fresh.
- B1 drive unification (T-79) and baseline parameter match (T-80) done (`b1_harness.py`, `b1_brian2.py`, `b1_nest.py`, `b1_jaxley.py`; `--smoke` verified).
- `artifacts/state.md` corrected (`artifacts_check.py` ACCEPT).

## Tier 1: blocked on a ruling

0. T-93 [A then H] Size-budget conflict (open row O-R5SIZE): the pinned 3168-row armcheck record (413 KB) puts the
   tracked tree 283 KB over budget and `check_record_integrity.py` refuses to edit a pinned record, so both gates
   cannot pass at once. Closes when: `python scripts/check_tree_size.py --check` and
   `python scripts/check_record_integrity.py --check` both exit 0 on one SHA. Blocked by: Hamm's route ruling.

## Tier 2: 1 to 2 hours each

1. T-D3 [A] Mejias 2016 G6 (Checklist D3, 2 pts). Closes when: G6 is scored and pinned.
2. T-86 [A] G1 CAMINOS PLoS panels (Checklist G1, 3 pts): export the Fig 5A panel, then build the Fig 3 and 4A
    mean-field rows. Closes when: all panels are in the notebook, exported, from pinned runs.

## Tier 3: half a day

14. T-F5 [A] T-32 candidate batching for AdEx, LIF and `Model.tune` (`_model_tune.py:451`) (checklist F5,
    1.5 pts). Closes when: the batching tests pass across seeds.
15. T-83 [A] P-053 sparse-weight fixture: rebuild the minimal lateral and top-down fixture with hand-zeroed sparse
    weights. Closes when: it either reproduces the weight gap (P-053 confirmed) or passes (closed).
16. T-85 [A] PCL inhibition isolation: score one run of the v4 tree (port 5d101eb3) under `no-local`. Observed
    2026-11-12: re-scoring the v4 directory reproduces v4 to three decimals, and complex R_ori falls from 5.30 to
    1.19 under `no-local`; tree drift is the alternative explanation this run separates. Closes when: that score
    JSON is pinned beside the others and the result file's attribution paragraph cites it.
17. T-90 [A] P-056 kernel-LFP deep-layer power. Closes when: a settled column shows deep alpha-beta power or the
    cause is named.

## Tier 4: a day or more

18. T-87 [A] F4 scored benchmark across N on CPU and the A4000 with recording controls (checklist B1-B3, 10 pts),
    after T-79 and T-80. Closes when: the receipt records all 6 metrics and is evaluated against the purchase gate.
19. T-B5 [A] LIF edge kernel on GPU (checklist B5, 2 pts). Closes when: the GPU receipt equals CPU within the
    declared tolerance.
20. T-C1 [A] Augmentation ensemble study with provenance and the S7 panel (checklist C1, 4 pts). Closes when: S7
    renders from pinned receipts.
21. T-92 [A] 0.5.10 exit gates 1-5 (30 pts): broad gate, CI, replications audit and dashboard fresh on one SHA;
    version and changelog; second review receipt; then Hamm's tag and publish. Closes when:
    `python scripts/run_test_gate.py broad` and `python scripts/check_replications.py --check` exit 0 on one SHA
    and the review receipt is committed.

## 0.5.10 rows still open, carried with their checks

22. T-26 [A] Seed and candidate batching for AdEx, LIF and `Model.tune`; Mejias G6; the LIF edge kernel on GPU;
    F2, F4, S16 per the release plan. Closes when: each lands with its test or receipt.
23. T-R28 [A] Hamm's 2026-10-28 rulings left: dashboard P1 then P2 (ruling text not in the repo,
    `execution_plans_0.5.9-0.5.10.md:72`); the jitter reader on larger sets; C4's arbitrary-angle S3 panel; C6
    `geometry="laminar"` with Brian2 and Jaxley baselines. Closes when: each has a commit with its test or receipt.
24. T-31 [A] AGSDR log-scale bounds (landed 733b3960) used in the GSDR reproduction; a new fit pilot needs a new
    prereg. Closes when: the GSDR lane names its log-scale parameters.
25. T-32 [A] 0.5.11 lanes picked 2026-11-08 (O-87; `etudes/papers/{lee2025,mazzoni2015}/NOTES.md`): Lee 2025 Fig 6
    and Mazzoni 2015 Fig 5 RWS proxy. Closes when: each has a signed P3 prereg.
26. T-54 [A then H] awesome-jax PR after 0.5.10 and 7 replicated papers. Closes when: the PR is open with Hamm's go.
27. T-11 CAMINOS NWB excerpt for jnwb `tests/`, closes when `git -C D:/repos/jnwb grep -n caminos -- tests/`
    returns its test; T-RG komposer `run_guard.py stop` cwd fallback, closes when that test passes from a foreign
    cwd.

## Owner queue (asked one at a time, in this order)

28. [H] OCMA round 4: re-rule the rater with no reachable second model family on this host. Cause observed
    2026-11-12: Group Policy sets `ExecutionPolicy=AllSigned` so `cursor-agent.ps1` cannot run; the direct
    `node.exe index.js` launch gets past that but Cursor refuses `--sandbox` on Windows; agy produced no output.
    Options: a macOS/Linux host with Cursor; opencode Muse as the second family; defer the round.
29. [H] T-72 NeuroAI 2027: the anonymous code mirror, then your read and submit. Deadline 2026-11-20 23:59 AoE.
30. [A] T-25 retired names in dated records: history allowlist (asked in jchat, jynx keeper).

## Beside the path (∥, at most 3 lanes)

31. ∥ [A] T-8 standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev.
32. ∥ [A] T-10b jnwb proxy-claim wording gate or a dated drop; T-42 CI plan steps at the first rc; T-49 closed lane
    folders by kind.
33. ∥ [A] T-O68 remaining dirty worktrees (`base053`, `bisw1`, `frozen-etude`, `w3`, `jaxfne-wt-p022`,
    `jaxfne-wt-suites`).

## Deferred, each with its trigger

AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy; architecture split on a measured bottleneck;
C5-C7 zero/NaN contract; the UNTESTED-exact refusal tail and O-67 skips; S27 population family; agent transport
(MCP); 0.5.4 items 1b and 0; the 2026-09-20 owner calls; Atlas ladder; the mechanism rule proposal; stored-only
fields when a caller needs one; the tracked `w3b_scan_run.log` at the next W3b refreeze.
Directions, open: augmentation runs; `pcl_stdp_h2`; structural HDP; primate primary-connection mode; theory F3-F5
and Lean proofs; Python 3.15 in CI once its wheels exist.
