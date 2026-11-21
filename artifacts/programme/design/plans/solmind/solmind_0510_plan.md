# Solmind 0.5.10 plan: execution contract, tasks, decision gates

Status: independently REVIEWED (audit 3 ACCEPT, 2026-11-21, §7); execution AUTHORIZED by Hamm 2026-11-21 for plan commit 5550cf07 (S19, "Go: merge and execute"), up to READY_FOR_OWNER_GO with no tag, main or PyPI. Rows behind open Hamm gates (S02-S04, S08, S10, S11, S16-S18, S20-S22) stay BLOCKED. Canonical execution authority remains JaxFNE's `artifacts/AGENTS.md`, `artifacts/control/skills/jaxfne-release/SKILL.md`, frozen release plan and current project stacks. This file routes the tasks in §8 and the decisions in §9 (merged 2026-11-21 from `solmind_0510_todo.md` and `solmind_0510_open.md`, finding A1); it does not replace or weaken any of them.

## 0. Objective, baseline, boundaries

Objective: make 0.5.10 releasable at the smallest justified scope and verified cost, without changing frozen scientific verdicts or owner-reserved publication authority. The frozen checklist in `artifacts/etudes/meta/plans/release_plan_0.5.9-0.6.0.md` governs points (70 package + 30 gate). Recorded 43/100 is historical (2026-11-19), not an acceptance result at the current HEAD. Under the frozen contract, Gate 5's six points are earned only after actual publication. Before publication, report `PREPUBLICATION READY/NOT READY/UNKNOWN` separately, never silently remap 94 to 100. Do not claim READY with a nonempty issue-log Open section, an undrained release todo section, missing mandatory check, or unknown required host.

Baseline of this plan: written on `jaxfne@dev@10c417adced394e799b922e78315a0be1f8a82dd`; audited against live `dev` `3eb6a6a64c1654637eb9e02011324c44ce10618a` (10c417ad..3eb6a6a6 changes only the three stacks); at execution, fetch the live `dev` SHA and rebase/reconcile the plan, not the scientific evidence. `_temp_solmind_0510_plans` is planning only, not an authorized implementation or release branch. Target integration `dev`; `main`, tags, GitHub Release, PyPI and publication require Hamm's explicit exact-SHA go. Never force-push or rewrite pinned records. Temporary branch retirement follows independent review and preservation of unique evidence.

## 1. One entrypoint, one accountable lead

The lead reads Jarness `main` `rules/start.md`, JaxFNE `artifacts/AGENTS.md`, `artifacts/context.md`, `artifacts/state.md`, `artifacts/open_stack.md`, `artifacts/todo_stack.md`, issue-log Open section, the frozen 0.5.10 release checklist, and this file. Check `git status`, `main/dev` SHAs, active worktrees and outstanding edits. Git and native gates outrank stale state prose. Read relevant scientific specs only when their task becomes ready. No routine full-repository rereads.

Lead owns integration, reviewer assignment and score bookkeeping, and proposes scientific trade-offs and statistics designs; Hamm rules on them (§3). Delegate bounded tasks to inexpensive workers; at most three concurrent lanes INCLUDING reviewers. One writer per file set. A non-writer reviewer examines every consequential change and reruns acceptance on the integrated SHA. Reviewers do not edit writer files.

A worker packet MUST state: task ID; base branch and full SHA; read/write paths; immutable paths; dependencies; expected minutes; objective and falsifier; estimand, units, controls and stopping rule when scientific; exact commands; output/receipt paths; success and stop criteria. A worker reports changed files, tested SHA, command outputs, failed checks, residual risks and one closing verdict. Missing receipt = UNKNOWN.

## 2. Dependency graph and lanes

| Package | Task IDs | Owner | Prerequisite | May share the 3 lanes with | Completion |
|---|---|---|---|---|---|
| A: baseline/authority | R01-R04, R07-R11, R70-R72 | lead; R07-R09 behavior: Hamm (S04) | live state and frozen release plan; S21 for R04 | read-only design for C | scores and defects reconciled; baseline gate receipt |
| B: owner-gated numerical repair | R05-R06 | Hamm (S02, S03); lead executes | Hamm ruling on S02/S03 | C, D | exact-SHA Mac/CI checks pass |
| C: ephaptic scientific audit | R12-R27 | lead proposes; Hamm signs preregs (S17) | existing operator and source/field contracts; S08 for R25 | A, D, E | mathematical/unit tests, adversarial controls, bounded claim verdict; defer physical claims when unproven |
| D: release packages | R28-R41 | lead; Hamm for S10, S11, S17 rows | preregs, equipment, owner decisions for affected rows | C, E | each frozen checklist row has its own receipt |
| E: docs/maintenance | R42-R50 | lead | live public API inventory; R46-R50 after R42-R45 selection; S16 for any added file | C, D | measured duplication reduction; API parity, clean tutorial and docs gates |
| G: issue-log drain | R56-R69 (P-066, P-056 and the P-025 regression drain through R17, R25, R70 in their own packages) | lead fixes; Hamm rules SCIENCE interpretation and any move without a fix | none beyond each row | A, C, D, E | issue-log Open empty, each entry with a resolution line in a dated closed batch or a dated Hamm move |
| F: release closure | R51-R55 | lead; R55 Hamm | A-E and G mandatory scope; no open blocker; S18, S19 | none for final candidate freeze | Gates 1-4 exact-SHA, independent ACCEPT; R55 owner-reserved |

"May share" names packages that can hold lanes at the same time; the cap of three concurrent lanes including reviewers (§1) governs every combination. Within packages, do not interpret task numbering as a strict total order. Work on ready tasks; block dependents only. If a package is not required by the frozen 0.5.10 checklist (for example an experimental ephaptic paper claim or a major docs merger), it MUST NOT delay release unless a reproducible safety/correctness defect or owner ruling makes it mandatory. Every issue-log Open entry is mandatory regardless of package, because the release skill blocks a release with a non-empty Open section (`artifacts/control/skills/jaxfne-release/SKILL.md`, RULES and STANDING ACCEPTANCE); this covers R17 (P-066) and R25 (P-056) in package C. Put discretionary work after the release or in a separately authorized manuscript lineage. Preserve the 15-stage paper programme as distinct from the 0.5.10 score.

## 3. Autonomy, decisions, escalation

Proceed autonomously on reversible, authorized, in-scope changes with checkable acceptance. If a fix takes <=15 minutes, make the minimal repair and test; otherwise record one todo with its closing check and continue another ready task. Two failed approaches require revising the approach, not more retries. Never relax a threshold, change a frozen preregistration, reinterpret a failed replication, repin an immutable file, or expand the release scope to force PASS.

Owner gates (§9): S01 score semantics (proceeds under the frozen score without a decision); S02 fig05 re-freeze; S03 numeric acceptance semantics; S04 API behavior for O-73..O-75; S06 ephaptic claim bounds; S08 P-056 inputs; S10 Mejias signature; S11 missing dashboard P1/P2 ruling; S15 release publication; S16 file budget; S17 prereg signatures for R20, R21, R36; S18 CI workflow enablement; S19 execution go; S20 frozen Owner row O-85..O-87; S21 matching-host definition; S22 compatibility scope beyond the frozen checklist; S23 repin approvals for output-changing fixes. S07 spatial units was ruled 2026-11-19.

Authority: signing or amending a preregistration, changing an acceptance criterion or threshold, re-freezing a figure, raising `budget_files`, enabling a disabled workflow, and moving an issue-log entry without a fix are Hamm's, unless a dated standing ruling covers that exact item. Agents may prepare and validate preregistrations and decision packets.

Conflicts: when two canonical sources disagree (a Hamm ruling, the frozen plan, a stack, a skill, this file), stop the affected item, record both sides with file:line, mark it BLOCKED, queue its ranked question under the asking rule below, and continue other lanes. Dated Hamm rulings and the frozen plan outrank this file. Ask ONE ranked question only when its answer blocks the next material action and no other authorized ready work exists. Never assume silence is approval. An optional task blocked by an owner decision stays BLOCKED with a precise packet; other lanes continue.

A consequential scientific uncertainty (e.g., ephaptic physical interpretation) yields a restricted claim or explicit UNKNOWN, not an invented result. An unresolved required gate blocks release. An issue-log entry closes either by a fix, with a resolution line and its evidence and a move to a dated closed batch (issue-log header), or by a dated Hamm move with an executable new closing check; it is never silently removed, downgraded or relabeled.

## 4. Mandatory checks and evidence

At the task level, use the smallest relevant native test plus an adversarial negative or mutant where it would otherwise pass vacuously. At integration, run the relevant native gate. For code changes, `python scripts/run_test_gate.py broad --no-stamp` on the matching host (S21) before push, so the receipt carries the run's own counts; for docs, `python scripts/run_test_gate.py docs`; `python scripts/run_test_gate.py report` is audit-only and is not a broad test. For release (Gate 2), on the release SHA: `python scripts/run_test_gate.py rc`, CI green, `python scripts/check_replications.py --check`, `python scripts/check_tree_size.py --check`, `python scripts/check_record_integrity.py --check`, `python scripts/build_dashboard_data.py --check`, `python scripts/build_dashboard_figures.py --check`, `python scripts/generate_replications_page.py --check`, `mkdocs build --strict`, `python scripts/run_test_gate.py wheel-smoke --venv-python <venv python>`, and for Gate 3 `python scripts/release_body.py --tag <tag> --check`, plus candidate-specific release validation in the release skill. Every packet that adds tracked files states its file-count delta against `budget_files` (S16). Use exact commands available on the current SHA; if a named command does not exist, stop that check and correct the packet, not invent a passing substitute.

For each PASS, record repository, branch, full SHA, host/runtime, command, exit status, test counts, sample size, artifact hash or path, reviewer, and timestamp. Distinguish IMPLEMENTED, TESTED and independently VERIFIED. Missing runtime/host/receipt is UNKNOWN, not PASS. A receipt counts only for the SHA it names; for any other SHA, including the candidate, it is UNKNOWN. A count from another host, environment or SHA (for example another agent's base) is not a baseline. All negative and null results are retained. Do not retune criteria after outcomes. Review the actual diff, not only an agent's summary.

The lead recomputes the frozen checklist after each package, recording each scored item and its receipt. Score can rise only for verified named checks; a merged commit alone earns no points. Track `score/100`, `coverage/100`, `prepublication gate verdict`, and outstanding owner gates. Avoid another shadow scoring system.

## 5. Scope-specific acceptance

**Ephaptic:** Check whether the implemented voltage-neighbor operator is dimensionally and numerically consistent across LIF/Izhikevich, with correct lag, spatial normalization, alpha=0 identity, dt refinement, zero/shuffled/perturbed controls, and imposed-field comparator. Distinguish phenomenological lateral voltage coupling from physically calibrated extracellular fields. If no physical calibration, do not advertise a validated ephaptic mechanism. Do not move it into RBD/HDP semantics without the required state/weight dynamics. Keep experimental components outside core release obligations unless the existing release acceptance requires them.

**Merger and prose:** Inventory duplicate modules, import graphs, generated mirrors, stable public paths, tests, notebooks and sealed artifacts. Choose one highest-return merger; require unchanged API behavior and checked imports, preserved history, meaningful reduction in duplication, and independent review. A large merger is not a target by itself. Measure baseline/final files, words, duplicate statements and tests. Do not delete archived evidence merely to lower counts.

**Tutorials:** Classify active/archived JaxFNE tutorials; inspect jomission examples at pinned SHA. Verify provenance, license, scientific claim status, dependencies, API compatibility and exact executable cells. Prefer 2-3 minimal clean-install tutorials over mass migration. Jomission is downstream scientific usage, not a validation authority for JaxFNE. No copied result can become a package guarantee.

## 6. Continuation and stopping

Each integration cycle: check remote SHA and local status -> choose highest-priority unblocked task -> run bounded packet -> inspect diff -> run relevant tests -> non-writer review -> exact-path commit/push only under existing authority -> update canonical todo/open/state as applicable -> recompute release score -> repeat. The Solmind planning files are design history; avoid a parallel live tracker after integration. Use git history and receipts for completed items, remove closed todo entries. If a task changes the plan, commit its revision to this file at its integrated path on `dev` (before integration: the planning branch) before proceeding.

Terminal states:
- `READY_FOR_OWNER_GO`: all mandatory prepublication release checks pass on ONE immutable candidate SHA, issue-log Open section empty, release todo section drained item by item (release skill STANDING ACCEPTANCE; on dev that section is `todo_stack.md` `## 0.5.10 rows still open, carried with their checks`), independent review ACCEPT, no required UNKNOWN; no tag/main/release yet.
- `PUBLISHED`: Hamm authorized exact SHA and release skill confirms tag, main, GitHub Release, PyPI, hashes and clean install; frozen checklist 100/100 only when all checks truly pass.
- `BLOCKED`: no authorized ready task remains. Report exact unresolved owner questions, impacted checklist points, best safe next action and receipts. Never report completion from static inspection or partial tests.

Handback (<=15 lines plus links): branch@SHA; verified score and coverage; ready/not-ready/unknown; tasks closed; files/receipts; gate results; remaining blockers with owner and one decision question if needed; next ready task. No repetitive status narratives.

## 7. Plan audit before autonomous dispatch

Acceptance of this plan as an agent goal requires a reviewer to check: (a) all tasks (R01-R72) and gates (S01-S23) have a routing home (audit 1 checked the original 55 and 15); (b) each mandatory release item 10A-G and Gates 1-5 has an owner and acceptance evidence; (c) no frozen acceptance changed; (d) every owner-only action is gated; (e) three-lane cap and reviewer independence hold; (f) simulated missing-host, failed-gate, conflicting-authority and stale-SHA scenarios produce BLOCKED/UNKNOWN rather than false PASS. Until this audit is executed, the plan is SPECIFIED, not independently VERIFIED.

### Audit 1 (2026-11-21): REVISE

Plan a971288d against dev 3eb6a6a6; ":NN" cites in this audit-1 table refer to the a971288d contract file, not to this merged file. Two independent reviewers (Sonnet high, Opus medium), both REVISE; union below, deduplicated. "v" = dispatcher re-opened the source and confirmed. Passed: R01-R55 and S01-S15 unique and each in one package; no frozen weight, threshold, prereg or score semantics changed; tag/main/Release/PyPI, re-freeze (S02), numeric criteria (S03), Mejias (S10), P-056 inputs (S08) gated to Hamm; every named command exists on dev with the flags used; 10c417ad..3eb6a6a6 changes only stacks; missing-host -> UNKNOWN (:7, :44) and failed-gate -> BLOCKED (:38, :61) have forcing sentences.

| # | Sev | Finding | Repair |
|---|---|---|---|
| A1 | blocker (v) | Integration fails `check_tree_size.py`: dev 2813 files = `budget_files` 2813; a971288d has 2816 (`check_tree_size.py:110-111`, in report/docs/broad gates). Later tasks add files too. | Hamm rules on `budget_files`, or free files with match-cleanup evidence first; merge the plan files; file-count delta in each packet. |
| A2 | major (v) | Issue-log Open holds 16 entries (P-040, 047, 052, 053, 055-066); only P-056 (R25) and P-066 (R17) have tasks; R03 assigns owners only. READY needs Open empty (:61; release skill), so no route to READY. :30 (package C non-delaying) contradicts this for P-056/P-066. :38 reads as requiring Hamm even for a fix with a resolution line. | Package G "issue-log drain" with one row per entry; scope :30 to exclude Open-listed defects; allow fix-closure by resolution line in :38. |
| A3 | major | Conflicting authority and stale-SHA receipts have no forcing sentence (§7f). Live conflicts: matching host (R04 "Linux/Windows" vs dev T-MATCH "Linux, jax 0.10.2"); paper-stage numbering (R20-R21 vs O-7 "provisional"); A2; A5. | Add: "Two canonical sources disagree: stop that item, record both with file:line, mark BLOCKED, ask one ranked question, continue other lanes; frozen plan and dated Hamm rulings outrank Solmind files." and "A receipt counts only for the SHA it names; any other SHA is UNKNOWN." Define matching host once. |
| A4 | major (v) | :15 "Lead owns scientific tradeoffs, statistics, authority" hands Hamm's decisions to an agent; prereg signing gated only for S10. R20, R21, R36 prespecify estimands/seeds with no Hamm gate (H3 makes signatures Hamm's). | :15 -> "Lead owns integration, reviewer assignment, score bookkeeping; proposes scientific trade-offs." Add to §3: signing or amending a prereg, changing an acceptance criterion, re-freezing a figure is Hamm's unless a dated standing ruling covers that item. |
| A5 | major (v) | S04 (O-73..O-75, public-API semantics) owner "Release lead"; canonical `open_stack.md:16-18` gives O-73/O-74 to Hamm (release), O-75 to Hamm. S04 absent from the gate list :36. | S04 owner Hamm, BLOCKED on the behavior ruling; agents prepare comparison and tests; add S04 to :36. |
| A6 | major (v) | Closing evidence weaker than frozen checks: R54 lacks "receipt committed" (Gate 4); R31/R32 omit vmap and float32/64 (B2); R40 omits the jitter reader (F6); R53 and §4 omit CI on the release SHA and name no dashboard command (Gate 2); `release_body.py` needs `--tag`; R10 closes on "skips justified" vs O-76 strict xfail or passing test. | Copy the frozen wording into each row; name `build_dashboard_data.py --check`, `build_dashboard_figures.py --check`, `generate_replications_page.py --check`, `run_test_gate.py rc`/`wheel-smoke`. |
| A7 | major | Owner acts not routed: Publish and CI (Release & Scheduled) workflows disabled since 2026-11-14 (dev state.md), needed by Gates 2 and 5; plan is "authorization for planning only" (:3) while dev T-PLAN-EXECUTE says "work continuously". | S row for workflow re-enable (Hamm); T-PLAN-EXECUTE blocked on Hamm's dated execution go. |
| A8 | major | Broad gate on dev fails `tests/test_script_repo_root_guard.py::test_scripts_and_artifacts_assert_repo_package` (`scripts/p056_resource_smoke.py` lacks the repo-root assert; dispatcher full suite on 98d42ce4, same failure on 3eb6a6a6). No row owns it. | P0 row in package A: add the assert; check: the test passes. |
| A9 | major | S07/R17 stale: Hamm ruled P-066 on 2026-11-20 (realized z in mm, divide by `column_height_mm`, stored in params); repair and dated ruling on `_temp_tfne-manuscript` 98d42ce4 (ISSUE_LOG P-066), not on dev. | S07 RULED 2026-11-20; R17 -> review and integrate 98d42ce4 (before/after table, broad gate, non-writer review). |
| A10 | major | No task for JAX/jaxley compatibility: dev pins `jax>=0.4.25,<0.12`, `jaxley>=0.13.0`; shim `jaxfne/bridges.py:19-47` documents jaxley<=0.13; jaxley 0.14.0 installed on vwin-pc. A tier-2 report (Sailor, candidate aa260a01, logs not verified here) cites a `jnp.clip(a_max=)` failure, order-dependent `tests/test_score_port_parity_checks.py` (outside R10/R11 scope), Izhikevich drift, engine 0.5.8 untested, and 12 failed on its base vs 3 here. | Task: minimal repro, shim coverage test, narrow fix, no pin change without evidence; extend R10/R11 to order dependence (isolation + random order); map drift to R06; R04 records SHA, host, env; foreign counts inadmissible. |

Minors: R and package rows lack an owner column; §2 lists C beside A, D, E (4 lanes vs cap 3); §6 :58 commits revisions to a branch T-PLAN-INTEGRATE deletes; S01 BLOCKED vs :36 "can proceed"; R17 under both S06 and S07; frozen Owner row (O-85..O-87) and Crazy Sailor O-1..O-9 unrouted; broad-gate stamp reuse prints no counts (require `--no-stamp` or a full-run line).

### Audit 1 dispositions (repair revision, 2026-11-21; Hamm's scope of 2026-11-21: repair A2, A3, A5, A6, A8, A9, A10 and minors; A1 free files first; A4 preparation authorized, signing reserved; A7 no workflow enablement or release execution)

| # | Disposition | Where | Closing check |
|---|---|---|---|
| A1 | PARTLY REPAIRED, gate S16 kept. Three plan files merged into this one (-2). `artifacts/etudes/meta/plans/execution_plans_0.5.9-0.5.10.md` (2026-10-31 lanes for 0.5.10, not a pinned record: 0 keys in `scripts/record_hashes.json`; one pointer, `todo_stack.md`) folded verbatim into Appendix A (-1). Net against dev 3eb6a6a6: 2813 -> 2813 files. A worker inventory of 24 byte-duplicate groups, 3 empty files and 9 leads found no other removable file without unpinning records or breaking references. Every later added file, including each dated closed issue batch, needs S16 or an equal removal. Audit 2 found the merged file made `programme/design/plans/` hold 8 files against the 7-file folder limit of `tests/test_artifacts_tree_degree.py`; the file now sits in `plans/solmind/` (dev's T-PLAN-INTEGRATE path), leaving `plans/` at 7 files and 4 folders. | §8 Appendix A; §4 last sentence; §9 S16 | `python scripts/check_tree_size.py --check` and `pytest tests/test_artifacts_tree_degree.py` exit 0 on the repaired SHA |
| A2 | REPAIRED. Package G, R56-R69: one row per undrained Open entry; §2 makes every Open entry mandatory; §3 allows fix-closure by resolution line. | §2, §3, §8 | 14 rows, one per entry; ids match the `## Open` headings of `ISSUE_LOG.md` |
| A3 | REPAIRED. Conflict and SHA-binding rules added; matching host has two definitions (T-MATCH "Linux, jax 0.10.2" vs state.md labelling a Windows run matching-host), so S21 is BLOCKED under the new conflict rule. Paper-stage cites in R20/R21 now name O-7 as provisional. | §3, §4, §8 R04, R20, R21, §9 S21 | scenario walk recorded under Audits 2-3 below |
| A4 | REPAIRED in text; gate S17 kept. §1 lead role; §3 authority rule; R20, R21, R36 prepare and validate, Hamm signs. | §1, §3, §8, §9 S17 | audit 3 criterion 3 ACCEPT |
| A5 | REPAIRED. S04 owner Hamm, BLOCKED on behavior; R07-R09 prepare comparison and tests. | §3, §8, §9 S04 | S04 owner equals `open_stack.md` O-73..O-75 |
| A6 | REPAIRED. Frozen wording copied into R10, R31, R32, R40, R53, R54; Gate 2 and Gate 3 commands named and verified to exist with their flags on 3eb6a6a6 (audit 2: `wheel-smoke` also needs `--venv-python`, now written); READY and the release invariant now also require the release todo section drained; R52 carries changelog and release SHA. | §4, §8 | audits 2-3 against `release_plan_0.5.9-0.6.0.md:81,102,109,110,111` |
| A7 | REPAIRED in text; gates S18, S19 kept. No workflow enabled, no execution begun. Dev's T-PLAN-EXECUTE still reads "work continuously"; this branch cannot edit dev's stack, so S19 records the conflict and T-PLAN-INTEGRATE sets T-PLAN-EXECUTE to "Blocked by: S19". | §9 S18, S19; status line | after integration: dev `todo_stack.md` T-PLAN-EXECUTE reads "Blocked by: S19" |
| A8 | REPAIRED. R70: the P-025 regression gets a new P-ID and the assert. | §8 R70 | the named test passes |
| A9 | REPAIRED. S07 RULED 2026-11-19 (audit 1 said 2026-11-20, taken from the dispatcher's packet; the source `ISSUE_LOG.md:423` on 98d42ce4 and the author dates of commits 3902bd21..9bd24e5c read 2026-11-19); R17 becomes review and integration of 98d42ce4. | §8 R17, §9 S07 | audit 3 criterion 1 ACCEPT (ruling line and dates) |
| A10 | REPAIRED. Compatibility measured on the pyproject matrix (R71); order dependence R72; drift and engine probe added to R06 and R04. | §8 R04, R06, R71, R72 | R71 matrix receipt |
| minors | REPAIRED: package owners (§2); lane wording (§2); plan-revision home (§6); S01 status; R17 single gate (S07 ruled, S06 keeps R12-R16, R18-R22); frozen Owner row -> S20 and R51; Crazy Sailor O-1..O-4, O-6..O-9 declared outside the 0.5.10 checklist (they score no 0.5.10 point; canonical `open_stack.md` keeps them); stamp reuse -> `--no-stamp` (§4). | as listed | audit 3 criterion 1 ACCEPT |

### Audits 2-3 (2026-11-21)

- Audit 2 on 30575b03 (Opus medium, Sonnet high): both REVISE. Blocker: the merged file made `programme/design/plans/` hold 8 files against the 7-file limit of `tests/test_artifacts_tree_degree.py`; repaired by the move to `plans/solmind/`. Minors repaired in 51b74933 (S22, S19 conflict, READY drain, `wheel-smoke --venv-python`, R52, S16 receipts, pinned-record inventory, appendix status).
- Audit 3 on 51b74933 (Opus medium, Sonnet high): both ACCEPT, no blocker or major. Scenario walk quoted: stale SHA §4 "A receipt counts only for the SHA it names"; missing host §4 "Missing runtime/host/receipt is UNKNOWN, not PASS"; failed gate §3 "An unresolved required gate blocks release"; conflicting authority §3 "stop the affected item, record both sides with file:line, mark it BLOCKED". Their minors are repaired in the following commit: S06 in the §3 list, S21 and S19 file:line cites, the release todo section named, R07-R09 authority, SCIENCE-row closing, R71 receipt fields, R71 after S22, S23 repin approvals, the ruling/repair split for P-066, closing-check cells.

## 8. Tasks

Status: execution authorized 2026-11-21 (S19); a row runs only when its own gates are clear. Read the frozen release plan, `artifacts/programme/design/plans/arxiv_paper_plan.md`, canonical stacks and issue log before execution.

**Score contract:** Frozen 0.5.10 checklist: 70 package points + 30 release-gate points, each scored only on its named check; last recorded 43/100 at 2026-11-19, NOT a current verified score. Gate 5 (6 points) requires publication; 100/100 under current authority means published, not merely ready. Owner's desired prepublication readiness=100 is a distinct proposed metric, NOT adopted; do not rewrite historical scoring without ruling.

**Dependencies:** R01-R11, R70 and R72 first; R71 after S22; R12-R27 ephaptic/theory can proceed as a bounded lane beside R28-R41 release packages and R56-R69; R42-R50 docs and code hygiene only after API inventory; R51-R55 sequential release closure. Max three lanes including reviewers. No unapproved changes to frozen figures, preregs, scores or release targets. Negative replication results remain evidence, not release failure unless their registered gate says so. Preserve original 15-stage manuscript authority and new figure remapping; its stage numbering is provisional until O-7 closes.

**Task protocol:** For each row, record base SHA, read/write scope, expected minutes, file-count delta, falsifier, positive/negative controls, exact closing command and non-writer review before execution. An output-changing fix (for example R61-R69) first lists every frozen receipt, prereg and pinned record it changes; any repin goes to Hamm (§3). The closing phrases below are acceptance designs, not observed PASS. Smallest sufficient change, no automatic scope expansion.

| ID | Priority | Action | Authority | Closing evidence |
|---|---|---|---|---|
| R01 | P0 | Reconcile release score against exact dev SHA | 10A-G and gates | Checklist recomputed with evidence per scored item. DONE 2026-11-21 on dev b3a21045: 43/100 holds (packages 43 of 70, gates 0 of 30), sum of A3 4, A4 3, B1 3, B4 2, C2 2, D1 4, D2 4, D4 2, E1 7, E2 3, F1-F4 5, G1 3, G3 1. Every cited repo path resolves; the cited tests (geometry_radius_rotation 23, build_config_declared_layers 5, simulate_seed_determinism 7, kernel_lfp 22, emitters_jansen_rit 19, supplement_generators 18, balanced_preset_acceptance 6, agsdr_log_scale 16, in_degree_refusal_message 4) all pass in the b3a21045 broad gate (8459 tests, 0 failed). G1's runs and E1's `approx_diff.json` sit in `F:/warehouse/` on vwin-pc, not in the repo: fig4a sha256 7edb56f6 and the plosr_mf manifest 7896ae1e match the recorded hashes; fig5a has no recorded hash |
| R02 | P0 | Audit source authority and 15-stage figure remapping | Release/paper plans | Owner rulings mapped; no frozen figure silently changed. Partial 2026-11-21: `check_record_integrity.py --check` on b3a21045 + P-058: 255 of 255 pinned, 0 violations, 0 moves (no frozen record changed). The 15-stage remapping is blocked: the owner's workflow is not in the repo (open O-7, EX-27) |
| R03 | P0 | Reconcile current open/todo/issue records | Stacks | Every inherited open defect has one owner and closing check. DONE 2026-11-21: each of the 15 `open_stack.md` lines names an owner, a date, a closing check and a todo row that exists; each of the 16 issue-log Open entries maps to one row (P-040 R56, P-047 R57, P-052 R58, P-053 R59, P-055 R60, P-056 R25, P-057..P-065 R61..R69, P-066 R17). P-040 and P-057 to P-065 moved 2026-11-21 to `artifacts/archive/0.5.x/issues_closed/issues_closed_2026-11-21.md` (S16 ruled; pinned), which closes R56 and R61 to R69; Open now holds P-047, P-052, P-053, P-055, P-056, P-066 |
| R04 | P0 | Run matching-host broad baseline at integration SHA; probe the installed engine | Gate 2 | `run_test_gate.py broad --no-stamp` receipt on the S21 host with SHA, host and environment; CI on the same SHA; an import probe outside the checkout records the imported `jaxfne` path, version and commit |
| R05 | P0 | Resolve O-FIG05 frozen equivalence panel | Gate 2 | Owner ruling plus macOS gate pass; no unauthorized refreeze |
| R06 | P0 | Resolve O-NUM platform bitwise assertions; include the Izhikevich drift reported 2026-11-21 (unverified tier-2 report on candidate aa260a01) | Gate 2 | Owner-approved numerics criteria, negative mutant, Mac/CI pass; drift reproduced or refuted with dtype, precision, integration path and tolerance recorded |
| R07 | P0 | Prepare O-73 positional time-unit options; Hamm picks behavior (S04) | `open_stack.md` O-73 | Source comparison; after the ruling, old usage fails demonstrably and safe behavior is tested |
| R08 | P0 | Prepare O-74 noninteger bin-width transition options; Hamm picks (S04) | `open_stack.md` O-74 | After the ruling, compatibility/deprecation contract and regression test |
| R09 | P0 | Compare O-75 Fleiss kappa against 0.5.8; Hamm picks definition (S04) | `open_stack.md` O-75 | 0.5.8 comparison; after the ruling, documented version behavior and test |
| R10 | P0 | Turn six non-strict xfails into meaningful gates (O-76) | Gate 2 | Each a strict xfail or a passing test (O-76); a planted break fails each |
| R11 | P0 | Audit silent data-dependent skips (O-77) | Gate 2 | Clean data-free clone lists all skips explicitly |
| R12 | P1 | Inventory ephaptic operator routes and units | TASK-EPHAPTIC | LIF/Izh voltage, timestep, alpha dimension table |
| R13 | P1 | Separate lateral voltage coupling from HDP/RBD taxonomy | TASK-EPHAPTIC | Single canonical operator description, no mislabeled dynamics |
| R14 | P1 | Derive ephaptic discrete update and alpha dimensional contract | TASK-EPHAPTIC | Unit derivation and dt-refinement falsifier |
| R15 | P1 | Pin zero-coupling identity on LIF and Izhikevich | TASK-EPHAPTIC | Bitwise/declared tolerance identity and mutant |
| R16 | P1 | Verify spatial kernel normalization and symmetry | TASK-EPHAPTIC | Row normalization, edge cases, permutation test |
| R17 | P1 | Review and integrate the P-066 repair (S07 ruled 2026-11-19: realized z in mm, field proxy divides by `column_height_mm`; the repair stores the height in `model.params`) from `_temp_tfne-manuscript` 98d42ce4 | TASK-UNITS-CSD | Before/after table of affected rules, broad gate on the merge SHA, non-writer review ACCEPT, P-066 resolution line and move. Held 2026-11-21: the branch (base 3eb6a6a6, 8 commits, 22 files) adds `tests/test_height_callsites.py` and `tests/test_laminar_z_mm.py` (S16 cleared 2026-11-21, budget 2825) and repins `tests/canary_register.json` positions mean/std, the golden-snapshot params keys and the lane6b edge+position digest (split; edge digest 0bc6b793, position digest 94cb0202), which needs S23 |
| R18 | P1 | Test ephaptic lag across supported execution routes | TASK-EPHAPTIC | Same causal lag in scalar and batched routes |
| R19 | P1 | Compare exogenous-field drive to endogenous operator | T-94 | Matched kernel and imposed field; differential effect receipt |
| R20 | P1 | Prepare and validate the prereg for zero/shuffle/spatial-perturbation controls; run after Hamm signs (S17) | Paper stage 7 (provisional, O-7) | Prespecified effect and null estimands signed by Hamm; report negatives |
| R21 | P1 | Draft bounded mechanistic claims before the ephaptic figure; Hamm freezes them (S17) | Paper stage 7 (provisional, O-7) | Association vs causal attribution clearly bounded; Hamm's dated freeze |
| R22 | P1 | Determine whether ephaptic core addition is necessary | Release scope | Keep experimental if not required by frozen 0.5.10 acceptance |
| R23 | P1 | Verify source domain (state,input) and proxy scope | TASK-SOURCE-SCOPE | LFP/CSD/EEG/MEG claims separated from physical solutions |
| R24 | P1 | Check kernel LFP factorization and analytic oracles | T-90 | Tone, gain, shuffle and zero controls remain non-vacuous |
| R25 | P1 | Resolve P-056 inputs (S08), then run and close P-056 | T-90 | Population/weights/drive/threshold recorded by Hamm before the run; P-056 resolution line and move |
| R26 | P1 | Test discrete-map consistency and dt refinement | TASK-DMAP | Declared order, bounded error, planted break |
| R27 | P1 | Test H-state invariance vs clamp vs stability | TASK-HINV | Three distinct claims and matching checks |
| R28 | P1 | Build AT-10-N20 receipt and F2/new Fig 3 | 10A A1 | Figure regenerates from pinned atlas receipt |
| R29 | P1 | Implement NeuronalTensor flattening identity/schema figure | 10A A2 | Identity/topology/sign/position preserved in tests |
| R30 | P1 | Recheck geometry mutants and rotation invariance | 10A A3-A4 | All required planted breaks fail |
| R31 | P1 | Score CPU scaling on preregistered sizes | 10B B2 | Hash-pinned CPU receipts: neuron-steps/s against N, compile against run, vmap, float32/64 |
| R32 | P1 | Score GPU scaling with fixed accuracy and seeds | 10B B2 | Hash-pinned GPU receipts with the same four metrics, matched to CPU and prereg |
| R33 | P1 | Score Brian2/NEST/Jaxley baselines | 10B B3 | Matched model classes or documented incompatibility |
| R34 | P1 | Verify LIF edge GPU kernel parity | 10B B5 | Declared CPU/GPU tolerance met with receipt |
| R35 | P1 | Render S6 and main scaling figure from receipts | 10B | Axes, units, uncertainty and provenance audited |
| R36 | P1 | Prepare the ensemble augmentation prereg (seeds, metrics); execute after Hamm signs (S17) | 10C C1 | Frozen seeds and metrics signed by Hamm; independent test |
| R37 | P1 | Render S7 determinism/provenance panel | 10C C1 | Pinned receipts and replayable figure |
| R38 | P1 | Score Mejias 2016 G6 after prereg signature | 10D D3 | Owner-signed estimands and honest verdict |
| R39 | P1 | Complete AdEx/LIF/Model.tune batching tests | 10F F5 | Seed parity, shapes, gradients and negative tests |
| R40 | P1 | Dashboard P1 then P2 after text recovered (S11); the jitter reader on larger sets | 10F F6 | Each with its commit and receipt (frozen F6) |
| R41 | P1 | Run replication conformance audit | 10G G2 | 0 unlisted violations; exceptions owner-authorized |
| R42 | P2 | Inventory active docs, archived tutorials and navigation | Docs | Each public page classified canonical/archive/stale |
| R43 | P2 | Inspect jomission subproject tutorial candidates | Docs | Source SHA, license, API compatibility and scientific claim status |
| R44 | P2 | Select minimal tutorial migration candidates | Docs | Only executable transferable examples; no stale copying |
| R45 | P2 | Build 2-3 clean-room tutorials using current public API | Docs | Fresh install runs, deterministic figures, no jomission results promoted; file delta covered (S16) |
| R46 | P2 | Merge duplicate docs by canonical subject home | Docs | Link map, redirects where needed, no lost unique semantics |
| R47 | P2 | Compact README, quickstart and API prose | Docs | No unsupported claims, reduced repetition, original meaning preserved |
| R48 | P2 | Consolidate redundant code files only after dependency map | Maintainability | One bounded merger, API parity and tests; no broad rewrite |
| R49 | P2 | Audit figure captions, terminology and abbreviations | Docs | Independent interpretability at final print size |
| R50 | P2 | Regenerate docs/nav/notebooks and strict build | Docs | mkdocs strict, link checks, notebook smoke and dashboard fresh |
| R51 | P0 | Close package items with exact evidence or owner deferral; put the frozen Owner row (O-85..O-87) to Hamm (S20) | Gate 1 | Every 10A-10G item accepted or dated owner ruling; each S20 yes added as an item with its own row |
| R52 | P0 | Prepare candidate version, changelog and release body | Gate 3 | Version identity, changelog, and `release_body.py --tag <tag> --check` PASS on the release SHA |
| R53 | P0 | Run full RC gates, clean install and adversarial faults | Gate 2 | Same candidate SHA: every §4 release command exits 0, CI green, wheel/sdist hashes and mutants |
| R54 | P0 | Obtain independent release and scientific review | Gate 4 | Non-writer ACCEPT, findings dispositioned, review receipt committed under `artifacts/release/` |
| R55 | P0 | Owner release authorization and publication | Gate 5 | Only after owner go: tag/main/Release/PyPI and receipt |
| R56 | P0 | Drain P-040 (SCIENCE, Destexhe 2009 lane) | Issue log | A fix with resolution line and move, or Hamm's dated move |
| R57 | P0 | Drain P-047 (FRICTION, `Configuration.connections` selectors) | Issue log | Failing test first, fix, resolution line and move. Held 2026-11-21: P-047 records `{"cell_type": "all"}` as `compiled_no_matching_edges`, a design pinned by `test_connections_no_match_marked_compiled_no_edges`; a fix (refusal or wildcard) changes public behavior, so Hamm picks: keep and move as by-design, refuse, or wildcard |
| R58 | P0 | Drain P-052 (surrogate fitness; frozen Owner row O-85) | Issue log, S20 | Hamm's O-85 ruling, then resolution line and move |
| R59 | P0 | Drain P-053 (SCIENCE, PCL event scan) | Issue log | A fix with resolution line and move, or Hamm's dated move |
| R60 | P0 | Drain P-055 (SCIENCE, PCL event scan, attributed 2026-11-12) | Issue log | A fix with resolution line and move, or Hamm's dated move |
| R61 | P0 | Drain P-057 (BUG, Izhikevich `simulate()` zero-gain rule) | Issue log | Failing test first, fix, resolution line and move |
| R62 | P0 | Drain P-058 (BUG portability, PCL port tests) | Issue log | Failing test first, fix, resolution line and move. Fixed 2026-11-21 (commit 31a80b1b; resolution line in P-058; reviewer ACCEPT); moved 2026-11-21 to `issues_closed_2026-11-21.md` |
| R63 | P0 | Drain P-059 (BUG, `sdr_transform` shared `key_delta`) | Issue log | Failing test first, fix, resolution line and move |
| R64 | P0 | Drain P-060 (BUG port fidelity, `gsdr.py` shared `key_delta`) | Issue log | Failing test first, fix, resolution line and move |
| R65 | P0 | Drain P-061 (BUG test robustness, Windows) | Issue log | Failing test first, fix, resolution line and move |
| R66 | P0 | Drain P-062 (SCIENCE, Izhikevich HDP kernel route) | Issue log | A fix with resolution line and move, or Hamm's dated move |
| R67 | P0 | Drain P-063 (BUG, `simulate_batch` dense backend edges) | Issue log | Failing test first, fix, resolution line and move |
| R68 | P0 | Drain P-064 (BUG, `simulate_batch` HDP rule params) | Issue log | Failing test first, fix, resolution line and move |
| R69 | P0 | Drain P-065 (BUG, homeostasis kernel delay) | Issue log | Failing test first, fix, resolution line and move |
| R70 | P0 | Record the P-025 regression under a new P-ID and add the repo-root assert to `scripts/p056_resource_smoke.py` | Gate 2 | `pytest tests/test_script_repo_root_guard.py::test_scripts_and_artifacts_assert_repo_package` passes; entry resolved and moved |
| R71 | P1 | Jaxley/JAX compatibility on the supported matrix | `pyproject.toml` pins; not a frozen Gate 2 item (S22) | Measured 2026-11-21. Stated by the dispatcher and not in the log, so UNKNOWN to a repo reader under §4: host vwin-pc (Windows, Python 3.14.3), code of 3eb6a6a6 (`git diff 3eb6a6a6 30575b03 -- jaxfne tests pyproject.toml` is empty), and the 7 Jaxley test files (`test_jaxley_bridge`, `test_jaxley_emitter_bridge_e2e`, `test_emitters_hh`, `test_jaxley_optional_dependency`, `test_jaxley_trace_bridge`, `test_ei_jaxley_izhikevich_parity`, `test_hh_multicompartment`): jaxley 0.13.0, 0.14.0 and 0.14.1, each with jax 0.10.1 and 0.11.2 (the newest below the `<0.12` pin), 98 passed in each of the 6 cells, 0 failed, 0 skipped. Neither jax has native `jnp.clip(a_max=)`. jaxley 0.13.0 has one `a_max` call (`solver_gate.py:9`), expected to be covered by the shim (inferred: all 98 pass, but the shim-off control is not run); jaxley 0.14.0 and 0.14.1 have none, so the shim is not on their path. The reported `jnp.clip(a_max=)` failure was not reproduced in the matrix run; R72 (2026-11-21) reproduced it on jaxley 0.13.0 with no shim in force (the shim-off control: two test files run alone fail with that `TypeError`, pass once they install the shim). Open: jax 0.4.25-0.9.x (inside the pin, untested), Linux and macOS, and a shim-off negative control on jaxley 0.13.0. The jax 0.10.2 of T-MATCH is not in the matrix. Closes when the reporter's failing command is reproduced or refuted, the shim-off control fails on jaxley 0.13.0, and the cells S22 requires are run. Receipt: `D:/tmp/_temp_a10_matrix.log` on vwin-pc (not in the repo, so unverifiable by repo readers); when R71 runs, commit its rerun under `artifacts/release/` with a header naming SHA, host, Python and `pip freeze` per cell (S16) |
| R72 | P0 | Audit order-dependent tests, starting with `tests/test_score_port_parity_checks.py` (reported 2026-11-21, unverified) | Gate 2 | Each suspect run alone and in two different orders; cause named (shared state, RNG, cache, fixture); fixed with a test that fails under the bad order, or refuted. DONE 2026-11-21 on vwin-pc (Windows, Python 3.14.3), fix on `_temp_r70`: cause is process-wide shim state. `scripts/score_port_parity_checks.py:158` and `tests/test_hh_multicompartment.py:30` imported jaxley directly, so on jaxley 0.13.0 the `jnp.clip(a_max=)` shim was in force only if an earlier test had gone through `jaxfne.bridges.require_jaxley`. Run alone on jaxley 0.13.0 with jax 0.10.1 and with 0.11.2: score_port 2 failed (`test_recomputed_pass_is_error_within_tolerance`, `test_recompute_matches_within_rtol_1e_9`; `TypeError: clip() got an unexpected keyword argument 'a_max'` at `jaxley/solver_gate.py:9`), multicompartment 1 failed. Both sites now call `require_jaxley()`: alone on jaxley 0.13.0/jax 0.11.2, 7 passed and 17 passed; on jaxley 0.14.0/jax 0.10.1, the two files plus `test_minor_batch_refusals.py` 51 passed. On jaxley 0.14.x (no `a_max` call) the order dependence does not arise: alone, reversed, and after 88 related tests, 7 passed. The other raw-jaxley test files (`test_ei_jaxley_izhikevich_parity`, `test_ei_jaxley_hh_field_readout`) pass alone on 0.13.0 (4, 3 passed); `scripts/generate_doc_page_atlases.py:703` integrates through `JaxleyBridge` |

**Release invariant:** Do not call 0.5.10 READY until all prepublication requirements pass on the exact candidate SHA, issue-log Open section is empty and the release todo section (dev `todo_stack.md` `## 0.5.10 rows still open, carried with their checks`) drained under the release skill, full package/CI/documentation checks pass and owner has reviewed the remaining publication action. Never tag/publish without exact owner go.

## 9. Decision gates and risks

Not the canonical `artifacts/open_stack.md`; read it and the issue log first; do not duplicate their ownership or close existing rows from this plan.

| ID | Status | Problem / decision | Owner | Closes when | Todo |
|---|---|---|---|---|---|
| S01 | OPEN (proceeds under frozen score) | Owner interpretation of readiness 100 vs frozen 6 publication points | Hamm | Approve distinct prepublication indicator or retain frozen score | R01 |
| S02 | BLOCKED | O-FIG05 frozen panel mismatch; no silent re-freeze | Hamm | Approved remedy and exact-SHA macOS check | R05 |
| S03 | BLOCKED | O-NUM platform-specific bitwise criteria | Hamm | Approved criteria and positive/negative tests | R06 |
| S04 | BLOCKED | O-73 to O-75 API behavior and time units (public-API semantics; `open_stack.md` owner Hamm) | Hamm | Behavior ruling, then tests | R07-R09 |
| S05 | OPEN | O-76 and O-77 non-vacuous test and skip semantics | lead | Strict xfail or passing test per O-76; clean-clone skip list | R10-R11 |
| S06 | OPEN | Ephaptic operator classified as lateral voltage coupling, not proven physical field mechanism | lead proposes, Hamm rules claims | Units/lag/kernel/exogenous control and bounded claims | R12-R16, R18-R22 |
| S07 | RULED 2026-11-19 | P-066 depth units: realized z in mm, field proxy divides by `column_height_mm` (ruling recorded at `ISSUE_LOG.md:423` on `_temp_tfne-manuscript` 98d42ce4); the repair stores the height in `model.params` (commit d17812e7) | Hamm | R17 closes | R17 |
| S08 | BLOCKED | P-056 column inputs and threshold are unspecified | Hamm | Frozen population, weights, drive, bands and criterion | R25 |
| S09 | OPEN | Benchmarks require matching GPU/CPU host and baseline receipts | lead | Pinned comparable runs and controls | R31-R35 |
| S10 | BLOCKED | Mejias G6 estimands and prereg unsigned | Hamm | Signed P3 and defined score target | R38 |
| S11 | BLOCKED | Dashboard P1/P2 original owner text not recovered | Hamm | Recover or rule requirements before implementation | R40 |
| S12 | OPEN | Archived tutorials may be stale; jomission content is downstream science, not package authority | lead | Per-example API/license/provenance audit and clean-run | R42-R45 |
| S13 | OPEN | File merger may disrupt imports, generated mirrors or sealed artifacts | lead | Dependency map and bounded merge with equivalence tests | R46-R48 |
| S14 | OPEN | Release gates require exact candidate SHA and non-writer review | lead | Full RC receipt and review ACCEPT | R51-R54 |
| S15 | BLOCKED | Tag/main/Release/PyPI owner-reserved | Hamm | Explicit exact-SHA go after gates | R55 |
| S16 | RULED 2026-11-21 | Hamm 2026-11-21: "Raise to 2825" (`budget_files` 2813 -> 2825, +12, for the known 0.5.10 additions: one closed batch, R17's two tests, about four receipts, R45). First use: `issues_closed_2026-11-21.md` (P-040, P-057 to P-065), 2814 files. Before the ruling: dev was at `budget_files` 2813 (`scripts/tree_size_allowlist.json`); this revision nets 0, but R17 (+2 test files on 98d42ce4), each dated closed issue batch, committed receipts (R04, R53, R54, R71), R45 and other rows add files | Hamm | A raise to a stated number, or a rule that each addition is paid by a removal | R04, R17, R45, R53, R54, R56-R71 |
| S17 | BLOCKED | Prereg signatures for R20, R21, R36 (preparation authorized 2026-11-21; signing reserved) | Hamm | Signed preregs | R20, R21, R36 |
| S18 | BLOCKED | Publish and CI (Release & Scheduled) workflows disabled since 2026-11-14; Gate 2 CI and Gate 5 need them | Hamm | Hamm enables them or names another CI route | R53, R55 |
| S19 | RULED 2026-11-21 | Execution go given by Hamm 2026-11-21 for plan commit 5550cf075f96c9cfa4a5f9ec69cccb3c66b5fe58: "Go: merge and execute" (merge into dev, then ready tasks up to READY_FOR_OWNER_GO; no tag, main or PyPI). Before the go: Dev 3eb6a6a6 `todo_stack.md:44-47` T-PLAN-EXECUTE reads "Work Solmind packages continuously ... Blocked by: T-PLAN-INTEGRATE", which conflicts with this gate; T-PLAN-INTEGRATE sets it to "Blocked by: S19" and repoints dev's `state.md`, `open_stack.md` O-PLAN, `todo_stack.md` T-PLAN rows and the T-R28 pointer to `execution_plans_0.5.9-0.5.10.md:72` (carried by this branch's `todo_stack.md` edit) to this path, R01-R72 and S01-S23 | Hamm | Dated go naming the plan SHA | all |
| S20 | BLOCKED | Frozen Owner row: P-052 surrogate fitness (O-85), T-62 Fig 1D raw data (O-86), a named figure for Lee 2025 and Mazzoni 2015 (O-87) | Hamm | A ruling on each; each yes adds a checklist item | R51, R58 |
| S21 | BLOCKED | Matching host: T-MATCH says "Linux, jax 0.10.2" (dev 3eb6a6a6 `todo_stack.md:57`); `state.md:3` labels a Windows run (022da7a0) the last matching-host broad gate | Hamm | One definition | R04 |
| S22 | BLOCKED | Compatibility scope beyond the frozen checklist: R71's untested cells (jax 0.4.25-0.9.x inside the pin, Linux, macOS) are not a frozen Gate 2 item | Hamm | Ruling: test the cells, raise the pin floor, or accept the measured matrix for 0.5.10 | R71 |
| S23 | OPEN (fires per fix) | Repin approvals: an output-changing fix in R17 or R56-R70 that changes a frozen receipt, prereg or pinned record needs Hamm's approval for each repin (§3) | Hamm | Each repin approved with a date, or the fix redesigned to change no pinned record | R17, R56-R70 |

Planning constraints:
- Do not add ephaptic claims to the core merely because an operator exists. Require dimensional, numerical, causal-control and physical-interpretation checks; otherwise label it experimental and leave core release scope unchanged.
- A documentation merger must reduce duplicate normative content without removing API contracts, stable URLs, redirects, reproducible tutorials, scientific negatives or provenance. Count actual words/files and tests before/after; size reduction alone is not success.
- Jomission tutorials are candidate *examples* only; do not import downstream scientific conclusions into JaxFNE library guarantees. Verify license, SHA, dependencies, frozen scientific gates, and public API compatibility.
- The owner-approved 0.5.10 100-point checklist is unchanged. The 15-stage manuscript programme and its revised figure mapping are separate. Never exchange points between them. Crazy Sailor rows O-1..O-4 and O-6..O-9 are outside the 0.5.10 checklist and stay in the canonical `open_stack.md`.
- This branch is a temporary planning exception to the three-branch target; integrate the reviewed plan into dev and retire this branch after handoff. Do not alter main, tags, release or PyPI.

## Appendix A. Execution plans of 2026-10-31 (verbatim)

Folded here 2026-11-21 (A1) from `artifacts/etudes/meta/plans/execution_plans_0.5.9-0.5.10.md` at 3eb6a6a6, verbatim below this paragraph except that each of its 7 headings is demoted two levels; superseded for 0.5.10 routing by §8 and §9. Historical record only: none of its directives (lanes, roles, push or authorization lines) bind execution. History: `git log --follow` on the old path.

### Execution plans: open items and todos through 0.5.10

Written 2026-10-31 from `artifacts/open_stack.md`, `artifacts/todo_stack.md` (open rows), `release_plan_0.5.9-0.6.0.md`
(packages 9A to 9G and the 0.5.10 list) and the inventory of the pre-rewrite todo (`git show 6be045f7:artifacts/todo_stack.md`,
98 old items: 47 carried, 36 landed, 14 unmapped, 1 dropped). Four plans cover every open row and every open stack
item. Rows that wait on Hamm join no lane.

#### Rules for every lane

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

#### Owner-gated rows (no lane)

| Row | Open item | Decision | Source |
|---|---|---|---|
| T-O73 | O-73 | rule private vs shared sources for the S1 drive (G3) | `open_stack.md`, `SOURCE_P1.md` line 18 |
| T-62 | Fig 1D | ruled 2026-11-02: excluded for 0.5.9, moved to 0.5.10 (raw Onorato data on request from Martin Vinck, lead contact) | Onorato et al. 2025, data and code availability |
| T-46 close test | | close section 7 at 0 gaps or narrow to the 0.5.9 gaps | todo section 2 |
| Signatures | O-70 | Izhikevich P6, Potjans P3 v2, Vogels P3 v2, then Destexhe and GSDR after pilots, PCL R4 prereg | todo section 4 |
| P-052 | | surrogate fitness for training | `state.md` |
| T-O74 to T-O79 | O-74 to O-79 | keep retired selfcheck names as history, or edit the records | `open_stack.md` |
| T-54 | | awesome-jax PR, gated on the 0.5.10 release and 7 papers | todo section 4 |
| T-11, T-O10 | O-10 | external: jnwb path; jynx keeper takes or refuses each proposal (conference #29) | todo section 5 |
| T-RG | | external: komposer `run_guard.py stop` fallback | todo section 5 |

#### Plan 1: Cell Reports replication (CAMINOS, Tahvili et al. 2025)

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

#### Plan 2: Replication core and conformance (PCL, pilots, GSDR, Izhikevich)

Rows: T-3 (R2 to R5), T-22 and T-25 (after R2), T-O70 (pilots only), T-32 (one replication a day), the non-CAMINOS
part of T-30, O-69 (PCL R2). Release: 0.5.9 packages 9C and 9E.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 2a | orchestrator, WSL | T-3 R2: the port's replay of the trained network's files through the chained scan, against the C++ reference (`~/pcl_cpp`, `data/events/shapes.h5`); then R3 residual attribution | R2 receipt in `artifacts/etudes/papers/pcl/`; R4 waits on its signature |
| 2b | Muse code | T-32 current open rows, rewritten first (see Plan 4 4f): core homes for ISI ratio and CV, aperiodic fit, tuning fit (T-32d); GSDR fit on `simulate_hh_batch` with `gsdr_mcdp_transform`, test first | each lands with its test or record |
| 2c | Haiku worker, high | T-O70 pilots: Potjans 1% and 2% at in-degree held, Vogels estimands, GSDR and Destexhe pilots on seeds 0 to 3 (scoring seeds 9 to 13 reserved); pilot receipts only, no signature | pilot receipts exist; each signature stays with Hamm |
| 2d | Haiku worker, high | T-22, T-25: PCL B3 on the core intra-step operator; trained weights load as A' (`with_trained_weights`), after 2a | both run on the core with tests |
| 2e | Haiku worker, high | T-30 non-CAMINOS entries: Izhikevich net.m order through `construct`; GSDR HH dispatch; Destexhe estimators, LEDGER, RESULT; pcl_h and Q-learning phase records and `sys.path` imports | `python scripts/check_replications.py --check` lists no violation without Hamm's exception |

#### Plan 3: Protocols, paper items and dashboard

Rows: T-R28 (the executable parts, open stack O-72), T-45, T-46 (the 8 gaps; close test gated), T-10b, T-8, O-68
(local copies, rescue then delete, see Plan 4 4a). Release: 0.5.9 packages 9A, 9B and 9G.

| Lane | Who | Work | Done when |
|---|---|---|---|
| 3a | Haiku worker, medium | O-72: dashboard P1 then P2 (Hamm's ruling 2026-10-28); regenerate. BLOCKED 2026-10-31: the P1 and P2 text is not in the repo (`todo_stack.md:42`, `open_stack.md:16`, `state.md:12` cite it; `dashboard/ARCHITECTURE_v2.md` predates the rulings). Needs Hamm or the keeper to commit or paste the text | `python scripts/build_dashboard_data.py --check` passes |
| 3b | Haiku worker, high (test-first; launched on Haiku, not Muse, 2026-10-31). Result 2026-10-31: all 35 definition-map rows already map to a symbol with a test; the `PV[model=IZH]` parse fix (`jaxfne/tfne.py` ~1027, +4) is NOT integrated, because the key is a label only and does not change the realized model (repo rule: change the realized object or refuse). Owner ruling needed: refuse until decided (recommended) or override. S2 generator is absent from `scripts/`, so S2 cannot regenerate. T-45 stays open | T-45: map D17, D21, D26, D28, D31, D32, D34, D35 to public symbols; object specializer `PV[model=IZH]` (test first) | each row names a public symbol with a test; `supplement.py` regenerates S1, S2, S5 |
| 3c | orchestrator | P-052 beta scan against the central finite difference, output-changing, with a changelog fragment | scan receipt; reviewer reads the diff |
| 3d | Muse code | C4 Euclidean radius cutoff (rotation-invariant candidates; arbitrary-angle S3 test); C6 `geometry="laminar"` with Brian2 and Jaxley baselines declared | tests pass; the arbitrary-angle test present |
| 3e | Haiku worker, high | T-46: the 8 gaps in `programme/paper/reproducibility_audit.md` section 7 that the 0.5.9 scope holds | section 7 lists the remaining gaps, each named |
| 3f | Haiku worker, high | T-10b: jnwb proxy-claim wording gate, with its test | gate lands with its test |

T-8 runs after each push that changes dev: dashboard refresh (skill `jaxfne-dashboard`).

#### Plan 4: Fields, records, verification and the 0.5.10 backlog

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
