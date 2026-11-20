# Solmind 0.5.10 autonomous execution contract

Status: PROPOSED execution protocol; owner authorization for planning only. Canonical execution authority remains JaxFNE's `artifacts/AGENTS.md`, `artifacts/control/skills/jaxfne-release/SKILL.md`, frozen release plan and current project stacks. This contract routes the 55 tasks in `solmind_0510_todo.md` and 15 decisions in `solmind_0510_open.md`; it does not replace or weaken any of them.

## 0. Objective, baseline, boundaries

Objective: make 0.5.10 releasable at the smallest justified scope and verified cost, without changing frozen scientific verdicts or owner-reserved publication authority. The frozen checklist in `artifacts/etudes/meta/plans/release_plan_0.5.9-0.6.0.md` governs points (70 package + 30 gate). Recorded 43/100 is historical (2026-11-19), not an acceptance result at the current HEAD. Under the frozen contract, Gate 5's six points are earned only after actual publication. Before publication, report `PREPUBLICATION READY/NOT READY/UNKNOWN` separately, never silently remap 94 to 100. Do not claim READY with a nonempty issue-log Open section, missing mandatory check, or unknown required host.

Baseline of this plan: `jaxfne@dev@10c417adced394e799b922e78315a0be1f8a82dd`; at execution, fetch the live `dev` SHA and rebase/reconcile the plan, not the scientific evidence. `_temp_solmind_0510_plans` is planning only, not an authorized implementation or release branch. Target integration `dev`; `main`, tags, GitHub Release, PyPI and publication require Hamm's explicit exact-SHA go. Never force-push or rewrite pinned records. Temporary branch retirement follows independent review and preservation of unique evidence.

## 1. One entrypoint, one accountable lead

The lead reads Jarness `main` `rules/start.md`, JaxFNE `artifacts/AGENTS.md`, `artifacts/context.md`, `artifacts/state.md`, `artifacts/open_stack.md`, `artifacts/todo_stack.md`, issue-log Open section, the frozen 0.5.10 release checklist, and the two Solmind plan files. Check `git status`, `main/dev` SHAs, active worktrees and outstanding edits. Git and native gates outrank stale state prose. Read relevant scientific specs only when their task becomes ready. No routine full-repository rereads.

Lead owns scientific tradeoffs, statistics, authority, integration, reviewer assignment, and score. Delegate bounded tasks to inexpensive workers; at most three concurrent lanes INCLUDING reviewers. One writer per file set. A non-writer reviewer examines every consequential change and reruns acceptance on the integrated SHA. Reviewers do not edit writer files.

A worker packet MUST state: task ID; base branch and full SHA; read/write paths; immutable paths; dependencies; expected minutes; objective and falsifier; estimand, units, controls and stopping rule when scientific; exact commands; output/receipt paths; success and stop criteria. A worker reports changed files, tested SHA, command outputs, failed checks, residual risks and one closing verdict. Missing receipt = UNKNOWN.

## 2. Dependency graph and lanes

| Package | Task IDs | Prerequisite | Can run alongside | Completion |
|---|---|---|---|---|
| A: baseline/authority | R01-R04, R07-R11 | live state and frozen release plan | read-only design for C | scores and defects reconciled; baseline gate receipt |
| B: owner-gated numerical repair | R05-R06 | Hamm ruling on S02/S03 | independent C/D tasks | exact-SHA Mac/CI checks pass |
| C: ephaptic scientific audit | R12-R27 | existing operator and source/field contracts; S07/S08 only for affected execution | A, D, E | mathematical/unit tests, adversarial controls, bounded claim verdict; defer physical claims when unproven |
| D: release packages | R28-R41 | preregs, equipment, owner decisions for affected rows | C, E | each frozen checklist row has its own receipt |
| E: docs/maintenance | R42-R50 | live public API inventory; R46-R50 after R42-R45 selection | C, D | measured duplication reduction; API parity, clean tutorial and docs gates |
| F: release closure | R51-R55 | A-E mandatory release scope; no open blocker | none for final candidate freeze | Gates 1-4 exact-SHA, independent ACCEPT; R55 owner-reserved |

Within packages, do not interpret task numbering as a strict total order. Work on ready tasks; block dependents only. If a package is not required by the frozen 0.5.10 checklist (for example an experimental ephaptic paper claim or a major docs merger), it MUST NOT delay release unless a reproducible safety/correctness defect or owner ruling makes it mandatory. Put discretionary work after the release or in a separately authorized manuscript lineage. Preserve the 15-stage paper programme as distinct from the 0.5.10 score.

## 3. Autonomy, decisions, escalation

Proceed autonomously on reversible, authorized, in-scope changes with checkable acceptance. If a fix takes <=15 minutes, make the minimal repair and test; otherwise record one todo with its closing check and continue another ready task. Two failed approaches require revising the approach, not more retries. Never relax a threshold, change a frozen preregistration, reinterpret a failed replication, repin an immutable file, or expand the release scope to force PASS.

Owner gates: S01 score semantics (can proceed under frozen score without decision); S02 fig05 re-freeze; S03 numeric acceptance semantics; S07 spatial units; S08 P-056 inputs; S10 Mejias signature; S11 missing dashboard P1/P2 ruling; S15 release publication. Ask ONE ranked question only when its answer blocks the next material action and no other authorized ready work exists. Never assume silence is approval. An optional task blocked by an owner decision stays BLOCKED with a precise packet; other lanes continue.

A consequential scientific uncertainty (e.g., ephaptic physical interpretation) yields a restricted claim or explicit UNKNOWN, not an invented result. An unresolved required gate blocks release. An open issue is never silently removed, downgraded, or relabeled 'moved' without a dated owner ruling and an executable new closing check.

## 4. Mandatory checks and evidence

At the task level, use the smallest relevant native test plus an adversarial negative or mutant where it would otherwise pass vacuously. At integration, run the relevant native gate. For code changes, `python scripts/run_test_gate.py broad` on the required matching host before push; for docs, `python scripts/run_test_gate.py docs`; `python scripts/run_test_gate.py report` is audit-only and is not a broad test. For release, also run `python scripts/check_replications.py --check`, `python scripts/check_tree_size.py --check`, `python scripts/check_record_integrity.py --check`, `mkdocs build --strict`, figure/dashboard freshness checks, clean wheel/sdist build and install smoke, and candidate-specific release validation in the release skill. Use exact commands available on the current SHA; if a named command does not exist, stop that check and correct the packet, not invent a passing substitute.

For each PASS, record repository, branch, full SHA, host/runtime, command, exit status, test counts, sample size, artifact hash or path, reviewer, and timestamp. Distinguish IMPLEMENTED, TESTED and independently VERIFIED. Missing runtime/host/receipt is UNKNOWN, not PASS. All negative and null results are retained. Do not retune criteria after outcomes. Review the actual diff, not only an agent's summary.

The lead recomputes the frozen checklist after each package, recording each scored item and its receipt. Score can rise only for verified named checks; a merged commit alone earns no points. Track `score/100`, `coverage/100`, `prepublication gate verdict`, and outstanding owner gates. Avoid another shadow scoring system.

## 5. Scope-specific acceptance

**Ephaptic:** Check whether the implemented voltage-neighbor operator is dimensionally and numerically consistent across LIF/Izhikevich, with correct lag, spatial normalization, alpha=0 identity, dt refinement, zero/shuffled/perturbed controls, and imposed-field comparator. Distinguish phenomenological lateral voltage coupling from physically calibrated extracellular fields. If no physical calibration, do not advertise a validated ephaptic mechanism. Do not move it into RBD/HDP semantics without the required state/weight dynamics. Keep experimental components outside core release obligations unless the existing release acceptance requires them.

**Merger and prose:** Inventory duplicate modules, import graphs, generated mirrors, stable public paths, tests, notebooks and sealed artifacts. Choose one highest-return merger; require unchanged API behavior and checked imports, preserved history, meaningful reduction in duplication, and independent review. A large merger is not a target by itself. Measure baseline/final files, words, duplicate statements and tests. Do not delete archived evidence merely to lower counts.

**Tutorials:** Classify active/archived JaxFNE tutorials; inspect jomission examples at pinned SHA. Verify provenance, license, scientific claim status, dependencies, API compatibility and exact executable cells. Prefer 2-3 minimal clean-install tutorials over mass migration. Jomission is downstream scientific usage, not a validation authority for JaxFNE. No copied result can become a package guarantee.

## 6. Continuation and stopping

Each integration cycle: check remote SHA and local status -> choose highest-priority unblocked task -> run bounded packet -> inspect diff -> run relevant tests -> non-writer review -> exact-path commit/push only under existing authority -> update canonical todo/open/state as applicable -> recompute release score -> repeat. The Solmind planning files are design history; avoid a parallel live tracker after integration. Use git history and receipts for completed items, remove closed todo entries. If a task changes the plan, commit its planning revision to the planning branch under the standing instruction before proceeding.

Terminal states:
- `READY_FOR_OWNER_GO`: all mandatory prepublication release checks pass on ONE immutable candidate SHA, issue-log Open section empty, independent review ACCEPT, no required UNKNOWN; no tag/main/release yet.
- `PUBLISHED`: Hamm authorized exact SHA and release skill confirms tag, main, GitHub Release, PyPI, hashes and clean install; frozen checklist 100/100 only when all checks truly pass.
- `BLOCKED`: no authorized ready task remains. Report exact unresolved owner questions, impacted checklist points, best safe next action and receipts. Never report completion from static inspection or partial tests.

Handback (<=15 lines plus links): branch@SHA; verified score and coverage; ready/not-ready/unknown; tasks closed; files/receipts; gate results; remaining blockers with owner and one decision question if needed; next ready task. No repetitive status narratives.

## 7. Plan audit before autonomous dispatch

Acceptance of this plan as an agent goal requires a reviewer to check: (a) all 55 tasks and 15 risks have a routing home; (b) each mandatory release item 10A-G and Gates 1-5 has an owner and acceptance evidence; (c) no frozen acceptance changed; (d) every owner-only action is gated; (e) three-lane cap and reviewer independence hold; (f) simulated missing-host, failed-gate, conflicting-authority and stale-SHA scenarios produce BLOCKED/UNKNOWN rather than false PASS. Until this audit is executed, the plan is SPECIFIED, not independently VERIFIED.
