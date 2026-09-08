# jaxfne issue log — 40-day frozen-use period

Active issue log for the v0.4.17 frozen period (approximately 2026-08-19 to
2026-09-28). During this period the released core is frozen:

```
observe -> reproduce -> log
```

not

```
observe -> patch core.
```

Every entry records at least:

```text
date
type
area
observation
severity
minimal reproduction
expected behavior
actual behavior
evidence
possible future change
```

Types: `BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`.

Core bugs are accumulated here and are NOT repaired during the 40-day period.
At the end of the period, this log becomes the evidence for the next
development cycle. A severe issue may be proposed as an emergency patch but
requires separate authorization.

---

## Open issues

### I-001
- **date:** 2026-08-19
- **type:** DOC
- **area:** public surface / release evidence
- **observation:** `jaxfne/public_surface.py:105` contains a stale internal
  comment reporting "259 symbols at 0.4.13 baseline"; the frozen 0.4.13 set
  was actually 260.
- **severity:** MINOR (cosmetic comment; the live 191 public-export claim in
  docs + contract + `__all__` is correct)
- **minimal reproduction:** `grep -n "259" jaxfne/public_surface.py`
- **expected behavior:** comment matches frozen 0.4.13 count or is removed
- **actual behavior:** comment says 259
- **evidence:** final-hostile-review 5bc0400, check A/B; verified at commit
  0678557 that the frozen set was 260
- **possible future change:** correct comment at next code-touch (not during
  frozen period)

### I-002
- **date:** 2026-08-19
- **type:** DOC
- **area:** release receipt / provenance
- **observation:** `artifacts/release/v0_4_17_release_receipt.json` records
  `branches.dev/main = cd5738a` (seal-time truth). `dev` now points at
  5bc0400 (intentional docs-only delta); `final_candidate=cd5738a` unchanged.
- **severity:** MINOR (receipt is write-once by design; not a core
  contradiction)
- **minimal reproduction:** compare `git rev-parse dev` with
  `branches.dev` in the receipt
- **expected behavior:** receipt is immutable; branch snapshots are
  seal-time truth
- **actual behavior:** branch snapshot differs from current dev HEAD after a
  docs-only commit
- **evidence:** final-hostile-review check F (receipt branches snapshot note)
- **possible future change:** final release receipt should snapshot the exact
  release commit/tag after S2 completes

### I-003
- **date:** 2026-08-19
- **type:** DOC
- **area:** build artifacts / packaging
- **observation:** untracked `dist/` holds stale 0.3.42 wheels. The 0.4.17
  wheel/sdist referenced by the release receipt are build-gate outputs, not
  committed. A rebuilt 0.4.17 wheel from HEAD is clean and leak-free.
- **severity:** MINOR (no release blocker)
- **minimal reproduction:** `ls dist/`
- **expected behavior:** 0.4.17 wheel/sdist published from the release build
- **actual behavior:** stale 0.3.42 wheels in `dist/`; 0.4.17 built per gate
- **evidence:** final-hostile-review check C; clean-room build log
- **possible future change:** publish the 0.4.17 wheel/sdist in S2; consider
  cleaning or ignoring `dist/`

### I-004
- **date:** 2026-08-19
- **type:** DOC
- **area:** provenance / observed values
- **observation:** value `1.2165` is an observed `m_EI` mean-weight-trace from
  a 0.4.8-era checkpoint artifact
  (`artifacts/mcc3_10s_checkpoint/mcc3_10s_metrics.json`), not a documented
  `drive_gain` result. No public doc ties it to drive_gain.
- **severity:** MINOR (no contradiction, but flagged for precision)
- **minimal reproduction:** `jq '.weights.B.m_EI.mean_W_t' <  ...metrics.json`
- **expected behavior:** observed values remain traceable to their exact
  source run
- **actual behavior:** value exists only in the old checkpoint artifact
- **evidence:** final-hostile-review check 23 note
- **possible future change:** when quoting observed values in docs, cite the
  exact checkpoint artifact and version

### I-005
- **date:** 2026-08-20
- **type:** FRICTION
- **area:** CI gate / release hygiene
- **observation:** the pre-freeze CI-gate precondition (freeze only when
  main==dev and both main workflows are green on the exact candidate)
  exposed that CI (Fast) had been RED since 2026-08-18: the ruff 0.16.2 hard
  gate failed on the frozen candidate itself (`jaxfne/jdna/genome.py`
  F401 unused imports `warnings`/`RuntimeConfiguration` + E741 ambiguous
  `l` x5; `jaxfne/optim/__init__.py` E402 staged imports), and the
  sdist-hygiene test could not run on main because `build` was not in the
  dev extras (only the separate build job installed it). The lint breakage
  originated with JDNA commit 70ae496; candidate cd5738a therefore never
  had green remote gates.
- **severity:** MAJOR as release-process friction (caught before release;
  repaired in 53e9870/2845c92/0ff37e4 without core-semantic changes)
- **minimal reproduction:** `gh run list --branch dev` for 2026-08-18..19;
  `python -m ruff check jaxfne` under ruff==0.16.2
- **expected behavior:** every candidate that reaches a freeze/release gate
  has green remote CI on the exact commit
- **actual behavior:** candidates cd5738a and 2845c92 both had red main
  workflows; only 0ff37e4 (with `build` in dev extras) returned both green
- **evidence:** CI run 32285103287 (dev green), 32321936243 + 32321936278
  (main Fast + Release & Scheduled green at 0ff37e4); receipt v3
- **possible future change:** the pre-freeze CI-gate precondition is kept as
  a standing rule: freeze/release candidates must show green remote gates on
  the exact commit being frozen, not a predecessor

### I-006
- **date:** 2026-08-20
- **type:** FRICTION
- **area:** harness / code hygiene
- **observation:** the mechanical lint repair `l -> layer` in
  `jaxfne/jdna/genome.py` (`_check_realized_constraints`) introduced a
  semantic regression: the comprehension local shadowed the outer loop
  variable, turning `layer.name == layer.name` into an always-true
  self-comparison. CI caught it (`ValueError: layer 'L2': developed 100
  neurons, genome declares 250`), and renaming the local to `cand` restored
  behavior (50/50 JDNA tests pass). Renaming a variable can change semantics
  when comprehension/loop scopes interact.
- **severity:** MINOR (caught by the behavioral tests; the pre-freeze
  condition turned a lint fix into a test-validated repair)
- **minimal reproduction:** commit 53e9870 CI run
  (tests/test_jdna_pseudogenome.py failures); fix in 2845c92
- **expected behavior:** lint repairs are behavior-inert
- **actual behavior:** a rename shadowed an outer loop variable and altered
  constraint-checking semantics
- **evidence:** CI run 32320214843 (green at 0ff37e4); 50/50 JDNA tests pass
- **possible future change:** when renaming locals, prefer fresh names not
  used in the enclosing scope; keep the behavioral test suite as the
  authority over lint tooling

### I-007
- **date:** 2026-08-20
- **type:** DOC
- **area:** release receipt / provenance
- **observation:** receipt v2 recorded `final_candidate=cd5738a`, but the
  pre-freeze CI precondition moved the frozen core to `0ff37e4` (lint repair
  + sdist-gate fix). Receipt v3 supersedes v2 and records
  `core_candidate=0ff37e4`, with `release_candidate` left null until S2
  finalizes the docs/README identity. This is the receipt-branch-drift
  pattern of I-002, resolved at the release identity level.
- **severity:** MINOR (evidence hygiene; v3 is authoritative)
- **minimal reproduction:** `git rev-parse main` vs `core_candidate` in v3
- **expected behavior:** receipt records the exact frozen core and, after
  S2, the exact release identity; if they differ, both are recorded
- **actual behavior:** v2 stale; v3 records the true frozen core
- **evidence:** receipt v3 `pre_freeze_ci_gates` history table
- **possible future change:** S2 completion must update
  `release_candidate`/`tag` in the receipt (or supersede v3) before any
  PyPI/GitHub release action

### I-008
- **date:** 2026-09-07
- **type:** DOC
- **area:** freeze authorities / version identity
- **observation:** Freeze authorities still assert an active v0.4.17 core
  freeze and 40-day frozen-use window (~2026-08-19 to 2026-09-28) while tip
  activates v0.4.21 RC. ISSUE_LOG header names v0.4.17 frozen period;
  `jaxfne-frozen-use` asserts ΔC_core=0; `SCIENTIFIC_WORKBENCH_STATE.md`
  last indexed 2026-08-21 @ HEAD caec1f71 with `jaxfne==0.4.17 do not edit`;
  `jaxfne-seal` pins v0.4.17 final-100-goals authority. Tip:
  `pyproject.toml` / `mkdocs.yml` / `CITATION.cff` / `docs/citation.md` =
  0.4.21; tags through v0.4.20; no v0.4.21 tag at filing time.
- **severity:** block (critic); DOC + FACT — agent/human routing contradiction
- **minimal reproduction:** compare ISSUE_LOG header + frozen-use skill +
  workbench state to `grep version pyproject.toml` and `git tag -l 'v0.4.*'`
- **expected behavior:** freeze narrative and authorities align with active
  RC / post-freeze policy at tip without editing frozen core
- **actual behavior:** authorities pin 0.4.17-era freeze; tip declares 0.4.21
- **evidence:** jcritic pass @ b08f20f7 (gh-only, Gate0 not run);
  `artifacts/audit/evidence_audit_critic_surfaces_2026-09-07.md` F-001;
  scripts not executed
- **possible future change:** docs/critic-surface PR to refresh ISSUE_LOG
  header, workbench index, and skill authority narrative; do not patch
  `jaxfne/` core to resolve

### I-009
- **date:** 2026-09-07
- **type:** DOC
- **area:** agent onboarding / skills routing
- **observation:** `docs/for_ai_agents.md` routes agents to
  `artifacts/skills/catalog-glossary-jaxfne/SKILL.md` and
  `artifacts/skills/jaxfne-worker-context-router/SKILL.md`; both absent.
  Live skills: jaxfne-{audit,core,frozen-use,release,repo,science,seal}
  only (7 SKILL.md files).
- **severity:** should-fix (critic); DOC
- **minimal reproduction:** `ls artifacts/skills/*/SKILL.md` vs lines 15–16
  of `docs/for_ai_agents.md`
- **expected behavior:** start-here routes reference present skills or defer
  explicitly to live code + `artifacts/AGENTS.md`
- **actual behavior:** two primary routes target missing skill folders
- **evidence:** jcritic pass F-002; gh-only static read; scripts not executed
- **possible future change:** docs PR replacing stale skill paths with live set

### I-010
- **date:** 2026-09-07
- **type:** DOC
- **area:** contributor documentation
- **observation:** `docs/contributing.md` line 32 says update `skills/`;
  canonical agent skills live under `artifacts/skills/`; root `skills/` absent.
- **severity:** should-fix (critic); DOC
- **minimal reproduction:** `grep 'update \`skills/\`' docs/contributing.md`;
  `test -d skills` (fail) vs `test -d artifacts/skills` (pass)
- **expected behavior:** contributor path matches canonical `artifacts/skills/`
- **actual behavior:** path points to absent root `skills/`
- **evidence:** jcritic pass F-003; gh-only; scripts not executed
- **possible future change:** one-line fix in contributing guide

### I-011
- **date:** 2026-09-07
- **type:** DOC
- **area:** harness audit scripts / MkDocs inventory
- **observation:** `audit_81_docs_pages.py` and
  `audit_public_private_boundary.py` hardcode 81 MkDocs nav pages in
  docstrings and success banners; static nav leaf count from `mkdocs.yml` is
  88 at tip.
- **severity:** should-fix (critic); DOC + FACT
- **minimal reproduction:** grep `81` in the two scripts; count nav leaves
  from `mkdocs.yml`
- **expected behavior:** audit page count matches nav or is derived dynamically
- **actual behavior:** hardcoded 81 stale by 7 pages
- **evidence:** jcritic pass F-004; static nav count only; scripts not executed
- **possible future change:** harness PR to parameterize/rename; optional
  clone-backed re-run after fix

### I-012
- **date:** 2026-09-07
- **type:** DOC
- **area:** release/seal skill authorities
- **observation:** `artifacts/skills/jaxfne-release/SKILL.md` and
  `jaxfne-seal/SKILL.md` AUTHORITIES pin
  `artifacts/release/v0_4_17_release_receipt.json` while package version at
  tip is 0.4.21 RC and tags exist through v0.4.20.
- **severity:** should-fix (critic); DOC + FACT
- **minimal reproduction:** read AUTHORITIES sections of both skills vs
  `pyproject.toml` version
- **expected behavior:** skills cite matching receipt or document supersession
- **actual behavior:** both cite v0.4.17 receipt only
- **evidence:** jcritic pass F-005; gh-only; scripts not executed
- **possible future change:** skills/docs PR with receipt-chain note or new
  authority path; do not mutate frozen receipt JSON in this filing

### I-013
- **date:** 2026-09-07
- **type:** DOC
- **area:** scientific workbench routing index
- **observation:** `artifacts/science/SCIENTIFIC_WORKBENCH_STATE.md` last
  indexed 2026-08-21 (dev, HEAD caec1f71), ~17 days behind tip b08f20f7
  (2026-09-07); frozen instrument block still asserts `jaxfne==0.4.17`.
- **severity:** should-fix (critic); DOC + FACT
- **minimal reproduction:** read line 4 and frozen-instrument section vs
  `git log -1 --format='%H %ci' HEAD`
- **expected behavior:** routing index reflects current tip or explicit stale
  banner with successor
- **actual behavior:** index frozen at 2026-08-21 / 0.4.17 / caec1f71
- **evidence:** jcritic pass F-006; gh-only; scripts not executed
- **possible future change:** update workbench index in critic-surface PR
  (routing only; evidence receipts unchanged)

### I-014
- **date:** 2026-09-07
- **type:** DOC
- **area:** public/private boundary audit scope
- **observation:** `audit_public_private_boundary.py` scans MkDocs nav pages
  only. `docs/for_ai_agents.md` is `exclude_docs` in `mkdocs.yml` but contains
  `scratch/CURRENT_TASK.md` and worker-context handoff language. Full script
  PASS/FAIL unknown without clone execution. Sampled public doctrine/status
  pages remain consistent on H≠homeostasis and proxy≠calibrated.
- **severity:** unknown / nit (critic); DOC
- **minimal reproduction:** compare `mkdocs.yml` exclude_docs,
  `audit_public_private_boundary.py` nav iteration, and
  `docs/for_ai_agents.md` lines 47–48
- **expected behavior:** audit scope documented; agent-excluded docs either
  covered by separate check or explicitly out of scope
- **actual behavior:** nav-only scan; excluded agent doc not audited; outcome
  not verified in this pass
- **evidence:** jcritic pass F-007; gh-only; scripts not executed
- **possible future change:** document scope boundary; optional clone-backed
  run of boundary + docs-page audit scripts