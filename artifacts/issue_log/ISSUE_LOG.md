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

## Issue history (drained 2026-09-20 — Open section empty; verdicts below)

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

### P-001
- **date:** 2026-09-20
- **type:** FRICTION
- **area:** scripts lint debt / release scope
- **observation:** repo-wide `python -m ruff check scripts/` reports ~180
  pre-existing errors (unused imports, E741 names, f-strings, semicolons) in
  legacy/research scripts. The release gate covers scoped paths only, so
  these do not block 0.4.25 — but they are noise for contributors and an
  unbounded surface for drive-by fixes.
- **severity:** MINOR (no gate impact)
- **minimal reproduction:** `python -m ruff check scripts/`
- **expected behavior:** scoped lint policy documented; legacy debt triaged
  per file (fix vs wont-fix with reason)
- **actual behavior:** 180 errors, no policy recorded
- **evidence:** ruff output 2026-09-20 (sweep session)
- **possible future change:** Batch F scoping decision in
  `artifacts/todo_stack.md` (touched paths + `jaxfne/` hot paths only);
  bulk legacy cleanup is 0.5.x. No drive-by lint renames (I-006).

### P-002
- **date:** 2026-09-20
- **type:** DOC
- **area:** docs landing / atlas grammar
- **observation:** `docs/index.md:46` still said "6-panel interactive atlas"
  after the B2 7-panel migration (found by vocab subagent pass).
- **severity:** MINOR (landing inconsistency; panels themselves correct)
- **minimal reproduction:** read `docs/index.md` line 46 vs line 53
- **expected behavior:** landing names the 7-panel grammar
- **actual behavior:** stale "6-panel" label
- **evidence:** subagent report ses_f4397cdadffel1X66fF6hjJsl5; fixed same
  turn, verified by `mkdocs build --strict`
- **possible future change:** none (resolved)

### P-003
- **date:** 2026-09-20
- **type:** FRICTION
- **area:** frozen etude evidence / reproducibility
- **observation:** frozen multiscale trajectories (V_m, Q hashes) do not
  reproduce under the tip tree OR under the protocol-pinned code (d5cf9a6 /
  593cfb3 worktrees); spikes + positions reproduce exactly everywhere, so
  the event structure is stable and only float details drift. JAX version
  at bundle time (2026-08-12) was never recorded — the likely drift source
  (XLA codegen), unverifiable post hoc.
- **severity:** MAJOR as evidence hygiene (frozen bundles lack the
  environment pin needed to re-derive them); MINOR for the 0.4.25 figures
  (claims re-verified, see below)
- **minimal reproduction:** `python scripts/rerun_etude_figures.py
  --etude multiscale` hash assertion (since replaced by two-tier check)
- **expected behavior:** frozen bundles re-derive bit-exactly, or record
  the full environment (JAX/lib versions) needed to do so
- **actual behavior:** V_m/Q hash drift; env unrecorded
- **evidence:** rerun hashes vs `cause_hashes` in
  `artifacts/etudes/multiscale_observation/metrics.json`
- **possible future change:** C5 publishes figures only with claims-tier
  verification (spike-exact quantities ==, float quantities within the
  protocol's own 1e-3 relative tolerance, geometry-exact r90 ==); future
  frozen protocols must record JAX/lib versions in the bundle (0.5.x
  provenance rule)

### P-004
- **date:** 2026-09-20
- **type:** FRICTION
- **area:** repo hygiene / unexplained modification
- **observation:** `scripts/generate_doc_page_atlases.py` carries
  formatter-style reflow (~450 diff lines) with no identified authoring
  action (no formatter invocation found in history; possibly an
  interrupted worker's editor). Content verified unaffected.
- **severity:** MINOR (verified harmless, see below)
- **minimal reproduction:** `git diff --cached --stat` on the file
- **expected behavior:** diffs attributable to a known action
- **actual behavior:** unattributed reflow alongside real edits
- **evidence:** `ruff check` + `ruff format --check` clean; `--list` runs;
  single-slug end-to-end rerun green; prior full regen bit-exact
- **possible future change:** resolved as harmless for 0.4.25; agent rule
  added (verify `git diff --stat` scale matches the intended change)

### P-006
- **date:** 2026-09-20
- **type:** DOC
- **area:** api reference accuracy
- **observation:** systematic signature drift between `docs/api` and live
  code found by delegated audit (143 checked): 13 divergences — omitted
  keyword-only optionals, one `**kwargs` fiction, one required-vs-optional
  inversion, one renamed-parameter pair; plus a fictional Metrics section
  (`energy_ratio` etc. keys do not exist) and a broken example
  (`signals.LFP` attribute does not exist; `diag["warnings"]` key does not
  exist) in `fields.md`/`validation.md`.
- **severity:** MAJOR for the broken example (TypeError/KeyError on copy),
  MINOR for omissions
- **minimal reproduction:** `inspect.signature` vs doc blocks; execute the
  conservation example headless
- **expected behavior:** documented signatures/names/keys match live code;
  examples execute
- **actual behavior:** as in observation
- **evidence:** worker audit ses_f3f62fb93ffen03bB3XxJ7Wr07 + independent
  re-verification 12/12 SOUND (one worker false claim reverted with proof)
- **possible future change:** fixed same turn (12 doc edits + example
  rewritten to live keys and executed); `audit_doc_code_integrity.py` gate
  added to prevent recurrence

### P-005
- **date:** 2026-09-20
- **type:** DOC
- **area:** release version choreography
- **observation:** the 0.4.25 version bump broke
  `test_published_version_matches_pyproject_when_not_mid_candidate` and
  `test_colab_md_version`: while the tree is mid-candidate (0.4.25
  unpublished), install/colab published-claims must stay on 0.4.24 with an
  explicit release-candidate label. CI Fast on main@0b929ab failed on exactly
  this (broad-gate step, both python jobs); local broad had run pre-bump so
  the break was invisible until CI.
- **severity:** MAJOR as release-blocker until fixed; MINOR in content
  (docs-only, no numerics)
- **minimal reproduction:** `pytest tests/test_docs_version_alignment.py`
  on the bumped tree (2 failed)
- **expected behavior:** candidate tree labels itself release-candidate;
  published-claims track PyPI until the release ships, then align
  post-release (established v0.4.24 pattern)
- **actual behavior:** bump moved published-claims early with no RC label
- **evidence:** CI push-run 35496154417 (broad-gate failure); local
  reproduction 2 failed → 13 passed after fix
- **possible future change:** fixed same turn (install.md RC label;
  colab.md published-claims restored); full post-bump broad rerun before
  re-release

### P-007
- **date:** 2026-09-20
- **type:** DOC + CODE (audit findings)
- **area:** deep package audit (structure/integrity/completeness)
- **observation:** delegated audit found real defects: `tune()` silently
  dropped mixed single/multi arguments; `validate_configuration` docstring
  mismatched behavior; `completion.py` rationale outdated post-TFNE2-07.
  Also confirmed unwired modules (units.py, pynwb_compat), entry
  fragmentation, and a broad UNTESTED-exact refusal tail; two worker
  claims (pool path, R4/R6 coverage) were false alarms corrected in review.
- **severity:** MINOR (fixed items) / DEFERRED (owner decisions)
- **minimal reproduction:** read `_model_tune.py:196-216`; inspect vs doc
  blocks; grep importers of units/pynwb
- **expected behavior:** no silent drops; docs match behavior; rationale
  current
- **actual behavior:** as in observation
- **evidence:** 3 new tests green; 59-test blast radius green; ruff +
  integrity gate green
- **possible future change:** fixed: tune warnings + tests, docstring,
  comment. Deferred to 0.5.x staging: units/pynwb wiring, emitter split,
  fragmentation, hdp_kwargs policy, silent-pass validators, refusal-tail
  tests, stale xfails/skips, compat aliases

---

## Open (0.5.1 seal, 2026-09-24)

### P-009
- **date:** 2026-09-24
- **type:** FRICTION
- **area:** timing assertion / field audit test
- **observation:** `test_source_generation_vs_projection_split` asserts
  projection wall < sim wall; fails intermittently on this machine
  (t_proj 0.81–1.13s vs t_sim 0.77–0.82s), including on pristine `fd46e1c`
  without any lane change — a pre-existing timing flake, not a 0.5.1
  regression.
- **severity:** MINOR (no output semantics involved; wall-time inequality only)
- **minimal reproduction:** `pytest tests/test_field01_audit.py::test_source_generation_vs_projection_split -q` repeated runs
- **expected behavior:** stable PASS or a warmup/margin discipline
- **actual behavior:** intermittent FAIL at ~1% margins
- **evidence:** 3/3 merged-tree + 1/1 pristine-baseline failures 2026-09-24; passes on other runs
- **possible future change:** owner decision — warm up the projection compile before timing, widen with a margin, or move to slow/release-only; do not weaken silently

### P-010
- **date:** 2026-09-24
- **type:** SCIENCE
- **area:** chunked continuation seed chain
- **observation:** chunked-vs-single run diverges 173 vs 174 spikes with
  identical seeds; suspected identical-seed-per-segment harness.
- **severity:** MAJOR as investigation (continuation identity is load-bearing); not a release blocker for 0.5.1 (chunked≡continuous required in 0.5.3)
- **minimal reproduction:** 0.5.1 matrix chunk cells vs single-shot (`bottlenecks_051.md` B8)
- **expected behavior:** chunked ≡ continuous bit-exact under the seed chain
- **actual behavior:** 1-spike divergence
- **evidence:** `artifacts/perf/bottlenecks_051.md` B8 (excluded from item 3, not bit-exact as run)
- **possible future change:** root-cause in 0.5.3 item 3 (seed-chain semantics)

### P-011
- **date:** 2026-09-25
- **type:** FRICTION
- **area:** Kaleido static export under xdist load
- **observation:** `test_vis_smoke[exporters.export_figures]` failed once in the
  broad gate with `RuntimeError: Couldn't close or kill browser subprocess`
  (choreographer/Kaleido browser teardown), on a tree that changed neither
  `jaxfne/vis/exporters.py` nor the test.
- **severity:** MINOR (teardown of the headless browser; no output semantics)
- **minimal reproduction:** broad gate (`-n auto`); alone it passes 3/3
- **expected behavior:** stable PASS
- **actual behavior:** intermittent FAIL under parallel load
- **evidence:** broad gate on `d0d7360` 2026-09-25 (1 failed / 4430 passed); isolated reruns 3/3 PASS
- **possible future change:** retry Kaleido teardown once, or serialize Kaleido tests (xdist group); open

### P-012
- **date:** 2026-09-25
- **type:** DEFECT (H11: tests depend on untracked local state)
- **area:** `tests/test_memory_brief.py::test_brief_paths_exist`,
  `tests/test_v033_two_neuron_ei.py::test_v033_all_json_files_parseable`
- **observation:** broad gate in a fresh `git worktree` at `b110493` failed both;
  the main tree at the same code passes both. memory.md names
  `artifacts/release_candidate/`, `artifacts/developer/` (gitignored) and
  `jaxfne/publication/` (untracked); the v033 test skips only when
  `outputs/v030_03_two_neuron_ei_multimodal` is absent, and in the fresh tree
  it existed but held no JSON.
- **severity:** MINOR (gate false-fails outside the author's checkout)
- **minimal reproduction:** `git worktree add <dir> b110493`, run the broad gate there
- **expected behavior:** PASS or explicit skip from a fresh clone
- **actual behavior:** 2 FAIL
- **evidence:** broad gate log 2026-09-25 (2 failed / 4422 passed in worktree); main tree rerun 2/2 PASS
- **possible future change:** brief-path test treats gitignored/untracked paths as local-only
  (or memory.md stops naming them); v033 test skips on an empty output dir; open

---

## Drain verdicts (0.4.25 sweep, 2026-09-20)

Entries above preserved verbatim. Open section is empty; every item closed
with the verdict and evidence below. New problems keep getting IDs here and
drain the same way.

| ID | Verdict | Evidence |
|---|---|---|
| I-001 | FIXED (comment + test docstring to live counts; assertions unchanged) | `d302105`, gate green |
| I-002 | SUPERSEDED (immutable receipts; v0.4.24 chain rules) | receipt chain + rollover draft |
| I-003 | SUPERSEDED (`dist/` absent; clean-room build at release) | tree state; E5 validators |
| I-004 | ADOPTED as Batch D acceptance (observed values cite exact artifacts) | todo_stack E6 |
| I-005 | ADOPTED as standing rule (green gates on exact sealed commit) | E4 gates; CI on release commit |
| I-006 | ADOPTED (behavioral suites gate lint renames; recorded in traps) | F3 + developer file |
| I-007 | SUPERSEDED (same as I-002) | receipt chain |
| I-008 | RULED (A9 freeze scope: numerics frozen, presentation + bit-exact ok) | todo_stack A9 |
| I-009 | VERIFIED-FIXED (stale routes absent from file) | grep 2026-09-20 |
| I-010 | FIXED (one-line skills path) | `d302105`, gate green |
| I-011 | SUPERSEDED (scripts absent from tree) | tree state |
| I-012 | SUPERSEDED (skills already cite `current_release_authorities.json`) | grep 2026-09-20 |
| I-013 | BANNERED (stale routing index points to successor; receipts untouched) | workbench header edit |
| I-014 | SUPERSEDED (boundary script absent; nav covered by orphan check) | tree state + E1 |
| P-001 | SCOPED (scripts/ legacy lint out of 0.4.25; bulk cleanup 0.5.x) | todo_stack F0 |
| P-002 | FIXED same turn (stale 6-panel label) | e65c035, strict build |
| P-003 | DOCUMENTED (float-drift finding; claims-tier figures published) | C5 reruns + manifests |
| P-004 | DOCUMENTED (unattributed reflow proven harmless; diff-stat rule added) | ruff + regen evidence |
| P-005 | FIXED (candidate discipline: RC label + published-claims restored + rollover reverted; full broad rerun green before re-release) | version-alignment gates |
| P-007 | FIXED-IN-PART (tune warnings + tests, docstring, comment; rest deferred to 0.5.x staging) | new tests + gates green |
| P-009 | FIXED (human decision 2026-09-24: warmup + margin). Both paths warmed; best-of-3 warmed walls with order-of-magnitude guard (t_proj < 4*t_sim): warmup alone was insufficient (steady-state sim runs faster than projection); the guard catches pathological slowdowns, not small drifts | `tests/test_field01_audit.py` |
| P-010 | FIXED (not a tolerance): plain path drew bulk noise, chained path per-step chain; plain now consumes the chain schedule. New stochastic stream human-approved 2026-09-24 | `865e74b`, `artifacts/programme/continuation_053.md` |
| P-008 | DOCUMENTED (two harness traps 2026-09-20): (1) test-file reflow recurred via edit tool (quote/blank-line normalization alongside intended deletions; tests green, harmless — P-004 pattern); (2) bare `git stash pop` popped a PRE-EXISTING unknown-ownership stash ("vis changes"), causing a merge conflict — repaired via `git reset --hard HEAD` (tree was sealed+pushed, nothing lost; stash entry preserved untouched). Rule: never bare-pop; pre-existing stashes are read-only | reset + import smoke |