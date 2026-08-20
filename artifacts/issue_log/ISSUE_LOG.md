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