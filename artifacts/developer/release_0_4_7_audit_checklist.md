# jaxfne 0.4.7 Release-Readiness Audit Checklist

**Purpose:** an independent, falsifiable checklist for a third-party auditor to run
against the `dev` branch (or whichever ref is the release candidate), without needing
to trust any prior AI-agent self-report — including the scorecard this same document
ships with. Every rule below has a concrete command or observation attached; "looks
right" is not a pass, only a command's actual output is.

**How to use:** run each rule's command yourself against a clean checkout of the
target ref. Record PASS/FAIL/PARTIAL and the raw output. Where this document lists a
"Claude's 2026-07-14 finding," treat it as a lead to re-verify, not a given — this
repo's own history has at least one confirmed case of an agent's self-score (93/100)
being wrong against a direct human read (real: 82/100), so a from-scratch check
matters more than a diff against the prior claim.

**Scope note:** this checklist is intentionally broader than any one prior internal
scorecard (e.g. `plans.json`'s jaxley-relative 9-category scorecard) — it adds
sections (branch protection, dependency vulnerability scanning, lint gate, PRP
internal consistency) that the internal passes did not previously cover.

---

## A. Correctness & Test Coverage

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| A1 | Full fast test suite passes with zero failures/errors on a clean checkout | `python3 -m pytest tests/ -q -m "not slow"`, then `grep -c "^FAILED\|^ERROR"` on the **raw, untruncated** output (not a `tail`-truncated view — a truncated view can hide a real failure count) | grep returns `0`; exit code `0` |
| A2 | Slow/marked-slow tests also pass (not just excluded) | `python3 -m pytest tests/ -q -m "slow"` | `0` failed/error |
| A3 | Full CI matrix (not just fast CI) is green on the candidate SHA | `gh run list --workflow="CI (Release & Scheduled)" --limit 1 --json headSha,conclusion` matched against the exact candidate SHA (`git rev-parse <ref>`, never hand-typed) | `conclusion: success` on the exact SHA, not a stale prior run |
| A4 | Notebook execution nightly gate has run against the **current** HEAD, not a stale prior commit | `gh run list --workflow=notebook_execution.yml --limit 1 --json headSha,conclusion,createdAt`; compare `headSha` to `git rev-parse <ref>` | `headSha` matches current ref, `conclusion: success` |
| A5 | No test is silently skipped/xfailed in a way that masks a real gap | `pytest --collect-only -q -m "not slow"` then cross-check every `xfail`/`skip` reason string against current code — is the skip reason still true? | Every skip/xfail has a dated, still-accurate reason, not a stale placeholder |
| A6 | Public API snapshot/compatibility tests pass | `python3 -m pytest tests/test_public_api_snapshot_v034.py tests/test_public_api_compatibility.py -q` | all pass |
| A7 | No test file imports from a path outside the package/test tree that would make results environment-dependent (e.g. a hardcoded absolute path to a different checkout) | `grep -rn "workspace/computational\|/Users/" tests/ jaxfne/` | zero hits outside test fixtures that are explicitly path-agnostic |

**Claude's 2026-07-14 finding (verify independently):** A1 run to completion against
`dev`@`aed6c10`: `2661 passed, 74 skipped, 34 deselected, 4 xfailed, 0 failed/error`
in 324.56s (`grep -c "^FAILED\|^ERROR"` on raw output → `0`). Re-run it yourself
against the exact ref you're auditing — don't reuse this number without re-verifying,
this branch moves fast (this exact number is already superseded by any commit after
`aed6c10`).

---

## B. Build & Packaging

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| B1 | sdist + wheel build cleanly from a clean checkout | `python3 -m build --outdir /tmp/audit_build` | exit `0`, both `.tar.gz` and `.whl` produced |
| B2 | Wheel installs into a fresh venv and imports without error | `python3 -m venv /tmp/av && /tmp/av/bin/pip install /tmp/audit_build/*.whl && /tmp/av/bin/python -c "import jaxfne; print(jaxfne.__version__)"` | import succeeds, version matches `pyproject.toml` |
| B3 | `pyproject.toml` version matches the package's `__version__` and any release tag being cut | `grep '^version' pyproject.toml`; `python3 -c "import jaxfne; print(jaxfne.__version__)"`; `git tag --points-at <ref>` | all three agree |
| B4 | Dependency lower bounds are still installable together (no unresolvable conflict) | `pip install --dry-run jax>=0.4.25 jaxlib>=0.4.25 numpy>=1.24 scipy>=1.10` in a clean env | resolves without conflict |
| B5 | No unpinned/floating dependency that could silently break a future install (`jax`/`jaxlib` version drift is the highest-risk one for this package) | read `pyproject.toml`'s `dependencies` list directly | lower bounds present for every dependency that matters to correctness |
| B6 | Root directory structure matches the repo's own documented "root freeze" policy — no undeclared new top-level files/dirs | `git ls-files --full-name \| grep -v '/'` (files) and `git ls-files \| awk -F/ 'NF>1{print $1}' \| sort -u` (dirs), diffed against `AGENTS.md`'s frozen-root list | no additions beyond the documented approved-exception list |
| B7 | `py.typed` marker present if the package claims type-hint support | `find jaxfne -name py.typed` | present |

**Claude's 2026-07-14 finding:** B1 passed (`Successfully built jaxfne-0.4.6.tar.gz and
jaxfne-0.4.6-py3-none-any.whl`); B7 passed (`jaxfne/py.typed` exists); B6 passed on
direct inspection (root: 8 tracked files, 10 tracked top-level dirs, matches
`AGENTS.md`'s frozen list, no unexpected additions found by `git status --short`).
B2/B4/B5 not independently re-verified this session — re-run them.

---

## C. Documentation

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| C1 | `mkdocs build --strict` succeeds (broken links/refs fail the build in strict mode) | `python3 -m mkdocs build --strict` | exit `0` |
| C2 | Every page reachable from disk is wired into `mkdocs.yml`'s nav (no orphan pages) | compare `find docs -name "*.md"` against `mkdocs.yml`'s `nav:` tree | zero orphans, or each orphan is a deliberate, documented exception |
| C3 | No leaked agent-facing/internal doctrine language in public-facing docs (README, `docs/**/*.md`) | `python3 scripts/audit_public_docs_language.py --check` | `pass: true`, `doc_violations: []` |
| C4 | README's quickstart/install instructions actually work from a clean environment | manually run README's own quickstart snippet in a fresh venv | runs without error, produces the described output |
| C5 | Docstrings exist for every public (non-underscore) top-level symbol in `jaxfne/__init__.py`'s `__all__` | write a short script: for each name in `jaxfne.__all__`, check `getattr(jaxfne, name).__doc__` is non-empty | 100% coverage, or a documented, deliberate list of exceptions |
| C6 | CHANGELOG is current — the latest released/tagged version has a dated entry | `git tag --sort=-creatordate \| head -1`; check `docs/changelog.md` has a matching `## v<version>` heading | present, dated |
| C7 | No dead/broken external links in docs (e.g. to `github.com/HNXJ/jaxfne` paths, readthedocs, PyPI) | a link-checker pass (e.g. `python3 -m mkdocs build --strict` catches internal ones; use `linkchecker` or similar for external ones — not run this session) | zero broken links |

**Claude's 2026-07-14 finding:** C1 passed (`mkdocs build --strict` exit 0, one
informational "not in nav" note for `_generated/version.md`, not a failure). C3 passed
(`pass: true`, 0 violations). C2's single orphan page (`_generated/version.md`) should
be checked by the auditor — is it deliberately excluded from nav, or an oversight? C4,
C5, C6, C7 not independently re-verified this session.

---

## D. API Stability & Versioning

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| D1 | Public API snapshot (`artifacts/public_api_before.json` or equivalent) matches the live package | `python3 scripts/snapshot_public_api.py --check` (or equivalent diff mode — confirm the exact script/flag exists first) | no drift, or drift is intentional and the snapshot was regenerated in the same commit |
| D2 | Every name added to `jaxfne/__init__.py`'s `__all__` in this release cycle has a corresponding test and doc entry | diff `__all__` between the last tagged release and `<ref>`; check each new name for a test + docs hit | 100% coverage of new public names |
| D3 | No breaking signature change to a function/class that shipped in the last PyPI release (0.4.5) without a documented deprecation path | for each public symbol present in both 0.4.5 and `<ref>`, diff `inspect.signature(...)` | zero unexplained breaking changes, or each is in `docs/changelog.md` under a "Breaking" heading |
| D4 | Deprecated symbols emit `DeprecationWarning` and are still functional (not silently removed) | grep for `DeprecationWarning` call sites; smoke-test each deprecated function once | warning fires, function still works |

**Claude's 2026-07-14 finding:** D1's underlying test (`tests/test_public_api_snapshot_v034.py`,
`tests/test_public_api_compatibility.py`) passed (5/5) this session — but confirm the
auditor's own `<ref>` still matches; this repo has a documented prior incident
(2026-07-08, `AGENT_CHANNEL.md`) where a commit added public names without
regenerating the snapshot and shipped CI-red for a period before being caught. D2–D4
not independently re-verified this session.

---

## E. Security & Secrets

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| E1 | No secrets (API keys, tokens, credentials) committed anywhere in history, not just the current tree | run a real secret scanner across full history: `gitleaks detect --source . -v` or `trufflehog git file://. --since-commit <root>` | zero findings, or each finding is a confirmed false positive with a documented reason |
| E2 | No dangerous code execution patterns (`eval`, `exec`, unguarded `pickle.load` on untrusted input, `subprocess(..., shell=True)`, `os.system`) in the package or scripts | `grep -rn "eval(\|exec(\|pickle.load(\|subprocess.*shell=True\|os.system(" --include="*.py" jaxfne/ scripts/` | zero hits, or each hit is justified (e.g. controlled internal-only input) |
| E3 | Dependencies have no known CVEs at the pinned/allowed version ranges | `pip-audit` (or `safety check`) against a resolved environment | zero unpatched high/critical CVEs |
| E4 | GitHub Security Policy is configured (`SECURITY.md` or repo security-policy setting) | `gh repo view <owner>/<repo> --json isSecurityPolicyEnabled` | `true` |
| E5 | `main` branch has protection (required status checks before merge, no direct unreviewed pushes) | `gh api repos/<owner>/<repo>/branches/main/protection` | a protection ruleset exists (even a minimal one — required CI check is the floor) |
| E6 | No overly broad file permissions or accidentally-committed local/personal files (`.DS_Store`, IDE configs, personal `.env`) tracked in git | `git ls-files \| grep -iE "\.DS_Store|\.env$|\.idea/|\.vscode/settings"` | zero hits |

**Claude's 2026-07-14 finding:** E1 only shallow-checked this session (a grep for a
few common key-format patterns, not a real history-wide scanner — **run E1 for real**,
this is the weakest-verified row in this whole checklist). E2 passed (zero hits). E3
not checked (no scanner installed locally this session). **E4 and E5 both FAILED**
this session: `isSecurityPolicyEnabled: false`, and `gh api .../branches/main/protection`
returned `404 Branch not protected` — `main` currently accepts direct pushes with no
required CI check and has no security policy configured. E6 passed (only gitignored
local artifacts found at root, nothing tracked).

---

## F. Scientific / Claim Discipline (jaxfne-specific truth gates)

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| F1 | No code path silently escalates a truth/claim-status field (e.g. flips `physical_amplitude_calibrated`/`claim_level` to a stronger claim without an explicit calibration bridge) | `grep -rn "claim_level\s*=\|physical_amplitude_calibrated\s*=" jaxfne/` and manually verify each assignment site only ever sets a value it's authorized to, never escalates | every write site is either a constructor default or `clamp_truth_gate_metadata()` — no bare escalating assignment |
| F2 | Manifests/receipts are write-once (no silent overwrite of a saved scientific artifact) | read `jaxfne/io.py`'s `save_receipt`/manifest-saving path; attempt to overwrite an existing receipt file without `overwrite=True` | raises, does not silently overwrite |
| F3 | Public docs never claim "validated"/"physical"/"proven"/"mechanism" for a result that lacks a manifest+hash receipt | `python3 scripts/audit_public_docs_language.py --check` (already covers negative-claim language; extend/manually spot-check for over-claiming positive language too, which the current script may not catch) | no overclaiming language without a backing receipt |
| F4 | Test suite includes at least one exact-numeric-receipt style regression test per major scientific claim/preset (e.g. `DEFAULT_HDP`'s stationarity) | grep test files for the preset name, confirm an assertion pins a specific numeric range, not just "runs without crashing" | numeric assertions present, not just smoke tests |
| F5 | `FRICTIONS_STACK.md` and `plans.json`/`progress.json` do not contain live, unresolved contradictions about the same claim | cross-read `FRICTIONS_STACK.md`'s "Resolved" section against `plans.json`'s `items[]` for the same F-ID; flag any case where one says resolved and the other still lists it open/blocked | zero contradictions, or each is explicitly reconciled |

**Claude's 2026-07-14 finding:** F5 **FAILED** — a live, real contradiction was found
this session: `FRICTIONS_STACK.md`'s F-017 entry states *"F-019 (formula redesign) is
resolved by this fix; not opened as a separate row"*, but `plans.json`'s `items[]`
still carries `F-019` with `status: "blocked"` and a `blocked_reason` saying it needs
explicit go-ahead before scoping starts. These two statements are inconsistent about
whether F-019 is resolved or still pending — not fixed in this session (deliberately
left for the auditor / a dedicated pass, since resolving it requires deciding which
statement is actually true, not just editing JSON to match one side). F1–F4 not
independently re-verified this session.

---

## G. CI/CD & Release Engineering

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| G1 | `origin/main` == `origin/dev` (or the intended release branch has no unmerged commits sitting only on `dev`) | `git log origin/main..origin/dev --oneline` and the reverse | both directions consistent with the intended release state, no surprise divergence |
| G2 | Release workflow (`publish.yml`/`release_ci.yml`) has run successfully at least once end-to-end on a non-production target (TestPyPI) before any real PyPI publish | `gh run list --workflow=publish.yml` history | at least one successful TestPyPI dry run on record |
| G3 | Every remote-mutation workflow (tag push, GitHub Release, PyPI publish) requires explicit human trigger, not automatic on every merge to `main` | read `.github/workflows/publish.yml`/`release_ci.yml` triggers | `workflow_dispatch` or a tag-push-only trigger, not `push: branches: [main]` auto-publish |
| G4 | Full CI matrix covers all classifiers claimed in `pyproject.toml` (Python 3.10/3.11/3.12) | read `.github/workflows/ci.yml`'s matrix; cross-check against `pyproject.toml`'s `classifiers` | matrix covers every claimed Python version |
| G5 | Coverage reporting is wired and current (if a coverage badge is claimed in README) | check README's coverage badge source vs an actual fresh `pytest --cov` run | badge reflects current, not stale, data |

**Claude's 2026-07-14 finding:** G1 passed at last check (`dev`==`main`==`aed6c10` as
of this session, pending the final CI confirmation for that exact commit — auditor
should re-verify at time of audit, this drifts fast). G2–G5 not independently
re-verified this session.

---

## H. Governance & Community Health

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| H1 | `CONTRIBUTING.md` reachable and accurate (build/test/PR instructions actually work) | follow it verbatim in a fresh clone | every step succeeds as written |
| H2 | A Code of Conduct exists, or its absence is a documented, deliberate project decision (not just silently missing) | `find . -iname "CODE_OF_CONDUCT*"` | present, or the absence is explained in a discoverable doc |
| H3 | Issue/PR templates exist to guide external contributors | `find .github -iname "*template*"` | present, or explicitly deferred with a reason |
| H4 | LICENSE file matches the license declared in `pyproject.toml`'s classifiers | `head LICENSE` vs `grep "License ::" pyproject.toml` | consistent (both MIT here) |
| H5 | `CITATION.cff` is valid and, if a DOI is claimed anywhere, the DOI actually resolves | `cffconvert --validate -i CITATION.cff` (if installed) and manually resolve any DOI link | valid CFF, any DOI live |

**Claude's 2026-07-14 finding:** H2 **FAILED as "silently missing"** —
`CODE_OF_CONDUCT.md` was removed from the repo on 2026-07-12 per `AGENT_CHANNEL.md`,
and while that log entry documents the removal, there's no discoverable *public*-facing
note (README, CONTRIBUTING) telling an external contributor why there's no CoC — worth
either restoring a minimal CoC or adding a one-line public explanation. H3 **FAILED**
— `find .github -iname "*template*"` returned zero files, no issue/PR templates
exist. H4 passed (MIT/MIT consistent). H1, H5 not independently re-verified this
session.

---

## I. Code Quality & Maintainability

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| I1 | Lint is clean, or lint is wired into CI as an enforced gate | `python3 -m ruff check jaxfne/` **and** `grep -n "ruff" .github/workflows/*.yml` | either zero lint errors, or lint is explicitly not a release gate by documented decision |
| I2 | No `F821` (undefined-name) lint findings — these can indicate a real latent bug, not just style | `python3 -m ruff check jaxfne/ --select F821` | zero, or each is confirmed a false positive (e.g. a string-quoted forward-reference type hint) |
| I3 | No duplicated hot-path logic beyond a documented, deliberate exception | grep for known duplication patterns (e.g. the Izhikevich step equation across `emitters.py`'s scan closures) | zero undocumented duplication, or each instance is tracked as a backlog item |
| I4 | Type-checking (mypy/pyright) is configured and passes, or its absence is a stated decision | `grep -n "mypy\|pyright" pyproject.toml`; if present, run it | configured+passing, or explicitly out of scope |
| I5 | No orphaned/dead code (functions never called from any test, script, notebook, or other package code) | a call-graph/dead-code tool (e.g. `vulture jaxfne/`) | zero high-confidence dead code, or each flagged item is a deliberately-kept public API surface |

**Claude's 2026-07-14 finding:** I1 **FAILED as an enforced gate** — `ruff check
jaxfne/` reports **509 errors** (353 auto-fixable with `--fix`), and `ruff` is
**not** wired into any CI workflow (`grep -n "ruff" .github/workflows/*.yml` returned
zero hits) — lint currently has zero release-blocking power. I2: **16 `F821`
undefined-name findings**, spot-checked and most look like string-quoted forward-
reference type hints missing a `TYPE_CHECKING`-guarded import (e.g. `"jax.Array"` in
`jaxfne/analysis/metrics.py`, `Optional` unimported in `jaxfne/fields/proxy.py:1190`)
— low-severity individually but unaudited; the auditor should confirm none of the 16
are load-bearing at runtime, not just assume from this note. I3 known, tracked
(`emitters.py`'s Izhikevich step duplicated across 11 scan closures — see
`progress.json`'s `jaxfne/emitters.py` entry `tbd`, deliberately deferred, not
hidden). I4 **FAILED** — no mypy/pyright config found in `pyproject.toml`. I5 not
checked (no dead-code tool run this session).

---

## J. Reproducibility & Performance

| # | Rule | Verify with | Pass criterion |
|---|------|-------------|-----------------|
| J1 | A documented example reproduces its own claimed output bit-for-bit (or within a stated tolerance) with a fixed seed | pick 2-3 tutorials/scripts that claim specific numeric results, re-run with the same seed | output matches within the stated tolerance |
| J2 | JIT recompilation guard (`N_compile <= 1`) holds for the documented hot paths | run the package's own `jax-jit-pmap-performance-guard`-style check / `validation.py`'s recompilation warnings on a representative simulate() call | no `Re-compilation guard alert` in normal (non-adversarial) usage |
| J3 | Performance claims in docs (e.g. scaling benchmarks) are backed by a receipt, not asserted from memory | find the benchmark script + its last-run output/receipt | receipt exists, is dated, and the claimed numbers match it |
| J4 | Large-scale runs referenced in docs (e.g. "100k neurons", "1M neurons") have an actual receipt on record, not just an aspirational plan-item | cross-check `plans.json`'s `not_started`/`proposed` items against any doc claiming the scale is already supported | no doc claims a scale that's still `not_started` in the backlog |

**Claude's 2026-07-14 finding:** not independently re-verified this session (out of
scope for the compute time available) — flagging J4 specifically as worth a close
look, since `plans.json` currently lists `test-1000n-fast-laminar-lfp-csd-hdp` and
`cortical-column-scaleup-ladder-100-to-1M` as `not_started`; confirm no doc page
overclaims these scales as already validated.

---

## Summary of this session's direct findings (starting point for the auditor, not a substitute)

**Passed, directly verified 2026-07-14:**
A1 (2661 passed, 0 failed/error), A6, B1, B6, B7, C1, C3, E2, E6, G1 (as of `aed6c10`), H4.

**Failed / real gaps found, not fixed this session (deliberately left for the audit,
not silently patched):**
- **E4/E5** — no `SECURITY.md`/security policy, no branch protection on `main`.
- **F5** — live contradiction between `FRICTIONS_STACK.md` (F-019 "resolved") and
  `plans.json` (F-019 `status: blocked`).
- **H2/H3** — no CODE_OF_CONDUCT (removed, not publicly explained), no issue/PR
  templates.
- **I1/I4** — 509 unenforced ruff lint errors (ruff not wired into CI), no
  mypy/pyright configured.
- Backlog item **`hdp-k-w-ctrl-default-runaway-gap`** (new, `plans.json`,
  2026-07-14): `DEFAULT_HDP`'s `K_w_ctrl=0.0` permits unbounded weight drift on
  long/custom runs outside the specific presets that have been verified — not a bug
  in any existing shipped behavior, but a real gap for anyone building new HDP
  networks directly on the documented default.

**Not independently re-verified this session** (every row not listed as passed or
failed above) — these are exactly the rows most worth an independent auditor's fresh
eyes, since they were carried forward from this repo's own prior self-reports rather
than freshly checked.
