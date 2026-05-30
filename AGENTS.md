# Agent Coordination

**Protocol version:** 1.0  
**Repo:** jaxfne  
**truth_mode:** truth_safe_unverified

---

## Branch ownership

| Agent | Owns | Never commits directly to |
|---|---|---|
| `claude-sonnet` | `main` — source edits, tests, version bumps, merges | `dev` |
| `gemini-cli` | `dev` — docs, roadmaps, large-context reads, bulk drafts | `main` |

Flow: `dev` → PR / fast-forward → `main` (Claude merges).

---

## Session start checklist (both agents)

```bash
cd /Users/hamednejat/workspace/main/jaxfne
git fetch origin
git log --oneline -3        # verify known HEAD matches below
cat AGENTS.md               # read active locks before touching anything
```

---

## Last known state

| Branch | SHA | Status |
|---|---|---|
| `main` | `release-candidate` | v0.3.21 release candidate, Etude No. 1 and template aligned |
| `dev` | `sync-source` | staging branch; merge to main only after validation |

**Version:** `0.3.21` release candidate; PyPI upload pending final release authorization  
**Tests:** 1787 passed, 64 skipped, 4 xfailed  
**Working tree:** clean  

**All phases complete:**
- Phase 0: Release-control scripts hardened
- Phase 1: Public builders API
- Phase 2: Configuration API (10 domains)
- Phase 3: Test repairs
- Phase 4: Suite No. 2 docs expansion
- Phase 5: Visualization API (9 functions)
- Phase 5b: Multi-laminar AGSDR (3 analysis functions + tutorial)

**Next safe action:** Run release validation gates, tag v0.3.21, and publish only after clean receipts

---

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| (none) | — | — | No active locks |

## Completed work log

| Agent | Scope | Commit | Notes |
|---|---|---|---|
| `gemini-cli` | v0.3.20: JAX recompilation guards & cache checks | `current` | validation.py: added CompilationRegistry and make_recompilation_guard; core.py: wrapped jit compilation loops; added test_v0320_recompilation_guards.py; repaired sync_release_metadata, audit_notebooks_and_assets, projection row-norm tests, and strengthened dtype tests. |
|---|---|---|---|
| `gemini-cli` | v0.3.19: sliced finite-difference spatial stencils with boundary fallbacks | `dba883d` | fields.py: added compiler loop-fusion docs; implemented robust stencil fallbacks for n_contacts <= 3; added parity and boundary checks; 1681/1681 pass. |
|---|---|---|---|
| `gemini-cli` | v0.3.18: distributed sharding mesh stubs and replicated param fallbacks | `b8c7fd3` | sharding_utils.py: make_population_mesh / make_candidate_sharding / make_replicated_sharding / get_sharding_context; 14 new tests; 1679/1679 pass. |
|---|---|---|---|
| `gemini-cli` | v0.3.17: precision-matching dtype invariants for AGSDR loops | `5867e0f` | _wdtype / _wdtype_outer patch; 12 new dtype tests; 1665/1665 pass; blueprint with conflicting NamedTuple AGSDRState redefinition rejected. |
|---|---|---|---|
| `gemini-cli` | v0.3.11: Stage B Visualizations, Jaxley Bridge, and Biophysical Alignment | `b06a14c` | Implemented jtfne.vis namespace, FigureResult, JaxleyBridge, E2E smoke tests, and arkhipov_allen_tfne_alignment.md. All tests passed. |
| `claude-sonnet` | v0.3.10: dev→main merge (Suite No. 3 correction + Suite No. 1 AGSDR naming) | `175b75b` | Merged dev into main after full validation: 1318/1318 tests pass, docs hygiene 210/210, mkdocs strict pass. main SHA: 175b75b. |
| `gemini-cli` | v0.3.10: Suite No. 3 noisy absolute-power scaling correction | `3a28bc1` | Revised Suite No. 3 to noisy AI regime, aggregate vs density-normalized proxy power, control verification table. 1318/1318 tests pass; MkDocs strict pass. |
| `gemini-cli` | v0.3.10: EEG/MEG/EMM proxy bundle tutorial | `b89e516` | Implemented tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb, docs/tutorials/09_v0310_eeg_meg_emm_proxy_bundle.md, targeted test suite, and doc hygiene check alignment. 1284/1284 tests pass. |
| `gemini-cli` | v0.3.9: Suite No. 3 low-frequency scaling proxy-readout tutorial | `b0cf731` | Implemented and integrated tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb, docs/tutorials/08_jaxfne_suite_no_3_low_frequency_scaling.md, 5 publication-ready figures, targeted test suite, and doc hygiene check alignment. 1276/1276 tests pass. |
| `gemini-cli` | v0.3.5: Suite No. 1 docs positive style repair | `659668e` | Refined Suite No. 1 docs, notebook cells, and index files to conform to positive-scope guidelines. 1037/1037 tests pass; MkDocs build passes cleanly. |
| `gemini-cli` | v0.3.4: Suite No. 2 spectrolaminar motif and explicit facade | `a2e231b` | Completed tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb, docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md, 13 publication-ready figures, and implemented the explicit chainable Configuration facade method suite. 1037/1037 tests pass. |
| `gemini-cli` | v0.2.31: Suite No. 2 compact API facade (simulate & vis.spectrolaminar) | `5851dde` | Implemented functional jtfne.simulate and high-fidelity jtfne.vis.spectrolaminar; 1036/1036 tests passing; clean working tree. |
| `claude-sonnet` | v0.1.0 post-RC hardening: CI workflow, release scripts, Colab docs, RELEASE_CHECKLIST | (pending commit) | scripts/release_rehearsal.sh, upload_testpypi.sh, upload_pypi.sh; .github/workflows/ci.yml; docs/COLAB_SMOKE_V010.md; docs/RELEASE_CHECKLIST.md; README patches; no code change; PyPI blocked by missing credentials |
| `claude-sonnet` | v0.0.23 package hardening (LICENSE, example naming 00-06, pytest reliability) | `77485e7` | MIT LICENSE added; examples renamed; 236 tests pass in 36s; 7/7 examples pass; per external audit |
| `claude-sonnet` | v0.0.23 packaging validation smoke (wheel/sdist build, twine check, fresh venv install, version bump) | `e45e93b` | 236 tests pass, 7/7 examples pass; pushed to origin/dev; awaiting hardening per audit |
| `gemini-cli` | v0.0.22 docs/packaging/Colab hardening | `27495a4` | Added Colab scaffold, packaging docs, version bump to 0.0.22 |
| `claude-sonnet` | v0.0.22 version alignment fix (pyproject.toml sync, test assertion updates) | `cd2fbd3` | Fixed misalignment from Gemini's v0.0.22 bump; fast-forwarded main |
| `claude-sonnet` | v0.0.21 config/runtime/source fidelity (Tasks C–J validation, test suite, doc updates) | `29bbe0a` | 236 tests pass, 7 examples pass; pushed to origin/dev |
| `claude-sonnet` | v0.0.20 semantic hardening (receipts/readouts/manifest/probes/sim validation) | `e24f4e5` | 216 tests pass; pushed to origin |
| `gemini-cli` | `docs/roadmaps/v0.0.18_longterm/` | `d7bf899` | 10 roadmap docs staged on dev-v0.0.18; captured at merge |
| `gemini-cli` | `README.md` hero snippet | `d7bf899` | run_receipt/compute_readout example; captured at merge |
| `claude-sonnet` | BETA audit + truth_mode fix | `07d2119` | blocking defect resolved; 3 new tests |
| `claude-sonnet` | README + .gitignore hardening | `ff385f2` | pre-merge hygiene pass |
| `claude-sonnet` | v0.0.18 roadmap commit | `d7bf899` | committed Gemini's staged work before branch merge |
| `claude-sonnet` | merge dev-v0.0.18 → main | `d7bf899` | ff-only; branch deleted |
| `claude-sonnet` | v0.0.19 docstring + API clarity | `69d3197` | canonical API marked; CHANGELOG.md added |
| `claude-sonnet` | v0.0.19 docstring + API clarity + v0.0.20 prep | `69d3197` | canonical API marked; CHANGELOG added; premature 0.1.0 commit reverted |

---

## Release-Control Doctrine

This section is the authoritative release-control state machine for jaxfne.
Scripts in `scripts/release/` implement these gates. Read before any release action.

### Release Target Identity

For every release, define exactly one `intended_release_sha` — the peeled commit SHA
of the annotated tag (not the tag object SHA). The machine-readable current state
is declared in the block below (only this block is parsed by scripts):

<!-- RELEASE_STATE_BEGIN -->
release_freeze: false
intended_release_sha: none
<!-- RELEASE_STATE_END -->

Before tag repair, GitHub Release edits, TestPyPI, or PyPI upload, all of the
following must be true:

```
origin/main == intended_release_sha
ci_head_sha == intended_release_sha
ci_conclusion == success
working_tree_clean == true
tag_peeled_sha == intended_release_sha  (once tag exists)
```

Run `python scripts/release/reconcile_release_target.py --version X.Y.Z --target-sha <SHA>`
to verify. Do not rely on a clean working tree as a substitute for CI success.

### Release Freeze Mode

When release CI is running or has passed for a candidate SHA, set:

```
release_freeze: true
intended_release_sha: <40-char commit SHA>
```

**During freeze, blocked:** root cleanup commits, docs formatting, receipt
relocation, unrelated feature/bug commits, branch hygiene.

**During freeze, allowed:** read-only diagnostics, CI status checks, tag repair
after explicit authorization, TestPyPI/PyPI after explicit authorization.

Freeze is lifted when: (a) release is published to PyPI, or (b) release is
explicitly cancelled by the user, or (c) user explicitly changes the target SHA.

Run `python scripts/release/assert_release_freeze.py` to verify freeze state.

### Remote Mutation Protocol

Classify every remote mutation before executing:

| Class | Examples | Authorization required |
|---|---|---|
| Read-only | git status, ls-remote, gh run view | Always allowed |
| Local-only | build, test, inspect dist/ | Allowed unless risky |
| Remote branch | git push origin main | Explicit task scope |
| Tag mutation | create, delete, recreate tag | Explicit authorization |
| Distribution | TestPyPI/PyPI upload | Explicit authorization |
| GitHub Release | create, edit, publish | Explicit authorization |

Do not combine mutation classes in one step. Each is a separate authorization gate.

### CI Monitoring Discipline

For long CI jobs, emit **one terminal-state receipt** when CI reaches:
`success`, `failure`, `cancelled`, or `timed_out`.

Do NOT emit repeated "still running" or heartbeat messages. One notification
on terminal state only.

Terminal receipt must include: run URL, status + conclusion, headSha, job matrix,
origin/main SHA, working tree status, no unauthorized mutations, next safe action.

### Annotated Tag Audit

For annotated tags, always report both:

```bash
git ls-remote origin refs/tags/vX.Y.Z          # tag object SHA
git ls-remote origin "refs/tags/vX.Y.Z^{}"     # peeled tag commit SHA
```

Use the **peeled tag commit SHA** as the release identity. Never use the tag
object SHA as the release commit target.

Run `bash scripts/release/print_tag_receipt.sh vX.Y.Z` to verify.

### Root Hygiene Preflight

Root cleanup must happen **before** release freeze, not during it.

Required before marking `release_freeze: true`:
- `git ls-files . | wc -l` — confirm only tracked files
- `find . -maxdepth 1 -type f` — confirm no root clutter
- `git status --short` — confirm clean

---

## Handoff protocol

When finishing a scope:
1. Update the **Active locks** table (clear your entry).
2. Add a row to **Completed work log**.
3. Include `AGENTS.md` in your final commit for that scope.

When starting a scope:
1. Run the session start checklist above.
2. Add a row to **Active locks before making any edits.
3. If another agent has a lock on your target file/dir — read only, do not write.

---

## Conflict resolution

If two agents edited the same file independently (diverged state):
- The agent that pushed last wins on remote.
- The other agent must `git fetch`, inspect the diff, and rebase or cherry-pick.
- Do not force-push `main`.
- Escalate to user if rebase is non-trivial.

---

## v0.3.21 Release Gates (Current)

| Item | Status | Command / Notes |
|---|---|---|
| Full pytest suite | ⏳ | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/` |
| MkDocs strict build | ⏳ | `mkdocs build --strict` |
| Package build | ⏳ | `python -m build` — creates `dist/jaxfne-0.3.21.tar.gz` and wheel |
| Twine check | ⏳ | `python -m twine check dist/*` |
| TestPyPI upload | ⏳ | `python -m twine upload --repository testpypi dist/*` (requires ~/.pypirc) |
| TestPyPI smoke | ⏳ | `pip install --index-url https://test.pypi.org/simple/ jaxfne==0.3.21 && python -c "import jaxfne; print(jaxfne.__version__)"` |
| PyPI upload | ⏳ | `python -m twine upload dist/*` (production release) |
| Tag v0.3.21 | ⏳ | `git tag -a v0.3.21 -m "Release v0.3.21: Etude No. 1, template, hygiene rules"` |
| GitHub Release | ⏳ | Create release from tag on GitHub |

All gates ready on `main` branch. No blockers from documentation or packaging.
