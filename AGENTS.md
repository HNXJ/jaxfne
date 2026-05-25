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
| `main` | `c1e89f9` | v0.0.23 hardening complete |
| `dev` | `cced014` (pushed) | v0.1.0 practical OOP core freeze; hardening pass complete; CI workflow added; release scripts added |

**Version:** `0.1.0`  
**Tests:** 244 passed, 0 failed (verify with pytest before release)  
**Working tree:** clean after hardening commit; PyPI blocked — no account/token access  
**Next safe action:** Create `~/.pypirc` with TestPyPI token, then run `./scripts/upload_testpypi.sh`

---

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| (none) | v0.1.0 hardening pass complete; CI + release scripts + Colab docs added; awaiting PyPI credentials | 2026-05-18 | cleared — no active lock |

---

## Completed work log

| Agent | Scope | Commit | Notes |
|---|---|---|---|
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

## Next planned work

| Item | Assigned | Branch | Notes |
|---|---|---|---|
| Create `~/.pypirc` with TestPyPI token | user | — | Obtain API token from https://test.pypi.org/manage/account/token/; mode 600 |
| TestPyPI upload | `claude-sonnet` | `dev` | `./scripts/upload_testpypi.sh` — blocked until ~/.pypirc exists |
| Colab smoke from TestPyPI | user / `claude-sonnet` | — | Follow `docs/COLAB_SMOKE_V010.md` Cell 1 + Cell 3 |
| Merge dev → main (ff-only) | `claude-sonnet` | `main` | After TestPyPI + Colab smokes pass |
| Tag v0.1.0 and PyPI upload | `claude-sonnet` | `main` | `JAXFNE_CONFIRM_REAL_PYPI=1 ./scripts/upload_pypi.sh` — blocked until above complete |
