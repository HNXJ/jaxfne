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
| `main` | `cd2fbd3` | v0.0.22 — fast-forwarded from dev |
| `dev` | `e45e93b` (pushed) | v0.0.23 package validation smoke complete; 236 tests pass; 7 examples pass |

**Version:** `0.0.23`  
**Tests:** 236 passed, 0 failed  
**Working tree:** clean; v0.0.23 packaging smoke validated; awaiting hardening pass per external audit

---

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| `claude-sonnet` | v0.0.23 hardening: pytest full-suite reliability + manifest field grammar normalization | 2026-05-18 | in progress per external audit PARTIAL ACCEPT report |

---

## Completed work log

| Agent | Scope | Commit | Notes |
|---|---|---|---|
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
| v0.0.23 hardening: pytest full-suite reliability | `claude-sonnet` | `dev` | per external audit; timeout issue in full suite (targeted tests pass) |
| v0.0.23 hardening: manifest field grammar normalization | `claude-sonnet` | `dev` | per external audit; CSD_sign_convention, solver_residual_l2_relative, validation_report, asset_hashes |
| v0.0.23 hardening: package layout integration | `claude-sonnet` | `dev` | per external audit; missing modules (runtime.py, objectives.py, validation.py), missing LICENSE, missing examples (00, 01, 02, 03) |
| TestPyPI validation | `claude-sonnet` | `main` | after v0.0.23 hardening complete |
| PyPI public release v0.1.0 | `claude-sonnet` | `main` | final freeze after TestPyPI validation |
