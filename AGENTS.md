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
| `main` | `869694c` | v0.0.19 — last clean state |
| `dev` | `e24f4e5` (pushed) | v0.0.20 semantic hardening committed + pushed; v0.0.21 fidelity pass in progress |

**Version:** `0.0.20`  
**Tests:** 216 passed, 0 failed  
**Working tree:** clean baseline; v0.0.21 fidelity pass starting

---

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| `claude-sonnet` | v0.0.21 config/runtime/source fidelity (Tasks C–J) | 2026-05-18 | active |

---

## Completed work log

| Agent | Scope | Commit | Notes |
|---|---|---|---|
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
| Push `dev` v0.0.20 to remote | `gemini-cli` | `dev` | after commit |
| Fast-forward `main` to `dev` | `claude-sonnet` | `main` | after push |
| v0.0.21 — config fidelity | `gemini-cli` | `dev` | optional hardening |
| v0.1.0 bump + tag | `claude-sonnet` | `main` | final freeze |
