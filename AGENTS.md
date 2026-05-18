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
| `main` | `69d3197` | v0.0.19 — clean, pushed |
| `dev` | `1cbbc2b` | v0.0.14 — behind main, not yet fast-forwarded |

**Version:** `0.0.19`  
**Tests:** 178 passed, 0 failed  
**Working tree:** clean

---

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| *(none)* | — | — | idle |

---

## Completed work log

| Agent | Scope | Commit | Notes |
|---|---|---|---|
| `gemini-cli` | `docs/roadmaps/v0.0.18_longterm/` | `d7bf899` | 10 roadmap docs staged on dev-v0.0.18; captured at merge |
| `gemini-cli` | `README.md` hero snippet | `d7bf899` | run_receipt/compute_readout example; captured at merge |
| `claude-sonnet` | BETA audit + truth_mode fix | `07d2119` | blocking defect resolved; 3 new tests |
| `claude-sonnet` | README + .gitignore hardening | `ff385f2` | pre-merge hygiene pass |
| `claude-sonnet` | v0.0.18 roadmap commit | `d7bf899` | committed Gemini's staged work before branch merge |
| `claude-sonnet` | merge dev-v0.0.18 → main | `d7bf899` | ff-only; branch deleted |
| `claude-sonnet` | v0.0.19 docstring + API clarity | `69d3197` | canonical API marked; CHANGELOG.md added |

---

## Handoff protocol

When finishing a scope:
1. Update the **Active locks** table (clear your entry).
2. Add a row to **Completed work log**.
3. Include `AGENTS.md` in your final commit for that scope.

When starting a scope:
1. Run the session start checklist above.
2. Add a row to **Active locks** before making any edits.
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
| Push `main` v0.0.19 to remote | `claude-sonnet` | `main` | pending |
| Fast-forward `dev` to `main` | `claude-sonnet` | `dev` | after push |
| v0.1.0 bump + tag | `claude-sonnet` | `main` | after v0.0.19 push |
