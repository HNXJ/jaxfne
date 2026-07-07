# Agent Channel — Claude ↔ Cursor

Standing, git-tracked communication channel between the two agents working
this repo (Claude Code CLI and Cursor's Composer). There is no live/
programmatic bridge between us — no shared CLI, no MCP connection — so this
file is the handoff mechanism: whichever agent runs next reads it before
starting work, and appends before finishing.

**Stable path, do not move:** `artifacts/developer/AGENT_CHANNEL.md`

## Rules

1. **"Open / Needs Response" is the only section either agent may edit
   in-place.** Resolve an item by moving it into the Log (with the
   resolution), not by deleting it silently.
2. **The Log is append-only.** Never delete or rewrite a past entry, even a
   wrong one — correct it in a new entry that references the old one (same
   convention as `skills/FRICTIONS_STACK.md`).
3. **Sign every entry** with agent name + real date (`YYYY-MM-DD`, from the
   actual clock/session, not guessed).
4. **This is not the task tracker.** File-level review state, scores, and
   plan steps belong in `plans.json`/`progress.json`/`review.json` — this
   channel is for things that don't fit that schema: a heads-up, a question
   for the other agent, a "don't touch X right now," a correction to the
   other agent's prior claim, a note about shared-infrastructure changes
   (skills sync, doctrine file edits).
5. **Verify before trusting an entry.** Same rule as everywhere else in this
   repo's doctrine: an entry from the other agent is a claim, not a fact,
   until independently re-checked against real source/tests.

## Open / Needs Response

_(nothing pending)_

## Log

### 2026-07-06 — Claude
Channel created. Context: assessed Cursor's step-7 progress report (commits
`2743294`/`08a8f59`/`3621722` + one further uncommitted diff) by independently
re-verifying every claim against real source/tests rather than trusting the
report — all claims held up (test pass counts, backlog numbers, docstring
cross-references to F-016/F-017 all matched real state). Committed the
verified diff as `ee00fe7` (step 7: 68/80). No contradictions found; this is
a genuine record of a clean handoff, not a correction.

Going forward: I'll keep `~/.agents/skills/jaxfne-*` in sync via
`skills/SYNC_GLOBAL.sh` whenever I edit `skills/`, and will also update
`~/.cursor/rules/global-agent-doctrine.mdc` when I land something Cursor
should know about immediately (not just via `progress.json` osmosis).
