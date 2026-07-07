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

### 2026-07-07 — Claude: step 7 handoff, pre-authorized PRP loop
Cursor: `progress.json`'s new top-level key **`step7_handoff_worklist_2026_07_07`**
is a full, prioritized worklist to close plans.json step 7 (68/80 -> target 80)
without needing another round-trip through me or Hamm:

- **P1** (13 items): the real `status="open"`/`"deferred"` entries — the direct
  path to step 7's DoD.
- **P2** (3 items): quick doc-hygiene batch (stale skills-count, one-version-lag
  banners).
- **P3**: the standing 43-entry tbd/tbi review sweep — opportunistic, after P1/P2.

Each item carries its own verification note; re-check everything yourself before
acting (I verified paths/counts 2026-07-07, but that's a claim, not a fact, same
rule as always). Exit criteria + commit/CI/merge sequence are spelled out in the
worklist's `exit_criteria` field — this queue is pre-authorized to run start to
finish. The one hard boundary: **do not touch tag/TestPyPI/PyPI/GitHub Release**
(release-mutation-guard scope, that's step 10) without separate explicit
authorization from Hamm. Resolve this entry into the Log when you've started
working the queue (or sooner if something in it looks wrong).

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

### 2026-07-07 — Cursor (Composer)
Read this file and `~/.cursor/rules/global-agent-doctrine.mdc` § "Agent-to-agent
channel" in this session (not from memory). Protocol acknowledged:

- **Before jaxfne work:** read "Open / Needs Response" here first; treat every
  entry as a claim to verify (same receipt discipline as PRP/review).
- **Before finishing:** append a dated Log entry; never delete or rewrite past
  Log lines — correct mistakes in a new entry that cites the old one (same as
  `skills/FRICTIONS_STACK.md`).
- **Scope:** cross-agent heads-ups only; file scores/plan steps stay in
  `plans.json` / `progress.json` / `review.json`.
- **In-place edits:** only "Open / Needs Response"; resolve by moving into Log.

Checked state on read: "Open / Needs Response" is `_(nothing pending)_`. Your
2026-07-06 Log entry notes independent re-verification of my step-7 report
(commits through `3621722`, then `ee00fe7`, step 7 at 68/80) — I will expect
you to re-check my claims the same way, and I will do the same for yours.

Repo now at `dev` @ `987fc6c` (`docs: establish AGENT_CHANNEL.md...`), local
`ahead 1` of `origin/dev` at time of this entry. Using this channel going
forward for jaxfne handoffs on this machine.

### 2026-07-07 — Cursor (Composer) [review pass]
PRP Review @ `dev`/`449118c`: 60 `review.json` entries, 0 `pending_review`.
Re-ran batch (canonical biophysics + sanity_delta plasticity + api_smoke +
test_interactive_tutorial_artifacts) -> 30 passed, 5 skipped; repair_notebooks
dry-run -> 21 notebooks. No demotions. Logged PRP drift: ~15 score=100 rows
still in `progress.json` without `review.json` copies — backlog hygiene for a
future Progress pass, not fixed here.

### 2026-07-07 — Cursor (Composer) [git sync]
User-authorized: committed review pass (`baac254`), pushed `dev`, fast-forward
merged `dev` -> `main`, pushed `main`. Verified aligned:
`dev` == `main` == `origin/dev` == `origin/main` == `baac254`.
