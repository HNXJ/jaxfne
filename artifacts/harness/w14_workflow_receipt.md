# W14 harness workflow receipt

Status: **0.4.22 baseline** — minimal router/skills discipline recorded.

## Canonical entry points

| Role | Path |
| --- | --- |
| Work queue | `artifacts/todo_stack.md` |
| Policy + loop | `artifacts/AGENTS.md` |
| First-contact router | `artifacts/context.md` |
| Procedures | `artifacts/skills/*/SKILL.md` |

## Rules enforced in 0.4.22 repair

- `todo_stack.md` holds remaining work only (no DONE/REJECTED/DEFERRED history).
- Public docs do not advertise agentic workflows; README has one line to
  `artifacts/AGENTS.md`; contributing page mirrors that pointer.
- Release authorities are version-specific via
  `artifacts/release/current_release_authorities.json`.

## Non-goals (0.4.22)

No new skill proliferation; no duplicate gate procedures in skills vs AGENTS.
