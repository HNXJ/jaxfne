# Documentation for AI agents

jaxfne treats **AI-agent readability as a first-class design goal**, alongside human docs on
[Read the Docs](https://jaxfne.readthedocs.io/). The repo ships two coordinated surfaces:

| Surface | Audience | Role |
|---------|----------|------|
| `docs/` + README | Humans | Tutorials, API reference, guides |
| `artifacts/skills/` + `artifacts/AGENTS.md` | AI coding agents | Verified workflows, API catalog, config recipes — same source-of-truth as the package, not a parallel spec |
| `artifacts/developer/` | Maintainers / agents, **local-only** | PRP backlog (`plans.json`, `progress.json`, `review.json`) and handoff notes (`AGENT_CHANNEL.md`) — gitignored since 2026-07-14, not present in a fresh clone |

## Start here (agents)

1. **Import:** `import jaxfne as jtfne` — only public entry point.
2. **API catalog:** read `artifacts/skills/catalog-glossary-jaxfne/SKILL.md` <!-- optional: not present in this checkout; catalog ground truth is live code + tests --> before hand-rolling PSD, LFP/CSD-proxy, or spectrolaminar logic.
3. **Task router:** `artifacts/skills/jaxfne-worker-context-router/SKILL.md` <!-- optional: not present in this checkout; routing is governed by `scratch/CURRENT_TASK.md` frontmatter + skill WHEN TO USE matching --> picks config / tensor / paradigm / vis skills.
4. **Lean orientation:** root `artifacts/AGENTS.md` (pointer only — depth lives in skills and docs).
5. **Roadmap:** `docs/fullroadmap.md` — canonical ordered action list (agent-facing, not in MkDocs nav).
6. **Backlog:** `artifacts/developer/progress.json` if present locally — verify scores before claiming a file is done. Not in a fresh clone; maintainers keep it locally.

## Object grammar

Two grammars, kept distinct (see [Configuration Grammar](guides/configuration_grammar.md)):

```text
Scientific/operator grammar:     Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest
Software execution grammar:      CircuitSpec -> construct -> Model -> simulate -> Signals
```

Paradigm, Objective, and Trainer are optional downstream workflow components, not stages of either grammar.

`construct()` is the single dispatch — extend it, don’t bypass.

## Truth gates (non-negotiable)

See [Scope & status](scope_and_status.md) for the authoritative gate table.

## Skills sync

Repo skills mirror to client installs via `scripts/harness/sync_skills.py --update`
(canonical `artifacts/skills/` → local tool mirrors; mirrors are generated and
never edited by hand). After changing a skill, run sync and update the harness manifest
together with the change. Local handoff notes live in `artifacts/developer/AGENT_CHANNEL.md`
<!-- optional: local-only, not present in a fresh clone -->.

## Multi-agent handoff

Maintainers and coding agents share this repo file-based (no live bridge). **Read**
`artifacts/developer/AGENT_CHANNEL.md` <!-- optional: local-only --> before starting;
**append** before finishing (never delete past entries).

## Human docs cross-links

- [Quickstart](quickstart.md) — three build paths, HDP, canonical column
- [Configuration Grammar](guides/configuration_grammar.md)
- [NeuronalTensor API](api/neuronal_tensor.md)
- [Jaxley interoperability](guides/jaxley_interop.md)
