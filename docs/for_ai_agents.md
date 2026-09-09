# Documentation for AI agents

jaxfne treats **AI-agent readability as a first-class design goal**, alongside human docs on
[Read the Docs](https://jaxfne.readthedocs.io/). The repo ships two coordinated surfaces:

| Surface | Audience | Role |
|---------|----------|------|
| `docs/` + README | Humans | Tutorials, API reference, guides |
| `artifacts/context.md` + `artifacts/skills/` + `artifacts/AGENTS.md` | AI coding agents | Router, verified workflows, and repository policy |
| `artifacts/developer/` | Maintainers / agents, **local-only** | Developer working notes — gitignored since 2026-07-14, not present in a fresh clone |

## Start here (agents)

1. **Router:** read `artifacts/context.md` — canonical first contact; task-to-skill table lives there.
2. **Import:** `import jaxfne as jtfne` — only public entry point.
3. **Public API:** verify symbols against `jaxfne.public_surface` and live `tests/` before hand-rolling operators.
4. **Repository policy:** `artifacts/AGENTS.md` (step completion, evidence discipline).
5. **Evidence:** `artifacts/publication/publication_evidence_index.json` and `artifacts/publication/frozen_manifest.json` for frozen publication inputs.

## Object grammar

Two grammars, kept distinct (see [Configuration Grammar](guides/configuration_grammar.md)):

```text
Scientific/operator grammar:     Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest
Software execution grammar:      CircuitSpec -> construct -> Model -> simulate -> Signals
```

Paradigm, Objective, and Trainer are optional downstream workflow components, not stages of either grammar.

`construct()` is the single dispatch — extend it, do not bypass.

## Release gates (non-negotiable)

See [Scope & status](scope_and_status.md) for the authoritative gate table.

## Skills sync

Repo skills mirror to client installs via `scripts/harness/sync_skills.py --update`
(canonical `artifacts/skills/` → local tool mirrors; mirrors are generated and
never edited by hand). After changing a skill, run sync and update the harness manifest
together with the change.

## Multi-agent handoff

Maintainers and coding agents share this repository via git. For active mode and
release-specific authorities, use `scratch/CURRENT_TASK.md` (mode is read by Gate 0)
together with `artifacts/release/current_release_authorities.json`.

## Human docs cross-links

- [Quickstart](quickstart.md) — three build paths, HDP, canonical column
- [Configuration Grammar](guides/configuration_grammar.md)
- [NeuronalTensor API](api/neuronal_tensor.md)
- [Jaxley interoperability](guides/jaxley_interop.md)
