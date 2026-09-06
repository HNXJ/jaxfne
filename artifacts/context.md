# jaxfne — context for AI agents

Entry point for an AI coding agent working on, or with, this repository.

This file is a **router**, not a specification. It names the invariants, the authoritative
sources, and the commands that settle a question. Depth lives in the files it points to, so
nothing here has to be kept in step with them by hand.

## One import

```python
import jaxfne as jtfne
```

`jaxfne` is the only supported public entry point, and `construct()` is the single dispatch
point — extend it rather than bypassing it. What counts as public is decided by
`jaxfne.public_surface` together with `jaxfne.__all__`, not by any list written in prose.

## Two grammars, kept distinct

```text
Scientific/operator:  Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest
Software execution:   CircuitSpec -> construct -> Model -> simulate -> Signals
```

Paradigm, Objective and Trainer are optional downstream workflow components, not stages of
either grammar.

## Where authority lives

| Question | Authoritative source |
|----------|----------------------|
| What is public, and in which tier? | `jaxfne.public_surface` |
| What does a function actually do? | the implementation and `tests/` |
| How is it meant to be used? | `docs/`, published as the documentation site |
| Fuller agent orientation | `docs/for_ai_agents.md`, `artifacts/AGENTS.md` |
| Reusable agent workflows | `artifacts/skills/` — in a repository checkout; not carried in the source distribution |
| Which checks block a release? | `scripts/run_test_gate.py` |
| CI, branch and documentation policy | `docs/ci_policy.md` |

Resolve a path, symbol or flag against the repository before relying on it. Any file that
points at other files can name something that no longer exists without erroring, so treat a
recalled detail as a hypothesis until it is re-checked.

## Settling a claim

Run a gate rather than asserting a result. The gate vocabulary is defined once, in
`scripts/run_test_gate.py`:

| Gate | Scope |
|------|-------|
| `dev` | curated architectural checks |
| `broad` | repository-wide, excluding slow markers |
| `release` | broad, plus slow markers and release example/docs checks |
| `rc` | full release-candidate verification |
| `publication` | frozen scientific experiments and artifact validation |

```bash
python scripts/run_test_gate.py broad
make test-rc
```

A command that exits zero is evidence for exactly what it ran, and for nothing wider. Give the
command and its output together, or say the claim is unverified.

## Working conventions

- Make the smallest change that reaches a passing gate, and leave unrelated invariants alone.
- Array shapes, units, coordinate frames, sample rates and index bases are part of the
  contract. State an intentional break at the site of the change.
- Relative and proxy readouts are distinct from calibrated physical quantities. Hold that
  distinction in code, docstrings and prose alike.
- Generated artifacts have generators. Regenerate them instead of editing them by hand.
- Surface a contradiction rather than resolving it by choosing: two sources disagreeing on one
  quantity, a declared path that does not exist, or a gate whose inputs contradict its outputs.
