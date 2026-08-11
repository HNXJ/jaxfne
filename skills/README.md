# jaxfne skills

Skills are procedural guidance for AI coding agents. They are not a second
mathematical specification and must not duplicate equations or claim semantics
from the project source documents.

## Authority order

For jaxfne work:

1. Mathematical meaning: the six-file revised project source set in
   `artifacts/project_sources/`.
2. Implemented behavior: `jaxfne/` and `tests/`.
3. Public explanation: `README.md` and `docs/`.
4. Procedures: this directory's skills.
5. Current state: `scripts/repo_state_snapshot.py`.
6. History: `artifacts/legacy/` and explicitly labeled historical files.

Public explanation and executable references remain in:

- `docs/operator_doctrine.md`
- `docs/scope_and_status.md`
- `docs/guides/objective_grammar.md`
- `docs/guides/tensor_field_workflows.md`
- `docs/tutorials/notebook_standard.md`

These documents do not replace the mathematical source set.

## Canonical editable skill source

The repository copy under `skills/` is canonical. Copies under
`~/.claude/skills/` and `~/.agents/skills/` are synchronized mirrors only.
Never edit those mirrors independently.

Use the synchronization script in read-only mode first:

```bash
bash skills/SYNC_GLOBAL.sh --check
```

Apply a reviewed synchronization explicitly:

```bash
bash skills/SYNC_GLOBAL.sh --apply
```

The script never removes unrelated or archived skill directories.

## First-class procedures

- `catalog-glossary-jaxfne` — verify current public API names before implementation.
- `jaxfne-worker-context-router` — select the owning module and validation lane.
- `jaxfne-harden` — consolidated implementation safeguards.
- `jaxfne-config` — `Configuration` construction procedure.
- `jaxfne-neural-tensor` — `NeuronalTensor` construction and runtime procedure.
- `jaxfne-neural-network` — construct/simulate/Signals/readout procedure.
- `jaxfne-objective-grammar` — route software stages without redefining mathematics.
- `jaxfne-modeling-optimization-schema` — selectors and serialized schema checks.
- `jaxfne-paradigm-design` — event and trial construction procedure.
- `jaxfne-vis-modules` — signal-driven visualization procedure.
- `jaxfne-spectrolaminar-suite` — spectrolaminar execution/readout procedure.
- `jaxfne-notebook-release-gate` — notebook and artifact validation procedure.
- `jaxfne-sha256-artifact-integrity` — content-identity procedure.
- `jaxfne-release-mutation-guard` — remote mutation authorization procedure.

## Consolidated hardening

`jaxfne-harden/SKILL.md` is the only editable hardening skill. The numbered
`00_INDEX.md`–`11_*.md` files are compatibility pointers and must not contain
independent rules. Historical friction notes belong in `FRICTIONS_STACK.md`
only as archival evidence; executable tests and current code decide behavior.

These files are not active skills:

- `PATCH.md` — historical consolidation note.
- `ANTIGRAVITY_PROMPT.md` — delegation note.
- `SYNC_GLOBAL.sh` — synchronization utility.

## Scope rule

Before changing a skill, remove current versions, SHAs, branch lists, test
counts, benchmark timings, line-number citations, and incident narratives
unless the item is explicitly labeled historical. Prefer a symbol, test name,
or generated state command over a line number or remembered fact.
