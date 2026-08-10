# AGENTS.md — jaxfne operational context

## Identity

jaxfne is a JAX computational scaffold. The public import is:

```python
import jaxfne as jtfne
```

Public wording follows `docs/scope_and_status.md`: outputs are Relative by
default; Absolute values require an explicit calibration step.

## Two grammars

Scientific operator grammar:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

Software execution grammar:

```text
CircuitSpec -> construct -> Model -> simulate -> Signals
```

`CircuitSpec` includes supported `Configuration` and `NeuronalTensor` tiers.
`Paradigm`, readouts, objectives, tuning, and export attach to this execution
path; compatibility aliases are not the scientific grammar.

Continuation state is shape-preserving and treats the H-state as an opaque JAX
array. The current HDP kernel uses the scalar special case; generic
continuation prose must not equate H-state with one homeostatic variable or
with a required physical unit. Internal relative coordinates remain valid
until an explicit calibration transformation at a semantic boundary.

## Authority and evidence

Mathematical meaning belongs to the six-file revised project source set under
`artifacts/project_sources/`. Do not duplicate its equations into skills or
agent context. Current public explanation and executable references are:

- `docs/operator_doctrine.md` — public operator contracts and stage vocabulary.
- `docs/scope_and_status.md` — public Relative/Absolute and status fields.
- `docs/guides/objective_grammar.md` — executable software sequence.
- `docs/guides/tensor_field_workflows.md` — source/field/probe contracts.
- `docs/tutorials/` and `docs/tutorials/notebook_standard.md` — public
  executable evidence protocol.

Implemented API truth belongs to live `jaxfne/` code and `tests/`. Public
explanation belongs to `README.md` and `docs/`. Skills are procedures and
references, not parallel scientific specifications. Generated state comes from
`scripts/repo_state_snapshot.py`; archival material is not current authority.

## Evidence classification and scope discipline

Reports keep these categories separate:

- `SPECIFIED`: required by the authoritative project specification.
- `IMPLEMENTED`: present in the current checkout.
- `TESTED`: covered by an executable test with a command receipt.
- `OBSERVED`: measured in one named run/environment.

Do not promote one category into another. Before implementation, declare the
target invariant, expected files, API/mathematical/numerical/claim/
documentation/compatibility deltas, forbidden adjacent changes, and the stop
condition. Stop when the requested invariant is resolved, targeted tests pass,
directly affected documentation agrees, and no directly blocking discrepancy
remains. Do not clean neighboring modules, add convenience helpers, expand
tutorials, or repair unrelated scientific behavior.

Skills and agent context teach recovery and verification procedures; they do not
store current equations, SHAs, versions, defaults, test counts, bug
inventories, or implementation line locations as durable facts.

## JAX and API invariants

- Use `jax.numpy` in numerical kernels and explicit PRNG keys for randomness.
- Use pure numerical functions and `jax.lax.scan` for hot time evolution.
- Use `jax.vmap` for batches/seeds when shapes and semantics permit.
- Use `jax.jit` only for stable numerical hot paths; report effective runtime
  state rather than requested state.
- Default to float32. Enable x64 explicitly before array construction.
- Keep plotting, JSON, serialization, and file I/O outside JIT.
- Preserve public APIs; prefer additive compatibility wrappers.
- Keep optional dependencies lazy and fail explicitly when unavailable.
- Use package APIs for reusable scientific computation; do not create notebook
  engines that duplicate package behavior.
- Preserve conservative truth/status metadata. `linear_solver` is compatibility
  metadata for the current proxy path, not proof of a solved PDE.

## Repository workflow

Before repository work, run:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git ls-remote origin refs/heads/main refs/heads/dev
```


Before naming an unfamiliar API or helper, read
`skills/catalog-glossary-jaxfne/SKILL.md` and verify the live symbol. Use
`skills/jaxfne-harden/SKILL.md` for implementation safeguards.

Preferred validation entrypoints:

```bash
python3 -m compileall -q jaxfne tests scripts
python3 scripts/repo_state_snapshot.py
python3 scripts/audit_public_docs_language.py --check
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q --tb=short
```

Run only the gates relevant to the task and report exact results. Do not claim
full validation when the full suite or docs build was not run.

## Planning, reports, and mutations

Use PRP only for implementation/release work that needs persistent planning.
Simple inspection, factual questions, and one-session context work do not
require PRP files. `AGENT_CHANNEL.md` is optional handoff state; do not create
or append it during a read-only task.

Reports include:

```text
API delta:
Mathematical delta:
Numerical delta:
Claim/evidence delta:
Documentation delta:
Compatibility delta:
```

Before remote or irreversible Git mutation, obtain applicable explicit
authorization. Never commit, push, merge, tag, release, or upload merely
because a workflow suggests it.
