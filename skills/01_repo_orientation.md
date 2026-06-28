# Repository Orientation Skill

## Purpose
Make the agent understand the repo before it edits the repo.

## What this repo is
- `jaxfne` is a JAX-native TFNE scaffold.
- The canonical import is `import jaxfne as jtfne`.
- The current scientific posture is proxy/scaffold only unless a file, test, and manifest prove otherwise.
- The operator story is:
  `Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation`

## What to inspect first
1. `jaxfne/__init__.py` and root exports.
2. `jaxfne/core.py` for configuration, construction, simulation, batching, and connectivity.
3. `jaxfne/fields/proxy.py` for source/field/probe semantics and projections (default `density_preserving`).
4. `jaxfne/neuronal_tensor.py` for tensor-first column construction.
5. `jaxfne/runtime.py` for precision, JIT, and VMAP helpers.
6. `jaxfne/builders.py` for canonical builders, layer sets (`CANONICAL_LAYERS_6L`), parameter semantics.
7. `jaxfne/vis/` for plotting (not `tutorial_panels.py` alone).
8. `skills/catalog-glossary-jaxfne/SKILL.md` before inventing helpers.
9. `tests/` for the actual contract surface.

## Working rules
- Use package-native APIs before writing notebook-local engines.
- Prefer compatibility wrappers over breaking changes.
- Keep optional dependencies lazy.
- Keep claims inside the current evidence boundary.
- Treat `*_proxy` as proxy readout, not physical measurement.

## Acceptance checks
- The agent can name the right module before writing a patch.
- The agent does not invent a public API from memory.
- The agent can explain the current file's role in one sentence.
