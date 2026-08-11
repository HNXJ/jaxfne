---
name: jaxfne-harden
description: >-
  Apply jaxfne implementation safeguards before changing analysis, connectivity,
  batching, projection, runtime, public APIs, parameters, optional bridges, or
  helper structure. This is a procedure; mathematical meaning belongs to the
  project source documents.
---

# jaxfne hardening

Read `AGENTS.md` and `catalog-glossary-jaxfne` first. Use the smallest rule set
that matches the change. Verify claims against live code and tests; do not copy
historical incident text into a new specification.

## Analysis and readouts

- Never synthesize fallback spectra, traces, summaries, or diagnostics after an
  upstream failure.
- Strict paths raise. Explicit synthetic paths identify synthetic data in the
  API, metadata, and rendered output.
- Keep numerical preparation separate from plotting and export.
- Keep proxy/readout labels aligned with the current status contract in
  `docs/scope_and_status.md`.

## Connectivity and batches

- Prefer edge-list or block-sparse representations when the graph is sparse.
- Do not claim O(max_edges) behavior if a JIT path still allocates an
  O(n_pre*n_post) score array.
- Compare dense and sparse semantics on a small deterministic case.
- For repeated seeds, candidates, or trials, prefer `vmap` when shapes and
  semantics permit; reuse compiled models.
- Use `lax.scan` for hot traceable time evolution.

## Runtime and fallback behavior

- Report effective dtype, backend, JIT, VMAP, and fallback state.
- Strict mode fails on unsupported compilation or precision paths.
- Do not describe a wrapper as optimized when it only catches wrap-time errors.
- Treat recompilation counts as observations for a specific stable signature,
  not as a universal no-recompile guarantee.

## Public contracts

- Verify names and signatures through the public import before editing.
- Extend public APIs additively where possible; use compatibility wrappers.
- A public `NotImplementedError` must be intentional, documented, and tested.
- New behavior must be reachable through the canonical
  `CircuitSpec -> construct -> Model -> simulate -> Signals` path when that is
  the intended API.
- Do not create notebook-only scientific engines when a package API exists.

## Parameter and schema discipline

- State whether a parameter is global, per-area, per-layer, per-neuron, or
  per-edge.
- Reject ambiguous scope instead of guessing.
- Keep serialized configuration and manifests JSON-safe with finite values.
- Use explicit schema migration when serialized shape changes.

## Experimental and optional boundaries

- Keep Jaxley, jax-fem, PyNWB, and experimental solvers lazy and explicitly
  fenced.
- A proxy projection, compatibility status, or declared solver path is not
  evidence of a calibrated physical result.
- Do not change scientific equations, topology semantics, or solver behavior
  under a context/governance-only task.

## Generalization

- Before adding a private helper, name the behavior rather than the first
  initiative or preset that used it.
- Before adding a near-duplicate function, check whether one parameter can
  preserve both behaviors without obscuring the contract.
- Search all call sites, tests, and re-export aggregators before renaming.

## Validation

Run the smallest relevant checks:

```bash
python3 -m compileall -q jaxfne tests scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest <targeted-tests> -q
python3 scripts/audit_public_docs_language.py --check
```

For context-only edits, also run the repository context audit and verify that
no source/model files changed.
