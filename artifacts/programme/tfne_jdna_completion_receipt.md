# JDNA developmental completion — defaults, K_D, origins. CLOSED.

**Branch:** `dev`
**Scope:** new `jaxfne/jdna/completion.py` + additive `value_origins` in
`develop` provenance. No realization, construction, simulation, or TFNE
bridge behavior changed. No new root exports (surface contract untouched).

## Defect

The boundary gave JDNA ownership of completion (positions, distributions,
allocation, stochastic realization, defaults) but the code had no completion
layer: `develop` silently defaulted geometry fields, recorded no per-value
origin, and TFNE's declared-but-inert geometry had no JDNA-side path —
leaving PARAM-04 repair nowhere to land except the forbidden bridge patch.

## Delivered

Project source 7 carries S29.1 (amendment recorded in the source header):
TFNE may be intentionally underdetermined; JDNA completes under `D + K_D`
with per-value origins; geometry realization belongs to JDNA.
Doctrine: `docs/doctrine/tfne_jdna_boundary.md` (new, nav-registered).

`jaxfne/jdna/completion.py` (submodule only):

- `COMPLETION_RULES` — the defaults table. `geometry_distribution` /
  `geometry_domain` default; `cell_allocation` derives only;
  `mechanism_tau_ms` is **required** (kinetics refused, not invented —
  this is why PARAM-03 stays blocked behind the mechanism vocabulary).
- `resolve(name, declared, ...)` — declared always wins; otherwise the
  rule applies; unknown names refused even when declared (typo-safe).
- `realize_geometry(declared, n, seed)` — declared bounds obeyed,
  omitted axes fall back to the unit cube, half-declared axes refused,
  non-uniform distributions refused, degenerate bounds refused.
  Returns float32 `(n, 3)` `(x, y, z)` plus per-axis/distribution/positions
  origins. K_D: same seed identical, different seeds differ.
- `complete_tfne(realization, seed)` — per-leaf positions with one split
  of the development key each, in sorted leaf order. Pure: the input
  realization is never mutated; the TFNE bridge is untouched.

`develop` gains `provenance["value_origins"]` (per-layer counts +
geometry-field origins, per-area pose origins). Additive metadata only:
phenotype bytes, hashes, and saves unchanged. One truth-gate test updated
for the exact key set, intent preserved.

## What this changes

Nothing realized: `develop` phenotypes bit-identical (provenance excluded
from the phenotype hash by construction); TFNE `realize()` output
identical (asserted by test); PARAM-04's pin test still passes because the
bridge still passes declarations through — while JDNA now owns a real
coordinate path with declared/default/sampled origins.

## Tests

- `tests/test_jdna_completion.py` (14): all five `resolve` arms,
  declared/obeyed vs omitted/defaulted geometry, K_D determinism and
  variation, empty populations, three fail-closed geometry cases,
  per-leaf `complete_tfne` with origins, bridge-untouched assertion,
  `develop` origins on canonical and bare-geometry genomes.
- Updated: `test_provenance_semantics_documented_and_real` (exact key set
  gains `value_origins`).
- TFNE 9e replication tests ride in the companion receipt.

## Evidence

```
python -m pytest tests/test_jdna_pseudogenome.py tests/test_jdna_scenarios.py
                 tests/test_jdna_truth_gate.py tests/test_jdna_compact_grammar.py
                 tests/test_jdna_completion.py tests/test_tfne_algebra.py
                 tests/test_tfne_execution.py tests/test_tfne_parameter_transfer.py
                 tests/test_public_surface_contract_v0413.py -q
    -> 189 passed

python scripts/run_test_gate.py dev
    -> 217 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3973 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+22 against the 3951 S20 run: 8 replication + 14 completion)
```

The receipt is necessarily edited after the gate it reports.

## Status

- Boundary + S6.1/S29.1 language: **CLOSED** (compiler conforms).
- JDNA completion layer: **CLOSED**.
- PARAM-04: still open, now owned by JDNA (geometry path exists with
  origins; the bridge pin test stays until a JDNA-realized coordinate
  claim replaces it — not patched here).
- PARAM-03: still blocked behind TFNE2-07; `mechanism_tau_ms: required`
  records the refusal in code.
- Next per stack: TFNE2-04 (frontiers, unblocks CTX-01).
