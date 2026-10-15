# TFNE2-07 (remainder) — CLOSED. Typed failures, S13, S11.

**Branch:** `dev`
**Scope:** failure taxonomy classes, projection-redundancy check,
ungrouped-X refusal. No adjacency, kinetics, topology, or kernel
semantics changed. Mechanism resolution untouched (constraints from the
mechanism turn preserved: no aliasing, `tfne_direct` placeholder,
execution-layer enforcement).

## Defect

S25 requires semantic failure classes; the compiler raised prose
`TFNEError` everywhere, so handlers could match only message text.
S13 (redundant explicit projections invalid) and S11 (ungrouped
same-rule `X` chains require grouping) had no check at all: duplicate
output double-counted edges silently, and non-associative chains
associated left without comment.

## Delivered

Taxonomy, all subclassing `TFNEError` (existing handlers keep working;
messages byte-identical, only the class changed):

```
TFNEAddressUnknown, TFNEAmbiguousExpansion, TFNEExclusionUnknown,
TFNEFrontierUnresolved, TFNEInvalidProportion, TFNEMechanismUnresolved,
TFNEMechanismNotPermitted, TFNEMissingPolicy, TFNEOrderViolation,
TFNEProjectionRedundant
```

Parser/lexer structural errors stay plain `TFNEError`: the vocabulary
is semantic, not parser-specific (S25). S20.1 order codes migrate as
`TFNEOrderViolation` (same family principle; nothing invented).

- S13: post-expansion, explicit (bare) projections are compared against
  canonically generated rule output as `(src, dst, mechanism)` over
  realized leaves with direction normalized. Any overlap is
  `E_PROJECTION_REDUNDANT` — S14 requires explicit additions disjoint
  from `G_0`, so partial overlap cannot be a valid addition, and
  leaf-level identity catches coarser scopes double-counting generated
  leaf routes. Mechanism comparison is textual (no front-loaded
  kinetics refusal).
- S11: direct syntactic nesting of bare `Cross` nodes under one rule
  (`A X[k] B X[k] C`, either side) is `E_AMBIGUOUS_EXPANSION` unless
  the rule declares `associative = true`. Braces, named-definition
  boundaries, and differing rules keep left-assoc behavior. New code,
  not sealed text (S25 names the class descriptively).

## What this changes

Only previously-silent invalid specs: redundant bare projections now
refused instead of double-counted; ungrouped same-rule X chains refused
instead of silently left-associated. Everything valid expands
byte-identically (targeted suites green with no fixture changes; broad
has one unrelated pre-existing timing failure, documented above).

## Tests

`tests/test_tfne_algebra.py`, section "9j":

- taxonomy classes subclass `TFNEError`/`ValueError`;
- 7 parametrized triggers asserting exact class + code (address,
  frontier, mechanisms at the kinetics point, order, proportion,
  missing policy);
- redundant bare projection refused with typed class; distinct
  mechanism allowed; partial/coarse overlap refused;
- ungrouped same-k refused; braces, definitions, differing rules, and
  the `associative` escape all pass with edge counts.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 144 passed

python scripts/run_test_gate.py dev
    -> 282 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> broad = 4039 passed + 1 failed (75 skipped, 37 deselected, 4 xfailed)
       (+22 outcomes against the 4018 TFNE2-06 run: 11 §9j tests;
       the other 11 are the mechanism-turn tests, confirming that
       turn's recorded 4029 figure arithmetically)
```

The single failure is `test_field01_audit.py::
test_source_generation_vs_projection_split`, a wall-clock assertion
(`t_proj < t_sim`) in a test exercising none of the changed code paths
(builder + simulate + projection only). Reproduced on the clean-tree
control with this turn's changes stashed (identical timing split), work
restored afterward — pre-existing timing sensitivity under machine
load, therefore no evidence of TFNE regression, but broad itself was
not PASS. Ruff clean.

The receipt is necessarily edited after the gate it reports.

## Status

- TFNE2-07: **CLOSED** (mechanism subset previously; classes/S13/S11
  now). S13 redundancy and S11 grouping join the conformance table as
  satisfied.
- Next in stack order: CTX-01 (definition layer; `in`/`out` unblocked
  since TFNE2-04).
- PARAM-03 / PARAM-04 untouched per scope.
