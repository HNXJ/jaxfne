# TFNE2-07 (mechanism subset) + TFNE-PARAM-03 — CLOSED.

**Branch:** `dev`
**Scope:** mechanism vocabulary, resolution, and kinetics repair only.
Identity, topology, weights, delays, geometry, and all other failure
classes unchanged. JDNA untouched (`mechanism_tau_ms` stays `required`).

## Defect

TFNE-PARAM-03: `mechanism identity != mechanism kinetics`. A declared
`AMPA` edge executed at 0.1ms, not AMPA's 2.0ms, because
`to_neuronal_tensor` built `Inter/AreaConnection` with no `static=`
argument, so every TFNE spec inherited the `StaticParams` placeholder
while equivalent hand-written tensors carried 2.0/5.0. Blocked behind
TFNE2-07 because TFNE mechanism names are arbitrary strings: resolving
`AMPA`→2.0 while leaving the also-arbitrary `GABA` at 0.1 would distort
E/I balance asymmetrically — worse than the uniform placeholder.

## Delivered

One traceable identity from TFNE through JDNA into the kernel:

```
declared mechanism -> resolved mechanism identity
                   -> mechanism parameters / kinetics
                   -> realized connection
                   -> kernel-consumed mechanism
```

`resolve_mechanism(name, rule_params)` (`jaxfne/tfne.py`) classifies every
encountered name — no heuristic aliasing:

| Class | Meaning | Example |
|---|---|---|
| CANONICAL | exact canonical-table entry; tau/reversal/sign from it | AMPA→2.0, GABA_A→5.0, NMDA→100, GABA_B→150 |
| EXPLICIT_ALIAS | alias by authoritative definition | none exist (reserved, uninhabited) |
| CUSTOM_DEFINED | name + sufficient definition (finite positive `tau_ms`) | `FOO` + `tau_ms = 3.5` |
| UNRESOLVED | unknown and undefined → `E_MECHANISM_UNRESOLVED` | GABA, typos |
| NOT_PERMITTED | known but inadmissible → `E_MECHANISM_NOT_PERMITTED` | reserved `tfne_*`, canonical+contradictory tau, bad tau |

Load-bearing decisions, each fail-closed rather than guessing:

- **GABA is UNRESOLVED, not aliased.** It is ambiguous (A: 5ms vs B:
  150ms, 30x apart), so no alias can be correct. The two executing GABA
  fixtures moved to GABA_A with this rationale recorded; their tested
  property (inhibitory-mechanism identity transfer) is preserved, and
  both mechanisms moved to canonical kinetics together.
- **Absent mechanism means direct coupling** (`tfne_direct`, placeholder
  0.1ms, documented not-a-receptor) — matching the realization path's
  long-standing default. Silence is not read as a receptor claim, and no
  existing direct-coupled trajectory moves.
- **Enforcement at the kinetics point** (`to_neuronal_tensor`), not at
  parse/resolve/realize: identity transfers fine unresolved (PARAM-01),
  and invention would occur only at kinetics. `realize()` with GABA still
  succeeds; execution refuses.
- **Canonical + contradictory tau refused**: a canonical identity with
  non-canonical kinetics is not permitted; identical tau is accepted as
  harmless consistency.
- **Single chokepoint**: resolved tau flows into `StaticParams(dT_ms)`
  (+ reversal metadata) on every `Inter/AreaConnection`;
  `to_configuration` inherits it from bridge metadata unchanged.
  `dt != synaptic tau` preserved (existing dt-invariance test passes
  with new values).

## What this changes

Executed kinetics for canonical/custom TFNE mechanisms (0.1 → table
values, all together): AMPA 2.0, GABA_A 5.0, NMDA 100.0, GABA_B 150.0.
Nothing else moves: tfne_direct stays 0.1, signs stay cell-type-derived,
indices stay declaration-ordered, counts/weights/topology identical.

## Tests

`tests/test_tfne_algebra.py`, section "9i" (6 functions): canonical
table parity (single source of truth asserted), direct-coupling default,
GABA/unknown refusal + realize-still-succeeds, reserved/contradictory/bad
refusals, custom definition sufficiency, tau digest sensitivity.

`tests/test_tfne_parameter_transfer.py`, PARAM-03 section (5 functions):
canonical kinetics executed per receptor; custom tau executed;
asymmetric AMPA+GABA_A `{2.0, 5.0}`; TFNE-vs-handwritten parity (`{2.0}`
both); unresolved refused at `to_configuration` after successful
`realize()`.

Converted fixtures: `test_d_mechanism_and_weight`,
`test_f_ordered_and_cross_rules_keep_distinct_parameters` (GABA→GABA_A,
rationale in comments).

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py
                 tests/test_tfne_parameter_transfer.py
                 tests/test_jdna_completion.py tests/test_jdna_pseudogenome.py
                 tests/test_jdna_scenarios.py tests/test_jdna_truth_gate.py
                 tests/test_jdna_compact_grammar.py
                 tests/test_public_surface_contract_v0413.py -q
    -> 245 passed

python scripts/run_test_gate.py dev
    -> 271 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 4029 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+11 against the 4018 TFNE2-06 run: 6 vocabulary + 5 PARAM-03)
```

The receipt is necessarily edited after the gate it reports.

Per-mechanism lifecycle report (declared → resolved → configured →
JDNA → realized → kernel-consumed):

| Declared | Resolved | Configured tau | JDNA state | Realized tau | Executed tau |
|---|---|---|---|---|---|
| AMPA | CANONICAL | 2.0 | declared (origin) | None (`declared_not_simulated`) | 2.0 |
| GABA_A | CANONICAL | 5.0 | declared | None | 5.0 |
| NMDA | CANONICAL | 100.0 | declared | None | 100.0 |
| GABA_B | CANONICAL | 150.0 | declared | None | 150.0 |
| FOO+3.5 | CUSTOM_DEFINED | 3.5 | declared | None | 3.5 |
| GABA | UNRESOLVED | refused | declared | None | refused |
| (absent) | direct 0.1 | 0.1 | n/a | None | 0.1 |

(`Realized tau None` is by design: the realized table records identity,
kinetics resolve downstream. `stored != realized != consumed` holds.)

## Status

- TFNE2-07 mechanism subset: **CLOSED**. Remaining TFNE2-07: full typed
  failure classes, S13 redundancy, S11 ungrouped-X rejection.
- TFNE-PARAM-03: **CLOSED**.
- TFNE-PARAM-04: untouched (geometry).
