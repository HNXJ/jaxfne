# O-NUM ruling packet (2026-11-22; base dev 21ec8655)

Decision for Hamm: what an exactness test asserts when float32 results come from differently compiled, shaped or
segmented XLA graphs, or from chaotic runs. A ruling closes O-NUM and unblocks R06, R10 (O-76) and T-NUM. This
packet applies nothing; tolerances in force today are listed below unchanged.

## Evidence (observed; sources cited)

| Case | Tests | Measured difference | Source |
|---|---|---|---|
| Gaussian drive, receptor-exponential route | 1 | 43 of 4000 elements, max 3 ULP (1.5e-5 mV); bit-exact with JIT off or XLA O0 | `mac_gate_attribution.md` #1 |
| Composition, different graph shape or segmentation | 4 | V_m up to 1594 ULP (1.5e-3 mV); spikes identical; O0 and no-JIT pass | #2-5; `continuation_053.md:82`: jit vs eager exactness for stochastic runs "is not a contract" |
| izh2003 known-answer receipts | 2 | N1 mean rate 8.5853 vs 8.5936 and 8.4953 vs 8.4791 at rtol 1e-12; chaotic (weights x(1+1e-6) change spike counts by -274 and +312); receipts record no platform or jaxlib | #6-7 |
| Seed batch vs serial, AdEx and LIF | 6 (O-76) | ulp-scale V_m, max about 5e-5 mV on some WSL runs | `tests/test_adex_route.py:1060-1064` docstring |

Tolerance in force today: the strict seed-batch contract tests already assert spikes with `array_equal` and V_m (or
H, w) with `atol = V_ATOL_MV = 1e-4` mV, `rtol = 0` (`tests/test_adex_route.py:1047-1048,1142,1174`;
`tests/test_lif_tensor_route.py:691-692,793,814,853`). The six O-76 tests are `xfail(strict=False)` bitwise
diagnostics beside them: four repeat the spike check and add bitwise V_m (`test_adex_route.py:1060,1086`;
`test_lif_tensor_route.py:705,731`), two compare final state only (`test_adex_route.py:1125`: V, w;
`test_lif_tensor_route.py:776`: H, w). The spike contract is therefore enforced by the strict tests; the xfails add
only the bitwise evidence. The composition cases (#2-5, up to 1.5e-3 mV) exceed the existing 1e-4 mV bound.

## Options (ranked)

1. **Recommended: extend the existing contract pattern.** Discrete outputs exact; continuous state within a per-test bound.
   - Spikes and event times stay `array_equal`.
   - The bound is fixed by procedure, not guessed:
     - bound = 10 x the largest difference measured over the evidence matrix (Mac, CI Linux, Windows);
     - the bound is admissible only if a mutant exceeds it in the same test. The mutant's size is chosen by a
       probe that shows the break; a weight x(1+1e-6) is the starting candidate, from the #6-7 positive control;
     - where no admissible bound exists, the test asserts spikes only and says so.
   - The seed-batch tests keep 1e-4 mV unless the matrix shows that bound is inadmissible.
   - Same-graph rerun identity in one process stays bitwise.
   - izh2003 known answers: the receipts record platform and jaxlib. The check becomes a rate comparison whose
     tolerance comes from between-seed spread. Fixing that tolerance is a scientific criterion for Hamm.
2. **Bitwise, pinned to the freeze platform.** Skip elsewhere with a stated reason (as `test_equivalence_gate_v20260815.py:94`
   does). This keeps today's assertions but loses cross-platform coverage; CI Linux never runs them.
3. **x64 for exactness tests.** Rejected on evidence: x64 changes the random draws (#6-7), so it is no rounding
   control. It also changes the dtype contract.

## Effects of option 1

- R10 (O-76): the six xfails duplicate strict contract tests. Either retire them, or keep them as platform-pinned
  diagnostics (option 2 applied to diagnostics only). No new tolerance is needed for R10.
- R06: the composition and Gaussian-drive tests take the bound procedure.
- No frozen receipt or prereg changes. The izh2003 receipts gain fields: this is an S23 repin, listed for approval.
- Work: one lane, about one day including the evidence matrix runs.
