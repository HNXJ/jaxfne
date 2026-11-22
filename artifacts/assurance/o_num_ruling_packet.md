# O-NUM ruling packet (2026-11-22; base dev 21ec8655)

Decision for Hamm: what an exactness test asserts when float32 results come from differently compiled, shaped or
segmented XLA graphs, or from chaotic runs. Closes O-NUM, unblocks R06 and R10 (O-76), T-NUM. Nothing here is
applied; no tolerance is in force.

## Evidence (observed; sources cited)

| Case | Tests | Measured difference | Source |
|---|---|---|---|
| Gaussian drive, receptor-exponential route | 1 | 43 of 4000 elements, max 3 ULP (1.5e-5 mV); bit-exact with JIT off or XLA O0 | `mac_gate_attribution.md` #1 |
| Composition, different graph shape or segmentation | 4 | V_m up to 1594 ULP (1.5e-3 mV); spikes identical; O0 and no-JIT pass | #2-5; `continuation_053.md:82`: jit vs eager exactness for stochastic runs "is not a contract" |
| izh2003 known-answer receipts | 2 | N1 mean rate 8.5853 vs 8.5936 and 8.4953 vs 8.4791 at rtol 1e-12; chaotic: weights x(1+1e-6) change spike counts by -274 and +312; receipts record no platform or jaxlib | #6-7 |
| Seed batch vs serial, AdEx and LIF | 6 (O-76) | ulp-scale V_m, max about 5e-5 mV on some WSL runs | `tests/test_adex_route.py:1060-1064` docstring |

Finding (2026-11-22): the six O-76 tests assert spikes and V_m in one body under `xfail(strict=False)`
(`tests/test_adex_route.py:1060,1086,1125`; `tests/test_lif_tensor_route.py:705,731,776`). Their docstrings call the
spike assertions contract, but a spike mismatch also xfails, so the spike contract cannot fail today.

## Options (ranked)

1. **Recommended.** Discrete outputs exact, continuous state bounded.
   - Spikes and event times stay `array_equal`, in strict tests split from the V_m checks.
   - V_m and other continuous state are checked against a per-test bound. The bound is fixed by procedure, not
     guessed:
     - bound = 10 x the largest difference measured over the evidence matrix (Mac, CI Linux, Windows);
     - the bound is admissible only if a mutant (weights x(1+1e-6), or alpha = 1e-6 for zero-coupling identities)
       exceeds it in the same test;
     - where no admissible bound exists, the test asserts spikes only and says so.
   - Same-graph rerun identity in one process stays bitwise.
   - izh2003 known answers: the receipts record platform and jaxlib. The check becomes a rate comparison whose
     tolerance comes from between-seed spread. Fixing that tolerance is a scientific criterion for Hamm.
2. **Bitwise, pinned to the freeze platform.** Skip elsewhere with a stated reason (as `test_equivalence_gate_v20260815.py:94`
   already does). This keeps today's assertions but loses cross-platform coverage, and CI Linux never runs them.
3. **x64 for exactness tests.** Rejected on evidence: x64 changes the random draws (#6-7), so it is no rounding
   control. It also changes the dtype contract.

## Effects of option 1

- R10: six strict spike tests, plus V_m bounds where a bound is admissible.
- R06: the composition and Gaussian-drive tests take the bound procedure.
- No frozen receipt or prereg changes. The izh2003 receipts gain fields: this is an S23 repin, listed for approval.
- Work: one lane, about one day including the evidence matrix runs.
