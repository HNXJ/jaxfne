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

## Ruling and measurements (2026-11-22)

Hamm approved option 1 on 2026-11-22, with conditions:
- spikes stay exact;
- zero observed difference keeps exactness;
- a bound counts only if, per test i, 10 x Delta_platform,i < Delta_mutant,i, where the mutant is a scientifically
  meaningful defect. Mutant size may not be raised until this passes;
- no xfail may mask a regression.

Hamm, 2026-11-22, after checkpoint aafbb027:
- Apply the inequality separately to each test, each observable and each mutant pairing, using the smallest
  relevant mutant deviation. The aggregate Mac maximum does not count.
- The 84-101 mV defects support the chosen mutants only. They do not show sensitivity to smaller defects or to
  other defect classes.
- The AdEx final-w xfail (`tests/test_adex_route.py:1125`) stays explicitly unresolved. R10 scopes or bounds it,
  and it is not retired without strict coverage.
- Windows/Linux agreement alone justifies no further work, and the gates are unchanged.

The 1.5e-2 mV composition bound is not approved. The Gaussian-drive bound is provisional until there is Mac
evidence.

Measured with `scripts/internal/onum_measure.py` at base dev a1cd6e02:
- Windows: Windows 11, Python 3.14.3, jax/jaxlib 0.10.1, CPU.
- Linux: WSL2 Ubuntu (glibc 2.39), Python 3.12.3, jax/jaxlib 0.10.1, CPU.
- The two hosts give identical numbers, mutants included. The weight mutants scale one side's recurrent weights,
  or the Gaussian std, by 1 + eps.

| Comparison | Windows, Linux max abs dV (mV) | Mac (recorded) | Weight-mutant max abs dV at eps = 1e-6 / 1e-5 / 1e-4 / 1e-3 / 1e-2 (mV) | Spikes under mutants |
|---|---|---|---|---|
| Gaussian drive (#1) | 0 | 1.5e-5 | 2.1e-3 / 1.8e-2 / 0.23 / 2.4 / 81 | no spike probe |
| Continuation, member 0 vs solo (#2) | 0 | aggregate only: <= 1.5e-3 over #2-5 | 7.4e-4 / 7.0e-4 / 2.7e-3 / 2.3e-2 / 0.23 | equal at every eps |
| Continuation, segments vs full | 0 | as above | (no mutant) | n/a |
| Three-area middle member (#3) | 0 | as above | 0 / 5.0e-5 / 6.5e-4 / 1.3e-2 / 0.14 | equal at every eps |
| Slice vs solo, eager (#4) and jit (#5) | 0, both members | as above | 7.4e-4 / 7.0e-4 / 2.7e-3 / 2.3e-2 / 0.23 | equal at every eps |

Consequences:
- Windows and Linux: exactness stays (zero difference). CI Linux passing the current `array_equal` assertions
  implies zero difference there too (inferred, not measured).
- On Mac, the per-test maxima are unknown. vmac answers on tailscale but refuses SSH on port 22 (2026-11-22).
- In this fixture, a weight error of up to 1% changes no spike. The composition oracle is therefore weak against
  weight errors, so the defect-class mutants below are measured.

Defect-class mutants: the faults each identity test exists to catch (`defects()` in `onum_measure.py`). Windows and
Linux give identical numbers.

| Test | Defect | max abs dV (mV) | Spikes |
|---|---|---|---|
| #1 Gaussian drive (P-045) | std applied as variance (9 for 3) | 86 | no probe |
| #1 | one cell type's std dropped | 88 | no probe |
| #1 | wrong noise seed | 101 | no probe |
| #2, #4, #5 | member seeds swapped | 84 | differ |
| #4, #5 | block offset by one neuron | 101 | differ |
| #3, #4, #5 | cross-model edge leak, p = 0.3, weight 0.05 / 0.5 | 93 / 97 | differ |
| #2 segments | continuation state dropped (second segment restarts) | 99 | n/a in that comparison |

Reading (inferred from these numbers; for Hamm to rule):
- Each defect gives Delta_mutant,i >= 84 mV. The inequality 10 x Delta_platform,i < Delta_mutant,i therefore holds
  wherever Delta_platform,i < 8.4 mV, and the largest recorded Mac difference is 1.5e-3 mV.
- Every composition defect also changes spikes, so the exact spike check alone detects them.
- The weight-scaling mutants are not defects these tests guard. On them the oracle is weak: no spike change up
  to 1% and sub-mV V_m changes. An oracle that must detect parameter errors needs a fixture that spikes more
  densely or a probe sensitive to weights (inferred).
- This does not replace the per-test Mac maxima. It only says which mutant class makes the inequality meaningful.

## Effects of option 1

- R10 (O-76): the six xfails duplicate strict contract tests. Either retire them, or keep them as platform-pinned
  diagnostics (option 2 applied to diagnostics only). No new tolerance is needed for R10.
- R06: the composition and Gaussian-drive tests take the bound procedure.
- No frozen receipt or prereg changes. The izh2003 receipts gain fields: this is an S23 repin, listed for approval.
- Work: one lane, about one day including the evidence matrix runs.
