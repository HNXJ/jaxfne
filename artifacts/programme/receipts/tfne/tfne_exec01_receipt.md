# TFNE-EXEC-01 — CLOSED. Execution qualified; TFNE-PARAM-01 opened.

**Branch:** `dev`
**Entry state:** `edb3838` (S14, S6/S7, S10 closed)
**Scope:** prove the executable claim, or qualify it. No change to compiler semantics.

## Claim under test

`jaxfne/tfne.py` and the doctrine pipeline table both asserted that TFNE output
executes in the existing JaxFNE kernels. `tests/test_tfne_algebra.py` states
"no simulation kernels are touched", and no test called `construct` or
`simulate` on `to_neuronal_tensor` output. Structural conversion was tested;
executable equivalence was asserted.

## Result

The chain runs. `tests/test_tfne_execution.py` drives it end to end:

```
TFNE source -> parse -> resolve -> realize
                     -> to_neuronal_tensor -> construct -> simulate
```

Receipts from a 15-neuron, 50-edge specification over 20 ms at dt 0.1 ms:

| Stage | Observed |
| --- | --- |
| NF | 1 relation `V1.L4 -> V1.L2`; 15 neurons; 50 edges |
| `to_neuronal_tensor` | area `V1`, layers `L4` (E 0.8 / PV 0.2) and `L2` (E 1.0) |
| `construct` | `Model`; `connectivity_summary()["n_edges"] == 50` |
| `simulate` | `Signals`; `V_m (200, 15)`, `spikes (200, 15)`, finite, `V_m` spanning -74.76 to 26.86 mV |

What transfers, verified rather than assumed:

- **Edge count**, exactly: 50 realized, 50 executed.
- **Cell-type split**, exactly: `(L4,E)->(L2,E)` 40 edges and `(L4,PV)->(L2,E)` 10 edges, which is the 8/2 allocation of `P = 0.8/0.2` over `N = 10` onto all 5 of `L2.E`.
- **Mechanism identity**: every executed edge carries an `AMPA` receptor.
- **Index map**: `I` addresses the simulated axis. `V1.L4` is `(0, 10)` and `V1.L2` is `(10, 15)`; the slices abut and the second ends at `V_m.shape[1]`, so a TFNE path selects the right columns rather than merely a valid range.
- **The kernel integrated**: `V_m.std() > 0` and `spikes` is binary. A constant trace would have meant arrays allocated but never stepped.

## TFNE-PARAM-01 opened

Connection parameters do not transfer.

```
TFNE declared / realized edge weight : 0.5   (uniform, from O[ff])
executed model edge weight           : +/-0.258199
tensor InterConnection w_mech        : 1.0
```

`realize()` and `to_neuronal_tensor()` are two independent compilations of one
source. The `weight` handling lives only in the realize path; the tensor bridge
builds `InterConnection`s without it, so construction substitutes its own
scaling and a sign convention by cell type. The executed model is therefore not
parameter-equivalent to the realization, though it is topology- and
identity-equivalent.

This is pinned by `test_declared_rule_weight_does_not_reach_the_executed_model`
so the gap cannot close or widen silently. The decision it needs — carry
`weight`/`probability`/`delay` across the bridge, or declare TFNE rule
parameters non-binding on execution and say so in the doctrine — is queued as
**TFNE-PARAM-01**, not taken here.

## Documentation corrected

Both places that implied `(s, h0, I)` is what executes now say otherwise. The
module docstring shows the two compilations explicitly, and the doctrine
pipeline table carries a note that the Execute row means "the kernels run the
tensor built from this source", not "the kernels consume `s`".

## Evidence

```
python -m pytest tests/test_tfne_execution.py -q  ->  5 passed in 4.44s
python -m pytest tests/test_tfne_algebra.py -q    ->  30 passed
python scripts/run_test_gate.py dev               ->  138 passed, 1 skipped, 2 deselected
                                                       docs language audit: pass
                                                       vocabulary check: pass
```

The dev gate is a curated list of nine modules and contains neither TFNE test
file, so its count is unchanged at 138 and is evidence only that the
architectural checks did not regress. Both TFNE modules are collected by the
blocking `broad` sweep -- 35 node ids across
`tests/test_tfne_algebra.py` and `tests/test_tfne_execution.py` under
`pytest tests -m "not slow" --collect-only` -- so the new evidence is gated,
not orphaned.

## Status

- TFNE-EXEC-01: **CLOSED** — executes, with topology and identity verified and the parameter gap named rather than hidden.
- TFNE-PARAM-01: **OPEN**, pinned by test.
- Next by authorization: `TFNE-IMPORT-01`, then the remaining conformance gaps.
