# 24-MIN-01 receipt — minimization entry (baseline + acceptance gate)

**Branch:** `dev`. No product code changed.

## Complexity baseline (dev @ `a370752`, measured not estimated)
| domain | files | lines |
| --- | --- | --- |
| `jaxfne/` source | 159 py | 62,475 |
| `tests/` | 354 test files | 60,064 |
| `docs/` prose | 152 md | 27,860 |
| public surface | 190 symbols (`__all__`) | — |
| simulation kernels | 14 (13 in `emitters.py` + registrable HDP) | — |
| package version | 0.4.23 | — |

## Acceptance gate (binds DOC/CODEMIN/TESTMIN/PKG/API/PRO items)
Each minimization item must demonstrate, against THIS baseline:
1. identical required semantics, API constraints, numerical behavior
   (EQUIV-01 table bounds), and scientific meaning;
2. measured before/after on the same machine/profile where perf-relevant;
3. verdict IMPROVEMENT only on a clear net reduction; otherwise
   NO_CHANGE with the measured reason (v0.4.23 SIMP-sequence precedent).
Reductions that move complexity into parameter-heavy helpers, weaken
evidence (diagnostics/receipts), or trade exactness for undeclared
tolerance fail the gate.
