# Item 8 workflow benchmark — results (2026-09-20)

Packet frozen at `1758078` before execution. Tree verified clean after
all runs (no worker modified anything). Same snapshot, same packet, both
arms share the base agent contract (limitation: marginal workflow value).

## Scores (frozen binary gates; denominator = applicable weights only)

| Task | RAW F | WORKFLOW F | H_raw | H_wf |
|---|---|---|---|---|
| A canonical route (den 9) | 1.00 (count 2, canonical path stated) | 1.00 (count 2 + protocol/nulls/V(claim) framing) | 0 | 0 |
| B TFNE/JDNA handoff (den 12) | 1.00 (n=3, 0 edges, digest) | 1.00 (invalid `E(2)` caught, expanded canonically, n=3, 0 edges) | 0 | 0 |
| C lifecycle inspection (den 13) | 1.00 (4 answers + line evidence, ΔX denied) | 1.00 (same + DERIVED/relative qualification) | 0 | 0 |
| D scientific claim (den 13) | 1.00 (9.96 Hz, scoped) | 1.00 (9.46 Hz, scoped, nulls stated) | 0 | 0 |
| E ambiguous request (den 5) | 1.00 (refused, 5 missing items) | 1.00 (refused, gates cited) | 0 | 0 |
| **Mean** | **1.00** | **1.00** | **0** | **0** |

ΔF = 0.00, ΔH = 0.

## Notes (observations, not scores)

- Digests differ between arms on Task B (52b9… vs a58b…): different
  spec-hash inputs from different expansions, both internally consistent
  (3 neurons, 0 edges). Not scored as failure; cross-arm digest equality
  was never a frozen gate.
- Task D configs differed (80/20 vs 75/25 E-fraction); both inside the
  claim band and both scoped. Same note applies.
- WORKFLOW outputs consulted more files per task (e.g. 11 vs 6 on A) and
  carried protocol/nulls/V(claim) framing RAW omitted. Fidelity identical.

## Friction

Per-run tool/token/turn telemetry is unavailable from this harness, so
C=(T,N_tool,N_tokens,N_turns) is unmeasured. Proxy only: mean files
consulted per task — RAW ~6.4, WORKFLOW ~9.2. Direction suggests higher
workflow friction; magnitude and cost unmeasured. No composite utility
constructed (would hide the trade-off).

## Conclusion (observed comparison only)

On these five diagnostics, the base contract already saturates fidelity
(F̄=1.0 both arms, zero hard failures); the workflow adds process framing
at apparently higher friction. Workflow value, if any, lies on harder
multi-stage tasks than these diagnostics. No optimization follows from
this result. If Item 8 is rerun, use tasks that discriminate (multi-stage
carryover, conflicting authorities, near-miss canonical paths).
