# Haiku 5.5 as a worker model: evaluation (2026-10-09)

Author: vwin-claude-opus55-jaxfne, dispatcher. Model under test: `claude-haiku-5-5` through the
`worker-med`, `worker`, `worker-xhigh` and `worker-max` agents (`~/.claude/agents/`). The agents share
one body and differ only in effort.

## Summary

- Haiku 5.5 is reliable at following a packet. Every lane that ran used its writable
  paths only, ran the acceptance command and kept to the report cap.
- On one deep audit with a known answer key, recall was medium 2.5/6, high 5/6, xhigh 3.5/6.
  Max returned nothing.
- xhigh made the one finding that the Sonnet 5.5 second-reviewer also missed: a stated
  conjecture (F1 L5) is false. Confirmed in exact rational arithmetic.
- Max effort is unusable under the default 32,000 output-token cap. Two runs exceeded it in a
  single response before any tool call.
- Findings from every tier are leads, not facts. Each needs checking (xhigh swapped two line
  numbers; medium raised low-value points).
- Recall did not rise monotonically with effort. With one task this is a single sample, not a
  trend.

## Method

- **Task.** A deep audit, report only, of jaxfne theory family F1 as declared at `b23ed59c`:
  - `artifacts/etudes/theory_f1/README.md` (setting, claims L1–L5 with proofs, checks);
  - the check script and its tests;
  - the two reference rule files the README describes.

  The snapshot was taken before review fixes. Each lane got its own copy and the same packet.
- **Answer key** (6 items), from the independent Sonnet 5.5 second-reviewer, with all items
  fixed in `3019589d`:
  - K1: L3 claims one trace per distinct (M, b); equal decays with different gains share one.
  - K2: L3's proof shows only that one trace cannot serve two decays, not that K − 1 traces
    cannot serve K.
  - K3: L3's proof needs nonzero decays, which the setting does not state.
  - K4: L2 says the lazy snapshot needs only τ_j; it also needs the stored per-edge value, so
    memory stays O(E).
  - K5: the L5 formula decays the accumulator; the code decays the increment.
  - K6: check 4 has no test, and its threshold C ≤ 10 is loose and undeclared.
- **Scoring.** Hit = 1, partial = 0.5. Scored by the dispatcher, so one rater with no blinding.
  Wall time and tokens come from the agent's task notification.

## Results

| Effort | Key recall | Wall time | Tokens | Tool calls | Verdict | Extra valid findings |
|---|---|---|---|---|---|---|
| medium | 2.5/6 (K5, K6; K1 partial) | 197 s | 62.5k | 6 | REVISE | K is global in code but per neuron in the README; check 3's README sentence reads inverted; one-edge-only check coverage |
| high | 5/6 (K3–K6; K1, K2 partial) | 248 s | 75.6k | 6 | REVISE | L1 assumes the rule's `pre_sp` equals `spikes[pre]`; one-edge-only check coverage |
| xhigh | 3.5/6 (K2, K5, K6; K1 partial) | 631 s | 67.7k | 6 | REVISE | **L5 is false** (see below); swapped two line citations |
| max | none | n/a | n/a | 0 | none | Hit the 32,000 output-token cap twice, the second time after a resume telling it to work in small steps |

**The xhigh result.** L5 conjectured that the PCL depression accumulator,
B_e(t) = B_e(t−1) + a·s_i(t)·Y_j(t−1) reset at postsynaptic spikes, does not factor into
per-neuron state. xhigh showed that it does. Y_j depends only on time since j's last spike
τ_j, so B_e(t) = a^(−τ_j)·(z_i(t) − z_i(τ_j)), where z_i(t) = Σ s_i(t′)·a^(t′) is per neuron.
- In exact rational arithmetic the two forms agree on 3 of 3 seeds.
- In float64 the relative error is 4e-16 at 300 steps, 8e-12 at 1000 and 0.28 at 3000,
  because a^(−τ) grows. An implementation would have to rebase z.

The dispatcher verified this. It contradicts a claim that the dispatcher wrote and that
second-reviewer accepted.

## Earlier observations (2026-10-08 and 2026-10-09, builder packets)

- **F1 check script and tests** (`worker`, high; 240 s, 18 tool calls):
  - correct on the first pass;
  - ran the required mutation (a snapshot stored one step late failed 3 of 5 tests);
  - listed its judgement calls in a table instead of choosing silently.
- **Packet trailer conflict.** A packet that dictated the commit trailer stopped the worker,
  because its attribution rule differs from the dispatcher's. Rule since then: the dispatcher
  commits.
- **Pilot classes.** Sweeps, gate runs, failing-test fixes, regenerations, claim checks,
  chunked runs, and a script plus a long run: all clean (`rules/worker-tier.md`).
- **List claims.** In one pilot run a worker checking a list of citations missed 2 of 4 that
  were absent. Packets now say to check every item.
- **match-checker** (Haiku 5.5, high): cut 13 of 707 words with every number intact. It also
  returned useful scope flags on a working paper.

## Harness efficiency

- Tool calls per audit are flat across effort (6). Effort buys thinking time, not
  exploration.
- Tokens per audit were 62–76k at every working tier, so effort barely changed token cost;
  wall time grew 3.2× from medium to xhigh.
- Hooks held throughout. No `python -c`, heredoc into source, unscoped staging or push from any
  worker.
- Max effort fails at the default output cap. It needs `CLAUDE_CODE_MAX_OUTPUT_TOKENS` raised,
  which is a settings change for Hamm.
- Agent files load at session start. A new agent (`worker-max`) was unavailable until the
  harness reloaded its list. The headless `claude -p` fallback failed: OAuth expired, and it
  did not recognise the model ID.

## Routing (applied in `~/.claude/rules/worker-tier.md`)

| Packet | Agent |
|---|---|
| One exact procedure: a lookup, one gate run | `worker-med` |
| Audit against a known checklist; build with a failing test first | `worker` (high) |
| A claim that may be false and needs a derivation; long chains of tool calls | `worker-xhigh` |
| Anything | not `worker-max` until the output cap is raised |
| Science review before integration | still `second-reviewer`, plus a dispatcher check of each finding |

## Limits

- One audit task, one answer key, one rater. Recall differences of one item are within noise.
- Lane order was not randomized. max ran later than the others, after the agent list reloaded.
- The answer key itself missed the L5 refutation, so recall against it understates xhigh.

## Open

- Repeat the comparison on 3 or more tasks of different kinds (a code audit, a numbers check,
  a proof) before treating the effort ranking as stable.
- Decide whether to raise the output cap so max can be tested (Hamm).
