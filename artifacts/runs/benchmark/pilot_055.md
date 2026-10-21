# 5d benchmark pilot (2026-09-26)

One task (AT-02), one trial per arm, opencode `muse-spark-1.3-contributor`
through opencode-delegate (code tier). Each arm ran in a fresh single-commit
repository built by `git archive` from db4518e with `prepare_worktree` applied
(answers removed; the direct arm also without `artifacts/skills` and
`jaxfne/agent.py`). Worker session directories were checked to equal the
checkout. Neither worker changed anything but `bench_solution.py`.

| Arm | Elapsed | Tokens in/out | model | execution | observation | plasticity | dynamics |
|---|---|---|---|---|---|---|---|
| skills | 228 s | 106k / 8k | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| direct | 528 s | 192k / 12k | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

The skills arm realized the TFNE spec through `jaxfne.agent`; its V_m is
bit-equal to the reference on all three arms. The direct arm hand-built a
Configuration and added a drive asymmetry the spec does not declare
(A at 8.0, B at 5.0): 12 spikes per arm against the reference's 8.

## Changes the pilot forced

- Scorer blind spot: the first scorer (structure, execution, observation,
  plasticity) gave the direct arm 1.0 everywhere. Added the `dynamics` class
  (sorted per-neuron spike counts; per-neuron mean V_m at atol 1e-3) and
  re-froze the task set; all other properties are unchanged across the two
  freezes.
- Packet rule: the spec lists every input that differs from library
  defaults; unstated inputs take the default. The pilot packets did not say
  so, so the direct arm's result above predates the rule.
- Isolation: arms run in fresh single-commit repositories, because a clone
  or linked worktree keeps the reference answers in git history and the
  delegate wrapper refuses linked worktrees for writers.

One trial per arm on one task shows the harness works; it is not evidence
about skills versus direct use.
