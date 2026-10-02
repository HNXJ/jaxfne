# 5d benchmark results: skill+tools vs direct repository use (2026-09-26)

Task set `agent_tasks_055.json` (11 Atlas tasks, 30 arms), base commit
`aa36312`, executor opencode `muse-spark-1.3-contributor` (code tier), 3 trials
per task per arm, 66 runs. Each run used a fresh single-commit repository with
the reference answers removed; the direct arm also lacked `artifacts/skills`
and `jaxfne/agent.py`. All 66 runs produced a scorable solution; no source-repo
violation. Two runs left a stray check script beside the solution (AT-01
skills t1, AT-02 direct t3); nothing else changed. Per-run data:
`results_055.json`.

Scores are the fraction of frozen properties reproduced (mean over runs).

| Arm | model | execution | observation | plasticity | dynamics | runs with every class 1.0 | median s | median tokens in |
|---|---|---|---|---|---|---|---|---|
| skills | 0.95 | 1.00 | 1.00 | 1.00 | 0.87 | 23/33 | 251 | 108k |
| direct | 0.85 | 1.00 | 1.00 | 1.00 | 0.69 | 13/33 | 221 | 100k |

Per task (mean class score over trials; runs with every class 1.0):

| Task | skills | direct |
|---|---|---|
| AT-01 | 1.00 (3/3) | 1.00 (3/3) |
| AT-02 | 1.00 (3/3) | 0.92 (2/3) |
| AT-03 | 1.00 (3/3) | 0.91 (1/3) |
| AT-04 | 1.00 (3/3) | 0.76 (0/3) |
| AT-04-R2 | 1.00 (3/3) | 0.90 (0/3) |
| AT-05 | 0.97 (1/3) | 1.00 (3/3) |
| AT-06 | 0.91 (1/3) | 0.88 (1/3) |
| AT-07 | 0.92 (0/3) | 0.94 (0/3) |
| AT-08 | 0.82 (1/3) | 0.72 (0/3) |
| AT-09 | 0.98 (2/3) | 0.93 (0/3) |
| AT-10 | 1.00 (3/3) | 1.00 (3/3) |

## Reading

- Observed: the skills arm reproduced the intended model more often: 23/33
  runs had every class right, against 13/33. The gap is in model structure
  (0.95 vs 0.85) and dynamics (0.87 vs 0.69). Execution, observation and
  plasticity were 1.00 in both arms.
- Observed: most direct-arm structural failures come from building a
  different network: wrong edge counts, weights, mechanisms and delays on
  AT-02 t3, AT-04 (6 of 9 arm-runs) and AT-08 (all 9). The skills arm failed
  structure only on AT-06 (weights, 2 runs) and AT-08 (6 arm-runs).
- Inferred, not established: per task, skills had more perfect runs on 6
  tasks, direct on 1 (AT-05), and 4 tied. A two-sided sign test over those
  7 untied tasks gives p = 0.125. The direction favours skills, but 11
  tasks × 3 trials do not establish it at 0.05.
- Cost: similar. The median skills run took 13% longer and used 7% more
  input tokens.

## Limitations

- P-013: tasks name arms without defining them, so multi-arm tasks (AT-07
  `repro`/`fixed`) lose points in both arms for guessed arm semantics.
- One executor model. Nothing here says how a stronger agent behaves.
- Properties are permutation-invariant summaries plus dynamics at atol 1e-3.
  A model can differ in ways they do not see (for example spatial
  geometry that changes no recorded quantity).
- Pilot record: `pilot_055.md` (that run predates the library-default rule
  in the packet and the dynamics class).
