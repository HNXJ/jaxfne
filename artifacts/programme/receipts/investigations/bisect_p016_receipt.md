# Receipt — NEXT 3 (P-016): bisect of the population-restoring regression pin

Question: which commit moved `test_population_restoring_etude_regression_metrics`
(vector-HDP terminal error 0.0028 vs frozen 0.0296, tol abs 0.02)?

Method: worktree `bis016`, one commit at a time, `python -m pytest`
`tests/test_hdp_population_restoring.py::test_population_restoring_etude_regression_metrics`
with cwd = worktree and `PYTHONPATH` = worktree (driver
`_temp_bis016.py`; each run logs `VERDICT` plus `identity: head=<short>`
and the imported `jaxfne.__file__`, closing the site-packages hole).
Seed fixed in test; Windows CPU float32; ~10 s per run.

Scope: 28 commits touching `jaxfne/_hdp_registrable_kernel.py`,
`jaxfne/hdp_rule.py`, `jaxfne/hdp_network.py`,
`tests/test_hdp_population_restoring.py`. First-parent chain into the
mover (verified adjacent, ancestor exits 0):

| Commit | Verdict | term_vec | Identity |
|---|---|---|---|
| `a0b848d` | GOOD | within pin (value not captured) | head+pkg logged |
| `e348ebf` | GOOD | within pin (value not captured) | head+pkg logged |
| `865e74b` | BAD | 0.002801 vs pin 0.029602 ± 0.02 | head+pkg logged |

A first cross-branch bracket (`51f8c9b` vs `865e74b`, merge-base
`5d679a1`) was built, refuted by review, and replaced with the true
chain above.

Pin freeze: `git diff e348ebf 865e74b --
tests/test_hdp_population_restoring.py
artifacts/etudes/hdp_controllability_reachability/metrics.json` is empty,
so the move is behavior, not a re-pin.

LOCATED CAUSE (hypothesis, corroborated cross-lane): `865e74b`
("0.5.3 item 3 (W1): P-010 chain-noise fix + continuation all-state")
changed the plain-path noise schedule; the realized RNG stream changed,
moving terminal error either way (here 0.0296 → 0.0028, smaller yet out
of pin). The exact stream edit was not verified here.
The same commit moved the mcc3 condition-C timing receipt (P-020).

Gaps: single test seed; Windows CPU float32 only (CI ubuntu not run for
these commits); parent exact values pin-bounded, not captured; no SKIP
band (every in-scope run decided; mover adjacent on first-parent).

Sign-off (unsigned): re-freeze vector terminal to the post-`865e74b`
realization with cause stated above. Human signs here.
