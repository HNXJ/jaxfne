# problem stack

Open problems only, most urgent first. Older open issues: `artifacts/issue_log/ISSUE_LOG.md`.
Delete a row when it is resolved; the evidence goes in the commit message.

| ID | Severity | Symptom | Evidence | Fix item |
|---|---|---|---|---|
| P-1 | major | `Simulation(poisson_drive={"target": "E"})` drives every neuron while run metadata records `"E"` | `jaxfne/_signals.py:707-726` never reads `target`; `_model_simulate.py:1284` records it | branch `fix-poisson-target` (refuse non-"all"; test, broad gate 0 failed, second review PASS); push pending |
| P-2 | major | `connections(probability=p)` docstring says Bernoulli; the mechanism-aware compiler draws exactly round(p·n_pre·n_post) edges | `jaxfne/_config.py:978` vs `jaxfne/connectivity.py:282-287` | G-2 feature: `connections(sampling="bernoulli")`, docstring corrected with it |
