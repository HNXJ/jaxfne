# problem stack

Open problems only, most urgent first. Older open issues: `artifacts/issue_log/ISSUE_LOG.md`.
Delete a row when it is resolved; the evidence goes in the commit message.

| ID | Severity | Symptom | Evidence | Fix item |
|---|---|---|---|---|
| P-2 | major | `connections(probability=p)` docstring says Bernoulli; the mechanism-aware compiler draws exactly round(p·n_pre·n_post) edges | `jaxfne/_config.py:978` vs `jaxfne/connectivity.py:282-287` | T-1 (G-2 F1: `connections(sampling="bernoulli")`, docstring corrected with it) |
