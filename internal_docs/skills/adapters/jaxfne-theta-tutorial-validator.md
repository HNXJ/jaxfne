# jaxfne-theta-tutorial-validator

**Triggers:** tutorial PR, theta gate, repo_smoke, etude smoke, full tutorial validation.

**Purpose:** Combined execution + validation gate for tutorials before merge or release.

**Modes:** `repo_smoke`, `one_tutorial`, `etude_smoke`, `full_optional` (slow).

**Full skill:** user-installed `jaxfne-theta-tutorial-validator` (may include `validator.py` in skill dir). Composes api-truth, jax-lint, style-conformance, test-runner, tutorial-executor, evidence-validator.
