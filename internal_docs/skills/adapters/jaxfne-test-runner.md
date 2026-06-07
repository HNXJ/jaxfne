# jaxfne-test-runner

**Triggers:** pytest, test failure, suite, run tests.

**Purpose:** Run jaxfne tests with the correct environment; report real pass counts.

**Required env:**

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.
python -m pytest tests/ -q --tb=short
```

Run targeted tests first when touching a specific module.

**Full skill:** user-installed `jaxfne-test-runner`. Pair with `jaxfne-repo-audit` before merge.
