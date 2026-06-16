# jaxfne-repo-audit

**Triggers:** audit, health, readiness, status, before commit, before merge, before release.

**Purpose:** PASS/FAIL snapshot of branch, version comparison, compile, pytest, and secret scan.

**Core commands:**

```bash
git status --short --branch
git log --oneline -3
python -m compileall jaxfne -q
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.
python -m pytest tests/ -q --tb=line
```

**Full skill:** user-installed `jaxfne-repo-audit`.
