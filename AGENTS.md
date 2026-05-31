# jaxfne Agent Coordination

Protocol version: 2.0  
truth_mode: truth_safe_unverified  
active release line: v0.3.21+

## Required start checks

```bash
git fetch origin
git status --short
git branch --show-current
git rev-parse HEAD
```

## Branch flow

```text
feature branch -> dev review when used -> main after receipts
```

Do not force-push `main`. Use immutable SHA URLs for final audits.

## Active locks

| Agent | Scope | Since | Status |
|---|---|---|---|
| (none) | No active locks | 2026-05-31 | clear |

## Release-control gates

Each remote mutation needs explicit authorization:

| Gate | Examples |
|---|---|
| branch push | `git push origin <branch>` |
| tag mutation | `git tag`, `git push origin vX.Y.Z` |
| distribution upload | TestPyPI/PyPI twine upload |
| GitHub release | create/edit/publish release |

Before release upload/tag, verify:

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Worker report format

```text
repo / branch / SHA
changed files
commands run
exact results
runtime facts
truth/evidence status
blockers
next safe action
```

## Doctrine pointers

Compact source doctrine lives in:

```text
internal_docs/source_doctrine/
```

Read `internal_docs/source_doctrine/CLAUDE.md` before changing tutorials, Etudes, objectives, visualization, or release scripts.