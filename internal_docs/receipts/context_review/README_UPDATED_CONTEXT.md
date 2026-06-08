# Updated jaxfne Context Files Bundle

This bundle replaces the active Gemini/Claude context files with a smaller, more invokable set.

## Main changes

- Updated stale v0.3.5-era context to 0.3.28+ architecture.
- Canonicalized `Config`, `Net`, and `FlatNet` names.
- Kept `Configuration`, `Model`, and `FlatModel` as migration aliases.
- Replaced passive six-skill set with three high-trigger super-skills.
- Added Gemini-specific context file.
- Added schema/version/JIT/release safeguards.
- Preserved truth/status gates.

## How to apply

Copy files into the repo root, preserving paths. Archive old narrow skill files instead of deleting them if you want history.

```bash
cp -R jaxfne_context_files_updated/* /path/to/jaxfne/
```

Then run:

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```
