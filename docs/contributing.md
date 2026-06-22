# Contributing

Contributions are welcome. Please:

1. Open an issue for bugs or features on [GitHub](https://github.com/HNXJ/jaxfne)
2. Fork the repository and create a feature branch (do not commit directly to `main`, `dev`, `agy`, `cur`, or `ops`)
3. Add tests and documentation for user-facing behavior changes
4. Run the validation commands below
5. Submit a pull request

## Validation commands

Minimal docs and package smoke (from repository root):

```bash
python3 scripts/evidence_inventory.py
python3 -m compileall -q scripts/evidence_figures jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py \
  -q --tb=short
```

For documentation-only changes, `mkdocs build --strict` plus `compileall` on any
touched scripts is sufficient. Use `import jaxfne as jtfne` in all examples.

## Code style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep proxy-readout language in docs (no calibrated EEG/MEG or mechanism claims without evidence)

## Documentation

- User-facing pages live under `docs/` and are built with MkDocs (`mkdocs.yml`)
- Theory and scope pages: [Mathematical Glossary Flow](mathematical_glossary_flow.md), [Limitations](limitations_and_future_plans.md)
- Publication audit: [Docs Quality Report](publication/docs_quality_report.md)

## Questions?

Open a [discussion](https://github.com/HNXJ/jaxfne/discussions) on GitHub.
