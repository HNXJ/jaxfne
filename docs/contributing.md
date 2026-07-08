# Contributing

Contributions are welcome. Please:

1. Open an issue for bugs or features on [GitHub](https://github.com/HNXJ/jaxfne)
2. Fork the repository and create a feature branch
3. Add tests and documentation
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## Validation commands

Before opening a pull request that touches docs or package code:

```bash
python3 scripts/evidence_inventory.py
python3 -m compileall -q scripts jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py \
  tests/test_signals_get_v0329.py -q --tb=short
```

Docs-only changes: run `mkdocs build --strict` at minimum. Regenerate showcase
figures with `python3 scripts/generate_showcase_figures.py` when
`docs/guides/showcases.md` changes.

Publication-quality docs status is tracked in
[Documentation quality report](publication/docs_quality_report.md).

## Code style

- Follow PEP 8
- Use type hints
- Add docstrings

## Questions?

Open a [discussion](https://github.com/HNXJ/jaxfne/discussions) on GitHub.
