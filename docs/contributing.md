# Contributing to jaxfne

Thank you for helping improve jaxfne. This guide covers setup, validation, and
expectations for human contributors.

## Quick links

- [Changelog](changelog.md)
- [Documentation](https://jaxfne.readthedocs.io/)
- [Scope & status](scope_and_status.md)
- Maintainer agent workflows, skills, and the internal backlog live outside this public documentation tree
- [Code of Conduct](https://github.com/HNXJ/jaxfne/blob/main/.github/CODE_OF_CONDUCT.md)
- [GitHub Issues](https://github.com/HNXJ/jaxfne/issues) · [Discussions](https://github.com/HNXJ/jaxfne/discussions)

## Development setup

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e ".[dev,viz]"
```

Use **`python3`** (3.10–3.12 tested). Canonical import: `import jaxfne as jtfne`.

## Before you open a PR

1. **Scope:** Small, focused changes. Preserve public APIs unless the PR explicitly documents a break.
2. **Claims:** Don't overclaim simulated outputs — see [Scope & status](scope_and_status.md).
3. **Visualization:** Simulation-signal plotting belongs in `jaxfne/vis/*` only.
4. **Tests:** Add or extend tests for behavior you change. Do not weaken tests to hide failures.
5. **Docs:** User-facing behavior changes need matching docs in the same PR.
6. **Skills:** If you change agent-facing workflows, update `skills/` and note that in the PR body
   (maintainer-internal; not part of the public documentation tree).

### Validation commands

```bash
python3 scripts/evidence_inventory.py
python3 -m compileall -q scripts jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py \
  tests/test_signals_get_v0329.py -q --tb=short
```

Broader local check (optional, slower):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests -q -m "not slow" --tb=short
```

Docs-only PRs: run `mkdocs build --strict` at minimum. If you change `docs/guides/showcases.md`,
regenerate figures with `python3 scripts/generate_showcase_figures.py`.

## Code style

- PEP 8, type hints where practical, docstrings on public APIs
- JAX: `jax.numpy`, explicit PRNG keys, `lax.scan` for time, `vmap` for batches
- Keep I/O, plotting, and JSON outside `jit`

## Pull request expectations

- Link related issues when applicable
- Describe validation you ran (exact commands + results)
- Keep proxy/scaffold outputs labeled as such (see [Scope & status](scope_and_status.md))

## Questions?

Open a [discussion](https://github.com/HNXJ/jaxfne/discussions) or issue on GitHub.
