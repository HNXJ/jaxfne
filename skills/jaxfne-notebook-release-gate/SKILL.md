---
name: jaxfne-notebook-release-gate
description: >-
  Validate jaxfne notebooks, tutorials, documentation, manifests, figures,
  and release-facing artifacts. Use when a task executes or changes a
  notebook, tutorial, docs page, or artifact bundle.
---

# jaxfne notebook and artifact procedure

This is a validation procedure. Tutorial-specific duration, timestep, and
artifact requirements come from that tutorial's contract; do not apply a
release-only gate to every notebook.

## Required execution evidence

For an executed notebook, record:

- checkout path and package import path;
- explicit seed, dtype, duration, and timestep;
- package-native construction and simulation path;
- finite outputs;
- strict JSON/manifests;
- expected figure artifacts;
- status metadata;
- smoke/full scope.

Use `jaxfne.__file__` as the first diagnostic line when executing manually.
For `nbclient`, use an explicit repository kernel or the portable helper built
from `sys.executable`; never trust an unnamed machine default kernel.

## Public documentation

README and docs should contain neutral mathematical/scientific content:
objects, equations, contracts, inputs/outputs, assumptions, units/status,
configuration, execution, validation, and examples.

Do not add agent instructions, development history, bug ledgers, roadmap
updates, promotional prose, or defensive commentary to public docs.

## Artifact checks

- JSON uses finite values and `allow_nan=False`.
- Figures are present and non-empty when required by the artifact contract.
- Manifests record configuration, runtime, provenance, and status.
- Hashes establish content identity, not scientific truth.

## Validation commands

Use the smallest relevant gate:

```bash
python3 -m compileall -q jaxfne tests scripts
python3 scripts/audit_public_docs_language.py --check
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest <targeted-tests> -q
```

Use the repository's notebook execution tests for full notebook validation.
Do not claim full notebook or release validation when it was not run.

## Stop conditions

Stop when execution uses the wrong package path, hides errors, emits NaN/Inf,
creates an unlabelled proxy/Relative output, or reports success without a
command receipt.
