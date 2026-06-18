---
name: jaxfne-notebook-release-gate
summary: Validate jaxfne notebooks, tutorials, docs, artifacts, JSON manifests, tests, package builds, and worker reports before declaring success.
trigger: Use whenever the task mentions notebook, tutorial, Colab, markdown, docs, mkdocs, artifact, manifest, validation_report, metrics, figures, PNG, Plotly, smoke, full execution, pytest, audit, release, version, build, twine, TestPyPI, PyPI, GitHub Release, or worker report.
---

# jaxfne Notebook Release Gate

## Purpose

Use this before reporting success for notebook/tutorial/docs/release tasks.

## Tutorial hard gates

```text
duration_ms >= 1000 in full mode
dt_ms = 0.1
dtype = float32
seed deterministic
finite outputs
strict JSON
PNG figures present
canonical import: import jaxfne as jtfne
package-native engine path
proxy-safe figure titles
SMOKE execution receipt
FULL execution receipt for release-facing changes
```

## Required tutorial sections

Keep public prose technical and concise. Required material:

```text
learning objectives
question/scope
configuration
simulation
probe/readout
figures
interpretation
failure modes
exercises
run/status metadata
```

Avoid public overclaim wording. Machine-readable gates may still exist in manifests.

## Validation commands

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
PYTHONPATH=. TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_smoke.ipynb
```

For release-facing notebook changes:

```bash
PYTHONPATH=. TFNE_SMOKE=0 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_full.ipynb
```

JSON checks:

```bash
python -m json.tool outputs/jaxfne_etude_no_1/manifest.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/validation_report.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/metrics.json >/dev/null
```

Build checks:

```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

## Release version rule

PyPI version must match the source package version. Do not publish from a commit whose `pyproject.toml` or `jaxfne.__version__` is stale.

## Artifact checks

```text
manifest.json: strict JSON, finite values, unified cfg block
validation_report.json: finite outputs, strict_json_pass, png_figures_present, status metadata
metrics.json: objective/trainer metrics if training exists
figures/*.png: present and non-empty
plotly/*.html: optional, if available
```

## Worker report template

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

## Stop conditions

```text
notebook passes only because errors are hidden
artifact JSON contains NaN/Inf
figure artifacts are missing
full-mode duration below gate
package version mismatch before PyPI release
main/tag/CI SHA mismatch
worker reports success without exact command receipts
```
