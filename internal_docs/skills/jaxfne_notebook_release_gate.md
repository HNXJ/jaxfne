---
summary: Use for notebooks, tutorials, Colab, visualizations, manifests, JSON audits, docs builds, package builds, GitHub releases, TestPyPI/PyPI, and worker release reports.
trigger: Use whenever the task mentions notebook, Colab, tutorial, Etude, Suite, visualization, raster, LFP, CSD, EEG, MEG, PSD, spectrogram, spectrolaminar, activity suite, network3d, Plotly, PNG, manifest, validation_report, metrics, mkdocs, docs, release, tag, PyPI, TestPyPI, twine, build, CI.
---

# jaxfne Notebook Release Gate Skill

## Notebook hard gates

```text
duration_ms >= 1000 in full mode
dt_ms = 0.1
dtype = float32
deterministic seed
finite outputs
strict JSON
PNG core figures required
Plotly HTML optional
proxy-safe titles
SMOKE and FULL receipts for release notebooks
```

## Visualization ownership

`jaxfne.vis` owns reusable plotting:

```text
raster
voltage/source/LFP/CSD/EEG/MEG traces
PSD
spectrogram
spectrolaminar suite
activity suite
connectivity weight maps
network3d
2-photon image proxy
```

Package-level options should replace notebook patches:

```text
dark theme
z flip/range/aspect/camera
figure height/scale
relative_power_mode="per_frequency_depth_max"
trained-vs-initial panels
```

## Relative-power rule

For reference-style spectrolaminar heatmaps with power shape `[freq, contact]`:

```python
relative_power = power / max(power, axis=1, keepdims=True)
```

Frequency remains Hz. Do not normalize the frequency axis unless explicitly requested.

## Release validation

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
PYTHONPATH=. TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_smoke.ipynb
PYTHONPATH=. TFNE_SMOKE=0 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_full.ipynb
python -m json.tool outputs/jaxfne_etude_no_1/manifest.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/validation_report.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/metrics.json >/dev/null
python -m build
python -m twine check dist/*
```

## Package release guard

Before tag/upload:

```text
pyproject.toml version matches intended tag
jaxfne.__version__ matches intended tag
dist filenames contain intended version
main CI is green on release commit
GitHub Release points to same tag/commit
post-install smoke passes
```

## Worker report

```text
repo / branch / SHA
changed files
commands run
exact results
runtime facts
status/evidence level
blockers
next safe action
```
