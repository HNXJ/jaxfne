---
name: jaxfne-vis-modules
description: >-
  Use package-level jaxfne.vis functions for signal-driven raster, trace,
  field, spectral, spectrolaminar, connectivity, 3D, and artifact plotting.
  Use when changing or auditing visualization behavior.
---

# jaxfne visualization procedure

Read `catalog-glossary-jaxfne` and `jaxfne-neural-network` first. Visualization
is a readout procedure; it must not redefine source or field mathematics.

## Signal-driven path

```python
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=0)
figure = jtfne.vis.lfp(signals)
```

Use the package functions for reusable simulation-output plots:

```text
raster, vm, rate, source, lfp, csd, eeg, meg, emm,
psd, spectrogram, bandpower, connectivity,
laminar_profile, geometry3d, summary, objective_report,
spectrolaminar, spectrolaminar_suite
```

Verify exact signatures and return types against the live package.

## Semantics and labeling

- Keep proxy/readout labels aligned with `docs/scope_and_status.md`.
- Use Relative/Absolute terminology from the public status contract.
- State normalization and projection choices in metadata or figure labels.
- Check time/contact/frequency axes explicitly.
- Do not turn a proxy plot into a physical or calibrated claim.

## Ownership

Reusable simulation-output plotting belongs in `jaxfne/vis/`.
One-off architecture diagrams, receipt layouts, and publication illustrations
may remain in `scripts/evidence_figures/` when they do not define reusable
scientific readout behavior. Route their export through the existing evidence
helpers.

Optional matplotlib and Plotly dependencies remain lazy. Do not import plotting
libraries into numerical kernels.

## Validation

- Test a real `Signals` object or a declared readout contract.
- Verify finite data and expected axes.
- Verify the figure contains the intended data, not only labels.
- Save required artifacts with provenance and hashes.
- Run targeted visualization tests and inspect generated media when applicable.
