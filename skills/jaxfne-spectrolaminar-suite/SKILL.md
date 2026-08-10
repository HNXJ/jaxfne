---
name: jaxfne-spectrolaminar-suite
description: >-
  Run and audit jaxfne depth-by-frequency and laminar readout workflows. Use
  for spectrolaminar suites, multi-trial laminar signals, and their figures.
---

# jaxfne spectrolaminar procedure

Read `catalog-glossary-jaxfne`, `jaxfne-config`, and `jaxfne-vis-modules`.
Spectral and field meaning belongs to the project source documents; this skill
only routes executable procedures.

## Choose the pipeline

- Single-run or scalable model: use the `Configuration` path, an edge-list
  backend when appropriate, `construct()`, and top-level `simulate()`.
- Multi-trial tutorial sweep: use `tutorial_utils.make_laminar_column_config`,
  `tutorial_utils.build_laminar_column`, and
  `tutorial_utils.simulate_laminar_trials`.

Do not mix the model object contract with the tutorial model-dictionary
contract.

## Readout procedure

1. Identify the signal key and its shape.
2. Record contact depths, sampling interval, and frequency grid.
3. Use package spectral/readout helpers.
4. Preserve Relative power normalization and state the normalization mode.
5. Use `jaxfne.vis.spectrolaminar` or `spectrolaminar_suite` for reusable
   signal-driven figures.
6. Record kappa synchrony and other diagnostics when the source specification
   requires them.

LFP, CSD, EEG, and MEG outputs remain proxy/readout channels unless the current
status contract and an external calibration/solver receipt say otherwise.

## Scale and memory

Inspect the selected connectivity backend and array shapes before large runs.
Do not claim a performance regime from a skill file. Measure construction,
simulation, memory, and compilation for the current checkout and configuration.

## Evidence

A valid suite receipt includes the package path, configuration, seed, runtime
state, signal shapes, frequency/depth axes, finite checks, status metadata,
figure paths, and hashes when artifacts are retained.

Run the relevant targeted tests and notebook gate. Interpret crossover or motif
results only after the corresponding execution receipt exists.
