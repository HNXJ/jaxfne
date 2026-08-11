# jaxfne Visualization and Readout Rules

## 1. Visualization is downstream evidence

Figures visualize package-generated `Model`, `Signals`, diagnostics, objective reports, or manifests. Notebook plotting may arrange panels and annotate results; it must not reimplement scientific simulation, source, field, probe, PSD, objective, or optimization engines when package APIs exist.

## 2. Readout naming

Titles, axes, captions, and legends must preserve the readout's epistemic status.

Preferred forms when uncalibrated:

```text
LFP-like proxy
CSD-like proxy
EEG-like proxy
MEG-like proxy
spectrolaminar proxy
relative source/readout amplitude
```

Do not shorten these to physical measurement names in publication figures unless the relevant calibration/solver validation exists.

## 3. Field operator semantics

A fixed kernel projection

\[
\Phi = KQ
\]

is a linear operator/projection. It is not a PDE solve merely because the map is linear.

An experimental PDE operator should be labeled as such and accompanied by its equation, discretization, boundary/reference conditions, residual, convergence and unit/calibration status.

## 4. Laminar projections

Projection normalization changes the meaning of the readout. Row normalization can remove attenuation information by forcing each contact's weights to sum to a constant. For probe layouts extending outside the modeled population, use a density-/magnitude-preserving operator when attenuation is part of the intended interpretation.

Every publication-facing projection should record:

```text
contact geometry
kernel/operator
normalization mode
source support
readout units/status
field operator/solver status
```

## 5. Spectral and spectrolaminar plots

Use package-native PSD/spectrolaminar computation. Document:

- sampling interval/rate;
- window/segment convention;
- trial and contact axes;
- frequency range;
- normalization/log convention;
- aggregation across trials/seeds;
- layer/contact mapping.

Never infer a laminar mechanism solely from a power heatmap. The figure supports a model/readout result; causal interpretation requires interventions/nulls.

## 6. Network visualization

3D circuit figures should derive positions, layer labels, cell types, and edges from the constructed model/tensor rather than a separately reconstructed drawing.

Interactive HTML is optional. Static PNG is required for release-facing docs/papers because it is stable, hashable, and renderable in archival contexts.

## 7. Figure architecture

For methods/publication figures, prefer panels that expose the operator chain:

```text
A circuit/tensor definition
B dynamics/state
C source construction
D field/probe operator
E readout
F objective/null/quantification
```

This makes assumptions visible and prevents a proxy output from appearing as a direct physical measurement.

## 8. Quantification precedes decoration

Every qualitative publication panel should have a quantitative companion when the claim depends on magnitude, selectivity, stability, frequency, depth, or optimization.

Do not manually draw trends that are not generated from the analyzed arrays. Plotting code must consume the same arrays used for reported statistics.

## 9. Reproducibility

Publication figures require:

```text
frozen repo SHA
configuration/manifest
seed(s)
input artifact hashes
analysis parameters
output PNG hash
script/notebook path
```

A figure regenerated from a different SHA is a new evidence artifact.

## 10. Failure rendering

If a simulation fails validity checks, figures should not hide the failure through clipping, normalization, smoothing, or axis limits. NaN/Inf, implausible state excursions, missing channels, or failed solver residuals must terminate or visibly invalidate the figure-generation path.

## 11. Recommended stable visual families

- circuit/tensor structure;
- raster/rate/state traces;
- source traces/maps;
- laminar field/readout proxy;
- PSD/spectrolaminar summaries;
- objective component traces;
- optimizer convergence/search diagnostics;
- HDP H/weight/rate trajectories;
- null/ablation comparisons;
- validation/convergence plots for experimental solvers.
