## Question

Does the jaxfne core reproduce the laminar multi-area circuit of Mejias, Murray,
Kennedy and Wang (2016, Sci Adv 2(11):e1601335): gamma (~40 Hz) in supragranular
and alpha (~10 Hz) in infragranular Wilson-Cowan modules, their interlaminar
coupling (gamma power entrained to alpha phase, alpha tied to supragranular
suppression), and the 30-area functional hierarchy from frequency-dependent
Granger causality (FF gamma, FB alpha; SLN correlated with DAI/mDAI)?

## Model summary

A laminar network of 30 macaque visual cortical areas on tract-tracing anatomy
(directed FLN weights, SLN laminar patterns, wiring-delay propagation; graph
density about 66%). Each area holds two Wilson-Cowan excitatory-inhibitory
modules: L2/3 with noise-driven gamma and L5/6 with alpha/low-beta, coupled by
L2/3E-to-L5E excitation and L5E-to-L2/3I inhibition. Feedforward projections run
L2/3E to L2/3E; feedback runs L5/6E to all four target populations, weighted
toward infragranular. Stimulus currents used: I = 8 background with I = 8 to V1
L2/3E for the full network; laminar-specific inputs (specificity S up to 8) move
areas in the functional hierarchy.

## Source record

`artifacts/etudes/papers/mejias2016/SOURCE_P1.md` (P1): equations in symbols,
the target-figure parameter set (I values, bands 30-70 and 8-15 Hz, 30 named
areas, SLN 1/0 scale, DAI/mDAI definitions), the Fig. 2, 3, 6 and 7 claims with
the paper's numbers, and the gaps (no rate emitter, no Granger estimator,
unreachable supplement, login-walled anatomy files).

## Status

P1: source record only, no scored run. Next per rule B8: pilot, then signature, then fresh seeds.
