# Protocol C — wave inference (0.4.17-C)

**Status:** C0 frozen (specification only)  
**Spec:** `artifacts/protocol_c/c0_wave_protocol_spec.json`  
**Prerequisite:** Experiment A closed (`experiment_a_v0417_b`); Protocol D delay dynamics frozen separately

## Scope separation (frozen)

| Symbol | Domain | 0.4.17-C |
|--------|--------|----------|
| \(C_D\) | Delay dynamics | **Protocol D** — out of scope for C implementation |
| \(C_W\) | Wave inference | **In scope** — estimator \(\widehat{\mathcal W}\) |
| \(C_F\) | Field feedback | **Closed** — no ephaptic upstream coupling |

## Causal hierarchy (frozen)

\[
\mathcal G \rightarrow (W,\tau) \rightarrow X(\mathbf r,t) \rightarrow Q(\mathbf r,t) \rightarrow \Phi(\mathbf r,t) \rightarrow \widehat{\mathcal W}
\]

Nothing computed by the estimator may feed upstream.

## Operational wave definition

A traveling wave requires **more than phase differences**. Preregistered elements:

- frequency band
- phase extraction method
- spatial coordinates
- minimum spatial coherence
- phase-gradient fit
- propagation direction convention
- phase velocity convention
- fit quality (\(R^2_{\mathrm{phase}}\))

## Classification (three-way)

\[
\texttt{classification} \in \{\texttt{TRAVELING\_WAVE},\texttt{NO\_WAVE},\texttt{UNRESOLVED}\}
\]

- **NO_WAVE** — valid negative outcome under frozen gates.
- **UNRESOLVED** — poor identifiability; must not be collapsed into NO_WAVE (W3b lesson).

Per condition report:

\[
(\text{classification}, f, \mathbf k, v_{\mathrm{phase}}, R^2_{\mathrm{phase}}, \text{coherence}, \text{uncertainty/null score})
\]

## Synthetic controls (frozen at C0)

**Positive:** \(\phi(\mathbf r,t)=A\cos(\mathbf k^\top\mathbf r-\omega t+\phi_0)\) with known \(\mathbf k\), \(f=\omega/2\pi\), \(v_{\mathrm{phase}}=\omega/|\mathbf k|\).

**Negative:** synchronous (\(\mathbf k=0\)), spatially random phase, standing-wave-like structure, noise-only.

False positives matter more than a visually impressive positive example.

## Checkpoints

| ID | Deliverable |
|----|-------------|
| C0 | This specification (no estimator code) |
| C1 | Estimator validation on synthetic fields (`c1_synthetic_validation_receipt.json`) |
| C2 | General `Model.simulate` `delay_state` continuation |
| C3 | Prospective neural geometry/delay experiment (spec frozen; run not authorized) |
| C4 | Frozen receipt before Figure 5 interpretation |

## Experiment A integration

The estimator may consume appropriately structured \(X\), \(Q\), or \(\Phi\) from Experiment A, but **must not** force the 40-neuron Experiment A dataset to produce a wave.

## Successful no-wave outcome (C4)

\[
\boxed{\text{validated estimator} + \text{prospective neural no-wave result}}
\]

Do not retune delays, bands, geometry, thresholds, or duration after the prospective run.

## Deferred

- Manuscript figure generation (after B and C frozen)
- JAX 0.11 compatibility (operational debt; independent branch)
