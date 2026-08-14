# Protocol C3 — prospective neural geometry/delay experiment (0.4.17-C)

**Status:** C3 **CLOSED** — prospective run executed; C4 interpretation frozen  
**Spec:** `artifacts/protocol_c/c3_neural_experiment_spec.json`  
**Execution:** `artifacts/protocol_c/c3_execution_receipt.json` (60 cells)  
**Interpretation:** `artifacts/protocol_c/c4_interpretation_receipt.json` (outcome **C**)  
**Prerequisites:** C1 estimator validated; C2 `delay_state` continuation validated @ `4179660`

## Causal chain (frozen)

\[
\boxed{
\mathcal G\rightarrow(W,\tau)\rightarrow X(\mathbf r,t)\rightarrow\widehat{\mathcal W}
}
\]

\(Q(\mathbf r,t)\) and \(\Phi(\mathbf r,t)\) are **secondary only** where existing mappings are already frozen. Nothing from \(\widehat{\mathcal W}\) feeds upstream.

\[
\boxed{
\text{Protocol D asks how signals are delayed;}
\qquad
\text{Protocol C asks whether resulting activity exhibits propagation.}
}
\]

## Runtime (frozen)

| Setting | Value |
|---------|--------|
| HDP | **off** |
| Homeostasis | **off** |
| RBD | **off** (baseline recurrent) |
| Delay continuation | C2 `delay_state` path if segmented |

C2 did **not** validate HDP+delay continuation; C3 must not enable HDP.

## Design

**Factors**

| Factor | Levels |
|--------|--------|
| Geometry layout | `ordered`, `shuffled` (Fisher–Yates position assignment) |
| Delay policy | `uniform`, `geometry_derived`, `delay_shuffled` |

**Six preregistered conditions**

1. `ordered_uniform`
2. `ordered_geometry_derived`
3. `ordered_delay_shuffled` — same delay **multiset** as (2), permuted across edges
4. `shuffled_uniform`
5. `shuffled_geometry_derived`
6. `shuffled_delay_shuffled`

Delay shuffle preserves the exact delay multiset while breaking geometry–delay correspondence.

## Frozen construction rules

- **Topology:** directed one-neighbor ring, \(N=24\), \(W_{ij}=6\), \(\tau_{\mathrm{syn}}=3\,\mathrm{ms}\)
- **Positions:** ring radius \(R=1\,\mathrm{mm}\); estimator uses arc-length coordinate \((N,1)\)
- **Uniform delays:** `delay_steps = 4` at `dt_ms = 0.5`
- **Geometry-derived:** \(\tau_{ij} = d_{ij}/v_c\) with grid-aligned `delay_steps`; neighbor arc
  \(d_{ij} = 2\pi R/N\); frozen \(v_c = (2\pi R/N)/(4\cdot dt_{\mathrm{ms}})\)
- **Seeds:** `1001`–`1010` (10 seeds × 6 conditions = 60 cells)
- **Duration:** `2000 ms`, `dt_ms = 0.5`, deterministic (`noise_scale = 0`)

## Estimator binding

Frozen C1 implementation (`estimate_traveling_wave` / `WaveEstimate`) with C0 band **8–13 Hz** and thresholds unchanged.

Per seed/condition output:

\[
(\mathrm{class}, f, \mathbf k, \hat{\mathbf d}, v_{\mathrm{phase}}, R^2, C_{\mathrm{spatial}},
\text{null score}, \text{reasons})
\]

\[
\mathrm{class}\in\{\texttt{TRAVELING\_WAVE},\texttt{NO\_WAVE},\texttt{UNRESOLVED}\}
\]

Do **not** collapse `UNRESOLVED` into `NO_WAVE`.

## Population endpoints

**Primary:** \(p_W = N_{\mathrm{TW}}/N_{\mathrm{total}}\)

**Identifiability:** \(p_U = N_{\mathrm{unresolved}}/N_{\mathrm{total}}\)

Velocity/direction summaries are **conditional** on `TRAVELING_WAVE` and must not hide nondetections.

## Directional conjecture (optional, not required for success)

\[
p_W(\text{ordered, geometry-derived}) > p_W(\text{ordered, uniform})
\]

C3 **success does not require** this inequality.

## \(v_c\) diagnostic

When `geometry_derived` and `TRAVELING_WAVE`, report \(| \hat v_{\mathrm{phase}} - v_c |/v_c\) **diagnostically only**. Network phase velocity need not equal axonal conduction velocity.

## Explicit prohibitions

- No HDP, W3/W3c, or ephaptic/field feedback
- No post-result tuning of delays, geometry, band, duration, or estimator gates
- No H4 \(\mathcal M_X\) as wave evidence
- No claim that heterogeneous delays alone imply propagation
- No claim that a detected wave is field-mediated

## Checkpoints

| ID | Status |
|----|--------|
| C3 | Specification frozen; prospective run **executed** (60 cells) |
| C4 | Frozen prospective receipt + interpretation (**outcome C**) |
