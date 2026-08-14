# Protocol D — biological RBS containment (0.4.17-D)

**Status:** D1 **CLOSED** — static \(H_{\mathrm{K}}\) expression frozen @ `b_{\mathrm{eff}}=H_{\mathrm{K}}b`  
**D0 spec:** `artifacts/protocol_d_biological_rbs/d0_intrinsic_ionic_rbs_spec.json`  
**D1 spec:** `artifacts/protocol_d_biological_rbs/d1_static_h_k_expression_spec.json`  
**D1 receipt:** `artifacts/protocol_d_biological_rbs/d1_static_expression_receipt.json`  
**Prerequisites:** Protocol C closed @ C4; Protocol H closed @ H4; W2 expression frozen

> **Naming:** This is **0.4.17-D biological RBS**, distinct from **0.4.16 edge-delay
> Protocol D₀/D₁** (`tests/test_edge_delay_protocol_d016.py`). Do not merge receipts
> or manuscript panels.

## Central question

\[
\boxed{
\text{Can one biologically interpretable RBS realization extend an existing emitter while recovering the classical emitter when RBS is frozen?}
}
\]

Containment thesis:

\[
\boxed{
\text{classical emitter}
\subset
\text{RBS-extended emitter}.
}
\]

## Causal chain (frozen)

\[
\mathbf H_{\mathrm{ion}}
\xrightarrow{\Gamma(H)}
E
\xrightarrow{}
X
\quad\text{(phenotype observables)}
\]

\[
\dot W = 0
\qquad
\text{(RBD only; HDP out of scope for 0.4.17-D)}
\]

## Ionic vector grammar (typed; one coordinate in D1)

\[
\mathbf H_{\mathrm{ion}} = (H_{\mathrm{Na}}, H_{\mathrm{K}}, H_{\mathrm{Ca}})
\]

| Coordinate | D1 scope |
|------------|----------|
| \(H_{\mathrm{Na}}\) | Reserved — not implemented in D1 |
| \(H_{\mathrm{K}}\) | **First biological realization** |
| \(H_{\mathrm{Ca}}\) | Reserved — not implemented in D1 |

**Resolution principle (frozen):**

\[
\boxed{
\text{RBS coordinate}
\neq
\text{claim of microscopic physical completeness.}
}
\]

An effective \(H_{\mathrm{K}}\) may later refine into
\((H_{g_K}, H_{[K]_i}, H_{[K]_o}, H_{\mathrm{availability}}, \ldots)\) without
changing the grammar.

## First coordinate: \(H_{\mathrm{K}}\)

**Physical typing (frozen):** effective **K-associated recovery/adaptation contribution**
in the reduced Izhikevich realization — not literal \(g_K\), not concentration.

Four separable specification layers:

| Layer | Content |
|-------|---------|
| **Physical interpretation** | Effective K-associated recovery/adaptation contribution; \(H_{\mathrm{K}}=1\) = nominal |
| **\(\Gamma(H)\) grammar** | \(g_{\mathrm{K}}^{\mathrm{eff}} = H_{\mathrm{K}}\, g_{\mathrm{K}}^{0}\) (containment-layer typed grammar) |
| **Izhikevich realization (D1)** | \(b_{\mathrm{eff}} = H_{\mathrm{K}}\, b\); \(\dot u = a(b_{\mathrm{eff}} v - u)\) |
| **\(F_H\)** | Deferred to D2: \(\tau_{\mathrm{K}}\dot H_{\mathrm{K}} = R_{\mathrm{K}}(H_{\mathrm{K}}) + \text{coupling}\) |
| **Null / reference** | \(H_{\mathrm{K}}\equiv 1,\ \dot H_{\mathrm{K}}=0 \Rightarrow E_{\mathrm{extended}}=E_{\mathrm{classical}}\) (bit-exact on \(V\), spikes) |

Local mathematical receipt:

\[
\frac{\partial \dot u}{\partial H_{\mathrm{K}}} = a\, b\, v.
\]

> \(H_{\mathrm{K}}\) is an effective relative K-associated recovery coordinate in the
> Izhikevich realization. It demonstrates typed RBS containment of a reduced biophysical
> dependency; it is not an explicit potassium concentration, Nernst potential, or
> Hodgkin–Huxley potassium-conductance model.

**What \(H_{\mathrm{K}}\) is not:** potassium concentration, Nernst potential, STDP,
or neurotransmitter availability.

## Static sweep (D1 — executed)

Expression criterion (direction **not** preregistered):

\[
H_{\mathrm{K}} \neq 1 \Rightarrow X(H_{\mathrm{K}}) \neq X(1).
\]

Frozen levels: \(H_{\mathrm{K}} \in \{0.8,\ 1.0,\ 1.2\}\), \(\dot H_{\mathrm{K}}=0\).

**D1 gates (all passed):** G1 containment (bit-exact classical at \(H_{\mathrm{K}}=1\));
G2 static-state integrity; G3 parameter locality; G4 bidirectional evaluation of all three levels.

Reported phenotypes vs \(H_{\mathrm{K}}=1\): \(\Delta V(t)\), \(\Delta u(t)\), \(\Delta N_{\mathrm{spike}}\),
\(\Delta t_{\mathrm{spike}}\) (primary direct map: \(u\)).

## Checkpoint ladder

\[
\boxed{
\begin{aligned}
D0 &: \text{biological RBS specification (this document)},\\
D1 &: \text{static typed-coordinate expression},\\
D2 &: \text{dynamic }F_H\text{ + recovery/stability},\\
D3 &: \text{biological phenotype protocol},\\
D4 &: \text{optional second RBS class (not mandatory for 0.4.17)}.
\end{aligned}}
\]

## Explicit prohibitions

- No HDP (\(\dot W \neq 0\))
- No STDP or neurotransmitter coordinate in D1
- No concentration/Nernst claims for \(H_{\mathrm{K}}\)
- No implementing all three ionic coordinates in D1
- No post-hoc retuning of \(\delta\), \(\Gamma\), \(F_H\), or drive after observation

## Manuscript discipline (Figure 5/6)

| Protocol | Panel role | Claim boundary |
|----------|------------|----------------|
| **C3** | Figure 5 | Tested ring/delay regimes → sufficient-quality oscillatory activity, **no** estimator-supported traveling waves; does **not** generalize to absence of waves in TFNE |
| **H4** | Figure 6 | Topology/delay memory extension **negative** |
| **D** | Figure 6 (planned) | Biological RBS containment under \(\dot W=0\) |

C3 and H4 are joint examples of **falsification discipline** but answer different
questions — keep separate panels and narrative.

## Checkpoints

| ID | Status |
|----|--------|
| D0 | Specification frozen |
| D1 | Static \(H_{\mathrm{K}}\) sweep **executed** (`b_{\mathrm{eff}}=H_{\mathrm{K}}b`) |
| D2–D4 | Specified; **not** authorized |
