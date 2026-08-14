# Protocol D — biological RBS containment (0.4.17-D)

**Status:** D2a **CLOSED** — autonomous \(H_{\mathrm{K}}\) F1 relaxation frozen  
**D0 spec:** `artifacts/protocol_d_biological_rbs/d0_intrinsic_ionic_rbs_spec.json`  
**D1 spec/receipt:** static \(b_{\mathrm{eff}}=H_{\mathrm{K}}b\) expression (closed)  
**D2a spec/receipt:** `d2a_autonomous_h_k_relaxation_spec.json` / `d2a_autonomous_relaxation_receipt.json`  
**D2b:** specified, **not** implemented  
**Prerequisites:** Protocol C closed @ C4; Protocol H closed @ H4; W2 expression frozen; D2a closed

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

## D2a — autonomous \(H_{\mathrm{K}}\) relaxation (executed)

\[
\boxed{
\tau_{\mathrm{K}}\dot H_{\mathrm{K}} = 1 - H_{\mathrm{K}},
\qquad
\kappa_{\mathrm{K}} = 0.
}
\]

Analytic: \(H_{\mathrm{K}}(t) = 1 + [H_{\mathrm{K}}(0)-1]e^{-t/\tau_{\mathrm{K}}}\).

Complete RBD path on top of D1:

\[
\boxed{
H_{\mathrm{K}}(t) \rightarrow b_{\mathrm{eff}}(t) \rightarrow X(t).
}
\]

**Gates passed:** discrete Euler contract; analytic consistency; \(H_{\mathrm{K}}(0)=1\) baseline
invariance (bit-exact classical); positivity; repeatability; \(|V_{\mathrm{ext}}-V_{\mathrm{class}}|\)
decays in tail as \(H_{\mathrm{K}}\rightarrow 1\).

**Coordinate semantics (standardized):** effective **K-associated recovery state** — not “channel
availability” until a realization models availability/inactivation kinetics explicitly.

## D2b — two-timescale activity→\(H_{\mathrm{K}}\) coupling (specified)

\[
\boxed{
\mathbf H_i =
\begin{bmatrix}
H_{A,i} \\
H_{K,i}
\end{bmatrix}
}
\]

**Causal chain:**

\[
\boxed{
S_i \rightarrow H_{A,i} \rightarrow H_{K,i} \rightarrow b_i^{\mathrm{eff}} \rightarrow X_i
}
\]

| Coordinate | Reference | Role |
|------------|-----------|------|
| \(H_{\mathrm{A}}\) | \(0\) | activity-history trace (not an ion) |
| \(H_{\mathrm{K}}\) | \(1\) | effective K-associated recovery state |

**Dynamics (frozen):**

\[
\tau_{\mathrm{A}}\dot H_{\mathrm{A},i} = -H_{\mathrm{A},i} + S_i(t),
\qquad
\tau_{\mathrm{K}}\dot H_{\mathrm{K},i} = (1 - H_{\mathrm{K},i}) + \kappa_{\mathrm{AK}} H_{\mathrm{A},i},
\qquad
\kappa_{\mathrm{AK}} > 0.
\]

**Timescales (frozen, not optimized for D3):** \(\tau_{\mathrm{A}}=25\) ms, \(\tau_{\mathrm{K}}=100\) ms, \(\tau_{\mathrm{A}}<\tau_{\mathrm{K}}\).

**Discrete causal ordering:**

\[
H_{\mathrm{A},n+1} = F_{\mathrm{A}}(H_{\mathrm{A},n}, S_n),
\qquad
H_{\mathrm{K},n+1} = F_{\mathrm{K}}(H_{\mathrm{K},n}, H_{\mathrm{A},n}).
\]

**State-level semantics (not phenotype):** \(S\uparrow \Rightarrow H_{\mathrm{A}}\uparrow \Rightarrow H_{\mathrm{K}}\uparrow\).  
**Not preregistered:** \(H_{\mathrm{K}}\uparrow \Rightarrow\) firing \(\downarrow\).

**Null hierarchy:**

- \(\kappa_{\mathrm{AK}}=0 \Rightarrow\) exact D2a
- \(S=0 \Rightarrow H_{\mathrm{A}}\to 0,\ H_{\mathrm{K}}\to 1\)
- \((H_{\mathrm{A}},H_{\mathrm{K}})=(0,1)\) = RBS reference equilibrium

**Post-stimulus analytic contract:** with \(S=0\), \(H_{\mathrm{A}}(t)=H_{\mathrm{A}}(0)e^{-t/\tau_{\mathrm{A}}}\); \(h_{\mathrm{K}}=H_{\mathrm{K}}-1\) obeys \(\tau_{\mathrm{K}}\dot h_{\mathrm{K}} = -h_{\mathrm{K}} + \kappa_{\mathrm{AK}} H_{\mathrm{A}}(0)e^{-t/\tau_{\mathrm{A}}}\).

**Spec:** `artifacts/protocol_d_biological_rbs/d2b_activity_h_k_coupling_spec.json`  
**Implementation:** not authorized.

## D3 — deferred

Repeated-stimulus fatigue/adaptation/recovery phenotype protocol. Requires D2b
implementation gates to pass first; D2b establishes state-writing only.

## Checkpoint ladder

\[
\boxed{
\begin{aligned}
D0 &: \text{biological RBS specification},\\
D1 &: \text{static typed-coordinate expression},\\
D2\mathrm{a} &: \text{autonomous }H_{\mathrm{K}}\text{ relaxation},\\
D2\mathrm{b} &: \text{activity}\rightarrow H_{\mathrm{K}}\text{ coupling (specified)},\\
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
| D1 | Static \(H_{\mathrm{K}}\) sweep **executed** |
| D2a | Autonomous F1 relaxation **executed** |
| D2b | Two-coordinate \((H_{\mathrm{A}},H_{\mathrm{K}})\) coupling — **specified, not implemented** |
| D3–D4 | Phenotype protocol / optional second class — not authorized |
