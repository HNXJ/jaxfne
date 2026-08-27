# Biological RBS Containment

*Protocol D, 0.4.17-D — biological RBS containment. Biological RBS realization; protocol identifier in provenance.*

**Status:** **0.4.17-D CLOSED @ D3** — biological RBS containment complete; D4 not authorized  
**D0 spec:** `artifacts/protocol_d_biological_rbs/d0_intrinsic_ionic_rbs_spec.json`  
**D1 spec/receipt:** static \(b_{\mathrm{eff}}=H_{\mathrm{K}}b\) expression (closed)  
**D2a spec/receipt:** `d2a_autonomous_h_k_relaxation_spec.json` / `d2a_autonomous_relaxation_receipt.json`  
**D2b spec/receipt:** `d2b_activity_h_k_coupling_spec.json` / `d2b_implementation_receipt.json`  
**D3 spec/receipts:** `d3_adaptation_recovery_phenotype_spec.json` /
`d3_execution_receipt.json` / `d3_interpretation_receipt.json`  
**Closure:** `d_closure_interpretation_receipt.json`  
**Next milestone:** 0.4.17-E (E0 composition spec frozen)  

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

## D2b — two-timescale activity→\(H_{\mathrm{K}}\) coupling (executed)

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

**Activity input (frozen):** \(S_n\in\{0,1\}\) — binary per-neuron spike indicator per Euler step (timestep-independent unit event; not scaled by \(\Delta t\)).

**Spec:** `artifacts/protocol_d_biological_rbs/d2b_activity_h_k_coupling_spec.json`  
**Receipt:** `artifacts/protocol_d_biological_rbs/d2b_implementation_receipt.json`

**Gates passed:** activity writing; causal \(H_{\mathrm{A}}\to H_{\mathrm{K}}\) transfer (one-step lag);
reference recovery \((H_{\mathrm{A}},H_{\mathrm{K}})\to(0,1)\); \(\kappa_{\mathrm{AK}}=0\) D2a reduction;
admissibility \(H_{\mathrm{K}}>0\); post-stimulus analytic/discrete two-timescale recovery;
\(W(t)=W(0)\). Node-local diagnostic: spikes write local \(H_{\mathrm{A},i}\) only.

**Outputs:** first-class traces \(H_{\mathrm{trace}}=[H_{\mathrm{A}}(t),H_{\mathrm{K}}(t)]\) (not collapsed to \(b_{\mathrm{eff}}\) alone).

**Scientific contract ends at state-space writing** — adaptation/recovery phenotype is deferred to D3.

## D3 — activity-dependent adaptation with recovery (executed)

\[
\boxed{
\text{baseline}
\rightarrow
\text{repeated stimulation}
\rightarrow
\text{recovery interval}
\rightarrow
\text{rechallenge}
}
\]

**Phenomenon (frozen label):** activity-dependent **adaptation with recovery** — not
“fatigue” as a formal microscopic mechanism.

**Pulse train:** \(m=6\) identical pulses (\(u_1=\cdots=u_m\)); amplitude \(15\),
duration \(40\) ms, onset-to-onset ISI \(60\) ms; no parameter changes across repetitions
or rechallenge.

**Primary response:** \(R_j = N_{\mathrm{spike}}^{(j)}\) in an \(80\) ms post-onset window.
Secondary: voltage integral, first-spike latency.

**Adaptation index (prospective):**

\[
A_{\mathrm{adapt}} = 1 - \frac{\overline R_{\mathrm{late}}}{\overline R_{\mathrm{early}}},
\qquad
\overline R_{\mathrm{early}}=\mathrm{mean}(R_1,R_2),
\qquad
\overline R_{\mathrm{late}}=\mathrm{mean}(R_5,R_6),
\]

defined when \(\overline R_{\mathrm{early}}>0\). Negative values (facilitation) are retained.

**Recovery index (secondary):**

\[
R_{\mathrm{recovery}}=
\frac{R_{\mathrm{rechallenge}}-\overline R_{\mathrm{late}}}
{\overline R_{\mathrm{early}}-\overline R_{\mathrm{late}}}.
\]

**Recovery intervals (from D2b timescales, not tuned post hoc):** short \(50\) ms
(\(2\tau_{\mathrm{A}}\)), medium \(100\) ms (\(\tau_{\mathrm{K}}\)), long \(250\) ms
(\(2.5\,\tau_{\mathrm{K}}\)).

**Null hierarchy:**

| Arm | Realization |
|-----|-------------|
| N0 | classical emitter / RBS off |
| N1 | static \(H_{\mathrm{K}}=1\) |
| N2 | \(\kappa_{\mathrm{AK}}=0\) (D2a dynamics) |
| D | full D2b \(H_{\mathrm{A}}\to H_{\mathrm{K}}\) |

Primary contrast: **D − N2** (dynamic \(H_{\mathrm{K}}\) restoration in both; only D
allows activity history to write \(H_{\mathrm{K}}\)).

**Hidden-state mechanism checks (D arm; not sufficient for adaptation):** \(\overline H_{\mathrm{A}}^{\mathrm{late}}>\overline H_{\mathrm{A}}^{\mathrm{baseline}}\),
\(\overline H_{\mathrm{K}}^{\mathrm{late}}>1\); recovery \((H_{\mathrm{A}},H_{\mathrm{K}})\to(0,1)\).

**Classification (three-way):** `ADAPTATION`, `NO_ADAPTATION`, `UNRESOLVED`.
`ADAPTATION` requires \(A_{\mathrm{adapt}}>\theta_A\), \(H_{\mathrm{K}}^{\mathrm{late}}>1+\theta_H\),
and sufficient early signal (\(\overline R_{\mathrm{early}}\ge 1\)). `NO_ADAPTATION` is a
valid outcome (including \(A_{\mathrm{adapt}}\le 0\) with adequate signal). `UNRESOLVED`
when early response is too sparse — silence is not adaptation.

**Frozen thresholds:** \(\theta_A=0.15\), \(\theta_H=0.01\).

**Response-window semantics (frozen):** 80 ms post-onset windows overlap the next
pulse (60 ms ISI) by 20 ms for train pulses 1–5; \(R_j\) is **not** an isolated
single-pulse measure.

**Execution:** 36 cells (3 seeds × 4 null arms × 3 recovery intervals).

**Outcome (frozen interpretation @ execution):** formal `ADAPTATION` classification
**not** assigned on D arm. \(A_{\mathrm{adapt}}\approx 0.29>\theta_A\) (spike-count
attenuation) but \(H_{\mathrm{K}}^{\mathrm{late}}\not>1+\theta_H\) (M2 gate).
Observable \(A_{\mathrm{adapt}}\) is **identical** across N0/N1/N2/D (D−N2 null on
spike phenotype). Hidden-state recovery with \(T_{\mathrm{rec}}\uparrow\) confirmed
(\(|H_{\mathrm{K}}(T_{\mathrm{rechallenge}})-1|\) decreases short→long).

**Q1/Q2/Q3 decomposition** recorded in `d3_interpretation_receipt.json` for Figure 6.

**Spec:** `artifacts/protocol_d_biological_rbs/d3_adaptation_recovery_phenotype_spec.json`  
**Receipts:** `d3_execution_receipt.json`, `d3_interpretation_receipt.json`

## Checkpoint ladder

\[
\boxed{
\begin{aligned}
D0 &: \text{biological RBS specification},\\
D1 &: \text{static typed-coordinate expression},\\
D2\mathrm{a} &: \text{autonomous }H_{\mathrm{K}}\text{ relaxation},\\
D2\mathrm{b} &: \text{activity}\rightarrow H_{\mathrm{K}}\text{ coupling (executed)},\\
D3 &: \text{adaptation/recovery phenotype (executed)},\\
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
| **D** | Figure 6 | Biological RBS containment under \(\dot W=0\); **closed @ D3** — architectural extension demonstrated; activity-written spike adaptation **not supported** (informative `NO_ADAPTATION`) |

C3 and H4 are joint examples of **falsification discipline** but answer different
questions — keep separate panels and narrative.

## Checkpoints

| ID | Status |
|----|--------|
| D0 | Specification frozen |
| D1 | Static \(H_{\mathrm{K}}\) sweep **executed** |
| D2a | Autonomous F1 relaxation **executed** |
| D2b | Two-coordinate \((H_{\mathrm{A}},H_{\mathrm{K}})\) coupling — **executed** |
| D3 | Adaptation/recovery phenotype — **executed** (`NO_ADAPTATION`) |
| D4 | Optional second RBS class — **not authorized** |

## Protocol D closed @ D3

\[
\boxed{
\text{RBS dynamics}
\neq
\text{mechanism-strength criterion}
\neq
\text{observable adaptation phenotype}
}
\]

**Figure 6 ladder (frozen):**

| Step | Status |
|------|--------|
| Static \(H_{\mathrm{K}}\to X\) | **demonstrated** |
| \(H_{\mathrm{K}}(t)\to 1\) | **demonstrated** |
| \(S\to H_{\mathrm{A}}\to H_{\mathrm{K}}\) state mechanism | **demonstrated** |
| Activity-written \(H_{\mathrm{K}}\to\) distinct spike adaptation | **not supported** |

**D3 key observations:** \(S\to H_{\mathrm{A}}\) demonstrated; \(H_{\mathrm{A}}\to H_{\mathrm{K}}\)
dynamic but \(H_{\mathrm{K}}^{\mathrm{late}}\approx 1.006 < 1.01\); \(R_j^{D}=R_j^{N2}\);
\(A_{\mathrm{adapt}}\approx 0.29\) identical across N0–D (attenuation not attributable to D2b
\(H_{\mathrm{K}}\) writing). Hidden-state recovery \(T_{\mathrm{rec}}\uparrow\Rightarrow
|H_{\mathrm{K}}(T_{\mathrm{rechallenge}})-1|\downarrow\) demonstrated separately from
observable spike recovery.

**Closure receipt:** `artifacts/protocol_d_biological_rbs/d_closure_interpretation_receipt.json`

**Methodological carry-forward to E:** every phenotype claim requires a mechanism-null contrast.
