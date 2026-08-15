# Protocol E — integrated TFNE composition (0.4.17-E)

**Status:** E0.1 ladder **frozen**; **E1–E4 closed**; **E5 specification frozen** (implementation not authorized); TFNE integration architecture complete through observation
**Milestone boundary:** `9589933` — 0.4.17 transitions from component validation (D) to TFNE grammar composition (E)

**Specs:**  
- E0: `artifacts/protocol_e_integration/e0_composition_spec.json`  
- E0.1: `artifacts/protocol_e_integration/e0_1_implementation_ladder_spec.json`  
- E1: `artifacts/protocol_e_integration/e1_hierarchy_runtime_spec.json`
- E2: `artifacts/protocol_e_integration/e2_delayed_coupling_spec.json`
- E3: `artifacts/protocol_e_integration/e3_rbs_composition_spec.json`
- E4: `artifacts/protocol_e_integration/e4_observation_chain_spec.json`
- E5: `artifacts/protocol_e_integration/e5_causal_perturbation_spec.json`

**Prerequisites:** Protocol D closed @ D3; Protocol C closed @ C4; Protocol H closed @ H4; W3 unresolved

## Central question (E0)

\[
\boxed{
\text{Can independently validated TFNE components compose into a
hierarchical multi-area system without semantic or dynamical failure?}
}
\]

E is a **composition experiment**, not a phenotype-manufacturing experiment.

## Composition target

\[
\boxed{
\text{geometry}
+
\text{heterogeneous populations}
+
\text{delays}
+
\text{RBS/RBD}
+
S\rightarrow F/P
\rightarrow
\text{one coherent hierarchical system}
}
\]

**Naming:** heterogeneous **populations/parameters** within one emitter family (e.g. Izhikevich) is not the same as heterogeneous **emitter equations** (\(F_X\) families).

## E0.1 implementation ladder (frozen)

\[
\boxed{
E1\ \text{hierarchy/runtime}
\rightarrow
E2\ \text{delayed coupling}
\rightarrow
E3\ \text{RBS composition}
\rightarrow
E4\ \text{observation chain}
\rightarrow
E5\ \text{integrated perturbation}
}
\]

### Integration monotonicity

> Adding a validated TFNE layer must not invalidate previously established lower-layer semantics.

\[
E1 \subset E2 \subset E3 \subset E4
\]

**Reduction contracts (required):**

| Contract | When | Must recover |
|----------|------|--------------|
| R\_E2→E1 | delays = 0 | E1-like coupling |
| R\_E3→E2 | RBS at reference \(H=H^\*\) | E2 dynamics |
| R\_E4→E3 | observation removed | E3 trajectories unchanged |

### Checkpoint scope summary

| ID | Adds | Excludes |
|----|------|----------|
| **E1** | Two-area laminar hierarchy, E/PV populations, FF/FB ownership, zero delay | RBS, delays, observation, phenotype |
| **E2** | \(\mathcal G \to \tau_{ij} \to X\), `delay_state` continuation | RBS, wave claims |
| **E3** | RBS/RBD on selected populations; \(H=H^\*\Rightarrow E3=E2\) | D3 adaptation target |
| **E4** | \(X,H\to Q\to\Phi\to P\to Y\) (Experiment-A semantics) | Stronger-than-proxy field claims |
| **E5** | Integrated perturbation | — (requires E1–E4) |

### E5 phenotype rule (from D3)

\[
\boxed{
\text{phenotype evidence} = \text{perturbation effect} - \text{mechanism-null effect}
}
\]

## E1 — hierarchy/runtime (authorized)

Minimal **A1/A2** laminar hierarchy with **E** and **PV** populations (same Izhikevich family, distinct parameters), explicit **FF** (A1 L2/3 E → A2 L4) and **FB** (A2 L5 E → A1 L2/3), **zero delay**, **no RBS**.

**Structural gates:** construction, identity round-trip recovery, FF/FB ownership on edge sets, finite deterministic execution, reproducibility, baseline structural reduction (inter-area disabled).

**Diagnostics:** per-edge provenance table (`local_A1`, `local_A2`, `FF_A1_to_A2`, `FB_A2_to_A1`).

**Receipt:** `artifacts/protocol_e_integration/e1_execution_receipt.json`

**Not required:** publication phenotype, spectral claims.

## E2 — typed delayed coupling (closed)

**Question:** Does adding typed provenance-class delays preserve everything E1 established?

**Added DOF:** \(\tau_{ij}>0\) via four provenance classes only (`local_A1`, `local_A2`, `FF_A1_to_A2`, `FB_A2_to_A1`).

**Reduction contract:**

\[
\boxed{
R_{E2\rightarrow E1}:\quad
\tau_{ij}=0\ \forall(i,j)\Rightarrow E2\equiv E1
}
\]

Bit-exact on \(V_m\), spikes, identity/provenance, and edge structure where ordering permits.

**Frozen delay values** (dt = 0.5 ms; \(\tau_{\mathrm{local}}\le\tau_{\mathrm{FF}}<\tau_{\mathrm{FB}}\)):

| edge_class | delay_ms | delay_steps |
|------------|----------|-------------|
| local_A1 | 1.0 | 2 |
| local_A2 | 1.0 | 2 |
| FF_A1_to_A2 | 2.0 | 4 |
| FB_A2_to_A1 | 4.0 | 8 |

**E1-receipt-derived:** `p_local=0.2` (documented in E1 execution receipt; **not** E1 preregistration).

**Gates (G1–G8):** E1 reduction, provenance preservation, delay ownership, finite delayed execution, delayed continuation (`delay_state`), exact event timing, hierarchy invariance, no spectral/functional overinterpretation.

**Continuation splits:** primary @ 400 ms; in-flight stress @ 120 ms.

**Excluded from E2 evidence:** RBS, field/probe chain, spectral/functional claims.

**Receipt:** `artifacts/protocol_e_integration/e2_execution_receipt.json`

**Closed:** G1–G8 pass; typed delay table and delay occupancy frozen.

## E3 — RBS composition (closed)

**Question:** Does hierarchy + typed delays compose with sparse D1/D2a \(H_K\) RBS without changing E2 at reference?

**Added DOF:** selected population receives \(H_K\) RBD state with \(b_{\mathrm{eff}} = H_K b\) (D1/D2a primitive only; **not** D2b activity writing).

**Reduction contract:**

\[
\boxed{
R_{E3\rightarrow E2}:\quad
H=H^\star,\ \dot H=0
\Rightarrow
E3\equiv E2
}
\]

**RBS owner** (verified against E1 identity table @ spec freeze):

| area | layer | cell_type | n_nodes | flat_indices |
|------|-------|-----------|---------|--------------|
| A2 | L5 | E | 7 | 70–76 |

**Modes:** `E3-null` (\(H_K=1,\ \dot H_K=0\), must match E2 through E3 path) and `E3-dynamic` (\(\tau_K\dot H_K = 1-H_K\), \(H_K(0)=1+\delta_H=1.2\) on owners, \(\tau_K=100\) ms).

**Non-owner semantics:** non-owners carry fixed reference \(H_K=1\) with F1 recurrence masked off (not an unallocated coordinate).

**Combined continuation:** segmented execution carries \(H_K\) and \(\mathcal{B}_t\) (`delay_state`); in-flight stress @ 120 ms with both \(H_K\neq 1\) and \(\mathcal{B}_t\neq 0\).

**Gates (G1–G9):** E2 reduction, ownership, non-owner invariance, typed expression, autonomous recovery (D2a Euler recurrence), delay compatibility, continuation, hierarchy invariance, composition-effect-only (no phenotype).

**Receipt:** `artifacts/protocol_e_integration/e3_execution_receipt.json`

**Closed:** G1–G9 pass; E2 delay digest and hierarchy fingerprints inherited unchanged.

## E4 — observation chain (closed)

**Question:** Can the E3 hierarchical trajectory generate independently declared multiscale observations while preserving the underlying neural/source state?

**Added DOF:** downstream-only observation \((X,H,\mathcal B)\xrightarrow{S}Q\xrightarrow{F}\Phi\xrightarrow{P}Y\) with no feedback into E3 dynamics.

**Reduction contract:**

\[
\boxed{
R_{E4\rightarrow E3}:\quad
\text{observation disabled}
\Rightarrow
E4\equiv E3
}
\]

**Stronger invariant (probe-independent):**

\[
\boxed{
(X,H,\mathcal B,Q,\mathcal G)_{P_1}
=
(X,H,\mathcal B,Q,\mathcal G)_{P_2}
}
\]

**Workflow:** simulate once (E3 path) \(\rightarrow\) freeze \(X,H,Q,\mathcal B,\mathcal G\) \(\rightarrow\) apply multiple \(F/P\). No re-simulation per probe.

**Experiment A inheritance:** reuse canonical relative \(Q\) (`signals.sources_canonical_relative_source`) and validated `project_laminar_sources` / `lfp_proxy_probe` / `csd_proxy_probe` semantics — **no integrated-model source variant**.

**Primary evidence:** native `V_m`, spikes, first-class `Q`; `lfp_ref` relative proxy; shallow/deep LFP contacts; CSD-from-LFP finite-difference relative proxy.

**Hierarchy-aware source table:** \(Q \rightarrow (\mathrm{area},\mathrm{layer},\mathrm{cell\_type},t)\) with conservation \(\sum_{a,\ell,c}Q_{a,\ell,c}(t)=\sum_i Q_i(t)\).

**Gates (G1–G10):** E3 reduction/neural invariance, single source-of-truth, source identity + T\_E4, probe independence, zero-source (declared operators), linearity (declared \(F\)), semantic status, hierarchy/provenance preservation, reproducibility, no phenotype claim.

**Diagnostic (outside G1):** E3-dynamic observation run confirms non-reference \(H_K\) trajectories accept the same observation chain (not phenotype evidence).

**Receipt:** `artifacts/protocol_e_integration/e4_execution_receipt.json`

**Closed:** G1–G10 pass; Experiment A semantics reused; source aggregation conservation verified.

## E5 — causal perturbation (spec frozen)

**Question:** Does a localized RBS perturbation produce a measurable hierarchical response beyond its matched mechanism-null?

**Adds:** prospective causal evidence only — **zero new TFNE architecture**.

**Owner:** \(\mathcal O_H =\) A2:L5:E (flat indices 70–76); perturbation \(H_K(t_0^+)=1.2\) on owners.

**Arms:**

| ID | \(H_K\) on \(\mathcal O_H\) | \(\Gamma_H\) | \(b_{\mathrm{eff}}\) |
|----|------------------------------|--------------|----------------------|
| **N0** | 1 (reference) | identity | \(b\) |
| **N1** | 1.2, matched dynamics | disabled | \(b\) |
| **D** | 1.2, matched dynamics | enabled | \(H_K b\) |

**Primary contrast:** \(\boxed{D - N_1}\) (not \(D\) vs N0 alone). Null invariants: \(H_K^{N1}=H_K^{D}\); \(\Gamma_H^{N1}=I\), \(\Gamma_H^{D}(H_K)=H_K\).

**Propagation assay:** \(H_K \rightarrow X_{\mathrm{owner}} \rightarrow X_{\mathrm{A2}} \rightarrow X_{\mathrm{A1}} \rightarrow Q \rightarrow Y\).

**Response vector:** \(\Delta R = (\Delta X_{\mathrm{owner}}, \Delta X_{\mathrm{A2/nonowner}}, \Delta X_{\mathrm{A1}}, \Delta Q, \Delta Y)\) with simple frozen metrics (voltage/spike/source/proxy norms and integrals; **no new spectral pipeline**).

**Primary prediction:** \(D-N_1 \neq 0\) at owner population (required for non-`NO_EFFECT`). Downstream propagation is secondary.

**Classification:** `NO_EFFECT` | `LOCAL_EXPRESSION` | `HIERARCHICAL_PROPAGATION` | `UNRESOLVED` — **HIERARCHICAL_PROPAGATION not required to close E5**.

**A1 interpretation:** any A1 effect is structural propagation via frozen **FB** pathway A2\(\rightarrow\)A1, not FF/FB functional/spectral claim.

**Workflow:** one neural trajectory per arm/seed \(\rightarrow\) freeze \(\rightarrow\) E4 \(F/P\) (no re-simulate per observation).

**Seeds:** `[11, 12, 13]` fixed (no added sample size without spec amendment).

**Post-close policy:** hard 0.4.17 scientific feature freeze; next work is publication evidence (Figures 1–7), not E6.

**Provenance housekeeping (prospective):** distinguish `execution_parent_sha` from `artifact_commit_sha` in publication manifest; do not retroactively rewrite E3/E4 write-once receipts.

**Not authorized:** E5 implementation, new architecture, spectral/adaptation/HDP phenotype claims.

## Explicit blocks (unchanged from E0)

| Block | Status |
|-------|--------|
| W3 closed-loop HDP | **unresolved** |
| D3 adaptation requirement | **NO_ADAPTATION frozen** |
| D4 second RBS class | **not authorized** |
| Monolithic E implementation | **prohibited** |

## Checkpoints

| ID | Status |
|----|--------|
| E0 | Composition question **frozen** |
| E0.1 | Implementation ladder + reduction contracts **frozen** |
| E1 | Hierarchy/runtime **closed** (G1–G6 pass; execution receipt frozen) |
| E2 | Delayed coupling **closed** (G1–G8 pass; execution receipt frozen) |
| E3 | RBS composition **closed** (G1–G9 pass; execution receipt frozen) |
| E4 | Observation chain **closed** (G1–G10 pass; execution receipt frozen) |
| E5 | Causal perturbation spec **frozen**; implementation **not authorized** |
