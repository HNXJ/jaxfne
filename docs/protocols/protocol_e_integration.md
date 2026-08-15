# Protocol E — integrated TFNE composition (0.4.17-E)

**Status:** E0.1 ladder **frozen**; **E1 closed**; **E2 closed** (implementation + execution receipt frozen)
**Milestone boundary:** `9589933` — 0.4.17 transitions from component validation (D) to TFNE grammar composition (E)

**Specs:**  
- E0: `artifacts/protocol_e_integration/e0_composition_spec.json`  
- E0.1: `artifacts/protocol_e_integration/e0_1_implementation_ladder_spec.json`  
- E1: `artifacts/protocol_e_integration/e1_hierarchy_runtime_spec.json`
- E2: `artifacts/protocol_e_integration/e2_delayed_coupling_spec.json`

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
| E3–E5 | Not authorized (spec deferred) |
