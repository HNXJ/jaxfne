# Protocol W — HDP parameter memory (specification only)

**Status:** SPECIFICATION OPEN · IMPLEMENTATION **not** authorized  
**Prerequisite:** Protocol H **closed** at H4 (`docs/doctrine/protocol_h_rbd_memory.md`, interpretation receipt `artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json`)  
**Out of scope:** H4 rescue, Protocol H extensions (H5), full RBD+HDP+delays+hierarchy composition before W is characterized

## 1. Scientific question

\[
\boxed{
\text{Can transient RBS history be written into a slower parameter state while preserving bounded, physically sane dynamics?}
}
\]

Protocol W is **independent** of the H4 topology/delay falsification. It must **not** be introduced to “fix” H4 or to recover the Checkpoint-1 conjecture that longer heterogeneous recurrence extends activity-expressed fading memory.

## 2. Causal hierarchy (composition grammar)

Protocol H established the RBD path with \(\dot W=0\):

\[
u \rightarrow H \rightarrow X.
\]

Protocol W opens the HDP path:

\[
u \rightarrow H \rightarrow W \rightarrow X.
\]

| Layer | State | Dynamics | Protocol |
|-------|-------|----------|----------|
| RBD | \(H\) | \(\dot H = F_H(\ldots)\) | H (closed) |
| HDP | \(W\) | \(\dot W = F_W(H,\ldots)\) | W (this spec) |
| Activity | \(X\) | \(F_x\) with typed gains | shared kernels |

RBD and HDP are **compositional**. W must be mathematically characterized on a **minimal topology** before composing

\[
\text{RBD} + \text{HDP} + \text{delays} + \text{hierarchy}.
\]

Complexity is added by composition, not piled together from the beginning.

## 3. What Protocol H established (inputs to W)

From H1–H4 (see H closure receipt):

| Proposition | Content | H evidence |
|-------------|---------|------------|
| **P1** | RBS retains perturbation history | H1a/H2 continuation, H3 \(M_H\) |
| **P2** | RBS history can influence neural activity | H1c \(G_H\), H3 \(\beta_H\) null structure |
| **P3** | Network geometry/delays extend activity-expressed memory (preregistered form) | **Not supported** (H4) |

Checkpoint H (frozen):

\[
\boxed{
\begin{array}{l}
\text{RBS state grammar} + \text{RBD dynamics} + \text{typed }H\!\rightarrow\!X\text{ coupling} \\
+ \text{exact delayed continuation} + \text{quantified fading-memory assay} \\
+ \text{prospective topology/delay falsification}.
\end{array}}
}
\]

W builds on this closure; it does not reopen H4 design choices (\(\Delta_{\max}\), decoder, \(F_H\), \(\beta_H\), perturbation magnitude, delay distributions).

## 4. Primary W experiment (pre-registered sketch)

**Minimal topology first** — not the 12-node heterogeneous ring. Use the smallest system where

\[
\Delta H_{\mathrm{pre/post}} \rightarrow \Delta W
\]

is **identifiable**, **bounded**, and **retentive/reversible** under explicit \(F_W\) rules.

Candidate observables (to be frozen before implementation):

- \(\|\Delta W\|\) after declared \(H\) perturbation or input protocol
- retention of \(\Delta W\) after \(H\) returns toward baseline
- activity readout \(X\) change attributable to \(\Delta W\) rather than transient \(H\) alone
- null: \(F_W \equiv 0\), shuffle controls, matched input without \(H\) excursion

**Not the primary W question:** decodability of perturbation identity across long rings (that was Protocol H4).

## 5. Conservation and boundedness grammar (mandatory before code)

Legacy toy HDP experiments already demonstrated failure modes: unconstrained \(H\)-gradient plasticity can **manufacture coupling** and **destabilize** the system. Protocol W implementation is blocked until the following are specified in writing:

### 5.1 \(F_W\) contract

\[
\dot W = F_W(H, W, x, I; \Theta_W),
\qquad
W \in \mathcal W_{\mathrm{adm}},
\]

with explicit:

| Requirement | Specification obligation |
|-------------|------------------------|
| **Domain** | Admissible set \(\mathcal W_{\mathrm{adm}}\) (sign, magnitude, sparsity, Dale) |
| **Conservation** | Quantities conserved or dissipated per step (if any); no hidden mass creation |
| **Boundedness** | Invariants or contractive terms preventing runaway \(W\) |
| **Gain sanity** | Maps from \(W\) to synaptic drive remain in biophysically declared range |
| **Separation** | \(F_W\) must not duplicate \(F_H\) state storage or smuggle \(H\rightarrow X\) through unbounded \(W\) |
| **Null recovery** | \(F_W=0\) recovers RBD-only Protocol H behavior on shared kernels |

### 5.2 Stability receipts (pre-implementation)

Before any W kernel is authorized:

1. **Local** — fixed point or bounded orbit under constant \(H\)
2. **Perturbation** — finite \(\Delta H\) produces finite \(\Delta W\)
3. **Coupling** — \(\partial X/\partial W\) and \(\partial W/\partial H\) paths do not create positive feedback without declared saturation
4. **Continuation** — if delays enabled later, full Markov state includes \(W_t\) in segment handoff

### 5.3 Forbidden shortcuts

- No tuning \(F_W\) after viewing held-out retention metrics
- No “rescue H4” objective in W loss functions or acceptance criteria
- No conflation of RBD fading memory with synaptic long-term memory without separate receipts

## 6. Implementation phases (ordered, not authorized)

| Phase | Deliverable | Gate |
|-------|-------------|------|
| **W0** | This specification + boundedness grammar frozen | **Current** |
| **W1** | Minimal \(F_W\) on smallest topology; \(\dot W\) only, RBD \(F_H\) fixed | Boundedness proofs + unit tests |
| **W2** | \(\Delta H \rightarrow \Delta W\) identification experiment | Prospective receipt |
| **W3** | Retention/reversibility of \(W\) traces | Prospective receipt |
| **W4** | Compose RBD + HDP (still minimal topology) | No regression of H nulls |
| **W5+** | Delays, hierarchy, larger graphs | Only after W1–W4 |

## 7. Relation to existing code

| Artifact | Protocol W role |
|----------|-----------------|
| `simulate_edge_recurrent_izhikevich_hdp` | Legacy reference implementation — **not** canonical W without boundedness audit |
| `simulate_edge_recurrent_izhikevich_rbd` | RBD substrate (\(\dot W=0\)); W must reduce to this when plasticity disabled |
| `docs/doctrine/rbs_rbd_hdp.md` | Mathematical authority for RBS/RBD/HDP grammar |

Do not promote legacy HDP paths to Protocol W without explicit \(F_W\) specification and falsification receipts.

## 8. Evidence and receipts

Each prospective W run (when authorized) exports:

- protocol ID (`protocol_w_hdp_parameter_memory`)
- git SHA, frozen \(F_W\) and \(\mathcal W_{\mathrm{adm}}\)
- topology receipt (minimal graph)
- perturbation protocol for \(\Delta H\)
- \(\Delta W\) trajectories and null controls
- boundedness/stability checks (finite, in-domain)

Failed prospective runs are preserved.

## 9. Stop rules

Stop and report rather than extend scope if:

- \(F_W\) is implemented before boundedness grammar is frozen
- W experiments use long heterogeneous rings “because H4”
- plasticity explains an effect that RBD already explains under Protocol H nulls
- \(\Delta W\) is not identifiable separately from transient \(H\)
- implementation begins without W0 specification sign-off

## 10. References

- `docs/doctrine/protocol_h_rbd_memory.md` — Protocol H (closed)
- `artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json` — H4 frozen negative result
- `docs/doctrine/rbs_rbd_hdp.md` — RBS/RBD/HDP definitions, falsification ladder
- `docs/doctrine/tfne_containment_architecture.md` — typed coupling maps
