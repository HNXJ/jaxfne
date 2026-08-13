# Protocol H — RBD state memory (fixed weights)

**Status:** H1a IMPLEMENTED; H1b SPEC OPEN; H2–H4 not started  
**Baseline:** `dev` @ `cf0eb43` (H1 kernel)  
**Prerequisite:** Protocol D₀/D₁ (`724aa32`) — finite edge-delay semantics  
**Out of scope:** Protocol W, HDP (\(\dot W \neq 0\)), D₂ geometry compiler, H1c implementation (until H1b frozen)

## 1. Scientific question

Before any synaptic plasticity, quantify how much **adaptation and memory** arise from

\[
\boxed{\text{RBS}+\text{RBD}+\text{delays},\qquad \dot{\mathbf W}=0}.
\]

If plasticity is enabled too early, a successful memory result is **ambiguous** between
hidden-state dynamics and synaptic modification. Protocol H isolates the intrinsic
**RBD memory kernel**.

**Checkpoint-1 conjecture (falsifiable, not assumed):** recurrent geometry and delay
heterogeneity can prolong or distribute fading state memory across \(\mathcal X_t\).
Protocol H must be able to **reject** this conjecture.

## 2. Experimental ladder

Protocol H is the first rung. Do not develop H and W simultaneously.

\[
\boxed{
\underbrace{\text{Protocol H}}_{\text{state memory}}
\rightarrow
\underbrace{\text{Protocol W}}_{\text{parameter memory via HDP}}
\rightarrow
\underbrace{\text{adaptation/omission/oddball}}_{\text{phenomena}}
\rightarrow
\underbrace{\text{surprise/prediction}}_{\text{theory test}}
}
\]

Protocol W opens only after the intrinsic RBD retention function \(M(\Delta)\) is
characterized under \(\dot W=0\). Its question is whether
\(\Delta\mathbf H \rightarrow \Delta\mathbf W\) extends memory onto another
timescale — not whether delays alone produce waves.

## 3. Markov state

The complete discrete-time Markov state is

\[
\boxed{
\mathcal X_t =
\bigl(
\mathbf x_t,\;
\mathbf H_t,\;
\mathbf W_t,\;
\mathcal B_t,\;
\ldots
\bigr)
}
\]

where:

- \(\mathbf x_t\) — emitter activity state (e.g. Izhikevich \((v,u)\), synaptic filters)
- \(\mathbf H_t\) — RBS coordinates per neuron, \(\mathbf H_i(t)\in\mathbb R^{d_H}\)
- \(\mathbf W_t\) — **frozen** in Protocol H (\(\mathbf W_{t+1}=\mathbf W_t\))
- \(\mathcal B_t\) — delay history (spike ring buffer) whenever any
  `delay_steps > 0` (Protocol D)

**Continuation contract:** segmented simulation must accept and return the **full**
\(\mathcal X_t\) needed by the selected kernel. Nonzero-delay continuation without
\(\mathcal B_t\) is invalid (D₀/D₁ currently reject `init_state` when delays are
active; Protocol H implementation must **lift** this by threading `spike_history`).

Relative coordinates use baseline-one where appropriate:
\(H_{ik}=z_{ik}/z_{ik}^\star\) with nominal \(H=1\), or reduced
\(H_{ik}=\mathcal R_k(\mathbf z_i)\).

## 4. RBS and RBD kernel (minimal)

### 4.1 RBS

\[
\mathbf H_i(t)\in\mathbb R^{d_H},\qquad d_H\ge 1.
\]

Protocol H **phase 1** requires scalar support (\(d_H=1\)) with a path to
\(d_H>1\). Initial condition default: \(H_i(0)=1\) (relative equilibrium).

### 4.2 RBD — two coupling directions

\[
\dot H_i = F_H(H_i, x_i, I_i;\kappa_{x\rightarrow H}),
\qquad
\dot x_i = F_x(x_i, I_i; G(H_i;\kappa_{H\rightarrow x})),
\qquad
\dot{\mathbf W}=0.
\]

| Checkpoint | Content | Status |
|------------|---------|--------|
| **H1a** | \(x/I \rightarrow H\) via \(\kappa_H I_i^{\mathrm{rel}}\) | Implemented (`simulate_edge_recurrent_izhikevich_rbd`) |
| **H1b** | Specify minimal \(H \rightarrow x\) gain \(G\) | `docs/doctrine/protocol_h_h1b_h_to_x_gain.md` |
| **H1c** | Implement selected gain (not authorized) | Blocked on H1b |

**H1a limitation:** \(F_x\) is **unchanged** in the current kernel. With
\(\kappa_H=0\), a localized \(H_k=1+\delta_H\) perturbation can relax in \(H\)
without affecting spikes or voltages. That validates RBS state dynamics but is
**not sufficient** for the RBD memory hypothesis \(H\rightarrow x\). Do not open
H3 \(M(\Delta)\) until H1c provides a null-recoverable \(H\rightarrow x\) interface.

Delayed recurrent input (when enabled):

\[
\mathbf I_i(t)= \sum_j \mathcal C_{ji}\bigl(\mathbf x_j(t-\tau_{ji}),\,W_{ji}\bigr).
\]

Activity may feed back into \(F_H\) via \(\kappa_H I_i^{\mathrm{rel}}\) — see §4.5.

### 4.5 \(I^{\mathrm{rel}}\) and baseline-one vs driven steady state

**Implemented definition:**

\[
I_i^{\mathrm{rel}} = I_{\mathrm{syn},i}/i_{\mathrm{ref}},
\qquad
I_{\mathrm{syn},i}=\sum_{e:\,\mathrm{post}(e)=i} w_e s_e .
\]

\(I^{\mathrm{rel}}\) is **not zero-centered**: ongoing recurrent activity with
nonnegative filter states typically yields \(\mathbb E[I^{\mathrm{rel}}]>0\).

Distinguish:

| Term | Meaning |
|------|---------|
| **Coordinate reference** | RBS baseline \(H=1\) where \(R(1)=0\) |
| **Driven steady state** | \(H^\*\) solving \(R(H^\*)+\kappa_H\mathbb E[I^{\mathrm{rel}}]=0\) |

When \(\kappa_H>0\), generally \(H^\*\neq 1\). Baseline-one is a **reference
calibration**, not a claim that the driven attractor is at \(H=1\). With
\(\kappa_H=0\) and no input coupling, \(H^\*=1\) for F1/F2.

### 4.6 H1 phase portrait (isolated, \(I^{\mathrm{rel}}=0\))

F1: \(R_1(H)=1-H\) — antisymmetric about \(H=1\).  
F2: \(R_2(H)=H^{-1}-1\) — stronger recovery below 1, slower decay above 1 despite
matched Jacobian at \(H=1\). Receipt: `tests/test_protocol_h_rbd_h1_phase_portrait.py`.

### 4.3 \(F_H\) candidate families (not one canonical equation)

The legacy HDP income/spending kernel (including the exploratory scalar term
\(\rho_{\mathrm{passive}}/H_i^2\), sometimes informally written “\(100/H\)”) is
**one candidate realization**, not canonical RBD theory.

Protocol H must compare **at least two** scalar \(F_H\) families under identical
falsification metrics:

**Pre-implementation correction (H1, post-`81700a4`):** F2 is **not** the literal
legacy \(\rho/H^2\) (“100/H”) scaling. It is the baseline-one matched inverse-state
candidate

\[
R_2(H)=H^{-1}-1,\qquad
\tau_H\dot H_i = R_2(H_i)+\kappa_H I_i^{\mathrm{rel}},
\]

documented as **ancestry** from the toy passive-income rule, not mathematical
identity. F1 and F2 share equilibrium \(H^\*=1\) and first-order relaxation
rate at \(H=1\): \(R_1'(1)=R_2'(1)=-1\).

| ID | Family | \(F_H\) restoring term (scalar \(H_i\), \(\kappa_H I_i^{\mathrm{rel}}\) optional) | Role |
|----|--------|-------------------------------------------------------------------------------------|------|
| **F0** | RBS-disabled null | \(\dot H_i=0\); \(H_i\equiv 1\) | Separates delay-only / activity-only memory |
| **F1** | Linear restoring | \(\tau_H\dot H_i = (1-H_i) + \kappa_H I_i^{\mathrm{rel}}\) | Analytic reference; \(H^\*=1\), \(\partial\dot H/\partial H|_{1}=-1/\tau_H\) |
| **F2** | Inverse-state matched | \(\tau_H\dot H_i = (H_i^{-1}-1) + \kappa_H I_i^{\mathrm{rel}}\); **requires \(H>0\)** | Same local Jacobian as F1 at \(H=1\); nonlinear away from equilibrium |

\[
R_1(H)=1-H,\qquad R_2(H)=H^{-1}-1,\qquad R_1(1)=R_2(1)=0,\qquad R_1'(1)=R_2'(1)=-1.
\]

F2 trajectories that reach \(H\le 0\) are **invalidated** (propagate non-finite
\(H\)); no hidden clipping or arbitrary epsilon floor.

Additional families (e.g. trace-filtered activity drive) may be added only if they
are **pre-registered** with the same \(M(\Delta)\) pipeline.

**Design constraint:** do not tune \(F_H\) to maximize visual memory. Select among
families by **pre-registered decodability metrics**, not raster aesthetics.

### 4.4 Timescales

Each \(F_H\) family must expose at least one explicit timescale (e.g. \(\tau_H\)
or \(\tau_0\)) in the protocol receipt. Timescale sweeps are optional secondary
axes; the primary matrix is §6.

## 5. Perturbation and retention metric

### 5.1 Perturbation

**Primary (pre-registered):** localized RBS intervention at time \(t_0\):

\[
H_k(t_0^-)=1 \rightarrow H_k(t_0^+)=1+\delta_H,\qquad H_{i\neq k}=1.
\]

This interrogates \(\Delta H \rightarrow x/H/\mathcal B \rightarrow\) distributed
fading state without the confound \(u\rightarrow x\rightarrow H\).

**Secondary:** pulse-current perturbation — introduces an additional transfer path
and is reserved for later cross-checks, not the first memory test.

Other examples (optional, pre-register before use):

Post-perturbation, external input returns to a **declared baseline** \(I^{\mathrm{base}}(t)\)
identical across trials that differ only in perturbation identity or magnitude.

### 5.2 Decoder

A decoder estimates perturbation identity or magnitude from future full state:

\[
\hat u_{t_0} = D\bigl(\mathcal X_{t_0+\Delta}\bigr).
\]

\(D\) may be linear (e.g. logistic/ridge on stacked features from \(\mathcal X\)),
a small train-once classifier, or another **pre-registered** decoder with fixed
hyperparameters. The decoder is fit on a **training fold** of perturbation labels;
reporting uses held-out trials.

Permitted features from \(\mathcal X_{t_0+\Delta}\) must be declared (e.g.
\(H\), \(v\), spike counts, synaptic state slices, \(\mathcal B_t\) summaries).
Using \(\mathcal B_t\) directly tests delay-buffer memory; excluding it tests
RBS-mediated memory only.

### 5.3 Retention function

\[
M(\Delta) =
\operatorname{Decodability}\bigl[
u(t_0);\;
\mathcal X(t_0+\Delta)
\bigr].
\]

Report \(M(\Delta)\) vs \(\Delta\) (in ms or steps), with confidence intervals
over seeds and perturbation instances. **No memory claim** without a quantitative
\(M(\Delta)\) curve (doctrine stop rule §16 in `rbs_rbd_hdp.md`).

Secondary metrics (optional, not substitutes): perturbation-aligned response
difference between matched-input trajectories, autocorrelation of \(H\), recovery
time constants — all pre-registered.

## 6. Primary experiment matrix

Fixed \(\dot W=0\), fixed \(F_H\) family per row block, shared decoder protocol:

\[
\begin{array}{c|c|c}
& \text{uniform delays} & \text{heterogeneous delays} \\
\hline
\text{short ring} & M(\Delta) & M(\Delta) \\
\text{long ring} & M(\Delta) & M(\Delta)
\end{array}
\]

Plus **RBS-disabled null** (F0) on each topology/delay class.

Definitions:

| Term | Specification |
|------|----------------|
| **short ring** | Minimal circumference supporting recurrent closure; pre-register \(N_{\mathrm{short}}\) |
| **long ring** | Larger \(N_{\mathrm{long}} \gg N_{\mathrm{short}}\) with matched local coupling statistics where possible |
| **uniform delays** | All edges share one `delay_steps` (or zero) |
| **heterogeneous delays** | At least two distinct positive delay classes on the ring |

Ring topology uses directed recurrent edges (one neighbor or k-nearest on a ring);
exact edge lists are frozen in the étude/script receipt.

## 7. Nulls and controls

| Null | Purpose |
|------|---------|
| **F0** (RBS off) | Memory from delays + activity only |
| **Matched input, no perturbation** | Decoder false-positive rate |
| **Instantaneous recurrence** (`delay_steps=0`) | Delay contribution |
| **Shuffle perturbation labels** | Metric sanity |
| **Identical seeds, zero noise** | Deterministic reproducibility |

HDP / \(\dot W \neq 0\) is **excluded** from Protocol H runs.

## 8. Implementation phases (ordered)

| Phase | Deliverable | Acceptance |
|-------|-------------|------------|
| **H0** | Parent protocol frozen | `81700a4` |
| **H1a** | `simulate_edge_recurrent_izhikevich_rbd`: \(\kappa_{x\rightarrow H}\), F0/F1/F2, D delays | `tests/test_protocol_h_rbd_h1.py` |
| **H1b** | \(H\rightarrow x\) gain specification + \(I^{\mathrm{rel}}\) semantics | `docs/doctrine/protocol_h_h1b_h_to_x_gain.md` |
| **H1c** | Implement selected gain interface | Blocked — not authorized |
| **H2** | Full-state continuation incl. \(\mathcal B_t\) | After H1c |
| **H3** | Localized RBS perturbation + \(M(\Delta)\) | After H1c; requires \(H\rightarrow x\) |
| **H4** | Matrix §6 + evidence receipt | After H3 |

**API note:** reuse compatibility names (`h_state_*`, `enable_hdp`) only where
semantically accurate; prefer a distinct **RBD-only** dispatch flag or kernel entry
(e.g. `enable_rbd` / `simulate_*_rbd`) rather than overloading HDP plasticity.
D-class renames remain out of scope unless explicitly authorized.

## 9. Relation to existing code

| Artifact | Protocol H role |
|----------|-----------------|
| `simulate_edge_recurrent_izhikevich` + delays | Activity + \(\mathcal B_t\) substrate (D) |
| `simulate_edge_recurrent_izhikevich_hdp` | **Not** Protocol H; includes \(\dot W\) and legacy multi-term \(F_H\) |
| `simulate_edge_recurrent_izhikevich_rbd` | **Protocol H1** kernel (\(\dot W=0\), F0/F1/F2) |
| `enable_homeostasis` / `homeostatic_ei` | Kernel-specific homeostasis (B-class); not generic RBD |

Extract F2's matched inverse-state form in the H1 kernel; do not treat the full
HDP path as Protocol H.

## 10. Evidence and receipts

Each H4 run exports:

- protocol ID (`protocol_h_rbd_memory`)
- git SHA, seeds, `dt_ms`, `n_steps`
- \(F_H\) family ID (F0–F2+)
- topology and delay receipt (`EdgeList` digest)
- perturbation specification
- decoder specification and train/test split
- \(M(\Delta)\) table and figure hash
- pass/fail vs pre-registered acceptance thresholds

Failed prospective runs are preserved. Do not tune the decoder or \(F_H\) after
viewing held-out \(M(\Delta)\) without declaring a new protocol revision.

## 11. Stop rules

Stop and report rather than silently extend scope if:

- Do not open H3 \(M(\Delta)\) without H1c \(H\rightarrow x\) (see H1b spec)
- \(F_H\) is selected by visual raster appeal without \(M(\Delta)\)
- continuation omits \(\mathcal B_t\) under nonzero delays
- plasticity (\(\dot W\)) is required to obtain the reported memory effect
- decoder performance is not above shuffle-null on F0
- a single \(F_H\) family is promoted to canonical RBD without comparative runs

## 12. References

- `docs/doctrine/rbs_rbd_hdp.md` — RBS/RBD/HDP definitions, §8 memory hypothesis
- `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` — mathematical authority
- `tests/test_edge_delay_protocol_d016.py` — Protocol D₀/D₁ tests
- `docs/doctrine/protocol_h_h1b_h_to_x_gain.md` — H1b \(H\rightarrow x\) gain specification
