# jaxfne doctrine — RBS / RBD / HDP

Repository-facing realization of the jaxfne hidden-state refactor. **Upstream
authority:** the six project-source markdowns under
`artifacts/project_sources/`, with `4_tfne_theory_and_neural_tensor.md` as the
principal theoretical source. Those sources require synchronized revision during
the semantic migration; until then, treat this document as the target doctrine
and the project sources as **pending upstream alignment**.

## Mission

Refactor jaxfne's hidden-state doctrine around a physically disciplined,
relative state-space formulation while preserving validated behavior and
compatibility.

Target hierarchy:

```text
TFNE
├── RBS — Relative Biophysical State
├── RBD — Relative Biophysical Dynamics
└── HDP — Hidden-state Dependent Plasticity
```

Preserve the canonical TFNE scientific grammar:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

This is primarily a semantic/theoretical refactor. Do not use it as
justification for unrelated API expansion.

## 1. Semantic migration

Retire the generic definitions:

```text
H = homeostatic state
HDP = homeostasis-dependent plasticity
```

Homeostasis is a possible property/regime of the dynamics, not the
definition of `H`.

Define:

\[
\boxed{\mathbf H_i(t)\in\mathbb R^{d_H}=\text{Relative Biophysical State (RBS)}}
\]

RBS is a finite-dimensional reduced representation of physically
realizable internal emitter state. Possible antecedents include relative
ionic concentrations/gradients, charge-related state, neurotransmitter
availability, vesicle/resource availability, ATP/energetic resources,
receptor/channel state, adaptation variables, and neuromodulatory state.

Where a physical quantity \(z_{ik}\) and reference \(z_{ik}^*\) are
explicit, prefer a relative coordinate such as

\[
H_{ik}=\frac{z_{ik}}{z_{ik}^{*}},
\]

so the nominal reference is \(H_{ik}=1\). A reduced coordinate may
instead represent several unresolved physical variables,

\[
H_{ik}=\mathcal R_k(\mathbf z_i).
\]

RBS may be abstract and dimensionless, but must not become an arbitrary
latent embedding. Each coordinate should admit a plausible biophysical
realization, even when that realization is not uniquely identifiable or
calibrated.

The scalar toy model is simply \(d_H=1\).

**Kernel-specific homeostasis is not erased.** A mechanism that genuinely
implements homeostatic regulation (e.g. `homeostatic_ei`, legacy
`enable_homeostasis` trace dynamics) retains homeostatic terminology for
that kernel. The migration applies to **generic** `H` doctrine:

\[
H_{\text{generic}}:\quad \text{“homeostatic state”}\rightarrow\text{RBS},
\qquad
\text{not}\quad
\text{homeostasis}\rightarrow\text{definition of }H.
\]

## 2. RBD — Relative Biophysical Dynamics

RBD is the general dynamics involving RBS:

\[
\dot{\mathbf x}_i=F_x(\mathbf x_i,\mathbf I_i;\mathbf H_i),
\]

\[
\dot{\mathbf H}_i=
F_H(\mathbf H_i,\mathbf x_i,\mathbf I_i,\mathbf W,\mathbf u_i,\ldots).
\]

For recurrent delayed input,

\[
\mathbf I_i(t)= \sum_j\mathcal C_{ji}
\left(\mathbf x_j(t-\tau_{ji}),W_{ji}\right).
\]

Discrete emitters should use the equivalent Markov transition:

\[
(\mathbf x_t,\mathbf H_t)\mapsto
(\mathbf x_{t+1},\mathbf H_{t+1}).
\]

RBD remains meaningful with fixed weights:

\[
\dot W=0.
\]

Therefore adaptation, history dependence, fading state memory, phase
memory, and delayed recurrent dynamics do not require plasticity by
definition.

Do not define RBD as memory, predictive coding, or surprise
minimization. Those are hypotheses/results to establish.

## 3. HDP — Hidden-state Dependent Plasticity

Refactor HDP to mean:

\[
\boxed{\text{Hidden-state Dependent Plasticity}}
\]

HDP is the subset in which persistent parameters depend on RBS and
possibly other state:

\[
\dot W_{ij} =
F_W(W_{ij},\mathbf H_i,\mathbf H_j,\mathbf x_i,\mathbf x_j,\ldots).
\]

Canonical distinction:

```text
RBS = hidden relative biophysical state representation
RBD = dynamics of activity/state involving RBS
HDP = persistent parameter/plasticity dynamics dependent on hidden state
```

Conceptual multiscale loop:

\[
\mathbf x\leftrightarrow\mathbf H\leftrightarrow\mathbf W.
\]

Do not require `W` for RBD. Do not call numerical optimization HDP or
biological learning.

## 4. TFNE integration

Preserve TFNE as the encompassing typed operator factorization.

Conceptually update the emitter to:

\[
E_\theta:
(\mathbf x_t,\mathbf H_t,\mathbf u_t,\boldsymbol\xi_t)
\mapsto (\mathbf x_{t+1},\mathbf H_{t+1}).
\]

The full recurrent Markov state may additionally contain weights,
previous spikes, conductances, delays/queues, receptor states, or other
variables required by the selected kernel.

The operator chain remains:

\[
(\mathbf x_t,\mathbf H_t,\mathbf W_t,\ldots)
\xrightarrow{E}
(\mathbf x_{t+1},\mathbf H_{t+1},\mathbf W_{t+1},\ldots)
\xrightarrow{S}Q_t \xrightarrow{F}\Phi_t
\xrightarrow{P}Y_t.
\]

Do not break or blur Source -> Field -> Probe semantics.

## 5. Relative-computation doctrine

Adopt:

\[
\boxed{\text{Compute relatively; calibrate physically; preserve the map.}}
\]

Prefer relative/dimensionless internal coordinates when:

1. scientific information is preserved;
2. required physical provenance/reference scales are retained;
3. later calibration remains possible where required;
4. normalization does not erase geometry, density, attenuation,
   conservation, or unit-relevant information.

Conceptually:

\[
\mathbf z_{\rm physical}
\xrightarrow{\mathcal N}
\tilde{\mathbf z}_{\rm relative}
\xrightarrow{\text{TFNE}} \tilde{\mathbf y}
\xrightarrow{\mathcal C}
\mathbf y_{\rm calibrated}.
\]

"Relative by default" does not mean units are irrelevant. Physical units
remain mandatory for calibrated physical claims.

For LFP-like, CSD-like, EEG-like, MEG-like, and related readouts,
preserve proxy-safe naming until source semantics, geometry, operator,
units, calibration, and validation justify stronger claims.

A fixed linear projection remains a projection, not a PDE solve.

## 6. Geometry/calibration boundary

Where physical calibration depends on geometry, make the boundary
explicit. A generic source calibration is

\[
Q_{\rm phys} =
\mathcal C_S(\tilde Q;\mathcal G,\Theta_{\rm cal}).
\]

Record as applicable:

```text
reference scales
normalization map
inverse/calibration map
geometry
source support
operator type
solver status
amplitude status
physical units
calibration provenance
```

Never reconstruct physical amplitude from a normalized representation if
the normalization discarded information required for that
reconstruction.

## 7. Quantization and reduced precision

Relative coordinates should make bounded, well-conditioned computation
easier and may enable reduced-precision/quantized execution.

But:

```text
relative != automatically quantization-safe
quantized != automatically faster
```

Validate reduced-precision paths against a reference path. Do not change
scientific semantics to fit a quantization scheme.

## 8. Memory/adaptation hypothesis

Working hypothesis: RBD can produce fading dynamical memory without
plasticity.

A past perturbation constitutes state memory when identical current
external input can produce different future responses because hidden
states differ:

\[
I^{(1)}(t)=I^{(2)}(t),\qquad
\mathbf H^{(1)}(t)\neq\mathbf H^{(2)}(t),
\]

implying generally

\[
\mathbf x^{(1)}(t)\neq\mathbf x^{(2)}(t).
\]

Recurrent topology and heterogeneous delays may distribute the trace
across RBS coordinates, activity, phase, delayed/in-flight events, and
hierarchy.

Do not claim heterogeneous delays necessarily prolong memory. Measure
retention/decodability.

HDP may convert transient RBD state into slower parameter memory:

\[
\text{input}\rightarrow\Delta\mathbf H\rightarrow\Delta\mathbf W.
\]

Treat this as a falsifiable hypothesis.

## 9. Surprise-minimization claim gate

Do not encode "surprise minimization" into RBS/RBD/HDP definitions.
Derive and test before claiming predictive or variational interpretations.

## 10. Minimal falsification program

| Test | Claim |
|------|-------|
| T1 | RBS state closure / full-state continuation |
| T2 | RBD equilibrium/stability (homeostasis established per equation, not by name) |
| T3 | H-only fading memory with \(W\) frozen |
| T4 | Topology/delay memory (no extension is valid) |
| T5 | HDP persistent parameter traces |
| T6 | Canonical phenomena (adaptation, oddball, omission, …) |
| T7 | Prediction/surprise — only after T1–T6 |

## 11. Repository migration procedure

Search all current uses of:

```text
homeostasis-dependent plasticity
homeostatic state
homeostatic potential
HDP
H state
H-state
H factor
H-factor
```

Classify each occurrence:

```text
A. generic doctrine requiring migration
B. kernel-specific homeostatic behavior that should remain homeostatic
C. historical material to preserve/archive as historical
D. API identifier requiring compatibility handling
```

Do not mechanically replace every occurrence of "homeostatic."

## 12. API compatibility

Do not break stable APIs merely to rename concepts. Legacy public identifiers
(`enable_hdp`, `hdp_params`, `DEFAULT_HDP`, `h_state_locality`, …) remain
compatibility surfaces unless evidence justifies a versioned migration.

## 13. Documentation targets

Upstream project sources and repository docs listed in the agent handout.
Requirements: one definition per concept; RBS/RBD/HDP used consistently;
proxy-safe readout labels retained; homeostasis as possible RBD property.

## 14. Scientific invariants

1. Repository state beats remembered context.
2. Every emitter exposes enough state for correct Markov continuation.
3. Relative coordinates retain physical provenance where physical claims may later be made.
4. Proxy readouts remain proxies until calibrated/validated.
5. Geometry-dependent calibration occurs at an explicit boundary.
6. A normalized projection is not a physical PDE solve.
7. Homeostasis, adaptation, memory, plasticity, prediction, and surprise minimization are distinct claims.
8. RBS/RBD/HDP unify mechanisms only where equations and falsification justify it.
9. Quantization is implementation optimization, not scientific evidence.
10. Qualitative figures do not substitute for quantitative metrics.

## 15. Desired conceptual grammar

```text
Physical internal state z
        |
        | normalization / reduction
        v
Relative Biophysical State H (RBS)
        |
        | Relative Biophysical Dynamics (RBD)
        v
Emitter activity/state x
        |
        +----> optional Hidden-state Dependent Plasticity (HDP) ----> W / parameters
        |
        v
Source -> Field -> Probe -> Objective -> Optimizer
        |
        v
Manifest / Validation
```

Compactly:

\[
\boxed{
\mathbf z
\rightarrow
\mathbf H_{\rm RBS}
\overset{\rm RBD}{\longleftrightarrow}
\mathbf x
\overset{\rm HDP}{\longrightarrow}
\mathbf W
}
\]

embedded inside TFNE.

## 16. Stop rules

Stop and report rather than silently resolve if:

```text
live implementation contradicts proposed state semantics
rename would break stable public API
relative transformation destroys calibration-relevant information
an RBS coordinate has no defensible physical/reduced-state interpretation
quantization changes qualitative dynamics
an HDP rule permits unintended negative/undefined physical weights
a memory claim lacks quantitative retention/decoding evidence
surprise minimization is asserted rather than derived/tested
```

Do not invent equations to make doctrine appear complete.
