# TFNE Theory, NeuralTensor, RBS/RBD/HDP

Repository-facing doctrine: `docs/doctrine/rbs_rbd_hdp.md`. This file is the
upstream mathematical authority for the RBS/RBD/HDP semantic migration.

## 1. TFNE as an operator factorization

Tensor-Field Neural Equations (TFNE) are best treated as a typed composition rather than a claim that every neural model obeys one physical field equation.

For parameters and metadata

\[
\Theta=(\theta,\psi,\gamma,\eta),
\]

define four principal operators:

\[
E_\theta: (x_t,u_t,\xi_t)\mapsto x_{t+1},
\]

\[
S_\psi:x_t\mapsto q_t,
\]

\[
F_\gamma:q_t\mapsto\phi_t,
\]

\[
P_\eta:\phi_t\mapsto y_t.
\]

The observable-like TFNE map is

\[
\mathcal T_\Theta=P_\eta\circ F_\gamma\circ S_\psi\circ E_\theta,
\]

and over a trajectory

\[
Y_{0:T}=\mathcal T_\Theta(X_0,U_{0:T},K).
\]

This factorization is the scientific invariant of jaxfne. Individual emitters, source maps, projections, probes, objectives, and optimizers are replaceable implementations of these typed roles.

## 2. Emitter

The emitter contains local neural dynamics coupled to optional hidden relative
biophysical state (RBS) and optional persistent-parameter plasticity (HDP).

### 2.1 Activity and RBS

Let \(\mathbf x_i\) denote fast neural activity/state (membrane voltage,
recovery variables, conductances, spikes, etc.). Let

\[
\mathbf H_i(t)\in\mathbb R^{d_H}
\]

denote **Relative Biophysical State (RBS)** — a finite-dimensional reduced
representation of physically realizable internal emitter state. RBS is
**relative biophysical state**, not necessarily a normalized copy of
individually identifiable physical variables. Two supported realizations:

\[
H_{ik}=\frac{z_{ik}}{z_{ik}^{*}},
\qquad
H_{ik}=\mathcal R_k(\mathbf z_i),
\]

where \(z_{ik}\) is a physical or quasi-physical quantity, \(z_{ik}^{*}\) a
declared reference (nominal \(H_{ik}=1\)), and \(\mathcal R_k\) a reduction map
over several unresolved physical variables.

**Relative Biophysical Dynamics (RBD)** is the general coupled evolution:

\[
\dot{\mathbf x}_i = F_x(\mathbf x_i,\mathbf I_i;\mathbf H_i),
\qquad
\dot{\mathbf H}_i = F_H(\mathbf H_i,\mathbf x_i,\mathbf I_i,\mathbf W,\mathbf u_i,\ldots).
\]

RBD remains meaningful with fixed weights (\(\dot W=0\)). Adaptation, history
dependence, fading memory, phase memory, and delayed recurrent coupling do
**not** require plasticity by definition.

Homeostasis is a possible **property or regime** of selected \(F_H\) or
\(F_x\) realizations (e.g. a kernel whose mechanism is genuinely homeostatic).
It is **not** the definition of RBS.

### 2.2 Recurrent input and delay history

For recurrent coupling with finite transmission delay,

\[
\mathbf I_i(t)=\sum_j \mathcal C_{ji}\bigl(\mathbf x_j(t-\tau_{ji}),W_{ji}\bigr).
\]

When \(\tau_{ji}>0\), the presynaptic spike/event history is part of the
Markov state.

### 2.3 Full Markov state and continuation

In discrete simulation form the emitter step is

\[
(\mathbf x_{t+1},\mathbf H_{t+1},\ldots)=E_\theta(\mathbf x_t,\mathbf H_t,\mathbf u_t,\boldsymbol\xi_t;\Delta t).
\]

For a recurrent network, the **actual Markov state** must include every variable
needed to compute the next step. Write the complete dynamical state as

\[
\boxed{
\mathcal X_t = (\mathbf x_t,\mathbf H_t,\mathbf W_t,\mathcal B_t,\ldots)
}
\]

where \(\mathcal B_t\) is **delay history** (spike/event buffers) whenever
nonzero edge delays are enabled; synaptic filter state, previous spikes,
conductances, and other kernel-specific carries belong here as required.

Continuation is exact only when this full state is preserved. Initializing
only \((\mathbf H,\mathbf W)\) is partial parameter/hidden-state initialization
unless equivalence to full continuation is proven.

Jaxley can supply a more detailed differentiable emitter. jaxfne should bridge
its output rather than reproduce its channel/morphology machinery.

## 3. Source operator

The source operator maps neural state into the quantity acted on by the field/readout operator:

\[
q_t=S_\psi(x_t).
\]

The source is not automatically current in amperes. Its semantics depend on the emitter and calibration.

A source contract declares:

\[
\psi=(\text{mode, gain, sign, support, normalization, calibration}).
\]

A run should use one coherent source interpretation. If synaptic contributions are already contained in a total membrane-current source, adding them again as a separate source term is double counting.

## 4. Field operator

### 4.1 Proxy regime

A common TFNE field proxy is a fixed linear map:

\[
\phi_t=K_\gamma q_t.
\]

This is useful because it is explicit, differentiable, composable, and testable. It does not by itself solve a physical field equation.

Linearity implies

\[
F(aq_1+bq_2)=aF(q_1)+bF(q_2),
\]

which should be property-tested for operators declared linear.

### 4.2 PDE regime

A physical/numerical field candidate instead starts from a governing equation. Schematically,

\[
\mathcal L_\gamma[\phi]=q,
\]

with boundary/reference conditions. A discretization yields

\[
A_\gamma\phi=q.
\]

Calling this a solver requires an actual numerical solution plus residual/convergence evidence. An experimental PDE implementation remains experimental until its geometry, units/calibration and external validity are established.

### 4.3 Orthogonal evidence axes

Do not collapse these into one label:

1. operator type: projection vs PDE solve;
2. numerical validation: experimental vs validated;
3. amplitude semantics: relative vs calibrated.

## 5. Probe operator

A probe samples/transforms the field or source representation:

\[
y_t=P_\eta(\phi_t).
\]

`P` includes channel selection, sensor/contact geometry, referencing, finite differences, leadfield-like maps, or other declared readout transforms.

Thus an LFP-like or EEG-like output is not a new emitter state; it is a derived readout whose interpretation depends on `S`, `F`, and `P` jointly.

## 6. NeuralTensor

`NeuronalTensor` is a structured circuit specification, not the full dynamical state tensor.

Conceptually define a circuit as

\[
\mathcal N=(\mathcal A,\mathcal L,\mathcal C,\mathcal G,\mathcal R),
\]

where:

- \(\mathcal A\): areas;
- \(\mathcal L\): layers within areas;
- \(\mathcal C\): neuron/cell types and fractions;
- \(\mathcal G\): geometry/positions;
- \(\mathcal R\): inter/intra-population connection rules.

Compilation maps the declarative tensor into an executable model:

\[
C_N:\mathcal N\rightarrow M.
\]

The `Configuration` tier supplies another constructor

\[
C_C:\mathcal C_{fg}\rightarrow M.
\]

Both land in the executable model space. There is no requirement that a lossless bijection exists between their input specification spaces.

## 7. Local nonlinear / global structured split

TFNE is especially useful when local dynamics are nonlinear but downstream source/readout operators have exploitable structure.

Local nonlinear examples:

```text
membrane dynamics (F_x)
relative biophysical state dynamics (RBD / F_H)
channel/receptor states
spike/reset rules
hidden-state-dependent plasticity (HDP / F_W)
kernel-specific homeostatic regulation (a possible RBD regime)
local adaptation
```

Potentially global structured operators:

```text
sparse recurrent coupling
source aggregation
fixed laminar projection
fixed leadfield-like projection
probe sampling
```

Do not force an operator to remain linear if its kernel depends on the evolving state. A state-dependent field/readout operator is a different mathematical object and should be represented explicitly.

## 8. RBS, RBD, and HDP

The hidden-state hierarchy inside TFNE is:

```text
TFNE
├── RBS — Relative Biophysical State (H)
├── RBD — Relative Biophysical Dynamics (F_x, F_H, …)
└── HDP — Hidden-state Dependent Plasticity (F_W when enabled)
```

Conceptual flow:

\[
\mathbf z_i
\xrightarrow{\mathcal N/\mathcal R}
\mathbf H_i\in\mathbb R^{d_H}
\xrightarrow{\mathrm{RBD}}
\mathbf x_i,
\]

with optional

\[
\dot{\mathbf W}=F_W(\mathbf W,\mathbf H,\mathbf x,\ldots)\qquad\text{(HDP)}.
\]

Relative computation doctrine: compute in relative coordinates where
information-preserving; calibrate physically only at explicit boundaries
(see `docs/doctrine/rbs_rbd_hdp.md`).

### 8.1 RBS

RBS coordinates should admit a plausible biophysical or reduced-state
interpretation but need not be uniquely identifiable or calibrated. The scalar
toy model is \(d_H=1\).

### 8.2 RBD

A canonical sparse edge-list realization uses synaptic income/spending and
optional restoring control in \(F_H\), for example

\[
\tau_{H,i}\dot H_i=f_H(H_i,r_i,I_i,W_i;\alpha),
\]

with detailed terms supplied by the selected kernel. **Kernel-specific
homeostatic mechanisms** (e.g. `homeostatic_ei`, legacy per-neuron
excitability traces) retain homeostatic terminology for that mechanism; they
are particular RBD realizations, not the definition of RBS.

For delayed recurrence, RBD includes the delay operator in \(\mathbf I_i(t)\)
and the delay buffer \(\mathcal B_t\) in \(\mathcal X_t\).

### 8.3 HDP — Hidden-state Dependent Plasticity

**HDP** means persistent parameter dynamics that depend on RBS (and possibly
activity), not generic homeostasis:

\[
\dot W_{ij}=F_W(W_{ij},\mathbf H_i,\mathbf H_j,\mathbf x_i,\mathbf x_j,\ldots).
\]

For the canonical difference-family weight equation on edges \(i\rightarrow j\),
write

\[
\Delta H_{ij}=H_j-H_i,
\qquad
w_{ij}=q_{ij}m_{ij},\qquad q_{ij}\in\{-1,+1\},\qquad m_{ij}=|w_{ij}|,
\]

\[
\dot m_{ij}
=q_{ij}K_{\mathrm{HDP}}\varphi(\Delta H_{ij})m_{ij}
+K_{w,\mathrm{ctrl}}(m^0_{ij}-m_{ij}).
\]

Here \(\varphi(x)=x\) for `signed_linear` and \(\varphi(x)=x|x|\) for
`signed_quadratic`. `hebbian_product`,

\[
\dot m_{ij}^{product}
=q_{ij}K_{\mathrm{HDP}}H_iH_jm_{ij}
+K_{w,\mathrm{ctrl}}(m^0_{ij}-m_{ij}),
\]

is a separate product modulation, not another approximation to
\(\Delta H_{ij}\). Public API names such as `enable_hdp`, `K_HDP`, and
`DEFAULT_HDP` remain compatibility surfaces; the acronym now denotes
**Hidden-state Dependent Plasticity**.

### 8.4 Null consistency

Do not use bare “HDP off” for distinct null experiments. Define:

\[
N_W^{\mathrm{HDP}}:
\quad
q_{ij}K_{\mathrm{HDP}}\varphi(\Delta H_{ij})m_{ij}=0,
\]

which is the **HDP weight-update null**;

\[
N_H:
\quad
\dot H_i=0,\qquad C_{\mathrm{spike}}s_i=0,
\]

which is the **RBS-dynamics null** (fixed hidden state, no spike spending); and

\[
N_{\mathrm{system}}:
\quad
(\mathcal X_t,W_t)_{\mathrm{HDP}}
=(\mathcal X_t,W_t)_{\mathrm{baseline}}
\]

under matched initial full state, inputs, and PRNG. \(K_{\mathrm{HDP}}=0\)
establishes only the first null when the independent weight-restoration term
is also null; it does not establish \(N_H\) or \(N_{\mathrm{system}}\).

### 8.5 Boundedness

For the supported parameter domain, establish or explicitly enforce

\[
|H_i(t)|<\infty,
\qquad
W_{min}\le W_{ij}(t)\le W_{max}
\]

or provide an equivalent stability condition. Projection/clipping establishes
numerical boundedness, not convergence or asymptotic stability. Distinguish
bounded, locally stable, asymptotically stable, and empirically stable over a
specified horizon. Long-run stability must be measured per supported preset
rather than assumed.

### 8.6 Full-state continuation

Exact segmented continuation requires the complete \(\mathcal X_t\) for the
selected kernel, including \(\mathcal B_t\) when delays are enabled. Example
carry for an edge-list HDP kernel:

\[
\mathcal X_t=(v_t,u_t,s_{t-1},z_t^{\mathrm{syn}},\mathbf H_t,W_t,\mathcal B_t),
\]

with component names matching the implementation. Initializing only
\((\mathbf H,W)\) is not automatically equivalent.

### 8.7 Timescale structure

RBS/HDP systems are naturally multiscale. If

\[
\tau_{\mathrm{neural}}\ll\tau_H,\tau_W,
\]

fast neural activity evolves against slowly changing hidden/plastic variables.
This supports analysis of adaptation, recovery, omission, and long-/short-
timescale effects without claiming that the chosen equations are a validated
biological mechanism.

## 9. Objectives and optimization

Given readout `Y`, define

\[
L=O(Y,T,G,N).
\]

An optimizer/search operator proposes

\[
\Theta_{k+1}=A(\Theta_k,L_k,\mathcal C,K_k).
\]

This closes the computational loop:

\[
\Theta\rightarrow E\rightarrow S\rightarrow F\rightarrow P\rightarrow O\rightarrow A\rightarrow\Theta'.
\]

The manifest/validation layer is outside the mathematical state update but inside the scientific method: it records enough information to reproduce and audit the loop.

## 10. Differentiability

When each selected operator is differentiable almost everywhere, TFNE supports gradients through the composition:

\[
\frac{\partial L}{\partial\theta}
=
\frac{\partial L}{\partial Y}
\frac{\partial P}{\partial\phi}
\frac{\partial F}{\partial q}
\frac{\partial S}{\partial x}
\frac{\partial E}{\partial\theta}.
\]

Spike/reset discontinuities may require surrogate or piecewise treatment depending on the emitter. Do not claim end-to-end differentiability for an arbitrary pipeline without a gradient test of the selected path.

## 11. Validation hierarchy

A strong TFNE result climbs this ladder:

```text
mathematical well-posedness
-> finite deterministic execution
-> operator/property tests
-> nulls and ablations
-> numerical convergence where relevant
-> source/field calibration where relevant
-> external-tool comparison
-> held-out empirical comparison
-> mechanism support
```

Each rung enables stronger language; no rung is implied merely by the existence of later software infrastructure.

## 12. Core theorem-like invariants to test

For supported deterministic configurations:

1. **Reproducibility:** identical configuration + key gives identical outputs within defined backend tolerance.
2. **Finite-state closure:** valid finite inputs do not silently yield NaN/Inf.
3. **Linear-operator property:** declared linear field/probe operators satisfy superposition numerically.
4. **Zero-source property:** for zero-offset linear operators, `F(0)=0`.
5. **Shape closure:** operator output axes match their declared tensor contracts.
6. **Dense/edge equivalence:** equivalent connectivity representations produce equivalent source/dynamics within tolerance where algorithms are intended to be equivalent.
7. **Continuation equivalence:** one uninterrupted run equals segmented full-state continuation within tolerance.
8. **Null consistency:** disabling an optional mechanism removes its contribution without changing unrelated semantics.
