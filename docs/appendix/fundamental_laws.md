# Fundamental-Law Appendix: From Charge Conservation to JaxFNE

**Status:** Informational (derivation) — Δscience = 0. No kernel, solver, or calibration status changes. This appendix walks from elementary physical laws to the JaxFNE equations that appear in the glossary and pipeline; it explicitly marks where the physical derivation ends and the JaxFNE modeling abstraction begins.

**Companion maps:** [Source and field equations](../source_field_equations.md) · [Mathematical glossary flow](../mathematical_glossary_flow.md) · [Computation basis](../computation_basis.md) · [References](../reference/references.md) · [Limitations and future plans](../limitations_and_future_plans.md)

> **Scope guard.** JaxFNE is a `computational_scaffold` with `field_solver_status = "linear_solver"` and `physical_amplitude_calibrated = False` by default. Every field/EEG/MEG/EMM/LFP/CSD output is a `proxy_readout` in relative proxy units, an uncalibrated physical claim — without physical amplitude calibration. This appendix is a **derivation narrative** that makes the physics→model steps explicit; it preserves the existing claim level, conductivity, and solver status without elevation.

> **What JaxFNE does not numerically solve in the shipped configuration.** JaxFNE does **not** solve the full Maxwell system, the Poisson/elliptic volume-conductor PDE, the Nernst–Planck ion electrodiffusion system, or a calibrated cable equation per compartment. Those laws are declared for reference; the shipped forward path evaluates **reduced emitters → linear source projection → Gaussian proxy field → probe readouts** as explicit computational proxies defined by kernel equations. Reserved PDE-solve regimes are catalogued in [Limitations](../limitations_and_future_plans.md).

---

## 1. How to read this appendix

For each equation class the appendix uses a three-part pattern:

1. **Physical law** — the textbook law and its domain of validity.
2. **Reduction / specialization** — the routinely-invoked biophysical specialization (e.g. Hodgkin–Huxley, single-exponential synapse).
3. **JaxFNE modeling abstraction** — the declared computational operator actually evaluated, with the explicit boundary where derivation ends.

A section-boundary box such as

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** …

accompanies every equation so that no proxy operator is mistaken for a physical law that has been solved.

All block equations use `\[ … \]` and inline equations use `\( … \)` per the `pymdownx.arithmatex` configuration in `mkdocs.yml`.

---

## 2. Elementary charge and current — \(I = dq/dt\)

### 2.1 Physical law

Electric charge \(q\) (coulomb, C) is conserved. In differential form charge conservation is the continuity equation

\[
\frac{\partial \rho}{\partial t} + \nabla\!\cdot\!\mathbf{J} = 0,
\]

where \(\rho(x,t)\) is volume charge density (C·m⁻³) and \(\mathbf{J}(x,t)\) is current density (A·m⁻²). For a control volume \(V\) bounded by surface \(S\),

\[
\frac{d}{dt}\int_V \rho\, dV = -\oint_S \mathbf{J}\!\cdot\! d\mathbf{S}.
\]

The elementary current through a surface is the rate of charge crossing it:

\[
I = \frac{dq}{dt}, \qquad [I] = \text{A} = \text{C·s}^{-1}.
\]

This is a definition grounded in Maxwell/continuity physics. It holds irrespective of the charge carrier (ions, electrons) or the medium.

### 2.2 Biophysical specialization

In neural tissue the relevant currents are ionic (Na⁺, K⁺, Ca²⁺, Cl⁻, …), capacitive (displacement of charge across the lipid bilayer), and synaptic (transmitter-gated channels). Their algebraic sum obeys Kirchhoff's current law at the membrane: transmembrane current entering the extracellular volume equals current leaving the intracellular compartment.

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** JaxFNE does not integrate a charge-continuity PDE to obtain \(I\). Individual emitter kernels declare a **model current** `I(t)` (e.g. Izhikevich `current_native` plus spike impulse proxy) whose time integral is *interpreted* as charge moved, not derived from electrodiffusion. The quantity `source_proxy` in `Signals` is a relative-unit proxy, `source_calibration_status = "uncalibrated_izhikevich_native_current"`.

---

## 3. Membrane current balance — \(C_m\, dV/dt = I_{\mathrm{ext}} - I_{\mathrm{ion}} - I_{\mathrm{syn}}\)

### 3.1 Physical law: the bilayer as a capacitor

A patch of cell membrane of area \(A\) separates intracellular and extracellular electrolytes by a ~5 nm dielectric. To first order it is a parallel-plate capacitor with specific capacitance

\[
C_m = \frac{\epsilon}{d} \approx 1\,\mu\text{F·cm}^{-2},
\]

so that stored charge and transmembrane voltage satisfy

\[
q = C_m\, V, \qquad I_{\mathrm{cap}} = C_m \frac{dV}{dt}.
\]

This is classical electrostatics applied to the membrane.

### 3.2 Adding ionic and synaptic branches

Hodgkin and Huxley [Hodgkin & Huxley 1952](../reference/references.md#neural-dynamics) measured and parameterized voltage-gated Na⁺ and K⁺ currents as

\[
I_{\mathrm{ion}} = \sum_k g_k(V,t)\,(V - E_k),
\]

with \(g_k\) the conductance and \(E_k\) the Nernst reversal potential of ion \(k\). Synaptic input adds a chemically-gated branch \(I_{\mathrm{syn}} = \sum_j g_{\mathrm{syn},j}(t)\,(V - E_{\mathrm{syn},j})\). External (injected / electrode / stimulus) current is \(I_{\mathrm{ext}}\). Kirchhoff's law at the patch gives the familiar balance

\[
C_m \frac{dV}{dt} = I_{\mathrm{ext}} - I_{\mathrm{ion}} - I_{\mathrm{syn}},
\]

or, equivalently,

\[
I_{\mathrm{cap}} + I_{\mathrm{ion}} + I_{\mathrm{syn}} = I_{\mathrm{ext}}.
\]

This is the **single-compartment** statement. For extended morphology the same balance becomes the cable equation; see §12.

### 3.3 How this law is used — and where JaxFNE stops solving it literally

A full biophysical simulator would at each timestep evaluate \(g_k(V,t)\) from gating ODEs (\(\dot m, \dot h, \dot n\)), compute each \((V-E_k)\) driving force, sum branches, and integrate \(V\) with a stiff solver (\(dt \sim 0.01\)–\(0.025\) ms). JaxFNE pathway:

| Layer | Equation evaluated in shipped code | Physical terms kept literally? |
|---|---|---|
| `C_m dV/dt` | Absorbed into reduced dynamics \((v,u)\) | No — no explicit farads; \(v\) in internal proxy units |
| \(I_{\mathrm{ion}}\) | Summarized inside \(\dot v = 0.04v^2+5v+140 - u + I\) (see §4) | No — kinetics are phenomenological |
| \(I_{\mathrm{syn}}\) | Linear filtered spike input `syn_state * exp(-dt/tau) + spike` | Yes — as a linear filter, not as \(g(V-E)\) conductance |
| \(I_{\mathrm{ext}}\) | `drive` + `drive_schedule` + `noise` | Yes — as an abstract current-like drive |

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The shipped JaxFNE emitters do **not** time-step \(C_m dV/dt = I_{\mathrm{ext}} - I_{\mathrm{ion}} - I_{\mathrm{syn}}\) with calibrated \(C_m\) (µF·cm⁻²), physical \(g_k\) (mS·cm⁻²), and mV-referenced \(E_k\). They time-step a declared reduced system \((z \equiv (v,u))\) whose right-hand side is chosen to *reproduce spiking statistics* of the physical system. Reversal potentials appear only as metadata in `ReceptorSpec` (`reversal_mV`), not in the dynamics (`docs/api/emitters.md`). Physical units are recoverable only via an explicit calibration map \(\mathcal C\) that is not applied by default.

**References:** Hodgkin & Huxley 1952; FitzHugh 1961; Morris & Lecar 1981; Brette & Gerstner 2005; reviews in Pettersen et al. 2012; Einevoll 2020 ([References](../reference/references.md)).

---

## 4. Reduced emitters — \(\dot z = F_\theta(z,u,t)\) and the Izhikevich reduction

### 4.1 Why reductions exist

Integrating §3 literally for \(10^2\)–\(10^4\) neurons with heterogeneous channels at \(dt < 0.05\) ms is computationally dominant. The dynamical-systems program (FitzHugh–Nagumo, Morris–Lecar, AdEx, Izhikevich) observes that the HH phase portrait near threshold is essentially two-dimensional — a fast excitation variable and a slower recovery variable. Izhikevich [Izhikevich 2003](../reference/references.md#neural-dynamics) showed that a quadratic integrate-and-fire plus linear recovery,

\[
\begin{aligned}
\frac{dv}{dt} &= 0.04\,v^2 + 5\,v + 140 - u + I(t), \\
\frac{du}{dt} &= a\,(b\,v - u), \\
\text{if } v &\ge 30\text{ mV-like: } v \leftarrow c,\; u \leftarrow u + d,
\end{aligned}
\]

reproduces ~20 cortical firing patterns by tuning only \((a,b,c,d)\) — a reduction justified a posteriori by bifurcation analysis, not by eliminating terms from §3 algebraically.

### 4.2 JaxFNE realization

JaxFNE's shipped emitters (`jaxfne/emitters.py:25`, `docs/api/emitters.md`) are Izhikevich-class:

\[
\frac{dz}{dt} = F_\theta(z, u, t), \qquad z = (v, u) \in \mathbb{R}^2,
\]

evaluated inside `jax.lax.scan` kernels `simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`, etc. The tensor form is component-wise identical. The bridge term \(F_\theta\) is the **Emitter → Source** bridge (see [Mathematical Glossary Flow](../mathematical_glossary_flow.md)).

Default parameters are per cell-type (`E, PV, SST, VIP …`), not per-preset; the `preset` string (e.g. `"cortical_eig"`) is a metadata tag, not a dynamics switch.

Other emitter classes (`GLIFEmitter`, `LIFEmitter`, `homeostatic_ei`) are typed holders for the same grammar point: a declared Markov transition \((\mathbf x_t,\mathbf H_t) \mapsto (\mathbf x_{t+1},\mathbf H_{t+1})\).

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The coefficients \(0.04, 5, 140\) and the threshold \(30\) are **not** measured conductances or Nernst potentials; they are rescaled so that \(v\) has a convenient numerical range for biphasic integration. The shipped kernels do not enforce ionic reversal bounds, Q10 temperature scaling, or gating-variable conservation as numerical invariants. Choosing a different emitter (e.g. a Jaxley compartmental model attached via `jaxfne.bridges`) is an explicit user decision that changes this boundary; it does not change the proxy status of downstream field readouts.

**References:** Izhikevich 2003; FitzHugh 1961; Nagumo et al. 1962; Morris & Lecar 1981; Brette & Gerstner 2005; Hines & Carnevale 1997 (NEURON); Deistler et al. 2025 (Jaxley) ([References](../reference/references.md)).

---

## 5. Synaptic filtering — \(\tau_s \dot s = -s + \sum_k \delta(t - t_k)\)

### 5.1 Physical origin

Transmitter release at times \(\{t_k\}\) opens postsynaptic channels. Full kinetic schemes (destexhe-type) are Markov chains with rise and decay rates. The simplest analytically tractable reduction is a first-order low-pass: each spike deposits a unit impulse that decays exponentially:

\[
\tau_s \frac{ds}{dt} = -s(t) + \sum_k \delta(t - t_k),
\]

with \(\tau_s\) the synaptic time constant. Its solution is the convolution

\[
s(t) = \sum_{k: t_k < t} e^{-(t - t_k)/\tau_s}\, \Theta(t - t_k),
\]

where \(\Theta\) is the Heaviside step. The discrete-time update evaluated in JaxFNE is therefore

\[
s_{n+1} = s_n e^{-dt/\tau_s} + \text{spike}_n,
\]

which is exactly the `syn_state` recurrence in `jaxfne/emitters.py`.

Multiple receptors are multiple decay constants: `standard_receptor_specs()` declares \(\tau_{\mathrm{AMPA}}=2\) ms, \(\tau_{\mathrm{GABA_A}}=5\) ms, \(\tau_{\mathrm{NMDA}}=100\) ms, \(\tau_{\mathrm{GABA_B}}=150\) ms (metadata; kernels instantiate the 2/5 ms E/I split by default, the 100/150 ms entries via `synaptic_tau_from_mechanism` / `synaptic_current_tensor`).

### 5.2 Standalone tensor and report

\[
\texttt{filtered} = \texttt{synaptic\_current\_tensor}(\texttt{spikes\_pre}, \tau_{ms}, dt_{ms}),
\]

is the factored-out per-edge operator (see `docs/api/emitters.md`). Its report carries `source_calibration_status = "metadata_only_uncalibrated"` and `physical_amplitude_calibrated = False`.

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The equation above is linear in \(s\) and ignores saturation, driving-force dependence \((V-E_{\mathrm{rev}})\), short-term depression/facilitation, and NMDA voltage dependence. Those effects belong to a richer kinetic model (not solved in the default path). JaxFNE's default excitatory/inhibitory tau split (2 ms / 5 ms) is a computational choice balanced for 10–100 Hz network dynamics, not a fit to a specific voltage-clamp dataset.

**References:** Standard receptor kinetics reviewed in Pettersen et al. 2012; Lindén et al. 2014; Hagen et al. 2018 ([References](../reference/references.md)).

---

## 6. Relative biophysical state — \(H_k = z_k / z_k^*\)

### 6.1 Definition

Let \(z_k\) be a physical internal quantity of neuron \(i\) (e.g. ionic availability, adaptation variable, vesicle pool, ATP proxy, STDP trace) and \(z_k^*\) its declared reference scale. The **relative biophysical state (RBS)** coordinate is

\[
H_{ik} = \frac{z_{ik}}{z_{ik}^*}, \qquad H_{ik}=1 \text{ at nominal reference},
\]

or, for reduced coordinates subsuming several unresolved variables,

\[
H_{ik} = \mathcal{R}_k(\mathbf z_i).
\]

The full per-neuron RBS vector is \(\mathbf H_i \in \mathbb{R}^{d_H}\). Conventions are typed: multiplicative availability coordinates use \(H^*=1\); deviation/trace coordinates (e.g. activity-history traces, STDP \(H_+, H_-\)) use \(H^*=0\). See [RBS/RBD/HDP](../doctrine/rbs_rbd_hdp.md) and [TFNE Containment Architecture](../doctrine/tfne_containment_architecture.md).

### 6.2 Normalization and calibration maps

Conceptually,

\[
\mathbf z_{\mathrm{physical}} \xrightarrow{\;\mathcal N\;} \tilde{\mathbf z}_{\mathrm{relative}} \xrightarrow{\;\text{TFNE}\;} \tilde{\mathbf y} \xrightarrow{\;\mathcal C\;} \mathbf y_{\mathrm{calibrated}},
\]

where \(\mathcal N\) is normalization (\(z \mapsto H\)) and \(\mathcal C = \mathcal N^{-1}\) is the (geometry-aware) calibration that recovers physical units where justified. JaxFNE preserves \(\mathbf z^*\) and the map \(\mathcal N\) as metadata; it does **not** apply \(\mathcal C\) to readouts by default.

### 6.3 Reduction illustrated: from \(H = z/z^*\) to the scalar Protocol H model

The RBS formalism subsumes the one-dimensional toy used in Protocol H. Setting \(d_H = 1\) and interpreting \(z\) as total recurrent drive recovers the scalar \(H\) dynamics tested in `simulate_edge_recurrent_izhikevich_rbd`:

\[
\tau_H \frac{dH_i}{dt} = (1 - H_i) + \kappa_H\, I_i^{\mathrm{rel}}\quad (f_1\text{ family}), \tag{6.1}
\]

\[
\tau_H \frac{dH_i}{dt} = (H_i^{-1} - 1) + \kappa_H\, I_i^{\mathrm{rel}}\quad (f_2\text{ family}), \tag{6.2}
\]

both of the form \(\dot H = F_H(H, I^{\mathrm{rel}})\) with the relative drive \(I_i^{\mathrm{rel}} = I_i^{\mathrm{rec}}/I_{\mathrm{ref}}\). The coupled gain acting on the emitter is

\[
G_H(H;\beta_H) = 1 + \beta_H\,(H - 1),\qquad I_i^{\mathrm{drive}} = I_i^{\mathrm{ext}} + G_H(H_i)\, I_i^{\mathrm{rec}},
\]

declared in `docs/doctrine/protocol_h_rbd_memory.md` and `docs/doctrine/tfne_containment_architecture.md`. The scalar model is thus **one realization** of \(H = z/z^*\); the same grammar extends to \(d_H \sim 10\)–\(100\) coordinates without changing the containment structure.

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The choice \(z_k^*\) and the reduction map \(\mathcal R_k\) are **modeling decisions**. The code stores \(\mathbf H\) as a JAX array of shape `(n_neurons, d_H)` (or `(n_neurons,)` for \(d_H=1\)) and integrates \(\dot{\mathbf H} = F_H(\mathbf H, \mathbf x, \mathbf I, \mathbf W, \ldots)\) via explicit Euler inside the same `scan` as the emitter. No invariant forces \(H\) to remain near 1; f2 propagation with \(H \le 0\) yields non-finite \(H\) by design so that misuse is observable — it does not silently clip.

---

## 7. Source operator — \(Q = \mathcal S(X, H, W, \ldots)\)

### 7.1 Physical meaning

In volume-conductor theory the field equation is sourced by transmembrane current density. Formally,

\[
q(x,t) = P_s[z(t), I(t), \chi(x)],
\]

where \(P_s\) maps neural state/current into spatial coordinates via the tissue geometry \(\chi(x)\). The pipeline symbol is

\[
Q = \mathcal S(X, H, W, \ldots),
\]

with \(X\) the emitter activity state, \(H\) the RBS, and \(W\) the connectivity/weights; additional arguments may include geometry \(\mathcal G\).

A minimal, widely-used specialization is the sum over point-like emitters:

\[
q_\alpha(t) = \sum_k w_{k\alpha}(x)\, s_k(t),
\]

where \(\alpha\) indexes spatial contacts, \(k\) indexes neurons, \(w_{k\alpha}\) are spatial coupling weights, and \(s_k\) is the source-bearing trace of neuron \(k\).

### 7.2 JaxFNE declaration: proxy sources

The shipped source is the canonical proxy

\[
Q^{(r)} = \texttt{source\_scale}\;\bigl(I_{\mathrm{native}} + g_{\mathrm{spike}}\cdot \texttt{spikes}\bigr),\qquad g_{\mathrm{spike}} = \texttt{DEFAULT\_SPIKE\_IMPULSE\_GAIN}=20,
\]

documented in `docs/api/emitters.md` (Canonical source representation) and assembled via `jaxfne.fields.construct_source_tensor` (which guards against double-counting synaptic current). Its status fields are:

- `source_projection_mode = "proxy_no_field_solve"`
- `source_decomposition = "proxy_reduced_emitter"`
- `source_calibration_status = "uncalibrated_izhikevich_native_current"`
- `source_scale` — explicit scalar preserving the map for later calibration

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The native current \(I_{\mathrm{native}}\) is the internal drive+recurrent+noise sum in Izhikevich proxy units, not amperes per membrane area. The code does **not** enforce \(\nabla\!\cdot\!\mathbf J = -q\) as a numerical conservation constraint on \(Q\); a scalar conservation proxy \(\int q\, dV \approx 0\) is *reported* (see §13), not *enforced*. Declaring a different \(Q\) (e.g. a Jaxley-derived voltage-proxy current) is done via `jaxfne.bridges` and is recorded via `source_calibration_status`, not via the default code path.

---

## 8. Source-to-field projection — \(\Phi = \mathcal K\, Q\)

### 8.1 Physical field equation (declared, not solved by default)

Under the quasi-static approximation (\(|\partial \mathbf B/\partial t|\) negligible for LFP/EEG frequencies; see Vorwerk et al. 2014; Hagen et al. 2018), extracellular potential obeys the elliptic (Poisson-type) volume-conductor equation

\[
\nabla\!\cdot\!\bigl(-\sigma_e(x)\,\nabla \phi_e(x,t)\bigr) = q(x,t),
\]

with Neumann boundary condition \(\mathbf n\!\cdot\!(\sigma_e \nabla \phi_e)=0\) on the outer surface and a mean-zero gauge \(\int \phi_e\, dV = 0\) (required because only voltage *differences* are physically meaningful). In operator notation,

\[
\mathcal L\, \Phi = Q,\qquad \mathcal L \equiv \nabla\!\cdot\!(-\sigma_e \nabla \cdot),\qquad \Phi \equiv \phi_e.
\]

Formally the solution is \(\Phi = \mathcal L^{-1} Q \equiv \mathcal K Q\), where \(\mathcal K = \mathcal L^{-1}\) is the Green's operator (leadfield) encoding geometry and conductivity.

### 8.2 JaxFNE proxy: Gaussian leadfield

The shipped operator replaces the PDE solve by an explicit, geometry-aware linear kernel:

\[
K_{\alpha k} = \exp\!\Bigl[-\tfrac12\Bigl(\tfrac{z_\alpha - d_k}{w}\Bigr)^2\Bigr]\quad\text{or row-normalized variant } K_{\alpha k}/\sum_j K_{\alpha j},
\]

\[
\Phi_{\mathrm{proxy}} = Q\, K^{\!\top},\qquad \phi_{\alpha}(t) = \sum_k K_{\alpha k}\, q_k(t),
\]

with \(z_\alpha\) the contact depth, \(d_k\) the neuron depth in \([0,1]\), and \(w\) the Gaussian width (default \(0.10\) relative-depth units). This is `jaxfne.fields.project_laminar_sources` / `project_sources_to_laminar_field` (`jaxfne/fields/proxy.py:148`).

The kernel is factored out as `FieldOutput.kernel` so that CSD proxy, LFP proxy, etc. can be recomputed without re-running projection.

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** No elliptic PDE is assembled or solved; no matrix \(\mathcal L\) is formed; no iterative solver residual is reported (fields `solver_residual_l2_relative`, `n_iterations`, `converged` are `None` in the proxy branch). Conductivity \(\sigma_e\) is the scalar `proxy` placeholder, not a tensor measured in S·m⁻¹. The equation \(\nabla\!\cdot\!(-\sigma_e\nabla\phi_e)=q\) is **declared metadata** (`field_solver_status = "linear_solver"`, `field_claim_level = "proxy_readout"`) rather than a numerically enforced constraint. Promoting this to a real solve is the `solved_poisson` / `reserved_admittive` / `reserved_maxwell` ladder in [Limitations](../limitations_and_future_plans.md) and [Computation basis](../computation_basis.md) — reserved, not shipped.

---

## 9. Probe and readout operator — \(Y = \mathcal P\,\Phi\)

### 9.1 General form

A probe extracts spatially-localized readouts from the field (and, where appropriate, from the source directly):

\[
Y_c(t) = \mathcal P_c[\phi_e(t), \mathbf J_e(t), \mathrm{CSD}(t), \ldots],
\]

where \(c\) indexes contacts. JaxFNE notation for the factorized laminar pipeline is

\[
Y = \mathcal P\,\Phi,\qquad \text{with readout family }\mathcal P \in \{\text{source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy, }\phi_e\text{-proxy, }J_e\text{ status}\}.
\]

### 9.2 CSD as a probe example

Current-source density is, physically, \(\mathrm{CSD} \equiv \nabla\!\cdot\!\mathbf J_e = -\nabla\!\cdot\!(\sigma_e \nabla \phi_e)\). In the laminar proxy the field analogue is the discrete second derivative along depth:

\[
\mathrm{CSD}_{\mathrm{proxy}}[c] = -\frac{\phi_{c+1} - 2\phi_c + \phi_{c-1}}{dz^2},
\]

edge-padded at the boundaries (`jaxfne/fields/proxy.py:1028`, `csd_tensor`). Sign convention in JaxFNE: `positive_equals_extracellular_source` (positive = outward transmembrane current, see [Source and field equations](../source_field_equations.md)).

Other probes are linear readouts \(y = \text{source} \cdot W^{\top}\) with a toy/declared leadfield \(W\) (`LinearReadout`, `eeg_proxy_transform`, `meg_proxy_transform`, `emm_proxy_transform`).

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** The shipped probes are **simulated proxies**: `operator_status = "simulated_proxy"`, `leadfield_status = "toy_or_declared_proxy"`, `units_or_status = "relative_proxy_units"`. Return values are reported with `physical_amplitude_calibrated = False` and are not in pA, mV, or µA·mm⁻³. The EMM-proxy in particular is a normalized signaling-energy metric, not ATP consumption.

---

## 10. Hidden-state dependent plasticity — \(\dot W = F_W\)

### 10.1 Physical motivation

Synaptic efficacy depends on spike timing (STDP), postsynaptic depolarization, neuromodulation, and resource state. Phenomenologically one writes a rate that depends on activity and possibly on hidden biophysical state:

\[
\dot W_{ij} = F_W(W_{ij}, \mathbf H_i, \mathbf H_j, \mathbf x_i, \mathbf x_j, \ldots).
\]

Two canonical realizations unified under the TFNE containment `docs/doctrine/tfne_containment_architecture.md` are:

- **STDP traces:** \(\tau_+ \dot H_+ = -H_+ + S_{\mathrm{pre}},\; \tau_- \dot H_- = -H_- + S_{\mathrm{post}}\), and \( \Delta W \propto H_+ S_{\mathrm{post}} - H_- S_{\mathrm{pre}}\).
- **BCM:** \(\dot W \propto \phi(\mathbf x, H_\theta)\, \mathbf x\), with sliding threshold \(H_\theta\) as a scalar RBS coordinate.

JaxFNE governance: [RBS/RBD/HDP](../doctrine/rbs_rbd_hdp.md) and [HDP guide](../guides/hdp.md); realization docs `docs/doctrine/protocol_w_hdp_parameter_memory.md`.

### 10.2 JaxFNE realization and ladder

RBD alone already yields state memory with \( \dot W = 0\) — the definition of Protocol H — so adaptation and fading memory do **not** require plasticity. HDP is the optional slower loop \(\mathbf x \leftrightarrow \mathbf H \leftrightarrow \mathbf W\). The `emitters_homeostatic_ei` ladder stages \((x,G,H)\) with independent \(( \tau_x, \tau_G, \tau_H )\), exercised separately from the Izhikevich emitters.

> **Boundary — Physical derivation ends here. JaxFNE abstraction begins:** Specific shipped rules (e.g. `signed_linear`, `hebbian_pairwise`, `cubic_penalty`) are **declared** phenomenological candidates, not measurements of a specific biological STDP curve. The weight update defaults in `simulate_edge_recurrent_izhikevich_hdp` are a null/identity configuration unless explicit gains (`alpha, beta, gamma, delta, C_spike, K_HDP`) are set. Numerical optimization (AGSDR, Adam) is a distinct operator class `A` and is not HDP; conflating them violates `docs/doctrine/rbs_rbd_hdp.md`.

---

## 11. Consolidated boundary map

| # | Physical law (textbook) | Canonical equation | Physical derivation ends — JaxFNE evaluation begins |
|---|---|---|---|
| 1 | Charge conservation / continuity | \(I = dq/dt\), \(\partial_t \rho + \nabla\!\cdot\!\mathbf J = 0\) | JaxFNE declares model current `I(t)` as proxy; no continuity PDE solved |
| 2 | Capacitive membrane + Hodgkin–Huxley branches | \(C_m dV/dt = I_{\mathrm{ext}} - I_{\mathrm{ion}} - I_{\mathrm{syn}}\) | Replaced by \(\dot z = F_\theta(z,u,t)\) with phenomenological \(F_\theta\) |
| 3 | Reduced spiking dynamics | Izhikevich (2003) reduction from HH | Coefficients are rescaled, not physical conductances; \(v\) in proxy units |
| 4 | Synaptic kinetics | \(\tau_s \dot s = -s + \sum \delta(t-t_k)\) | Linear single-exponential filter; no \(g(V-E_{\mathrm{rev}})\), no saturation |
| 5 | Relative biophysical state | \(H_k = z_k/z_k^*\) or \(\mathcal R_k(\mathbf z)\) | \(\mathcal N\) preserves map; \(\mathcal C\) not applied; \(d_H=1\) scalar is one realization |
| 6 | Source density | \(Q = \mathcal S(X,H,W,\ldots)\); \(q = P_s[z,I,\chi(x)]\) | \(Q^{(r)} = \texttt{source\_scale}(I_{\mathrm{native}}+g_{\mathrm{spike}}\,\texttt{spikes})\) with uncalibrated status |
| 7 | Quasi-static field equation | \(\nabla\!\cdot\!(-\sigma_e\nabla\phi_e)=q\), \(\Phi=\mathcal K Q\) | Gaussian kernel \(K_{\alpha k}=\exp[-0.5((z_\alpha-d_k)/w)^2]\); no PDE solve; `field_solver_status = "linear_solver"` |
| 8 | Probe / readout | \(Y = \mathcal P\Phi\); CSD \(= -\nabla\!\cdot\!(\sigma_e\nabla\phi_e)\) | Discrete proxy CSD \(=-(\phi_{c+1}-2\phi_c+\phi_{c-1})/dz^2\); all probes `proxy_readout` |
| 9 | Plasticity | \(\dot W = F_W(W,\mathbf H,X,\ldots)\) | Declared rule (`signed_linear`, `hebbian`, …); RBD (\(\dot W=0\)) memory is already falsifiable without it |

---

## 12. What the shipped package leaves outside the default solve — and why it still matters

Explicitly **not solved numerically** in the default `linear_solver` configuration:

- Maxwell's equations or Poynting-flux dynamics (\(\partial_t u_{\mathrm{em}} + \nabla\!\cdot\!\mathbf S + \mathbf J\!\cdot\!\mathbf E = 0\) appears in [Mathematical Glossary Flow](../mathematical_glossary_flow.md) as *conservation motivation* for diagnostics, not as an integrated physics solve; reserved Maxwell/admittive regimes in `docs/tensor_electromagnetics_scope.md`).
- The elliptic volume-conductor PDE for \(\phi_e\) (reserved `solved_poisson`).
- Nernst–Planck/drift-diffusion for ions or full cable-equation morphology (compartmental detail is imported via Jaxley bridges when desired, not via the Izhikevich default).
- Thermodynamic/metabolic energy accounting (EMM-proxy is a normalized cost proxy, section §9).
- Any calibrated physical amplitude mapping (requires geometry, conductivity, electrode model, and empirical reference data per [Calibration guide](../guides/calibration.md)).

What **is** evaluated, reproducibly and differentiably in JAX:

- A complete, deterministic dynamical chain \(X \to H \xrightarrow{S} Q \xrightarrow{\mathcal K} \Phi \xrightarrow{\mathcal P} Y\) with explicit shapes \([T,N] \to [T,X] \to [T,R]\), finite-valued outputs, and JSON-safe manifests (see [Computation basis](../computation_basis.md)).
- Typed, replaceable operators — any stage can be upgraded to a more physical realization (e.g. Jaxley emitters, FEM field solve) inside the same grammar without breaking the pipeline rule (see [TFNE Containment](../doctrine/tfne_containment_architecture.md)).
- Proxy diagnostics that audit the scaffold (conservation proxy residual, gradient norms, bandpower) while remaining at `claim_level = "computational_scaffold"`.

---

## 13. Dimensions, units, and the calibration boundary

**Shipped units are relative proxy units** by construction. A physical-unit claim would require all of:

\[
\mathcal G\;(\text{geometry})\; +\; \sigma_e\;(\text{conductivity})\; +\; \mathcal K_{\mathrm{leadfield}}\; +\; \text{electrode model}\; +\; \text{empirical reference}\; +\; \mathcal C\;(\text{calibration map})\; +\; \text{validation residual}.
\]

The current proxy keeps these separated so that adding them later is an explicit, evidenced step rather than a silent unit change. In particular:

- Emitter state \(v\) is "mV-like" but not mV; its scale is set by the Izhikevich rescaling, not by a patch-clamp calibration.
- Synaptic state \(s\) is dimensionless in the filter equation; its product with \(W\) is in internal weight·activity units.
- Source \(Q\) is in `source_proxy` units (`source_scale` preserves the relative→physical map without applying it).
- Field \(\Phi\) and readouts \(Y\) are in `relative_proxy_units` until a `CalibrationSpec` with `amplitude_status = true` and accompanying validation evidence is presented — which the shipped code never auto-elevates.

The governing rule is [Relative-Quantity Grammar](../doctrine/relative_quantity_grammar.md):

\[
\boxed{\text{Compute relatively; calibrate physically; preserve the map.}}
\]

---

## 14. References

This appendix cites the canonical bibliography at [References](../reference/references.md). Key entries invoked by equation:

- **Charge/membrane/HH basis:** Hodgkin & Huxley 1952; FitzHugh 1961; Nagumo et al. 1962; Morris & Lecar 1981; Brette & Gerstner 2005.
- **Reduced/population dynamics:** Wilson & Cowan 1972; Montbrió et al. 2015; Coombes 2005; Cook et al. 2022.
- **Extracellular field and observation:** Lindén et al. 2014; Hagen et al. 2018; Herreras 2016; Pettersen et al. 2012; Einevoll 2020; Vorwerk et al. 2014.
- **Compartmental/physical emitters (when opting out of the reduction):** Hines & Carnevale 1997; Deistler et al. 2025; Cannon et al. 2014 (LEMS/NeuroML).
- **RBS/RBD/HDP governance and containment:** `docs/doctrine/rbs_rbd_hdp.md`; `docs/doctrine/tfne_containment_architecture.md` (RBS definition \(H_k = z_k/z_k^*\), containment grammar).
- **Calibration and reproducibility:** Wilkinson et al. 2016 (FAIR); Nosek et al. 2015; Heil et al. 2021; Rübel et al. 2022 (NWB).

Every reference above is DOI-resolved in [References](../reference/references.md); no new bibliographic entries are introduced here.

---

## 15. Document history

- 2026-08-28 — Created as an informational derivation appendix at `dev` head `aad8ce5` (requested derivation baseline `350730a`). Δscience = 0; no code, solver, or `physical_amplitude_calibrated` changes. Equations verified against `mkdocs.yml` arithmatex delimiters `\(…\)` / `\[…\]` and against existing rule files cited above.
