# Mathematical Glossary Flow

## Purpose

Mathematical glossary flow is the documentation format used to teach TFNE/jaxfne equations by connecting formal mathematics to physical intuition, computational implementation, and statement boundaries.

Each important equation is presented in:
1. Simple/familiar form
2. Tensor/general form (when useful)
3. Complete term glossary
4. Worded-equation (plain-language translation)
5. Critical bridge term (how this equation connects the pipeline)
6. Run boundary (what scientific status this equation carries)
7. Implementation location (where in code/report this shows up)

This structure ensures that equations remain teachable, grounded, and appropriately stated.

---

## Core TFNE Equations

### 1. Emitter Dynamics

**Formal equation:**

$$\frac{dz}{dt} = F_\theta(z, u, t)$$

**Tensor form:** component-wise, same as above.

**Term glossary:**
- $z(t) \in \mathbb{R}^{n_\theta}$: neural state vector (membrane voltages, gating variables, synaptic conductances, etc.)
- $u(t)$: input (stimulus, noise, recurrent current)
- $F_\theta$: declared neural dynamics function (parametrized emitter)
- $\theta$: emitter parameters (time constants, thresholds, conductance scales, etc.)
- $t$: time

**Worded-equation:**

$$\boxed{\text{state change} = \text{declared neural dynamics applied to state, input, parameters, and time}}$$

**Critical bridge term:**

$F_\theta$ is the **Emitter → Source** bridge. It transforms parameter choices and input currents into state evolution that later becomes membrane current and spike signals.

**Run boundary:**

- **Computational scaffold** unless $F_\theta$ is calibrated to empirical neural response
- **Not biological proof** without comparison to experimental data
- **Izhikevich default** in jaxfne; other emitters supported via custom models

**Implementation:**
- `jaxfne.emitters.IzhikevichParams` — parameter container
- `jaxfne.emitters.simulate_eig_izhikevich()` — forward simulation
- `jaxfne/core.py` — Model interface

---

### 2. Source Projection

**Formal equation:**

$$q(x,t) = P_s[z(t), I(t), \chi(x)]$$

**Tensor form:**

$$q_\alpha(x,t) = \sum_k w_{k\alpha}(x) \cdot (a_k \cdot z_k(t) + b_k \cdot I_k(t))$$

where $\alpha$ indexes spatial contact points, $k$ indexes neurons, and $w$ are spatial projection weights.

**Term glossary:**
- $q(x,t)$: field source density (current per unit volume or contact area)
- $P_s$: source projection operator (maps neural state/current to spatial domain)
- $z(t)$: neural state (from emitter)
- $I(t)$: input/drive current
- $\chi(x)$: spatial contact basis (probe geometry, source locations)
- $w_{k\alpha}(x)$: spatial projection weight from neuron $k$ to contact $\alpha$ (anatomical distance, contact coupling)
- $a_k$: state-to-source projection scalar for unit $k$ (scales how much of the emitter state contributes to the source; a Relative-value coefficient)
- $b_k$: input-to-source projection scalar for unit $k$ (scales how much of the drive/input current contributes to the source; a Relative-value coefficient)

**Worded-equation:**

$$\boxed{\text{field source density} = \text{projection of neural state and current into spatial tissue coordinates}}$$

**Critical bridge term:**

$P_s$ is the **Source → Field** bridge. It transforms time-domain neural currents into spatially-localized sources that can be convolved with field operators.

**Run boundary:**

- **Physical only** when `source_calibration_status` includes calibrated or empirical current evidence
- **Proxy** in default linear_solver path (no physical current units stated)
- **Model current** (Izhikevich model current + spike impulse proxy) by default
- **No double-counting** of synaptic current (forbidden pattern: using both synaptic conductance-based current and spike-based source in the same field computation)

**Implementation:**
- `jaxfne.fields.project_laminar_sources()` — source projection kernel
- `jaxfne/core.py` → `Signals` — output structure
- Manifest field: `source_calibration_status`

---

### 3. Ohmic Extracellular Current

**Formal equation:**

$$\mathbf{J}_e = -\sigma_e \nabla \phi_e$$

**Tensor/index form:**

$$J_e^i = -\sigma_e^{ij} \partial_j \phi_e$$

**Term glossary:**
- $\mathbf{J}_e(x,t)$: extracellular current density (A/m²)
- $\sigma_e(x)$: extracellular tissue conductivity tensor (S/m)
- $\phi_e(x,t)$: extracellular voltage (mV)
- $\nabla \phi_e$: voltage gradient

**Worded-equation:**

$$\boxed{\text{extracellular current density} = \text{passive conductive tissue response to voltage gradient}}$$

**Critical bridge term:**

$\sigma_e$ is the **Field → Current** bridge. It encodes how tissue converts electrostatic voltage gradients into current flow.

**Run boundary:**

- **Proxy** in the current linear_solver path (assumed isotropic, no PDE solve)
- **Physical only** when:
  - Field solver is active and verified (`field_solver_status != linear_solver`)
  - Conductivity is calibrated (`conductivity_status == calibrated_physical`)
  - Solution satisfies conservation constraints
- **Isotropy assumed** in current proxy mode (conductivity is scalar)

**Implementation:**
- `jaxfne.fields.project_laminar_sources()` — applies conductivity implicitly via kernel
- Manifest: `field_solver_status`, `boundary_condition`, `gauge`

---

### 4. Field Compatibility Equation (Poisson-like)

**Formal equation:**

$$\nabla \cdot (-\sigma_e \nabla \phi_e) = q$$

**Tensor/index form:**

$$\partial_i (-\sigma_e^{ij} \partial_j \phi_e) = q$$

**Term glossary:**
- $\nabla \cdot$: divergence operator
- $q(x)$: source density (from source projection)
- $\phi_e$: extracellular potential

**Worded-equation:**

$$\boxed{\text{divergence of conductive extracellular current} = \text{declared source density}}$$

**Critical bridge term:**

$q$ is the **Source → Field** boundary condition. It ensures the field solution (when computed) is compatible with declared sources.

**Run boundary:**

- **Current default**: `linear_solver` — a Relative-value declared projection
- **Reserved path**: `specified_future_solver` for an Absolute-value field solver (see [Limitations and future plans](limitations_and_future_plans.md))

**Implementation:**
- `jaxfne.fields.project_laminar_sources()` — declares but doesn't solve
- Manifest: `field_solver_status == "linear_solver"` means this equation is metadata only

---

### 5. Current-Source Density (CSD)

**Formal equation:**

$$\mathrm{CSD}(x,t) = \nabla \cdot \mathbf{J}_e(x,t) = -\nabla \cdot (\sigma_e \nabla \phi_e)$$

**Tensor form:**

$$\mathrm{CSD} = \partial_i J_e^i$$

**Term glossary:**
- $\mathrm{CSD}$: current-source density (A/m³)
- Positive CSD: extracellular current divergence (current flowing outward) = current source = outward transmembrane current
- Negative CSD: current convergence (current flowing inward) = current sink = inward transmembrane current

**Worded-equation:**

$$\boxed{\text{current-source density} = \text{local divergence of extracellular current density}}$$

**Critical bridge term:**

CSD turns a field-current pattern into a source/sink-like readout. It is the primary laminar-field observable.

**Run boundary:**

- **CSD-proxy** unless source AND field are calibrated/solved
- **Sign convention** in jaxfne: `positive_equals_extracellular_source` (positive CSD = current flowing outward)
- **Not a direct biological measurement** (no physical conductivity or solved field)

**Implementation:**
- `jaxfne.fields.project_laminar_sources()` — computes CSD proxy directly
- Manifest: `csd_sign_convention`, `field_model_status == "proxy_readout"`

---

### 6. Probe Operator (General)

**Formal equation:**

$$Y_c(t) = P_c[\phi_e(t), \mathbf{J}_e(t), \mathrm{CSD}(t), \ldots]$$

**Tensor form:** operator-dependent (contact-wise, depth-wise, or spatially integrated).

**Term glossary:**
- $Y_c(t)$: channel readout (voltage, current, field quantity at contact $c$)
- $P_c$: probe operator (e.g., voltage sampling, CSD extraction, LFP bandpass)
- $c$: contact index (spatial location on probe)

**Worded-equation:**

$$\boxed{\text{channel readout} = \text{declared probe operator applied to field and source-derived quantities}}$$

**Critical bridge term:**

$P_c$ is the **Field → Probe** bridge. It extracts spatially localized measurements from the field solution.

**Run boundary:**

- **Simulated/proxy readout** unless:
  - Geometry (lead field) is calibrated to real electrodes
  - Field is solved (not proxy)
  - Validation against empirical data exists
- **Eight multimodal operators** in jaxfne: SPK, Vm, source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy
- See [Probe Operators](guides/probe_operators.md) for operator-specific statement boundaries

**Implementation:**
- `jaxfne.fields.probe_laminar_modes()` — operator definitions
- `jaxfne/core.py` → Model.compute_readout() — readout computation
- Manifest: `operator_status` for each probe operator

---

### 7. EMM-Proxy (Electro-Mechanical Metabolic-like Proxy)

**Formal equation:**

$$\mathrm{EMM}_\text{proxy}(t) = w_s \|\mathrm{source}(t)\|_2^2 + w_E \|\nabla\phi_e(t)\|_2^2 + w_J \|\mathbf{J}_e(t)\|_2^2$$

**Term glossary:**
- $\mathrm{EMM}_\text{proxy}$: dimensionless activity cost
- $w_s, w_E, w_J$: scalar weights (default: normalized by signal variance)
- $\|\cdot\|_2$: Euclidean (L2) norm

**Worded-equation:**

$$\boxed{\mathrm{EMM\text{-proxy}} = \text{weighted normalized source activity + field-gradient activity + current-density activity}}$$

**Critical bridge term:**

EMM-proxy is a **relative within-run cost/activity index**, not a biological energy-consumption measurement. It combines source, field-gradient, and current-density norms to measure "computational cost" or "activity intensity."

**Run boundary:**

- **Proxy-only**
- **Computational cost metric** — combines source and field norms, not ATP consumption or metabolic rate
- **Valid for relative within-run comparison** (e.g., which timestep is most active)
- **Requires empirical calibration** for any biological interpretation statements

**Implementation:**
- `jaxfne.fields.probe_laminar_modes()` → EMM-proxy operator
- Manifest: `source_model` includes EMM-proxy definition
- See [Probe Operators](guides/probe_operators.md) for full details

---

## Conservation-Law Rules (Reserved Reference)

### Poynting's Theorem

**Formal equation:**

$$\frac{\partial u_{em}}{\partial t} + \nabla \cdot \mathbf{S} + \mathbf{J} \cdot \mathbf{E} = 0$$

where

$$u_{em} = \frac{1}{2}\left(\epsilon_0 |\mathbf{E}|^2 + \frac{1}{\mu_0}|\mathbf{B}|^2\right)$$

$$\mathbf{S} = \frac{1}{\mu_0} (\mathbf{E} \times \mathbf{B})$$

**Term glossary:**
- $u_{em}$: electromagnetic field energy density (J/m³)
- $\mathbf{S}$: Poynting vector (energy flux density, W/m²)
- $\mathbf{J} \cdot \mathbf{E}$: power density transferred to charges (W/m³)
- $\epsilon_0$: permittivity of free space
- $\mu_0$: permeability of free space

**Worded-equation:**

$$\boxed{\text{field energy change} + \text{energy flux leaving region} + \text{work done on charges} = 0}$$

**Critical bridge term:**

$\mathbf{J} \cdot \mathbf{E}$ is the **field-to-matter power density bridge**. It quantifies electromagnetic power transferred to moving charges.

**Run boundary:**

- Recorded here as conservation-principle motivation for the field diagnostics
- Full Maxwell/stress-energy tensor dynamics belong to the reserved regimes in [Limitations and future plans](limitations_and_future_plans.md)

**Why included:**
Poynting's theorem provides a conservation-principle foundation for the field diagnostics and the reserved electrodynamic regimes catalogued in [Limitations and future plans](limitations_and_future_plans.md).

---

## Proof and practice flows

These short flows tie formal reasoning to jaxfne implementation locations. Each stops at the proxy/scaffold boundary unless a solved field is explicitly enabled.

### Linearity and superposition (proxy stage)

**Formal:** for fixed projection weights $w_{k\alpha}$, laminar readouts satisfy
$\mathcal{R}[q_1 + q_2] \approx \mathcal{R}[q_1] + \mathcal{R}[q_2]$ when sources are combined before projection.

**Practice:** combine emitter outputs into one `[T, N]` source array, then call
`project_laminar_sources` once. Do not project populations separately and add
readouts unless you have verified the weights are identical — the default
Gaussian kernel is linear in source amplitude but not in neuron placement.

**Code:** `jaxfne/fields/proxy.py` (`project_laminar_sources`), `validate_projection_invariants`.

**Boundary:** superposition holds for the **proxy operator**, not for a future
nonlinear PDE solve (P2+ in [Tensor Electromagnetics Scope](tensor_electromagnetics_scope.md)).

### Source projection

**Formal:** $q_\alpha(t) = \sum_k w_{\alpha k}\, s_k(t)$ with $w$ from depth/contact geometry.

**Practice:** build `positions` from `model.neuron_table()`, pass `[T,N]` sources from
`signals.get("source")` or `sig.sources`. Choose `mode="density_preserving"` (default)
for off-population attenuation; `mode="row_normalize"` only when every contact lies
inside the modeled population.

**Code:** `jaxfne/fields/proxy.py`, `jtfne.project_laminar_sources`.

### CSD finite difference

**Formal:** $\mathrm{CSD}(z,t) \approx \partial^2 \phi_e / \partial z^2$ on the contact axis.

**Practice:** CSD-proxy is computed along the laminar contact grid after spatial
projection — it is a discrete second difference on `phi_e_proxy`, not a volume
divergence solve. Use `csd_tensor` when you need the stage factored out of the
full projection chain.

**Code:** `jaxfne/fields/proxy.py`, `jaxfne/fields/diagnostics.py` (`csd_tensor`).

### Conservation compatibility (diagnostic only)

**Formal:** $\int q\,\mathrm{d}V \approx 0$ is checked via the spatial mean of $q$ over contacts.

**Practice:** `compute_conservation_proxy_diagnostics` reports
`source_conservation_proxy_residual` — a scalar summary, not a PDE residual.
Values near zero indicate spatial balance of the **proxy source**, not enforced
continuity at a tissue boundary.

**Code:** `jaxfne/fields/diagnostics.py`, [Conservation Proxy Diagnostics](conservation_proxy_diagnostics.md).

### Proxy-vs-solver ladder

| Rung | What runs | Status field |
|------|-----------|--------------|
| L0 | Emitter only (`V_m`, spikes) | no field |
| L1 | Gaussian projection + FD CSD | `field_solver_status = "linear_solver"` |
| L2 | Declared geometry metadata | `PhysicalFieldSolverSpec` (metadata) |
| L3 | Elliptic/volume solve | reserved (`solve_physical_field` raises) |

Ascending the ladder requires new evidence at each rung; metadata alone does not
upgrade `physical_amplitude_calibrated` or `field_claim_level`.

---

## Implementation Checklist

When adding a new equation to jaxfne documentation:

- [ ] Include formal equation (LaTeX)
- [ ] Include tensor/index form (if useful for generalization)
- [ ] Define every term in glossary
- [ ] Provide worded-equation in plain language
- [ ] Identify critical bridge term and pipeline location
- [ ] State run boundary (computational scaffold? proxy? calibrated physical?)
- [ ] Link to code location and manifest fields
- [ ] Verify no forbidden phrasing (`real EEG`, `biological metabolism`, `mechanism proof`)

---

## See Also

- [Source-Field Equations](source_field_equations.md) — source bookkeeping, forbidden patterns
- [Computation Basis](computation_basis.md) — extensibility rules
- [Probe Operators](guides/probe_operators.md) — detailed operator statements
- [Scope and Limitations](limitations_and_future_plans.md) — boundary conditions
- [TFNE Operator Doctrine](operator_doctrine.md) — per-stage contract table indexing these equations
