# Electromagnetism of Brain Signals

## Purpose

This guide connects classical electromagnetism to the jaxfne proxy-readout pipeline.
It is written for readers who know basic vector calculus and want to see how
extracellular potentials, current-source density (CSD), and scalp/magnetometer
readouts relate to the code paths in `jaxfne.fields` and `jaxfne.vis`.

jaxfne does **not** solve Maxwell or Poisson PDEs in the shipped laminar-proxy
regime. The equations below define the **target physics** and the **proxy ladder**
that the package implements today.

---

## 1. From membrane current to extracellular source

Neurons inject transmembrane current into the extracellular medium. At a coarse
scale, define a source density \(q(x,t)\) (A/m³) that enters the field equations.

**Formal:**

\[
q(x,t) = P_s\big[z(t), I(t), \chi(x)\big]
\]

**In jaxfne:** `project_laminar_sources` maps emitter state and spike impulses to
contact-wise source proxies. See [Source and Field Equations](../source_field_equations.md).

**Interpretation boundary:** default sources use uncalibrated Izhikevich model
current plus a spike impulse gain — proxy units, not measured μA/mm².

---

## 2. Quasi-static extracellular potential (volume conductor)

At neural frequencies, displacement currents are negligible. The extracellular
potential \(\phi_e\) satisfies a Poisson-like equation:

\[
\nabla \cdot \big(-\sigma_e \nabla \phi_e\big) = q
\]

**Ohmic constitutive law:**

\[
\mathbf{J}_e = -\sigma_e \nabla \phi_e
\]

**Worded mechanism:** extracellular current flows down voltage gradients through
conductive tissue; divergence of that current equals the declared source.

**Shipped status:** `field_solver_status = "linear_solver"` — the equation is
recorded in manifests but **not solved**. Instead, jaxfne applies a
Gaussian-leadfield laminar projection (superposition of spatial kernels).

**Implementation:** `jaxfne.fields.project_laminar_sources`, manifest
`field_solver_status`, `boundary_condition`, `gauge`.

---

## 3. Current-source density (CSD)

CSD is the divergence of extracellular current density:

\[
\mathrm{CSD}(x,t) = \nabla \cdot \mathbf{J}_e = -\nabla \cdot (\sigma_e \nabla \phi_e)
\]

**Proxy path in jaxfne:** finite-difference divergence along the laminar depth
axis applied to the source or potential proxy — a **CSD-like readout**, not a
calibrated biophysical measurement.

**Sign convention:** positive CSD corresponds to extracellular current divergence
(`positive_equals_extracellular_source` in manifests).

**Visualization:** `jtfne.vis.csd(sig)` returns a matplotlib figure titled with
proxy-safe language.

---

## 4. LFP-like laminar readout

Local field potentials sample \(\phi_e\) (or a band-limited projection) at
electrode contacts. In the laminar proxy:

\[
Y_\mathrm{LFP}(t,c) = \sum_k w_{ck}\, s_k(t)
\]

where \(w_{ck}\) are spatial coupling weights and \(s_k\) are per-unit source
contributions.

**Normalization matters:** `mode="row_normalize"` forces each contact's weights
to sum to 1, which can **erase distance attenuation** for contacts placed outside
the modeled population. Use `mode="density_preserving"` when probe geometry extends
beyond the neuron cloud. See [Probe Operators](probe_operators.md).

---

## 5. EEG-like and MEG-like scalp projections

Scalp EEG and MEG sensors integrate fields over sensor geometry. jaxfne exposes
**linear proxy projections**:

| Modality | Proxy form | Physical status (shipped) |
| --- | --- | --- |
| EEG-like | \(Y_\mathrm{EEG}(t,ch) = \sum_c L_{ch,c}\, \phi_c(t)\) | uncalibrated lead field |
| MEG-like | \(Y_\mathrm{MEG}(t,ch) = \sum_c G_{ch,c}\, J_c(t)\) | uncalibrated gain matrix |

Lead fields \(L\) and gain matrices \(G\) are declared operators, not fitted to
empirical head models unless you supply calibration data.

**Tutorial:** [09: EEG/MEG/EMM Proxy Bundle](../tutorials/09_v0310_eeg_meg_emm_proxy_bundle.md).

---

## 6. EMM proxy versus physical energy accounting

The EMM-proxy combines normalized norms of source, potential gradient, and
current-density proxies:

\[
\mathrm{EMM}_\text{proxy}(t) = w_s \|q(t)\|_2^2 + w_E \|\nabla\phi_e(t)\|_2^2 + w_J \|\mathbf{J}_e(t)\|_2^2
\]

This is a **within-run activity index**, not ATP metabolism or electromagnetic
energy density. Poynting-flux and stress-energy diagnostics are reserved for
future solver regimes — see [Conservation Proxy Diagnostics](../conservation_proxy_diagnostics.md).

---

## 7. Local nonlinear / global linear split

| Layer | Nonlinearity | Linearity |
| --- | --- | --- |
| Emitter (Izhikevich) | per-neuron dynamics, spikes | — |
| Recurrent synapses | threshold, short-term saturation | weight matrix multiply |
| Source projection | spike impulses | superposition over units |
| Field / probe | — | linear kernels on sources |

This split is why superposition checks are meaningful for source→field paths even
when single-neuron dynamics are strongly nonlinear.

---

## 8. Proof and practice flows

### Linearity / superposition (proxy path)

If sources are fixed linear functionals of state, then
\(q = q^{(1)} + q^{(2)} \Rightarrow Y \approx Y^{(1)} + Y^{(2)}\) for linear
probe operators. Validate with two independent drive masks and compare summed
readouts — see [Conservation Proxy Diagnostics](../conservation_proxy_diagnostics.md).

### Source projection

Verify `signals.field` keys (`lfp_proxy`, `csd_proxy`) exist after `simulate`
with laminar field config. Check manifest `source_projection_mode`.

### CSD finite difference

Inspect depth-axis spacing in `model.neuron_table()` z-coordinates; CSD proxy
resolution follows contact spacing, not histological layer thickness.

### Conservation compatibility

`source_conservation_proxy_residual` measures spatial mean of source — a soft
diagnostic, not a PDE residual. A reserved elliptic solver would require
boundary conditions, gauge fixing, and residual norms — see
[Limitations and future plans](../limitations_and_future_plans.md).

### Proxy-versus-solver ladder

```text
Emitter → Source proxy → Laminar leadfield → Probe proxy   (shipped)
Emitter → Source → Poisson solve → Calibrated leadfield    (reserved)
Emitter → Source → Maxwell/stress-energy solve             (reserved)
```

---

## See also

- [Mathematical Glossary Flow](../mathematical_glossary_flow.md) — equation cards
- [Source and Field Equations](../source_field_equations.md) — manifest contracts
- [Probe Operators](probe_operators.md) — operator catalog
- [Computation Basis](../computation_basis.md) — collapsible tensor-field scaffold
