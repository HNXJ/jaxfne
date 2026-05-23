# Tensor-Network Ancestry and Basis-Transform Doctrine

**Status:** v0.2.29 conceptual documentation  
**Version:** v0.2.28-aligned  
**truth_mode:** truth_safe_unverified  
**Scope:** Terminology, historical context, architectural parallels; no implementation claims

---

## Purpose

This document situates TFNE within two distinct meanings of **tensor network**:

1. **Pellionisz/Llinás (1980s–2000s):** Tensor network as a geometric framework for sensorimotor transforms and cerebellar learning in neuroscience
2. **Modern ML/Physics (2010s–present):** Tensor network as a factorized state compression and contraction algorithm (tensor trains, MPS, PEPS, etc.)

jaxfne adopts the **basis-transform architecture** from the first tradition (multi-scale coordinate projections: emitter basis → source basis → field basis → readout basis) while remaining distinct from both traditions' full scope.

This document clarifies what TFNE does, what it does not implement, and why the basis-transform concept matters for modularity and extensibility.

---

## Part 1: Two Meanings of "Tensor Network"

### Pellionisz/Llinás Neuroscience Meaning

**Era:** Pellionisz & Llinás (1980–2010), continued by Porrill, Dean, and others

**Core idea:** Neurons solve coordinate-transform problems. A cerebellar Purkinje cell receives:
- Sensory (retinotopic, somatosensory) input in **one coordinate frame** (retinal, skin)
- Motor efference copy in **another frame** (joint angles, velocity)
- Output command in **a third frame** (muscle activation)

The cerebellum learns to map between frames via a metric tensor.

**Mathematical core:**

$$V^{motor} = g^{motor,sensory}_{ij} S^{sensory}_{j} + \text{memory terms}$$

where $g$ is a learned tensor (metric) that transforms sensory coordinates into motor coordinates.

**Why "tensor network"?** Multiple sensory and motor dimensions are contracted via a learned coupling tensor.

### Modern ML/Physics Meaning

**Era:** Vidal (2003–present), developed in quantum information and condensed-matter physics

**Core idea:** A large high-dimensional tensor can be factorized into a **network of smaller tensors** connected along bonds (virtual indices).

**Example: Tensor Train (TT) or Matrix Product State (MPS)**

$$T(i_1, i_2, \ldots, i_n) = C_1(i_1) \times C_2(i_2) \times \cdots \times C_n(i_n)$$

(where $\times$ denotes contraction along virtual indices)

**Why "tensor network"?** The high-dimensional tensor is built by contracting a network of lower-rank local tensors.

**Applications:**
- Quantum state compression
- Efficient PDE solvers
- Variational autoencoders (VAE) with factorized latents

---

## Part 2: TFNE Basis-Transform Architecture

TFNE adopts the **coordinate-transformation philosophy** (Pellionisz/Llinás) but does NOT implement:
- Learned metric-tensor coefficients
- Cerebellar learning dynamics
- Full sensorimotor circuit models

### TFNE Basis Contract

TFNE organizes computation as **cascaded basis transforms**:

**Formal expression:**

$$
\boxed{
Y^{readout}_a = P_{a\alpha c} Z^{field}_{\alpha c}
}
$$

**Expanded (four basis transforms):**

$$
X^{probe}_a = P_{a, basis(source \to probe)} \left( F_{basis(field \to source)} \left( Q_{basis(emitter \to field)} V^{state}_{state\_index} \right) \right)
$$

**Worded (English):**

1. **Emitter basis:** Neural state (membrane voltage, gating variables, synaptic conductances) in a neuron-indexed frame
2. **Source basis:** Currents or source densities projected into a spatial frame (contact positions, voxels)
3. **Field basis:** Extracellular potentials or proxy potentials at measurement locations
4. **Readout basis:** Multiple simultaneous readouts (spikes, raw voltage, LFP proxy, CSD proxy, etc.)

**Bridge terms** (which basis transforms are computed):

| Transform | Implemented | Status |
|-----------|-------------|--------|
| Emitter → Source | ✓ (always) | Mandatory; state projection via current/conductance emission |
| Source → Field | ◑ (optional) | Proxy-only or PDE-based (future) |
| Field → Readout | ✓ (selective) | User chooses probe operators (8 available: SPK, Vm, source, LFP, CSD, EEG, MEG, EMM) |

### Why Basis Transforms Matter

1. **Modularity:** Each stage is independent. A source can exist without field solve; a field without EEG readout.
2. **Claim boundary:** Each basis transform carries its own truth status. Source projection is deterministic; field solve is (currently) proxy-only.
3. **Extensibility:** New bases (ionic current frame, spectral frame, etc.) fit the same architecture without breaking the pipeline.
4. **Tensor structure:** The cascade is naturally a tensor contraction chain: state → source density → field potential → readout metrics.

---

## Part 3: Probe-Basis Tensor Example

TFNE probe readouts implement the final basis transform:

**Formal:**

$$Y_a = P_{a,i,j} \, Z_{i,j}$$

where:
- $a$: readout index (spike, voltage, LFP, CSD, etc.)
- $P_{a,i,j}$: probe operator (includes geometry, filtering, etc.)
- $Z_{i,j}$: field basis (spatial location $i$, time step $j$)
- $Y_a$: scalar or vector readout

**Example (LFP proxy):**

$$\text{LFP\_proxy}(t) = \sum_{contacts} w_{contact} \, \phi_e(contact, t)$$

(where $\phi_e$ is extracellular potential and $w_{contact}$ are weights from contact geometry)

**Non-claim:** This is not a real LFP sensor. It is a proxy computed from simulated sources. No calibration to empirical recordings is claimed.

---

## Part 4: What TFNE Does (and Does Not Claim)

### TFNE Does

✓ Organize emitter → source → field → readout as a modular tensor-contraction pipeline  
✓ Support multi-basis workflows (e.g., Izhikevich emitter in mV, source in nA, LFP proxy in arbitrary units)  
✓ Provide 8 probe operators for simultaneous multimodal readouts  
✓ Validate claim gates: proxy-only, no biological metabolism claims, no solver oversteps  
✓ Enable future extensions: new emitters, new field solvers, new probes—all within the same basis-transform architecture

### TFNE Does NOT Claim

✗ **No cerebellar metric-tensor learning:** jaxfne does not implement Pellionisz/Llinás sensorimotor transforms or learned basis-transform coefficients  
✗ **No tensor-train/MPS compression:** jaxfne is not a tensor-network factorization library (no tensor-train states, MPS, PEPS, etc.)  
✗ **No Maxwell/Poisson/stress-energy solver implementation from tensor-network ancestry:** jaxfne's source-field framework is a proxy spatial projection in the default v0.2.x path (no PDE solvers, no boundary-condition enforcement). TFNE's source and field basis transforms are fundamental to the electromagnetic observable framework; they are not absent from it.  
✗ **No biological validation:** Proxy readouts (LFP, CSD, EEG, MEG) are named *proxy* because they are not validated against empirical measurements  
✗ **No sensorimotor proof:** Using basis transforms does not demonstrate that neural circuits implement metric-tensor learning  
✗ **No condensed-matter quantum analogy:** jaxfne tensors are not quantum states and do not use variational tensor-network algorithms

---

## Part 5: Relationship to BasisSpec Contract

jaxfne's `BasisSpec` (introduced in v0.2.25) formalizes the basis-transform idea:

```python
from jaxfne.core import BasisSpec

# Example: declare basis transforms
emitter_basis = BasisSpec(name="izhikevich_state", units="mV", n_dims=4)
source_basis = BasisSpec(name="synaptic_current", units="nA", n_dims=50)
field_basis = BasisSpec(name="laminar_potential", units="proxy_mV", n_dims=16)
readout_basis = BasisSpec(name="multimodal", units="mixed", n_dims=8)

# Pipeline preserves basis contracts through transformations
```

**Doctrine:** BasisSpec makes the tensor-coordinate structure explicit and testable. It is the operational instantiation of "basis transform" as a software contract.

---

## Part 6: Future Optional Path (Not Implemented)

A **future cerebellar/sensorimotor tutorial** could use jaxfne's basis-transform architecture as a teaching tool:

**Hypothetical example (NOT IMPLEMENTED):**

```python
# Hypothetical cerebellar learning model
# Would require:
# 1. A cerebellar-circuit emitter (Purkinje/Granule dynamics)
# 2. A metric-tensor learning rule in the source basis
# 3. Validation against empirical cerebellar response

# Such a tutorial would be a NEW module with its own:
# - Claim gates
# - Validation tests
# - Manuscript references
# - Disclaimer that this is exploratory, not a biological proof
```

This path is **deferred and not promised.** If pursued, it would:
- Require separate validation evidence
- Use separate claim gates (distinct from computational_scaffold)
- Be a distinct research module, not a core jaxfne feature

---

## Part 7: Distinction from Other Tensor-Network Meanings

| Term | Meaning | jaxfne Role | Claim? |
|------|---------|-------------|--------|
| **Tensor network (Pellionisz)** | Sensorimotor coordinate transforms, metric-tensor learning | ✓ Architectural inspiration | ✗ No implementation |
| **Tensor network (ML/Physics)** | Factorized state compression (MPS, PEPS, TT) | ✗ Not used | ✗ No claim |
| **Basis transform** | Cascade of coordinate changes (emitter → source → field → readout) | ✓ Core TFNE principle | ✓ Implemented |
| **Tensor contraction** | Algebraic operation summing over shared indices | ✓ Mathematical formalism | ✓ Implicit in readouts |

---

## Summary: Why Basis-Transform Doctrine Matters

1. **Architecture:** Basis transforms make TFNE's modularity explicit. Users understand why sources can exist without fields.
2. **Extensibility:** New bases (ionic channels, spectral, population-level) fit the same framework.
3. **Claim clarity:** Each basis transform has its own truth status and claim gate.
4. **Teaching:** The basis-coordinate idea connects TFNE to classical computational neuroscience (Pellionisz, Koch, Arleo) while remaining distinct.
5. **Humility:** By referencing Pellionisz/Llinás and NOT claiming their results, we honor the intellectual history while respecting scope boundaries.

---

## See Also

- **[Computation Basis](computation_basis.md)** — Collapsible tensor-field scaffold and dimension contracts
- **[Mathematical Glossary Flow](mathematical_glossary_flow.md)** — Seven core TFNE equations with claim boundaries
- **[Source/Field Equations](source_field_equations.md)** — Source bookkeeping and field proxy details
- **[Manuscript Alignment](manuscript_alignment.md)** — How TFNE code maps to published manuscript sections

---

**Status:** v0.2.29 conceptual documentation (no implementation claims)  
**truth_mode:** truth_safe_unverified
