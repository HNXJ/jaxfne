# Tutorial Template (v0.3)

**Status:** canonical template for all v0.3 tutorial notebooks and documentation  
**Applies to:** v0.3.1 through v0.3.30 (every numbered tutorial phase)  
**Canonical import:** `import jaxfne as jtfne`  
**truth_mode:** tutorial_exploratory_not_biological_truth  

---

## Required Tutorial Structure

Every v0.3 tutorial must contain all 13 sections below, in order.
Sections may be brief, but none may be omitted.

---

### Section 1 — Learning Objectives

State 2–5 concrete learning objectives using active-verb phrasing.

**Example:**
> After completing this tutorial, you will be able to:
> 1. Construct a single-neuron Izhikevich configuration using `jtfne.configuration()`.
> 2. Run a simulation and retrieve spike and voltage traces from `Signals`.
> 3. Read the manifest to verify claim gates.
> 4. Distinguish what the tutorial demonstrates from what it does not claim.

---

### Section 2 — Biological/Computational Question

State the question or problem this tutorial addresses.
Use hedged language. Do not overclaim biological validity.

**Example:**
> **Question:** How does a regular-spiking Izhikevich neuron respond to step-current injection,
> and what proxy-field readout does jaxfne produce for a single laminar contact?
>
> **Computational framing:** This is a demonstration of the emitter → source → proxy-field
> pipeline for one unit. It does not validate biological spike timing or field amplitudes.

---

### Section 3 — Mathematical Glossary Flow

List the relevant symbols and equations used in this tutorial.
Reference the canonical document (e.g., `docs/mathematical_glossary_flow.md`) and define
any tutorial-specific terms.

**Required format (use a table):**

| Symbol | Meaning | Units (if applicable) | Status in this tutorial |
|--------|---------|-----------------------|------------------------|
| $v$ | Membrane voltage (Izhikevich) | mV (model units) | Simulated |
| $u$ | Recovery variable | — | Simulated |
| $I$ | Total drive current | pA (proxy, uncalibrated) | Proxy |
| $\phi_e$ | Extracellular potential | μV proxy (not physical) | Proxy |

For the full equation glossary, see [Mathematical Glossary Flow](mathematical_glossary_flow.md).

---

### Section 4 — Canonical `jtfne` Import (Code Cell)

The first code cell in every tutorial must be the install cell.
The second code cell must be the canonical import.

**Install cell (required in every Colab notebook):**
```python
!pip install jaxfne
```

**Import cell (required in every tutorial, local or Colab):**
```python
import jaxfne as jtfne
import jax.numpy as jnp
import json
```

**Forbidden aliases:** Do not use `jtnfe`, `jtFNE`, `jaxFNE`, `jfne`, or any other variant.
Use `jtfne` everywhere in the tutorial — in code, in prose, and in comments.

---

### Section 5 — Configuration Block (Code Cell)

Use the chained configuration API:

```python
cfg = (
    jtfne.configuration()
    .network(
        name="<tutorial_name>",
        kind="cortical_column",
        n=<n_neurons>,
        layers=["L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(
        name="laminar_probe",
        modes=["spk", "vm", "source", "lfp_like", "csd_like"],
        n_contacts=16,
    )
)
```

Add a prose cell explaining the key choices and what each parameter means for this tutorial.
Do not claim the parameters are empirically calibrated unless calibration evidence is provided.

---

### Section 6 — Simulation Block (Code Cell)

```python
model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0)
signals = model.simulate(sim)

print(f"V_m shape: {signals.V_m.shape}  # (n_time, n_neurons)")
print(f"duration: {signals.V_m.shape[0] * 0.1:.1f} ms")
```

Add a prose cell explaining what `signals` contains and what the shapes mean.
Include: `n_time`, `n_neurons`, time resolution (dt_ms).

---

### Section 7 — Probe/Readout Block (Code Cell)

```python
readout = model.probe(signals, modes=["spk", "vm", "source", "lfp_like", "csd_like"])
```

For tutorials that use specific readout operators, show the relevant arrays and shapes.
Add a prose cell explaining what each readout represents and its proxy status.

---

### Section 8 — Manifest and Claim Gates (Code Cell)

Every tutorial must end with a manifest check:

```python
manifest = model.manifest(signals)

print("truth_mode          :", manifest["truth_mode"])
print("claim_level         :", manifest["claim_level"])
print("field_solver_status :", manifest["field_solver_status"])
print("physical_amplitude  :", manifest["physical_amplitude_claim_allowed"])
print("field_claim_level   :", manifest["field_claim_level"])

# Verify JSON safety
json.dumps(manifest, allow_nan=False)
print("manifest: JSON-safe ✓")
```

Expected output:
```
truth_mode          : truth_safe_unverified
claim_level         : computational_scaffold
field_solver_status : laminar_proxy_no_pde
physical_amplitude  : False
field_claim_level   : proxy_readout_only
manifest: JSON-safe ✓
```

**Do not suppress or skip this cell.** It is the receipt that anchors the tutorial's
claim status.

---

### Section 9 — Figures

Generate at least one figure from the simulation data.
Figures must be generated from real data in this notebook run — not pre-created or manually
copied.

Required figure practices:
- Title must include tutorial name and a non-overclaiming label
- Axes must be labeled with units (use "proxy" or "a.u." if not physical)
- Do not label axes with physical units (μV, nA, etc.) unless the source is calibrated
- Caption must state proxy status

**Example caption:**
> *Spike raster for 10-neuron E/I network (v0.3.3 tutorial). Rows = neurons sorted by
> cell type. Each dot = one threshold-crossing event. Amplitudes are proxy units.
> No physical-amplitude or biological-mechanism claim.*

---

### Section 10 — Interpretation

Write 3–6 sentences interpreting the tutorial output.
Use hedged language:

**Allowed:**
> "The simulation shows that..."
> "The proxy CSD readout suggests..."
> "The spectral content of the proxy LFP is consistent with..."

**Forbidden:**
> "This proves that..."
> "The real LFP amplitude is..."
> "This validates the biological mechanism of..."

---

### Section 11 — Failure Modes

List 2–4 known failure modes or edge cases for this tutorial:

| Failure Mode | Symptom | Mitigation |
|-------------|---------|------------|
| All-silent network | spike raster empty, V_m flat near -65 mV | Check drive current; increase input gain |
| Synchrony explosion | all neurons fire simultaneously | Reduce recurrent E→E weight |
| NaN in manifest | `json.dumps` raises ValueError | Check for inf/NaN in signals; use finite inputs |
| Field all-zero | phi_e and CSD are exactly zero | Source projection returned zero; check source mode |

---

### Section 12 — Exercises

List 2–5 exercises for the reader.

**Example:**
> 1. Change `n` from 10 to 50. How does the spike raster change?
> 2. Set `seed=1` instead of `seed=0`. Are results reproducible across seeds?
> 3. Change `preset` from `"cortical_eig"` to `"fs_interneuron"`. What changes?
> 4. Remove the `PV` cell type. How does synchrony change?
> 5. Increase `duration_ms` to 500 ms. Does the network reach a steady state?

---

### Section 13 — What This Tutorial Does NOT Claim

This section is **required and may not be omitted or shortened.**

---

**This tutorial is a computational scaffold.**

Outputs are simulated proxy readouts generated under manifest claim gates.
No biological mechanism is proven by this tutorial alone.

Specifically, this tutorial does **not** claim:

- The Izhikevich native current is a calibrated membrane current
- The spike timing matches any specific biological neuron type
- The proxy LFP/CSD amplitudes correspond to physical microvolts
- The proxy EEG/MEG outputs correspond to measured scalp or MEG signals
- The spectral content of proxy readouts matches in vivo spectral profiles
- The network dynamics have been validated against any empirical dataset
- Any anatomical or physiological parameter has been calibrated

**To make physical-amplitude, biological, or mechanistic claims, you must supply:**
- Calibration evidence (empirical recordings with known units)
- A solved field (Poisson or Maxwell equations with physical conductivity)
- Validated geometry and boundary conditions
- A peer-reviewed validation protocol

Until these are supplied, all outputs remain computational scaffolds with
`physical_amplitude_claim_allowed: False`.

---

## Tutorial Checklist

Before submitting a v0.3 tutorial for review:

- [ ] All 13 sections present
- [ ] Canonical import: `import jaxfne as jtfne` (no forbidden aliases)
- [ ] First code cell: `!pip install jaxfne` (Colab) or import-only (local)
- [ ] Configuration uses chained `.network().emitter().field().probe()` pattern
- [ ] Simulation block shows shapes and duration
- [ ] Manifest block runs and prints all 5 claim fields
- [ ] `json.dumps(manifest, allow_nan=False)` passes
- [ ] At least one figure generated from real simulation data
- [ ] Figure axes labeled (no physical units unless calibrated)
- [ ] Figure caption states proxy status
- [ ] Interpretation uses hedged language
- [ ] Failure modes table included
- [ ] Section 13 (non-claim statement) present and complete
- [ ] No forbidden language in prose (see language audit in `docs/ci_policy.md`)

---

## Claim Status

Every tutorial produced from this template carries:

```
truth_mode: tutorial_exploratory_not_biological_truth
claim_level: computational_scaffold
physical_amplitude_claim_allowed: False
field_claim_level: proxy_readout_only
```

These are not advisory labels. They are enforced by the manifest and verified by
automated language audits. Do not remove or soften Section 13.
