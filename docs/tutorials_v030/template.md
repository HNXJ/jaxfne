# v0.3 Tutorial Template: 13-Section Required Structure

**Version:** v0.3.0+  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified  

---

## Overview

All v0.3 tutorial notebooks must follow this exact 13-section structure. This ensures consistency, reproducibility, clarity, and truth-boundary enforcement across the entire v0.3 atlas.

---

## Required 13-Section Structure

### **Section 1: Learning Objectives**

**Purpose:** Explicit list of what the learner will understand after completing the notebook.

**Format:**
```markdown
## Learning Objectives

By completing this tutorial, you will:

1. [Objective 1 in clear, measurable language]
2. [Objective 2]
3. [Objective 3]
4. [Objective 4 or more as needed]

**Time estimate:** [X minutes for X ms of simulation + [Y minutes of analysis]

**Prerequisites:**
- v0.X.N: [prerequisite notebook, if any]
- Familiarity with [concept, e.g., "JAX arrays and vmap"]
```

**Key rules:**
- Use action verbs: "understand," "implement," "analyze," "reproduce"
- Avoid vague claims: "learn about neural networks" ❌ → "simulate Izhikevich spike generation" ✓
- Link to prior tutorials if prerequisites exist
- Realistic time estimates (include both simulation and post-hoc analysis)

---

### **Section 2: The Biological/Computational Question**

**Purpose:** Situate the tutorial in a real scientific context, clearly stating what question it addresses.

**Format:**
```markdown
## The Biological Question

### Real neuroscience question
[Briefly describe the biological phenomenon the tutorial models. E.g., "How do neurons generate action potentials? What mechanisms underlie spike timing and frequency adaptation?"]

### Computational analog
[Translate the biological question into a computational model question. E.g., "In the Izhikevich model, how do the voltage variable V_m and recovery variable u interact to produce spike-like events? How do parameters a, b, c, d control spike initiation and reset?"]

### Scope boundary
[Explicitly state what this tutorial does NOT address. E.g., "This tutorial models spike generation at the single-neuron level. It does NOT address whole-brain networks, biological calibration, or proof of real neural mechanisms. It is a computational scaffold for teaching and design exploration."]
```

**Key rules:**
- Reference real neuroscience (literature citations welcome, not required for v0.3)
- Connect biology ↔ computation explicitly
- State truth boundary upfront (prevents reader misconception)
- Avoid overclaiming ("demonstrates a neural mechanism" ❌ → "demonstrates a possible mechanism in silico" ✓)

---

### **Section 3: Mathematical Glossary**

**Purpose:** Define all symbols, equations, and mathematical objects used in the tutorial.

**Format:**
```markdown
## Mathematical Glossary

### Variables

| Symbol | Name | Units | Meaning |
|--------|------|-------|---------|
| V_m | Membrane potential | mV | Voltage across neuronal membrane relative to resting |
| u | Recovery variable | mV (dimensionless in original) | Slow adaptation/repolarization mechanism |
| I | Input current | pA or nA | External current stimulus |
| t | Time | ms | Simulation time |

### Parameters

| Symbol | Name | Default | Range | Meaning |
|--------|------|---------|-------|---------|
| a | Time scale | 0.02 | [0.001, 0.2] | Decay rate of recovery variable |
| b | Sensitivity | 0.2 | [−0.5, 1.0] | Coupling of recovery to voltage |
| c | Reset | −65 | [−90, −30] | Post-spike reset potential |
| d | Recovery jump | 8.0 | [0, 20] | Jump in recovery variable after spike |

### Equations

**Izhikevich model:**

$$\frac{dV_m}{dt} = 0.04 V_m^2 + 5 V_m + 140 - u + I$$

$$\frac{du}{dt} = a(b V_m - u)$$

**If $V_m \geq 30$ mV, then $V_m \leftarrow c$ and $u \leftarrow u + d$**

(Describe what each term represents in plain language)

### Abbreviations

| Abbrev. | Meaning |
|---------|---------|
| Hz | Hertz (spikes per second) |
| mV | Millivolts |
| ms | Milliseconds |
```

**Key rules:**
- Make a table; don't hide definitions in prose
- Include units always
- Cite original papers if equations are from literature
- Describe equations in words, not just symbols

---

### **Equation Display Rule (Cross-Section Policy)**

**CRITICAL:** Every theoretical/mathematical section in the tutorial must display equations in LaTeX notation, not only describe them in words.

**Correct pattern:**

All equations must use LaTeX display math syntax (`$$...$$`):

```markdown
$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I(t)$$
```

**Incorrect pattern:**

Describing equations only in words without LaTeX is not sufficient:

```markdown
The voltage changes based on a cubic term, linear terms, and input current.
```

**Required equation structure:**

Every displayed equation must be followed by:

1. **Equation (displayed in LaTeX):** Show the equation visually
2. **Term Glossary:** Define each symbol, units, ranges
3. **Worded Equation:** Explain in plain English what the equation computes
4. **Implementation Location:** Where in jaxfne code this is computed
5. **Claim Boundary:** What this equation claims and what it does not claim

**Example format (v0.3 tutorial section):**

```markdown
### Izhikevich Voltage Dynamics

The voltage follows the canonical Izhikevich cubic-linear form:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I(t)$$

**Term Glossary:**
- $v$ = membrane potential (mV), range [−90, +50]
- $u$ = recovery variable (nA), represents slow negative feedback
- $I(t)$ = external input current (pA), scenario-dependent
- $t$ = simulation time (ms)

**Worded Equation:**
The voltage update combines a cubic nonlinearity (fast spike initiation), linear terms (shaping fast and slow dynamics), and recovery-driven hyperpolarization (slow repolarization), driven by external input.

**Implementation:** `jaxfne/emitters/izhikevich.py:simulate_izhikevich_step()`

**Claim Boundary:**
This equation is a mathematical model, not validated against electrophysiological recordings. The native current $I(t)$ is not empirically calibrated. The model is a computational scaffold for understanding spike dynamics and design exploration, not a proof of biological mechanism.
```

**Additional LaTeX equation examples (for reference):**

Source projection:
$$q(x,t) = P_s[z(t), I(t), \chi(x)]$$

Field approximation (proxy, no PDE):
$$Y_c(t) = P_c[\phi_e, \mathbf{J}_e, \mathrm{CSD}](t)$$

Population firing rate:
$$r(t) = \frac{1}{N\Delta t}\sum_{n=1}^{N}\sum_{t'} \mathbb{I}[\text{spike}_n(t')]$$

**Rules:**
- Every mathematical section must include a displayed equation (not words only)
- Each equation must be followed by its term glossary, worded version, implementation location, and claim boundary
- Use `$$...$$` for display math in markdown/Jupyter
- Avoid bare variable names without definition
- Always link to code where equation is implemented
- Always state claim boundary (especially for proxy/approximate equations)

---

### **Section 4: Canonical Import**

**Purpose:** Set up the required JAX and jaxfne import conventions.

**Format:**
```markdown
## Canonical Import and Setup

Import jaxfne using the required alias:

\`\`\`python
import jax
import jax.numpy as jnp
from jax import vmap, jit

import jaxfne as jtfne
from jtfne.emitters import IzhikevichParams, simulate_eig_izhikevich
from jtfne.fields import project_laminar_sources, compute_conservation_proxy_diagnostics
from jtfne.io import json_safe, save_json

# Optional: Plotly (if used in this tutorial)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
\`\`\`

**Import rules:**
- ✓ `import jaxfne as jtfne` (required alias)
- ❌ `from jaxfne import *` (forbidden)
- ❌ `import jaxfne` (bare import forbidden; must alias as jtfne)
- ✓ Guarded Plotly import (optional, but use try/except if needed)
```

**Key rules:**
- Exact import alias `jtfne` (no alternatives)
- Guard optional imports (Plotly, matplotlib) with try/except
- Do not import secrets or hardcoded parameters

---

### **Section 5: Configuration and Model Construction**

**Purpose:** Define the simulation configuration, emitter parameters, and model setup.

**Format:**
```markdown
## Configuration and Model Setup

### Parameters

\`\`\`python
# Simulation parameters
duration_ms = 100.0  # Total simulation time
dt_ms = 0.1         # Timestep
n_steps = int(duration_ms / dt_ms)

# Neuron parameters (Izhikevich phenomenological)
params = jtfne.IzhikevichParams(
    a=0.02,  # Time scale (adap)
    b=0.2,   # Recovery coupling
    c=-65.0, # Spike reset (mV)
    d=8.0,   # Recovery jump
)

# Network parameters
n_neurons = 10  # Population size

# Geometry (laminar_proxy_no_pde mode)
dx_mm = 0.010  # Spatial resolution (0.010 mm per neuron)
dy_mm = 0.010
dz_mm = 0.010
\`\`\`

### Model construction

\`\`\`python
# Set up random seed for reproducibility
key = jax.random.PRNGKey(seed=0)

# Initial conditions [time=0]
V_m_init = jnp.full(n_neurons, -65.0)  # Resting potential (mV)
u_init = jnp.full(n_neurons, params.b * -65.0)  # Recovery at rest

# Configuration
config = jtfne.configuration()  # Use default TFNE config

# Construct model
model = jtfne.construct(
    config,
    emitters=[...]  # Specify emitters (list of callable neural update rules)
)
\`\`\`

**Key rules:**
- Declare all parameters explicitly (no magic numbers in simulation loop)
- Use realistic ranges (2–25 Hz firing, −100 to +40 mV voltages, physiological time constants)
- Initialize with plausible values (not NaN, not infinite)
- Geometry metadata must be declared explicitly (dx=dy=dz=0.010 mm)

---

### **Section 6: Simulation and Data Generation**

**Purpose:** Run the model and collect the raw simulation outputs.

**Format:**
```markdown
## Simulation

### Run the model

\`\`\`python
# Simulate the network
signals = model.simulation(
    duration_ms=duration_ms,
    dt_ms=dt_ms,
    random_key=key,
)

# Check outputs
print("Signals shape:", {
    'time_ms': signals.time_ms.shape,
    'V_m': signals.V_m.shape,
    'spikes': signals.spikes.shape,
})

# Verify finite values
assert jnp.all(jnp.isfinite(signals.V_m)), "V_m contains NaN/Inf"
assert jnp.all(jnp.isfinite(signals.spikes)), "spikes contains NaN/Inf"
print("✓ Numerical stability: all values finite")
\`\`\`

### Post-processing

\`\`\`python
# Extract spike times
spike_times = [jnp.where(signals.spikes[:, i])[0] * dt_ms for i in range(n_neurons)]
spike_times = [st.tolist() for st in spike_times]

# Compute firing rates
firing_rates_hz = jnp.array([len(st) / (duration_ms / 1000.0) for st in spike_times])
print(f"Mean firing rate: {firing_rates_hz.mean():.1f} ± {firing_rates_hz.std():.1f} Hz")

# Check acceptance gate: firing rate 2–25 Hz
assert jnp.all(firing_rates_hz >= 2.0), f"Firing rate < 2 Hz: {firing_rates_hz}"
assert jnp.all(firing_rates_hz <= 25.0), f"Firing rate > 25 Hz: {firing_rates_hz}"
print("✓ Firing rate gate: PASS")
\`\`\`

**Key rules:**
- Simulation path must be JAX-native and jittable (no Python loops in hot path)
- Verify finiteness immediately after simulation
- Extract spike times and compute basic statistics
- Report acceptance gate status (pass/fail)

---

### **Section 7: Probe and Multimodal Readout**

**Purpose:** Apply the 8 probe operators and compute multimodal readouts.

**Format:**
```markdown
## Probes and Multimodal Readout

### Apply probe operators

\`\`\`python
manifest = model.manifest(signals)

# Extract 8 probe results
probe_results = manifest['probe_report']

print("Probe operators:")
for op_name in ['spikes', 'V_m', 'source', 'lfp_proxy', 'csd_proxy', 'eeg_proxy', 'meg_proxy', 'emm_proxy']:
    if op_name in probe_results:
        result = probe_results[op_name]
        print(f"  {op_name}: {result['operator_status']}")
\`\`\`

### Multimodal visualization

\`\`\`python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Subplot 1: Spike raster
ax = axes[0]
for i in range(min(20, n_neurons)):  # Show first 20 neurons
    spike_times_i = jnp.where(signals.spikes[:, i])[0] * dt_ms
    ax.vlines(spike_times_i, i, i + 0.8, color='black', lw=0.5)
ax.set_xlim([0, duration_ms])
ax.set_ylim([0, min(20, n_neurons)])
ax.set_ylabel('Neuron index')
ax.set_title('Spike raster')

# Subplot 2: Voltage trace (sample neurons)
ax = axes[1]
for i in [0, 5, 10]:
    if i < n_neurons:
        ax.plot(signals.time_ms, signals.V_m[:, i], label=f'Neuron {i}', alpha=0.7)
ax.set_xlim([0, duration_ms])
ax.set_ylabel('V_m (mV)')
ax.set_title('Membrane voltage (sample neurons)')
ax.legend()
ax.grid(alpha=0.3)

# Subplot 3: Firing rate over time (sliding window)
ax = axes[2]
window_ms = 10.0
window_steps = int(window_ms / dt_ms)
firing_rate_t = []
for i in range(0, len(signals.time_ms) - window_steps):
    spikes_in_window = jnp.sum(signals.spikes[i:i+window_steps, :])
    fr = (spikes_in_window / window_steps / dt_ms) * 1000.0  # Convert to Hz
    firing_rate_t.append(fr)
ax.plot(signals.time_ms[:-window_steps], firing_rate_t, color='blue', lw=1)
ax.set_xlim([0, duration_ms])
ax.set_ylabel('Population rate (Hz)')
ax.set_xlabel('Time (ms)')
ax.set_title('Firing rate dynamics (sliding window)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/v030_XX_multimodal_readout.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/v030_XX_multimodal_readout.png")
\`\`\`

**Key rules:**
- Verify all 8 probe operators present in manifest
- Generate at least one PNG figure (required for acceptance)
- Use multimodal readouts (spikes, voltage, field proxies)
- Save figures with clear filenames (v030_XX_..._artifact_name.png)

---

### **Section 8: Manifest and Claim Gates**

**Purpose:** Extract and validate the JSON manifest, verify claim gates are frozen.

**Format:**
```markdown
## Manifest and Claim Gates

### Extract manifest

\`\`\`python
manifest = model.manifest(signals)

# Check manifest structure
print("Manifest keys:", list(manifest.keys()))
# Expected: ['basis', 'probe_report', 'validation_report', 'conservation_proxy_diagnostics', 'metadata']

# Extract claim gates
basis = manifest['basis']
print("\nClaim gates (frozen in v0.2.30):")
print(f"  physical_amplitude_claim_allowed: {basis['physical_amplitude_claim_allowed']}")
print(f"  biological_metabolism_claim_allowed: {basis['biological_metabolism_claim_allowed']}")
print(f"  claim_level: {basis['claim_level']}")
print(f"  field_solver_status: {basis['field_solver_status']}")
\`\`\`

### Verify claim gates

\`\`\`python
# Hard validation: claim gates must be frozen
assert basis['physical_amplitude_claim_allowed'] == False, "CLAIM GATE VIOLATED"
assert basis['biological_metabolism_claim_allowed'] == False, "CLAIM GATE VIOLATED"
assert basis['claim_level'] == 'computational_scaffold', "CLAIM GATE VIOLATED"
assert basis['field_solver_status'] == 'laminar_proxy_no_pde', "CLAIM GATE VIOLATED"

print("\n✓ All claim gates frozen and immutable")
print("\nTruth status:")
print("  - This tutorial demonstrates a computational model.")
print("  - Outputs are proxies, not physical measurements.")
print("  - No biological validation or calibration.")
print("  - No Maxwell/Poisson solvers (laminar_proxy_no_pde mode).")
print("  - Safe for exploration and teaching; not for empirical claims.")
\`\`\`

### Save manifest

\`\`\`python
# Validate manifest is JSON-safe
try:
    import json
    manifest_json = jtfne.json_safe(manifest)
    with open('outputs/v030_XX_manifest.json', 'w') as f:
        json.dump(manifest_json, f, indent=2)
    print("✓ Saved: outputs/v030_XX_manifest.json (JSON-safe)")
except Exception as e:
    print(f"✗ Failed to save manifest: {e}")
    raise
\`\`\`

**Key rules:**
- Extract claim gates explicitly
- Hard validation: assert all gates match expected frozen values
- Save manifest to JSON (no NaN/Inf)
- Report truth status clearly

---

### **Section 9: Figures and Artifacts**

**Purpose:** Generate publication-quality figures and store artifact metadata.

**Format:**
```markdown
## Figures and Artifacts

### Figure 1: [Figure description]

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

# Plot logic here
ax.plot(..., label='...')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('[Unit]')
ax.set_title('Figure 1: [Title]')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/v030_XX_figure_1_name.png', dpi=150, bbox_inches='tight')
plt.close()

# Compute hash for integrity validation
import hashlib
with open('outputs/v030_XX_figure_1_name.png', 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()
print(f"✓ Figure 1: {sha256}")
\`\`\`

### Figure 2: [Figure description]

[Repeat pattern for each figure]

### Artifact metadata

\`\`\`python
# Record artifact hashes and metadata
artifacts = {
    'figure_1': {
        'filename': 'v030_XX_figure_1_name.png',
        'sha256': '[computed above]',
        'description': '[brief description]',
        'figure_type': 'raster | trace | heatmap | spectrum | etc.',
    },
    'figure_2': { ... },
    ...
}

# Save to JSON
with open('outputs/v030_XX_artifacts.json', 'w') as f:
    json.dump(artifacts, f, indent=2)
print("✓ Saved: outputs/v030_XX_artifacts.json")
\`\`\`

**Key rules:**
- Generate at least 2–3 PNG figures per tutorial
- Use high DPI (150+) and clear labels
- Compute SHA256 hash for integrity validation
- Record all figure metadata in JSON

---

### **Section 10: Interpretation and Analysis**

**Purpose:** Interpret the results, explain what the figures reveal, and connect back to the biological question.

**Format:**
```markdown
## Interpretation and Analysis

### What do the results show?

[Describe what the simulation produced. Reference the figures. Explain the key observations in plain language.]

Example:
> The spike raster (Figure 1) shows that the network produces highly regular, bursting activity with a population firing rate of 12.3 Hz (well within the 2–25 Hz acceptance range). Individual neurons fire in a loosely synchronized manner, with each spike visible in the voltage trace (Figure 2) as a rapid depolarization followed by recovery. The LFP-proxy (Figure 3, not shown) reveals dominant oscillatory power in the 10–15 Hz range, consistent with alpha-band activity in cortical recordings.

### Connection to biological question

[Relate the computational results back to the real neuroscience question posed in Section 2.]

Example:
> This demonstrates that the Izhikevich model can capture key features of action potential generation: rapid upstroke (V_m rises from −65 to +30 mV in ~1 ms), spike threshold (crossing ~−40 mV), repolarization (recovery variable u drives return to baseline), and adaptation (increasing u slows subsequent spikes). However, the model is phenomenological—it does not account for ionic currents, channel kinetics, or dendritic compartments that the Hodgkin-Huxley model captures.

### Scope limitations

[Explicitly state what the tutorial does NOT demonstrate.]

Example:
> This tutorial does NOT:
> - Validate the Izhikevich model against real neurons (no patch-clamp data comparison)
> - Prove that the brain uses this mechanism (it's a possible mathematical description, not biological proof)
> - Model dendritic computation, synaptic conductances in detail, or multi-compartment morphology
> - Claim biological realism beyond phenomenological spike shape matching

These limitations are expected and intended: v0.3 tutorials are computational scaffolds for teaching, not biological validation engines.

\`\`\`

**Key rules:**
- Interpret results in plain language
- Reference specific figures
- Connect back to the biological question (Section 2)
- Explicitly state scope limitations
- Avoid overclaiming

---

### **Section 11: Failure Modes and Edge Cases**

**Purpose:** Discuss what can go wrong, how to detect it, and how to fix it.

**Format:**
```markdown
## Failure Modes and Edge Cases

### Failure 1: Non-finite values (NaN/Inf)

**Symptom:** Simulation produces NaN or Inf values in V_m or u.

**Causes:**
- Timestep too large (dt_ms > 1 ms for Izhikevich)
- Parameters outside plausible range (e.g., a < 0, d > 100)
- Numerical instability in state update

**Detection:**
\`\`\`python
assert jnp.all(jnp.isfinite(signals.V_m)), "Voltage contains NaN/Inf"
\`\`\`

**Fix:**
1. Reduce dt_ms to 0.05 ms or smaller
2. Check parameter ranges match literature values
3. Verify initial conditions (V_m_init, u_init) are reasonable

### Failure 2: Dead neurons (0 Hz firing rate)

**Symptom:** One or more neurons never spike (spike_times = []).

**Causes:**
- Resting potential above spike threshold (c too high)
- Recovery coupling too strong (b too large)
- Insufficient input current

**Detection:**
\`\`\`python
for i, fr in enumerate(firing_rates_hz):
    if fr < 2.0:
        print(f"Warning: Neuron {i} firing rate {fr:.1f} Hz (below 2 Hz minimum)")
\`\`\`

**Fix:**
1. Lower spike reset threshold c (try c = −65 mV)
2. Reduce recovery coupling b (try b = 0.2)
3. Add input current or increase noise

### Failure 3: Explosive firing (> 25 Hz)

**Symptom:** Population firing rate > 25 Hz, network appears unstable.

**Causes:**
- Parameter set produces instability (e.g., a too small)
- Input current too large
- Excessive synaptic feedback

**Detection:**
\`\`\`python
if firing_rates_hz.mean() > 25.0:
    print(f"Warning: Mean firing rate {firing_rates_hz.mean():.1f} Hz (above 25 Hz maximum)")
\`\`\`

**Fix:**
1. Increase time constant a (try a = 0.05)
2. Reduce input current
3. Add inhibitory feedback (negative synaptic weights)

[Add 2–3 more failure modes relevant to the specific scenario]

\`\`\`

**Key rules:**
- List at least 3–5 common failure modes
- Describe detection and diagnosis
- Provide specific remedies
- Connect to acceptance gates

---

### **Section 12: Exercises and Extensions**

**Purpose:** Suggest follow-up experiments, modifications, and learning exercises.

**Format:**
```markdown
## Exercises and Extensions

### Exercise 1: Parameter sweep

Modify the parameter sweep to explore how spike frequency changes with parameter `a`:

\`\`\`python
a_values = jnp.linspace(0.01, 0.1, 10)
firing_rates = []

for a_val in a_values:
    params_sweep = jtfne.IzhikevichParams(a=a_val, b=0.2, c=-65, d=8.0)
    # Run simulation with modified params
    # Extract firing rate
    firing_rates.append(...)

plt.plot(a_values, firing_rates)
plt.xlabel('Parameter a')
plt.ylabel('Firing rate (Hz)')
plt.title('Frequency tuning by adaptation')
plt.savefig('outputs/v030_XX_exercise_1_parameter_sweep.png')
\`\`\`

### Exercise 2: Noise robustness

Add Gaussian noise to the input current and repeat the simulation. How does noise affect firing rate and variability?

### Extension 1: Multi-neuron heterogeneity

Repeat the tutorial with heterogeneous parameters (different a, b, c, d per neuron). How does diversity affect population activity?

### Extension 2: Adaptive synaptic strength

Implement synaptic plasticity (STDP or homeostatic scaling) and rerun. How does plasticity change the network dynamics?

[Add 2–3 more exercises and extensions specific to the scenario]

\`\`\`

**Key rules:**
- Provide code snippets (runnable Python)
- Design exercises that deepen understanding
- Suggest extensions that push toward v0.3.X+ scenarios
- Keep difficulty levels graduated

---

### **Section 13: Non-Claim Statement (Mandatory)**

**Purpose:** Final, explicit, immutable statement of truth status and boundaries. This section must appear in every v0.3 tutorial.

**Format:**
```markdown
## Non-Claim Statement

### What this tutorial IS

✓ An executable, reproducible computational scaffold  
✓ A demonstration of how the Izhikevich model operates  
✓ A teaching tool for understanding neural dynamics  
✓ Suitable for exploration, design iteration, and hypothesis generation  
✓ A foundation for understanding more complex models (Hodgkin-Huxley, multi-compartment, networks)

### What this tutorial IS NOT

❌ A biological validation or calibration of neural mechanisms  
❌ A proof that the Izhikevich model describes real neurons  
❌ An empirical study; no comparison to experimental data  
❌ A claim of physiological realism beyond phenomenological spike shape matching  
❌ Evidence for or against any specific neural computation  
❌ A whole-brain simulator or large-scale network model  

### Scientific boundaries (immutable as of v0.2.30)

- **truth_mode:** truth_safe_unverified
- **claim_level:** computational_scaffold
- **physical_amplitude_claim_allowed:** False
- **field_solver_status:** laminar_proxy_no_pde (no Maxwell/Poisson PDE solvers)
- **source_calibration_status:** uncalibrated (teaching proxy)

This tutorial uses v0.2.30 of jaxfne, a stable baseline for v0.3 tutorial work. The package version will not be bumped unless a tutorial reveals a real bug in the core library or a genuinely missing public API.

### Invitation to contribute

If you:
- Find a bug in the code (please open a GitHub issue)
- Identify misleading language (please request a clarification)
- Have ideas for future extensions (v0.3.X+ phases)

Please reach out via GitHub Issues or Discussions. This is a living, community-driven tutorial atlas.

### Citation

If you use this tutorial in research or teaching, please cite:

> jaxfne v0.2.30 Tutorial Atlas. Hamed Nejat and contributors. https://github.com/HNXJ/jaxfne/docs/tutorials_v030/

(Exact citation format may be updated as v0.3 stabilizes.)

---

**End of Section 13: Non-Claim Statement**

This marks the end of this tutorial. The next scenario in the learning path is [link to v0.3.X+1].

\`\`\`

**Key rules (MANDATORY):**
- Must appear as Section 13 in EVERY v0.3 tutorial notebook
- Must explicitly list what the tutorial IS and IS NOT
- Must include immutable claim gates
- Must state that physical_amplitude_claim_allowed = False
- Must prevent reader misconception about scope, calibration, biological proof

---

## Quality Checklist for v0.3 Tutorial Authors

Before committing a tutorial notebook, verify:

- [ ] **Section 1:** Learning objectives clear and measurable
- [ ] **Section 2:** Biological question, computational analog, scope boundary all stated
- [ ] **Section 3:** All symbols defined in glossary table; units present
- [ ] **Section 4:** Import alias `jtfne` used consistently; no forbidden imports
- [ ] **Section 5:** All parameters declared explicitly; geometry metadata present
- [ ] **Section 6:** Simulation runs without error; finiteness validated; acceptance gates checked
- [ ] **Section 7:** All 8 probe operators present in manifest; at least one PNG figure saved
- [ ] **Section 8:** Claim gates extracted and validated; manifest JSON-safe
- [ ] **Section 9:** ≥2 PNG figures with SHA256 hashes; artifact metadata JSON saved
- [ ] **Section 10:** Interpretation in plain language; connection to biological question; scope limitations stated
- [ ] **Section 11:** ≥3 failure modes described with detection and fix
- [ ] **Section 12:** ≥2 exercises and ≥2 extensions suggested
- [ ] **Section 13:** Non-claim statement present and complete; Section 13 is last cell

### Code quality checks
- [ ] No syntax errors; notebook runs end-to-end without errors
- [ ] All figures saved with filenames: `v030_XX_description.png` (XX = scenario number, padded)
- [ ] SHA256 hashes computed and stored in JSON
- [ ] Manifest JSON valid (tested with `json.dumps(manifest_json)`)
- [ ] No hardcoded paths (use relative paths or `outputs/` directory)
- [ ] No API calls to external services (all computation local)

### Acceptance gate validation
- [ ] Firing rate 2–25 Hz for all populations (or explicitly justified deviation)
- [ ] All numerical values finite (assert jnp.all(jnp.isfinite(...)))
- [ ] JSON manifest serializable and safe (no NaN/Inf)
- [ ] JAX simulation path jittable (no Python loops in hot path)
- [ ] Geometry metadata declared (dx=dy=dz values explicit)
- [ ] Figure artifacts present and hashed
- [ ] Claim gates frozen (physical_amplitude_claim_allowed=False)

---

## Examples and Template Notebooks

A complete example template notebook will be provided in:
- `tutorials/v030_template_example.ipynb` (auto-generated from this markdown)

Authors can copy this notebook and fill in the content for their specific scenario.

---

## See Also

- [Scenario Index](scenario_index.md) — 15-scenario spine
- [README](README.md) — Overview, acceptance gates, learning path
- [v0.3 Tutorial-Scenario Plan](../v030_tutorial_scenario_plan.md) — Doctrine
- [Canonical Imports](canonical_imports.md) — Import conventions
