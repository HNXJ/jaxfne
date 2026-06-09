# Glossary of Methods

This page documents the numerical methods, algorithms, and technical approaches used in jaxfne.

## Integrators and Solvers

### Forward Euler (`euler_step`, `euler_scan`)

A simple first-order numerical method for integrating ordinary differential equations:

$$y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$$

**Use case:** State vector evolution (e.g., neural membrane voltage over time).

**Properties:**
- O(Δt) local truncation error
- Explicitly used for Izhikevich and GLIF emitters
- Stable for sufficiently small timesteps
- Implemented via `jax.lax.scan` for efficient batching

**Not a field solver:** This is a temporal integrator for neural state, not a spatial PDE solver.

## Field Projection Methods

### Laminar Proxy (LFP, CSD, EEG, MEG)

Laminar proxies estimate extracellular signals from neuronal sources without solving Maxwell's equations or assuming tissue conductivity.

#### Local Field Potential (LFP)

Computed as a weighted spatial projection of source currents:

$$\text{LFP} \propto \int \text{CSD}(z') \cdot K(|z - z'|) \, dz'$$

where $K$ is a Gaussian kernel and CSD is current source density.

**Status:** Proxy-based readout, not field solve. No physical amplitude claims.

#### Current Source Density (CSD)

Estimated from intracellular ionic currents using source geometry:

$$\text{CSD} = -\nabla \cdot \mathbf{I}_\text{transmembrane}$$

**Status:** Computational diagnostic. Approximates multi-compartment transmembrane current distribution.

#### EEG Proxy

EEG signals are computed as a far-field approximation (single dipole moment):

$$\text{EEG} \propto \sum_i I_i \cdot \text{position}_i$$

Assumes dipole far-field and negligible conductivity variation.

**Status:** Extreme approximation. Physical amplitude claims not allowed.

#### MEG Proxy

MEG is computed as a weighted integral of source currents over space. Similar to EEG but sensitive to magnetic dipole moment:

$$\text{MEG} \propto \oint \mathbf{I}(z) \times \mathbf{r}(z) \, dz$$

**Status:** Proxy only. No physical amplitude claims.

---

## Truth Gates and Claim Labels

### Truth Modes

- **`truth_safe_unverified`** — Computational scaffold. Results are not empirically validated and should not be published as neural biology without external validation.

### Claim Levels

- **`computational_scaffold`** — Intermediate computational output. Not a ground-truth measurement.
- **`proxy_readout_only`** — Proxy-based approximation. No claim of physical accuracy.

### Physical Amplitude Claims

By default, **`physical_amplitude_claim_allowed = False`**. This prevents downstream code from interpreting proxy outputs as physical measurements (mV, µV, etc.) without explicit calibration.

---

## Optimization Methods

### AGSDR (Adaptive Gradient + Spectral Descent Response)

A two-phase optimizer that combines:

1. **Gradient phase:** Optax-based optimization (Adam, SGD)
2. **Spectral phase:** Adaptive dampening based on loss spectral properties

Used for tuning Izhikevich parameters and readout weights.

**References:** See `jaxfne.optim.AGSDR` and `jaxfne.optim.agsdr_transform`.

### Gradient Estimation Methods

- **SDR** — Spectral descent response
- **GSDR** — Generalized spectral descent response
- **AGSDR** — Adaptive generalized spectral descent response

All use `jax.grad()` under the hood but apply spectral adaptive damping.

---

## Configuration and Construction

### Network Construction

Networks are built via `jaxfne.construct(config)` which:

1. Parses the configuration schema
2. Allocates emitter state (Izhikevich parameters, etc.)
3. Builds sparse connectivity matrices
4. Initializes PRNG keys for stochastic elements

### Simulation State

Simulations track:

- **Membrane voltage** $V_m(t)$ [mV]
- **Spike times** (binary or graded)
- **Field outputs** (LFP, CSD, EEG, MEG if requested)

State is immutable across time (functional programming via JAX).

---

## References

- **Field proxies:** Reimann et al. (2013), Koch & Segev (1998)
- **Izhikevich model:** Izhikevich (2003) "Simple model of spiking neurons"
- **Optimization:** Boyd & Vandenberghe (2004), Nesterov & Nemirovskii (1994)

---

**See also:**
- [API Reference](../api/index.md)
- [Tutorials](../tutorials/index.md)
- [Guides](../guides/index.md)
