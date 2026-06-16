# Biophysical Model-Comparison Notes

These notes describe how the Tensor-Field Neural Equation (TFNE) formulations in
`jaxfne` relate to established point-neuron, multicompartment, and extracellular-readout
formulations from the computational-neuroscience literature. The goal is to map the
package's operator chain onto familiar mathematical building blocks.

---

## 1. Operator chain

The standard `jaxfne` workflow maps a multiscale circuit onto a tensor-parallel
computation graph:

```
[Emitter (Local Spiking / HH Dynamics)]
        │
        ▼ (Presynaptic Spikes / Conductances)
[Synapse / Connectivity Layer]
        │
        ▼ (Recurrent Drive & Aggregate Inputs)
[Source Layer (Transmembrane Current Bookkeeping)]
        │
        ▼ (Laminar / Spatial Geometry Weighting)
[Passive Field / Linear Readout]
        │
        ▼ (Extracellular Contact Probes)
[LFP / CSD / EEG / MEG Readout Proxies]
```

Local nonlinearities (voltage gates, synaptic receptors, adaptation variables) live
inside the **Emitter** and **Synapse** components, while spatial propagation and
electrode readouts are represented as parallelized **Linear Readout** operators.

---

## 2. Core modeling formulations

`jaxfne` adopts canonical mathematical formulations used across computational-neuroscience
benchmarks:

### 2.1. Local conductance & adaptation
Multicompartment Hodgkin-Huxley-style gate kinetics, calcium shells, and voltage-reset
adaptation are represented inside the generalized **Emitter** loop.

### 2.2. Network connectivity & synaptic kernels
Point-neuron approximations and compartmental models map recurrent connectivity weights
through explicit synaptic trace equations. Postsynaptic current updates follow:

- **Exponential Synapse**:
  $$s_j[t+1] = s_j[t]\exp\left(-\frac{\Delta t}{\tau_j}\right) + z_j[t]$$
- **Alpha Synapse**:
  $$k_j(t) = \frac{t}{\tau_j}\exp\left(1-\frac{t}{\tau_j}\right)$$
- **Double-Exponential Synapse**:
  $$k_j(t) = A_j \left(e^{-t/\tau_{decay,j}} - e^{-t/\tau_{rise,j}}\right)$$

### 2.3. Dual multiscale representation
TFNE architectures separate cellular dynamics from low-dimensional network descriptions
(e.g., GLIF vs. full biophysical compartment models), so emitter classes can be swapped
while a unified linear readout layer is retained.

### 2.4. Extracellular signal readouts
Transmembrane currents are projected onto extracellular recording probes (e.g., laminar
silicon probes) using a linear transfer-resistance proxy model:
$$Y_c(t) = \sum_n W_{cn} S_n(t)$$

---

## 3. Proxy-readout scope

Extracellular metrics (LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy) are
numerical proxy readouts for objective functions, optimization, and system-level
comparison. They use uncalibrated proxy units under the package truth gates
(`field_solver_status = "linear_solver"`, `physical_amplitude_calibrated = False`).
See [Limitations and future plans](../limitations_and_future_plans.md) for the centralized
scope statement.

---

## 4. Canonical coding pattern

```python
import jaxfne as jtfne

# 1. Instantiate the bridge (e.g. JaxleyBridge)
bridge = jtfne.bridges.JaxleyBridge(
    model=jaxley_model,
    source_mode="transmembrane_current",
    compartment_axis="last"
)

# 2. Extract transmembrane source bookkeeping (proxy units)
sources = bridge.extract_sources(simulation_result)

# 3. Plot proxy signals using the visualizer namespace
fig = jtfne.vis.lfp(sources)
```

---

## References

- Arkhipov et al. (2018), Gouwens et al. (2018), Billeh et al. (2020), Rimehaug et al. (2023) —
  point-neuron, multicompartment, and extracellular-readout formulations whose mathematical
  structure the TFNE operator chain mirrors.
