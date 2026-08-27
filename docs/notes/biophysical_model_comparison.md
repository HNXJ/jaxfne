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

## 5. Positioning: what would make this non-redundant with existing tools (honest status, 2026-07-17)

Sections 1-4 describe structural analogy to established formulations, not a capability
or superiority claim (see `scripts/evidence_figures/fig08_adjacent_tools_comparison.py`,
which explicitly disclaims speedup/accuracy/biological-validity claims). This section
states plainly what is and is not true today, and names the specific unbuilt capability
that would make jaxfne's combination of features non-redundant with existing tools,
rather than leaving that question unanswered.

**True today:**
- jaxfne's spiking simulation (`simulate_edge_recurrent_izhikevich*`) is implemented in
  JAX and differentiable through `jax.lax.scan`, in the same family as Jaxley.
- HDP (`simulate_edge_recurrent_izhikevich_hdp`) adds a per-neuron resource-state
  homeostatic controller (`H_i`) driving synaptic weight plasticity — a mechanism not
  present in this specific form in Jaxley, Brian2, or NEST.
- The source-to-field readout (`jaxfne.fields.project_laminar_sources`) is a static,
  non-learned linear projection (`field_solver_status = "linear_solver"`), not a
  differentiable solve and not an elliptic/PDE field equation. Per
  `docs/source_field_equations.md`, the physically-grounded source modes
  (`total_membrane_current`, `decomposed_cap_ion_syn`) remain reserved, not implemented.

**Not yet true — the actual, unbuilt differentiation opportunity:** no tool surveyed
here (jaxfne included, as of this writing) offers a single differentiable pipeline from
spiking dynamics, through homeostatic plasticity, through to a field/LFP-CSD readout —
i.e., computing `dL/d(any network or plasticity parameter)` for a loss `L` against a
recorded extracellular signal. Brian2/NEST are not differentiable at all; Jaxley is
differentiable through multicompartment biophysics but has no field-projection layer or
HDP-style homeostasis; jaxfne has the plasticity mechanism and the JAX substrate, but
its field projection is not yet a differentiable solve. Building that solve (replacing
the current `linear_solver` placeholder) is the concrete target that would turn "field"
in Tensor-Field Neural Equations into a demonstrated capability rather than a name.
Tracked via `artifacts/publication/publication_evidence_index.json`.

**Update (2026-07-18):** `fig08_adjacent_tools_comparison.py`'s "no speedup claims"
disclaimer covers a *general, systematic* performance claim, which still doesn't exist.
A first, narrow, honestly-caveated quantitative data point now does: a matched
Izhikevich sparse-network task (same N, same ~100-in-degree connectivity, same
duration, CPU, default settings both sides) showed jaxfne 4.7-18.9x faster than Brian2
at N=1,000 and 1.2-11.3x faster at N=5,000 (construct/simulate respectively). See
[Brian2 Benchmark Receipt](brian2_benchmark_receipt.md) for the full setup, honest
caveats (different RNG streams, small-scale smoke test not a benchmark suite, Brian2's
default rather than maximally-tuned backend), and reproduction instructions.

---

## References

- Arkhipov et al. (2018), Gouwens et al. (2018), Billeh et al. (2020), Rimehaug et al. (2023) —
  point-neuron, multicompartment, and extracellular-readout formulations whose mathematical
  structure the TFNE operator chain mirrors.
