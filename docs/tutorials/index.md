# Tutorials

Learn jaxfne by working through progressively detailed examples. Each tutorial builds on the previous one.

## Notebook standard

All tutorials follow the **[Colab notebook standard](notebook_standard.md)**. This standard ensures tutorials are:

- Reproducible in fresh Colab environments with minimal dependencies
- CPU-safe with optional GPU acceleration
- Portable across platforms (nbconvert-compatible)
- Properly cleared and version-verified before commit

Start with the [notebook standard](notebook_standard.md) to understand the structure and validation guidelines used in all tutorials.

## Tutorial stack

The tutorial progression teaches the source-to-field/readout workflow, from single-neuron models to multi-area laminar circuits:

| Number | Topic | Type | Focus | Version |
|--------|-------|------|-------|---------|
| **Suite 1** | Computational Biophysics | Interactive Colab | 4-part course: models → circuits → readouts → optimization | v0.3.3+ |
| **Suite 2** | Corticospectrolaminar Motif | Runnable Notebook | Compact V1/PFC spectrolaminar motif and visual analysis | v0.3.4+ |
| **Suite 3** | Low-Frequency Scaling | Runnable Notebook | Scale-dependent low-frequency proxy readouts and boundary validation | v0.3.9+ |
| **09** | EEG/MEG/EMM Proxy Bundle | Runnable Notebook | Separate sensor pathways for scalp potential, magnetic field, and metabolic proxy | v0.3.10+ |
| **10** | Sensory Omission & Oddball | Runnable Notebook | Expected sensory stimuli, unexpected deviants, and sensory omissions | v0.3.13+ |
| — | Continuous Sequential Omission Oddball | Runnable Notebook | `general_sequential_oddball_paradigm` backbone + per-event `target_indices` L4-E-only targeting in a 100-neuron V1 column | v0.4.0+ |
| **06** | Chainable Configuration (100-neuron E/I) | Runnable notebook | New Configuration API: method chaining, E/I population dynamics | v0.3.6+ |
| **07** | v0.3.7 Source Bookkeeping | Interactive HTML | 3D visualization of source/field/probe workflow | v0.3.7+ |
| **08** | v0.3.8 LFP/CSD Readout | Runnable notebook | Laminar contact projection, Gaussian kernels, CSD-proxy | v0.3.8+ |
| **01** | Single-neuron Multimodal | Runnable notebook | Izhikevich emitter, spikes, voltage, field readouts | v0.2.8+ |
| **02** | Two-neuron E/I | Runnable notebook | Coupling, recurrent dynamics | v0.2.9+ |
| **03** | 100-neuron Network | Runnable notebook | Population dynamics, stability | v0.2.10+ |
| **04** | V1 Six-layer Column | Documentation guide | Laminar anatomy, depth-specific readouts | v0.2.11+ |
| **05** | V1-PFC Dual Column | Documentation guide | Cross-area interaction, traveling waves | v0.2.14+ |
| — | NeuronalTensor (tensor-first circuits) | Runnable notebook | Declarative Areas/Layers/NeuronTypes, HDP homeostatic plasticity | 0.4.7+ |


## Featured: jaxfne Suite No. 1

**[Computational Biophysics](06_jaxfne_suite_no_1_computational_biophysics.md)** (interactive Colab)

A comprehensive 4-part course covering:
- Part 1: Single-neuron models and biophysics
- Part 2: Vectorized circuits and connectivity
- Part 3: Laminar cortical columns with readout operators (LFP-proxy, CSD-proxy, spectral analysis)
- Part 4: Hypothesis tuning via optimization

**22 figures**, export metrics, and immutable scope fields throughout. CPU-safe, runs in 2–3 minutes on Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb)

---

## Featured: jaxfne Suite No. 2

**[Corticospectrolaminar Motif](07_jaxfne_suite_no_2_spectrolaminar_motif.md)** (interactive notebook)

A comprehensive tutorial covering:
- Part 1: Declaring cortical column anatomy and multi-column loops
- Part 2: Vectorized JAX-first population simulations
- Part 3: Sampling multimodal sensor proxies (MUA, LFP, CSD, EEG, MEG, EMM)
- Part 4: High-resolution spectrolaminar visualizations
- Part 5: Evoked responses and baseline spectrolaminar heatmaps
- Part 6: CPU-safe parameter search and tuning trajectories

**13 figures**, strict JSON evidence manifests, and JAX-based vis tools.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)

---

## Featured: jaxfne Suite No. 3

**[Scale-Dependent Low-Frequency Structure in Proxy Field Readouts](08_jaxfne_suite_no_3_low_frequency_scaling.md)** (interactive notebook)

A comprehensive tutorial covering:
- Part 1: Declaring scale-dependent proxy configurations
- Part 2: Vectorized population simulations across varied sizes
- Part 3: Computing relative power spectral densities (PSD) and bandpower metrics
- Part 4: Investigating synchrony and Fano proxies by scale
- Part 5: Exporting strict validation JSON manifests and reports

**Five figures**, strict verification constraints, and low-frequency scaling diagnostics.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Featured: jaxfne v0.3.10 EEG/MEG/EMM Proxy Bundle

**[Multimodal Sensor Projections and EEG/MEG/EMM Proxy Bundle](09_v0310_eeg_meg_emm_proxy_bundle.md)** (interactive notebook)

A comprehensive tutorial covering:
- Part 1: Declarative setup and population simulation (100 neurons)
- Part 2: Scalp potential projections via EEG-proxy operators
- Part 3: Oriented magnetic field projections via MEG-proxy operators
- Part 4: Energy cost timeline projections via metabolic EMM-proxy operators
- Part 5: Validation receipt and independent panel figures

**Three panel figures** and structured validation manifest exports.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb)

---

## Featured: jaxfne v0.3.13 Sensory Omission & Oddball Detection

**[Sensory Omission & Oddball Detection Paradigm](10_v0313_omission_oddball.md)** (interactive notebook)

A comprehensive tutorial covering:
- Part 1: Declarative expected sensory stimuli and unexpected deviant tone setups
- Part 2: Configuring sensory omission conditions (expected silence)
- Part 3: Running stimulus-locked trials with windowed time segmentation
- Part 4: Comparing expected vs deviant vs omission population activity and LFP/CSD proxy readouts
- Part 5: Exporting JSON-safe paradigm reports and validation manifests

**Five plots** and structured validation manifests.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v0313_omission_oddball.ipynb)

---

## Featured: jaxfne v0.4.0 Continuous Sequential Omission Oddball

**[`jaxfne_v040_continuous_omission_oddball.ipynb`](https://github.com/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v040_continuous_omission_oddball.ipynb)**
(runnable notebook) — the current reference example for the
`general_sequential_oddball_paradigm` backbone (see `jaxfne-paradigm-design`),
run end to end on a 100-neuron V1 column:

- Part 1: 4-slot sequential paradigm timing (fixation + p1-p4 + delays), one
  slot declared as the omission via `omission_tokens`
- Part 2: a per-event `target_indices` key, built from `model.neuron_table()`,
  stamped onto each event dict passed to `StimulusSchedule(events=...)` to
  drive only L4 E cells per slot — the worked example for per-neuron-subset
  stimulus targeting referenced from
  [`AGENTS.md`](https://github.com/HNXJ/jaxfne/blob/main/AGENTS.md) and the
  README's "Which workflow should I use?" section
- Part 3: Epoch-aligned population spiking and laminar LFP/CSD-proxy readouts
  across expected, deviant, and omission conditions
- Part 4: JSON-safe paradigm + validation manifest export

Computational scaffold throughout: `claim_level=computational_scaffold`,
`field_solver_status=linear_solver`, `physical_amplitude_calibrated=False`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v040_continuous_omission_oddball.ipynb)

---

## Featured: jaxfne v0.3.6 Chainable Configuration API

**[100-Neuron Excitatory-Inhibitory Population](06_v036_100_neuron_ei_population.md)** (interactive notebook)

Introduces the **new fluent Configuration API** (method chaining) for streamlined model composition:

```python
cfg = (jtfne.Configuration()
    .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    .column(name="L2/3_column", layers=["L2/3"], n=100)
    .cell_types({"E": 0.75, "PV": 0.25})
    .connectivity()
    .set_emitter(family="izhikevich", preset="cortical_eig")
    .probes(["SPK", "Vm", "source", "LFP-proxy", "CSD-proxy"]))
```

---

## Featured: NeuronalTensor (tensor-first circuits)

**[jaxfne_neuronal_tensor_first.ipynb](https://github.com/HNXJ/jaxfne/blob/main/tutorials/jaxfne_neuronal_tensor_first.ipynb)** (runnable notebook, `nbclient`-executed; see also the standalone script version: [08_neuronal_tensor_first.py](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py))

`NeuronalTensor` is a second, declarative way to define a circuit — an
explicit `Areas x Layers x NeuronTypes` data model that JSON round-trips and
converges on the same `Model`/`Signals` as the `Configuration` path used by
every tutorial above:

```python
tensor = jtfne.NeuronalTensor(areas=[...])
model  = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

It also carries the package's canonical 1000-neuron V1 column
(`jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")`) and is
the path to use for **HDP homeostatic plasticity** (per-cell-type cube-law
adaptation time constant) — see [HDP guide](../guides/hdp.md) § "Tensor-first"
and the [API reference](../api/neuronal_tensor.md).

A comprehensive tutorial covering:
- Part 1: Biological question (balanced E/I coupling)
- Part 2: Configuration via method chaining
- Part 3: Simulation and population readouts
- Part 4: Manifest with scope metadata
- Part 5: Five figures
- Part 6: Scope boundaries and limitations

**Intermediate difficulty**, CPU-safe, runs in ~1–2 minutes on Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v036_100_neuron_ei_population.ipynb)

---

## Beginner tutorials

**[Single-neuron Multimodal](01_single_neuron_multimodal.md)**

Start here. Build, simulate, and inspect a single Izhikevich neuron with spikes, voltage, and readout operators.

**[Two-neuron E/I](02_two_neuron_ei.md)**

Excitatory and inhibitory neurons connected. Observe recurrent dynamics and coupling effects.

## Intermediate tutorials

**[100-neuron Network](03_network_100_ei.md)**

A balanced network of excitatory and inhibitory neurons. Explore local population activity and stability.

**[V1 Six-layer Column](04_v1_column.md)**

A laminar model inspired by primate V1 with six layers (L1, L2/3, L4, L5, L6) and depth-specific readouts.

## Advanced tutorial

**[V1-PFC Dual Column](05_v1_pfc_dual_column.md)**

Two cortical columns (V1 and PFC) with a feedforward inter-areal connection, driving a continuous AAAB local-oddball adaptation paradigm with real trial-to-trial HDP weight/homeostasis carryover. A working script (not yet a polished notebook) -- verified stable over 100 chained trials.


## Running tutorials

Tutorials live under `tutorials/` as Jupyter notebooks. Headless CI runners live under `examples/` (e.g. `v031_*`, `v033_*`).

```bash
jupyter notebook tutorials/jaxfne_v031_single_neuron.ipynb
```

Or execute headless:

```bash
python examples/v031_single_izhikevich_neuron.py
```

## Quick example: Single-neuron primer

```python
import jaxfne as jtfne

cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=100.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=0)

idx_e = model.select(cell_type="E")
spk = signals.get("spk", cell_type="E")
print(f"E spike count: {int(spk.sum())}")
```

## Next steps

After tutorials:

- **[Guides](../guides/index.md)** for how-to articles and workflow tips
- **[API reference](../api/index.md)** for full class/function documentation
- **[Jaxley interoperability](../guides/jaxley_interop.md)** for using external models
