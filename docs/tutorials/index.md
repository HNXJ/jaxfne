# Tutorials

Learn jaxfne by working through progressively detailed examples, organized into
three families: **Suites** (multi-part interactive courses), **Versioned
tutorials** (the numbered single-topic progression), and **Étude notebooks**
(focused research-style circuits grouped by theme).

## Notebook standard

All tutorials follow the **[Colab notebook standard](notebook_standard.md)**. This standard ensures tutorials are:

- Reproducible in fresh Colab environments with minimal dependencies
- CPU-safe with optional GPU acceleration
- Portable across platforms (nbconvert-compatible)
- Properly cleared and version-verified before commit

Start with the [notebook standard](notebook_standard.md) to understand the structure and validation guidelines used in all tutorials.

---

## Suites

Multi-part interactive courses. Each is a single notebook covering a full
workflow arc (models → circuits → readouts → optimization).

| Suite | Topic | Focus | Version |
|-------|-------|-------|---------|
| **[Suite 1](06_jaxfne_suite_no_1_computational_biophysics.md)** | Computational Biophysics | 4-part course: single-neuron models → vectorized circuits → laminar readouts (LFP/CSD-proxy, spectral) → optimization. 22 figures. | v0.3.3+ |
| **[Suite 2](07_jaxfne_suite_no_2_spectrolaminar_motif.md)** | Corticospectrolaminar Motif | Cortical column anatomy → JAX-first population sim → multimodal proxies (MUA/LFP/CSD/EEG/MEG/EMM) → spectrolaminar visualization → CPU-safe tuning. 13 figures. | v0.3.4+ |
| **[Suite 2 (Evoked L4)](08_jaxfne_suite_no_2_evoked_l4_drive.md)** | Evoked L4 Drive | Compact evoked-response variant of Suite 2's motif, baseline-vs-driven L4 layer contrast. | v0.3.4+ |
| **[Suite 3](08_jaxfne_suite_no_3_low_frequency_scaling.md)** | Low-Frequency Scaling | Scale-dependent proxy configs → population sims across sizes → PSD/bandpower → synchrony/Fano diagnostics → strict validation export. 5 figures. | v0.3.9+ |

[![Open Suite 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb)
[![Open Suite 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)
[![Open Suite 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Versioned tutorials

The numbered single-topic progression, teaching the source-to-field/readout
workflow from single-neuron models to multi-area laminar circuits. Grouped by
level.

**Beginner**

| # | Topic | Focus | Version |
|---|-------|-------|---------|
| [**01**](01_single_neuron_multimodal.md) | Single-neuron Multimodal | Izhikevich emitter, spikes, voltage, field readouts | v0.2.8+ |
| [**02**](02_two_neuron_ei.md) | Two-neuron E/I | Coupling, recurrent dynamics | v0.2.9+ |

**Intermediate**

| # | Topic | Focus | Version |
|---|-------|-------|---------|
| [**03**](03_network_100_ei.md) | 100-neuron Network | Population dynamics, stability | v0.2.10+ |
| [**04**](04_v1_column.md) | V1 Six-layer Column | Laminar anatomy, depth-specific readouts | v0.2.11+ |
| [**06**](06_v036_100_neuron_ei_population.md) | Chainable Configuration | New fluent `Configuration` method-chaining API | v0.3.6+ |
| [**07**](07_v037_source_bookkeeping.md) | Source Bookkeeping | 3D visualization of source/field/probe workflow | v0.3.7+ |
| [**08**](08_v038_lfp_csd_readout.md) | LFP/CSD Readout | Laminar contact projection, Gaussian kernels, CSD-proxy | v0.3.8+ |
| [**09**](09_v0310_eeg_meg_emm_proxy_bundle.md) | EEG/MEG/EMM Proxy Bundle | Separate sensor pathways: scalp potential, magnetic field, metabolic proxy | v0.3.10+ |
| [**10**](10_v0313_omission_oddball.md) | Sensory Omission & Oddball | Expected stimuli, unexpected deviants, sensory omissions | v0.3.13+ |

**Advanced**

| # | Topic | Focus | Version |
|---|-------|-------|---------|
| [**05**](05_v1_pfc_dual_column.md) | V1-PFC Dual Column | Cross-area interaction, traveling waves, continuous AAAB local-oddball adaptation with real trial-to-trial HDP carryover (100 chained trials, verified stable) | v0.2.14+ |
| [**11**](11_multi_laminar_cortical_agsdr.md) | Multi-area Laminar Model | Per-event `target_indices` L4-E-only targeting, `general_sequential_oddball_paradigm` backbone | v0.4.0+ |
| [**12**](12_izhikevich_single_emitter_explorer.md) | TFNE-Izhikevich Explorer | Single-emitter parameter exploration | — |
| [**13**](13_canonical_column_etude.md) | Canonical Cortical Column | The canonical 1000-neuron laminar column reference | 0.4.7+ |

**NeuronalTensor (tensor-first circuits)** — [`jaxfne_neuronal_tensor_first.ipynb`](https://github.com/HNXJ/jaxfne/blob/main/tutorials/jaxfne_neuronal_tensor_first.ipynb)
(script version: [`08_neuronal_tensor_first.py`](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py)).
`NeuronalTensor` is a second, declarative `Areas x Layers x NeuronTypes` circuit
definition that JSON round-trips and converges on the same `Model`/`Signals` as
the `Configuration` path used above. Carries the canonical 1000-neuron V1
column and is the path for **HDP homeostatic plasticity** — see the
[HDP guide](../guides/hdp.md) § "Tensor-first" and the
[API reference](../api/neuronal_tensor.md).

```python
tensor = jtfne.NeuronalTensor(areas=[...])
model  = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

---

## Étude notebooks

Research-style circuits under [`tutorials/etudes/`](https://github.com/HNXJ/jaxfne/tree/main/tutorials/etudes),
each a focused, runnable study rather than a step-by-step lesson. Grouped by
theme; all execute cleanly under CI (`tests/test_notebook_execution_suite.py`).

**Foundational**

| Étude | Topic |
|-------|-------|
| [No. 1 — Base](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_base.ipynb) | Computational-scaffold base circuit; tensor-first vs `Configuration` comparison |
| [No. 1 — Multi-laminar AGSDR](11_multi_laminar_cortical_agsdr.md) | Multi-area laminar model with AGSDR tuning (also a versioned tutorial, above) |

**Spectrolaminar family**

| Étude | Topic |
|-------|-------|
| [No. 2 — Spectrolaminar Power](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_2_spectrolaminar_power.ipynb) | TFNE-Izhikevich spectrolaminar motif, single-trial |
| [No. 3 — V1 Spectrolaminar 1k](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb) | 1000-neuron scalable spectrolaminar V1 column (see [showcases](../guides/showcases.md)) |
| [No. 7 — Multi-trial Spectrolaminar](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_7_multitrial_spectrolaminar.ipynb) | Multi-trial continuous simulation → spectrolaminar motif aggregation |

**Homeostasis & plasticity family**

| Étude | Topic |
|-------|-------|
| [No. 4 — Homeostatic V1 Column](04_v1_column.md) | Continuous pause/resume simulation with HDP homeostasis (also a versioned tutorial, above) |
| [No. 8 — Continuous Adaptation](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_8_continuous_adaptation.ipynb) | Continuous drive adaptation via HDP |

**Biophysical & multi-area family**

| Étude | Topic |
|-------|-------|
| [No. 5 — Enhanced Biophysical Column](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_5_enhanced_biophysical_column.ipynb) | Enhanced biophysical cortical column |
| [No. 6 — Multi-Area Network](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_6_multi_area_network.ipynb) | Connecting multiple cortical columns |

**Oddball / omission paradigm family**

| Étude | Topic |
|-------|-------|
| [No. 9 — Local Oddball](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_9_local_oddball.ipynb) | Simple local oddball task |
| [No. 10 — Global/Local Oddball](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_10_global_local_oddball.ipynb) | Combined global/local oddball task |
| [No. 11 — Local Omission](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_11_omission_local.ipynb) | Local omission task |
| [No. 12 — Continuous Omission (COOP)](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_12_omission_global_coop.ipynb) | Continuous omission oddball paradigm (COOP) |

**Thalamocortical**

| Étude | Topic |
|-------|-------|
| [TCM V1 6-Population](https://github.com/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_tcm_v1_6pop.ipynb) | Thalamocortical model, 6-population cortical column |

---

## Running tutorials

Tutorials live under `tutorials/` as Jupyter notebooks (étude notebooks under
`tutorials/etudes/`). Headless CI runners live under `examples/` (e.g.
`v031_*`, `v033_*`).

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
