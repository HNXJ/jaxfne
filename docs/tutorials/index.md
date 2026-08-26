# Tutorials

Tutorials teach **how to use the jaxfne grammar**: progressively detailed
examples organized by mathematical purpose (circuit construction, source/field
readout, paradigms, optimization). For frozen scientific demonstrations, see
**[Études](../etudes/index.md)**.

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

| Suite | Topic | Focus |
|-------|-------|-------|
| **[Suite 1](06_jaxfne_suite_no_1_computational_biophysics.md)** | Computational Biophysics | Single-neuron models → vectorized circuits → laminar readouts (LFP/CSD-proxy, spectral) → optimization |
| **[Suite 2](07_jaxfne_suite_no_2_spectrolaminar_motif.md)** | Corticospectrolaminar Motif | Cortical column anatomy → population simulation → multimodal proxies → spectrolaminar visualization |
| **[Suite 2 (Evoked L4)](08_jaxfne_suite_no_2_evoked_l4_drive.md)** | Evoked L4 Drive | Baseline-vs-driven L4 layer contrast within the spectrolaminar motif |
| **[Suite 3](08_jaxfne_suite_no_3_low_frequency_scaling.md)** | Low-Frequency Scaling | Scale-dependent proxy configs → population PSD/bandpower → synchrony/Fano diagnostics |

[![Open Suite 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb)
[![Open Suite 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)
[![Open Suite 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Single-topic progression

Progressive lessons on the source-to-field/readout workflow, from single-neuron
models to multi-area laminar circuits.

**Beginner**

| # | Topic | Focus |
|---|-------|-------|
| [**01**](01_single_neuron_multimodal.md) | Single-neuron Multimodal | Izhikevich emitter, spikes, voltage, field readouts |
| [**02**](02_two_neuron_ei.md) | Two-neuron E/I | Coupling, recurrent dynamics |

**Intermediate**

| # | Topic | Focus |
|---|-------|-------|
| [**03**](03_network_100_ei.md) | 100-neuron Network | Population dynamics, stability |
| [**04**](04_v1_column.md) | V1 Six-layer Column | Laminar anatomy, depth-specific readouts |
| [**06**](06_v036_100_neuron_ei_population.md) | Chainable Configuration | Fluent `Configuration` method-chaining API |
| [**07**](07_v037_source_bookkeeping.md) | Source Bookkeeping | Source/field/probe workflow and metadata |
| [**08**](08_v038_lfp_csd_readout.md) | LFP/CSD Readout | Laminar contact projection, Gaussian kernels, CSD-proxy |
| [**09**](09_v0310_eeg_meg_emm_proxy_bundle.md) | EEG/MEG/EMM Proxy Bundle | Scalp potential, magnetic field, metabolic proxy pathways |
| [**10**](10_v0313_omission_oddball.md) | Sensory Omission & Oddball | Expected stimuli, deviants, sensory omissions |

**Advanced**

| # | Topic | Focus |
|---|-------|-------|
| [**05**](05_v1_pfc_dual_column.md) | V1-PFC Dual Column | Cross-area interaction, trial-chained HDP carryover |
| [**11**](11_multi_laminar_cortical_agsdr.md) | Multi-area Laminar Model | Per-event targeting, sequential oddball paradigm backbone |
| [**12**](12_izhikevich_single_emitter_explorer.md) | TFNE-Izhikevich Explorer | Single-emitter parameter exploration |
| [**13**](13_canonical_column_etude.md) | Canonical Cortical Column | Canonical 1000-neuron laminar column reference |

**NeuronalTensor (tensor-first circuits)** — [`jaxfne_neuronal_tensor_first.ipynb`](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb)
(script version: [`08_neuronal_tensor_first.py`](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py)).
`NeuronalTensor` is a second, declarative `Areas x Layers x NeuronTypes` circuit
definition that JSON round-trips and converges on the same `Model`/`Signals` as
the `Configuration` path used above. Carries the canonical 1000-neuron V1
column and is the path for **H-state / HDP adaptation** — see the
[H-state / HDP guide](../guides/hdp.md) and the
[API reference](../api/neuronal_tensor.md).

```python
tensor = jtfne.NeuronalTensor(areas=[...])
model  = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

---

## Étude notebooks

Notebook études under
[`tutorials/etudes/`](https://github.com/HNXJ/jaxfne/tree/main/tutorials/etudes)
remain runnable from the repository. The primary **documented étude** with
committed metrics is indexed under **[Études](../etudes/index.md)** (HDP
controllability / reachability). The table below lists legacy notebook études
by theme.

**Foundational**

| Étude | Topic |
|-------|-------|
| [No. 1 — Base](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_1_base.ipynb) | Computational-scaffold base circuit; tensor-first vs `Configuration` comparison |
| [No. 1 — Multi-laminar AGSDR](11_multi_laminar_cortical_agsdr.md) | Multi-area laminar model with AGSDR tuning (also a versioned tutorial, above) |

**Spectrolaminar family**

| Étude | Topic |
|-------|-------|
| [No. 2 — Spectrolaminar Power](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_2_spectrolaminar_power.ipynb) | TFNE-Izhikevich spectrolaminar motif, single-trial |
| [No. 3 — V1 Spectrolaminar 1k](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb) | 1000-neuron scalable spectrolaminar V1 column (see showcases (`docs/guides/showcases.md` — repository-internal reference, excluded from the built site)) |
| [No. 7 — Multi-trial Spectrolaminar](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_7_multitrial_spectrolaminar.ipynb) | Multi-trial continuous simulation → spectrolaminar motif aggregation |

**Homeostasis & plasticity family**

| Étude | Topic |
|-------|-------|
| [No. 4 — Homeostatic V1 Column](04_v1_column.md) | Continuous pause/resume simulation with HDP homeostasis (also a versioned tutorial, above) |
| [No. 8 — Continuous Adaptation](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_8_continuous_adaptation.ipynb) | Continuous drive adaptation via HDP |

**Biophysical & multi-area family**

| Étude | Topic |
|-------|-------|
| [No. 5 — Enhanced Biophysical Column](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_5_enhanced_biophysical_column.ipynb) | Enhanced biophysical cortical column |
| [No. 6 — Multi-Area Network](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_6_multi_area_network.ipynb) | Connecting multiple cortical columns |

**Oddball / omission paradigm family**

| Étude | Topic |
|-------|-------|
| [No. 9 — Local Oddball](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_9_local_oddball.ipynb) | Simple local oddball task |
| [No. 10 — Global/Local Oddball](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_10_global_local_oddball.ipynb) | Combined global/local oddball task |
| [No. 11 — Local Omission](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_11_omission_local.ipynb) | Local omission task |
| [No. 12 — Continuous Omission (COOP)](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_12_omission_global_coop.ipynb) | Continuous omission oddball paradigm (COOP) |

**Thalamocortical**

| Étude | Topic |
|-------|-------|
| [TCM V1 6-Population](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_tcm_v1_6pop.ipynb) | Thalamocortical model, 6-population cortical column |

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
