# Tutorials

Tutorials teach **how to use the jaxfne grammar** as a cumulative arc that
reuses the same model from definition to comparison. Each step carries the same
variables forward (`genome → tensor → model → signals → H → dynamics → compare`), endpoint is the
canonical **1000-neuron PseudoGenome** `canonical-v1-column-1000n`. For frozen scientific
demonstrations, see **[Études](../etudes/index.md)**.

> **Canonical invariant (endpoint 1000n, Δscience=0):** every step below starts from
> `load_canonical_pseudogenome("canonical-v1-column-1000n")` and reuses the prior step's object.
> Smoke mode (`SMOKE=1`, `n≈100`, `duration_ms≈100`) runs in ~30 s; the documented main path is 1000n
> (`duration_ms=1000.0`, `dt_ms=0.5`). `PseudoGenome --develop(K_D)--> NeuronalTensor --construct(K_S)--> Model --simulate(K_S)--> Signals`
> with `K_D ≠ K_S ≠ K_A` (see [PseudoGenome guide](../guides/jdna.md)). No new API.

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

## Cumulative (canonical 1000n) progression — define → develop → inspect → simulate → observe → add state → add dynamics → compare

Each row reuses the same `genome → tensor → model → signals` chain. Endpoint is the
frozen `canonical-v1-column-1000n` PseudoGenome (6 layers 100/250/200/100/200/150,
48 intra-area rules, `fraction_jitter_sigma=0.01`); see `artifacts/tutorial_cumulative_audit.md`
(full audit, this repo) for the table and receipts.

| # | Verb | Topic | Focus | Reuses |
|---|------|-------|-------|--------|
| [**01**](01_define_genome.md) | define | PseudoGenome (generative rules) | Load `canonical-v1-column-1000n`, `validate_genome`, `genome_rules_hash`, `declared_constraints`, tolerance bands | — (origin) |
| [**02**](02_develop_genome.md) | develop | Genome → Phenotype | `develop(G,K_D)` determinism, `seed=0` vs `1` jitter within bands, `phenotype_sha256`, provenance | `genome` from 01 |
| [**03**](03_inspect_tensor.md) | inspect | Realized vs configured | `neuron_table()`, per-layer counts, 48 rules, configured vs realized vs effective, `save/load_neuronal_tensor` round-trip | `tensor` from 02 |
| [**04**](04_simulate_tensor.md) | simulate | Construct → Signals | `construct(tensor, RuntimeConfiguration)` → `simulate` → `Signals`; `EdgeList.n_edges≈215k` (p=1.0 bipartite), positions after `Pose3D` | `tensor` from 02 |
| [**05**](05_observe_fields.md) | observe | Field & probe operators | LFP/CSD/EEG/MEG/PSD post-hoc on *frozen* `X,Q`; authority `K_a≠K_b ⇒ Y_a≠Y_b`; LFP-not-CSD for crossover | `model,signals` from 04 |
| [**06**](06_add_state.md) | add state | H / RBS container | `PlasticParams.H` → `h_state`, `with_hdp_initial_state`, `checkpoint/restore` (inert until dynamics) | `tensor,model` from 04 |
| [**07**](07_add_dynamics.md) | add dynamics | HDP adaptation | `enable_hdp` (`DEFAULT_HDP` stable vs `DESYNC`), `H_trace/w_trace`, `K_HDP/K_ctrl/K_w_ctrl` | `model` from 06 |
| [**08**](08_compare_nulls.md) | compare | Nulls, lesions, authority | Shuffled, `LESION_SPEC`, multi-area 3000n via `merge_neuronal_tensors`, `kappa`, effective = ΔX | all prior |

> **One chain, one model.** The first cell of 02-08 is `# continued — genome/tensor/model/signals from previous step` and does not rebuild a different `n` or `Configuration`. The fluent `Configuration` builder remains first-class (see box below) but the cumulative main line uses `develop(G,K_D)`; both converge on `construct → simulate` per [Configuration Grammar](../guides/configuration_grammar.md).

**Archive (pre-cumulative, isolated tutorials) — still reachable, not the recommended path:**

| # | Topic | Focus | Relation to cumulative |
|---|-------|-------|------------------------|
| [**01**](01_single_neuron_multimodal.md) | Single-neuron Multimodal | Izhikevich emitter, spikes, voltage, field readouts | Folded into 03 as 5-line *n=1 vs 1000* contrast box |
| [**02**](02_two_neuron_ei.md) | Two-neuron E/I | Coupling, recurrent dynamics | Same — boxed contrast in 03 |
| [**03**](03_network_100_ei.md) | 100-neuron Network | Population dynamics, stability | Smoke preamble for 02-05 (`if SMOKE: n=100`) |
| [**04**](04_v1_column.md) | V1 Six-layer Column (600n) | Laminar anatomy, depth-specific readouts | Historical 600n example → redirect to 01 canonical 1000n |
| [**06**](06_v036_100_neuron_ei_population.md) | Chainable Configuration | Fluent `Configuration` method-chaining API | Box in 04: `Configuration` vs `NeuronalTensor` two on-ramps, one compiler |
| [**07**](07_v037_source_bookkeeping.md) | Source Bookkeeping | Source/field/probe workflow and metadata | Merged into 05 §1 |
| [**08**](08_v038_lfp_csd_readout.md) | LFP/CSD Readout | Laminar contact projection, Gaussian kernels, CSD-proxy | Merged into 05 §2 |
| [**09**](09_v0310_eeg_meg_emm_proxy_bundle.md) | EEG/MEG/EMM Proxy Bundle | Scalp potential, magnetic field, metabolic proxy pathways | Merged into 05 §3 |
| [**10**](10_v0313_omission_oddball.md) | Sensory Omission & Oddball | Expected stimuli, deviants, sensory omissions | Variant box in 08 (`omission_oddball_paradigm`) |
| [**05**](05_v1_pfc_dual_column.md) | V1-PFC Dual Column | Cross-area interaction, trial-chained HDP carryover | Multi-area variant in 08 (`merge_neuronal_tensors` / `canonical-v1-v4-pfc-multiarea`) |
| [**11**](11_multi_laminar_cortical_agsdr.md) | Multi-area Laminar Model | Per-event targeting, sequential oddball paradigm backbone | Tuning demo in 07 (AGSDR) + knock-out demo in 08 |
| [**12**](12_izhikevich_single_emitter_explorer.md) | TFNE-Izhikevich Explorer | Single-emitter parameter exploration | Prerequisite appendix (browser Euler ≠ JAX kernel) |
| [**13**](13_canonical_column_etude.md) | Canonical Cortical Column | Canonical 1000-neuron laminar column reference | **Rewritten** to cross-link to 01-08; code blocks now `load_canonical_pseudogenome → develop → construct` |

**NeuronalTensor (tensor-first circuits)** — [`jaxfne_neuronal_tensor_first.ipynb`](https://github.com/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb)
(script version: [`08_neuronal_tensor_first.py`](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py)).
`NeuronalTensor` is the phenotype data model: the declarative `Areas x Layers x
NeuronTypes` object that `develop(G,K_D)` returns and `construct` compiles. It
JSON round-trips and converges on the same `Model`/`Signals` as the `Configuration`
path, but is the object the cumulative arc carries (steps 02→04). It is also the
path for **H-state / HDP adaptation** — see the [H-state / HDP guide](../guides/hdp.md) and the
[API reference](../api/neuronal_tensor.md). The cumulative path uses `develop(G,K_D)`; `Configuration`
is the fluent builder for bespoke circuits — both converge on `construct → simulate`
per [Configuration Grammar](../guides/configuration_grammar.md).

```python
# cumulative main line (steps 01→04):
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)   # G --D(K_D)--> N
model  = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)          # same compiler regardless of on-ramp
```

---

## Études (frozen demonstrations — not tutorials)

Notebook études under [`tutorials/etudes/`](https://github.com/HNXJ/jaxfne/tree/main/tutorials/etudes)
remain runnable from the repository. The four **documented études** with committed
metrics/bundles are indexed under **[Études](../etudes/index.md)** (HDP controllability/reachability,
multiscale observation, Experiment A, heterogeneous emitters). Those are publication-grade
frozen protocols; they are **not** the tutorial progression.

Legacy notebook études (12 + TCM) remain under `artifacts/tutorials/etudes/` for
runnable archival use; see [Études](../etudes/index.md). The table formerly duplicated here has been
removed to avoid implying those notebooks are the tutorial path. For evidence reuse,
the observations in cumulative step 05 and comparison in step 08 are derived from the same
observation stack those études freeze (see [Multiscale observation](../etudes/multiscale_observation.md)).

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

## Quick example: Canonical 1000n primer (same chain as steps 01→04)

```python
import jaxfne as jtfne

genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")  # define (01)
tensor = jtfne.develop(genome, seed=0)                                   # develop (02)
# inspect (03): len(tensor.areas[0].layers)==6, sum(n_neurons)==1000, rules==48
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))  # simulate (04)
signals = jtfne.simulate(model)

print(f"neurons: {len(model.neuron_table())}")         # 1000
print(f"spike count: {int(signals.get('spikes').sum())}")
# observe (05): LFP/CSD on frozen signals; add state (06) / dynamics (07) / compare (08) carry this same model
```

> The former single-neuron primer (`suite2_four_celltype_config(n=1)`) is retained as a 5-line contrast box inside step 03; it is not the canonical model. For the fluent-builder on-ramp, see [Configuration Grammar](../guides/configuration_grammar.md).

## Next steps

After tutorials:

- **[Guides](../guides/index.md)** for how-to articles and workflow tips
- **[API reference](../api/index.md)** for full class/function documentation
- **[Jaxley interoperability](../guides/jaxley_interop.md)** for using external models
