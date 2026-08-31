# Notebook Status and Classification

**Total notebooks: 40** (release-facing: 29, archived: 10, template: 1)

## Release-facing notebooks (29)

Release-facing notebooks are part of the standard tutorial suite and subject to strict grammar, API, and scope validation.

- **jaxfne-sanity-checker-notebook-01.ipynb** — Core validation test; demonstrates API correctness and output stability
- **jaxfne_etude_no_1_base.ipynb** — Pedagogical etude; multi-laminar scaffold with AGSDR tuning workflow
- **jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb** — Pedagogical etude; 1000-neuron spectrolaminar V1 cortex column (has a dedicated artifact-contract test: `tests/test_etude3_v1_spectrolaminar_1k.py`)
- **jaxfne_etude_no_4_homeostatic_V1_column.ipynb** — Pedagogical etude; homeostatic V1 column with continuous pause/resume simulation
- **jaxfne_etude_no_5_enhanced_biophysical_column.ipynb** — Pedagogical etude; enhanced biophysical cortical column
- **jaxfne_etude_no_6_multi_area_network.ipynb** — Pedagogical etude; multi-area network, connecting columns
- **jaxfne_etude_no_7_multitrial_spectrolaminar.ipynb** — Pedagogical etude; multi-trial continuous simulation to spectrolaminar motif
- **jaxfne_etude_no_8_continuous_adaptation.ipynb** — Pedagogical etude; multi-trial continuous adaptation (homeostasis + adaptive synapses)
- **jaxfne_etude_no_9_local_oddball.ipynb** — Pedagogical etude; simple local oddball task
- **jaxfne_etude_no_10_global_local_oddball.ipynb** — Pedagogical etude; global-local oddball task
- **jaxfne_etude_no_11_omission_local.ipynb** — Pedagogical etude; local omission paradigm
- **jaxfne_etude_no_12_omission_global_coop.ipynb** — Pedagogical etude; global omission paradigm (COOP)
- **jaxfne_etude_tcm_v1_6pop.ipynb** — Pedagogical etude; TCM V1 6-population cortical column
- **jaxfne_suite_no_1_computational_biophysics.ipynb** — Suite notebook; Jaxley integration and biophysics scope (executed in CI: `tests/test_suite_no1_notebook_execution.py`)
- **jaxfne_suite_no_2_evoked_l4_drive.ipynb** — Suite notebook; evoked response paradigm and readout
- **jaxfne_suite_no_2_spectrolaminar_motif.ipynb** — Suite notebook; spectrolaminar motif analysis
- **jaxfne_suite_no_3_low_frequency_scaling.ipynb** — Suite notebook; frequency domain analysis and power scaling
- **jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb** — Suite notebook; oscillatory dynamics in laminar circuit (executed in CI: `tests/test_suite_no4_notebook_execution.py`)
- **jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb** — Version tutorial; proxy readout bundle (EEG/MEG/EMM alternatives)
- **jaxfne_v0313_omission_oddball.ipynb** — Version tutorial; oddball/omission paradigm for novelty detection
- **jaxfne_v031_single_neuron.ipynb** — Version tutorial; single-unit warmup and emitter validation
- **jaxfne_v032_parameter_sweep.ipynb** — Version tutorial; parameter sensitivity and gain/drive sweeps
- **jaxfne_v033_two_neuron_ei.ipynb** — Version tutorial; E-I pair dynamics and balance
- **jaxfne_v035_small_recurrent_ei.ipynb** — Version tutorial; recurrent E-I population dynamics
- **jaxfne_v036_100_neuron_ei_population.ipynb** — Version tutorial; scaled E-I population readouts
- **jaxfne_v038_lfp_csd_readout.ipynb** — Version tutorial; LFP/CSD proxy readout mechanics
- **jaxfne_v040_continuous_omission_oddball.ipynb** — Version tutorial; v0.4.0 continuous sequential omission oddball paradigm
- **jaxfne_v040_homeostasis_plasticity_dc_noise_sweep.ipynb** — Version tutorial; v0.4.0 homeostasis/plasticity DC-noise sweep
- **jaxfne_neuronal_tensor_first.ipynb** — Version tutorial; 0.4.7 NeuronalTensor-first circuit definition (Areas/Layers/NeuronTypes), JSON round-trip, HDP homeostatic plasticity via explicit `RuntimeConfig` override; executed with `nbclient` 2026-06-25 (outputs are real captured run results, not placeholders)

## Archived notebooks (10)

Archived notebooks are excluded from strict validation. They may use legacy patterns, experimental APIs, or non-standard scope.

- **jaxfne_mechanism_01_relative_state_X_H_X.ipynb** — Mechanism tutorial notebook (relative state dynamics)
- **jaxfne_mechanism_02_rbd_memory_Xt_Ht1.ipynb** — Mechanism tutorial notebook (RBD memory dynamics)
- **jaxfne_mechanism_03_hdp_H_W.ipynb** — Mechanism tutorial notebook (HDP weight dynamics)
- **jaxfne_etude_no_2_spectrolaminar_power.ipynb** — Pedagogical etude with local simulation helpers; requires API extraction before release-facing status
- **jaxfne-sanity-delta-test-hierarchical-global-local-oddball.ipynb** — Experimental delta-test for hierarchical oddball; not part of release suite
- **jaxfne-v034-stdp-ltp-ltd-adaptation.ipynb** — Experimental plasticity notebook; STDP/LTP/LTD mechanisms (pre-release, biophysics incomplete)
- **jaxfne-v035-coop-paradigm-stability.ipynb** — Experimental cooperative/competition paradigm (pre-release, scope unstable)
- **jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb** — Legacy etude; replaced by jaxfne_etude_no_1_base.ipynb with cleaner workflow
- **tutorial_madelane_2026_jaxfne_spectrolaminar.ipynb** — Madelane research notebook; custom analysis beyond standard scope
- **jaxfne_colab_gpu_tpu_100k_column.ipynb** (added 2026-07-14) — Manual-run Colab notebook for real GPU/TPU timing on a 100k-neuron column (`plans.json` item `colab-notebook-gpu-tpu-100k-column`); requires real GPU/TPU hardware this repo's CI doesn't have, so it's intentionally excluded from `nbclient`-based execution coverage — run it yourself in Colab, not via `test_notebook_execution_suite.py`

## Template (1)

- **templates/jaxfne_notebook_template.ipynb** — Authoring scaffold for new tutorials/etudes (setup/grammar/Colab-badge boilerplate); not itself a tutorial, excluded from both release-facing and archived validation.

## Validation status

- Release-facing: strict grammar, export API, scientific scope, tensor-field consistency
- Archived: documented exceptions, not subject to strict rules
- All 29 release-facing notebooks have real `nbclient`-based execution coverage (verified 2026-06-25, corrects an earlier wrong claim in this doc): Suite No. 1 and Suite No. 4 each have a dedicated test file; `jaxfne_neuronal_tensor_first.ipynb` has `tests/test_neuronal_tensor_notebook_execution.py` (added 2026-06-25); the remaining 26 are parametrized in `tests/test_notebook_execution_suite.py` (5 of those 26 -- Étude 8-12 -- are intentional skeleton placeholders and correctly `xfail`). All of the above are gated `@pytest.mark.slow`/`@pytest.mark.notebook` and run via `.github/workflows/notebook_execution.yml` (nightly 03:00 UTC + manual dispatch), not the push-triggered fast lane.
