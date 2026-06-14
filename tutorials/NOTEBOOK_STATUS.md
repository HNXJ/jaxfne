# Notebook Status and Classification

**Total notebooks: 24** (release-facing: 15, archived: 9)

## Release-facing notebooks (15)

Release-facing notebooks are part of the standard tutorial suite and subject to strict grammar, API, and scope validation.

- **jaxfne-sanity-checker-notebook-01.ipynb** — Core validation test; demonstrates API correctness and output stability
- **jaxfne_etude_no_1_base.ipynb** — Pedagogical etude; multi-laminar scaffold with AGSDR tuning workflow
- **jaxfne_suite_no_1_computational_biophysics.ipynb** — Suite notebook; Jaxley integration and biophysics scope
- **jaxfne_suite_no_2_evoked_l4_drive.ipynb** — Suite notebook; evoked response paradigm and readout
- **jaxfne_suite_no_2_spectrolaminar_motif.ipynb** — Suite notebook; spectrolaminar motif analysis
- **jaxfne_suite_no_3_low_frequency_scaling.ipynb** — Suite notebook; frequency domain analysis and power scaling
- **jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb** — Suite notebook; oscillatory dynamics in laminar circuit
- **jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb** — Version tutorial; proxy readout bundle (EEG/MEG/EMM alternatives)
- **jaxfne_v0313_omission_oddball.ipynb** — Version tutorial; oddball/omission paradigm for novelty detection
- **jaxfne_v031_single_neuron.ipynb** — Version tutorial; single-unit warmup and emitter validation
- **jaxfne_v032_parameter_sweep.ipynb** — Version tutorial; parameter sensitivity and gain/drive sweeps
- **jaxfne_v033_two_neuron_ei.ipynb** — Version tutorial; E-I pair dynamics and balance
- **jaxfne_v035_small_recurrent_ei.ipynb** — Version tutorial; recurrent E-I population dynamics
- **jaxfne_v036_100_neuron_ei_population.ipynb** — Version tutorial; scaled E-I population readouts
- **jaxfne_v038_lfp_csd_readout.ipynb** — Version tutorial; LFP/CSD proxy readout mechanics

## Archived notebooks (9)

Archived notebooks are excluded from strict validation. They may use legacy patterns, experimental APIs, or non-standard scope.

- **jaxfne_etude_no_2_spectrolaminar_power.ipynb** — Pedagogical etude with local simulation helpers; requires API extraction before release-facing status
- **jaxfne-sanity-delta-test-hierarchical-global-local-oddball.ipynb** — Experimental delta-test for hierarchical oddball; not part of release suite
- **jaxfne-v0333-colab-evidence.ipynb** — Colab prototype; notebook evidence collection (pre-release exploration)
- **jaxfne-v0333-colab-orientation.ipynb** — Colab prototype; notebook orientation task (pre-release exploration)
- **jaxfne-v0333-colab-task-smoke.ipynb** — Colab prototype; notebook task smoke test (pre-release exploration)
- **jaxfne-v034-stdp-ltp-ltd-adaptation.ipynb** — Experimental plasticity notebook; STDP/LTP/LTD mechanisms (pre-release, biophysics incomplete)
- **jaxfne-v035-coop-paradigm-stability.ipynb** — Experimental cooperative/competition paradigm (pre-release, scope unstable)
- **jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb** — Legacy etude; replaced by jaxfne_etude_no_1_base.ipynb with cleaner workflow
- **tutorial_madelane_2026_jaxfne_spectrolaminar.ipynb** — Madelane research notebook; custom analysis beyond standard scope

## Validation status

- Release-facing: strict grammar, export API, scientific scope, tensor-field consistency
- Archived: documented exceptions, not subject to strict rules
