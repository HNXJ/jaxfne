# Predictive Coding Light (N'dri et al. 2025): P1 source record

Status: draft from a Muse read of README.md, PORT_SPEC.md, CALIBRATION.md, prereg/P3_fig3.md, prereg/P3_fig3_v5.md and port/pcl_event_ref.py docstring, plus the Nat Commun 16:8880 full text via nature.com, 2026-10-18; unverified by the dispatcher.

All items labeled **observed** (with source) or **inferred**. Values neither the lane records nor the passages read give are **unknown**. No passage quoted over 15 words. Port outcomes cited only as pointers (README.md gate sections, CALIBRATION.md, prereg result files).

Target: N'dri, Barbier, Teulière, Triesch, "Predictive Coding Light", Nat Commun 16:8880 (2025), doi:10.1038/s41467-025-64234-z (observed README.md:3-6). Read 2026-10-18 from the publisher full text (page marked open access). The record covers the lane's two dashboard figures, Fig. 2 and Fig. 3.

## 1. Network (paper Methods; PORT_SPEC.md:8-17)

- Input: DVS events (t in µs, x, y, polarity), 346x260, one camera; HDF5 group `events` (observed PORT_SPEC.md:12).
- Simple layer: 9x9 positions x 64 depths per patch; receptive field 10x10x2 (polarity); overlap 3 (stride 7); patches x {150}, y {100}: one patch (observed PORT_SPEC.md:13).
- Complex layer: 6x6 x 32; receptive field 4x4x64 simple cells; overlap 3 (observed PORT_SPEC.md:14).
- Weight sharing: one simple weight tensor per (patch, depth); the local-inhibition vector shared per depth (observed PORT_SPEC.md:15, citing SpikingNetwork.cpp:188).
- Inhibition: local (other depths at same x, y; plastic, shared), lateral dynamic (range 4 in x and y, all depths, plastic per pair), top-down (complex to the simple cells they cover, plastic per pair). The default config enables local only (DefaultConfig.cpp:15); PCL's lateral and top-down inhibition come from each trained network's own config (observed PORT_SPEC.md:16).
- Propagation: event-driven and depth-first: a spiking simple cell updates its weights, then sends local inhibition, then lateral inhibition (only when activation is on), then excites complex cells, which can spike and send top-down inhibition, all at the same timestamp (observed PORT_SPEC.md:17, citing SpikingNetwork.cpp:41-179).
- A unit can excite some targets and inhibit others; the network violates Dale's law (observed article Discussion).

## 2. Neuron (paper Methods Eqs. 2-3; PORT_SPEC.md:19-29)

On each input of weight w at time t: `V <- V·exp(-Δt/τm)`, `A <- A·exp(-Δt/τSRA)`, `V <- V + w - ηRP·exp(-(t-ts)/τRP) - A`, clamped at -20; spike if `V > θ`, then `V <- Vreset` and `A <- A + ΔSRA`. Inhibitory inputs subtract w with the refractory and adaptation terms. Complex cells have no A term (observed PORT_SPEC.md:21-25).

Parameters (observed PORT_SPEC.md:27-37): θ 30 (default config) vs 10 (paper Table 2) simple, 3 complex; Vreset -10 both; τm, τRP, τSRA 18, 5, 100 ms simple and 50, 5, none complex; τLTP, τLTD 7, 7 ms simple and 40, 40 ms complex; ηRP 10 both; excitatory L1 NORM_FACTOR 50 simple / 1000 complex; local-inhibition ETA_INH 1500 / 600; lateral / top-down norms 6500 / 2000; ηLTP, ηLTD, ηILTP, ηILTD 0.00077, -0.00021, 0.00046, -0.00046 simple and 0.002, 0.002, 0.008, -0.008 complex. Thresholds differ between the default config and Table 2, so each trained network's own `configs/*.json` overrides decide (observed PORT_SPEC.md:63-64). Simple adaptation increment defaults to 0 (observed CALIBRATION.md:38).

## 3. Plasticity (paper Methods Eqs. 4-9; PORT_SPEC.md:39-54)

Causal STDP on every connection, applied at each postsynaptic spike over the events stored since the last update; soft-bound potentiation/depression with L1 renormalization of incoming weights per connection type to λ (observed README.md:16-21 and the article Methods). Code-level readings from the C++, read as data (observed PORT_SPEC.md:42-54): the excitatory update pairs LTP against LTD with the previous post spike, clipped at w >= 0; effective rates are scaled (ηLTP·NORM_FACTOR, ηILTP·ETA_INH); the soft-bound constants (3, 0.33, 50, 0.02 simple; 4, 0.25 excitatory and 25, 0.04 inhibitory complex) are hard-coded and absent from the paper equation; the complex-cell excitatory rule is counter-gated and pairs the last two spikes; the default config's complex ηLTD is +0.002, so its LTD branch potentiates. The code is authoritative for replication; each Methods-vs-code difference is a declared deviation (observed PORT_SPEC.md:54).

## 4. Procedure per targeted figure

- Fig. 2 (observed article Results and README.md:25-28): 4 neurons; neuron 0 driven by a Poisson train; neurons 1-3 copy 90 / 50 / 10 % of its spikes with 1 ms delay. Neuron 0 grows inhibitory synapses onto neurons 1-3 from equal initial values under the causal iSTDP rule with L1 competition. Training: 140 one-second samples x 10 epochs; test: 35 samples; excitatory weights fixed at 15 mV, threshold 10 mV.
- Fig. 3 (observed article Results; P3_fig3.md:23-36; P3_fig3_v5.md:19-22): unsupervised training on natural-image event streams; probing with sinusoidal counterphase gratings. Lane stimuli: `test7`, 8 orientations x 18 spatial-frequency indices x 11 phase indices x 2 trials; level 2 of the axes is spatial frequency, level 3 is phase, phase 10 repeats phase 0 so phases 0-9 are used. Each stimulus spans about 1.82 s of events. Responses average over the 64 simple cells and 32 complex cells with 2 trials; every numeric criterion is the port's, the paper states its claims qualitatively (observed P3_fig3.md:10-21).

## 5. Claims per figure the lane targets

- Fig. 2 (observed article Results): the iSTDP rule grows inhibitory synapses stronger with higher predictability of the postsynaptic spikes, and the resulting inhibition suppresses the predictable spikes. Lane acceptance: learned weights and suppression both ordered neuron 1 > 2 > 3 in every network (observed README.md:23-29). Paper Fig. 2d values were unavailable to the lane, so there is no numeric comparison (observed README.md:42-43).
- Fig. 3 (observed article Results): after training, simple-cell fields fit Gabor functions and complex cells pool same-orientation/frequency simple cells with variable phase/polarity; both sets tune for orientation and spatial frequency; only simple cells tune for spatial phase (complex curve virtually flat); complex cells frequency-double relative to simple cells under counterphase gratings. Lane prereg C1-C3 restates this qualitatively (observed P3_fig3.md:10-21); the reference curves for the v5 spatial-frequency test are digitized from the publisher full-size PNG (observed P3_fig3_v5.md:24-35).
- Figs. 4-6 exist and are outside the lane's current scope: surround, orientation-tuned and cross-orientation suppression (Fig. 4); gesture and N-MNIST classification with random-removal controls (Fig. 5); population sparseness (Fig. 6) (observed article Results; sequencing inferred from PORT_SPEC.md:75-76 and the dashboard open items).

## 6. Data and code availability with licence

- Paper: open access (observed nature.com article page).
- Reference code: github.com/comsee-research/Predictive-Coding-Light, C++ event-driven (observed README.md:4-6, PORT_SPEC.md:5-6). Read as text only; the transcription lists its source files with line ranges (observed port/pcl_event_ref.py:1-13).
- Archived code: Zenodo 10.5281/zenodo.16836801, CC-BY 4.0 (observed PORT_SPEC.md:3-4).
- Trained networks and data: Zenodo 10.5281/zenodo.16813454, local copy `F:\warehouse\pcl\zenodo` (observed PORT_SPEC.md:4-5). That archive's licence is **unknown** here.
- Fig. 3 source image: CC BY-NC-ND 4.0, kept outside the repo (`F:\warehouse\pcl\paper\`), SHA-256 checked per run (observed P3_fig3_v5.md:26-28).
- Natural-image training set identity and licence: **unknown** (the lane records read name only "natural images").
- `test7` stimulus provenance: **unknown** (only its axes were measured from the events).
- Figs. 5-6 datasets (DVS128 Gesture, N-MNIST) are named in the article (observed article Results); their access and licences are **unknown** here.
- The authors' tuning-curve data archive is not opened, by decision (observed dashboard/sources.json, pcl-fig3 open items).

## 7. Gap list

- G1: Lateral and top-down maps load in libstdcxx-emulated unordered-map order; the files do not record the true iteration order, so the order is a declared assumption with a sensitivity run (observed port/pcl_event_ref.py:19-22; P3_fig3.md:28-29).
- G2: The analysed simple column (4,4) rests on the convention that `neuron_id = 2560` in the authors' `tuningCurves.ipynb` means column 40 = (4,4) under column-major ids (observed P3_fig3.md:40-43).
- G3: The complex set (columns (2,2), (2,3), (3,2), (3,3), 128 cells) follows a geometric tie rule, while the paper averages over 32 complex cells (observed P3_fig3.md:45-51).
- G4: The v5 curve test assumes the notebook's 5 phases are test7's even phases (A1), period proportional to 20 - i (A2), and the v4 response window (A3), each with a sensitivity run (observed P3_fig3_v5.md:52-58).
- G5: The two trials are pooled, never used as independent folds (their column-40 count correlation is 0.95) (observed P3_fig3.md:34-35).
- G6: Simple threshold 10 mV (paper Table 2) vs 30 mV (default config); per-network configs decide (observed PORT_SPEC.md:29, 63-64).
- G7: Default complex ηLTD +0.002 potentiates; kept as the code does (observed PORT_SPEC.md:52-53).
- G8: "static" in layerInhibitions is read as "local" (declared alias); the C++ checks only "local" (observed port/pcl_event_ref.py:23-25).
- G9: Sums use numpy (pairwise), so weights agree with the C++ to about 1e-15 relative, not bitwise (observed port/pcl_event_ref.py:31).
- G10: Initial training weights are unreproducible bit for bit (C++ `std::random_device`) (observed port/pcl_event_ref.py:16-17).
- G11: The paper's Fig. 2d numeric values were unavailable; Fig. 2 has no numeric comparison (observed README.md:42-43).
- Port-result pointers only: Gate P0/K0/K0b and C1/C1b outcomes (README.md); core-gap closures and estimand table (CALIBRATION.md); v4/v5 scoring (prereg/P3_fig3_result.md, prereg/P3_fig3_v5_result.md).
