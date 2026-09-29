# Literature reproduction études: P0 candidate table

Stage P0 of `literature_reproduction_plan.md`, 2026-09-29. **Unverified.**
Every row comes from a web-search result (title and link as returned); nothing
is read at the source yet. Authors, exact years, parameters and observables are
filled at P1 from the primary source. Screening marks are provisional guesses
from snippets and are corrected at P1. Criteria 1–6 are the plan's screening
list; "?" means the snippet does not say.

Searches run: laminar microcircuit + LFP/CSD; spectrolaminar motif;
predictive coding and omission with SST/VIP; extracellular field forward
models; homeostatic plasticity in microcircuits; reduced single-neuron
biophysics and LFP.

## Candidates

| # | Title (as returned) | Link | Type | Field / observable | HDP hook | Crit. met (1-6) | P0 call |
|---|---|---|---|---|---|---|---|
| 1 | A ubiquitous spectrolaminar motif of local field potential power across the primate cortex (Nat Neurosci, 2023) | [nature.com](https://www.nature.com/articles/s41593-023-01554-7) | data + hypothesis | laminar power: gamma superficial, alpha-beta deep, L4 crossing | slow deep-layer adaptation could shape the beta arm | 3 (1? 4?) | shortlist: the target observable, not a model |
| 2 | Is there a ubiquitous spectrolaminar motif of local field potential power across primate neocortex? (Nat Neurosci, 2025) | [nature.com](https://www.nature.com/articles/s41593-025-02167-y) | data re-analysis | same motif; counter-evidence across areas and species | same | 3 (1? 4?) | shortlist with #1: model must state where it agrees with each side |
| 3 | A cortical microcircuit model reveals distinct inhibitory mechanisms of network oscillations and stability (bioRxiv, 2025) | [biorxiv](https://www.biorxiv.org/content/10.1101/2025.02.23.639719.full.pdf) | model | E/PV/SOM network, self-generated gamma near 35 Hz; inhibitory roles in stability | inhibitory stability is an HDP question | 1, 3, 5, 6 | shortlist: closest match to jaxfne populations |
| 4 | Cortical networks with multiple interneuron types generate oscillatory patterns during predictive coding (bioRxiv 2024; PubMed 40929236) | [pubmed](https://pubmed.ncbi.nlm.nih.gov/40929236/) | model | PV/SST/VIP; oscillations in inference; cell-type inactivation | adaptation of prediction-error balance | 1, 3, 5, 6 | shortlist |
| 5 | Modelling Predictive Coding in V1: Layer 2/3 Circuits for Prediction Error Computation through Compartmentalized Spiking Neurons (bioRxiv, 2025) | [biorxiv](https://www.biorxiv.org/content/10.1101/2025.11.01.686040v1.full) | model | sign-specific prediction errors, omission, VIP ramp | learned prediction gating | 1, 3, 5 | likely GAP: two-compartment neurons; check emitters at P2 |
| 6 | Cell-type-specific encoding of prediction and reward in cortical microcircuits during novelty detection (bioRxiv, 2025) | [biorxiv](https://www.biorxiv.org/content/10.1101/2025.05.13.653877v2.full) | ? | cell-type responses to novelty and omission | ? | 3, 5 (1? 4?) | hold: model or data unknown |
| 7 | The laminar organization of cell types in macaque cortex and its relationship to neuronal oscillations (bioRxiv, 2024) | [biorxiv](https://www.biorxiv.org/content/10.1101/2024.03.27.587084.full.pdf) | data | cell-type depth profile vs oscillation bands | none obvious | 2, 5 | supports #1-2 parameters (layer composition); not a target |
| 8 | Resolving the mesoscopic missing link: Biophysical modeling of EEG from cortical columns in primates (NeuroImage, 2022) | [pubmed](https://pubmed.ncbi.nlm.nih.gov/36031184) | model | layer-specific currents to dipole and ERP components | none obvious | 1, 2, 3 (5?) | hold: needs a solved field; jaxfne ships proxy |
| 9 | On the validity of electric brain signal predictions based on population firing rates (PMC, 2025) | [pmc](https://pmc.ncbi.nlm.nih.gov/articles/PMC12052147/) | method | when rate-based field proxies hold | none | 2, 3 | shortlist for the readout side: tests what a proxy may claim |
| 10 | Emergence of Balanced Cortical Activity via Calcium-Regulated Synaptic Homeostasis (bioRxiv, 2025) | [biorxiv](https://www.biorxiv.org/content/10.1101/2025.09.04.674182.full.pdf) | model | balanced activity; E and I adaptation with limited resources | direct HDP analogue | 1, 3?, 6 | shortlist: strongest HDP hook |
| 11 | Metastable dynamics emerge from local excitatory-inhibitory homeostasis in the cortex at rest (Netw Neurosci, 2025) | [mit press](https://direct.mit.edu/netn/article/9/3/938/128759/Metastable-dynamics-emerge-from-local-excitatory) | model | metastability from E-I homeostasis, multi-area | direct HDP analogue | 6, multi-area | hold: whole-brain scale; check reducibility |
| 12 | The interplay between homeostatic synaptic scaling and homeostatic structural plasticity maintains the robust firing rate of neural networks (eLife) | [elifesciences](https://elifesciences.org/articles/88376) | model | firing-rate robustness | structural plasticity | 1, 6 | reject: structural change is outside the HDP grammar |

Also seen, not screened in: Rosetta Stone of Neural Mass Models (arXiv
2512.10982, neural-mass scale); Biophysical and computational insights from
modeling human cortical pyramidal neurons (Front Neurosci 2025, single-cell
review); Modular arrangement of synaptic and intrinsic homeostatic plasticity
within visual cortical circuits (PNAS 2025, experimental); Stability and
learning in excitatory synapses by nonlinear inhibitory plasticity (bioRxiv
2022); Energy optimization induces predictive-coding properties in a
multi-compartment spiking network (2024); Local field potentials primarily
reflect inhibitory neuron activity (Sci Rep, before the window).

## Provisional shortlist for P1 (6)

| Slot | Row | Why |
|---|---|---|
| Readout target | #1 + #2 | the spectrolaminar motif and its dispute; the proxy readout can express it |
| Microcircuit oscillation | #3 | E/PV/SOM, gamma and stability; nearest to existing populations |
| Predictive coding | #4 | PV/SST/VIP, oscillations, cell-type inactivations; omission étude exists |
| Readout validity | #9 | states what a rate-based proxy can claim; guards our wording |
| HDP-native | #10 | calcium-regulated E and I homeostasis; HDP on/off has a clear question |

Held: #5 (compartments), #6 (unclear), #8 (solved field), #11 (whole brain).
Rejected: #12 (structural plasticity).

## Next (P1)

1. Read each shortlisted source directly; record authors, year, DOI,
   parameters and observables with locators.
2. Drop any row whose model or observables the source does not give.
3. Human confirms 4–6 targets, then one pilot goes through P2–P5.
