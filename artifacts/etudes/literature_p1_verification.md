# Literature reproduction études: P1 verification (partial)

Stage P1 of `literature_reproduction_plan.md`, 2026-09-29. Rows follow
`literature_p0_candidates.md`.

## How it was read, and its limit

Sources were fetched one at a time. Access used: PMC full text, the PLOS
article page, and Crossref records (title, authors, date, DOI, abstract) for
bioRxiv preprints, whose full text returned HTTP 429 or 503. The fetch tool
summarizes a page with a small model, so every quote below is **as returned by
that tool** and is re-checked against the PDF at P2 before a parameter is used.
A row marked "pending" has no verified parameters yet.

## Verified records

| # | Citation (as returned) | Kind | Verified content | Not yet verified |
|---|---|---|---|---|
| 1 | Mendoza-Halliday D, Major AJ, Lee N, Lichtenfeld MJ, Carlson B, Mitchell B, Meng PD, Xiong Y, Westerberg JA, Jia X, Johnston KD, Selvanayagam J, Everling S, Maier A, Desimone R, Miller EK, Bastos AM. Nat Neurosci 2024;27(3):547–560. DOI 10.1038/s41593-023-01554-7 (PMC10917659) | data, no mechanistic model | gamma "50–150 Hz" peaks in layers 2/3; alpha-beta "10–30 Hz" peaks in layers 5/6; relative-power crossover marks layer 4 (Abstract; Fig. 1, Fig. 4e); 14 macaque areas, five monkeys, 810 probe recordings (942 expanded) (Methods) | FLIP code location; exact band edges used in each analysis |
| 2a | Mackey CA, Duecker K, Neymotin S, Dura-Bernal S, Haegens S, Barczak A, O'Connell MN, Jones SR, Ding M, Ghuman AS, Schroeder CE. "Is there a ubiquitous spectrolaminar motif of local field potential power across primate neocortex?" bioRxiv 2024-09-19, DOI 10.1101/2024.09.18.613490; published as Nat Neurosci, DOI 10.1038/s41593-025-02167-y (link as returned by Crossref) | data re-analysis | abstract: gradient in "61–64% of our recordings"; crossing point identified layer 4 in "29–33% of our recordings"; motif "common but not universal" | full text, area list, exact criteria, published-version year (Nature page gave a login redirect) |
| 2b | Major AJ, Abdaltawab A, Phillips JM, et al. "A ubiquitous spectrolaminar motif across independent studies, including Mackey et al.'s own data." bioRxiv 2025-05-08, DOI 10.1101/2025.05.07.652644 (PMC12247776) | data re-analysis, rebuttal | bands "10–19 Hz" and "75–150 Hz"; ω gradient metric; motif detected in 65% of A1, 67% of V1, 64% of Belt probes in Mackey's data; vFLIP2 layer error "0.08mm" | the "A. J. Major et al. reply" record (PMC13504716) was seen in search, not read |
| 3 | Tahvili F, Vinck M, di Volo M. "A cortical microcircuit model reveals distinct inhibitory mechanisms of network oscillations and stability." bioRxiv 2025-02-24, DOI 10.1101/2025.02.23.639719 | model, preprint | abstract: mechanism named CAMINOS; distinct causal roles of inhibitory classes; "stochastic gamma oscillations with drive-dependent frequency"; "asymmetric PV-SOM connectivity" as the key ingredient; prediction: higher SOM/PV density along the hierarchy lowers frequency (gamma to alpha/beta) and raises seizure susceptibility | neuron model, populations, weights, drive, dt (full text blocked); the "8000 E, 800 PV, 600 SOM, ~35 Hz" figure came from a search snippet and is unverified |
| 4 | Lee K, Pennartz CMA, Mejias JF. "Cortical networks with multiple interneuron types generate oscillatory patterns during predictive coding." PLOS Comput Biol, 2025-09-10, DOI 10.1371/journal.pcbi.1013469 (preprint DOI 10.1101/2024.10.27.620494) | model, published | **rate-based, not spiking**; 2 areas; area 1 L4 1024 E and L2/3 microcircuits (E, PV, SST, VIP), area 2 L4 784 E/784 PV and L5 784 E; membrane time constant 20 ms for E and I; ~6 Hz rhythm (Fig 4C), frequency set by the time constant (Fig 4D); PV, SST and VIP silencing effects (Fig 6); enhanced deviant response (Fig 5); trained on image data with plastic inter-area synapses; code github.com/Kwangjun-uva/CoCoPC | dt; whether jaxfne has a rate-based emitter (check at P2) |
| 9 | Ness TV, Tetzlaff T, Einevoll GT, Dahmen D. "On the validity of electric brain signal predictions based on population firing rates." PLOS Comput Biol 2025, DOI 10.1371/journal.pcbi.1012303 (PMC12052147) | method | kernel method for LFP/EEG/MEG from firing rates works best with spatially clustered input, correlated spike trains, large out-degree, strong signals; fails with uniform dendritic input, uncorrelated trains, small out-degree; relative error "inversely proportional to the signal amplitude" (Section 2.5, Fig 11C); layer-5 pyramidal cell (Hay et al. 2011), LFPy 2.3, NEURON 8.2, NEST 3.6; code github.com/torbjone/kernel_validity_paper | none needed for its role below |
| 10 | van Vreeswijk C (spelled "van Vreewsijk" by the record), Farkhooi F (spelled "Farzada" by the record). "Emergence of Balanced Cortical Activity via Calcium-Regulated Synaptic Homeostasis." bioRxiv 2025-09-05, DOI 10.1101/2025.09.04.674182; no published version linked | model, preprint | abstract: E and I synapses co-adapt through calcium; calcium mean encodes firing rate, variance encodes irregularity; inhibitory synapses set by the mean, excitatory by the variance; stabilizes rates, preserves irregular spiking, yields balanced E-I | equations, parameters, network size, dt (full text blocked); author-name spelling to confirm against the PDF |

## Row 3: model read from the authors' code (2026-09-29)

Full text stays blocked (bioRxiv 429). The published version is Cell Reports
2025 (title of the authors' code repository). Repository
`mdivolo/Tahvili-et-al-Cell-Reports-2025-code`, pushed 2025-07-08, **no
license**: read-only, parameters are recorded here, no code is copied. Values
below are read from its notebook (control condition of Fig. 1); the paper's
own methods are unread, so each is re-checked at P2.

| Item | Value in the notebook |
|---|---|
| Neuron | AdEx, conductance synapses, Brian2, dt 0.1 ms, 4 s run |
| Populations | 8000 E, 800 FS/PV, 600 SST (598 SST + 1 Poisson unit), p = 0.017, delay 0 |
| Common | C 200 pF, gL 10 nS, V_T -50 mV, V_reset -65 mV, Ee 0, Ei -80 mV, refractory 5 ms, tau_w 500 ms |
| E | a 4 nS, b 130 pA, DeltaT 2 mV |
| PV | a 0, b 0, DeltaT 0.5 mV |
| SST | a 4 nS, b 25 pA, DeltaT 1.5 mV, EL -55 mV |
| Synaptic taus | E to PV 1 ms, E to SST 2 ms, others 5 ms |
| Asymmetry | SST to PV weight nonzero; PV to SST and SST to SST weights 0 |
| Drive | 4 Hz Poisson to E and PV, N_ext = 0.017 x 8000 inputs, 1.25 nS each; none to SST |

## Feasibility flag for P2

jaxfne has no AdEx emitter (no match in `jaxfne/`); its spiking family is
Izhikevich. Row 3 needs (1) AdEx, or an Izhikevich mapping declared as a GAP,
(2) conductance synapses with per-connection taus, (3) zero-weight asymmetric
PV/SST wiring, (4) 10 000 neurons at 0.1 ms. Each is a GAP row at P2, not an
assumption. Row 10: no code repository found (GitHub title-term search
returned none); still needs the PDF.

## Findings that change the P0 plan

1. **#1 and #2 are one contested observable, not two targets.** The bands
   differ by source (50–150 and 10–30 Hz in the 2024 paper; 75–150 and 10–19 Hz
   in the 2025 rebuttal), and detection rates differ by data set and criterion.
   A jaxfne column can be asked whether it produces a superficial-gamma,
   deep-alpha/beta gradient, and the tolerance must name which band definition
   and detection rule. Neither source is a mechanism to reproduce.
2. **#3 links the spectrolaminar question to a hierarchy prediction**
   (SOM/PV density along the hierarchy sets the frequency). That is testable
   in a multi-area jaxfne network, with HDP on or off.
3. **#4 is rate-based and trained.** Its model class, image training and
   plastic inter-area synapses make it a poor first target. Hold until P2
   shows whether a rate-based emitter exists; keep its qualitative results
   (PV, SST, VIP silencing, deviant enhancement) as a comparison list.
4. **#9 is a wording guard, not an étude target.** It needs morphology and
   NEURON; it states when a rate-based field proxy may be trusted. It informs
   how the readout in every étude is described.
5. **#10 is the HDP-native target**, but its equations are unread. It is
   also spiking with calcium variables; check they fit the HDP state
   grammar at P2.

## Status per row after P1

| # | P1 status | Next |
|---|---|---|
| 1, 2a, 2b | verified (metadata and observables) | fix the band definition and detection rule at P3 |
| 3 | model parameters read from the authors' code (no license); paper methods unread | GAP list at P2; re-check against the PDF |
| 4 | verified; demoted to comparison | check for a rate-based emitter at P2 |
| 9 | verified; role changed to wording guard | none |
| 10 | metadata and abstract verified; equations pending | full text needed |

Human, please: supply PDFs for #3 and #10 (bioRxiv blocks automated fetches), or
say to keep retrying; and confirm the change of targets: #1+#2 as one observable,
#3, #10 as first candidates, #4 comparison, #9 wording guard.
