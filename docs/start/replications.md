# Replications

Published models rerun with jaxfne. Each paper has a page on the
[replication dashboard](https://hnxj.github.io/jaxfne/dashboard/) with its question, preregistration, criteria,
verdicts and figures. Stages P0–P7 run from selection to insight; a replication counts
as done with jaxfne only when it runs through the library (P6).

[Open the replication dashboard](https://hnxj.github.io/jaxfne/dashboard/)

Replica is the share of the paper's result panels reproduced, out of 100 (pass 1, partial 0.5,
fail or not run 0); 100 is a full replica. A dash means no panel inventory yet.

| Paper | Class | Release | Stage | Replica | Paper | Library | Methods |
|---|---|---|---|---|---|---|---|
| [CAMINOS reference model, Fig. 1](https://hnxj.github.io/jaxfne/dashboard/#model/caminos)<br>Tahvili, Vinck and di Volo, PLoS Comput Biol 2026 | mixed | 0.5.7 | P6 | 21/100 | mixed: Reproduced under v7 (amended after v6 R4 failed): f_pk, rates and R4. Missed: against PLoS CB (primary, Hamm 2026-10-14) the PV→SOM lag by the ref. 42 method is 6.47 ms against 8.5 ms (off, e 0.24), a finding that needs X. Library path, scored apart: P6 v2 PASS through the public API (Q1–Q5, C1). The 0.1 ms delay sensitivity gives the preprint's ≈ 7 ms | P6 PASS | — |
| [CAMINOS figure claims, Figs. 1-6 (drive, silencing, randomization, connectivity, SOM/PV ratio)](https://hnxj.github.io/jaxfne/dashboard/#model/caminos-cellrep)<br>Tahvili, Vinck and di Volo, Cell Reports 44:116131 (2025) | in progress | 0.5.9 | P2 | 44/100 | — | — | — |
| [Predictive coding light, Fig. 3 (simple and complex cell tuning)](https://hnxj.github.io/jaxfne/dashboard/#model/pcl-fig3)<br>N'dri, Barbier, Teulière and Triesch, Nat Commun 2025 | in progress | 0.5.6 | P5 | — | — | — | — |
| [Predictive coding light, Fig. 2 (inhibitory STDP removes predictable spikes)](https://hnxj.github.io/jaxfne/dashboard/#model/pcl-fig2)<br>N'dri, Barbier, Teulière and Triesch, Nat Commun 2025 | in progress | 0.5.6 | P1 | — | — | — | — |
| [Hodgkin–Huxley reference against a reduced jaxfne neuron (AT-01)](https://hnxj.github.io/jaxfne/dashboard/#model/hh-at01)<br>Hodgkin and Huxley 1952, through Jaxley | in progress | 0.5.5 | P3 | — | — | — | — |
| [Q-learning on jaxfne network features](https://hnxj.github.io/jaxfne/dashboard/#model/qlearning)<br>Watkins and Dayan 1992 (learning rule) | in progress | 0.5.6 | P3 | — | — | — | — |
| [Izhikevich 2003, Figs. 2 and 3 (single-cell patterns and the 1000-cell network)](https://hnxj.github.io/jaxfne/dashboard/#model/izhikevich2003)<br>Izhikevich, IEEE Trans Neural Netw 2003 | in progress | 0.5.7 | P5 | 58/100 | — | — | — |
| [GSDR, Fig. 6 (a 50-neuron E, SST-like and PV-like circuit fitted to recorded spectra)](https://hnxj.github.io/jaxfne/dashboard/#model/gsdr2026)<br>Nejat, Sherfey and Bastos, PLOS One 2026 | in progress | 0.5.8 | P2 | — | — | — | — |
| [Destexhe 2009, Fig. 6B (self-sustained asynchronous irregular state in an AdEx cortical network)](https://hnxj.github.io/jaxfne/dashboard/#model/destexhe2009)<br>Destexhe, J Comput Neurosci 2009 | in progress | 0.5.8 | P2 | 0/100 | — | — | — |
| [Brunel 2000, Fig. 8 (SR, SI fast, AI and SI slow states of a sparse LIF network)](https://hnxj.github.io/jaxfne/dashboard/#model/brunel2000)<br>Brunel, J Comput Neurosci 2000 | reproduced (conditional) | 0.5.7 | P5 | — | reproduced (conditional): Every estimand in its band and Q1-Q3 pass on seeds 4-8 under signed v4. Condition: Q3 (AI at C) reads the paper's wording, stationary global activity plus CV > 0.1, chosen after the seed-0 pilot failed the classifier's asynchronous bounds (disclosed in v4); at C the cells stay more regular than those bounds (CV 0.40-0.41, kappa 0.20). | — | — |
| [Potjans and Diesmann 2014 cortical microcircuit, Fig. 6 and Table 5](https://hnxj.github.io/jaxfne/dashboard/#model/potjans2014)<br>Potjans and Diesmann, Cereb Cortex 24(3):785-806 | in progress | 0.5.9 | P2 | 0/100 | — | — | — |
| [Vogels et al. 2011, Fig. 4 (inhibitory STDP balancing a recurrent network) and Figs. 1-2 (single-cell detailed balance)](https://hnxj.github.io/jaxfne/dashboard/#model/vogels2011)<br>Vogels et al., Science 334(6062):1569-1573 | in progress | 0.5.8 | P2 | — | — | — | — |
| [Mejias et al. 2016 laminar multi-area model, Figs. 2-3 and 6-7](https://hnxj.github.io/jaxfne/dashboard/#model/mejias2016)<br>Mejias, Murray, Kennedy and Wang, Sci Adv 2(11):e1601335 | in progress | 0.5.9 | P2 | 0/100 | — | — | — |
| [Analytic EEG/MEG forward checks of jaxfne.fields.head](https://hnxj.github.io/jaxfne/dashboard/#model/head-forward)<br>Sarvas 1987 (MEG); Frank 1952 (EEG) | core method | 0.5.7 | — | — | — | — | pass |
| [Emitter closed-form checks of the jaxfne emitter kernels](https://hnxj.github.io/jaxfne/dashboard/#model/emitter-closed-form)<br>closed-form and hand-computed references in the pinned tests (no paper value) | core method | 0.5.8 | — | — | — | — | pass |
| [Core checks of the field and probe invariants in jaxfne.fields](https://hnxj.github.io/jaxfne/dashboard/#model/field-invariants)<br>Nicholson and Freeman 1975 (CSD three-point stencil); Gauss's law in one dimension | core method | 0.5.8 | — | — | — | — | pass |
| [Hand-value plasticity checks of the jaxfne core](https://hnxj.github.io/jaxfne/dashboard/#model/plasticity-hand-values)<br>Vogels et al., Science 334(6062):1569-1573; docs/guides/state/hdp.md (barrier equilibrium); protocol D2a F1 relaxation | core method | 0.5.8 | — | — | — | — | pass |
| [Objective and optimizer stages against closed forms and independent references](https://hnxj.github.io/jaxfne/dashboard/#model/objective-optimizer-references)<br>analytic references; no paper value | core method | 0.5.8 | — | — | — | — | pass |
| [Core-method numerics checks: integrators and autodiff gradients](https://hnxj.github.io/jaxfne/dashboard/#model/numerics-references)<br>Closed-form exponential and Euler iterate; central finite differences (no paper values) | core method | 0.5.8 | — | — | — | — | pass |
| [Core-vs-oracle parity checks of the AdEx, multi-compartment HH and MCDP routes](https://hnxj.github.io/jaxfne/dashboard/#model/port-oracle-parity)<br> | core method | 0.5.8 | — | — | — | — | pass |

## Not started

| Release | Model | Needs |
|---|---|---|
| 0.5.8 | Jansen and Rit 1995: EEG and visual evoked potentials from coupled columns | a neural-mass emitter in the core |
| 0.5.9 | Lee, Pennartz and Mejias, PLoS Comput Biol 2025: interneuron silencing, deviant enhancement | event input |
| 0.5.10 | Spectrolaminar motif (Nat Neurosci 2023 and the 2025 reply) | the 0.5.9 laminar networks |
| 0.5.10 | Mazzoni et al. 2015: LFP proxies | the 0.5.9 laminar networks |
| after 0.5.10 | PCL Fig. 4: suppression effects | the PCL full-network mapping (0.5.8) |
| after 0.5.10 | PCL Figs. 5–6 | the PCL suppression data and N-MNIST/DVS sets |
| after 0.5.10 | Calcium-regulated synaptic homeostasis (bioRxiv 2025) | methods read; calcium state as an HDP coordinate |

Generated by `scripts/generate_replications_page.py` from `dashboard/sources.json`.
