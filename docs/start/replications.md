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
| [CAMINOS reference model, Fig. 1](https://hnxj.github.io/jaxfne/dashboard/#model/caminos)<br>Tahvili, Vinck and di Volo, PLoS Comput Biol 2026 | mixed | 0.5.7 | P6 | 48/100 | mixed: Reproduced under v7 (amended after v6 R4 failed): f_pk, rates and R4. Missed: against PLoS CB (primary, Hamm 2026-10-14) the PV→SOM lag by the ref. 42 method is 6.47 ms against 8.5 ms (off, e 0.24), a finding that needs X. Library path, scored apart: P6 v2 PASS through the public API (Q1–Q5, C1). The 0.1 ms delay sensitivity gives the preprint's ≈ 7 ms | P6 PASS | — |
| [CAMINOS figure claims, Figs. 1-6 (drive, silencing, randomization, connectivity, SOM/PV ratio)](https://hnxj.github.io/jaxfne/dashboard/#model/caminos-cellrep)<br>Tahvili, Vinck and di Volo, Cell Reports 44:116131 (2025) | in progress | 0.5.9 | P2 | 71/100 | — | — | — |
| [Predictive coding light, Fig. 3 (simple and complex cell tuning)](https://hnxj.github.io/jaxfne/dashboard/#model/pcl-fig3)<br>N'dri, Barbier, Teulière and Triesch, Nat Commun 2025 | pending | 0.5.6 | P5 | 67/100 | — | — | — |
| [Predictive coding light, Fig. 2 (inhibitory STDP removes predictable spikes)](https://hnxj.github.io/jaxfne/dashboard/#model/pcl-fig2)<br>N'dri, Barbier, Teulière and Triesch, Nat Commun 2025 | in progress | 0.5.6 | P1 | — | — | — | — |
| [Hodgkin–Huxley reference against a reduced jaxfne neuron (AT-01)](https://hnxj.github.io/jaxfne/dashboard/#model/hh-at01)<br>Hodgkin and Huxley 1952, through Jaxley | in progress | 0.5.5 | P3 | — | — | — | — |
| [Q-learning on jaxfne network features](https://hnxj.github.io/jaxfne/dashboard/#model/qlearning)<br>Watkins and Dayan 1992 (learning rule) | in progress | 0.5.6 | P3 | — | — | — | — |
| [Izhikevich 2003, Figs. 2 and 3 (single-cell patterns and the 1000-cell network)](https://hnxj.github.io/jaxfne/dashboard/#model/izhikevich2003)<br>Izhikevich, IEEE Trans Neural Netw 2003 | pending | 0.5.7 | P6 | 58/100 | — | P6 PASS | — |
| [GSDR, Fig. 6 (a 50-neuron E, SST-like and PV-like circuit fitted to recorded spectra)](https://hnxj.github.io/jaxfne/dashboard/#model/gsdr2026)<br>Nejat, Sherfey and Bastos, PLOS One 2026 | pending | 0.5.8 | P3 | 0/100 | — | — | — |
| [Destexhe 2009, Fig. 6B (self-sustained asynchronous irregular state in an AdEx cortical network)](https://hnxj.github.io/jaxfne/dashboard/#model/destexhe2009)<br>Destexhe, J Comput Neurosci 2009 | pending | 0.5.9 | P3 | 4/100 | — | — | — |
| [Brunel 2000, Fig. 8 (SR, SI fast, AI and SI slow states of a sparse LIF network)](https://hnxj.github.io/jaxfne/dashboard/#model/brunel2000)<br>Brunel, J Comput Neurosci 2000 | reproduced (conditional) | 0.5.7 | P5 | — | reproduced (conditional): Every estimand in its band and Q1-Q3 pass on seeds 4-8 under signed v4. Condition: Q3 (AI at C) reads the paper's wording, stationary global activity plus CV > 0.1, chosen after the seed-0 pilot failed the classifier's asynchronous bounds (disclosed in v4); at C the cells stay more regular than those bounds (CV 0.40-0.41, kappa 0.20). | — | — |
| [Potjans and Diesmann 2014 cortical microcircuit, Fig. 6 and Table 5](https://hnxj.github.io/jaxfne/dashboard/#model/potjans2014)<br>Potjans and Diesmann, Cereb Cortex 24(3):785-806 | pending | 0.5.9 | P3 | 2/100 | — | — | — |
| [Jansen and Rit 1995, Fig. 3 (regime sequence of the single-column model across the C sweep)](https://hnxj.github.io/jaxfne/dashboard/#model/jansenrit1995)<br>Jansen and Rit, Biol Cybern 73(4):357-366 (1995) | reproduced (conditional) | 0.5.10 | P3 | — | —: Conditional: owner decision 2 of the signed prereg (ruled as drafted) replaces the paper's unstated noise hold and RKF45 with a 2 ms hold and RK4 at dt 0.1 ms. The pass holds under that declared hold and integrator only. | — | — |
| [Vogels et al. 2011, Fig. 4 (inhibitory STDP balancing a recurrent network) and Figs. 1-2 (single-cell detailed balance)](https://hnxj.github.io/jaxfne/dashboard/#model/vogels2011)<br>Vogels et al., Science 334(6062):1569-1573 | pending | 0.5.9 | P3 | 0/100 | — | — | — |
| [Mejias et al. 2016 laminar multi-area model, Figs. 2-3 and 6-7](https://hnxj.github.io/jaxfne/dashboard/#model/mejias2016)<br>Mejias, Murray, Kennedy and Wang, Sci Adv 2(11):e1601335 | in progress | 0.5.9 | P2 | 0/100 | — | — | — |
| [Analytic EEG/MEG forward checks of jaxfne.fields.head](https://hnxj.github.io/jaxfne/dashboard/#model/head-forward)<br>Sarvas 1987 (MEG); Frank 1952 (EEG) | core method | 0.5.7 | — | — | — | — | pass |
| [Emitter closed-form checks of the jaxfne emitter kernels](https://hnxj.github.io/jaxfne/dashboard/#model/emitter-closed-form)<br>closed-form and hand-computed references in the pinned tests (no paper value) | core method | 0.5.8 | — | — | — | — | pass |
| [Core checks of the field and probe invariants in jaxfne.fields](https://hnxj.github.io/jaxfne/dashboard/#model/field-invariants)<br>Nicholson and Freeman 1975 (CSD three-point stencil); Gauss's law in one dimension | core method | 0.5.8 | — | — | — | — | pass |
| [Core checks of the spectral Granger causality in jaxfne.analysis](https://hnxj.github.io/jaxfne/dashboard/#model/granger-geweke)<br>Geweke 1982, JASA 77:304-313 (spectral Granger causality from a fitted VAR) | core method | 0.5.10 | — | — | — | — | pass |
| [Hand-value plasticity checks of the jaxfne core](https://hnxj.github.io/jaxfne/dashboard/#model/plasticity-hand-values)<br>Vogels et al., Science 334(6062):1569-1573; docs/guides/state/hdp.md (barrier equilibrium); protocol D2a F1 relaxation | core method | 0.5.8 | — | — | — | — | pass |
| [Objective and optimizer stages against closed forms and independent references](https://hnxj.github.io/jaxfne/dashboard/#model/objective-optimizer-references)<br>analytic references; no paper value | core method | 0.5.8 | — | — | — | — | pass |
| [Core-method numerics checks: integrators and autodiff gradients](https://hnxj.github.io/jaxfne/dashboard/#model/numerics-references)<br>Closed-form exponential and Euler iterate; central finite differences (no paper values) | core method | 0.5.8 | — | — | — | — | pass |
| [Core-vs-oracle parity checks of the AdEx, multi-compartment HH and MCDP routes](https://hnxj.github.io/jaxfne/dashboard/#model/port-oracle-parity)<br> | core method | 0.5.8 | — | — | — | — | pass |

## Not started

| Release | Model | Needs |
|---|---|---|
| 0.5.11 | Lee, Pennartz and Mejias, PLoS Comput Biol 2025: Fig. 6 interneuron silencing | a ReLU rate transfer and Hebbian training; Fig. 7 motif search first (lee2025/NOTES.md) |
| 0.5.11 | Mazzoni et al. 2015: Fig. 5 reference weighted-sum LFP proxy | a multicompartment reference LFP (Dryad deposit, or jaxfne kernel LFP declared as substitute); Fig. 7C first (mazzoni2015/NOTES.md) |
| after 0.5.10 | PCL Fig. 4: suppression effects | PCL Fig. 3 scored (R5-approximate, 0.5.10) |
| after 0.5.10 | PCL Figs. 5–6 | the PCL suppression data and N-MNIST/DVS sets |
| after 0.5.10 | Calcium-regulated synaptic homeostasis (bioRxiv 2025) | methods read; calcium state as an HDP coordinate |
| unscheduled | Spectrolaminar motif (Nat Neurosci 2023 and the 2025 reply) | the laminar networks |

## Reference audit (OCMA)

H1 mechanism realized in the core, H2 reproduction evidence, H3 dashboard readiness; each is scored -1 to +1 by a three-model council. Action is the first rule of the OCMA glossary that the scores meet.

| Reference | Section | H1 | H2 | H3 | Action |
|---|---|---|---|---|---|
| Hodgkin and Huxley (1952) | Neural dynamics | 0.44 | 0.40 | 0.78 | exists |
| Izhikevich (2003) | Neural dynamics | 0.81 | 0.55 | 0.90 | exists |
| Brette and Gerstner (2005) | Neural dynamics | 0.55 | 0.13 | 0.77 | study-list |
| Tahvili et al. (2026) | Neural dynamics | 0.93 | 0.55 | 0.78 | exists |
| Wang and Buzsáki (1996) | Neural dynamics | 0.00 | 0.10 | 0.47 | study-list |
| Billeh et al. (2020) | Neural dynamics | 0.09 | 0.33 | 0.82 | study-list |
| Frank (1952) | Sources and extracellular observation | 0.50 | 0.60 | 0.90 | exists |
| Hagen et al. (2016) | Sources and extracellular observation | 0.43 | 0.35 | 0.53 | study-list |
| Nicholson and Freeman (1975) | Sources and extracellular observation | 0.13 | 0.45 | 0.90 | exists |
| Sarvas (1987) | Sources and extracellular observation | 0.50 | 0.65 | 0.90 | exists |
| Lindén et al. (2014) | Sources and extracellular observation | 0.06 | 0.00 | 0.60 | study-list |
| Hagen et al. (2018) | Sources and extracellular observation | 0.06 | 0.03 | 0.60 | study-list |
| Herreras (2016) | Sources and extracellular observation | 0.60 | 0.00 | 0.33 | roadmap |
| Pettersen et al. (2012) | Sources and extracellular observation | 0.04 | 0.00 | 0.60 | study-list |
| Einevoll et al. (2013) | Sources and extracellular observation | 0.06 | 0.00 | 0.60 | study-list |
| Einevoll (2020) | Sources and extracellular observation | -0.40 | 0.07 | 0.53 | study-list |
| Vorwerk et al. (2014) | Sources and extracellular observation | -0.40 | 0.00 | 0.33 | study-list |
| FitzHugh (1961) | Background: reduced and population models | 0.15 | 0.00 | 0.67 | study-list |
| Nagumo et al. (1962) | Background: reduced and population models | 0.08 | 0.00 | 0.53 | study-list |
| Morris and Lecar (1981) | Background: reduced and population models | -0.04 | 0.00 | 0.67 | study-list |
| Wilson and Cowan (1972) | Background: reduced and population models | 0.50 | 0.00 | 0.60 | study-list |
| Montbrió et al. (2015) | Background: reduced and population models | 0.17 | 0.00 | 0.43 | study-list |
| Coombes (2005) | Background: reduced and population models | 0.10 | 0.00 | 0.40 | study-list |
| Cook et al. (2022) | Background: reduced and population models | 0.10 | 0.00 | 0.40 | study-list |
| Ermentrout and Kleinfeld (2001) | Background: reduced and population models | 0.17 | 0.03 | 0.43 | study-list |
| Muller et al. (2018) | Background: reduced and population models | 0.20 | 0.00 | 0.33 | study-list |
| Geweke (1982) | Background: reduced and population models | 1.00 | 0.46 | 0.74 | core-check (build) |
| Clopath et al. (2010) | Background: circuits, rhythms and plasticity | 0.06 | 0.00 | 0.20 | study-list |
| Arkhipov et al. (2018) | Background: circuits, rhythms and plasticity | 0.07 | 0.00 | 0.20 | study-list |
| Tahvili et al. (2026) | Research studies (study list) | 1.00 | 0.80 | 0.82 | exists |
| Tahvili et al. (2025) | Research studies (study list) | 0.71 | 0.63 | 0.55 | exists |
| N'dri et al. (2025) | Research studies (study list) | 0.55 | 0.65 | 0.80 | exists |
| Nejat et al. (2026) | Research studies (study list) | 0.50 | 0.28 | 0.55 | exists |
| Destexhe (2009) | Research studies (study list) | 0.77 | 0.47 | 0.95 | exists |
| Brunel (2000) | Research studies (study list) | 0.93 | 0.72 | 0.78 | exists |
| Potjans and Diesmann (2014) | Research studies (study list) | 1.00 | 0.57 | 0.65 | exists |
| Vogels et al. (2011) | Research studies (study list) | 0.80 | 0.50 | 0.72 | exists |
| Mejias et al. (2016) | Research studies (study list) | 0.40 | 0.07 | 0.55 | exists |
| Batuev and Sukhov (2026) | Research studies (study list) | 0.00 | 0.00 | 0.13 | study-list |
| Kopell et al. (2000) | Research studies (study list) | 0.17 | 0.00 | 0.13 | study-list |
| Börgers and Kopell (2003) | Research studies (study list) | 0.17 | 0.00 | 0.36 | study-list |
| Prinz et al. (2004) | Research studies (study list) | 0.17 | 0.00 | 0.24 | study-list |
| Bastos et al. (2012) | Research studies (study list) | 0.17 | 0.00 | 0.20 | study-list |
| Bastos et al. (2015) | Research studies (study list) | 0.10 | 0.00 | 0.36 | study-list |
| van Kerkoerle et al. (2014) | Research studies (study list) | 0.10 | 0.00 | 0.44 | study-list |
| Mendoza-Halliday et al. (2024) | Research studies (study list) | 0.60 | 0.15 | 0.30 | roadmap |
| Tiesinga and Sejnowski (2009) | Research studies (study list) | 0.23 | 0.00 | 0.28 | study-list |
| Brunel and Wang (2003) | Research studies (study list) | 0.30 | 0.00 | 0.28 | study-list |
| van Vreeswijk and Sompolinsky (1996) | Research studies (study list) | 0.10 | 0.03 | 0.17 | study-list |
| Renart et al. (2010) | Research studies (study list) | 0.17 | 0.03 | 0.28 | study-list |
| Litwin-Kumar and Doiron (2012) | Research studies (study list) | 0.13 | 0.00 | 0.20 | study-list |
| Ostojic (2014) | Research studies (study list) | 0.17 | 0.00 | 0.40 | study-list |
| Tsodyks et al. (1997) | Research studies (study list) | 0.17 | 0.00 | 0.40 | study-list |
| Ozeki et al. (2009) | Research studies (study list) | 0.17 | 0.00 | 0.40 | study-list |
| Litwin-Kumar et al. (2016) | Research studies (study list) | 0.17 | 0.00 | 0.40 | study-list |
| Hertäg and Sprekeler (2020) | Research studies (study list) | 0.17 | 0.00 | 0.00 | study-list |
| Mazzoni et al. (2015) | Research studies (study list) | 0.47 | 0.00 | 0.55 | study-list |
| Mazzoni et al. (2008) | Research studies (study list) | 0.17 | 0.00 | 0.00 | study-list |
| Lindén et al. (2011) | Research studies (study list) | 0.20 | 0.00 | 0.20 | study-list |
| Hagen et al. (2016) | Research studies (study list) | 0.33 | 0.04 | 0.55 | study-list |
| Lee et al. (2025) | Research studies (study list) | 0.10 | 0.00 | 0.47 | study-list |
| Ness et al. (2025) | Research studies (study list) | 0.23 | 0.00 | 0.45 | study-list |
| Páscoa dos Santos and Verschure (2025) | Research studies (study list) | 0.17 | 0.00 | 0.33 | study-list |
| Vogels and Abbott (2005) | Research studies (study list) | 0.27 | 0.00 | 0.53 | study-list |
| Zenke et al. (2015) | Research studies (study list) | 0.17 | 0.00 | 0.33 | study-list |
| Song et al. (2000) | Research studies (study list) | 0.14 | 0.00 | 0.67 | study-list |
| Billeh et al. (2020) | Research studies (study list) | 0.17 | 0.30 | 0.63 | study-list |
| Schmidt et al. (2018) | Research studies (study list) | 0.06 | 0.00 | 0.33 | study-list |
| Chaudhuri et al. (2015) | Research studies (study list) | 0.03 | 0.00 | 0.33 | study-list |
| Joglekar et al. (2018) | Research studies (study list) | 0.06 | 0.00 | 0.33 | study-list |
| Veit et al. (2017) | Research studies (study list) | 0.17 | 0.00 | 0.33 | study-list |
| Wilmes and Clopath (2019) | Research studies (study list) | 0.17 | 0.00 | 0.53 | study-list |
| Kumar et al. (2008) | Research studies (study list) | 0.17 | 0.00 | 0.33 | study-list |
| El Boustani and Destexhe (2009) | Research studies (study list) | 0.17 | 0.00 | 0.53 | exists |
| di Volo et al. (2019) | Research studies (study list) | 0.30 | 0.03 | 0.53 | study-list |
| Spreizer et al. (2019) | Research studies (study list) | 0.14 | 0.00 | 0.20 | study-list |
| Rosenbaum et al. (2017) | Research studies (study list) | 0.14 | 0.00 | 0.20 | study-list |
| Huang et al. (2019) | Research studies (study list) | 0.14 | 0.00 | 0.20 | study-list |
| Keane and Gong (2015) | Research studies (study list) | 0.14 | 0.00 | 0.20 | study-list |
| Mejias and Longtin (2012) | Research studies (study list) | 0.14 | 0.00 | 0.20 | study-list |
| Wang (2002) | Research studies (study list) | 0.06 | 0.00 | 0.33 | study-list |
| Hansel and van Vreeswijk (2012) | Research studies (study list) | 0.11 | 0.00 | 0.33 | study-list |
| Wacongne et al. (2012) | Research studies (study list) | 0.11 | 0.00 | 0.33 | study-list |
| Spratling (2008) | Research studies (study list) | 0.11 | 0.00 | 0.33 | study-list |
| Rao and Ballard (1999) | Research studies (study list) | 0.11 | 0.00 | 0.33 | study-list |
| Markram et al. (2015) | Research studies (study list) | 0.17 | 0.00 | 0.25 | study-list |
| van Vreeswijk and Farkhooi (2025) | Research studies (study list) | 0.07 | 0.00 | 0.15 | study-list |
| Jansen and Rit (1995) | Research studies (study list) | 0.66 | 0.60 | 0.72 | core-check |

Generated by `scripts/generate_replications_page.py` from `dashboard/sources.json`.
