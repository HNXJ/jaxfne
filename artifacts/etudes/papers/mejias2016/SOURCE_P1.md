# mejias2016: P1 source record

Target: Mejias JF, Murray JD, Kennedy H, Wang X-J (2016), Feedforward and feedback frequency-dependent interactions in a large-scale laminar network of the primate cortex, Sci Adv 2(11):e1601335, doi:10.1126/sciadv.1601335. Read from the open-access PMC record (PMCID PMC5262462, fetched 2026-10-19; nothing saved in the repo), cross-checked against nothing else: the Supplementary Methods PDF (`1601335_SM.pdf`, 1.5 MB, listed on the PMC page) is not fetchable with this run's tools (see §4).

Pointers below are article sections and figure captions of the PMC record. Status is observed (paper states it), inferred (computed from stated values), or unknown (paper does not give it in reachable text). No passage is quoted.

## 1. Model equations (Results, Figs. 2-6; Methods/Computational model)

Local module (observed, Results/Fig. 2A, Methods/Computational model): one nonlinear Wilson-Cowan-type firing-rate pair (excitatory plus inhibitory population) per laminar module. Supragranular module (L2/3) tuned to a noise-driven gamma rhythm near 40 Hz; infragranular module (L5/6) tuned to alpha near 10 Hz or low-beta 15-30 Hz. Full differential equations, time constants, weights and noise amplitudes live in the unreachable Supplement (Methods/Computational model says so explicitly).

Interlaminar motif (observed, Results/Fig. 3A): two projections only, the strongest found between L2/3 and L5. Excitatory L2/3E to L5E; L5E to L2/3I (net inhibitory downward influence).

Interareal motif (observed, Results/Fig. 4): feedforward approximated as source L2/3E to target L2/3E (standing in for L4 targeting, L4 itself unmodeled). Feedback from source L5/6E to all four target populations (E and I, both layers), weighted stronger onto L5/6E than onto L2/3E; the exact ratio was tuned, with a 0.8 supra/infra variant shown in fig. S1. Area signal for spectral analysis is a weighted combination of L2/3 and L5/6 activity mimicking depth-electrode recording (Results/Fig. 5A).

Directionality indices (observed, Results/Figs. 5-6): `DAI_{V1->V4}(f) = (GC_{V1->V4}(f) - GC_{V4->V1}(f)) / (GC_{V1->V4}(f) + GC_{V4->V1}(f))` (Eq. 1); `mDAI = (DAI(gamma) - DAI(alpha)) / 2` with gamma 30-70 Hz and alpha 6-18 Hz (Eq. 2, sign per the negative SLN-DAI correlation in alpha).

## 2. Parameters (Results figure legends and captions; Methods/Anatomical data)

| Parameter | Paper value | Status |
|---|---|---|
| Supragranular rhythm | noise-driven gamma, ~40 Hz (Fig. 2A) | observed |
| Infragranular rhythm | alpha ~10 Hz or low-beta 15-30 Hz (Fig. 2A) | observed |
| FF / FB bands | gamma 30-70 Hz; alpha/low-beta 8-15 Hz (Abstract, Fig. 5); mDAI alpha average 6-18 Hz (Fig. 6) | observed |
| Fig. 3A background | I = 8 to both layers' E populations | observed |
| Fig. 3B inputs | I = 6 supra-E, I = 8 infra-E | observed |
| Fig. 4A FF stim | I = 15 to V1 supra- and infra-E plus background I = 2 supra / I = 4 infra | observed |
| Fig. 4D FB stim | I = 15 to all V4 E plus background I = 1 | observed |
| Fig. 5 bidir stim | I = 8 to all E populations | observed |
| Fig. 6 full net | I = 8 to V1 L2/3E plus background I = 6 to all E populations | observed |
| Fig. 7B laminar sweep | lower-area L2/3E 4 to 6 to 8 while L5/6E 8 to 6 to 4; higher area fixed I = 6; specificity S = -4, 0, 4 | observed |
| Fig. 7E-H jump test | laminar specificity S = 8 in area 8l vs S = 0 | observed |
| Areas | 30 named areas (Methods/Anatomical data: V1, V2, V4, TEO, 9/46d, F5, 8m, 7A, DP, 2, 5, 7B, STPr, STPi, STPc, PBr, TEpd, 24c, F1, F2, F7, ProM, 8L, 9/46v, 46d, 8B, MT, 7m, 10, LIP) | observed |
| Graph density | ~66% (Results/Large-scale) | observed |
| FLN plot threshold | projections with FLN > 0.005 drawn (Fig. 6B caption) | observed |
| SLN scale | 1 purely FF, 0 purely FB (Results/Fig. 6A) | observed |
| Hierarchy subset | V1, V2, V4, DP, 8m, 8l, TEO, 7A (Methods/Anatomical data, Fig. 6) | observed |
| FF anatomy | supragranular origin, targets L4 (Results/Interareal) | observed |
| FB anatomy | infragranular origin, targets supra- and infragranular avoiding L4 (Results/Interareal) | observed |
| FB supra/infra variant | ratio 0.8 (fig. S1 caption) | observed |
| Wilson-Cowan constants, weights, noise, delays, dt, duration, seeds, GC settings | in Supplement / authors' code, not in reachable text | unknown |

## 3. Claims per target figure (Results, figure captions)

Fig. 2, intralaminar (observed): raising E-population input strengthens the gamma peak (strongest in 30-60 Hz; Fig. 2B) and shifts it from ~30 toward ~40 Hz; power and frequency saturate at high inputs (Fig. 2C). Limitation stated: contrast enhancement of high-gamma (>100 Hz) spiking broadband is absent in the rate model (Results/Fig. 2B).

Fig. 3, interlaminar (observed): coupling adds a strong alpha component to the L2/3 spectrum and shifts the L5/6 alpha peak upward (Fig. 3A). L2/3 gamma power entrains to L5/6 alpha phase (Fig. 3B). Drive to L5/6E lowers L2/3 gamma power and firing rate while raising L5/6 alpha, giving an inverse supra-rate vs alpha-power relation (Fig. 3C).

Fig. 6, 30-area network (observed): stimulus-driven gamma strongest in early visual areas (V1, V2); relatively raised alpha in ventral-stream areas (V4, TEO). SLN-DAI correlation positive in gamma, negative in alpha/low-beta (Fig. 6E); mDAI-SLN correlation highly significant (Fig. 6F). Functional hierarchy: early visual bottom, FEF (8l, 8m) above, ventral and dorsal extrastriate streams top, same-stream areas clustered (Fig. 6G-H).

Fig. 7, hierarchy dynamics (observed): laminar-specific input to the lower area raises FF gamma DAI and the pair's hierarchical distance (Fig. 7B-C); in the full net, S = 8 input to 8l shifts alpha/gamma power (Fig. 7E-F), raises 8l-to-8m gamma DAI (Fig. 7G) and jumps 8m upward (Fig. 7H); jump size grows with S (Fig. 7I).

Supporting microstimulation (observed, Fig. 4 insets): FF stim gives a highly significant V4 L2/3 gamma-power increase (***P < 0.001) with a small nonsignificant V4 L5/6 alpha decrease; FB stim gives a drastic V1 L2/3 gamma-power drop with a strong V1 L5/6 alpha-power increase (***P < 0.001).

## 4. Data, code and licence

- Paper licence: CC BY-NC 4.0 (observed, PMC copyright notice).
- Authors' code location and licence: laboratory website of X.-J.W. (observed, Data and materials availability); no URL or licence stated there, repository unknown.
- Data location and licence: `core-nets.org` files `LIP_Dataset.xlsx`, `Neuron_2015_Table.xlsx`, `PNAS_2013_Distance_Matrix.xlsx` (observed, Methods/Anatomical data); licence unstated; the site served a login wall on 2026-10-19 (observed).
- Supplement tried and unreachable this run (observed): PMC `bin/1601335_SM.pdf` and the index html (JS interstitial), Europe PMC mirror (403), publisher DC1 over http and https (403), science.org article and supplement endpoints (403). Equations, parameter tables, wiring distances (fig. S4), connectivity matrices (figs. S2-S3) and analysis routines stay unknown until the supplement or code is read.

## 5. Gaps (each item observed or inferred; jaxfne paths checked in this tree)

Have (observed): spiking emitter families LIF/AdEx/HH/Izhikevich (`LIFParams` in `jaxfne/emitters_lif.py`, `IzhikevichEmitter` in `jaxfne/emitters.py`); constant and Poisson drive (`XSource` in `jaxfne/inputs.py`); per-edge delays (`edge_delay_steps_from_ms` in `jaxfne/emitters.py`, `resolve_edge_delay_steps` in `jaxfne/emitters.py`); compiled connection rules (`compile_connection_rules` in `jaxfne/connectivity.py`); spectral-peak and regime estimators (`_spectral_peak_excess` in `jaxfne/analysis/metrics.py`, `activity_regime` in `jaxfne/analysis/metrics.py`).

- G1 (observed): no Wilson-Cowan (rate) emitter in `jaxfne/` (grep for wilson/w-c rate models: no match); the paper's unit is a rate E-I pair, so P2 needs rate dynamics: Hamm authorized a rate-population (Wilson-Cowan-type) core emitter for this lane under B3, behind the P4 perturbation gate (todo T-32f, 2026-10-18).
- G2 (inferred): the two-module laminar motif (gamma supra vs alpha infra) is not a core construct; wiring it is a P2 build on the public API.
- G3 (inferred): interlaminar specificity (L2/3E to L5E, L5E to L2/3I) fits the core's per-edge rules as a P2 configuration question.
- G4 (inferred): interareal laminar patterns (FF onto L2/3E; FB onto all four, infra-weighted) likewise; the FB supra/infra ratio is a tuned value from the unreachable supplement.
- G5 (observed): no spectral Granger-causality or coherence estimator in `jaxfne/analysis` (grep: no match); DAI/mDAI need new tested estimators at P2.
- G6 (observed): full numerics (time constants, weights, noise, delays, dt, seeds) unknown per §4; the authors' code would settle G4-G6.
- G7 (inferred): a 30-area by 4-population network through the public API is a P2 scale question, not a claim.
