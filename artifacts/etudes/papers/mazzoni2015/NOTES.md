# Mazzoni et al. 2015: deep-read notes (O-87)

Target: Mazzoni A, Lindén H, Cuntz H, Lansner A, Panzeri S, Einevoll GT (2015), Computing the Local Field Potential
(LFP) from Integrate-and-Fire Network Models, PLoS Comput Biol 11(12):e1004584, doi:10.1371/journal.pcbi.1004584
(Crossref-verified 2026-11-08; CC BY). Local copy `F:/warehouse/papers_o87/mazzoni2015/paper.pdf`; `L<n>` is line n
of `paper_lines.txt` beside it. Status tags as in the other lanes. Code and data: LIF model ModelDB 152539; 3D LFP
code via LFPy; generated data on Dryad doi:10.5061/dryad.j5r51 (L58-69; Dryad contents not yet inspected: unknown).

## Question and scope

Which quantity computed from a point-neuron LIF network best predicts the LFP that the same synaptic input would
generate in a 3D network of multicompartment neurons (L24-34). Model only; layer 2/3-like column, current-based
synapses by default (L237-248, L1290-1293).

## Argument map

| Step | Content | Locator |
|---|---|---|
| Premise | LFP needs spatially separated transmembrane currents; point LIF neurons have none | L22-24 |
| Method | LIF: 4000 E, 1000 I, p 0.2; thalamic Poisson input plus an OU "cortical" input | L221-226, Table 1-2 (L279-485) |
| Method | Ground truth: LIF spike times injected into 5000 multicompartment cells in two stacked cylinders (r 250 um, h 250 um); AMPA over the whole tree, GABA in the lower cylinder | L230-248, L420-423 |
| Method | Proxies z-scored and time-shifted; delay and scale fit per depth by R^2 | L921-929 |
| Result | Six proxies ranked: sum abs I 0.83 > -GABA 0.81 > AMPA 0.78 > sum I, Vm 0.69 > FR 0.51 (mean R^2 over depth, 1.5 spikes/ms) | L955-960 |
| Result | WS proxy (Eq. 3): alpha > 0 at every depth, tau_AMPA 5-7 ms, tau_GABA -1 to 1 ms, R^2 up to 0.93 | L1012-1017 |
| Result | RWS (Eq. 4): alpha 1.65, tau_AMPA 6 ms, tau_GABA 0; close to WS across depths and lateral positions, > 90% | L1019-1034, L1042-1054 |
| Result | Across input 1-6 spikes/ms: WS > 0.91, RWS 0.88-0.91 (1-3) and 0.87 (6); at 0.5 no proxy works | L1107-1118 |
| Result | Robust to morphology above a ~100 um bush separation, to synapse layout unless homogeneous, and to conductance synapses (RWS 0.91-0.93) | L1182-1204, L1243-1289, L1301-1312 |
| Conclusion | RWS is a fixed formula for LFP from LIF output when one pyramidal population dominates | L1354-1386 |

## Claims table

| Claim | Evidence | Standing | Caveat |
|---|---|---|---|
| RWS = Norm[sum_pyr AMPA(t - 6 ms) - 1.65 sum_pyr GABA(t)] explains > 90% of LFP variance over a wide range of electrode positions; 88-91% for inputs 1-3 spikes/ms | L1023-1032, L1052-1053, L1113-1114 | Provisional (one model family) | AMPA positive, GABA negative sign convention (L916-917); scale f_RWS fit per position |
| Interneuron (stellate) contribution to the LFP is negligible | L885-888, Fig 3 | Provisional | due to stellate morphology (L1190-1191) |
| Proxy ranking is stable across input rates 1-6 spikes/ms | L1117-1118, Fig 6C | Provisional | |
| Gamma peak frequency predicted by all proxies above 1 spikes/ms | L1164-1165, Fig 7C | Provisional | WS/RWS underestimate gamma power at 6 spikes/ms (L1164) |
| Homogeneous synapse placement gives no usable LFP or proxy | L1247-1249, L1466-1467 | Provisional | |

## Quantities

| Quantity | Value | Unit | Condition | Locator |
|---|---|---|---|---|
| N_E / N_I, p | 4000 / 1000, 0.2 | count, prob | | L434-438 |
| V_thr, V_reset | 18, 11 | mV | both types | L441-446 |
| tau_m | 20 (E), 10 (I) | ms | | L447-449 |
| tau_ref | 2 (E), 1 (I) | ms | | L450-452 |
| AMPA rise/decay | 0.4/2 (onto E), 0.2/2 (onto I) | ms | | L459-465 |
| GABA rise/decay | 0.25/5 | ms | | L454-458 |
| J_GABA, J_AMPA-cort, J_AMPA-th | -1.7/-2.7, 0.42/0.7, 0.55/0.95 | mV | onto E / onto I | L466-474 |
| Thalamic input | 0.5:0.5:3 and 6 | spikes/ms | | L476-478 |
| OU tau, sigma | 16, 0.25 | ms, mV | | L479-484 |
| Reference input | 1.5 | spikes/ms | Figs 1-4 | L1132 |
| RWS alpha, tau_AMPA, tau_GABA | 1.65, 6, 0 | -, ms, ms | mean over depths | L1019-1022 |
| Best sum-abs-I lag | 1 | ms | Fig 4D-E | L966-968 |
| Simulated duration | 10.1, first 100 ms removed | s | LIF runs | L1665-1666 |
| Conduction delay, synaptic latency | 1, 1 | ms | | L1613, L1637 |
| Integration step | 0.05 | ms | | L1665 |

## Disputes

- Laminar population analysis convolves firing rates with a kernel; RWS is the synaptic-current analogue (L1393-1398).
- jaxfne P-056 (kernel-LFP deep-layer power < 0.5% of L2/3) concerns a multi-population column; Mazzoni's formula is
  stated for one dominant population, with a linear combination suggested for several (L1378-1386).

## Open questions

- Does RWS hold for a multi-layer column (several pyramidal populations)? Settle: per-population RWS summed with
  depth weights against a kernel or multicompartment ground truth (L1380-1386 proposes this).
- Contents of the Dryad deposit (LIF currents? 3D LFP per depth?): decides whether a ground-truth oracle exists
  without rebuilding the 3D network.

## Candidate target figure

**Fig 5, the RWS proxy.** Estimand: depth-averaged R^2 between RWS (alpha 1.65, tau_AMPA 6 ms, tau_GABA 0) and the
ground-truth LFP at 1.5 spikes/ms, with the WS fit's optimal alpha per depth (Fig 5A-B, L1012-1034). Criterion
proposal: RWS mean R^2 >= 0.88, the low end of the 88-91% the text gives for 1-3 spikes/ms (L1113-1114; the
1.5 spikes/ms value itself is read from Fig 6C), and WS alpha in the band around 1.65 shown in Fig 5A (proposal, for
the prereg; bands read from the figures).

Fit to the core: the LIF network (Table 2) is current-based LIF on jaxfne's spiking path; AMPA and GABA sums are
synaptic currents (`jaxfne/fields/proxy.py` `synaptic_current`, `exponential_synaptic_trace`), so the proxies are
core readouts. The ground truth is the cost driver: (a) the Dryad deposit if it holds the 3D LFP for a given LIF run
(oracle, cheapest); (b) jaxfne's kernel LFP (`jaxfne/fields/kernel_lfp.py`, Hagen 2016 cells) as a declared
substitute for the LFPy morphologies (a different ground truth, must be stated); (c) LFPy outside the core as an
oracle arm. Cheap stage 0, LIF only: Fig 7C, gamma peak frequency against thalamic input above 1 spikes/ms (L1164-1165; the
tested inputs are 0.5:0.5:3 and 6, L476-478).
