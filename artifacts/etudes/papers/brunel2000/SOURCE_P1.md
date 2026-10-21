# Brunel 2000: P1 source record

Target: Brunel N (2000), Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons, J Comput Neurosci 8:183-208, doi:10.1023/A:1008925309027. Read from the author's own copy (linked as `pdfs/brunel00JCNS.pdf` on the author's publication page, fetched 2026-10-19; nothing saved in the repo), cross-checked against the NEST `brunel_delta` example documentation for parameters only.

Pointers below are journal pages and sections. Status is observed (paper states it), inferred (computed from stated values), or unknown (paper does not give it). No passage is quoted.

## 1. Model equations (paper §2-§4, pp. 184-190)

Model A, identical excitatory and inhibitory cells (observed, §2, p. 184):

- Membrane: `tau dVi/dt = -Vi + R Ii(t)` (Eq. 1).
- Synapses are delta currents with one fixed delay: `R Ii(t) = tau * sum_j Jij * sum_k delta(t - tkj - D)` (Eq. 2).
- Weights: external excitatory `J`; recurrent excitatory `J` (external equals recurrent); inhibitory `-gJ`.
- External drive: independent Poisson processes at rate `nuext` on `Cext` synapses per cell.
- Spike when `Vi` reaches `theta`; reset to `Vr`; absolute refractory `taurp`.
- Reference rate: `nuthr = theta / (J * CE * tau)`, the external rate holding a cell at threshold without feedback.

Model B, distinct E/I cells (observed, §2, p. 185): `tauE`, `tauI`; `JEE = JE`, `JEI = gE * JE`, `JIE = JI`, `JII = gI * JI`; external rates `nuE,ext`, `nuI,ext`; four delays `Dab`.

Mean field (observed, §3, p. 186): recurrent mean `mul(t) = CE * J * (1 - g*gam) * nu(t-D) * tau` with `gam = CI/CE`; external mean `muext = CE * J * nuext * tau`; noise `sigl = J * sqrt(CE * (1 + gam*g^2) * nu(t-D) * tau)`, `sigext = J * sqrt(CE * nuext * tau)`; cross-cell input correlations neglected in the `C/N -> 0` limit. States with constant `nu` are asynchronous; time-varying `nu` synchronous.

Stationary rates (observed, §4.1, pp. 188-190): high-activity branch Eq. 23, `nu0 = (1/taurp) * (1 - (theta - Vr) / (CE*J*(1 - g*gam)))`, nearly independent of `nuext`; low-activity branch Eq. 24, `nu0 = (nuext - nuthr) / (g*gam - 1)`, quasi-linear in `nuext` with gain `1/(g*gam - 1)`.

## 2. Parameters (observed, §2, p. 185; Fig. 8 set §6, p. 196)

| Parameter | Paper value | Status |
|---|---|---|
| `NE`, `NI` | 0.8N, 0.2N (80% excitatory) | observed |
| `gam = CI/CE` | 0.25 (`CE = 4*CI`) | observed |
| `Cext` | `CE` | observed |
| `tauE` | 20 ms | observed |
| `theta` | 20 mV | observed |
| `Vr` | 10 mV | observed |
| `taurp` | 2 ms | observed |
| Free (model A) | `g`, `nuext`, `J`, `CE`, `D` | observed |
| Fig. 8 set: `J` | 0.1 mV | observed |
| Fig. 8 set: `CE`, `CI` | 1000, 250 | observed |
| Fig. 8 set: `D` | 1.5 ms | observed |
| Fig. 8 sims: `NE`, `NI` | 10000, 2500 (connection probability 0.1) | observed |
| Fig. 1 set: `CE`, `J` | 4000, 0.2 mV | observed |
| Derived `nuthr`, Fig. 8 set | 10 Hz (`20 / (0.1*1000*20 ms)`) | inferred |
| Integrator, dt, duration, seeds | not stated in the sections read | unknown |

## 3. Claims per target figure (observed, Results pp. 189-197)

Fig. 8, four `(g, nuext/nuthr)` points (caption and §6, pp. 196-197; global activity in 0.1 ms bins, rasters of 50 random cells):

- A, SR `(3, 2)`: almost fully synchronized, regularly firing at high rates (no numeric rate stated).
- B, SI fast `(6, 4)`: global ~180 Hz sim (190 theory); single cells irregular at 60.7 Hz sim (55.8 theory).
- C, AI `(5, 2)`: stationary global activity, irregular cells at 37.7 Hz sim (38.0 theory).
- D, SI slow `(4.5, 0.9)`: global 22 Hz sim (29 theory); single cells irregular at 5.5 Hz sim (6.5 theory).

Fig. 1, stationary branches (p. 189): CV of the ISI ~0 for `g < 4` (cells near refractory saturation); jumps above 2 near the balanced point `g = 4` with the rate collapse; declines after, toward 1 at very low rates; whole low-activity branch CV near or above 1.

Fig. 2, phase diagrams (§5.1, p. 192): AR stable for `g < 4`, AI stable for `g > 4`, bounded by Hopf lines; SR when excitation dominates, SI when inhibition dominates; small triangular quasiperiodic region between SR and SI; quiescent state stable below the short-dashed saddle-node line; near `g = 4` and `nuext > nuthr`, high-frequency `~k/D` instabilities except when `D/taurp` is near an integer (the `D = taurp = 2 ms` diagram shifts these lines to lower `g`).

Fig. 3, oscillation frequencies (p. 193): fast branch near `1/4D`, i.e. 167, 125, 83 Hz for `D = 1.5, 2, 3 ms` (`g = 8, 6, 5`); slow branch near 20 Hz at low `nuext`, set by the membrane constant; large `D` plus large `g` merges slow into fast continuously.

Finite-size note (§6, p. 196): simulated AI global activity is not quite stationary and SI oscillations are somewhat irregular; the paper attributes both to finite `N`.

## 4. Data, code and licence

- Paper licence: (c) 2000 Kluwer Academic Publishers (observed, journal p. 183); subscription journal, author's copy freely posted on the author's own page with no explicit licence statement observed.
- Authors' code location and licence: unknown (no code statement found in the article).
- Data location and licence: none (theory plus simulation paper; no recordings).
- Cross-check only: NEST `brunel_delta` example documentation (reference [1] = this paper; parameters `dt 0.1 ms`, `delay 1.5 ms`, `g 5.0`, `eta 2.0`, `epsilon 0.1`, `order 2500`, `tauMem 20 ms`, `theta 20 mV`, `t_ref 2.0 ms`, `J 0.1 mV`), read as data, never run.

## 5. Gaps (each item observed or inferred; jaxfne paths checked in this tree)

Have (observed): LIF emitter family (`jaxfne/emitters_lif.py`, construct wiring `jaxfne/_construct_lif.py`); per-edge delays (`resolve_edge_delay_steps`, `edge_delay_steps_from_ms` in `jaxfne/emitters.py`); external Poisson drive (`Simulation(poisson_drive=...)`, `jaxfne/_config.py`); sparse random E/I wiring (`compile_connection_rules`); regime, rate and spectral estimators (`jaxfne/analysis/metrics.py`: `activity_regime`, `fano_factor`, spectral peak).

- G1 (inferred): plain Brunel LIF (absolute reset to `Vr`, hard refractory, no adaptation) against the core LIF kernel, which carries PCL refractory/adaptation terms (`jaxfne/emitters_lif.py`); P2 checks whether the extras zero out exactly.
- G2 (inferred): exact fixed indegree (every cell exactly `CE` excitatory plus `CI` inhibitory inputs); core random wiring exists, the exact rule needs a P2 check.
- G3 (inferred): one fixed delay `D` on every synapse including external drive; the core supports per-edge delays (a superset), so uniform-`D` is a P2 configuration question.
- G4 (observed): simulation numerics (integrator, dt, duration, seeds) are unstated in the paper; P2 declares them as deviations with a convergence check.
- G5 (inferred): a 12,500-cell sparse network through the public API is a P2 scale question, not a claim.
- G6 (observed): authors' code location unknown, so the Fig. 8 numbers rest on the paper text plus the NEST cross-check until the pilot reproduces them.
