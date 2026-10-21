# potjans2014 port specification (P2)

Paper: Potjans TC, Diesmann M (2014), Cereb Cortex 24(3):785-806,
doi:10.1093/cercor/bhs358. Source record: `SOURCE_P1.md`. Every paper value
below is observed there (§2-§3); every core behaviour is observed in this
tree; probe numbers are observed from the P2 probes in this run (N=800,
seed 0, dt=0.5 ms). Status per row: exact (port equals paper), deviation
(port differs, declared here), gap (paper needs what no public path offers;
settled by a test in `test_potjans2014_api.py`).

Lane route (checklist E1): `NeuronalTensor` +
`RuntimeConfiguration(emitter="lif")` -> `construct(tensor, runtime)` ->
`Model.simulate(Simulation(...))` (pattern:
`brunel2000_api.py:154-175`). Estimators from `jaxfne.analysis`
(checklist E3). `Configuration.set_emitter(family="lif")` is not the route;
the tensor route is the only lif path (as in brunel lane).

## Model form (checklist E5)

- A (tensor): one area `mc`, four layers `L23/L4/L5/L6`, eight
  `(layer, E/PV)` populations; default poses, no spatial extent. The paper
  network is unstructured (random, no geometry), so declared-per-rule
  delays with no position dependence are exact.
- x (input): recurrent delta arrivals plus one uniform Poisson channel.
- h (state): membrane voltages V (mV); no adaptation, no conductance
  state. Paper Eq: `tau_m dV/dt = -(V - E_L) + R_m * I_syn`
  (`SOURCE_P1.md` §1).
- y (readout): spike rasters; per-population `activity_regime`
  (rate_hz, cv_isi) and `wang_buzsaki_kappa`
  (`jaxfne/analysis/metrics.py:241-697`).
- Geometry: one area, layers carry default `Geometry3D`; no term reads
  positions.

## Per-element rows

| # | Element (paper) | Core path (file:line) | Status |
|---|---|---|---|
| 1 | 8 populations, N per Table 5 (`SOURCE_P1.md` §2) | `NeuronType.make("E"/"PV", emitter="lif", ...)` per layer with paper fractions; `L23/L4/L5/L6` layers in one area (`potjans2014_api.py:88-147`); fractions split counts exactly (`jaxfne/neuronal_tensor.py`, `_counts_from_fractions` idiom as in brunel lane) | exact (probe: 214/61/227/57/50/11/149/31 at N=800) |
| 2 | LIF membrane tau_m=10 ms (`SOURCE_P1.md` §2) | `emitter_params={tau_m_ms:10, ...}` per triple, applied in `jaxfne/_construct_lif.py:62-141`; decay `V*exp(-dt/tau)` in `jaxfne/emitters_lif.py:321` | exact (realized columns pinned by `test_lif_params_exact`) |
| 3 | Fixed threshold theta=-50 mV, reset V_reset=-65 mV, clamp tau_ref=2 ms (`SOURCE_P1.md` §2) | Reset to `v_reset_mV` exact (`jaxfne/emitters_lif.py:364`); hard-refractory window `t_ref_ms=2.0` holds V at reset and blocks spikes for `round(2.0/dt)` steps (`jaxfne/emitters_lif.py:332-339`); lane declares `t_ref_ms=2.0` | exact gaps after D-VSHIFT (gap 15 mV pinned; refractory pinned by brunel `test_hard_refractory` on the same kernel) |
| 4 | Absolute voltage origin (E_L, hence theta/V_reset absolute) | Kernel leaks toward 0 mV (`jaxfne/emitters_lif.py:321`); E_L and R_m unstated in the paper (`SOURCE_P1.md` §1) | deviation D-VSHIFT (declared: origin at V_reset, theta=+15 mV, reset=0 mV; all stated gaps kept) |
| 5 | Strictness of threshold crossing | Core spikes on strict `V > v_thresh` (`jaxfne/emitters_lif.py:326`) | deviation (measure-zero, named, as in brunel row 4) |
| 6 | No voltage floor stated | `v_min_mV=-1000.0`: floor never binds in range | deviation (declared; inert unless V leaves range, checked per run by finite-V assert) |
| 7 | No adaptation, no relative-refractory tail | `eta_rp_mV=0.0, delta_sra_mV=0.0` (PCL defaults carry `eta_rp=10, tau_rp=5`, `jaxfne/emitters_lif.py:55-65`) | exact after zeroing (pinned by `test_lif_params_exact`) |
| 8 | Random wiring, binomial in/out-degree, K fixed per projection, multapses allowed (`SOURCE_P1.md` §1) | `sampling="density"` with `probability=p` per rule: realized unique count exactly `round(p*n_pre*n_post)` (`jaxfne/connectivity.py:801-812`); every sampler draws distinct pairs only (`jaxfne/connectivity.py:501-546`) | deviation D-DENSITY (declared stand-in: fixed-K unique pairs) + gap G-MULT (no with-replacement sampler; `test_gap_multapse_sampler`) |
| 9 | Synaptic strengths Gaussian w +- delta_w, sign-preserving, L4e->L2/3e doubled (`SOURCE_P1.md` §2) | One constant per rule: realized edge `|w_mech*g_mech|/sqrt(total_n)` mV, sign from mechanism vocabulary (`jaxfne/neuronal_tensor.py:2581-2592,2595-2628`); lane sets `w_mech = mag*sqrt(N)` with mag 0.1 (E), doubled 0.2 on L4e->L2/3e, 0.4 (I, g=4) | deviation D-WCONST (constant means) + gap G-WDIST (no per-rule spread; `test_gap_weight_distribution`); doubling exact-as-constant (probe: all L4e->L2/3e edges 0.2 mV) |
| 10 | Delays Gaussian d_e/d_i +- delta_d, sign-preserving, snapped to h multiples (`SOURCE_P1.md` §2) | One constant per rule, realized by round-half-up (`jaxfne/connectivity.py:46-76`); dt=0.5 gives E 1.5 ms exactly 3 steps, I 0.8 ms rounds to 2 steps | deviation D-DCONST (constant means; I value declared 1.0 ms, grid-snapped) + gap G-DDIST (no per-rule spread; `test_gap_delay_distribution`) |
| 11 | Exponential synaptic currents tau_syn=0.5 ms (`SOURCE_P1.md` §1) | `lif_step` sums signed arrivals into V directly, no current state (`jaxfne/emitters_lif.py:321-325`) | gap G-TAUSYN (no synaptic filter on the lif route; `test_gap_tau_syn_filter`) |
| 12 | External input per-population k_ext trains at 8 Hz (`SOURCE_P1.md` §2) | `Simulation(poisson_drive={rate_hz:8.0, amplitude:0.1, counts:2000, seed})`: per step per cell one `binomial(counts, rate*dt/1000)` draw, the exact sum distribution of that many independent trains (`jaxfne/_signals.py:743-797`); only `target="all"` is implemented (`jaxfne/_signals.py:768-772`) | deviation D-EXT (uniform 2000-train stand-in; rate per train exact) + gap G-POPDRIVE (no per-population drive; `test_gap_per_population_drive`) |
| 13 | Thalamic Th population, 902 Poisson units, Th column of the map (`SOURCE_P1.md` §2) | No Poisson-unit cell type on the lif route; Th drive folded into the uniform external drive | deviation D-TH (declared: no Th cells; Thalamocortical projections unwired) |
| 14 | pA weight scale (w=87.8 pA, g=-4) with C_m=250 pF (`SOURCE_P1.md` §2) | LIF route carries mV arrivals directly; no C_m, no E_L, no pA->mV map | deviation D-SCALE (stand-in mV scale 0.1/0.4; paper pA scale not portable) |
| 15 | No autapses stated (paper silent; independent draws admit self pairs) | Self-edges dropped by default (`allow_self` unset) | deviation (declared; paper unknown, as in brunel row 17) |
| 16 | No background current stated beyond synaptic input | Undeclared triple drive defaults nonzero; lane declares `drive=0.0` on all triples (probe: realized 0.0, no-drive network silent after step 0) | exact after declaration (pinned by `test_zero_drive_silent`; V0=V_reset so no step-0 volley) |
| 17 | Synaptic time constant dT (NEST kernel detail) | `StaticParams.dT_ms` required by P-023 validation but inert on the lif route (probe pattern as in brunel row 14) | deviation (declared-but-inert; value 2.0 ms carried for inspection only) |
| 18 | Integrator, dt, duration, seeds (paper: NEST h=0.1 ms; V0 unstated) | Exact exponential decay per step (`jaxfne/emitters_lif.py:321`); dt=0.5 ms binary-exact; V0=rest=reset (silent start); seed 0 pilots only | deviation D-DT/D-V0 (declared; convergence check at P3) |
| 19 | Thalamic transient protocol (+15 Hz, 10 ms) and Fig. 10/11 claims (`SOURCE_P1.md` §5 G5) | Deferred: not P1 target figures, no port rows | out of scope (not a gap in the scored route) |
| 20 | Estimands C1/C4/C5: per-population rates, CV of ISIs, synchrony (Fig. 6, `SOURCE_P1.md` §3) | `activity_regime` rate_hz/cv_isi/label + `wang_buzsaki_kappa`, per population via `score_by_population` (all `jaxfne.analysis`, checklist E3; no lane-local estimator) | exact (estimator choice; values scored at P3) |

## Status counts

- exact: 8 (rows 1, 2, 3-gaps, 5, 7, 11-geometry, 16, 20; row 9-doubling as constant)
- declared deviations: 12 (D-VSHIFT, D-SCALE, D-DENSITY, D-WCONST,
  D-DCONST, D-EXT, D-TH, D-DT, D-V0, threshold strictness, v_min floor,
  dT carrier, no-autapse)
- gaps: 5 (G-MULT, G-TAUSYN, G-POPDRIVE, G-WDIST, G-DDIST), each with an
  `xfail(strict=True)` test naming the gap id and the smallest proposed
  public API (see OPEN in the lane report).

## Numerics declared

- Clock-driven, one `lif_step` per 0.5 ms bin; decay `exp(-dt/tau)` exact
  in float32; ring-buffer delayed arrivals; no sub-step spike
  interpolation; no synaptic filtering (G-TAUSYN).
- Pilot: N=800 (214/61/227/57/50/11/149/31), 55 density rules, 30708
  edges, T=1.0 s, seed 0, CPU, float32; outputs under
  `scratch/potjans2014_pilot/` (gitignored).
