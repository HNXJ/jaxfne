# potjans2014 port specification (P2)

Paper: Potjans TC, Diesmann M (2014), Cereb Cortex 24(3):785-806,
doi:10.1093/cercor/bhs358. Source record: `SOURCE_P1.md`. Every paper value
below is observed there (§2-§3); every core behaviour is observed in this
tree; probe numbers are observed from the first P2 probes (N=800, seed 0,
dt=0.5 ms; rows 1 and 16) and the P2 refresh pilot (scale 0.01 and 0.02,
seed 0, dt=0.1 ms; Numerics). Status per row: exact (port equals paper), deviation
(port differs, declared here), gap (paper needs what no public path offers;
settled by a test in `test_potjans2014_api.py`).

Lane route (checklist E1): `NeuronalTensor` +
`RuntimeConfiguration(emitter="lif")` -> `construct(tensor, runtime)` ->
`Model.simulate(Simulation(...))` (pattern:
`run` in `brunel2000_api.py`). Estimators from `jaxfne.analysis`
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
  (`activity_regime` in `jaxfne/analysis/metrics.py`; `wang_buzsaki_kappa` in `jaxfne/analysis/metrics.py`).
- Geometry: one area, layers carry default `Geometry3D`; no term reads
  positions.

## Per-element rows

| # | Element (paper) | Core path | Status |
|---|---|---|---|
| 1 | 8 populations, N per Table 5 (`SOURCE_P1.md` §2) | `NeuronType.make("E"/"PV", emitter="lif", ...)` per layer with paper fractions; `L23/L4/L5/L6` layers in one area (`build_spec` in `potjans2014_api.py`); fractions split counts exactly (`jaxfne/neuronal_tensor.py`, `_counts_from_fractions` idiom as in brunel lane) | exact (probe: 214/61/227/57/50/11/149/31 at N=800) |
| 2 | LIF membrane tau_m=10 ms (`SOURCE_P1.md` §2) | `emitter_params={tau_m_ms:10, ...}` per triple, applied in `apply_lif_triple_param_overrides` in `jaxfne/_construct_lif.py`; decay `V*exp(-dt/tau)` in `lif_step` in `jaxfne/emitters_lif.py` | exact (realized columns pinned by `test_lif_params_exact`) |
| 3 | Fixed threshold theta=-50 mV, reset V_reset=-65 mV, clamp tau_ref=2 ms (`SOURCE_P1.md` §2) | Reset to `v_reset_mV` exact (`lif_step` in `jaxfne/emitters_lif.py`); hard-refractory window `t_ref_ms=2.0` holds V at reset and blocks spikes for `round(2.0/dt)` steps (`lif_step` in `jaxfne/emitters_lif.py`); lane declares `t_ref_ms=2.0` | exact gaps after D-VSHIFT (gap 15 mV pinned; refractory pinned by brunel `test_hard_refractory` on the same kernel) |
| 4 | Absolute voltage origin (E_L, hence theta/V_reset absolute) | Kernel leaks toward 0 mV (`lif_step` in `jaxfne/emitters_lif.py`); E_L and R_m unstated in the paper (`SOURCE_P1.md` §1) | deviation D-VSHIFT (declared: origin at V_reset, theta=+15 mV, reset=0 mV; all stated gaps kept) |
| 5 | Strictness of threshold crossing | Core spikes on strict `V > v_thresh` (`lif_step` in `jaxfne/emitters_lif.py`) | deviation (measure-zero, named, as in brunel row 4) |
| 6 | No voltage floor stated | `v_min_mV=-1000.0`: floor never binds in range | deviation (declared; inert unless V leaves range, checked per run by finite-V assert) |
| 7 | No adaptation, no relative-refractory tail | `eta_rp_mV=0.0, delta_sra_mV=0.0` (PCL defaults carry `eta_rp=10, tau_rp=5`, `LIF_CELL_TYPE_DEFAULTS` in `jaxfne/emitters_lif.py`) | exact after zeroing (pinned by `test_lif_params_exact`) |
| 8 | Random wiring, binomial in/out-degree, K fixed per projection, multapses allowed (`SOURCE_P1.md` §1) | `sampling="multinomial"` with `in_degree=k` per AreaConnection: K = round_half_up(k * n_post) synapses drawn with replacement (`_candidate_pairs_multinomial` in `jaxfne/connectivity.py`, `in_degree` path at its call site). k = K_full / N_post_full with K_full = ln(1-C)/ln(1-1/(N_pre N_post)) at the Table 5 C (`mean_in_degree` in `potjans2014_api.py`), held at every scale (`prereg/P3_v2.md` Route). Self-edges excluded by default (row 15) | exact on the paper's independent-draw wiring (§1). K_full is the paper's synapse count for connection probability C under independent draws (Materials and Methods, wiring paragraph); SOURCE_P1 records the rule (§1), not the formula. Realized K per projection at scale 0.01: `test_in_degree_exact_totals` |
| 9 | Synaptic strengths Gaussian w +- delta_w, sign-preserving, L4e->L2/3e doubled (`SOURCE_P1.md` §2) | One Gaussian per rule through the core's spread: `PlasticParams(w_mech = mean_mV * sqrt(total_n), w_std = SD_mV)` (`build_spec`). The core divides the mean by sqrt(total_n) (`_connection_edge_weight` in `jaxfne/neuronal_tensor.py`) but takes `w_std` raw, so the spread is passed in mV as realized. Means: 87.8 pA (E), 4 x 87.8 pA (I, g = -4), 2 x 87.8 pA (L4e->L2/3e), each times the row 14 factor. SD proportional to the mean, delta_w / w = 8.8 / 87.8 on every rule: 8.8 pA (E), 35.2 pA (I), 17.6 pA (L4e->L2/3e) (reading: g and the doubling scale the amplitude distribution, §1 "larger by factor g"). `|.|` fold at zero (`connectivity.py`), 10 SD from the mean | deviation D-WSPREAD (spread reading declared; §2 states delta_w for E only). Realized mean and SD per rule: `test_weights_per_rule` (5 SE) |
| 10 | Delays Gaussian d_e/d_i +- delta_d, sign-preserving, snapped to h multiples (`SOURCE_P1.md` §2) | One Gaussian per rule through `delay_std_ms`: E 1.5 +- 0.75 ms, I 0.8 +- 0.4 ms (`build_spec`). Each draw is folded at zero (absolute value in `compile_connection_rules` in `jaxfne/connectivity.py`), snapped to the dt = 0.1 ms grid (round half up), and a positive draw inside one step takes one step (`to_edge_list`, spread branch; the constant path still refuses). Realized SD is the folded-Gaussian SD, below the nominal SD | deviation D-DSPREAD: the fold and the one-step floor are the core's rule; the paper's truncation rule is not stated (G2). Realized mean within 4 SE + 0.01 ms and SD equal to the folded SD: `test_delays_per_rule` |
| 11 | Exponential synaptic currents tau_syn=0.5 ms (`SOURCE_P1.md` §1) | `tau_syn_ms = 0.5` on every neuron (`_LIF_POTJANS`): a first-order current filter g <- g exp(-dt/tau_syn) + arrivals, with g entering V each step (`lif_step` input in `jaxfne/emitters_lif.py`; G-TAUSYN closed in the core). One arrival of weight a deposits a / (1 - exp(-dt/tau_syn)) in total, so a = row 14 gives w tau_syn / C_m | exact (`test_tau_syn_realized`; the discrete step is row 18's integrator) |
| 12 | External input per-population k_ext trains at 8 Hz (`SOURCE_P1.md` §2) | `Simulation(poisson_drive=[...])`, one entry per population (8): target the population's (area, layer, cell_type) triple, `counts` = the Table 5 reference k_ext, `rate_hz` 8, `amplitude` = w tau_syn / C_m = 0.1756 mV (`build_drive` in `potjans2014_api.py`). Per step per cell one `binomial(counts, rate*dt/1000)` draw, the exact sum of that many independent trains (`_make_poisson_drive` in `jaxfne/_signals.py`); triple targets resolve through `resolve_poisson_drive_target`; the list form is realized on the lif route (`jaxfne/inputs.py`). G-POPDRIVE closed in the core | deviation D-EXT: the core adds each arrival's amplitude unfiltered in its own step, so the drive has no synaptic filter (declared; the total per arrival is the full deposit, as row 14). Tests: `test_external_drive_per_population`, `test_drive_entry_changes_only_its_population` |
| 13 | Thalamic Th population, 902 Poisson units, Th column of the map (`SOURCE_P1.md` §2) | No Poisson-unit cell type on the lif route; no Th cells, no Th projections, and no Th drive: the drive of row 12 is the k_ext of the eight cortical populations only | deviation D-TH (declared: Th not realized; the nine Th probabilities are unwired and the Th input is absent) |
| 14 | pA weight scale (w=87.8 pA, g=-4) with C_m=250 pF (`SOURCE_P1.md` §2) | pA->mV by the lane (the core takes mV arrivals): per-step edge increment (w/C_m) tau_syn (1 - exp(-dt/tau_syn)) mV, so one PSC deposits w tau_syn / C_m (0.1756 mV at 87.8 pA; `_step_mv_per_pa`, `_deposit_mv_per_pa` in `potjans2014_api.py`). No E_L or R_m is used (the leak is the kernel's) | exact (C_m, tau_syn and the total deposit per arrival; the origin shift is D-VSHIFT, row 4) |
| 15 | No autapses stated (paper silent; independent draws admit self pairs) | Self-edges dropped by default (`allow_self` unset) | deviation (declared; paper unknown, as in brunel row 17) |
| 16 | No background current stated beyond synaptic input | Undeclared triple drive defaults nonzero; lane declares `drive=0.0` on all triples (probe: realized 0.0, no-drive network silent after step 0) | exact after declaration (pinned by `test_zero_drive_silent`; V0=V_reset so no step-0 volley) |
| 17 | Synaptic time constant dT (NEST kernel detail) | `StaticParams.dT_ms` required by P-023 validation but inert on the lif route (probe pattern as in brunel row 14) | deviation (declared-but-inert; value 2.0 ms carried for inspection only) |
| 18 | Integrator, dt, duration, seeds (paper: NEST h=0.1 ms; V0 unstated) | Exact exponential decay per step (`lif_step` in `jaxfne/emitters_lif.py`); dt = 0.1 ms, the paper's h; V0=rest=reset (silent start); seed 0 pilots only | deviation D-V0 (V0 unstated; declared); dt equals h |
| 19 | Thalamic transient protocol (+15 Hz, 10 ms) and Fig. 10/11 claims (`SOURCE_P1.md` §5 G5) | Deferred: not P1 target figures, no port rows | out of scope (not a gap in the scored route) |
| 20 | Estimands C1/C4/C5: per-population rates, CV of ISIs, synchrony (Fig. 6, `SOURCE_P1.md` §3) | `activity_regime` rate_hz/cv_isi/label + `wang_buzsaki_kappa`, per population via `score_by_population` (all `jaxfne.analysis`, checklist E3; no lane-local estimator) | exact (estimator choice; values scored at P3) |

## Status counts

Counted from the status column of rows 1-20 after the P2 refresh:

- exact: 9 (rows 1, 2, 3, 7, 8, 11, 14, 16, 20).
- deviation: 10 (rows 4, 5, 6, 9, 10, 12, 13, 15, 17, 18).
- out of scope: 1 (row 19).
- gap: 0. The five core gaps stay closed: G-MULT b4d4dc5c, G-TAUSYN 5a8d4ad4,
  G-POPDRIVE 47c8076b, G-WDIST + G-DDIST 9dc6afd3; G-MULT review fixes b2c2b117.
  The lane uses them from the P2 refresh.
- Retired stand-ins: D-DENSITY (row 8), D-WCONST and D-DCONST (rows 9-10),
  D-SCALE (row 14), the uniform 2000-train drive (row 12), the 1.0 ms I delay (row 10).

## Numerics declared

- Clock-driven, one `lif_step` per 0.1 ms step; decay `exp(-dt/tau)` in
  float32; first-order synaptic filter tau_syn 0.5 ms (row 11); ring-buffer
  delayed arrivals; no sub-step spike interpolation.
- P2 refresh pilot (unscored, seed 0, T 1 s, CPU float32, in-degree held;
  receipts `C:/workspace/jaxfne_receipts/potjans2014/pilot_p2/`): scale 0.01,
  772 cells, 2,999,076 synapses, construct 57 s, simulate 49 s, peak 2.5 GB;
  scale 0.02, 1,544 cells, 5,998,183 synapses, construct 86 s, simulate 102 s,
  peak 4.0 GB. Ran with an 8.8 pA spread on every rule (before the row 9 reading).
