# brunel2000 port specification (P2)

Paper: Brunel N (2000), J Comput Neurosci 8:183-208, doi:10.1023/A:1008925309027.
Source record: `SOURCE_P1.md`. Every paper value below is observed there (§2-§4);
every core behaviour is observed in this tree; probe numbers are observed from
the P2 probes in this run (N=50, seed 0, dt=0.5 ms). Status per row: exact
(port equals paper), deviation (port differs, declared here), gap (paper needs
what no public path offers; settled by a test in `test_brunel2000_api.py`).

Lane route (checklist E1): `NeuronalTensor` + `RuntimeConfiguration(emitter="lif")`
-> `construct(tensor, runtime)` -> `Model.simulate(Simulation(...))` (pattern:
`tests/test_inputs_xsource_t24.py:92-99,217-231`). Estimators from
`jaxfne.analysis` (checklist E3). `Configuration.set_emitter(family="lif")`
raises; the tensor route is the only lif path
(`tests/test_emitter_family_validation_v0330.py`).

## Model form (checklist E5)

- A (state): membrane voltages V (mV), spikes; no adaptation, no conductance
  state. Paper Eq. 1: `tau dVi/dt = -Vi + R Ii(t)` (`SOURCE_P1.md` §1).
- x (input): recurrent delta arrivals plus one external Poisson channel per cell.
- h (parameters): J, g, nuext, CE, CI, D below; fixed weights (no plasticity).
- y (readout): spike rasters; `activity_regime` (rate_hz, cv_isi, label) and
  `wang_buzsaki_kappa` (`jaxfne/analysis/metrics.py:482-573,621-697`).
- Geometry: one area `net`, one layer `L0`, no spatial extent; delays are
  declared per rule, not derived from positions. The paper network is
  unstructured (random, no geometry), so this is exact.

## Per-element rows

| # | Element (paper) | Core path (file:line) | Status |
|---|---|---|---|
| 1 | LIF membrane, tau=20 ms, theta=20 mV, Vr=10 mV (`SOURCE_P1.md` §2) | `NeuronType.make("E"/"PV", emitter="lif", emitter_params={tau_m_ms:20, v_thresh_mV:20, v_reset_mV:10, ...})`; fields in `jaxfne/emitters_lif.py:67-70`, applied per triple in `jaxfne/_construct_lif.py:62-141` | exact (probe: realized columns equal the declaration) |
| 2 | Rest 0 mV, V0 unspecified (paper states no V0; cells start below threshold) | V0 = 0.0, silent with no drive (probe observed) | deviation (declared: V0=0; paper unknown, `SOURCE_P1.md` §2) |
| 3 | Absolute reset to Vr, hard refractory taurp=2 ms (`SOURCE_P1.md` §1) | Reset to `v_reset_mV` exact (`jaxfne/emitters_lif.py:327`); no hard-refractory term exists: `eta_rp_mV=0, delta_sra_mV=0` zeroes the PCL extras, and `lif_step` (`jaxfne/emitters_lif.py:288-330`) then re-fires the next step under strong drive | gap G1 (extras zero out exactly, pinned by `test_lif_extras_zero_closed_form`; missing hard refractory pinned by `test_hard_refractory_gap`, xfail) |
| 4 | Strictness of threshold crossing ("reaches theta") | Core spikes on strict `V > v_thresh` (`jaxfne/emitters_lif.py:263-264,295`) | deviation (measure-zero; named, not tested) |
| 5 | No voltage floor | `v_min_mV=-1000.0`: floor never binds in range (probe: V within [0, ~40] mV at pilot drive) | deviation (declared; floor value inert unless V leaves range, checked per run by finite-V assert) |
| 6 | No adaptation, no relative-refractory tail | `eta_rp_mV=0.0, delta_sra_mV=0.0` (PCL defaults carry `eta_rp=10, tau_rp=5`, `jaxfne/emitters_lif.py:54-65`) | exact after zeroing (pinned by `test_lif_extras_zero_closed_form`) |
| 7 | Recurrent weights J (E), -gJ (I), delta currents (`SOURCE_P1.md` §1) | Realized edge weight `|w_mech*g_mech|/sqrt(total_n)` mV, sign from mechanism vocabulary (`jaxfne/neuronal_tensor.py:2464-2475,2478-2511`); lane sets `w_mech = J*sqrt(N)` (E), `g*J*sqrt(N)` (I); edges execute signed mV arrivals (`jaxfne/emitters_lif.py:1404-1424`) | exact (probe: realized E +0.1, I -0.5 mV at J=0.1, g=5) |
| 8 | Fixed indegree CE/CI per cell (`SOURCE_P1.md` §2: 1000/250) | Density sampler: exact total `round(p*n_pre*n_post)` per rule (`jaxfne/connectivity.py:290-344`), binomial per-cell indegree (probe: range 1..10 at mean 5.0); no exact-fixed-indegree sampler exists (`density`, `bernoulli`, `capped_bernoulli` only, `jaxfne/connectivity.py:727-733`; the cap binds above, never fills below, `jaxfne/connectivity.py:432-472`) | gap G2 (exact totals pinned by `test_density_exact_totals`; exact per-cell indegree pinned by `test_exact_indegree_gap`, xfail) |
| 9 | One fixed delay D=1.5 ms on every synapse (`SOURCE_P1.md` §2) | `delay_ms=1.5` per rule, realized by round-half-up (`jaxfne/connectivity.py:45-75`); dt=0.5 gives exactly 3 steps; edges refuse delay < 1 step (`jaxfne/emitters_lif.py:527-528`) | exact for recurrent edges (pinned by `test_uniform_delay_steps`; all 246 probe edges delay 3) |
| 10 | External drive delayed by D like recurrent input (Eq. 2 covers all synapses) | Current-route poisson has no delay key; `delay_ms` refused (`jaxfne/inputs.py:357-382`); arrivals apply same-step (`jaxfne/_signals.py:742-776`) | gap G3-external, deviation (pinned by `test_external_drive_same_step_gap`, xfail; uniform recurrent D exact per row 9) |
| 11 | External drive: Cext independent Poisson trains per cell, rate nuext each, amplitude J (`SOURCE_P1.md` §1) | `Simulation(poisson_drive={rate_hz: Cext*nuext, amplitude: J, seed})`: independent per-neuron Bernoulli trials, p = rate*dt/1000, at most one event per step per cell (`jaxfne/_signals.py:742-776`); `target` must be `"all"` | deviation (Bernoulli-for-Poisson negligible at p << 1; one-event cap binds at p -> 1; rate cap below) |
| 12 | Full-scale drive rates (up to Cext*nuext = 40 kHz at point B) | `prob > 1` refused (`jaxfne/_signals.py:769-773`): at dt=0.5 only rates <= 2000 Hz run; point B (4000 Hz at N=1250, 40 kHz at full scale) is inexpressible, points A/C (2000 Hz at N=1250) only as degenerate always-on drive | gap G3-scale (pilot runs point D, p=0.45; A-C deferred to P3 protocol decision, recorded in `CALIBRATION.md`) |
| 13 | No constant background current (rest follows input only) | Undeclared triple drive defaults to 5.0 (E) / 3.0 mV per step (probe observed); lane declares `drive=0.0` on both types (probe: realized 0.0, silent no-drive run) | exact after declaration (pinned by `test_zero_drive_silent`) |
| 14 | Synaptic time constant (delta currents: none) | `StaticParams.dT_ms` required by P-023 validation but inert on the lif route (probe: dT 2.0 vs 8.0 bit-identical edges and weights) | deviation (declared-but-inert; value 2.0 ms carried for inspection only) |
| 15 | Integrator, dt, duration, seeds (paper: unknown, `SOURCE_P1.md` §2) | Exact exponential decay per step (`jaxfne/emitters_lif.py:290`); dt=0.5 ms binary-exact; seed 0 pilots only | deviation G4 (declared; convergence check at P3) |
| 16 | Population sizes NE/NI = 0.8N/0.2N, sparse p=0.1 (`SOURCE_P1.md` §2) | Fractions 0.8/0.2 split counts exactly (probe: 40/10 at N=50); sparse wiring via `AreaConnection(probability=0.1)` (`jaxfne/neuronal_tensor.py:685-688`); `InterConnection` has no probability (all-to-all, `jaxfne/neuronal_tensor.py:2761-2764`) so the lane uses same-area `AreaConnection`s | exact |
| 17 | No autapses stated (paper silent) | Self-edges dropped by default (`allow_self` None) | deviation (declared; paper unknown) |
| 18 | Estimands: single-cell rates, CV, global frequencies (Fig. 8, `SOURCE_P1.md` §3) | `activity_regime` rate_hz/cv_isi/label + `wang_buzsaki_kappa`, Welch peak via `spectral_peak_excess` (all `jaxfne.analysis`, checklist E3; no lane-local estimator) | exact (estimator choice; values scored at P3) |

## Numerics declared (G4)

- Clock-driven, one `lif_step` per 0.5 ms bin; decay `exp(-dt/tau)` exact in
  float32; ring-buffer delayed arrivals; no sub-step spike interpolation.
- Pilot: N=1250 (NE=1000, NI=250, CE~=100, CI~=25 at p=0.1), T=0.5 s, seed 0,
  CPU, float32; outputs under `scratch/brunel2000_pilot/` (gitignored).
- Scale decision (P2): full N=12500 projects to ~15.6M edges; drive-rate cap
  (row 12) already bars points A-C at full scale on this route, so no full
  run precedes the P3 protocol decision.
