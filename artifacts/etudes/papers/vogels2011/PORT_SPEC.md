# vogels2011 port specification (P2)

Paper: Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011), Science
334(6062):1569-1573, doi:10.1126/science.1211095. Source record:
`SOURCE_P1.md`. Every paper value below is observed there (§1-§2, authors'
code symbols `inhib_plasticity.m` / `isp_brian.py` read as text); every core
behaviour is observed in this tree; probe numbers are observed from the P2
probes in this run (N=20, seed 0, dt=0.1 ms). Status per row: exact (port
equals paper), deviation (port differs, declared here), gap (paper needs
what no public path offers; settled by a test in
`test_vogels2011_api.py`).

Lane route (checklist E1): `NeuronalTensor` (emitter `"adex"`) +
`RuntimeConfiguration(emitter="adex")` -> `construct(tensor, runtime)` ->
`Model.simulate(Simulation(...))`. The conductance route is the only core
path with reversal-potential synapses: the LIF route is current-based and
carries reversals as metadata only (`jaxfne/emitters.py:4462-4464`;
`jaxfne/emitters_lif.py:1`). Estimators from `jaxfne.analysis`
(checklist E3): `activity_regime` (rate_hz/cv_isi/kappa/label,
`jaxfne/analysis/metrics.py:621-697`, kappa at `:482`).

## Model form (checklist E5)

- A (state): membrane voltages V (mV), adaptation currents w, conductances
  g_ampa/g_gaba (nS), refractory timers; spikes. Paper §1: conductance LIF,
  `dv/dt = (-gl*(v-el) - (g_ampa*v + g_gaba*(v-er))*w + bgcurrent)/memc`
  (`SOURCE_P1.md` §1).
- x (input): recurrent conductance arrivals plus constant 200 pA background
  current per cell; no external Poisson drive on the Fig. 4 route.
- h (parameters): C=200 pF, gL=10 nS, EL=-60 mV, VT=-50 mV, E_rev 0/-80 mV,
  tau_ampa=5 ms, tau_gaba=10 ms, Tref=5 ms, weights below; I-to-E plastic in
  the paper, fixed in the port (gap G-IE).
- y (readout): spike rasters; `activity_regime` (rate_hz, cv_isi, kappa,
  label). Target equilibrium: excitatory rate rho0 = 3 Hz (G-source: the
  paper's stated realized numbers are unreachable, `SOURCE_P1.md` §5).
- Geometry: one area `net`, one layer `L0`, no spatial extent; delays are
  declared per rule, not derived from positions. The paper network is
  unstructured (random, epsilon-sparse), so this is exact.

## Per-element rows

| # | Element (paper) | Core path (file:line) | Status |
|---|---|---|---|
| 1 | Leak + threshold: gl=10 nS, memc=200 pF (tau 20 ms), el=-60 mV, vt=-50 mV (`SOURCE_P1.md` §2) | `NeuronType.make("E"/"PV", emitter="adex", emitter_params={C_pF:200, gL_nS:10, EL_mV:-60, VT_mV:-50, ...})`; AdEx defaults in `jaxfne/emitters_adex.py:29-42`, overrides applied per triple in `jaxfne/_construct_adex.py:364-429` | exact (probe: realized C/gL/EL/VT equal the paper on every cell) |
| 2 | Reset to el, refractory 5 ms (`SOURCE_P1.md` §1-§2) | `Vreset_mV=-60.0` override, `Tref_ms=5.0` (already the Table-1 default, `jaxfne/emitters_adex.py:29-42`) | exact (probe: realized Vreset/Tref equal the paper) |
| 3 | No adaptation in the paper cells (plain LIF) | `a_nS=0.0, b_pA=0.0` overrides on both types (E Table-1 defaults carry `a=4, b=50`; PV defaults already 0, `jaxfne/emitters_adex.py:29-42`) | exact after zeroing (probe: realized a/b are 0) |
| 4 | No exponential spike-sharpness term (LIF, not AdEx) | `Delta_mV` stays at Table-1 defaults (2.0 E / 0.5 PV): no zero is expressible on the conductance route | deviation D-CELL (declared; the AdEx `exp((V-VT)/Delta)` current near threshold has no paper counterpart; scored at P3 against the authors' code) |
| 5 | V0 unstated (cells start below threshold) | V0 = EL (kernel rest, `jaxfne/emitters_adex.py:738`); no V0 declared | deviation (declared; paper unknown, `SOURCE_P1.md` §2) |
| 6 | Conductance synapses tau_ampa=5 ms, tau_gaba=10 ms, EAMPA=0 mV, EGABA/er=-80 mV (`SOURCE_P1.md` §2) | `StaticParams(dT_ms=5.0/10.0, reversal_potentials_mV={...})`; reversal mandatory on the adex route (`jaxfne/neuronal_tensor.py:2902-2909`); kernel `I = sum_r g[:,r]*(E_r-V)` (`jaxfne/emitters_adex.py:487-504`) | exact (probe: realized receptor tau (5.0, 10.0), reversal (0.0, -80.0)) |
| 7 | Fixed weights E-to-all 0.3, I-to-I 3 (nS, `SOURCE_P1.md` §2) | Realized weight `polarity * w_mech*g_mech/sqrt(total_n)` nS (`jaxfne/neuronal_tensor.py:2464-2475,2932-2933`); lane sets `w_mech = 0.3*sqrt(N)` (E), `3.0*sqrt(N)` (I-to-I) | exact (probe: realized E +0.3, I -3.0 nS) |
| 8 | I-to-E plastic: init 1e-10, pairing potentiation + per-pre-spike depression (`eta=1e-2`, `alpha=2*rho0*tau_stdp`), cap gmax=100 (`SOURCE_P1.md` §1-§2) | Core STDP kernel updates only excitatory-presynaptic weights (`update_mask = exc_mask[None, :]` in `stdp_weight_update`, `jaxfne/plasticity.py:93`); no I-plastic rule in the HDP registry (`register_hdp_rule`, `jaxfne/hdp_rule.py:131`; known targets `edge_weight/drive_bias` only, `:60`) | gap G-IE (pinned by `test_istdp_ie_gap`, xfail; init/growth/cap ride the same missing rule) |
| 9 | I-to-E weight in the port (plasticity absent) | Fixed `w_mech = 3.0*sqrt(N)` (the I-to-I value), sign negative from the PV source (`jaxfne/neuronal_tensor.py:2932`) | deviation D-IE (declared stand-in; keeps inhibition stabilizing for the smoke pilot, scored at P3) |
| 10 | Sparse epsilon=0.02, NE=8000, NI=2000, NI=NE/4 (`SOURCE_P1.md` §2) | Density sampler: exact total `round(p*n_pre*n_post)` per rule (`jaxfne/connectivity.py:290-344`); fractions 0.8/0.2 split counts exactly | exact (probe: realized totals equal round(0.02*n_pre*n_post) per rule) |
| 11 | Fixed indegree per cell | Brian random sparse wiring is binomial per cell, matching the density sampler; no exact-indegree claim in the sources read | exact (sampler matches the authors' code form) |
| 12 | Delay: none stated in the sources read (Brian default) | Adex route refuses a rule without a delay (`jaxfne/neuronal_tensor.py:2943-2951`); lane declares `delay_ms=0.1` per rule, realized by the route as exactly 1 step at dt=0.1 | deviation D-DELAY (declared assumption; paper unknown) |
| 13 | Background current bgcurrent=200 pA on every cell (`SOURCE_P1.md` §2) | Per-triple `drive=200.0` bridged as `tensor_triple_drive` pA (`jaxfne/_construct_adex.py:306-356`), realized as `I_ext_pA` (`jaxfne/_construct_adex.py:699-700`) | exact (probe: realized 200.0 pA on every cell; pilot spikes) |
| 14 | Single-cell setup: 8 tuned input channels, per-group rates, 1000 trains (`SOURCE_P1.md` §2) | One `Simulation` carries one Poisson rate for all targets (`compile_poisson_sources`, `jaxfne/inputs.py:385-458`: single `rate_hz`, shared by `target_cell_types`) | gap G-SC (pinned by `test_per_channel_rates_gap`, xfail; differentiation additionally blocked by G-IE) |
| 15 | Integrator, dt, duration (paper: dt=0.1 ms coded; solver/order unstated) | Conductance decay `exp(-dt/tau)` (`decay="exp"`); dt=0.1 ms exact; pilot T=1.0 s, seed 0 (checklist P2) | deviation (declared; convergence check at P3) |
| 16 | Population sizes NE/NI = 0.8N/0.2N | Fractions split counts exactly (probe: 16/4 at N=20); sparse wiring via same-area `AreaConnection(probability=0.02)` (`jaxfne/neuronal_tensor.py:685-688`); `InterConnection` has no probability (all-to-all) so the lane uses `AreaConnection`s | exact |
| 17 | No autapses stated | Self-edges dropped by default (`allow_self` None) | deviation (declared; paper unknown) |
| 18 | Estimands: excitatory rate to rho0=3 Hz; Before (0.8-1.0 s) vs After (last 0.2 s) rasters (`SOURCE_P1.md` §2-§3) | `activity_regime` rate_hz/cv_isi/kappa/label (checklist E3; no lane-local estimator) | exact (estimator choice; values scored at P3) |

## Numerics declared

- Clock-driven, one AdEx step per 0.1 ms bin; conductance decay `exp`
  per step; ring-buffer delayed arrivals; no sub-step spike interpolation.
- Pilot: N=200 (NE=160, NI=40, ~1.6k edges at epsilon=0.02), T=1.0 s,
  dt=0.1 ms, seed 0, CPU, float32; outputs under
  `scratch/vogels2011_pilot/` (gitignored).
- Scale decision (P2): full N=10000 (~2M edges at epsilon=0.02) is deferred
  to the P3 protocol; the pilot settles the route, not the scale.
