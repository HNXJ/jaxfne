# mejias2016 port specification (P2)

Paper: Mejias JF, Murray JD, Kennedy H, Wang X-J (2016), Sci Adv
2(11):e1601335, doi:10.1126/sciadv.1601335. Source record: `SOURCE_P1.md`.
Every paper value below is observed there (§1-§2) or unknown there (§4);
every core behaviour is observed in this tree; probe numbers are observed
from the P2 probes in this run (single-area module, N=40, seed 0, dt=0.5
ms). Status per row: exact (port equals paper), deviation (port differs,
declared here), gap (paper needs what no public path offers; settled by a
test in `test_mejias2016_api.py`).

Lane route (checklist E1): `NeuronalTensor` +
`RuntimeConfiguration(emitter="lif")` -> `construct(tensor, runtime)` ->
`Model.simulate(Simulation(...))` (pattern: `construct` in `jaxfne/_construct_core.py`,
`tests/test_inputs_xsource_t24.py:92-99`). Estimators from `jaxfne.analysis`
(checklist E3): `activity_regime` (`activity_regime` in `jaxfne/analysis/metrics.py`), Welch
peak via `_spectral_peak_excess` (`_spectral_peak_excess` in `jaxfne/analysis/metrics.py`),
`wang_buzsaki_kappa` (`wang_buzsaki_kappa` in `jaxfne/analysis/metrics.py`), PSD via
`spectrolaminar_psd_jax` (`spectrolaminar_psd_jax` in `jaxfne/analysis/spectral.py`). No lane-local
estimator. `Configuration.set_emitter(family="lif")` is not the lane path;
the tensor route is the only lif path.

## Model form (checklist E5)

- A (state): membrane voltages V (mV), spikes; no adaptation, no rate state.
  Paper: one nonlinear Wilson-Cowan firing-rate pair (E plus I) per laminar
  module (`SOURCE_P1.md` §1). The core ships the rate emitter (exact row 1:
  `jaxfne/emitters_rate.py`, `jaxfne/_construct_rate.py`); the lane pilot
  still runs the declared spiking stand-in (deviation S1): lif E-PV pairs
  per module, extras zeroed (`LIF_CELL_TYPE_DEFAULTS` in `jaxfne/emitters_lif.py`).
- x (input): tonic triple drive (declared 0.0) plus one Poisson channel per
  cell (`_make_poisson_drive` in `jaxfne/_signals.py`); paper: tonic currents I plus noise, both
  of unknown process (`SOURCE_P1.md` §4, gap G6).
- h (parameters): J-equivalent gains, delays, background mapping below;
  fixed weights (no plasticity). Paper time constants, weights, noise
  amplitudes unknown (`SOURCE_P1.md` §4, gap G6).
- y (readout): spike rasters; `activity_regime` (rate_hz, cv_isi, label),
  `wang_buzsaki_kappa`, Welch PSD peak; area signal as equal-weight L23+L56
  sum (deviation S2: paper uses a depth-electrode-weighted combination,
  weights unknown, `SOURCE_P1.md` §1).
- Geometry: one area `V1` (two-area `V1,V4` builder for the FF/FB rows),
  layers `L23`,`L56`, default `Geometry3D` (`Layer` in `jaxfne/neuronal_tensor.py`)
  with no space-dependent terms; delays are declared per rule, not derived
  from positions (deviation S3: the paper network has distance-dependent
  wiring, distances unknown, `SOURCE_P1.md` §4).

## Per-element rows

| # | Element (paper) | Core path | Status |
|---|---|---|---|
| 1 | Local module unit: nonlinear Wilson-Cowan E-I rate pair per layer (`SOURCE_P1.md` §1) | `NeuronType(emitter="rate")` + `RuntimeConfiguration(emitter="rate")` -> `construct_rate_tensor_model` (`jaxfne/_construct_rate.py`); transfers in `jaxfne/emitters_rate.py`, registered under the name "rate" by `register_emitter` in `jaxfne/emitter_registry.py` | exact (pinned by `test_wilson_rate_emitter`; lane pilot still runs the lif stand-in S1 below) |
| 2 | Stand-in S1: lif E-PV pairs per module, extras zeroed | `NeuronType.make("E"/"PV", emitter="lif", emitter_params={tau_m_ms:20, v_thresh_mV:20, v_reset_mV:10, ...})`; fields in `LIFParams` in `jaxfne/emitters_lif.py`, applied by `lif_step` in `jaxfne/emitters_lif.py` | deviation (declared: spikes stand in for rates; no rate claim; pilot runs this) |
| 3 | Supra gamma ~40 Hz, infra alpha ~10 Hz / low-beta 15-30 Hz (`SOURCE_P1.md` §2) | Constants, weights, noise amplitudes unknown (`SOURCE_P1.md` §4) | gap G6 (no test: nothing to fail; settled at P3) |
| 4 | Interlaminar motif: L23E to L5E; L5E to L2/3I, the two strongest projections (`SOURCE_P1.md` §1) | Same-area `AreaConnection`s (`AreaConnection` in `jaxfne/neuronal_tensor.py`); no other cross-layer rules declared | exact (pinned by `test_interlaminar_motif_exact`) |
| 5 | FF: source L2/3E to target L2/3E, standing in for L4 targeting, L4 unmodeled (`SOURCE_P1.md` §1) | Cross-area `AreaConnection` V1-L23-E to V4-L23-E (`AreaConnection` in `jaxfne/neuronal_tensor.py`) | exact (pinned by `test_ff_fb_motif_exact`) |
| 6 | FB: source L5/6E to all four target pops, stronger onto L5/6E; supra/infra 0.8 variant (`SOURCE_P1.md` §1) | Four cross-area rules with distinct `w_mech`, infra:supra 2:1 declared; exact ratio tuned, value unknown (`SOURCE_P1.md` §4) | exact as configuration, values gap G6 (pinned by `test_ff_fb_motif_exact`; ratio under G6) |
| 7 | Area signal: weighted L2/3+L5/6 combination mimicking depth electrode (`SOURCE_P1.md` §1) | Equal-weight layer sum in the lane scorer; weights unknown | deviation S2 (declared; scoring choice, not dynamics) |
| 8 | Tonic inputs: Fig. 3A I=8 to both E pops; Figs. 3B-7 values (`SOURCE_P1.md` §2) | Triple `drive` (`NeuronType` in `jaxfne/neuronal_tensor.py`) plus per-cell Poisson (`_make_poisson_drive` in `jaxfne/_signals.py`); paper current units vs lif mV | deviation (declared mapping: constant drive 0.0, background as Poisson rate 400 Hz amplitude 2 mV; probe: finite V, spikes present) |
| 9 | Noise process driving each module (`SOURCE_P1.md` §1) | `Simulation(poisson_drive={rate_hz, amplitude, seed})`: independent per-neuron Bernoulli trials, at most one event per step (`_make_poisson_drive` in `jaxfne/_signals.py`); `XSource` kinds (`XSource` in `jaxfne/inputs.py`) | deviation (declared; paper process unknown, gap G6) |
| 10 | Per-edge delays (paper values unknown, `SOURCE_P1.md` §4) | `delay_ms` per rule, grid-aligned to steps (`edge_delay_steps_from_ms` in `jaxfne/emitters.py`); dt=0.5 gives exactly 2 steps for 1.0 ms | exact route (pinned by `test_uniform_delay_steps`); values under G6 |
| 11 | Multi-area tensor: 30 named areas, ~66% density (`SOURCE_P1.md` §2) | `NeuronalTensor` areas plus `AreaConnection(probability=...)` (`AreaConnection` in `jaxfne/neuronal_tensor.py`); exact total `round(p*n_pre*n_post)` per rule | exact route (pinned at 2 areas by `test_ff_fb_motif_exact` and totals by `test_density_exact_totals`); 30-area scale gap G7, deferred to P3 |
| 12 | SLN/FLN anatomy, hierarchy subset (`SOURCE_P1.md` §2) | Lane-assigned per-pair gains; matrices served from a login wall, unknown (`SOURCE_P1.md` §4) | gap G6 (no test: input data, not core behaviour) |
| 13 | Estimands: layer rates, spectral peaks, gamma/alpha bands (`SOURCE_P1.md` §3) | `activity_regime`, Welch peak, kappa, `spectrolaminar_psd_jax` (all `jaxfne/analysis`, checklist E3) | exact (estimator choice; values scored at P3; pinned by `test_score_keys`) |
| 14 | Directionality: spectral Granger causality, DAI/mDAI Eqs. 1-2 (`SOURCE_P1.md` §1) | `granger_spectral` + `directionality_index` (DAI Eq. 1) in `jaxfne/analysis/granger.py`, exported from `jaxfne/analysis`; `cross_area_coherence` in `jaxfne/fields/proxy.py` stays a continuous-proxy operator, not the spike-signal estimator | exact as the DAI Eq. 1 pairwise estimator (pinned by `test_dai_granger`); the multivariate mDAI form is not ported |
| 15 | Integrator, dt, duration, seeds (paper unknown, `SOURCE_P1.md` §4) | Exact exponential decay per `lif_step` (`lif_step` in `jaxfne/emitters_lif.py`); dt=0.5 ms binary-exact; seed 0 pilots only | deviation (declared; convergence check at P3) |
| 16 | Population sizes, sparse wiring (paper: full 30-area counts unknown) | Fractions split counts exactly (probe: 16/4 per layer at n=20); `dT_ms` required but inert on the lif route (value 2.0 ms carried for inspection) | deviation (declared-but-inert dT; sizes declared per builder) |

## Numerics declared (G6)

- Clock-driven, one `lif_step` per 0.5 ms bin; ring-buffer delayed
  arrivals; no sub-step spike interpolation; no hard refractory (lif route
  re-fires the next step under strong drive, as in the brunel lane).
- Pilot: single-area two-layer module, n=20 per layer (40 cells), T=1.0 s,
  seed 0, CPU, float32; outputs under `scratch/mejias2016_pilot/`
  (gitignored).
- Scale decision (P2): the 30-area network is rows 11-12 of this spec only;
  no 30-area run precedes the P3 protocol decision.

## Status counts

16 rows: 8 exact (1, 4, 5, 6-config, 10-route, 11-route, 13, 14),
6 declared deviations (S1 stand-in row 2, S2 readout, 8, 9, 15, 16),
2 gap references (G6 rows 3, 6-values, 12; G7 row 11-scale).
