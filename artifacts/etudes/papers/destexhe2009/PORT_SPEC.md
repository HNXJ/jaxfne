# Destexhe 2009 port spec (jaxfne core)

Source values: `SOURCE_P1.md`. Engine rule (Hamm 2026-10-13): every run uses
the core grammar `Configuration -> construct -> Model -> simulate/Signals`
through the AdEx route (`construct_adex_model`, `adex_kernel_args`,
`adex_external_drive_kwargs`, `simulate_adex`), exactly as
`artifacts/etudes/caminos/caminos_api.py` (route c) runs it. Structure mirrors
`caminos_api.py` (library-path module + CLI runner, rasters + manifest,
scored-dir refusal). Optimizers for the residual protocol:
`jaxfne/optim/core.py` (`agsdr`, `optax_adam`).

## Stage order (smallest first)

1. **S1 — single cells (Fig. 1).** RS / FS / LTS at the §1 table, ±250 pA
   steps, dt = 0.1 ms: RS adapts, FS does not, LTS rebounds after release.
2. **S2 — cortical self-sustained AI (Fig. 6B).** N = 2000 (1600 RS + 400 FS),
   p = 2% all four blocks, ge = 6 nS, gi = 67 nS, τE = 5 ms, τI = 10 ms,
   E_rev = 0 / −80 mV, weak adaptation (b = 5 pA). Estimands: alive at 10 s,
   CV_ISI = 2.47, CC = 0.005, PY rate order.
3. **S3 — controls.** Fig. 6A (strong adaptation, b = 40 pA: transient, dead
   within seconds); Fig. 7B (N = 500 + 5% LTS: AI); Fig. 9 row (minimal N).
4. **S4 — thalamic / thalamocortical / two-layer** (Figs. 3–5, 10–13) after S2
   scores.

## Mapping to jaxfne objects

| Paper object | jaxfne object |
|---|---|
| aeIF cell + type table | `jaxfne.adex_destexhe` (core addition): `DESTEXHE_CELL_DEFAULTS` in absolute units (§1 table) applied to an adex-family Model by `apply_destexhe_params`; carriers E/PV/SOM keep the route's labels, only the eleven parameter fields change (ten route fields plus `Vpeak_mV`, the authors' −50 mV detection, F6) |
| cortical RS+FS network | `Configuration.network(cell_types={"E": 0.8, "PV": 0.2}).emitter(family="adex", adex_decay="exp")`, E carries RS, PV carries FS |
| conductance synapse (τE/τI, reversals) | two `mechanisms()` receptors (`tau_ms=(5.0, 10.0)`, `reversal_mV=(0.0, −80.0)`), `decay="exp"` (NEURON exact-exp reading) |
| random 2% blocks, no autapses (G10) | `connections()` capped-Bernoulli rules (`sampling="capped_bernoulli"`, the authors' per-cell cap, F7) compiled by `compile_connection_rules` inside `construct_adex_model`; `edge_seed` from `connectivity()` |
| no synaptic delays | `delay_ms = 0.1` = 1 step at dt = 0.1 ms (minimum the kernel accepts; fidelity note F1) |
| 50 ms kick then free run | one `simulate_adex` call over kick + free steps with the core `ext_t_start_steps`/`ext_t_stop_steps` drive window (`t_start_ms=0`/`t_stop_ms=50` on the run's `Simulation`; `run_continuous` / `fig6b_cont`), so w, g, ref and in-flight spikes carry through the boundary. Two-phase `run` (phase 2 from phase-1 V, w/g/ref restart) stays as the F2 control. Kick: 286 Hz ge kicks to a seeded 0.25 fraction of RS (`subset_drive_kwargs`, code value F5); FS cells get no drive |
| CV_ISI, CC (Eqs. 3–4) | `cv_isi`, `pairwise_cc` on the spike raster (5 ms bins, 500 disjoint pairs) |
| residual fit | `agsdr` on L = Σe² (non-differentiable path); `optax_adam` only for differentiable X terms |

## Fidelity notes (port-vs-paper, not fits)

- F1: 0.1 ms uniform delay (kernel minimum) vs no delay.
- F2: phase-2 restart drops w, g and ref from the 50 ms kick (V carried).
- F3: threshold VT + 5Δ with explicit Euler vs NEURON's unstated detection.
- F4: autapses excluded (paper silent).
- F5: kick weight/shape assumed (G5): 300 Hz private Poisson sources, 6 nS
  AMPA kicks to the RS population only (the kernel targets whole types).
  Authors' code value (G8): N/5 cells × 20 synapses × exponential 70 ms
  ≡ Poisson ≈ 285.7 Hz aggregate, 50 ms — i.e. kick fraction 0.25 of RS
  at 286 Hz, outside the paper's stated 2–10% range.
- F6: spike detection: authors' code detects at ≈ VT (−50 mV,
  `v_thresh = v_spike ≈ VTR`); the core kernel default is VT + 5Δ
  (−37.5 mV). The port carries the authors' value on every mapped cell
  (`Vpeak_mV = −50.0` in `DESTEXHE_CELL_DEFAULTS`, applied through
  `destexhe_overrides`); unmapped carriers keep the core default
  (NaN = absent). Core default unchanged (CALIBRATION.md step 6).
- F7: connection rule: authors' code caps in-degree per cell
  (C_E/C_I inputs, ordered scan); the core compiles capped Bernoulli
  (`sampling="capped_bernoulli"`, in-lane fidelity fix, CALIBRATION.md
  step 6). Residual: ordered-scan vs independent-cap draw order is
  unproven equivalent; ≈7% exc / ≈14% inh trim at N = 2000 stands.

## Module layout

- `destexhe_api.py`: S2 constants, `build_spec` (public API + core cell
  table), `build_edges` (wiring read off the constructed model),
  `single_cell`, `epsp_peak`, `run` (two-phase core run), `run_continuous`
  (one core `simulate_adex` call with the `t_start_ms`/`t_stop_ms` drive
  window; `kick_fraction` via `subset_drive_kwargs`, `kick_rate_hz`,
  `v0_seed`, `ongoing`, `probability`), `cv_isi`,
  `pairwise_cc`, `mean_rate_hz`, `write_run` (rasters + manifest; refuses
  `scored` output).
- `run_destexhe.py`: CLI (`fig1`, `fig6b`, `fig6b_cont`, `score`) importing the helper.
- `tests/test_destexhe2009_port.py`: single-cell patterns, wiring counts,
  conductance jumps, metric known-answers.
- Core addition: `jaxfne/adex_destexhe.py` (+ `from . import adex_destexhe`
  in `jaxfne/__init__.py`), tested by `tests/test_adex_destexhe.py`.
