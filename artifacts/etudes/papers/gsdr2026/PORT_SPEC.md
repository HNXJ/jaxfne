# GSDR Fig. 6 circuit: P2 port specification

Source decision (Hamm 2026-10-13, `SOURCE_P1.md` §4): follow the code that
produced Fig. 6, the `misc/main.py` run of `HNXJ/GSDR` (read as data, never
run). Where the paper text differs, the difference is a declared deviation
(§9). Live code verified by re-read 2026-10-14: `gsdr/models.py`
(`Inoise`, `GradedAMPA/GABAa/GABAb`, `build_net_eig`), `misc/main.py`
(`net_eig`, `noise_current_ac` use, loss, GSDR settings), `gsdr/simulation.py`
(`noise_current_ac`), `gsdr/pipeline.py` (`get_loss_fn` kappa variant).

Conventions: voltages mV, time ms, currents per the paper's Eq. 1 sign
convention (positive outward current hyperpolarizes). Lane runs are float32
on CUDA; the reference script ran float64 on CPU (§9, deviation D7).

## 1. Composition

| Item | Value | Source |
|---|---|---|
| Excitatory cells (E) | 36 | paper Results; `misc/main.py: Ne = 36` |
| Global inhibitory (SST-like) | 8 | paper Results; `misc/main.py: Nig = 8` |
| Local inhibitory (PV-like) | 6 | paper Results; `misc/main.py: Nil = 6` |
| Total | 50 | sum |

## 2. Cell model

Stock Jaxley `HH()` plus one per-cell `Inoise` OU channel; `radius = length = 1.0`
(`gsdr/models.py: build_net_eig`, `misc/main.py: net_eig`). Membrane equation is
the paper's Methods Eq. 1: `C_m dV/dt = −I_input − ΣI_intrinsic − ΣI_synaptic`.
HH rate parameters, `C_m` and solver: **read from jaxley 0.14.0** (downloaded
wheel, read as data 2026-10-14): classic rate shapes (`channels/hh.py`
`m/h/n_gate`, term-by-term identical to the lane's rest-relative form),
`gNa = 0.12, gK = 0.036, gLeak = 0.0003` (S/cm²),
`eNa = 50, eK = −77, eLeak = −54.3` (mV), `C_m = 1.0` (µF/cm², the compartment
default), gates stepped by exact exponential Euler (`solver_gate.py`),
voltages by `bwd_euler` (the `integrate` default), compartments init at
`v = −70.0` mV with gates at their rate fixed point (`HH.init_state`), `exp`
arguments clipped at 20 (`save_exp`). The lane assumed `EL = −54.364`
(classic textbook); the fidelity fix sets the read `−54.3`.

## 3. OU noise channel (`Inoise`)

Per cell, `mean = 0.0`; `tau ∼ U(10, 50)`, `amp = clip(U(−0.1, 0.1), 0, 0.05)`
(observed `misc/main.py: net_eig`; the `gsdr/models.py: build_net_eig` builder
uses `U(5, 50)` / clip `[0, 0.1]` — the Fig. 6 run values win). Per-cell seed
random (`np.random`, unrecorded → lane seed is étude-local). Update
(observed `gsdr/models.py: Inoise.update_states`):

```text
n += (mean − n)/tau · dt + sigma · sqrt(2/tau) · xi · sqrt(dt), xi ∼ N(0,1)
I_noise = −n        (compute_current returns −states["n"])
```

`init_state`: `n = mean`, step counter drives a per-step folded PRNG key.

Units (read 2026-10-14): the noise channel declares
`current_is_in_mA_per_cm2`, so its current `−n` is a density in mA/cm²
(combined `*1000` into the uA/cm² common unit). The lane kernel works in
uA/cm² throughout, so the runner scales the reference amplitudes by 1000.

## 4. Graded synapses

Activation (observed, all three classes in `gsdr/models.py`):

```text
act(Vpre) = 0.5 · (1 + tanh((Vpre − Vth)/slope))
ds/dt = −s/tauD + act · (1 − s)/tauR
I_syn = g · s · (Vpost − E_rev)
s(0) = 0.1 (AMPA, GABAa); 0.01 (GABAb)
```

| Class | g default | E_rev | tauD (Fig. 6 run) | tauR | slope | Vth | Used |
|---|---|---|---|---|---|---|---|
| GradedAMPA | 2.5 | 0.0 | 2.0 (class default 5.0) | 0.2 | 5.0 | −20.0 | E→all |
| GradedGABAa | 5.0 | −80.0 | 5.0 global / 2.0 local | 0.5 | 5.0 | −20.0 | I→ targets below |
| GradedGABAb | 1.0 | −95.0 | 200.0 / tauR 10.0 | 10.0 | 5.0 | −20.0 | defined, never connected |

Paper states only that rise/decay are fixed per references; every value above
is code-only. Local-I tauD: 2.0 in `misc/main.py: net_eig`, 5.0 in
`gsdr/models.py: build_net_eig` — the Fig. 6 value is 2.0.

Units (read 2026-10-14): synapses are point processes — conductances in uS,
currents in nA (`synapses/synapse.py` base docstring) — divided by the post
compartment area into the uA/cm² common unit
(`convert_point_process_to_distributed`: `I/A*1e5`). The lane kernel takes
densities, so the runner converts reference uS by `1e5/AREA_CELL_UM2`
(§14).

## 5. Wiring (observed `misc/main.py: net_eig`)

| Rule | Pre → post | Edges |
|---|---|---|
| E→all AMPA | 36 E → all 50 | 1800 |
| Global-I→all GABAa | 8 → all 50 | 400 |
| Local-I→10 % GABAa | 6 → 5 posts (`int(50·0.1)`, `jax.random.choice(PRNGKey(1))`, no replacement) | 30 |

Total 2230 trainable conductances. `gsdr/models.py: build_net_eig` keys the
local choice on the build seed instead of 1; the Fig. 6 run uses key 1.

## 6. Conductance init and clamps (observed `misc/main.py`)

`gAMPA ∼ U(−4, 10)` clipped to `[0, 8]`; `gGABAa ∼ U(2, 10)` clipped to
`[0, 10]`; optimizer clamp transforms bound both to `[0, 10]`.

## 7. Stimulus (observed `misc/main.py`, `gsdr/simulation.py`)

`noise_current_ac(i_delay = 500, i_dur = 500, amp_n = 0.0, amp_b = input,
spect = [120.0], dt = 0.1, t_max = 1500)`: inside `[500, 1000]` ms the current
is `amp_b · sin(2π·120·t/1000)` (the `amp_n = 0` OU part contributes 0), else 0.
Delivered to every other E cell (`range(0, 36, 2)`, 18 cells). Batch inputs
`[0, 1, 1] · 0.6`, i.e. amplitudes `(0.0, 0.6, 0.6)` nA. Recording: all cells,
branch 0.

Units (read 2026-10-14): `data_stimulate` takes nA, converted by
`convert_point_process_to_distributed` (`I/A*1e5` into uA/cm²) and added with
a positive sign into the voltage update, i.e. positive depolarizes
(`modules/base.py: _get_external_input`, `_step` constant terms) — the
lane's depolarizing-positive convention was already correct (D9 resolved,
no sign change).

## 8. Integration grid (code-only, observed `misc/main.py`)

`dt = 0.1` ms, `t_max = 1500` ms (15001 steps), stimulus `500–1000` ms,
`jx.integrate` with 2-level checkpointing. Paper dt/duration/integrator:
**unknown**.

## 9. Fig. 6 loss (observed `misc/main.py: loss`)

- Target grid `linspace(15, 60, 40)` Hz; labels from data spectra
  (`selected_unit_spiking`: baseline window `:200`, stim window `600:1200` of
  trial slice `5:6`, mean over units; min-subtracted, max-normalized).
- Model PSD: mean voltage over cells in window `100–1400` ms → rFFT power /
  `(N·fs)` → interp onto grid → divide by max (+1e-6).
- Per-sample `psd = Σ (label · ((pred+eps) − (label+eps)))²`, eps = 1e-6;
  `total = mean(psd)·990 + 9·mean(exp(5 − F) + exp(F − 50))`,
  `F` = per-neuron rate from upward −20 mV crossings / 1.5 s.
- `gsdr/pipeline.py: get_loss_fn` variant (packaged form): pre `100–500` /
  stim `500–1000` / post `1000–1400` windows against separate stim/off targets
  plus `100·Σ|kappa|` (pre/stim/post). The Fig. 6 run used the `main.py` form
  (no kappa term; kappa reported as a metric, Fig. 6 caption). The pipeline
  form is the declared alternative for the calibration stage.

## 10. GSDR settings of the Fig. 6 run (observed `misc/main.py`)

Inner SDR `lr = 0.1, momentum = 0.4`; `GSDR(deselection_threshold = 3.0`
(= paper's `C`), `a_init = 0.0, a_dynamic = True, lambda_d = 1.0,
checkpoint_n = 5, mcdp = True)`; 500 epochs. MCDP (observed
`misc/main.py: compute_correlations`): squared Pearson correlation of pre
voltage vs post voltage shifted +10 steps, Gaussian-smoothed
(`k = int(sqrt(N))`), z-scored per (AMPA/GABAa) group. Per-weight update per
paper Methods Eqs 2.1–2.2: `Δ = (L·D·δ(λ)(1−α) + α)·R_pre,post`;
`W → [0, max_g]`, `α → [0, 1]`, dynamic α random-walks ±0.1 under selection;
NaN loss reverts to the best state.

## 11. Paper/code deviations (declared)

- D1 stim amplitude 1 nA (paper) vs 0.6 (code run).
- D2 stimulus onset "time 0" (paper Fig. 6) vs 500–1000/1500 ms (code).
- D3 "random half of E" (paper) vs every-other-E, fixed 18 cells (code).
- D4 local-I tauD 2.0 (`main.py`) vs 5.0 (`models.py` builder); run uses 2.0.
- D5 OU ranges `U(10,50)`/clip 0.05 (`main.py`) vs `U(5,50)`/clip 0.1 (builder).
- D6 `C = 3.0` / checkpoint 5 / `a_init = 0.0` (run) vs package defaults
  `C = 2.0` / checkpoint 10 / `a_init = 0.5`.
- D7 lane dtype float32 on CUDA vs reference float64 on CPU
  (`jax_enable_x64`, `jax_platform_name = "cpu"` in `misc/main.py`).
- D8 stimulus is a gated 120 Hz sine (code), not a DC step convolved with a
  pulse train (paper text).
- D9 `data_stimulate` current sign convention: resolved 2026-10-14 (positive
  depolarizes; lane unchanged).
- D10 HH kinetic parameters and `C_m`: resolved 2026-10-14 (read from jaxley
  0.14.0; only `EL` moved, −54.364 → −54.3, §2).

## 12. Assumed values (unknown in both paper and code)

- A1 HH kinetics/`C_m`: resolved (read from jaxley, §2).
- A2 OU seeds: the reference draws a random seed per cell at build; the lane
  passes one key per trial (per-cell, per-step draws come from the split
  stream). A3 conductance init RNG stream (lane: PRNG seed 0).
A4 local-post choice key 1 (code-observed, kept). A5 spike threshold −20 mV
(code-observed, kept). A6 PSD window `100–1400` ms (code-observed, kept).

## 13. Core map (P3: Fig. 6 circuit on the jaxfne core)

New core modules (additive only; no existing public function changed):

| Port item (§) | Core home |
|---|---|
| HH kinetics, OU step, graded tanh synapse, edge-list kernel (§§2–4) | `jaxfne/emitters_hh.py`: `make_hh_params`, `hh_step`, `hh_gate_inf`, `ou_step`, `graded_syn_step`, `graded_syn_current`, `simulate_hh` |
| Spec → edges: population, receptors, rules, edge seed (§§1, 4–5) | `jaxfne/_construct_hh.py`: `construct_hh_model`, `hh_kernel_args` |
| Reachability | `jaxfne/__init__.py` gains `from . import emitters_hh` (`__all__` unchanged) |

Declaration mapping used by `artifacts/etudes/gsdr2026/run_gsdr.py:build_spec`
(and `tests/test_gsdr2026_model.py`):

- Population: `network(n=50, cell_types={E: 36/50, Ig: 8/50, Il: 6/50})`.
- Receptors: `mechanisms(kind="receptor")` named `ampa` (tau_d 2.0, E_rev 0.0,
  tau_r 0.2), `gaba_global` (tau_d 5.0, E_rev −80.0, tau_r 0.5), `gaba_local`
  (tau_d 2.0, E_rev −80.0, tau_r 0.5); all slope 5.0, Vth −20.0, s0 0.1.
- Rules (all `probability=None` = exact full bipartite; compiled with
  `allow_self_connections=True` since the reference includes self-edges):
  `e_all` E→all (1800, ampa), `ig_all` Ig→all (400, gaba_global),
  `il_local` Il→`ids` posts (6×5=30, gaba_local). Total 2230.
- The local-post mask (`choice(PRNGKey(1), arange(50), (5,))`) is a
  `target={"ids": posts}` selector, not a sampled probability.
- Per-edge conductances (§6 init), OU hyperparams (§3), stimulus (§7), and
  metrics/loss (§9) stay etude-local in `run_gsdr.py`: they are run inputs
  and readouts, not circuit structure. `construct()` dispatch is untouched
  (route-(c) precedent: `artifacts/etudes/caminos/caminos_api.py`).
- Refusals: `delay_ms` set (kernel has no delay ring), negative weight,
  undeclared/missing mechanism, non-`receptor` kind, any `emitter()`
  declaration, mixed receptor `s0` (kernel takes one scalar).

## 14. Lane reductions (lane vs reference code; declared simplifications)

- S1 one compartment per cell: the reference cell is 2 branches × `ncomp=2`
  compartments (4 total). The lane carries one compartment with the
  whole-cell area and capacitance; there is no axial coupling.
- S2 whole-cell area `AREA_CELL_UM2 = 80π ≈ 251.33` µm²: each compartment
  covers `2π·r·l` with `r = 1.0` µm (default) and `l = 10.0` µm (default —
  the `cell.length = 1.0` assignments are plain-attribute no-ops since
  jaxley defines no `__setattr__`, so the default stands), i.e. 20π µm² each.
  Point-process inputs convert with the whole-cell area (total current is
  preserved): stimulus `0.6 nA → 238.7 uA/cm²`, synapse `g uS → g·397.89
  mS/cm²`, OU `σ mA/cm² → σ·1000 uA/cm²`. Spreading the stimulus over the
  whole cell (reference: one of four compartments) is part of S1.
- S3 per-cell OU seed stream (§12 A2): one trial key, not per-cell seeds.
- S4 float32 on CUDA (D7); the reference ran float64 on CPU.
