# GSDR lane P3 baseline calibration record (OBSERVED, unscored)

Date: 2026-10-14. Code: `3dd02ed060390ecf711e04860a1d6d46e854ef83` (one SHA
per directory, enforced by the runner). JAX 0.10.1, device `cuda:0`
(NVIDIA RTX A4000), float32, dt 0.1 ms, 15001 steps. Outputs:
`scratch/gsdr2026/baseline/` (3 V traces, per-trial conductances,
`manifest.json`). Prereg: `PREREG_P3.md` (draft, frozen before the run
except per-trial provenance fix, which re-ran the baseline).

## Results (model readouts only; no data labels, no loss scored)

| trial | amp (nA) | g_seed | ou_seed | mean rate (Hz) | rate penalty | kappa |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 0 | 0 | 0.547 | 89.8994 | 0.3708 |
| 1 | 0.6 | 1 | 1 | 0.587 | 85.5665 | 0.4147 |
| 2 | 0.6 | 2 | 2 | 0.520 | 92.7880 | 0.3377 |

Wiring asserts passed in every trial (1800 AMPA / 400 global GABAa /
30 local GABAa). Per-cell rates (manifest): nearly all cells 0.667 Hz
(1 spike / 1.5 s), remainder silent. Model PSDs (15–60 Hz, max-normalized)
are in `manifest.json` per trial.

## Reading (no calibration claimed)

Mean rates (~0.5 Hz) sit far below the 5–50 Hz in-bounds band, so the
one-sided penalty (~90) dominates; the 0.6 nA stimulus barely moves the
rate (trial 0 vs 1–2). Nothing is tuned post hoc: per the prereg, a miss
goes to the fidelity suspects in order — D9 (stimulus sign convention),
then A1/D10 (assumed HH kinetics/`C_m`) — and a paper/code disagreement
on a claim-changing value stops the lane. Live fields, if any, carry
`physical_amplitude_calibrated=False` (CALIBRATED != REDUCED_PHYSICAL !=
RELATIVE_PROXY).

## Step 1 — simplify (RESIDUAL_PROTOCOL.md §steps.1, OBSERVED)

Smallest model showing the miss: single E cell f-I on the lane kernel,
dt 0.1 ms. 0 Hz at 0–0.6 uA/cm²; NaN-dominated ~1 Hz at 3–50 uA/cm².
`hh_step` NaN at step 21 at I = 10 uA/cm², dt = 0.1 (99.6 % NaN over the
trace); dt = 0.01 stable at 70 Hz. Gated 120 Hz sine at raw 0.6 (≈0.6
uA/cm², unconverted) → 0 Hz; converted 0.6 nA (≈238.7 uA/cm²) → 1.33 Hz.
E→I pair barely follows. Verdict: the miss is a units + integration
problem, not wiring (wiring asserts passed throughout).

## Fidelity fixes (not fits; each with a fail-first test)

Kernel (`jaxfne/emitters_hh.py`), read from jaxley 0.14.0 source as data:
gates by exact exponential Euler (`solver_gate`, `emitters_hh.py:99`),
voltage by backward Euler linearized at frozen currents (`:173`);
`EL_mV = -54.3` (`:42`), rest `HH_V_REST_MV = -70.0` (`:69`);
`hh_gate_inf` fixed-point gate init (`:106`, used at `:475`);
`_save_exp` clips exp() arguments at 20.0 (`:71`, `:77`);
non-finite guards keep old state (`:209`, `:229`, `:444`, `:451`).
Fail-first: 6 kernel tests in `tests/test_emitters_hh.py` (all failed
pre-fix, e.g. `EL must be the reference -54.3 mV, got -54.364`).

Runner (`artifacts/etudes/gsdr2026/run_gsdr.py`): whole-cell area
`AREA_CELL_UM2 = 80π` µm² (`:68`, S1/S2 one compartment per cell);
`NA_TO_UACM2 = US_TO_MSCM2 = 1e5/80π ≈ 397.89` (`:70`–`:71`); stimulus
0.6 nA → ≈238.7 uA/cm² (`:178`); OU σ ×1000 mA→uA/cm² (`:72`–`:73`,
`:164`); rule weights and init conductances converted (`:111`–`:118`,
`:138`). Fail-first: 5 conversion tests in
`tests/test_gsdr2026_fidelity.py` (all failed pre-fix, e.g. stimulus
0.6 vs 238.7). Post-fix CPU: 33 passed (`33 passed in 92.96s`); model
tests 11 passed; 300 ms smoke finite, 2.667 Hz both amps (stim window
starts 500 ms, identical expected).

## Fidelity-1 run (OBSERVED, scored against the 5–50 Hz band)

Code `3dd02ed060390ecf711e04860a1d6d46e854ef83`, JAX 0.10.1, `cuda:0`,
float32, dt 0.1 ms, 15001 steps. Outputs:
`scratch/gsdr2026/fidelity1/` (manifest + per-trial V/g + rerun twin
`scratch/gsdr2026/fidelity1_rerun/`). GPU wall ≈ 8 s for 3 trials
(in-rerun log: 6.9/2.8/5.2 s).

Estimand: paper Fig. 6 baseline activity in the 5–50 Hz in-bounds band
(`run_gsdr.py:56`; no point y — no MT/MST data in the lane, so the
verdict is band membership, protocol e undefined).

| trial | amp (nA) | seeds (g, ou) | z0 pre-fix (Hz) | z post-fix (Hz) | penalty | kappa | band |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0, 0 | 0.547 | 0.533 | 91.34 | 0.35 | off |
| 1 | 0.6 | 1, 1 | 0.587 | 21.64 | 0.00 | 0.84 | **in** |
| 2 | 0.6 | 2, 2 | 0.520 | 0.493 | 96.38 | 0.28 | off |

Trial 1 (OBSERVED): all 50 cells 18.7–22.7 Hz (≈20–22 Hz uniform);
trial 0/2: nearly all cells 0.667 Hz (1 spike / 1.5 s), remainder
silent — same shape as the pre-fix baseline. The fidelity fix moves one
stim-trial draw into the band and leaves the other two ~0.5 Hz:
seed (network-draw) variance spans the band edge at fixed stimulus.

Determinism twin (same SHA, same seeds): trials 0/2 bit-identical
(0/50 cells differ); trial 1 mean 21.64 → 21.69 Hz, 5/50 cells differ,
max per-cell diff 1.33 Hz (2 spikes / 1.5 s). Both in-band; the verdict
is unaffected. Source of the active-regime wobble unidentified → OPEN.

## Verdict and next

Mixed: 1 of 3 trials in-band; the stim condition (trials 1–2) splits
21.6 vs 0.5 Hz across network draws. Per the protocol this is still
off → next is residual X + `jaxfne.optim.agsdr` fit (θ fixed), absorb
into D, held-out seeds 4–8, then the `gsdr` run. Not started.

## Step 2 — Fig. 6 data-label y (OBSERVED, unscored)

Recipe (`data_labels.py`, read as data from `misc/main.py:109-141`):
trial 5 (`5:6`), baseline `:200`, stim `600:1200`, mean over the MT/MST
units (info groups `(122, 40, 95)`, slice `(162, 257)`), rFFT power
`/(N*fs)` at `fs=1000`, interp onto `linspace(15, 60, 40)`,
`uniform_filter1d` size 5, min-subtracted, max-normalized. Warehouse
files stay out of the repo (hashes in `data_labels.py:31-34`,
`794b5ee6…` / `759349b2…`, verified in
`tests/test_gsdr2026_labels.py`). Vectors + scalars:
`/home/nejath/gsdr_src/y_labels.json` (outside the repo, like the data).

| quantity | baseline | stim |
|---|---|---|
| PSD peak (Hz) | 40.38 | 38.08 |
| gamma band (35–60) | 0.4347 | 0.3769 |
| beta band (13–30) | 0.0844 | 0.2250 |
| alpha / theta | 0 / 0 (outside the 15–60 grid) | 0 / 0 |
| onset rate mean (Hz, thr 20.0) | 5.79 (std 8.59, 48/95 silent) | 2.14 (std 5.47, 52/95 silent) |
| kappa, pairwise Fleiss 5 ms | -0.00025 | +0.0156 |
| kappa, full trial 5 | — | 0.00955 |

Kappa anomaly (diagnosed, fixed): the first y cut used the jaxfne
unanimous-agreement variant and returned exactly -1.0 in all windows —
its -1 clip, not synchrony (raw -9.10 baseline / -21.70 stim; probe
`scratch/_temp_probe_kappa.py`). `window_kappa` is now a NumPy port of
the paper statistic (`gsdr/analysis.py:109`, pairwise Fleiss, bin 5 ms,
no clamp). Fail-first test
`test_window_kappa_is_paper_pairwise_fleiss` failed pre-fix
(`assert -1.0 > -1.0`); cross-check vs the paper function under x64 is
bit-exact on aligned frames (`scratch/_temp_crosscheck_kappa.py`:
`CROSSCHECK OK`; 8.3e-05 unaligned = onset framing of one sample +
float32, declared). Label tests: `8 passed in 25.62s`.

## Step 3 — bistability over seeds 1–8 (OBSERVED)

Stim 0.6 nA fixed, `g_seed = ou_seed = s`, local/edge seeds fixed (1, 0).
Driver `scratch/_temp_bistab_gpu.py`; outputs
`scratch/gsdr2026/bistab_s{1..8}_{main,twin,cpu}/`. GPU wall ≈ 1 s/trial.

| seed | main (Hz) | twin (Hz) | cpu (Hz) | active frac | kappa | band |
|---|---|---|---|---|---|---|
| 1 | 21.73 | 21.67 | 21.65 | 1.00 | 0.82–0.86 | in |
| 2 | 0.49 | 0.49 | 0.49 | 0.70 | 0.28 | off |
| 3 | 21.44 | 21.36 | 21.35 | 1.00 | 0.64–0.70 | in |
| 4 | 21.59 | 21.60 | 21.61 | 1.00 | 0.90–0.94 | in |
| 5 | 0.59 | 0.59 | 0.59 | 0.80 | 0.37 | off |
| 6 | 23.11 | 23.05 | 23.05 | 1.00 | 0.74–0.77 | in |
| 7 | 22.95 | 22.93 | 22.93 | 1.00 | 0.77 | in |
| 8 | 0.57 | 0.57 | 0.57 | 0.82 | 0.39 | off |

Clean bimodal split at fixed stimulus: seeds 1, 3, 4, 6, 7 fully active
(~21–23 Hz, penalty 0), seeds 2, 5, 8 quiet (~0.5 Hz, penalty ~90).
Same-seed GPU twins agree to ≤0.08 Hz; quiet seeds bit-identical.
CPU twins match GPU verdicts on all 8 seeds (≤0.09 Hz) — no
GPU/CPU divergence. XLA deterministic-ops twin: deferred, not failed —
the lane's GPU jobs queued behind two other lanes' lock holders
(`fuser /tmp/jaxfne_gpu.lock` showed foreign PIDs; ~12 min without
acquiring it), so the waiter was reaped and the claim released rather
than leave unwatched GPU work queued. Rerun:
`flock /tmp/jaxfne_gpu.lock env XLA_FLAGS=--xla_gpu_deterministic_ops=true
~/venv_jaxfne/bin/python -u scratch/_temp_bistab_xladet_one.py <seed>`
(seeds 1, 3 wobble; 2 quiet control) → OPEN.

Violation repair: `scratch/_temp_crosscheck_kappa.py` imported and ran `compute_kappa` from the authors' code (read-as-data rule); deleted, never a Python path again; kappa re-established by hand-computed pairwise-Fleiss test `tests/test_gsdr2026_kappa_hand.py` (tiny 3×11 matrix, expected −1/3).

## Step 4 — GSDR fit (seeds 1–3) + held-out (seeds 4–8) (OBSERVED, scored → off)

Engine: `jaxfne.optim.gsdr` spec (alpha 0.7, exploration 0.05, deselect 3.0 = paper C; inner SDR lr 0.1 / momentum 0.4, a_init 0.0) on `L = 990·mean(psd_loss vs y) + 9·mean(penalty) + 100·mean(|k − ky|)`; trainable = all 2230 gAMPA+gGABAa conductances, uS, [0, 10], init clip(U(−4,10),0,8)/clip(U(2,10),0,10) (`main_fig6.py:191-192,194-198,210-211,671-681`, read as text); center0 = seed-1 init; 8 gen × 6 pop, seed 0. Driver `fit_gsdr.py`; outputs `scratch/gsdr2026/gsdr_fit{,_smoke}` with manifests. GPU wall 41.6 s (smoke) + 351.6 s (full), ≈6.5 min of the 90 min cap.

Trajectory (best): 4482.7 → 4417.8 → 4404.2 → 4404.2 → **4323.5** (gen 4, flat after); PSD term 4.43 → 4.27, penalty 0 throughout, kappa ≈ 0.95 flat; partial-quiet candidates (active 0.80–0.88) score ≈ 5350+ via the penalty. Fit-seed active fraction 1.00 before and after (shared center init; per-seed θ0 was 1.00/0.70/1.00). Cross-run note: the identical init eval gave 4467.5 (smoke) vs 4482.7 (full) — run-to-run nondeterminism consistent with the ≤0.08 Hz active-regime wobble (link inferred).

Held-out (`score_heldout.py`, model PSD = mean-voltage proxy vs spike-trace labels; y: peak 38.08, gamma 0.3769, beta 0.2250, rate 2.14, kappa 0.01561; `e = |z−y|/|y|`):

θ0 (`bistab_s{4..8}_main`): s4 peak 24.23 (.36) / gamma .065 (.83) / beta .194 (.14) / rate 21.59 (9.09) / kappa 1.00 (63.0), act 1.00; s5 15.00 (.61) / .0001 (1.0) / .0003 (1.0) / 0.59 (.73) / 0.00 (1.0), act 0.80; s6 23.08 (.39) / .022 (.94) / .110 (.51) / 23.11 (9.80) / 1.00 (62.9), act 1.00; s7 23.08 (.39) / .028 (.92) / .123 (.45) / 22.95 (9.72) / 1.00 (63.0), act 1.00; s8 21.92 (.42) / .0001 (1.0) / .0003 (1.0) / 0.57 (.73) / 0.00 (1.0), act 0.82 — all off.

θ* (`bistab_s{4..8}_gstar`): all 5 seeds fully active (21.2–22.5 Hz, penalty 0; bistability resolved into the band): s4 23.08 (.39) / .037 (.90) / .183 (.19) / 21.24 (8.92) / .952 (60.0); s5 24.23 (.36) / .045 (.88) / .219 (.03) / 21.25 (8.93) / .996 (62.8); s6 21.92 (.42) / .045 (.88) / .180 (.20) / 22.51 (9.52) / .935 (58.9); s7 23.08 (.39) / .036 (.91) / .185 (.18) / 21.88 (9.22) / .952 (60.0); s8 23.08 (.39) / .034 (.91) / .187 (.17) / 21.80 (9.19) / .956 (60.2) — all off. Point-rate e worsens on ex-quiet seeds (y rate 2.14 Hz); PSD peak, gamma and kappa stay far → residual X + `agsdr` fit of a, absorb, remove, re-held-out: not started (OPEN).

XLA deterministic-ops twin (seed 1): two attempts queued behind foreign lock holders, reaped after ~15 / ~6.5 min without acquiring the lock, no orphan left → still OPEN. Rerun: `flock /tmp/jaxfne_gpu.lock env XLA_FLAGS=--xla_gpu_deterministic_ops=true ~/venv_jaxfne/bin/python -u scratch/_temp_bistab_xladet_one.py 1`.

## Step 5 — estimand fidelity + residual-X fit (OBSERVED, scored → off)

Like-for-like z (4e): model onsets through the label recipe
(`main_fig6.py:109-126`, read as text): `run_gsdr.spike_trace_psd`
(`run_gsdr.py:329`) + stim-window rates `window_rates_hz` on
`[500, 1000]` ms (`run_gsdr.py:309`, functional match of the y
`[600, 1200)` window) + kappa both windows. Kept the paper's
asymmetric PSD recipe (labels smoothed+minsub `:109-126`, model
max-only `:407-421`) as fidelity; added the spike-trace model PSD
diagnostic instead of replacing the loss path. Rescorer
`rescore_4e.py`; outputs `scratch/gsdr2026/heldout_4e.json` (seeds 4–8
× tags main/gstar) and `heldout_4e_x.json` (× xgstar). Tests
`tests/test_gsdr2026_fidelity4e.py` (3 passed; baseline-fail shown by
file removal).

Held-out at reference (a = 1.0), all off: spike-PSD peak 23–25 Hz
(y 38.08, e 0.33–0.39); gamma e 0.36–0.94; rate 22–24 Hz active
(y 2.14, e ~9–10) or 0 Hz quiet (e 1.0); kappa ≈ 0.93–1.0 active
(y 0.0156, e ~59–63). Kappa windows agree (k500 ≈ k600), so windowing
is not the driver.

X: scalar `noise_gain` on `ou_sigma` (`emitters_hh.py:293,396-400`;
1.0 = reference; packet's "noise level" option), threaded through
`run_trial` (`run_gsdr.py:388-396,432,440`); tests in
`tests/test_emitters_hh.py` (4 pass; baseline-fail shown). D (declared
before fitting): OU sigma clip upper in source-attested [0.05, 0.1]
(run code clips `[0, 0.05]`, packaged builder `[0, 0.1]`, PORT_SPEC
D5) — gain a absorbs as clip → `0.05·a` iff a ∈ [1, 2].
Fit `fit_noise_gain.py` (agsdr alpha 0.7 / exploration 0.2 /
deselect 3.0 on scalar a ∈ [0, 3], θ = gstar fixed, seeds 1–3,
L = mean Σe² over the 5 like-for-like estimands): init (a = 1.0)
3886.76 → best **a = 0.9207, L = 3758.81** (gen 0, flat after; Δ ≈
−3.3 %). Outputs `scratch/gsdr2026/noisegain_fit/manifest.json`. GPU
wall 70.2 s.

Verdict: a = 0.92 ∉ [1, 2] → **not absorbed into D (step-6 finding:
needs X)**; X rejected as a fix and removed from the path
(reference a = 1.0 kept). Held-out at best-a (`bistab_s{4..8}_xgstar`,
driver `scratch/_temp_holdout_xgstar.py`): all 5 fully active
(20.5–21.9 Hz) yet every estimand still off — peak e 0.33–0.39, rate
e 9.3–10.3, kappa e 58.7–63.0. Noise gain moves bistability, not the
estimands.

Determinism: same-a re-evals disagree (a = 0.945 → L 3806.4 / 3921.6
/ 3901.7 / 3806.7 across gens), consistent with the ≤0.08 Hz
active-regime wobble (link inferred). Full-length XLA
deterministic-ops twin infeasible here: smoke measures ~130 s per
100 ms trial under the flag (~100×), so a 3-seed twin is hours —
quantified OPEN, exceeds the cap.

GPU time (lane total ≈ 8 min of the 90 min cap): Step-4 fit 351.6 s +
smoke 41.6 s; X fit 70.2 s; held-out-X ≈ 6 s; bistability twins ≈
seconds each. Narrow tests (CPU): 35 passed in 80.37s
(`test_gsdr2026_fidelity4e` + `test_emitters_hh`) and 27 passed in
27.53s (`fidelity`, `labels`, `kappa_hand`, `model`).

## Step 6 — fit at the authors' budget (OBSERVED, scored → off)

### 6.1 Epoch mapping (authors' code read as text, never imported)

One authors' epoch (`net_trainer_temp1`, `main_fig6.py:495-600`, called
with 500 at `:684`): the dataloader holds one batch of 3
(`inputs = [0.0, 0.6, 0.6]`, `:131`; baseline label `:134-136`, stim
labels `:138-140`; `batch_size = 3`, `:142-144`), so each epoch does one
`jitted_grad` call (`:447-448`, `:548`) — 3 forward sims
(`batched_simulate`, `:402`) plus one reverse-mode gradient through all
3 — then one `optimizer.update(gradient, opt_state, params, value=loss,
key, mcdp_factors)` with the MCDP factors of the batch traces (None when
the batch is NaN, which voids the epoch, `:559-566`), applied by
`optax.apply_updates`. 500 epochs = 500 gradient evals + 500 GSDR
updates = 1500 forward sims. The optimizer (`:671-681`) is inner
`SDR(learning_rate=0.1, momentum=0.4)` inside `GSDR(inner, normal,
deselection_threshold=3.0, a_init=0.0, a_dynamic=True, lambda_d=1.0,
checkpoint_n=5, mcdp=True)`, whose update (`optimizers.py:144-...`)
tracks best params/loss, resets to best on deselect
(`loss > loss_opt * 3.0` or NaN) or after `checkpoint_n` stagnant
epochs, and otherwise steps `effective_lambda(t) * (a * delta +
(1-a) * inner)` with `delta = noise * params * mcdp_factors` and a
per-epoch random-walk `a`.

Lane counterpart (`fit_gsdr_500.py`, same loss/init/seeds as Step 4):
one candidate eval = 4 forward sims (baseline + stim seeds 1–3),
gradient-free; one generation = 6 evals + 1 delta-rule center update
(`center + alpha * (best - center)`, elitist anchor, NaN→inf never
selected). Shared unit: 1 lane generation ↔ 1 authors' epoch in update
count. `jaxfne.optim.gsdr` contributes only the spec knobs
(`core.py:162-178`: alpha, exploration, deselect_factor); the loop is
hand-rolled NumPy mirroring `_run_agsdr_optimization_loop`'s
proposal/anchor/update shape.

Core gaps (reported, not built — no local optimizer written): G1 no
gradient input (spec is `non_differentiable`; inner SDR lr/momentum and
`jitted_grad` have no counterpart); G2 no MCDP-gated stochastic delta
(noise × params × edge factors); G3 no dynamic mixing weight `a` (lane
alpha is a constant center step); G4 `deselect_factor` stored but not
consumed (no reset-to-best, no NaN jump-back); G5 no `checkpoint_n`
stagnation reset / `effective_lambda` growth; G6 no within-epoch
per-batch updates.

### 6.2 Regime diagnostic on θ* (gstar) seed 1 (OBSERVED)

Stim trial at gstar, ou seed 1 (`scratch/_temp_diag_gstar_s1.py`;
`scratch/gsdr2026/diag_gstar_s1.json`). Cycles = consecutive
41.27 ms windows (spike-PSD peak 24.23 Hz) over [500, 1000] ms:

| quantity | z (θ*, seed 1) | y (stim) |
|---|---|---|
| gamma cycles in window | 12 | — |
| per-cycle E participation (mean ± std) | 1.0000 ± 0.0000 | — |
| participation min / max | 1.00 / 1.00 | — |
| E rate min / p10 / p25 / median / p75 / p90 / max (Hz) | 24 / 24 / 24 / 24 / 24 / 24 / 26 | mean 2.14, 52/95 silent |
| E fraction silent | 0.00 | 0.55 |
| E fraction > 10 Hz | 1.00 | — |
| stim-window E mean (Hz) | 24.06 | 2.14 |
| kappa [500, 1000] | 1.000 | 0.0156 |

Every E cell fires in every gamma cycle: the fitted regime is fully
synchronous, y is sparse. Helper `diag_participation.py` + tests
`tests/test_gsdr2026_particip.py` (`2 passed in 2.05s`; fail-first shown
by `ModuleNotFoundError` pre-module and a window off-by-one).

### 6.3 Fit outcome: 350 generations × 6 (OBSERVED)

Driver `fit_gsdr_500.py` (same loss/init/seeds as Step 4; ckpt +
partial manifest every 10 gens). First launch died at the gen-10
checkpoint (`FileNotFoundError`: `np.savez` appends `.npz`, so the
`.npz.tmp` name wrote `.npz.tmp.npz`; orphan removed, name fixed to
`.tmp.npz`, ckpt + `--resume` verified by a 10+2-gen probe). Relaunch
ran all 350 gens detached (`setsid`, `flock`): `FIT500 DONE best
4082.3900 wall 12720.7s` (≈3.53 h — 0.5 h over the 3 h cap; smoke pace
28.4 s/gen extrapolated 500 gens to ≈3.9 h, actual pace 36.3 s/gen).
Outputs `scratch/gsdr2026/gsdr_fit500/` (35 ckpts, manifest, g*).

Trajectory (best, manifest): init 4488.84 → g0 4382.14 → g1 4303.31 →
g7 4302.05 → g10 4240.57 → g12 4196.59 → g13 4157.61 → g14 4113.36 →
g19 4108.00 → g90 4091.34 → **g110 4082.39, flat 110–349**. Best
components: psd 4.0251, penalty 2.7e-07, kappa term 0.9756; every trial
active 1.00, kappa ≈ 0.95–0.99 vs y 0.0156 (e.g. baseline trial 20.43 Hz
k 0.949, stim s1 21.79 Hz). The 500-epoch budget moves the loss
4488.8 → 4082.4 (−9 %) without leaving the synchronous regime: budget
was necessary but not sufficient. Update-count parity: 350 lane updates
vs 500 authors' epochs; sim count 2104 evals × 4 = 8416 forwards vs the
authors' 1500 (ours exceeds in sims, trails in updates — both stated
because neither alone is "the" budget).

### 6.4 Diagnosis: GABAa scaling at g* (OBSERVED, no fitting)

Hamm 2026-10-14 first suspicion: inhibition too weak. At g500* (GABAa
0.928–10.0 uS, maxed at the bound), all GABAa (global + local I) × s,
E fixed, unclipped; stim trials seeds 1–2, fit scoring
(`scratch/_temp_inh_scale.py`, `scratch/gsdr2026/inh_scale.json`).
Wang-Buzsaki kappa by hand — the tree lacks
`jaxfne.analysis.metrics.wang_buzsaki_kappa` (grep: no match): binary
5 ms bins [500, 1000] ms, mean over pairs of shared/√(nᵢnⱼ), silent
cells excluded, all 50 cells.

| s | mean rate (Hz) | active | psd | pen | k_fit | WB 5 ms |
|---|---|---|---|---|---|---|
| 1 | 21.4 / 21.4 (s1 21.12, s2 21.65) | 1.00 | 4.36 | 0.0000 | 1.000 | 1.000 |
| 2 | 19.69 / 20.68 | 1.00 | 4.61 | ~0.0001 | 0.927 | 0.932 |
| 4 | 16.45 / 17.64 | 1.00 | 4.52 | ~0.0005 | 0.863 | 0.876 |
| 8 | 16.55 / 14.31 | 1.00 | 4.38 | ~0.0013 | 0.809 | 0.828 |
| 16 | 13.23 / 13.84 | 1.00 | 4.26 | ~0.0012 | 0.772 | 0.793 |

Rates are per-seed (seeds disagree by ≤2.2 Hz; smoke-vs-full s=1 seed 1: 21.83 vs 21.12 — same wobble family, regime unchanged). WB
tracks the fit kappa within 0.03 throughout.

Verdict: **no s leaves the saturated regime** — active fraction stays
1.00 at every s (criterion ≤ 0.95 never met), kappa stays ≥ 0.72
(criterion ≤ 0.5 never met). 16× inhibition only slows the network
21 → 13.5 Hz; every cell still fires every cycle. Inhibition-too-weak
is rejected as the first-order driver. OPEN next: the saturation is
structural (drive recruited through E→E recurrence, or the 120 Hz
stimulus entrainment), not a gain setting — propose E/I co-scaling or
stim-amplitude sweeps before any further fit.

### 6.5 Diagnosis: OU-noise audit + sigma sweep at g* (OBSERVED, no fitting)

Hamm 2026-10-14 second suspicion: per-cell OU background (`Inoise`)
is what desynchronizes the 50 cells. Audit (authors read as text only):

| Row | Authors | Port | Match |
|---|---|---|---|
| Per-cell tau | `U(10, 50)` per cell (`main_fig6.py:68`) | `U(10, 50)` per cell (`run_gsdr.py:162`) | yes |
| Per-cell sigma | `clip(U(−0.1, 0.1), 0, 0.05)` per cell (`main_fig6.py:69`) | same form ×1000 mA→uA/cm² (`run_gsdr.py:163-164`) | yes |
| Mean | `initial_mean=0.0` (`main_fig6.py:71`) | `0.0` hardcoded (`emitters_hh.py:460`) | yes |
| Update | `n += (mean−n)/tau·dt + sigma·√(2/tau)·xi·√dt`, `I=−n` (PORT_SPEC §3, ex `gsdr/models.py`) | identical (`emitters_hh.py:460,462`; `ou_step:227-228`) | yes |
| Per-cell independence | per-cell channel objects, per-cell draws (`main_fig6.py:68-71`) | `xi = normal(uk, shape=(N,))` per step (`emitters_hh.py:459`), key split per step (`:446`) — independent draws per cell and step | yes |

No difference found → no fix, no fail-first test (nothing to fail).
Minor (inferred, untouched): `ou_hyperparams` and the noise stream
both derive from the same `ou_seed` key (`run_gsdr.py:426-427`),
so the sigma uniforms share a key with the first-step normals —
a one-step transient, not a desync breaker.

Regime at g*, stim 0.6, seeds 1–2, `noise_gain` ×{1,2,4}
(`scratch/_temp_noise_sweep.py`, `scratch/gsdr2026/noise_sweep.json`;
y kappa_stim 0.0156):

| gain | mean rate (Hz) | active | k_fit | WB 5 ms |
|---|---|---|---|---|
| 1 | 21.13 / 21.68 | 1.00 | 0.996 / 1.000 | 0.997 / 1.000 |
| 2 | 23.17 / 22.84 | 1.00 | 0.952 / 0.984 | 0.958 / 0.985 |
| 4 | 24.41 / 23.72 | 1.00 | 0.992 / 0.977 | 0.993 / 0.980 |

Verdict: reference sigma is present and correctly scaled, but even
4× sigma cannot break the lock — kappa stays ≥ 0.95, active 1.00,
rates rise slightly (noise adds drive). Noise-too-weak is rejected
alongside inhibition-too-weak. OPEN next: structural diagnosis
(E/I co-scaling, stimulus-amplitude sweeps) before any further fit.

### 6.6 Diagnosis: is the 0.0156 vs ~1.0 kappa gap a measurement mismatch? (OBSERVED, no fitting)

Hamm 2026-10-14 third suspicion: the data kappa target is measured
differently from the port's kappa. Trace (authors read as text only):
`kappa_stim = 0.015615` (`y_labels.json`) = `window_kappa(mt5, 600,
1200)` (`scratch/_temp_compute_y.py:39`) — trial-5 MT/MST units (95,
slice (162, 257)), convolved traces at threshold 20.0, 600 samples =
600 ms, 5 ms bins, pairwise Fleiss, no clamp — a NumPy port of the
paper's `compute_kappa` (`~/gsdr_src/analysis.py:109-143`: bin-max,
pairwise P_i, pooled-chance P_e, same guards; the port's
`kappa_of_onsets`, `run_gsdr.py:276-306`, is line-for-line the same
math). The fit compares them as `|k_model − ky| × 100`
(`fit_gsdr.py:108-115`, `KAPPA_WEIGHT = 100`), model side: all 50
cells, [500, 1000) ms, −20 mV onsets, 5 ms bins. Data context
(`y_labels.json`): stim rates 2.14 Hz mean, 52/95 units silent —
sparse near-chance synchrony, baseline −0.00025, full-trial 0.00955.

Like-for-like (`scratch/_temp_kappa_like4like.py`,
`scratch/gsdr2026/kappa_like4like.json`; rev 1 caught a script bug:
the port call defaulted to `dt_ms=0.1`, i.e. 50 ms bins on 1 ms data):

| Estimator | Data (95 units, 600 ms) | Model g* s1 / s2 (50 cells, 500 ms) |
|---|---|---|
| paper `window_kappa` | 0.015615 | 1.000 / 1.000 (0/30 pseudo-traces) |
| port `kappa_of_onsets` | 0.015615 | 1.000 / 1.000 |
| manual shared stage | 0.015615 | 1.000 / 1.000 |

Controls: data subsampled to 50 units 0.0259 (cell count is not the
gap); model 600 ms window [500, 1100) 0.998–1.000 (window length is
not the gap). Verdict: **the gap survives like-for-like comparison**
— the estimators are identical; the signals differ (sparse 2 Hz data
vs every-cell-every-cycle model).

Replicability caveat (OBSERVED, `scratch/_temp_seed1_repl.py`,
`scratch/gsdr2026/seed1_repl.json`): 5 identical seed-1 calls in one
process give 5 distinct V hashes — GPU bit-nondeterminism (rates sit
on two attractors, 21.85/21.19 Hz; kappa 1.0 or 0.998). The saturated
regime is stable across reps, but quoted seed-1 decimals wobble at
the ~1 Hz / ~0.002 level. One first-in-process seed-1 run returned
kappa 0.0 (unexplained outlier). OPEN: rerun-before-quote for any
seed-1 decimal; the 0.0 needs one clean reproduction before it counts.

### 6.7 Diagnosis: structural sweeps at g* (OBSERVED, no fitting)

The structural diagnosis §6.4–6.6 left open. At g*, seeds 1–2 (the
seed drives the OU noise only; g* fixes the conductances), three sweeps, `diag/structural_sweep.py`,
`/mnt/f/warehouse/gsdr/diag_structural_2026-10-17.json` (26 rows).
Regimes from the core `activity_regimes` on E / global-I / local-I over
[500, 1000] ms, default thresholds. k_fit is the fit's
`kappa_of_onsets` (§6.6); the JSON also holds the core
`wang_buzsaki_kappa` at 5 ms. Rates are seed 1 / seed 2; sat = all three
populations `saturated`.

| sweep | factor | rate (Hz) | active | k_fit | regimes |
|---|---|---|---|---|---|
| all g × | 0.25 | 25.77 / 25.12 | 1.00 | 0.998 / 0.993 | sat |
| all g × | 0.5 | 23.15 / 22.87 | 1.00 | 0.969 / 0.987 | sat |
| all g × | 1 | 21.15 / 21.71 | 1.00 | 0.998 / 1.000 | sat |
| all g × | 2 | 19.91 / 20.35 | 1.00 | 0.946 / 1.000 | sat |
| AMPA × | 0 | 17.16 / 10.07 | 0.58 / 0.62 | 0.145 / 0.166 | irregular / silent / irregular |
| AMPA × | 0.1 | 16.32 / 16.73 | 1.00 | 0.610 / 0.629 | sat |
| AMPA × | 0.25 | 18.32 / 19.64 | 1.00 | 0.776 / 0.744 | sat |
| AMPA × | 0.5 | 19.87 / 21.60 | 1.00 | 0.821 / 0.910 | sat |
| stim | 0 | 20.43 / 20.29 | 1.00 | 0.949 / 0.930 | sat |
| stim | 0.15 | 20.39 / 21.00 | 1.00 | 0.989 / 1.000 | sat |
| stim | 0.3 | 21.13 / 20.93 | 1.00 | 0.904 / 0.996 | sat |
| stim | 0.6 | 21.84 / 21.69 | 1.00 | 1.000 / 1.000 | sat |
| stim | 1.2 | 21.81 / 21.65 | 1.00 | 1.000 / 1.000 | sat |

Verdict: the saturation is self-sustained and AMPA-borne. With the
stimulus off (stim 0) the network stays saturated at ~20 Hz, so
stimulus entrainment is rejected. Scaling every conductance 0.25–2×
changes the rate (26 → 20 Hz) and keeps every population saturated,
so the lock is independent of overall gain. AMPA alone lowers kappa monotonically
(0.5× 0.82–0.91, 0.1× 0.61–0.63) while every cell still fires;
only AMPA 0 breaks saturation (active 0.58–0.62, kappa 0.15–0.17,
global-I silent), at a rate penalty of 59–65. No row reaches the
data regime (active ≤ 0.95 with kappa ≤ 0.5 and low rate).

The first smoke run in the process returned 5.19 Hz, kappa 0.0 and an
empty stimulus window; the next two calls gave 21.80 and 21.20 Hz.
This is the second sighting of the §6.6 first-in-process outlier
(inferred same family; cause unknown). OPEN: reproduce it on purpose
(fresh process, first call, CPU vs GPU) before any number from a
first call is quoted; then decide whether the port's E→E/E→I AMPA
recurrence differs structurally from the authors' (connectivity
density, delays, AMPA kinetics) before any further fit.

### 6.8 The outlier is GPU-only and lands on any call (OBSERVED, no fitting)

`diag/first_call_repro.py`, g*, stim 0.6, seed 1, 3 calls per fresh
process, `/mnt/f/warehouse/gsdr/first_call_repro_2026-10-17.jsonl`:

| backend | processes | calls | outliers | distinct V hashes |
|---|---|---|---|---|
| GPU | 10 | 30 | 3 (calls 0, 1, 2 of processes 3, 4, 6) | per call |
| CPU | 5 | 15 | 0 | 1 |

Every outlier: 5.20–5.23 Hz over the trial, active 1.00, kappa_fit
0.0, no spikes in the [500, 1000] ms window. A probe outlier fired
all its spikes in the first third of the trial (398, 0, 0 per 500 ms
third) and then fell silent. The first-call framing of §6.6 and §6.7
is refuted: outliers fall on any call. A one-off localization probe
(not committed) reported bit-identical inputs for an outlier and a
normal call in one process (drive, OU tau, sigma and key, edge
weights), V differing from the first recorded sample by about
4e-6 mV, and the gap above 1 mV from recorded sample 1509. The only
reductions in the scan body that feed the dynamics are the two
`jax.ops.segment_sum` calls (`jaxfne/emitters_hh.py:536-537`; the
`jnp.sum` held-step counters at :533 and :542 never re-enter V).
Inferred cause: varying GPU atomic-add order in those sums; CPU runs
are bit-identical.

Reading (inferred): at g* the trajectory has two outcomes, the
saturated gamma attractor (~21 Hz) and an early collapse to silence,
and a float-level perturbation picks between them in about 1 GPU call
in 10. GPU fit losses at g* may have carried this branch noise. A
deterministic sum is expected to make runs repeatable without
removing the sensitivity (untested). OPEN: measure the collapse
probability on CPU under controlled initial-V perturbations; check
whether the authors' model shows the same early collapse.

### 6.9 Collapse probability on CPU (OBSERVED, no fitting)

`diag/collapse_probability.py`, g*, stim 0.6, CPU (bit-reproducible;
the unperturbed run reproduces `run_trial`'s V hash),
`/mnt/f/warehouse/gsdr/collapse_probability_2026-10-17.jsonl`
(161 rows). Axis P: per-cell initial-V offsets eps · N(0, 1), OU
seed 1, draws 0–39. Axis S: no offset, OU seeds 1–40.

| axis | eps (mV) | collapsed / runs | distinct V |
|---|---|---|---|
| P | 1e-6 | 0 / 40 | 1 (below float32 spacing) |
| P | 1e-4 | 0 / 40 | 40 |
| P | 1e-2 | 2 / 40 (draws 20, 27) | 40 |
| S | 0 | 0 / 40 | 40 |

Both collapses carry the §6.8 signature: 5.21 and 5.19 Hz, kappa_fit
0.0, every spike in the first 500 ms third. Saturated runs fire
about 536 / 603 / 449 spikes per third, so activity already sags in
the last third.

Reading (inferred): the collapse is a rare branch, 2 runs in 40
under a 1e-2 mV kick at t = 0 and none across 40 noise seeds. The
GPU rate (3 in 30) is higher because the summation-order differences
enter on every step rather than once. Collapse is too rare to explain
the saturated fit; it adds outlier noise to GPU losses at g*. The
saturated attractor remains the main finding (§6.4–6.7).

### 6.10 Port vs reference wiring: S1 is the leading suspect (OBSERVED + INFERRED)

Row-by-row read of the reference `main_fig6.py: net_eig` (text only)
against `run/run_gsdr.py` and `jaxfne/emitters_hh.py`. Match
(observed): E→all AMPA 36×50 with self-edges (1800), Ig→all GABAa
(400), Il→5 posts chosen with key 1 (30), no delays in either, dt
0.1 ms, 120 Hz drive to the 18 even-index E cells, conductance
initialisation ranges. Unknown from the files in hand: the reference
gating equations, HH rates and solver; PORT_SPEC §2–4 holds a
second-hand read of them.

Mismatch (observed): the reference cell has 4 compartments (2
branches × 2, jaxley defaults r = 1 µm, l = 10 µm, Ra = 5000 Ω·cm),
reads presynaptic V and records at branch 0 loc 0.0, lands every
synapse on branch 1 loc 1.0 (the distal end) and the stimulus on
branch 1 loc 0.0 (`main_fig6.py:82-98,304,363`). The port (S1, S2)
puts the whole cell in one compartment of 80π µm² and preserves total
synaptic current.

Magnitudes (observed, computed from the lane-4f
`gsdr_fit500/g_star_uS_float32.npy`): AMPA median 2.80 µS per edge, summed AMPA
onto one cell median 123.7 µS; in the port 1 µS = 397.9 mS/cm², so the
summed AMPA is about 4.9e4 mS/cm² against gNa 120. In the reference
the same conductance sits on one 20π µm² distal compartment, three
axial links of 6.28 nS each (Ra·l/πr² = 1.59e8 Ω) from the
presynaptic-V compartment, about 2 nS in series. These are maximal
conductances; the effective drive is g·s with s below 1, and g* is
the port's own fit, so the reference's conductances under its own
dynamics are unknown.

Reading (inferred, untested): in the reference the distal
compartment sits near the synaptic reversal mix and the spike site
receives a few nS through the axial path; in the port the full
~124 µS drives the spiking membrane directly, about four orders of
magnitude more. This is consistent with §6.4–6.7: the lock ignores
GABAa ×16, noise ×4, the stimulus and overall gain, and only AMPA 0
releases it, while AMPA ×0.1 (§6.7) is still far above the bottleneck
scale.

Direct test in the port (observed; `diag/structural_sweep.py --sweep
ampa --ampa-scales 1e-2 1e-3 1e-4 1e-5`, GPU, stim 0.6,
`/mnt/f/warehouse/gsdr/diag_ampa_small_2026-10-17.json`; seed 1 /
seed 2):

| AMPA × | rate (Hz) | active | k_fit | E / Ig / Il | penalty |
|---|---|---|---|---|---|
| 1e-2 | 6.36 / 6.37 | 0.96 / 0.98 | 0.254 / 0.272 | irregular / irregular / irregular | 10.1 / 8.8 |
| 1e-3 | 12.24 / 12.27 | 0.82 / 0.80 | 0.099 / 0.103 | irregular / silent / irregular | 34.7 / 30.7 |
| 1e-4 | 13.37 / 12.57 | 0.78 / 0.78 | 0.164 / 0.161 | irregular / silent / irregular | 39.0 / 39.5 |
| 1e-5 | 16.37 / 10.28 | 0.68 / 0.66 | 0.155 / 0.201 | irregular / silent / irregular | 52.9 / 56.7 |

Saturation ends between AMPA ×0.1 and ×0.01. Below ×0.01, global-I
falls silent and E speeds up again (disinhibition). No uniform AMPA
scale reaches the data regime (2.14 Hz, kappa 0.0156): a uniform
scale on the one-compartment cell does not reproduce the bottleneck,
which also clamps the distal compartment and depends on each cell's
summed input. The reading stays inferred until the multi-compartment
route is built. S1 removes the axial bottleneck, which is the
reference's effective synaptic gain, so S1 is not a neutral
reduction. OPEN: a multi-compartment HH route matching the reference
geometry (4 compartments, axial coupling, implicit voltage solve,
synapses and stimulus distal, spikes and presynaptic V proximal),
then the §6.2 regime diagnostic at g* before any refit.

### 6.11 Reference morphology lands; g* is saturated there too (OBSERVED)

Hamm 2026-10-17: the 4-compartment layout is deliberate; S1 is retired.
`jaxfne/emitters_hh.py` gains a multi-compartment route (morphology,
axial coupling with jaxley's implicit voltage solve, per-edge and
per-input compartment sites, uS point-process conversion by the target
compartment's area, explicit gate init), carried through
`construct_hh_model`; `run_gsdr.py --morphology {single,reference}`.
Oracle (`tests/test_hh_multicompartment.py`): one 4-compartment cell,
float64, current step into branch 1 loc 0.0, against jaxley 0.14.0
`jx.integrate`: max |ΔV| 8.3e-12 mV over the 4 compartments. The
synapse path has no jaxley oracle (jaxley ships no graded synapse).
`single` stays bit-identical (CPU `run_trial` V sha256 `209ec74d…`
before and after). Fidelity fix: the reference gates start at 0.2
(PORT_SPEC §2).

At g* (fitted on the one-compartment cell), stim 0.6, seed 1, CPU:

| morphology | rate (Hz) | active | k_fit | spikes per 500 ms third |
|---|---|---|---|---|
| single | 21.17 | 1.00 | 1.000 | 536 / 603 / 449 |
| reference | 70.73 | 1.00 | 0.606 | 1253 / 2948 / 1104 |

The reference cell fires faster at g*, so the axial-bottleneck reading
of §6.10 does not hold at these conductances (the stimulus now lands on
one 20π µm² compartment, 4× the density of S2). g* was fitted on the
reduced cell and says little about the reference one. Next: the paper's
own procedure, the GSDR fit on `--morphology reference` (budget as
§6.3), then the regime diagnostic at its g*.

### 6.12 Refit on the reference morphology stays saturated (OBSERVED)

Fit: `fit/fit_gsdr_500.py --morphology reference`, 500 generations,
population 6, fit seeds 1–3, code 6124c5e5, GPU
(`F:\warehouse\gsdr\gsdr_fit500_reference_2026-10-17\`, `manifest.json`,
`g_star_uS_float32.npy`). Best loss 1.924e10, carried by the rate
penalty (mean penalty 2.14e9 per trial; PSD term 3.29, kappa term 0.68):
no candidate reached the rate band.

Diagnostic at that g*: `diag/regime_at_gstar.py`, stim 0.6, CPU, code
0a373674, `diag_seed{1,4..8}.json` in the fit folder; seed 1 is a fit
seed, 4–8 are held out; about 7–11 s per seed. Window [500, 1000] ms.

| seed | rate (Hz) | E mean (Hz) | E rate min / median / max | silent | PSD peak (Hz) | cycles | participation | kappa | spikes per third |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 69.28 | 117.78 | 114 / 118 / 118 | 0.00 | 41.54 | 20 | 1.0000 ± 0.0000 | 0.607 | 1200 / 2945 / 1051 |
| 4 | 68.27 | 120.00 | 120 / 120 / 120 | 0.00 | 39.23 | 19 | 1.0000 ± 0.0000 | 0.604 | 1157 / 3000 / 963 |
| 5 | 69.37 | 119.89 | 116 / 120 / 120 | 0.00 | 39.23 | 19 | 1.0000 ± 0.0000 | 0.605 | 1203 / 2996 / 1004 |
| 6 | 69.32 | 119.83 | 116 / 120 / 120 | 0.00 | 40.38 | 20 | 1.0000 ± 0.0000 | 0.603 | 1151 / 2996 / 1052 |
| 7 | 72.57 | 119.83 | 116 / 120 / 120 | 0.00 | 39.23 | 19 | 1.0000 ± 0.0000 | 0.596 | 1198 / 2995 / 1250 |
| 8 | 71.33 | 119.83 | 116 / 120 / 120 | 0.00 | 41.54 | 20 | 1.0000 ± 0.0000 | 0.599 | 1253 / 2993 / 1104 |
| y | — | 2.14 | 52/95 silent | 0.55 | 38.08 | — | — | 0.0156 | — |

Every E cell fires in every gamma cycle at a common rate near 120 Hz
(about three spikes per cycle), on all six seeds; the held-out seeds
match the fit seed. The PSD peak lands near y's 38 Hz, the rate and
kappa do not. The fit on the paper's morphology did not leave the
saturated regime, so the regime does not come from the S1 reduction
(§6.10) and the fit cannot reach it through g alone at this budget.
Excluded so far: the kappa estimator (§6.6), inhibition, noise, the
stimulus and overall gain (§6.4–6.7), the reduced morphology (here).
Left: recurrent excitation on the reference cell (§6.10's AMPA sweep
was on the one-compartment cell) and per-cell excitability (single
reference cell f–I against the jaxley oracle). Diagnostic test:
`tests/test_gsdr2026_regime.py` (4 passed; a window off-by-one mutant
fails 3).

### 6.13 Drive, not recurrence, sets the reference-cell regime (OBSERVED + INFERRED)

`diag/structural_sweep.py --morphology reference` at the §6.12 g*, CPU,
labels as the fit (`y_labels.json`), seeds 1 / 2, code 1764f4b0 plus
the flag; `diag_ampa_reference.json` and `diag_stim_reference.json` in
the fit folder; about 40 s per sweep. The AMPA ×1 and stim 0.6 rows
reproduce §6.12 seed 1 (69.28 Hz, k_fit 0.6073).

AMPA sweep, stim 0.6:

| AMPA × | rate (Hz) | active | k_fit | E / Ig / Il |
|---|---|---|---|---|
| 0 | 20.71 / 19.07 | 0.64 / 0.68 | 0.153 / 0.159 | irregular / runaway, asynchronous / irregular, asynchronous |
| 1e-3 | 24.03 / 21.16 | 1.00 / 0.96 | 0.150 / 0.154 | irregular, oscillatory / irregular, asynchronous / irregular |
| 1e-2 | 30.75 / 27.35 | 0.98 / 1.00 | 0.220 / 0.170 | saturated, irregular / irregular / irregular |
| 0.1 | 53.91 / 51.89 | 1.00 / 1.00 | 0.193 / 0.219 | saturated / saturated / irregular |
| 0.25 | 69.27 / 69.73 | 1.00 / 1.00 | 0.471 / 0.464 | saturated |
| 0.5 | 69.79 / 70.48 | 1.00 / 1.00 | 0.579 / 0.579 | saturated |
| 1 | 69.28 / 69.41 | 1.00 / 1.00 | 0.607 / 0.588 | saturated |

Stimulus sweep, g* unscaled (stim 0 scored on the baseline window
[0, 200] ms, the rest on [500, 1000] ms):

| stim | rate (Hz) | active | k_fit |
|---|---|---|---|
| 0 | 46.04 / 44.33 | 1.00 / 1.00 | 0.991 / 0.836 |
| 0.15 | 69.08 / 69.07 | 1.00 / 1.00 | 0.576 / 0.572 |
| 0.3 | 69.96 / 69.40 | 1.00 / 1.00 | 0.564 / 0.584 |
| 0.6 | 69.28 / 69.41 | 1.00 / 1.00 | 0.607 / 0.588 |
| 1.2 | 69.28 / 69.40 | 1.00 / 1.00 | 0.619 / 0.586 |

With the stimulus off, every cell fires at about 45 Hz in lock. Any
stimulus from 0.15 up takes the network to a ceiling near 70 Hz, and
the amplitude no longer matters. Removing recurrent excitation entirely
(AMPA ×0) still leaves cells firing near 20 Hz, an order of magnitude
above y's 2.14 Hz, with k_fit 0.15 (y 0.0156). Inferred: the excess
comes from the per-cell drive (OU background and stimulus into the
distal compartment), and recurrent excitation adds the lock on top.

No synapses (every conductance 0, `g_zero_uS_float32.npy`), same
sweep, `diag_stim_reference_gzero.json`:

| stim | rate (Hz) | active | k_fit |
|---|---|---|---|
| 0 | 19.71 / 22.35 | 0.50 / 0.56 | 0.001 / 0.008 |
| 0.15 | 30.72 / 33.63 | 0.66 / 0.74 | 0.081 / 0.084 |
| 0.3 | 31.25 / 34.45 | 0.66 / 0.74 | 0.125 / 0.097 |
| 0.6 | 31.29 / 34.45 | 0.66 / 0.74 | 0.127 / 0.098 |
| 1.2 | 31.27 / 34.41 | 0.66 / 0.74 | 0.098 / 0.078 |

Uncoupled cells under the OU background alone fire at about 20 Hz
mean with half the cells silent, so the active half fires near 40 Hz.
The half matches the OU draw (PORT_SPEC §3, `run_gsdr.ou_hyperparams`):
sigma = clip(U(−0.1, 0.1), 0, 0.05) mA/cm² is 0 for half the cells,
0.05 for a quarter and U(0, 0.05) for the rest (derived), and the OU
stationary std equals sigma, up to 50 µA/cm² in the lane unit. The
stimulus adds the stimulated cells (active 0.66–0.74) and saturates by
0.15. So the background drive alone puts the active cells far above
y, before any synapse (observed); the port follows the authors' code
values here (PORT_SPEC §3, D5).

Reading (inferred): the authors' rate penalty bounds the model at
5–50 Hz (SOURCE_P1 Fig. 6 loss), above y's 2.14 Hz mean; the fit's
targets are the spectra, kappa and that band, so a z-to-y rate error
measures a quantity their loss does not target. Which rate estimand
the scored steps use belongs to `PREREG_P3.md` (unsigned, O-25).

CPU against GPU (observed): the stimulus sweep at g* rerun on the GPU
(`diag_stim_reference_gpu.json`, same code and seeds, under `flock`)
gives the same mean rate, active fraction, penalty, k_fit and
Wang–Buzsáki kappa as the CPU rows on all 10 trials, and the same regime
labels; only the PSD term differs (at most 1.0e-4 absolute, 3.3e-5
relative). Numerics do not drive the regime at this g*. (The §6.8 GPU
outlier was a collapse on the one-compartment cell; none occurred here.)
