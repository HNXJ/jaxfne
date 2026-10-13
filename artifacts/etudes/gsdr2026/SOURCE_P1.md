# Nejat, Sherfey, Bastos 2026 (GSDR): P1 source record

Status: draft from a Muse read of the article, PMC text and `HNXJ/GSDR` (read as data, nothing run), 2026-10-13. Unverified by the dispatcher; values marked unknown wait for the Methods tail and S1. Hamm is an author: gaps go to him.

All items labeled **observed** (with source) or **inferred**. No passage quotes more than 15 words.

## 1. The 50-neuron circuit

- Composition (observed, Results + `misc/main.py`): 36 excitatory + 14 inhibitory (8 global SST-like + 6 local PV-like). Paper: PV-like restricted to ~10% of neurons (Supp Fig 6); SST-like unrestricted.
- Cell model (observed, Methods Eq 1): `C_m dV/dt = −I_input − ΣI_intrinsic − ΣI_synaptic`, positive-outward-hyperpolarizing convention; intrinsic = Na/K/leak. Code (observed, `models.py:build_net_eig`): stock Jaxley `HH()` + per-cell `Inoise` OU channel, `radius=length=1.0`. Capacitance value and HH rate parameters: **unknown** (paper names `C_m` only; Jaxley defaults not read).
- Synapses — all values code-only (`models.py`), paper says only that rise/decay fixed per refs: `GradedAMPA`: E=0.0, tauD=2.0 in builders (class default 5.0), tauR=0.2, slope=5.0, Vth=−20.0, default g=2.5. `GradedGABAa`: E=−80.0, tauD=5.0, tauR=0.5, slope=5.0, Vth=−20.0, g=5.0. `GradedGABAb` (E=−95, tauD=200, tauR=10) defined but **unused** in circuit (observed: never connected in builders).
- Connectivity (observed, `models.py` + `main.py:net_eig`): E→all cells AMPA; global-I→all GABAa; local-I→random 10% subset (5/50 posts) GABAa, via `fully_connect`.
- Inputs/noise (observed both): paper — stimulus is 1 nA noisy DC step convolved with 120 Hz pulse train, 500 ms; background = stationary random current per neuron. Code (`main.py`, `simulation.py`): `noise_current_ac` (delay 500, dur 500, amp_n=0, amp_b=input, 120 Hz) to half of E (every other E cell); OU `Inoise` per cell. Amplitude mismatch: code inputs `[0,1,1]*0.6` (=0.6, observed) vs paper 1 nA.
- dt/duration/integrator (code-only, observed `main.py`): dt=0.1 ms, t_max=1500 ms, stim 500–1000 ms, `jx.integrate` with 2-level checkpointing. Paper values: **unknown** (Methods tail unretrieved). Paper spectra average per-neuron PSDs (observed, Results).
- Tuned conductances (code-only, observed `main.py`): init gAMPA∼U(−4,10)→clip[0,8], gGABAa∼U(2,10)→clip[0,10]; clamp bounds [0,10] both.

## 2. GSDR

- Update (observed, Methods Eqs 2–4 + Algs 1–2): `U_t = U + (D(1−α)+α)·R`; `D=(ΔU)(ΔL)`; per-weight `Δ=(L·D·δ(λ)(1−α)+α)·R_pre,post`; MCDP `R`=z-scored pre/post membrane-potential correlation with post shifted by synaptic tau; R=1 where undefined. α∈[0,1]: 0=supervised, 1=MCDP+stochastic; dynamic α random-walks ±0.1 uniform under same selection rule. Clamps: W→[0,max_g], α→[0,1].
- Genetic rule (observed): store best (L_opt,U_opt); revert if `L ≥ C·L_opt`; code adds stale-checkpoint revert (`checkpoint_n`) and NaN revert (observed `optimizers.py`).
- Hyperparameters (observed, code-only): package defaults C=2.0, a_init=0.5, checkpoint_n=10; Fig-6 run (`main.py`) C=3.0, a_init=0.0 dynamic, checkpoint_n=5, inner SDR lr=0.1/momentum=0.4, 500 epochs. Paper C value: **unknown**. `AGSDR` adaptive-α exists in repo only (inferred: post-paper addition).
- Fig 3 loss (observed): per-condition spectral ratio — cond1 gamma>beta, cond2 beta>gamma; trained weights, fixed taus; total gain ~constant, weight variance redistributed.
- Fig 4 loss (observed): beta/middle-gamma ratio; local weights frozen, 3 top-down gains trained (E, SST-like, PV-like), init 0; cond1 stays <1, cond2→~1.8 via +slow-I / −fast-I gains.
- Fig 6 loss (observed paper): normalized log-ratio of model vs data spectra (pre+stim) + one-sided MXE low-spike penalty (Eq 5, exact form **unknown**). Code realization (`pipeline.py`, `main.py`): label-weighted squared PSD error ×990 + exp firing-rate penalty (bounds 5/50 Hz, ×9) + |kappa| sum (×100); target grid 15–60 Hz / 40 bins.

## 3. Claims per figure

- Fig 2 (observed): 3 runs × 1000 iters on noisy uniform-perturbed Rastrigin; GSDR escapes local minima, GD/SGD stick. Qualitative plot, no numeric fields quoted.
- Fig 3 (observed): conditional objective met; E spiking suppressed in beta condition. Bands inconsistent (see §4).
- Fig 4 (observed): ratio →~1.8 in cond2; cond1 ratio drifts slightly pre-vs-post (stochasticity note).
- Fig 6 (observed): pre→mid(~trial 200)→post desynchronization; kappa falls during stim and below strong-PING control (Supp 15); stim to random half of E, 0–500 ms.
- Needs-data, out of scope (observed Fig 5 + README): MT/MST recordings (100 trials, 80 neurons, ~37.5–38 Hz evoked gamma, +2.5 Hz session drift per Supp 14, PFC beta ~20 Hz, baseline beta ~15–20 Hz, stim gamma ~35–40 Hz). Replication uses only its spectra as targets; raw `oxm0818` via Drive link.
- Supplements (observed mentions only): S13 = 1000-neuron augmentation; S15 = strong-PING control; S16/S17/S18 = Izhikevich baseline/stimulus/PSDs qualitative match; S12 = early/late spectra; S6 = connectivity/weights; S3 = Fig-4 spectra. Full S1 file **absent** (fetch 404).

## 4. Gaps, inconsistencies, jaxfne needs

- Gamma band (observed): 35–100 (Fig-3 text) vs 40–90 (Fig-3 caption) vs >35 (Fig-4 text) vs 55–75 middle gamma (Fig-4) vs 35–60 (code `band_definitions`). Beta: 15–30 vs ~10–30 vs 13–30 (code). **Inferred**: adopt per-task bands; Fig-6 code grid (15–60 Hz, 40 bins) is the replicable choice.
- Mismatches (observed): stim 1 nA vs 0.6; onset "time 0" vs 500–1000/1500 ms; "random half of E" vs every-other-E; local-I tauD 2.0 (`main.py`) vs 5.0 (`models.py`); OU tau/amp ranges differ between the two builders; GSDR defaults differ (C 2 vs 3, checkpoint 10 vs 5, a_init 0.5 vs 0.0).
- Absent (searched PLOS/PMC/methods text + repo raws): Eq-5 form, parameter Tables, paper dt/duration/integrator, HH capacitance, DynaSim-vs-Jaxley split, recording methods tail.
- jaxfne needs (inferred): Jaxley-backed HH emitters + OU-noise channel; graded AMPA/GABAa with tanh activation (Vth −20, slope 5) and settable tauR/tauD/E_rev; E→all / global-I→all / local-I→10% wiring; half-E 120 Hz-convolved step stimulus; per-neuron PSD + band-ratio metrics; kappa synchrony; MXE-style low-rate penalty; GSDR loop (SDR inner + shifted-correlation MCDP + α mixture + C-threshold revert + clamps); Izhikevich emitter variant for S16–18.
