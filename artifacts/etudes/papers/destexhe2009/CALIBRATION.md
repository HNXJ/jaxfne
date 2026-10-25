# Destexhe 2009 calibration C3 (baseline θ₀, seeds 1–3 + Fig. 6A control)

Baseline z at the paper's parameters (PORT_SPEC.md θ₀, PREREG_P3.md).
No fit performed; degrees of freedom D untouched. Verdict: OFF, unscored pilot (ruling
2026-10-18: scored against the unsigned draft `PREREG_P3.md`) —
no self-sustained activity at θ₀, so E1 fails and E2/E3 score a transient,
not an AI state. (Step 6 below supersedes this verdict: seeds 4–8 ran there as an unscored pilot
at the declared baseline, so they are spent; a scored run needs a new protocol on fresh seeds.)

## Provenance

- Code SHA `3dd02ed060390ecf711e04860a1d6d46e854ef83` (one SHA per output
  directory, enforced by `destexhe_api.write_run`).
- JAX `0.10.1`, devices `['cuda:0']` (GPU runs under
  `flock /tmp/jaxfne_gpu.lock`); scoring CPU-only.
- Rasters: `scratch/destexhe2009/s2/spikes_seed{1,2,3}_fig6b.npz`,
  `scratch/destexhe2009/s2ctrl/spikes_seed1_fig6a.npz`; manifests in the same
  directories; per-run reports `scratch/destexhe2009/report_*.json`.
- GPU wall: seed 1 `17.8 s`, seed 2 `24.4 s`, seed 3 `18.9 s`,
  Fig. 6A control `17.9 s` (plus a misconfigured control run `17.9 s`,
  `strong_adaptation=false`, discarded by deleting its directory and rerunning
  with the flag; manifests distinguish them).

## Observed: every run dies within 2 s (spikes per 1-s bin, 10 s free run)

| run | s0 | s1 | s2–s9 | ever active |
|---|---|---|---|---|
| fig6b seed 1 | 47802 | 0 | 0 | 1718/2000 |
| fig6b seed 2 | 48299 | 20995 | 0 | 1769/2000 |
| fig6b seed 3 | 31566 | 0 | 0 | 1306/2000 |
| fig6a seed 1 (b = 40 pA) | 3721 | 0 | 0 | 1114/2000 |

The 50 ms kick ignites a ~1–2 s reverberation (most cells fire), then silence.
Seed 2 stays strongly active through the second second (20995 spikes) and dies
abruptly between 2–3 s.

## Estimands (PREREG_P3.md bands; E3 absolute scale 0.01)

| Id | y (paper) | z seed 1 / 2 / 3 | e | band |
|---|---|---|---|---|
| E1 alive at 10 s | self-sustained | false / false / false | — | FAIL |
| E2 CV_ISI = 2.47 | 2.47 | 1.615 / 1.682 / 1.340 | 0.346 / 0.319 / 0.458 | off (≤ 0.05 to close) |
| E3 CC = 0.005 | 0.005 | 0.108 / 0.090 / 0.137 | abs 0.104 / 0.085 / 0.132 | off (≤ 0.01 to close) |
| E4 rates | RS < FS, RS 1–10 Hz | 2.35/2.56, 3.41/3.67, 1.51/1.85 | — | reported, not gated: holds |
| C1 Fig. 6A control | dead within seconds | dead, 0.19 Hz, CV 0.52 | — | predicted direction holds |

E2/E3 are computed over transient spikes only (all spikes fall in the first
1–2 s); against a self-sustained y they are off regardless. E1 fails on its
own: z is dead, and the E1 criterion needs z alive AND the control dead.

## Not a fit

Nothing was tuned: θ₀ exactly as preregistered (kick 300 Hz private ge-sized
sources to E only, 50 ms; F2 V-carried restart; 0.1 ms uniform delay). The
candidate list below is protocol for a follow-up, in PREREG D order.

## Follow-up protocol (declared, not executed)

1. F2-continuous-run hypothesis first (no parameter change): the phase-2
   restart drops in-flight synaptic conductances g (V carried; w/g/ref
   restart), so the free run must reignite from V alone where the paper's run
   continues with conductances intact. Test: one continuous 10.05 s run with
   the kick drive gated to the first 50 ms; if it sustains, F2 explains the
   miss and the port — not θ₀ — was wrong.
2. If still dead: kick rate 200–400 Hz (paper range), then kick shape (F5:
   which cells the kick targets), then ge/gi (paper states them exactly;
   deviation needs a reason). Adaptation b stays out of D.
3. Score any fit on held-out seeds 4–8 per PREREG_P3.md.

## Scorer correction during this calibration

`run_destexhe score` read "last 1 s" as `spikes[:, -win:]` (last win cells,
clamped to the whole raster) and reported every spiking run alive. Fixed to
`spikes[-win:, :]`; all reports above use the fixed scorer. Regression test
`tests/test_destexhe2009_port.py::test_score_alive_reads_last_second_of_time`
fails on the old line (`assert True is False`) and passes on the new one.

## Step 1: F2 continuous-run hypothesis (REJECTED 2026-10-14; unscored pilot, ruling 2026-10-18)

One continuous 10.05 s run per seed (1–3), kick Poisson drive gated to the
first 50 ms, no phase restart: `simulate_adex_gated_drive`
(`jaxfne/adex_destexhe.py`), reached through `destexhe_api.run_continuous`
(`fig6b_cont`). Rasters `scratch/destexhe2009/s2cont/spikes_seed{1,2,3}_fig6b_cont.npz`;
report `scratch/destexhe2009/report_s2cont.json`. GPU wall: seed 1 `57.6 s`,
seed 2 `115.0 s`, seed 3 `107.7 s` (plus a 20-cell GPU smoke `3.4 s`).

Observed per-1-s bins (free run, kick steps dropped):

| run | s0 | s1 | s2–s9 | ever active |
|---|---|---|---|---|
| cont seed 1 | 51411 | 14759 | 0 | 1747/2000 |
| cont seed 2 | 46989 | 8623 | 0 | 1721/2000 |
| cont seed 3 | 21741 | 0 | 0 | 1397/2000 |

Estimands (same bands; E3 absolute scale 0.01):

| Id | y (paper) | z seed 1 / 2 / 3 | e | band |
|---|---|---|---|---|
| E1 alive at 10 s | self-sustained | false / false / false | — | FAIL |
| E2 CV_ISI = 2.47 | 2.47 | 1.708 / 1.530 / 1.210 | 0.309 / 0.381 / 0.510 | off |
| E3 CC = 0.005 | 0.005 | 0.0957 / 0.0908 / 0.1485 | abs 0.091 / 0.086 / 0.144 | off |
| E4 rates | RS < FS, RS 1–10 Hz | 3.24/3.57, 2.76/2.86, 1.06/1.18 | — | reported, not gated: holds on transient |

Verdict (unscored pilot): still dead within 2 s (seed 3 within 1 s). The phase-2 restart
dropping w/g/ref does not explain the miss; F2 rejected. E2/E3 again score a
transient, not an AI state.

Fidelity fix shipped for this step (not a fit): the time-gated Poisson drive
was inexpressible with the core's stationary `ext_lambda`, so
`jaxfne/adex_destexhe.py::simulate_adex_gated_drive` adds `ext_active_steps`
(one-line gate on the Poisson counts; RNG split pattern unchanged, first K
steps bit-exact vs the ungated run). Tests
`tests/test_adex_destexhe.py::test_gated_drive_{prefix_matches_ungated_kick,full_open_matches_ungated,closed_matches_no_drive,refuses_bad_window}`
fail on baseline (`ImportError: cannot import name
'simulate_adex_gated_drive'`) and pass with it; API smoke
`tests/test_destexhe2009_port.py::test_run_continuous_writes_free_window_only`
covers `run_continuous`.

Fidelity audit of every other port-vs-paper difference (no other fix):

| Difference | Finding |
|---|---|
| integration step (NEURON, dt unstated, G4) | no fix: paper states no dt; port dt = 0.1 ms explicit Euler stays |
| refractory 2.5 ms clamp | matches: `adex_step` clamps Vreset for `round(2.5/0.1)` steps |
| adaptation reset w += b | matches: `w + dw + b` on spike |
| conductance quanta ge = 6 / gi = 67 nS | matches stated values; kernel adds `\|w\|` per arrival, exp decay |
| connection p = 2%, N = 2000 (1600 + 400) | matches; autapses excluded (paper silent, G10) |
| kick 300 Hz (paper range 200–400), 50 ms | matches stated range/duration |
| threshold VT + 5Δ (F3) | no fix: paper detection unstated |
| no delays (F1) | no fix: 0.1 ms is the kernel minimum (refused at 0) |
| NEURON simulator method | no fix: unstated details, not replicable |
| kick fraction: paper 2–10% of neurons vs port 100% of RS (80% of net) | REAL difference, kept as D (kick shape, F5), not silently fixed; also corrects the stale G5 record ("5% of cells" never implemented). Driving 8–40x the paper's cells biases toward over-kick, the wrong direction to explain death by under-drive |

Provenance: code HEAD `3dd02ed060390ecf711e04860a1d6d46e854ef83` with
uncommitted tree state at run time (new `jaxfne/adex_destexhe.py`
`simulate_adex_gated_drive`, new `destexhe_api.run_continuous` +
`run_destexhe fig6b_cont`; `s2cont/manifest.json` pins the HEAD SHA —
rasters are post-baseline tree, labelled calibrated reproduction).

## Step 2: kick-D exhausted (REJECTED 2026-10-14; unscored pilot, ruling 2026-10-18)

D in declared order (PREREG_P3.md): kick rate 200–400 Hz (paper range),
kick fraction shape F5 (fraction of RS kick targets; 0.05 of RS ≈ 4% of the
net is paper-range 2–10%). Delay has no range (kernel minimum 0.1 ms).
Continuous gated runs, 10 s free, seed 1 unless named; rasters
`scratch/destexhe2009/s2kf/spikes_seed*faithful_kf*.npz`; report
`scratch/destexhe2009/report_s2kf_pilot.json` (seed 1) plus CPU rescore
below (seeds 2–3).

Observed per-1-s bins (free run):

| run | s0 | s1 | s2 | s3 | s4–s9 | ever active |
|---|---|---|---|---|---|---|
| kf05 seed 1 (0.05 @300 Hz) | 48152 | 39970 | 39118 | 7420 | 0 | 1871/2000 |
| kf10 seed 1 (0.10 @300 Hz) | 50416 | 16324 | 0 | 0 | 0 | 1758/2000 |
| kf05r200 seed 1 (0.05 @200 Hz) | 43509 | 29632 | 0 | 0 | 0 | 1786/2000 |
| kf05r400 seed 1 (0.05 @400 Hz) | 46131 | 626 | 0 | 0 | 0 | 1706/2000 |
| kf02 seed 1 (0.02 @300 Hz) | 39953 | 0 | 0 | 0 | 0 | 1661/2000 |
| kf03 seed 1 (0.03 @300 Hz) | 43395 | 0 | 0 | 0 | 0 | 1637/2000 |
| kf05 seed 2 | 45839 | 4264 | 0 | 0 | 0 | — |
| kf05 seed 3 | 45257 | 18760 | 0 | 0 | 0 | — |

Seed 1 scores: kf05 CV 1.778 CC 0.0679 rate 6.73; kf10 CV 1.596
CC 0.0973 rate 3.34; kf05r200 CV 1.646 CC 0.0826 rate 3.66; kf05r400
CV 1.653 CC 0.1004 rate 2.34; kf02 CV 1.544 CC 0.1075 rate 2.00; kf03
CV 1.621 CC 0.1122 rate 2.17; all `alive_last_s=false`. Seeds 2–3 kf05:
seed 2 CV 1.595 CC 0.0891 rate 2.51 alive false; seed 3 CV 1.687
CC 0.0984 rate 3.20 alive false.

Verdict (unscored pilot): every kick-D point dies (best kf05 seed 1 lasts ~3.4 s, all
others ≤ 2 s). E1 FAILs everywhere, so E2/E3 again score a transient.
Kick rate × kick fraction cannot sustain; no θ* from kick-D.

Fidelity addition for this step: `subset_drive_kwargs`
(`jaxfne/adex_destexhe.py:146`) keeps the core drive validation and
rate/weight/receptor/delay/key, then keeps a seeded fraction of RS
 targets; `run_continuous` (`artifacts/etudes/papers/destexhe2009/destexhe_api.py:409`)
exposes it as `kick_fraction` plus `kick_rate_hz`, reached through
`run_destexhe.py fig6b_cont`. No paper value moved.

## Step 3: diagnostics + residual X_bg (DECLARED 2026-10-14; unscored pilot, ruling 2026-10-18)

Diagnostics (report `scratch/destexhe2009/report_s2diag.json`): `nokickv0`
(heterogeneous V0 in [−70, −50] mV, no drive) gives 0 free spikes;
`kf05v0` dies ≤ 2 s (44950/14226/0, CV 1.622 CC 0.0893 rate 2.96,
alive false). No AI attractor at θ₀ without drive.

X_bg form: ongoing background Poisson drive to E, ge-sized (6 nS AMPA),
gain a = `kick-rate-hz` (Hz). Implemented as `ongoing=True` in
`run_continuous`: the gated drive run with `ext_active_steps` full-open,
proven equal to the ungated run by
`tests/test_adex_destexhe.py::test_gated_drive_full_open_matches_ungated`.
X_bg is foreign to the paper (the paper states a 50 ms kick then a free
run); a is the residual gain.

a trajectory, seed 1, 10 s free (`scratch/destexhe2009/s2bg`,
report `scratch/destexhe2009/report_s2bg.json`):

| a (Hz) | bins s0–s9 | CV | CC | rate | alive |
|---|---|---|---|---|---|
| 1 | 0/0/0/0/0/0/2/0/0/0 | nan | nan | 0.0001 | false |
| 2 | 2/3/4/6/3/5/5/4/3/2 | nan | nan | 0.00185 | true |
| 3 | 17/15/11/15/12/14/13/20/14/12 | nan | −0.0005 | 0.00715 | true |
| 4 | 29/41838/4903/42/4065/46755/24017/33/40/37360 | 3.890 | 0.0711 | 7.95 | true |
| 5 | 32575/14506/34670/3124/30047/44178/30194/241/43367/66 | 2.800 | 0.0475 | 11.65 | true |
| 8 | 45844/37621/40312/33639/37017/36232/37693/32318/32893/44699 | 1.816 | 0.0179 | 18.91 | true |
| 12 | 46808/39416/36410/42748/42443/37026/35796/42149/38540/36604 | 1.760 | 0.0155 | 19.90 | true |
| 20 | 49915/49043/44458/42282/39840/40609/44835/41549/43424/44452 | 1.769 | 0.0099 | 22.02 | true |

Per-carrier rates (seed 1 labels: 1600 E + 400 PV): bg04 RS 7.77 FS
8.68; bg05 RS 11.37 FS 12.78; bg08 RS 18.39 FS 20.99; bg12 RS 19.38
FS 21.98; bg20 RS 21.50 FS 24.08. RS < FS holds; RS leaves the 1–10 Hz
E4 order once the run is steady (bg05 already 11.37 Hz).

Estimand errors (bands: E2 e ≤ 0.05, E3 abs ≤ 0.01): bg05 e_cv 0.133,
abs_cc 0.0425; bg08 e_cv 0.265, abs_cc 0.0129; bg12 e_cv 0.288, abs_cc
0.0105; bg20 e_cv 0.284, abs_cc 0.0049. Only bg20 closes E3, none closes
E2, none closes E2+E3 jointly with rate in-band. L = e_cv² +
(abs_cc/0.01)² falls 18.1 → 1.7 → 1.2 → 0.3 across 5 → 8 → 12 →
20 Hz, minimal at bg20 with E2 still off.

GPU wall this step: bg08 `58.0 s`, bg12 `68.4 s` (run output); prior
bg01–bg05 + bg20 per manifest `wall_s`.

## Step 4: absorption interim + step-6 status (OPEN; unscored pilot, ruling 2026-10-18)

D with ranges and reasons: kick rate 200–400 Hz (paper range, in D);
kick fraction 0.02–1.0 (F5 shape; 0.02–0.10 of RS ≈ paper 2–10% of the
net, larger fractions over-kick controls); delay no range (kernel
minimum); ge/gi paper-states-exactly (6/67 nS, deviation needs a
reason — not moved here); b not in D (Fig. 6A vs 6B control contrast).
`agsdr` interface read (`jaxfne/optim/core.py:252`) but no fit launched;
manual a scan used so far.

Smallest |a|: a = 2 Hz is the first technically alive point (37 spikes
in 10 s, rate 0.00185 Hz); a = 4 Hz is the first intermittent Up/Down
run; a = 5 Hz the first bursty-alive run across seconds; a = 20 Hz the
first steady-alive run with CC in-band. No single a closes E2+E3 with
E4 in-band, so a ↛ 0 through allowed D (kick-D dead, E1 FAIL) and a ↛
closure through X_bg alone. ge/gi absorption would move paper-stated
parameters and is not attempted in this lane step.

Held-out seeds 4–8 unused and reserved. No θ*; the claim stays
interim: X_bg sustains where θ₀ dies, with |a| as above, but the
single-gain X_bg does not reproduce E2+E3. Concentrated-vs-diffuse
background and a genuine D-absorption fit (with λ ramp on a) remain
the two open moves before a final needs-X verdict.

## Step 5: authors'-code fidelity audit + code-value kick + N-trend (2026-10-14; unscored pilot, ruling 2026-10-18)

Code source (read as data, never run): ModelDB accession 126466
("Asynchronous irregular and up/down states in excitatory and inhibitory
NNs (Destexhe 2009)"), simulation environment PyNN, GitHub mirror
`github.com/ModelDBRepository/126466`. Vendored read-only under
`scratch/destexhe2009/code126466/` (README sha256 `1dc7f2fa…0780`,
`demo_cx05_N500b_LTS.py` `82ad0448…41be`, `demo_cx_Up-Down.py`
`b109cfc2…41d6`; sizes match the API listing). The entry holds no Fig. 6
demo; the audit reads `demo_cx05_N500b_LTS.py` (Fig. 7B net: the same
cell/synapse/connection/kick code path as the Fig. 6 cortical net).
pyNN/NEURON are absent from `~/venv_jaxfne` (checked), so nothing was
executed. Fig-6-specific readings are labelled inferred. Resolves G8.

Fidelity table (line refs `code126466/demo_cx05_N500b_LTS.py`, port refs
live code):

| # | Item | Authors' code | Port | Match? |
|---|---|---|---|---|
| 1 | C, gL, EL, VT, Δ, τw | C = 200 pF (`cm` 148–155), τm = 20 ms, EL = −60, VT = −50 (86), Δ = 2.5 (82), τw = 600 (81) | identical (DESTEXHE_CELL_DEFAULTS) | yes |
| 2 | a / b RSweak | a = 0.001 µS → 1 nS (73, 153), b = 0.005 nA = 5 pA (75, final assignment) | 1 nS / 5 pA | yes |
| 3 | a / b FS | a = 0.001 µS → 1 nS (78, 161), b = 0 (79) | 1 nS / 0 | yes — G1 resolved, port assumption correct |
| 4 | Vreset / refractory | VBOT = −60 (88), REFRACTORY = 2.5 ms (89; PyNN gets +DT → 2.6, 152) | −60, 25 steps = 2.5 ms | yes (≈; 0.1 ms negligible) |
| 5 | Spike detection | `v_thresh` = VTR = −50, `v_spike` = VTR+1e-6 (152–153): detection ≈ VT | `VT + 5Δ` = −37.5 (`jaxfne/emitters_adex.py:127,142`) | NO (F6, real; core-owned, no in-lane fix) |
| 6 | dt / method | DT = 0.1 (58), fixed-step (`use_cvode=False`, setup 272) | dt = 0.1 explicit Euler | yes (≈; NEURON backward-Euler detail unstated, no fix) |
| 7 | ge / gi / τE / τI / reversals | 0.006/0.067 µS = 6/67 nS (99–100), 5/10 ms (95–96), 0/−80 mV (97–98) | identical | yes |
| 8 | Connection rule | capped Bernoulli: per cell up to C_E = int(N_E·p) exc + C_I = int(N_I·p) inh inputs, ordered scan, `i != j` (194–224); p inverse-N rescaled (108) | pure Bernoulli p, autapses excluded | PARTIAL (F7; cap trims ≈7% exc / ≈14% inh at N = 2000, normal-approx, inferred; core-owned, no in-lane fix) |
| 9 | N / fractions | 80/20 E/I (104–106; file N = 500, Fig. 6 N = 2000 per paper) | 1600 + 400 | yes |
| 10 | Delay | `delay=DT` edges + kick (209, 224, 264) | 1 step = 0.1 ms | yes (F1 stands) |
| 11 | Kick | N_STIM = N/5 = 20% net, first cells, all excitatory (118, 258); 20 synapses/cell (120) × exponential ISI 70 ms (121, 242–264) ≡ Poisson ≈ 285.7 Hz aggregate; weight AMPA (264); 50 ms (119) | code-value probe kf 0.25 of RS @286 Hz = same distribution (`_construct_adex.py:512`: λ = n·rate·dt/1000) | code value reproduced; paper TEXT says 2–10% — text-vs-code discrepancy documented, both probed |
| 12 | Initial conditions | `initialize(v=-60, w=0)` uniform (174) | V0 = EL, w = 0 | yes |
| 13 | Reset order (V reset, w += b, ref) | PyNN EIF semantics at detection; no finer order readable | `adex_step` resets V, `w += b`, ref set | no finding |

F6 mechanism note: with detection at VT + 5Δ the port's cells spend
milliseconds in the exponential upstroke, where arriving inhibition
(driving force E_I − V large) can still veto the spike; the authors'
cells reset at VT with no veto window. Direction: the port
under-spikes relative to the authors' implementation at the same
conductance drive — the wrong direction is not guaranteed, but F6 is
the only structural mismatch found, and it is the prime suspect for
the death. Fixing it means a detection-threshold change in the core
kernel (`adex_step`) — out of this lane (no new kernel code) —
proposed as core follow-up, not executed.

In-lane change (plumbing, not a model fix): `run_continuous` forwards
`probability` (Fig. 9 needs inverse-N rescaled p), recorded in the
manifest with `n_edges`; CLI `--probability`. Test
`tests/test_destexhe2009_port.py::test_run_continuous_forwards_probability`
fails on the unmodified signature
(`TypeError: run_continuous() got an unexpected keyword argument
'probability'`) and passes with it. No model parameter moved.

S2 at the code-value kick (kf 0.25 @286 Hz, N = 2000, p = 0.02,
continuous 10 s free; rasters `scratch/destexhe2009/s2code/`; report
`scratch/destexhe2009/report_s2code.json`):

| run | s0 | s1 | s2–s9 |
|---|---|---|---|
| seed 1 | 44801 | 9516 | 0 |
| seed 2 | 47232 | 0 | 0 |
| seed 3 | 47296 | 6456 | 0 |

Estimands (same bands; E3 absolute scale 0.01):

| Id | y (paper) | z seed 1 / 2 / 3 | e | band |
|---|---|---|---|---|
| E1 alive at 10 s | self-sustained | false / false / false | — | FAIL |
| E2 CV_ISI = 2.47 | 2.47 | 1.635 / 1.544 / 1.647 | 0.338 / 0.375 / 0.333 | off |
| E3 CC = 0.005 | 0.005 | 0.0875 / 0.0932 / 0.1015 | abs 0.082 / 0.088 / 0.096 | off |
| E4 rates | RS < FS, RS 1–10 Hz | 2.72 / 2.36 / 2.69 overall | — | reported, not gated |

N-trend (Fig. 9 direction, 0% LTS, p rescaled inverse-N, code kick,
seed 1; rasters `scratch/destexhe2009/s2ntrend/`):

| N / p | bins s0–s9 | CV | CC | alive |
|---|---|---|---|---|
| 2000 / 0.02 (s2code seed 1) | 44801/9516/0/… | 1.635 | 0.0875 | false |
| 1000 / 0.04 | 29014/1462/0/… | 1.681 | 0.1192 | false |
| 500 / 0.08 | 1916/0/0/… | 0.699 | 0.1372 | false |

Smaller N dies faster — the paper's Fig. 9 direction — but the port
dies even at N = 2000 where the paper sustains (minimal N 1800 at 0%
LTS). The port is not near the edge; the miss is structural (F6
suspect), not a size effect.

Held-out seeds 4–8: not run. Nothing sustained on seeds 1–3, so there
is no candidate to score; 4–8 stay reserved per the residual protocol
(step 5 scores a fit, and there is none).

GPU wall this step: smoke `3.2 s` + `3.4 s`; batches `41.6 / 40.3 /
40.2 / 49.4 / 38.6 s`; total ≈ 217 s. JAX `0.10.1`, cuda.

Verdict (unscored pilot): still OFF at θ₀ and at the authors'-code kick value. Step-4
interim stands (|a| trajectory for X_bg unchanged); new finding: the
port's spike-detection threshold (F6) is the prime structural
suspect, with a proposed core experiment (detection at VT). No
paper-stated parameter moved; no θ*.

## Step 6: core migration and fidelity F6/F7 (2026-10-14; unscored pilot, ruling 2026-10-18)

What changed (port-side only, no core edit, no paper value moved):

1. Fork deleted: `jaxfne/adex_destexhe.py::simulate_adex_gated_drive`
   is gone; `destexhe_api.run_continuous` issues one `simulate_adex`
   call with the core `ext_t_start_steps`/`ext_t_stop_steps` drive
   window (`t_start_ms=0`/`t_stop_ms=50` on the run's `Simulation`).
   Fork-vs-window receipt
   (`scratch/destexhe2009/fork_window_equality.json`):
   `{"n_steps": 60, "kick_steps": 20, "fork_spikes": 2,
   "core_spikes": 2, "bit_exact": true}` — the window reproduces the
   fork bit-exactly on the receipt world. Guard test
   `tests/test_adex_destexhe.py::test_gated_fork_removed_core_window_is_the_drive`
   plus four core-window tests (full-open/prefix/pre-window/refusal).
2. F6 plumbed: `DESTEXHE_CELL_DEFAULTS` carries `Vpeak_mV = -50.0`
   (authors' detection, `demo_cx05_N500b_LTS.py:86,152-153`) onto every
   mapped cell via `destexhe_overrides`; unmapped carriers keep NaN
   (core default). Hamm ruling 2026-10-14: the authors' Vpeak is
   primary, the core default `VT + 5Δ` stays beside it as a declared
   sensitivity. Consequence pinned in
   `tests/test_destexhe2009_port.py::test_s2_quantum_spikes_resting_post`:
   one 6 nS S2 quantum spikes a resting RSweak cell under the authors'
   detection (1 post spike) and stays subthreshold under the old
   default (0 post spikes); the EPSP-scaling probe moved to 2/4 nS
   (ratio 1.94).
3. F7 applied port-side: `build_spec` uses the core-owned
   `sampling="capped_bernoulli"` (the authors' per-cell cap).
   Residual: ordered-scan vs independent-cap draw order unproven.
4. `run_continuous` forwards `probability` (manifest records it with
   `n_edges`); `single_cell`/`epsp_peak` carry `Vpeak_mV`.

S2 rerun at the code-value kick (0.25 of RS @286 Hz, p = 0.02,
continuous 10 s free; rasters `scratch/destexhe2009/s2core/`;
report `scratch/destexhe2009/report_s2core.json`; scorer
`run_destexhe score` bands unchanged: E2 e ≤ 0.05, E3 abs ≤ 0.01):

| seed | bins s0–s9 | alive | CV (e) | CC (abserr) | rate RS/FS Hz |
|---|---|---|---|---|---|
| 1 | 96469/93256/72283/0… | false | 2.426 (0.044) | 0.1109 (0.1059) | 12.85/14.12 |
| 2 | ~88–107k all s | true | 2.613 (0.143) | 0.0051 (0.0001) | 49.18/49.40 |
| 3 | ~92–101k all s | true | 2.664 (0.194) | 0.0038 (0.0012) | 47.31/50.19 |
| 4 | ~85–94k s0–s7, 0 s8–s9 | false | 2.356 (0.114) | 0.0379 (0.0329) | 34.86/39.49 |
| 5 | 87328/77073/41119/0… | false | 2.402 (0.068) | 0.1051 (0.1001) | 9.86/11.93 |
| 6 | ~91–100k all s | true | 2.551 (0.081) | 0.0105 (0.0055) | 46.59/50.48 |
| 7 | ~86–101k s0–s8, 10926 s9 | true | 2.567 (0.097) | 0.0159 (0.0109) | 40.87/44.31 |
| 8 | ~73–93k s0–s6, 0 s7–s9 | false | 2.449 (0.021) | 0.0518 (0.0468) | 28.56/34.26 |

RS < FS holds on all 8 seeds. E2 closes only on dead seeds (1, 8);
on alive seeds e = 0.081–0.194 (off). E3 closes on alive seeds 2, 3,
6; seed 7 misses by 0.0009. Mean rates on sustained seeds are
40–50 Hz — far above the E4 1–10 Hz order: a hyperactive attractor,
not the paper's AI state.

Sensitivity at the core default (same protocol, `Vpeak_mV = NaN` on
every cell via post-construction `_replace`; rasters
`scratch/destexhe2009/s2sens/`, report
`scratch/destexhe2009/report_s2sens.json`):

| seed | bins s0–s9 | alive | CV (e) | CC (abserr) | rate Hz |
|---|---|---|---|---|---|
| 1 | 35946/0… | false | 1.466 (1.004) | 0.0951 (0.0901) | 1.80 |
| 2 | 46672/32292/0… | false | 1.670 (0.800) | 0.0790 (0.0740) | 3.95 |
| 3 | 42929/9561/0… | false | 1.607 (0.863) | 0.0961 (0.0911) | 2.62 |

The default reproduces the fork-era death (steps 1–5: CV ~1.5–1.7,
CC ~0.08–0.10, dead ≤ 2 s). F6 is confirmed as the structural
switch: the authors' detection sustains (hyperactive), the core
default dies. Capped sampling trims edges (n_edges ≈ 73.3–73.5k vs
≈ 80k fork-era Bernoulli) — the wrong direction to explain sustain
by over-drive, and the default-Vpeak run shares the same wiring.

Provenance: code HEAD `6f85f5c3` with uncommitted lane tree at run
time (migration + Vpeak table + capped sampling; `s2core/manifest.json`
pins the HEAD SHA — rasters are post-baseline tree). JAX `0.10.1`,
cuda. GPU wall s2core: seeds 1–8 `81.0 / 62.0 / 84.5 / 81.3 / 75.2 /
49.5 / 52.7 / 81.3 s`; s2sens seeds 1–3 `70.0 / 77.4 / 54.7 s`.
Held-out note: seeds 4–8 are scored here at the declared non-fit
baseline per dispatcher order (no fit exists); any future fit
re-scores them per PREREG_P3.md.

Verdict (unscored pilot): supersedes steps 1–5 OFF. E1 PASSES on 4/8 seeds under the
authors' Vpeak (2, 3, 6 sustained; 7 alive but collapsing in s9);
E3 closes on 2, 3, 6; E2 stays off on every alive seed; E4 rate
order badly off (hyperactive ~5x). No θ* — no seed closes
E1+E2+E3 jointly with E4 in-band. The miss moved: from death to a
hyperactive attractor. Open moves: rate-order reduction without
losing sustain (ge/gi absorption still untouched, needs a reason),
and the X_bg trajectory of step 4 is unchanged.
