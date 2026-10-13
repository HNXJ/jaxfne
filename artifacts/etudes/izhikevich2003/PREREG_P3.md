# Izhikevich 2003 P3 preregistration

Status: draft v7 for Hamm's batch sign-off. The v6 estimator failed its calibration gate
(2026-10-13, receipt `calibration/calibration_v6_estimator_7ba10446.json`): false positives ≤ 5 %
from θ = 4 and every 40 Hz row ≥ 95 %, but the 10 Hz AR(2) rows detected 0.858–0.944 at every θ
in 8–13 Hz, and 0.864–0.938 on the six AR(1) floors after widening to 7–14 Hz (white 0.966–0.976). The misses were the local-maximum rule: f_pk, the
largest residual, fell on the bump's flank, where the PSD is not a strict local maximum (a
diagnostic outside the receipt, draws $j = 0$–299 of rows 16 and 32 at θ = 4 in 7–14 Hz: 23 and
39 local-maximum failures, against 2 each with f_pk off target). v7 revises step 3 as the gate requires;
the alpha band returns to 8–13 Hz and the gate reruns from the start. v5 review (2026-10-13): REVISE on three minor
points (float time comparisons, calibration seeds and AR(2) form, the widening trigger), fixed
here; `PORT_SPEC.md` line 12 was changed after its PASS to point at the fixed-point start below.
v4 failed review (2026-10-13):
B7 covered the spectral estimator only, θ was chosen on false positives alone while detection at
2θ sits near 95 % by construction (9 Welch windows), the stationary positives missed the paper's
"occasional" episodes, the TC step of −10 holds v near −76 mV (−90 needs −36.5), the −65/−70
starts are off rest, and the C8 prominence side, sample times, Welch settings and key split were
unstated. v1 failed review (2026-10-12):
parameters not inlined, C2/C8 detectors undefined, C6 had no rule for an empty search, the
reused estimator's settings differ from CAMINOS and were not pinned, and "true peak excess" had
no computation. v2 failed on seven smaller ambiguities (C6 search ignored the no-burst rule; C2
tail, N2 formula and cell count, C8 edges, S3 baseline, one SHA for all scored runs). v3 failed on
seven more (C5 spike minimum, C2 bridging ISI, C6 run count, C8 first sample, the estimator
function, the true-peak-excess computation, the alpha band check). No scored run before the signature. Smoke runs and timing pilots are not scored.

Spec: `PORT_SPEC.md` (second review PASS 2026-10-12). Source: `SOURCE_P1.md`. The paper's claims
are qualitative except "around 8 Hz" and "10 and 40 Hz"; every band below is ours.

## Prior exposure

No run of this port exists. The classic outputs of `net.m` (a rhythmic raster with population
bursts) are widely reproduced and known to us in general terms; no number from them sets a band.

## Fig. 2: single cells

Runs: kernel `simulate_eig_izhikevich`, one cell per run, `dt_ms = 0.1`, float32,
`noise_scale = 0`, `W = 0`, `drive = 0`, key `jax.random.PRNGKey(0)` (noise is off). Each cell
starts at its resting fixed point at $I = 0$: $v^*$ the lower root of $0.04v^2 + (5 - b)v + 140 = 0$,
$u^* = bv^*$; $v^* = -70.0$ for $b = 0.2$ (RS, IB, CH, FS), −64.4 for $b = 0.25$ (LTS, TC), −62.5
for $b = 0.26$ (RZ), computed in float64 and cast. All Fig. 2 runs use one code SHA.

Time: kernel output row $k$ is the state after step $k$ and is stamped $t = (k + 1)\,dt$; the
input of row $k$ of `drive_schedule` acts during step $k$. A spike at row $k$ is at $(k + 1)\,dt$.
Windows, ISIs and every time comparison are computed in integer rows (10 ms = 100 rows); times
are converted to ms only for reporting.

| Type | a | b | c | d |
|---|---|---|---|---|
| RS | 0.02 | 0.2 | −65 | 8 |
| IB | 0.02 | 0.2 | −55 | 4 |
| CH | 0.02 | 0.2 | −50 | 2 |
| FS | 0.1 | 0.2 | −65 | 2 |
| LTS | 0.02 | 0.25 | −65 | 2 |
| TC | 0.02 | 0.25 | −65 | 0.05 |
| RZ | 0.1 | 0.26 | −65 | 2 |

Definitions. Spikes are the kernel's threshold crossings. The analysis window is the 300 ms after
the input event (step onset, release, or pulse end). A burst is a maximal run of ≥ 2 consecutive
spikes whose inner ISIs are all < 10 ms. Rate = spike count in the window / 0.3 s. ISI ratio = last
ISI / first ISI in the window.

| Id | Type | Input (run length) | Pass |
|---|---|---|---|
| C1 | RS | $I = 0$ for 10 ms, then 10 (310 ms) | ≥ 4 spikes; ISI ratio ≥ 1.5 |
| C2 | IB | as C1 | an initial burst, then a tail of ≥ 3 spikes with every ISI ≥ 10 ms (definition below) |
| C3 | CH | as C1 | ≥ 3 bursts; inter-burst frequency (1 / mean interval between burst onsets) in 15–50 Hz |
| C4 | FS | as C1 | rate ≥ 50 Hz; ISI ratio ≤ 1.2 |
| C5 | LTS | as C1 | ≥ 4 spikes; rate above the RS rate of C1; ISI ratio ≥ 1.2 |
| C6 | TC tonic | $I = 0$ for 10 ms, then $I_+$ (310 ms) | ≥ 3 spikes, no burst, ISI CV (N2's formula) ≤ 0.3 |
| C7 | TC rebound | $I = 0$ for 10 ms, $-36.5$ for 100 ms, then 0 (410 ms) | a burst whose first spike falls within 50 ms of release |
| C8 | RZ | $I = 0$ for 10 ms, 5 for 1 ms (assumed), then 0 (311 ms) | no spike; ≥ 2 local maxima of $v$ after the pulse, the second lower than the first; the frequency 1 / (time between the first two maxima) is reported |

C7's step holds TC at its fixed point −90.0 mV ($I = -36.5$ solves $0.04v^2 + 4.75v + 140 + I = 0$
at $v = -90$), the hyperpolarized state `PORT_SPEC.md` names; S4 reports $I = -10$ (fixed point
−76.0 mV).

C5 is relational because the paper's claim is: LTS fires faster than RS and adapts. Both rates
come from the scored runs C1 and C5.

C6 runs up to 10 times, at $I_+ = 1, 2, …, 10$ in ascending order, and stops at the first run with
≥ 3 spikes, no burst and ISI CV ≤ 0.3 in the window; every run's outcome is reported. C6 passes
if such a run exists and fails otherwise. C6 therefore tests that TC fires regularly and tonically
from rest at some step amplitude up to 10; the search over amplitudes is intended.

A ratio, rate or ISI that is undefined because a run has too few spikes fails its criterion.

C2: spikes in the window are partitioned into bursts and single spikes first. The initial burst
is the burst that contains the first spike; it must exist. The tail is every spike after it.
Tail ISIs are the intervals with both spikes in the tail; fewer than 3 tail spikes, or a tail ISI
< 10 ms, fails C2.

C8 local maxima: the window is the samples with $t$ > pulse end through the end of the run;
maxima are the interior samples $v_i$ of the window (its first and last sample excluded) with
$v_i > v_{i-1}$, $v_i \ge v_{i+1}$ and prominence ≥ 0.05 mV. Prominence is $v_i$ minus the larger
of the two minima of $v$ taken between $v_i$ and the nearest higher-or-equal sample on each side;
on the left with no such sample the minimum runs to the window start, on the right to the window
end.

## Fig. 3: network

Runs: per `PORT_SPEC.md` (W = 2S and noise × √2 at `dt_ms = 0.5`), float32, seeds 1–5, one code
SHA. Keys: `k_r, k_w, k_noise = jax.random.split(jax.random.PRNGKey(seed), 3)`, in that order, for
the parameter draw $r$, the weights and the kernel's noise key. Duration 10.2 s; the first 200 ms are discarded. Signal: the population rate of all 1000 cells
in 1 ms bins (the paper's step), minus its mean.

| Id | Estimand | Pass |
|---|---|---|
| N1 | mean rate over all cells, per seed | 4–16 Hz in every seed (a factor-2 band around "around 8 Hz") |
| N2 | median over cells with ≥ 10 spikes of the ISI coefficient of variation, std(ISI, ddof = 0) / mean(ISI), per seed | 0.6–1.5 in every seed ("Poisson-like"), and ≥ 500 cells qualify; otherwise N2 fails |
| N3 | alpha: peak excess PE in 8–13 Hz (7–14 Hz if the calibration widened it) | PE ≥ θ in every seed |
| N4 | gamma: peak excess PE in 30–50 Hz | PE ≥ θ in every seed |

Detection is calibrated at a true peak excess of 3θ. Between θ and 3θ it is not guaranteed, so a
true peak there can fail "every seed"; the 1.5θ and 2θ calibration rows give that power.

Peak excess. The model form and the fitting routine are those of
`artifacts/etudes/caminos/caminos_analysis.py` (`aperiodic_log10`, `fit_aperiodic`); the band,
fit range, exclusion and edge rule are this study's, set here and passed as parameters to a new
function `peak_excess(signal, fs, band, fit_hz=(2, 200), excl_hz=2, edge_hz=1)` in
`artifacts/etudes/izhikevich2003/izh_analysis.py`. CAMINOS's `spectral_peak`, whose constants are
fixed to CAMINOS's bands, is not used. Fit bounds and starts are `fit_aperiodic`'s (k ≥ 0,
χ ∈ [0, 5], start k = 1, χ = 1); a fit that does not converge, a PSD with a zero bin in the
fit range (no logarithm), or a band with no strict PSD local maximum (step 3) counts as no peak.

1. Welch PSD: `scipy.signal.welch(x, fs=1000, window="hann", nperseg=2000, noverlap=1000,
   detrend=False, scaling="density")` (0.5 Hz bins, 9 windows over the kept 10 s).
2. Fit log10 P = a − log10(k + f^χ) by bounded least squares on 2–200 Hz.
3. f_pk = the frequency of the largest residual among the in-band bins that are strict local
   maxima of the PSD; a band with no such bin has no peak.
4. Refit on 2–200 Hz with f_pk ± 2 Hz excluded.
5. PE = P(f_pk) / background(f_pk).
6. The band call holds when PE ≥ θ, f_pk is a local maximum of the PSD (true by step 3), and
   f_pk lies ≥ 1 Hz from both band edges.

**Calibration gate (before signature, synthetic signals only).** A 10 s population count at
N = 1000 and 8 Hz mean rate, 1 ms bins, through steps 1–6; 1000 draws per null row, 500 per
detection row. Rows are numbered $i = 0, 1, …$ in this order: the 7 null floors as listed, then
the 28 detection rows (floor-major, then 10 before 40 Hz, then sinusoid before AR(2)), then the
reported-only rows in the order listed. Draw $j$ of row $i$ uses one generator
`numpy.random.default_rng([20261013, i, j])`, which draws the floor first, then the component (its
phase, AR(2) innovations or episode placement).

- Null floors (7): white Poisson; Poisson on a lognormal AR(1) rate with τ = 8 and 30 ms at
  variance 0.5×, 2× and 8× the Poisson count variance, listed τ-major (τ = 8: 0.5×, 2×, 8×; then
  τ = 30). The count in bin $t$ is Poisson($\lambda_t$), $\lambda_t = 8\,e^{\sigma z_t - \sigma^2/2}$,
  $z_t = \phi z_{t-1} + \sqrt{1-\phi^2}\,\varepsilon_t$, $\phi = e^{-1/\tau}$ ($\tau$ in bins),
  $z_0, \varepsilon_t \sim N(0, 1)$; $\operatorname{Var}\lambda_t = 64(e^{\sigma^2} - 1) = m \cdot 8$ for
  multiple $m$, so $\sigma^2 = \ln(1 + m/8)$. The generator draws $z$ first, then the counts.
- θ: the smallest of 2, 3, 4, 6 for which (i) every null floor has a false-positive rate ≤ 5 % in
  each band and (ii) every detection row at true peak excess 3θ detects ≥ 95 %. If some θ meets
  (i) and every 40 Hz row at it, but the 10 Hz rows fail (ii) at every such θ, the alpha band widens
  once to 7–14 Hz and the search repeats. Any other failure, or no θ after widening, revises the
  estimator here before signature.
- Detection rows (28): each floor plus an additive component at 10 Hz and, separately, at 40 Hz,
  in two forms: a random-phase sinusoid and a narrowband AR(2) bump of 3 Hz FWHM
  ($x_t = 2r\cos\omega\,x_{t-1} - r^2 x_{t-2} + e_t$, $\omega = 2\pi f_0/1000$,
  $r = e^{-\pi \cdot 3/1000}$, $x_0 = x_1 = 0$, 11 000 innovations drawn in one call, the first
  1000 samples discarded). The component is scaled per floor
  so that the true peak excess at the target is 3θ. True peak excess is (C + F) / F at the target
  bin, where F is the floor's mean Welch PSD over its null draws $j = 0$–199 and C is the
  component's mean Welch PSD over 200 draws of the component alone, from
  `default_rng([20261013, 1000 + i, j])`, $j = 0$–199; the component adds to the floor, so C + F is
  the expected PSD of a detection draw. C is computed at unit amplitude (sinusoid amplitude 1,
  AR(2) innovations N(0, 1)) and scales with amplitude squared, so the amplitude is
  $\sqrt{(3\theta - 1)F/C}$ exactly (no bisection is needed); F and C are fixed estimates and their
  sampling error is reported. The same draws $j$ serve every θ; only the amplitude changes. A
  detection counts when the band call holds with f_pk within 1.5 Hz of the target.
- Reported only: detection at true peak excess 1.5θ and 2θ on every floor, both forms; and
  intermittent rows, the 3θ component (the continuous row's amplitude) present only in 4 of the
  20 aligned 500 ms slots (20 % of the 10 s; slots from `rng.choice(20, 4, replace=False)` after
  the component), the paper's "occasional" synchronized episodes, on the white and the τ = 30 ms,
  8× floors. Reported-only rows are ordered 1.5θ rows, 2θ rows (each as the detection rows), then
  intermittent rows (white before τ = 30 ms 8×, then 10 before 40 Hz, then sinusoid before AR(2)).

**Detector calibration (before signature, known answers).** Unit tests of `izh_analysis.py` on
constructed inputs; each must hold exactly.

- Burst detector: a doublet train (inner ISI 5 ms, 100 ms apart) gives one burst per doublet; a
  regular 50 Hz train (ISI 20 ms) gives none; ISIs of exactly 10.0 ms give none (the rule is
  < 10 ms); a 3-spike run at 4 ms ISI is one burst.
- Rate and ISI ratio: a train with ISIs 10, 20, 30 ms starting 5 ms into the window gives rate
  4 / 0.3 s and ISI ratio 3.
- C8 maxima: $v(t) = -62.5 + 2e^{-t/0.05}\sin(2\pi \cdot 20\,t)$ (t in s, 0.1 ms samples) gives ≥ 2
  maxima, decreasing, with frequency 20 Hz within 0.1 Hz; a monotone relaxation
  $-62.5 + 2e^{-t/0.05}$ plus a ±0.01 mV alternating ripple gives no maximum.
- N2: 1000 Poisson trains at 8 Hz over 10 s give a median CV in 0.6–1.5; 1000 regular 8 Hz trains
  give a median CV below 0.6.

The calibration table is written into this file before signature.

**Completeness.** Every scored run (C1–C8 and N1–N4) uses one code SHA. C1–C5, C7 and C8: one run
each at the settings above; C6: the ascending search above. N1–N4: exactly seeds 1–5 at
T = 10.2 s. A missing or differently configured scored run scores as incomplete, which counts as a
fail. Sensitivity runs (S1–S3) are labelled as such and do not enter scoring. A seed with no spikes
fails N1–N4.

## Sensitivity (reported only)

- S1: the reference arm `net_reference.py` (float64 NumPy transcription of `net.m`), seeds 1–5:
  N1–N4 on both arms, their differences, and the mean extra $u$ increment per spike from the
  first-half-step overshoot (`PORT_SPEC.md`). If S1 moves N1–N4 across a band edge, a kernel
  variant arm that reproduces the overshoot is preregistered separately.
- S2: float64 with `jax_enable_x64` on, seed 1; the run asserts the output dtype.
- S3: `dt_ms = 0.25` with W = 4S (the paper's S) and `noise_scale` = 2 × the paper's σ (10 for E,
  4 for I), seed 1.
- S4: C7 with $I = -10$ during the step (fixed point −76.0 mV).

## Outcome language

- C1–C8 pass: "Izhikevich 2003 Fig. 2 firing patterns reproduce on the jaxfne Izhikevich kernel."
- N1–N4 pass: "The Izhikevich 2003 1000-neuron network reproduces in jaxfne: mean rate, irregular
  firing and alpha and gamma peaks."
- A failure is reported with its numbers. Parameters stay as declared; a change is a new,
  separately preregistered run.
