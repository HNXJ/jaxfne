# Destexhe 2009 preregistration P3 (draft, UNSIGNED — no scored run may use it)

Lane 5, cortical self-sustained AI first (PORT_SPEC.md S2 = paper Fig. 6B).
Estimands are the paper's values (y) from SOURCE_P1.md; z is the jaxfne AdEx
route at the paper's parameters θ₀ (no fit). Bands: RESIDUAL_PROTOCOL.md
(close: every e ≤ 0.05; off: any e > 0.05 → calibration steps).

## Fixed port settings (θ₀)

N = 2000 (1600 RSweak + 400 FS), p = 2% all four blocks, ge = 6 nS,
gi = 67 nS, τE/τI = 5/10 ms, reversals 0/−80 mV, dt = 0.1 ms, decay exp,
threshold VT + 5Δ, 0.1 ms uniform delay (F1), 50 ms kick (300 Hz private
sources, 6 nS AMPA kicks to RS only; F5/G5), 10 s free run (V-carried restart;
F2). km RNG: `connectivity(edge_seed=seed)`, drive `seed=seed`.

## Estimands

| Id | y (paper Fig. 6B) | z | e = \|z−y\|/\|y\| | criterion |
|---|---|---|---|---|
| E1 alive | self-sustained (no death time given; control 6A dies "within seconds") | spikes in last 1 s of 10 s run | — | z alive AND 6A control dead by 10 s |
| E2 CV_ISI | 2.47 | `cv_isi` over full free run | ≤ 0.05 | close band |
| E3 CC | 0.005 | `pairwise_cc`, 5 ms bins, 500 disjoint pairs | absolute error ≤ 0.01 (scale named: y near 0) | close band |
| E4 rates | PY 1–3 Hz order (CAMINOS R3 analogue; paper inset only) | `mean_rate_hz` per carrier | — | RS < FS, RS within 1–10 Hz (small-network text notes ~40 Hz; reported, not gated) |

E3 uses absolute error: e_rel = |z − 0.005| / 0.005 is unstable at CC ≈ 0;
the named scale is 0.01 (twice the asynchronous threshold 0.1/10 … precisely:
asynchronous iff CC < 0.1, so |err| ≤ 0.01 keeps the verdict).

## Seeds

Fit (if needed): seeds 1–3. Score: held-out seeds 4–8. This draft declares no
fit; the baseline z is reported at seeds 1–3 in CALIBRATION.md.

## Controls (same θ₀, predicted direction)

- C1 Fig. 6A (`strong_adaptation=True`): transient, dead within seconds.
- C2 single cells (Fig. 1): RS adapts, FS regular, LTS rebounds
  (tests/test_destexhe2009_port.py).

## Degrees of freedom D (declared before any fit; unused at baseline)

Empty at baseline. If E2/E3 score off, candidates in order: kick rate
(200–400 Hz, paper range), kick fraction shape (F5), delay (0.1 ms fixed by
kernel; no range), ge/gi (paper states exactly; deviation needs a reason).
Adaptation b is NOT in D (the paper states it; Fig. 6A vs 6B is the control).
