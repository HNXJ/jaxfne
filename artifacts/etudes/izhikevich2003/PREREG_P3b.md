# Izhikevich 2003 P3b preregistration: net.m update-order arm

Status: **draft v3**, unsigned (Hamm asked for the draft 2026-10-13). Nothing scored runs before
signature. P3 (signed v7) stays FAIL whatever P3b finds. P3b asks a single question: do the paper's
Fig. 3 claims hold when the kernel uses `net.m`'s own update order?

v1 review (Muse, 2026-10-13): REVISE. Fixed here:
- the shared-draw change is now named;
- the gate is injectable and has a fallback bar;
- every implementer choice is pinned.

v2 review (Muse, 2026-10-13): REVISE. Fixed here: the network constructor is named, the gate's
generators and injection are pinned, gate step 1 has a rounding-flip bar, the remaining
implementer choices are pinned, and the kernel arm's S1 numbers are listed.

Trigger: `PREREG_P3.md` §Sensitivity, S1. The reference arm fails N3 in every seed while the
kernel passes it (`RESULT_P3.md`, corrected S1 table).

## Prior exposure

Known before this draft, on the S1 reference arm, seeds 1–5, corrected scoring:

| Statistic | Reference arm |
|---|---|
| N1 | 7.05–7.23 Hz |
| N2 | 0.387–0.399 |
| N3 | seed 1: f_pk 11.0 Hz, PE 0.96. Seeds 2–5: f_pk 8.0–8.5 Hz, PE 6.65–11.34, inside the 1 Hz edge rule |
| N4 | passes every seed |
| Extra u per spike | 3.04–3.09 |

The S1 kernel arm (the scored P3 kernel) on the same seeds: N1 8.48–8.83 Hz, N2 0.400–0.410, N3
and N4 pass every seed (`RESULT_P3.md`, corrected S1 table).

Also known: the CV, CV2 and Fano numbers in `DIAGNOSTIC_P3.md`. Not known: any output of the P3b
arm; `kernel_net_m.py` does not exist at this draft.

The variant reproduces the reference's update order, so the expected outcome is FAIL on N2 and N3.
That expectation is a prediction; it sets no criterion. Every band, θ and setting below is P3 v7's,
unchanged, except those this section names.

## Arm

An étude-local JAX function, `kernel_net_m.simulate_net_m(params, S, thal, v0, u0)` in
`artifacts/etudes/izhikevich2003/kernel_net_m.py`. It changes no library API.
- Inputs: `thal` has shape (n_steps, 1000).
- Outputs: spike rows (n_steps, 1000) bool and the extra u per spike. Extra u is defined exactly as
  `net_reference.simulate` defines it.

One `jax.lax.scan` step is 1 ms, in `net.m`'s order:
1. `fired = v >= 30` on the v left by the previous step.
2. Reset: v ← c, u ← u + d for the fired cells.
3. I = thal[t] + `S @ fired`, with no other drive (drive = 0).
4. Two v half-steps of 0.5 ms with the same I:
   v ← v + 0.5 (0.04 v² + 5 v + 140 − u + I).
   There is no clamp, so a cell that crosses 30 in the first half-step overshoots in the second.
5. u ← u + a (b v − u) on the v after both half-steps.

Row t of the output is `fired` from step 1 of step t (time t ms).

Changes from the scored P3 arm, all of them `net.m`'s:

| | Scored P3 arm | P3b arm |
|---|---|---|
| Step | 0.5 ms | 1 ms, two v half-steps and one u step |
| Weights | W = 2S | W = S |
| Thalamic draws | independent per 0.5 ms step, × √2 | one per ms, shared by both half-steps: 5 N(0,1) to E, 2 N(0,1) to I |
| Threshold test | every 0.5 ms | every 1 ms, with overshoot |

Network and keys are P3's:
- `k_r, k_w, k_noise = jax.random.split(PRNGKey(seed), 3)`.
- a, b, c, d and W = S from `run_izh.build_sens_fig3_network(seed, dtype, w_multiple=1.0)`; its
  noise, drive and v₀/u₀ outputs are not used.
- v₀ = −65, u₀ = b v₀.
- thal[t] = `jax.random.normal(jax.random.fold_in(k_noise, t), (1000,))` × 5 for E and × 2 for I.

## Equivalence gate (before signature)

`net_reference.simulate` gains one optional argument, `thal` (float64, shape (n_steps, n)). When it
is given, row t replaces the step's draw and `rng` is never read (it may be None); `drive` stays 0.
Its default path is unchanged and checked byte-for-byte against `e67f3b86` on seed 1. Both arms
take a, b, c, d and S from the same `net` dict (`net_reference.build_network`), cast to the arm's
dtype. The gate runs in float64, with x64 enabled before any array is built.

1. **Small, noise-free.** 50 cells: `build_network(np.random.default_rng(20261013), n_e=40,
   n_i=10)`. thal = 6.0 for every cell and row (the constant drive, given as thal to both arms),
   1000 ms. Rasters must be identical row for row, and extra u must agree at rtol 1e-9. A differing
   row passes only as a rounding flip: every cell that differs there has |v − 30| < 1e-9 at that
   test in both arms; then the step-2 bar applies to the rest of the run.
2. **Full network.** `build_network(np.random.default_rng(1))`, 1000 ms. thal from
   `np.random.default_rng(2)`: per row, 5·standard_normal(800) then 2·standard_normal(200),
   concatenated, the same arrays passed to both arms.
   - Up to the first row where the rasters differ (reported), rasters must be identical and extra u
     must agree at rtol 1e-9.
   - Over the whole run, total spike counts must agree within 1 %.
   - One-ulp threshold flips between JAX and NumPy can cascade, so full-run raster identity is not
     required.
3. **Precision report.** The variant with net and thal cast to float32 against the float64 variant,
   on the inputs of 2: spike-count difference and first divergent row (reported only).

A failure of 1 or 2 revises the variant before signature.

## Scored run

- Seeds 1–5, 10.2 s, the first 200 ms discarded, float32.
- One code SHA: `git rev-parse HEAD`, covering `run_izh.py`, `kernel_net_m.py` and
  `izh_analysis.py`. `p3b` exits 2 before any run when `git status --porcelain` lists a tracked
  change.
- Command: `run_izh.py p3b --seeds 1 2 3 4 5 --T 10.2 --out F:\warehouse\izh2003\p3b_scored --signed`.
- Each npz carries `dt_ms = 1.0`. The manifest carries `"arm": "net_m_order"` and the code SHA.
- Scoring: `run_izh.py p3b-score`, which reads `dt_ms` from each npz and calls
  `score_network_seed` unchanged. Signal: the population rate of
  all 1000 cells in 1 ms bins minus its mean, P3's signal.
- Criteria: N1–N4 as in `PREREG_P3.md` §Fig. 3:
  - same estimands and the "every seed" rule;
  - Welch settings from its step 1;
  - θ = 4 and alpha 8–13 Hz from `calibration/calibration_v7_15f36087.json`;
  - `izh_analysis.peak_excess` unchanged;
  - P3's rule for a seed with no spikes.
- `p3b-score` refuses:
  - a manifest whose `arm` is not `net_m_order`;
  - code SHAs that differ across seeds;
  - a missing seed;
  - any `dt_ms` other than 1.0.
- Verdict: P3b PASS if N1–N4 all pass, else FAIL with the failing ids.

## Reported only

- Seed 1 in float64: N1–N4.
- Extra u per spike for every seed.
- Variant against S1: per-seed values of N1–N4 side by side, with min–max ranges and no test. The
  two arms draw from different RNGs.

## Outcome language

- PASS: "the paper's Fig. 3 claims N1–N4 hold under `net.m`'s update order in jaxfne".
- FAIL: "N1–N4 fail under `net.m`'s update order", naming the ids.

Either way, P3 (the library kernel) keeps its FAIL. Fig. 2 (C4, C8) is outside P3b; see
`DIAGNOSTIC_P3.md`.
