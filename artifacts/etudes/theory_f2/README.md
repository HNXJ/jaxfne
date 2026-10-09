# F2: replication under divergence (declaration, 2026-10-09)

Family F2 of `artifacts/programme/theory_programme_proposal.md`. Declared before the
script exists or any run starts. Source evidence: K1h-nf, where a one-ulp change of the
initial weights moved decoding by up to 0.069, and K1h, where `pcl_stdp_h2` split from
`pcl_stdp` on one ulp in one weight.

## Question

In the PCL column (`artifacts/etudes/pcl/pcl_column_hdp.py`, rule `pcl_stdp`), how does
the difference between two runs grow during training when their initial weights differ
by one float32 ulp, and over how many sequences does pointwise equality stay meaningful?

## Design

- Seeds 20, 21, 22 (not used by any earlier gate). Phase 1 only: 600 training
  sequences with the excitatory plastic types (0, 1, 4, 5), the same runner and keys as
  `pcl_column_hdp.main`.
- Arms, run in lockstep sequence by sequence:
  - **base**: initial weights `w0`, presentation key `k_p1`.
  - **nudge**: `nudge_ulp(w0, 1)`, same key.
  - **ref**: `w0` with an independent presentation key (`fold_in(k_p1, 1000)`). Its
    distance from base sets the saturation level of two unrelated training histories.
  - **frozen**: the frozen runner on `w0` and on `nudge_ulp(w0, 1)` over base's 600
    sequences. Each sequence restarts the membrane state, so this measures amplification
    within one sequence with no carry-over through weights.
- Measures after each sequence n, over the plastic edges and all simple and complex
  cells:
  - D_w(n) = ‖w_a(n) − w_b(n)‖₁ / ‖w_a(n)‖₁;
  - D_s(n) = Σ|c_a − c_b| / Σ(c_a + c_b), with c the per-cell spike counts of sequence n.
  - D_sat = D_w(base, ref) averaged over the last 50 sequences.

## Hypotheses

- **H2.1, growth.** log D_w(base, nudge) rises linearly in n, so the growth is
  exponential, with rate λ > 0 per sequence. λ is fitted by least squares over the window
  where 1e-6 ≤ D_w ≤ 0.1·D_sat. Rejected if that window has fewer than 10 sequences
  or the slope's 95% interval includes 0.
- **H2.2, horizon.** The horizon n* = the first n with D_w(base, nudge) ≥ 0.5·D_sat is
  reached within 600 sequences in all three seeds. Rejected in any seed where it is not.
- **H2.3, no carry-over without plasticity.** In the frozen arm D_s does not grow with
  n: the least-squares slope of D_s on n has a 95% interval that includes 0 or lies
  below it.

## Consequence if H2.1 and H2.2 hold

A pointwise equality gate on this column has a meaningful horizon of about n*
sequences. Later replication gates compare against a measured noise floor, as K1h-nf
did, rather than demand equality. If H2.2 fails, pointwise gates remain valid over a
full phase 1 and K1h's split needs another explanation.

## Outputs

`artifacts/etudes/theory_f2/f2_divergence.py` writes `confirm/f2_seed{20,21,22}.json`
with D_w and D_s per sequence for each arm pair, D_sat, λ with its interval, the window,
n* and wall time. The verdicts are computed by a separate function that reads only the
JSON files.

## Result (2026-10-09, seeds 20–22, 600 sequences, about 1580 s per seed)

| seed | first spike difference | D_w just before → at it | max D_w / D_sat | λ per sequence (95% interval), window length | n* |
|---|---|---|---|---|---|
| 20 | sequence 61 | 2.4e-7 → 2.4e-4 | 0.061 | 1.09e-3 (1.02e-3, 1.17e-3), 540 | none |
| 21 | sequence 110 | 2.8e-7 → 2.3e-4 | 0.113 | 6.55e-3 (6.26e-3, 6.84e-3), 420 | none |
| 22 | sequence 357 | 3.0e-7 → 8.5e-4 | 0.092 | 2.66e-3 (2.42e-3, 2.90e-3), 244 | none |

D_sat was 0.047–0.049. Verdicts by the declared rules:
- **H2.1** is supported, but its model does not fit; see the reading below.
- **H2.2** is rejected in all three seeds.
- **H2.3** is supported: in the frozen arm D_w stayed at 8.8e-8 and D_s showed no trend. Seeds
  20 and 22 pass on an all-zero D_s series (a degenerate interval); seed 21 reached at most 0.03
  in a single sequence.

**Reading.** The divergence is not a smooth exponential. Until the first sequence in which
per-cell spike counts differ (sequence 61, 110 and 357), the counts are identical; spike times
were not compared. Over that stretch D_w stays at float32 rounding level (7e-8 to 3e-7). At that
sequence D_w jumps by three orders of magnitude. It then drifts slowly, at λ of 1e-3 to 7e-3 per
sequence, and its maximum is 6–11% of the distance between unrelated training histories (seed
20 ends at 4.6%). After the first difference, mean D_s was
0.011–0.027, against 0.16–0.17 for the reference arm. The fitted λ describes this slow drift,
so H2.1 passes by the letter of its rule while the declared model, exponential growth from
one ulp, does not describe the data.

**Consequence, and a departure from the declaration.** The declaration said that if H2.2
failed, pointwise gates would remain valid over a full phase 1. That conditional assumed that
failing to reach 0.5·D_sat meant staying equal. The data show a third case it did not anticipate:
a jump that ends equality but stays below 0.5·D_sat. Pointwise equality of two implementations
holds up to the first spike-count difference, which came 61 to 357 sequences in and varied by
seed. Beyond it, runs one ulp apart stay close, at about a tenth of the reference distance at
most, but they are not equal. A replication gate on this column should therefore compare
against a measured one-ulp noise floor, as K1h-nf did, and not demand equality past the first
difference. K1h's split is consistent with this picture; that was not tested here.

**Departures and limits.**
- Seed 21's window is the set of sequences meeting the declared bounds, which is not
  contiguous. The first contiguous run (sequences 110–446) gives λ = 8.5e-3; the verdict is
  unchanged.
- The least-squares interval assumes independent residuals. The drift series is
  autocorrelated, so the intervals are likely too narrow.
- Phase 1 only, one rule (`pcl_stdp`), float32.
