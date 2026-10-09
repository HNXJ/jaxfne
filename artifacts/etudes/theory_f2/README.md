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
