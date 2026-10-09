# F4: criterion reachability (declaration, 2026-10-09)

Family F4 of `artifacts/programme/theory_programme_proposal.md`, opened by Hamm on 2026-10-09.
Declared before any script exists. Source evidence: control O1
(`artifacts/etudes/pcl/pcl_orient_control.py`). Hand-set oriented kernels in the standalone
column decoded orientation at 1.00 but reached a median OSI of only 0.266–0.272, below the 0.3
criterion of gates C1 and K1.

## Question

For the declared stimulus (drifting bars: `pcl_column.STIM`, width 2 px, speed 0.13 px/ms,
8 orientations, both directions) and the OSI of `pcl_column.osi`, how high can the median
OSI of a simple-cell population with the O1 kernel family go? Is 0.3 reachable at all,
before any learning or spiking enters?

## Lemma (exact)

**L1 (baseline bound).** For a nonnegative tuning curve r_k over the 8 orientations
θ_k = kπ/8, OSI = |Σ_k r_k e^{2iθ_k}| / Σ_k r_k ≤ 1 − 8·min_k r_k / Σ_k r_k.

Proof. Σ_k e^{2iθ_k} = 0 for 8 equally spaced θ_k over [0, π), so the minimum m can be
subtracted inside the modulus: |Σ (r_k − m) e^{2iθ_k}| ≤ Σ (r_k − m). ∎ Tier T2 (pending
second-reviewer).

So OSI ≥ 0.3 requires the weakest orientation to carry at most 70% of the mean response.
Equivalently, OSI is the ratio of the tuning curve's second circular Fourier coefficient
to its mean.

## Idealized model (the bound to compute)

- **Input.** The expected event images of `stimulus_step`: rate `events` on ON pixels where the
  bar enters and on OFF pixels where it leaves, with `noise_hz` set to 0. This is
  deterministic; no Poisson sampling. All c0 offsets of the test protocol are covered by a
  uniform grid of 32 offsets over the drift range, and both directions are used. Sequences
  last `seq_ms` at `DT`.
- **Unit.** One simple cell of each of the 16 O1 features at each of the 16 positions: the O1
  kernel (`oriented_kernels`, normalized as in O1) dotted with the 7×7×2 input patch.
  Response r = Σ_t max(drive(t) − h, 0), with threshold h. No spikes, no inhibition, no
  noise.
- **Population measure.** The median OSI over units with Σ_k r_k > 0, as in `pcl_column.osi`.
  The responsiveness rule (rate ≥ 0.5) has no spike analogue here and is replaced by
  r > 0.
- **Ceiling.** OSI*(h) is computed over 41 thresholds, h from 0 to 0.95 of each unit's
  maximum drive in steps of 0.025, applied per unit. The ceiling is max_h of the median OSI.

## Hypotheses

- **H4.1 (unreachable even ideally).** The ceiling is below 0.3. If it holds, the 0.3
  criterion cannot be reached by this kernel family under this stimulus with any threshold,
  so O1's failure is a property of stimulus, receptive-field size and measure, not of the
  column. Rejected if the ceiling is 0.3 or above.
- **H4.2 (the idealization bounds the column).** O1's simulated median OSIs (0.266–0.272) lie
  at or below the ceiling. Rejected if any exceeds it. A rejection would mean spiking or
  inhibition sharpens tuning beyond the linear–threshold ideal, and that the ceiling is no
  bound for the column.
- **Prediction, not a test.** The ceiling is recomputed at receptive-field sizes 9, 11 and
  13 px, with lobes unchanged across the bar and therefore longer along it. The smallest size
  at which the ceiling reaches 0.3 is reported as design guidance for any future tuning
  gate; "none" if no size does.

## Consequence

Adopt process change P4 for tuning gates: before a gate declares an OSI criterion, the F4
script computes the ceiling for its stimulus and kernel family, and the declaration states
it. If H4.1 holds, the criterion of C1 and K1 is recorded as having been unreachable for
this family under this stimulus. Gate verdicts already committed are not changed.

## Outputs

`artifacts/etudes/theory_f4/f4_ceiling.py` writes `f4_ceiling.json`: OSI*(h) per threshold,
the ceiling and the threshold that attains it, per-unit tuning curves at that threshold, the
RF-size sweep and the verdicts.
