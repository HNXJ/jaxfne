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
subtracted inside the modulus: |Σ (r_k − m) e^{2iθ_k}| ≤ Σ (r_k − m). ∎ Tier T2 (second-reviewer,
2026-10-09).

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
- **Ceiling.** OSI*(h) is computed over 39 thresholds, h from 0 to 0.95 of each unit's
  maximum drive in steps of 0.025, applied per unit. The ceiling is max_h of the median OSI.

**Correction before any ceiling was computed (2026-10-09).** The count was written as 41; the
range and step give 39. Building the checks also showed a second lemma:

**L2 (no tuning without a threshold).** At h = 0, r is linear in the events. When each sweep
fully crosses the receptive field, every pixel receives one ON and one OFF event at every
orientation. So Σ_t drive(t) is the same for all orientations, and OSI = 0 for every unit.
Any OSI in this model therefore comes from the threshold, through coincidence of events
within a time step, and the ceiling is attained at some h > 0. Tier T2 (second-reviewer,
2026-10-09); found by the worker building the checks.

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

## Result (2026-10-09, `f4_ceiling.py`, 256 units, RF 7)

| h (fraction of each unit's max drive) | 0 | 0.075 | 0.15 | 0.175 | 0.3 | 0.4 | ≥ 0.425 |
|---|---|---|---|---|---|---|---|
| median OSI | 0.000 | 0.077 | 0.286 | 0.332 | 0.379 | 0.818 | 1.000 |

- **H4.1 is rejected by the declared rule; the Reading below finds the rejection
  uninformative.** The ceiling is 1.000, reached at h = 0.425, and median OSI passes 0.3
  between h = 0.15 and 0.175. At h ≥ 0.425 most units respond to one orientation only, so
  OSI = 1 by definition.
- **H4.2 holds.** O1's medians (0.267, 0.266, 0.272) are below the ceiling. With a
  ceiling of 1.000 this is as uninformative as H4.1.
- The RF sweep (9, 11, 13 px) gave the same ceiling of 1.000.
- L2 holds numerically: OSI is 1.4e-15 at h = 0.
- On this grid, the 8 kernels at odd orientations (22.5° + k·45°) prefer an adjacent even
  orientation. Bars at 0°, 45°, 90° and 135° fall on the pixel grid and give more coincident
  events. This shifts preferred orientations; it does not limit OSI.

**Reading (revised after second-reviewer, 2026-10-09).** The declared ceiling is degenerate.
Each unit's threshold is a fraction of its own maximum drive over the whole protocol. Near
that maximum, only the single best stimulus clears the threshold, so under this rule
OSI → 1 for any kernel. The responsiveness rule of the column was replaced by r > 0,
so near-silent units count. H4.1's rejection is therefore true by the declared rule but
uninformative, and it does not show that the criterion is reachable at the column's activity
level.

What stands:
- L1.
- L2: with no threshold, the summed response is identical for all orientations, so any
  tuning comes from the nonlinearity acting on coincident events.
- The idealized median OSI crosses 0.3 at h between 0.15 and 0.175. Whether O1 operates at a comparable activity level is not measured, because
  the O1 files hold no per-presentation counts.
- The grid anisotropy above.

The working paper's sentence on "a ceiling of the stimulus and measure" is neither confirmed
nor refuted by F4. The P3 dependency stays open.

**Amendment A is withdrawn before its run.** Its threshold rule inherits the same
degeneracy, so H4.3 could not discriminate.

## Amendment A (declared 2026-10-09; withdrawn before its run, see Reading): integration window

Kept as the record of what was declared. Not run.

- **Model.** As above, except the drive passes through a leaky integrator before the
  threshold: v(t) = v(t−1)·e^(−DT/τ) + drive(t), and r = Σ_t max(v(t) − h, 0). The thresholds
  are the same 39 fractions of each unit's max v. τ ∈ {DT, 2, 5, 10, 18, 50} ms; τ = DT is
  close to the per-step model above. There is no spike reset, refractoriness, noise or
  inhibition.
- **H4.3.** The ceiling falls as τ grows, and at τ = 18 ms (the simple cells' τ_m) it is
  below 0.3. If it holds, membrane integration alone is enough to keep this kernel family
  under the criterion. Rejected if the ceiling at 18 ms is 0.3 or above; then integration
  alone does not explain O1, and noise, reset and inhibition remain candidates. Monotonic
  decrease is reported, not tested.
- **Output.** `f4_ceiling.json` gains `integration_sweep`, the ceiling and its h for each τ.
