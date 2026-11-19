# Plasticity Boat: frozen criteria v1 (100-neuron development model)

Frozen 2026-11-19 before any test of the revised model. Owner-approved changes: position plus velocity sensing, and removal of the built-in tonic drive for the 100-neuron model only. The 10-neuron fixture is unchanged. Any change to this file needs owner approval and a new version. Nothing here was tuned on outcomes.

## Model (dev-100)

- 80 E and 20 PV neurons; the four connection types as in the 10-neuron fixture; 9900 edges.
- Built-in tonic drive removed: per-type declared drive 0.0 for E and PV (`NeuronType.drive = 0.0`). Declared drive, not a post-hoc array edit.
- Motor groups, 5 E neurons each: +x = [0-4], -x = [5-9], +y = [10-14], -y = [15-19].
- Non-motor E [20-79]: distance drive g * e / (e + 1), g = 24, e = |p|.
- Motor E: directional drive g * u(z), g = 24, u(z) = max(0, z) / (max(0, z) + 1).
- Decoder: force = f0 * (c+x - c-x, c+y - c-y), f0 = 5. Force of interval k applied in interval k+1.

## Sensing

- Position only (negative control N1): z_+x = -p_x, z_-x = p_x, z_+y = -p_y, z_-y = p_y.
- Position plus velocity (primary): z_+x = -(p_x + tau * v_x), z_-x = p_x + tau * v_x, and likewise for y. Primary tau = 0.1 s (5 control intervals of 20 ms). Sensitivity tau = 0.5 s is reported and is not used to select.

## Resting and responsiveness criteria (replace B1 and B2 for the dev model)

- R1 intrinsic silence: no external input (no distance, no directional), boat fixed, 100 intervals. Mean E rate <= 1 Hz and mean PV rate <= 1 Hz.
- R2 stationary task input: distance drive held at e = 5, boat fixed, 600 intervals. Mean E rate in [5, 100] Hz. The upper bound 100 Hz is the B2 bound, unchanged. The lower bound 5 Hz is new. E rate minus R1 rate >= 5 Hz.
- R3 directional response: boat fixed at (-5, 0), (5, 0), (0, -5), (0, 5), 100 intervals each. Mean rate of the group pointing toward the target minus the opposite group >= 5 Hz at each position.
- R4 no saturation: every neuron's mean rate <= 100 Hz in R2 and R3.
- R5 build check: drive array of the dev model is 0.0 for all 100 neurons. The 10-neuron fixture drive array is identical before and after the change (hash compared).

## Positive control and regression

- P1 (primary, position plus velocity): 20 development starts, K = 600, seed 0, starts from numpy.random.default_rng(2025).uniform(0, 2 pi, 20), R = 5. PASS iff every start has fixation occupancy >= 0.9, where occupancy = fraction of errors[480:600] with error <= 1.0. Same threshold as EX-24 A3; not weakened.
- N1 (negative control, position only): same starts, same criterion. Reported; expected to fail; not a gate.
- G0 regression: 10-neuron A0 replay exact (max abs diff <= 1e-6 against lane_C F1 theta = 0); 10-neuron EX-24 start_00 errors identical to the stored JSON.

## Outcome rule

- P1 passes and R1 to R5 pass: the dev model may proceed to the eight-strategy protocol freeze. The competition is not run until that freeze is recorded.
- P1 fails: the dev model does not proceed. Report the failure and its trace. Do not change tau, g, f0, d0, K, the radius, or the starts.
- Any of R1 to R5 fails: report unchanged. Failures are kept; no tuning.

## Development evidence

All results under this file use development starts only. Held-out starts (default_rng 2026) stay unused until the owner's methodological lock. The EX-24 twenty-start run remains development evidence, not confirmatory evidence.
