# Plasticity Boat: criteria v2 (100-neuron development model, labelled revision)

Owner decision, 2026-11-19: approve a labelled dev-100 criteria v2. Frozen v1 (`criteria_v1_frozen.md`, sha256 d57fd981…037ee) is retained unchanged as failed evidence. v1 results stay in `artifacts/_temp/plasticity_boat/dev100/`. v2 results go to `artifacts/_temp/plasticity_boat/dev100_v2/`.

v2 is written before any v2 run. Every selection rule below is fixed here and was not chosen from outcomes.

## Carried from v1 (unchanged)

- Plant: dt 0.02 s, damping 0.5, p* = 0, identity actuator, R = 5, K = 600, seed 0.
- Membrane, synaptic and plasticity parameters of the 100-neuron model (`dev100_model.py`). Drive of all 100 neurons declared 0.0.
- Network: 80 E (0-79), 20 PV (80-99), 9900 edges. Motor groups: +x 0-4, -x 5-9, +y 10-14, -y 15-19.
- Directional motor drive g_d * u(z), g_d = 24, d0 = 1. Motor drive is not changed in v2.
- Sensing: P1 = position plus 0.1 s velocity (primary); N1 = position only (negative control, not a gate); T05 = 0.5 s velocity (sensitivity, not a gate).
- Gate P1: every validation start has fixation occupancy >= 0.9, where occupancy = fraction of errors[480:600] with error <= 1.0.
- R1 intrinsic silence (E and PV mean <= 1 Hz). R2 stationary input (E mean in [5, 100] Hz; PV in [0, 100] Hz; E(R2) - E(R1) >= 5 Hz). R3 directional response (toward minus opposite group mean >= 5 Hz at each of four positions). R4 no saturation (every neuron <= 100 Hz in R2 and in each R3 block). R5 build check. No threshold is weakened.

## Changed in v2

### C1. Sensory gain g_s (non-motor E distance drive only)

- Non-motor E drive = g_s * e / (e + d0), e = |p|. v1 value g_s = 24.
- Motor E drive is unchanged (g_d = 24). Only the stationary sensory gain of non-motor E is changed.
- Sweep grid, descending: g_s in {24, 18, 12, 9, 6, 4.5, 3}. g_s = 24 reproduces v1 R2 (failure preserved).
- Membrane, synaptic and plasticity parameters are not changed.

### C2. Decoder: population mean

F_g(t) = kappa * (1 / N_g) * sum_{i in g} n_i(t), N_g = 5.

- n_i(t) = spike count of neuron i in the 20 ms control interval t (units: spikes per interval).
- F_x = F_{+x} - F_{-x}; F_y = F_{+y} - F_{-y}; force units = kappa * spikes per interval per neuron. Force of interval t applied in interval t+1 (unchanged timing).
- kappa grid, ascending: {5, 10, 15, 20, 25}. Under mean decoding, kappa = 25 gives the same force as v1 (sum decoding, kappa = 5).
- v1 sum decoding is not used in v2 except in the regression check V0 below.

### C3. Selection rule (applied in this order, no outcome-based choice)

1. Resting sweep. For each g_s in the grid (fixed boat, no decoder), run R1 to R4. The pass set P = {g_s : R2, R3 and R4 all PASS}.
2. Calibration. Let g* = the largest g_s in P. On the calibration set (C4, P1 sensing), run kappa in ascending order. The first kappa with all 10 calibration starts at occupancy >= 0.9 is selected.
3. If no kappa passes at g*, move to the next smaller g_s in P and repeat step 2. If no (g_s, kappa) pair passes, the v2 track stops and is reported. No further grid is added.

### C4. Start sets

- Calibration set (new in v2): numpy default_rng(2027).uniform(0, 2 pi, 10), R = 5. Development only. Used only for step 2 of C3.
- Validation set: the 20 v1 development starts (default_rng(2025), R = 5). Used only for the validation runs under the selected (g_s, kappa). Their outcome is not used to select g_s or kappa.
- Held-out starts (default_rng(2026)) stay unused.

### C5. Validation

Under the selected (g_s, kappa): run P1, N1 and T05 on the 20 validation starts with K = 600, seed 0.

### C6. Firing rates

Report, for the selected configuration:
- zero input (R1, E and PV mean Hz);
- stationary task input (R2, E and PV mean Hz, max single-neuron Hz);
- moving task input (closed-loop validation runs: E mean Hz, PV mean Hz, max single-neuron Hz over the 600 intervals, per start, from the per-neuron spike totals).

## Gates and outcome

- P1 PASS and R1 to R5 PASS under the selected (g_s, kappa): the dev model may proceed to the eight-strategy protocol freeze (EX-26; the strategy list still needs owner confirmation). The competition is not run until that freeze is recorded.
- P1 FAIL, or any of R1 to R5 FAIL: report unchanged. No tuning after outcomes. Any further change needs criteria v3 and owner approval.
- The v1 20/20 P1 result does not transfer to v2 (the decoder is different). v2 is validated separately.

## Verification required before freeze

- V0 regression: the v2 rollout with g_s = 24, sum decoding, kappa = 5 reproduces the stored v1 P1 start 00 `errors` exactly (max abs difference 0.0).
- Independent review (a reviewer other than the writer) of: the selection rule followed, the decoder units, the start sets, the v1 regression, the firing-rate computation, and held-out unused.
- Protocol freeze is recorded only after both checks pass. The recorded sha256 of this file and of `dev100_v2.py` go into the gate record.

## Development evidence

All v2 results use development starts only. Held-out starts stay unused until the owner's methodological lock. Each v2 result is labelled v2 and cites this file.
