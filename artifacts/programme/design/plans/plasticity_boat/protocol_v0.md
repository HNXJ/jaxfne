# Plasticity Boat: scope and protocol, v0.1

Status: PROPOSED 2026-11-19. Not frozen. Confirmatory runs are held until the owner freezes §3 to §5.
Owner decision (2026-11-19): 10-neuron fixture for validation and falsification only; 100-neuron common LIF circuit for the primary eight-strategy competition; short computational neuroscience paper, 8 pages (internal target); draft by 2026-11-20.

## 1. Scope

- 10-neuron LIF fixture (M2, contract §12): validation, debugging, falsification. Frozen values are unchanged. Existing results (contract §13.8 R-01 to R-12) and failures are preserved, not re-labelled.
- 100-neuron LIF circuit: primary competition. Not yet built.
- A competition result is claimed only after it is reproduced from scripts and passes the §4 controls.

## 2. Critical path (before any 100-neuron run)

1. **Directional observability.** The fixture's −x motor has no distance drive, so F1 pushes +x from every start. Evidence: R-09 (8 angles), R-11 (3 angles), R-12 (dev starts, see below). A distance-only sensor cannot express direction, so the competition would measure the sensor, not the plasticity rule.
   - Option A (recommended): signed-position population encoder. Two sensory populations per axis, drive proportional to max(0, ±p_axis). Direction is an explicit input, stated in the paper.
   - Option B: scalar distance plus a sign bit per axis. Simpler, but the bit is an engineered shortcut.
   - Owner chooses A or B.
2. **Positive control.** A known-correct directional controller, with the directional encoder on the 10-neuron fixture, must bring the boat to p* from every dev start. Failure means the pipeline is broken; no competition runs. (EX-9 in todo_stack.md.)
3. **Negative controls.** Zero drive and stationary inputs keep the boat at R = 5 (F0 already holds at 5.000).
4. **Evaluation validity.** Paired starts; identical circuit initialisation and seeds across strategies; identical K; metric frozen before any confirmatory run.

## 3. Eight strategies (proposed; 100-neuron circuit)

| id | plasticity rule | role |
|----|-----------------|------|
| S1 | none (static weights) | baseline |
| S2 | none; drive scaled by H (F2) | H effect without weight change |
| S3 | H-scaled PV→E plasticity (F6) | inhibitory plasticity |
| S4 | three-factor, one shared reward for all E→motor synapses (new) | ablates channel identity |
| S5 | channel-specific three-factor eligibility, reward −Δe (F3) | primary rule |
| S6 | S5 with running-mean reward baseline (F3b) | reward-baseline control |
| S7 | S5 with the distance signal shuffled across time (F5) | reward-structure control |
| S8 | reward-free pair-based STDP on E→motor (new) | unsupervised control |

- F4 is dropped: it equals F2 by construction (critic finding 7b).
- F2b is reported only in the 10-neuron registry, not among the eight.
- S4 and S8 are not implemented. They must be built and pass the §2 controls first.

## 4. Metrics and controls (freeze before confirmatory)

- Fixed episode length K = 600 control intervals (12 s), declared here.
- Primary metric (proposed, resolves EX-20): fixation occupancy = fraction of the final 20% of intervals with distance ≤ 1.0. K-independent in its threshold; reported with the final distance.
- Secondary: final distance at K; drift rate after the transient.
- Controls on the same starts: S1 static, zero drive, stationary input, directional positive control.

## 5. Splits and statistics

- Development starts: 20, θ ~ Uniform[0, 2π), `numpy.random.default_rng(2025)`, R = 5. Used for debugging and falsification only. Never used to choose a strategy.
- Held-out starts: 40, `default_rng(2026)`, used once, after the owner freeze and all §2 controls pass.
- Unit of inference: the start. Bootstrap over starts, 10 000 resamples, `default_rng(0)`, 95% percentile CI. Paired contrasts against S1.
- Effective-variation gate before any CI: at least 0.9·N distinct values per contrast, otherwise reported as degenerate with no CI.

## 6. Locks and authority

- Methodological lock: 10-neuron fixture values (locked, M2). 100-neuron circuit and §3 to §5 locked on owner freeze.
- Not authorised: submission, publication, release, pushes to dev or main. Author identities and submission metadata wait for stage 13.

## 7. Feasibility by 2026-11-20 (honest)

| Deliverable | Status by deadline |
|-------------|--------------------|
| 10-neuron falsification: F1 bias, dev starts, sign split | achievable (R-09, R-11, R-12; R-12 complete, 20 of 20 starts) |
| Directional positive control on 10 neurons | achievable only if Option A is chosen today |
| 100-neuron circuit built and controlled | not expected by deadline |
| Eight-strategy competition results | not expected by deadline; not claimed |

The competition results will not be claimed in the 2026-11-20 draft. If the owner needs them, the deadline moves.
