---
name: jaxfne-science
description: Scientific simulation, falsification, HDP/TFNE analysis, Etudes, and quantitative evidence.
metadata:
  audience: agents
---
# jaxfne scientific procedure

## WHEN TO USE
SCIENCE work: scientific simulation, falsification, HDP/TFNE analysis,
Etudes, and quantitative evidence.

## AUTHORITIES TO READ
1. Repository AGENTS.md (truth gates, evidence grammar).
2. Live package-native source/field/probe/spectral/objective operators.
3. Protocol doctrine when the experiment has one (e.g.,
   `docs/doctrine/protocol_h_rbd_memory.md` for Protocol H).

## INVARIANTS
- Hypothesis, observables, controls/nulls, protocol, metrics, seeds, and
  acceptance/falsification criteria are declared before the decisive run.
- The smallest model that can falsify the claim is used.
- Failed prospective runs are preserved; a protocol change after failure
  freezes a new protocol and identifies why the change is scientifically
  justified.
- Software closure and scientific validation are separate: MCCs test
  integration; Etudes/validation experiments test scientific claims.
- No biological mechanism is inferred beyond the implemented model;
  relative/proxy quantities remain valid computational observables with
  explicit status.
- For RBS/RBD work use `F_X + F_H` with \(\dot W=0\) until Protocol W; do
  not assume scalar H or synaptic plasticity unless the experiment specifies
  it. Protocol H (`docs/doctrine/protocol_h_rbd_memory.md`) quantifies state
  memory via \(M(\Delta)\) before HDP.

## PROCEDURE
1. Declare hypothesis, observables, controls/nulls, protocol, metrics,
   seeds, and acceptance/falsification criteria before the decisive run.
2. Use the smallest model that can falsify the claim. Reuse package-native
   source/field/probe/spectral/objective operators.
3. Verify metric definitions independently on simple known data before
   interpreting a surprising result.
4. Separate software closure from scientific validation. MCCs test
   integration; Etudes/validation experiments test scientific claims.
5. Preserve failed prospective runs. If a protocol changes after failure,
   freeze a new protocol and identify why the change is scientifically
   justified.
6. For adaptive systems distinguish: plant controllability, actuator rank,
   restorative alignment, local stability, finite-amplitude reachability,
   and constraint/saturation behavior.
7. For RBS/RBD work use `F_X + F_H` with \(\dot W=0\) until Protocol W; do
   not assume scalar H or synaptic plasticity unless the experiment
   specifies it. Protocol H (`docs/doctrine/protocol_h_rbd_memory.md`)
   quantifies state memory via \(M(\Delta)\) before HDP.
8. Report neurophysiology when relevant: spikes/rates, native
   potentials/state, H/parameter trajectories, source, field/probe readouts,
   oscillatory/spectral dynamics, representation/calibration status, and
   pathologies.
9. Do not infer biological mechanism beyond the implemented model.
   Relative/proxy quantities remain valid computational observables with
   explicit status.
10. Keep exploratory large trajectories local/generated. Consolidate
    durable evidence into compact receipts, hashes, figures, and reproducible
    scripts/Etudes.

## STOP CONDITIONS
- Decisive run without declared falsification criteria; metric unverified
  on known data; freeze boundary ambiguity; identity or integrity failure.

## REQUIRED VERIFICATION
- Metric verification on known data; command/output trace in the receipt;
  evidence state and claim level per claim; durable evidence consolidated.

## FORBIDDEN INFERENCES
- Mechanism claims beyond the implemented model; unlabeled proxy/relative
  quantities; re-tuning a frozen protocol after observing its outcome.

## COMPLETION
- Receipt + trace + evidence states; durable evidence compacted; next
  checkpoint updated.