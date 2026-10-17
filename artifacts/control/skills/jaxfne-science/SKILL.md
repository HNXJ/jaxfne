---
name: jaxfne-science
description: Scientific simulation, falsification, HDP/TFNE analysis, Etudes, and quantitative evidence.
metadata:
  audience: agents
---
# jaxfne scientific procedure

## WHEN
SCIENCE work: scientific simulation, falsification, HDP/TFNE analysis, Etudes, and quantitative evidence.

## AUTHORITIES
1. Repository `artifacts/AGENTS.md` (required tests, evidence pipeline).
2. Protocol rules (e.g. `docs/doctrine/rbs_rbd_hdp.md`).

## RULES
- Numerical dtype is part of the declared protocol. Release gates and the current etudes use float32. Do not promote arrays to float64 unless that protocol says so.
- Hypothesis, observables, nulls, protocol, metrics, and acceptance criteria declared beforehand.
- Failed prospective runs are preserved (never tuned after observing validation).
- No biological mechanism inferred beyond implemented model; proxy quantities have explicit status.
- RBS: H is finite-dimensional relative state; H != homeostasis.
- Claim-conditioned verification: plan evidence from the scientific claim
  (V=V(claim)) — a "runs" claim needs execution, a rate claim needs signal
  analysis plus numerical adequacy, a plasticity claim needs an intervention
  on the mutable rule.

## STEPS
1. Declare protocol and falsification criteria before run.
2. Use smallest model capable of falsifying the claim; reuse package-native operators.
3. Verify metric definitions on simple known data before interpreting results.
4. Run prospective simulation and record raw + interpretation receipts.

## STOP
- Missing control/null; ambiguous metric definition; unrecorded prospective run.

## VERIFY
- Receipt with exact parameter and hash provenance.

## DONE
- Evidence preserved with declared polarity (POSITIVE, NEGATIVE, UNRESOLVED).
