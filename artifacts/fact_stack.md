# Authorized facts — jaxfne

Small set of **stable, human-authorized** project facts. **Not evidence.**
Repository code, tests, and receipts remain evidence and may contradict a fact.

**Agents:** read, use, test, and challenge these facts. **Do not add, remove, or
change any fact without explicit human authorization.** If evidence contradicts a
fact, flag the contradiction for human review — do not silently rewrite this file.

---

## Plasticity (HDP)

- **HDP** = Hidden-state Dependent Plasticity.
- HDP is the **general plasticity abstraction**, not one homeostatic mechanism.
- General parameter dynamics may depend on the state a rule requires:
  `dΘ/dt = P(X, H, B, Θ)` — each rule uses only the variables it needs.
- A **named plasticity mechanism** is a particular choice of dependent state,
  state dynamics, plasticity law, and modified parameter / effective gain.
- **H ≠ HDP.** RBS / hidden state (`H`) and plasticity (`Θ` or `W`) are distinct.
- Do **not** require a separate plasticity engine merely because a mechanism has
  a conventional name (STDP, STP, homeostasis, etc.).

## State (H / RBS / RBD)

- **RBS** (`H`) is a finite-dimensional dependency-state container — not
  intrinsically homeostasis and not one scalar controlling all operators.
- **RBD** is state dynamics involving `H`; plasticity is not required for RBD
  (`Ẇ = 0` is valid).

## Semantics layers

- **configured ≠ realized ≠ executed ≠ effective** — distinct lifecycle stages.
- **relative ≠ calibrated** — internal coordinates vs physical calibration.
- **proxy ≠ physical measurement** — projection/readout is not a solved field
  or calibrated instrument output unless explicitly validated.

## Representation compaction

- Compact storage must be **derivable from authoritative metadata**, not merely
  correlated with it. Observed correlations do not authorize universal compaction
  rules (e.g. sign→τ maps apply only where explicitly qualified).

## Programme gates (authorized constraints)

- **REP-03:** do not lower `_SPARSE_DIRECT_N` without bit-exact sparse-direct ≡
  dense equivalence evidence.
