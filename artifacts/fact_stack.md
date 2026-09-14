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
- Generic finite-state rules register against declared state layouts,
  event/history inputs, RNG semantics, and a finite declared target set;
  structural change, exogenous ports, and undeclared targets are outside
  the grammar — not silently supported.

## State (H / RBS / RBD) and continuation ownership

- **RBS** (`H`) is a finite-dimensional dependency-state container — not
  intrinsically homeostasis and not one scalar controlling all operators.
- **RBD** is state dynamics involving `H`; plasticity is not required for RBD
  (`Ẇ = 0` is valid).
- Continuation preserves the **complete declared dynamic state** — membrane /
  recovery / spike-history / synaptic (`X`), `H`, weights / efficacy (`W`/`Θ`),
  delay ring (`B`), controller coordinates (`K`) where declared, and rule-owned
  auxiliary / bias state. No hidden state may live outside the carrier.

## Semantics layers

- **configured ≠ realized ≠ executed ≠ effective** — distinct lifecycle stages.
- **relative ≠ calibrated** — internal coordinates vs physical calibration.
- **proxy ≠ physical measurement** — projection/readout is not a solved field
  or calibrated instrument output unless explicitly validated.

## Representation and equivalence

- Compact storage must be **derivable from authoritative metadata**, not merely
  correlated with it. Observed correlations do not authorize universal compaction
  rules (e.g. sign→τ maps apply only where explicitly qualified).
- Scientific semantics outrank storage: a compact form is allowed only when
  equivalent under the declared criterion.
- Exactness is required where declared (bit-exact identity and continuation);
  elsewhere divergence is judged against predeclared observable-specific
  bounds, never ad hoc.

## Programme gates (authorized constraints)

- **REP-03:** do not lower `_SPARSE_DIRECT_N` without bit-exact sparse-direct ≡
  dense equivalence evidence.
- Minimum complexity subject to complete required semantics; no simplification
  that changes required semantics, API, numerics, or meaning.
- Public and runtime compatibility is preserved unless a release explicitly
  changes it; aliases are retained and deprecations declare replacements.
