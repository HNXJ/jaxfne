# Remaining work — v0.4.23 → v0.4.24

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.22 shipped** (tag `v0.4.22`, PyPI, GitHub release, RTD stable).
Published predecessor: `v0.4.21` (immutable).

---

# v0.4.23

Entry: v0.4.22 shipped; W3/W7 classifications hold zero `UNKNOWN`; W8 baseline current.

- 23-HDP-02 — OPEN: scalar-etude residual diagnosed as stale frozen expectation (null-HDP ≡ off by identity theorem; see receipt §Timeline); needs human pick: re-baseline scalar frozen value OR redesign scalar arm; blocks seal
- 23-VERIFY-01 — full 0.4.23 RC + CI + seal + publish chain
- 23-STACK-01 — review and rewrite this file from sealed v0.4.23 before v0.4.24 work

**Ordered dependencies (remaining):** 23-DELAY-01 fully closed (delayed HDP receipt + compaction NO_CHANGE closeout); 23-HDP-02 OPEN and blocking seal (human protocol pick required: re-baseline scalar frozen value OR redesign scalar arm); disconnected-null boundary now covered on the engaged path — remaining human confirm is only the diag-None-for-identity contract (sealed in HDP-01); 23-SIMP-01 CLOSED per `artifacts/programme/v0_4_23_simp01_receipt.md` (HP-05 storage-setup defect fixed; class green with delays engaged); 23-SIMP-02 closed as NO_CHANGE per `artifacts/programme/v0_4_23_simp02_receipt.md` (parameter-heavy helper, net ~0); 23-SIMP-03 closed as NO_CHANGE per `artifacts/programme/v0_4_23_simp03_receipt.md` (T×E already gated; residual T×N not worth variants); 23-SIMP-04 closed as NO_CHANGE per `artifacts/programme/v0_4_23_simp04_receipt.md` (no per-mode winner); 23-SIMP-05 closed as NO_CHANGE per `artifacts/programme/v0_4_23_simp05_receipt.md` (semantic branches dominate; universal kernel would be parameter-heavy); 23-STOCH-01 closed as NO_CHANGE per `artifacts/programme/v0_4_23_stoch01_receipt.md` (K_t owned; cont-vs-cont exact incl. delay+HDP+noise; bulk-vs-cont boundary declared); 23-REC-01 CLOSED as IMPROVEMENT per `artifacts/programme/v0_4_23_rec01_receipt.md` (strided capture + memory_report + continuation toggle fix); 23-EDGE-01 CLOSED as IMPROVEMENT per `artifacts/programme/v0_4_23_edge01_receipt.md` (EDGE_OWNED: compact-weight tune repair; HP-07 + 3/4 population green); 23-LAW-01 CLOSED as IMPROVEMENT per `artifacts/programme/v0_4_23_law01_receipt.md` (aux-carrying structured rule; primitive exercised, numerics unchanged); 23-FIN-01/23-SYSID-01/23-JDNA-01 closed as NOT_REQUIRED per `artifacts/programme/v0_4_23_conditional_not_required_receipt.md` (conditions false; no dependents); REP-03 gate retains `_SPARSE_DIRECT_N=5000` until bit-exact equivalence (`artifacts/fact_stack.md`). Jomission downstream migration remains **unverified** post-COMPAT-JOM-01. Open pre-existing defects (not introduced here): population `h_dim` IndexError (material; proposed 23-EDGE-01), disconnected-null `diag is None` (needs human decision on forced HDP engagement).

---

# v0.4.24

Entry: v0.4.23 shipped; W8 shows cumulative programme receipts from `8823520` baseline.

- 24-ENT-01 — confirm 0.4.24 entry criteria (after 23-STACK-01)
- 24-W16-6 — neurobiophysical geometry / bounded augmentation (W16.6); named observable, metric, ε
- 24-FIELD-01 — field/LFP proxy computational audit with declared error bounds
- 24-JAX-01 — JAX execution profile programme (measure before JIT changes)
- 24-EQUIV-01 — per-observable equivalence acceptance (exact or `d≤ε` pre-declared)
- 24-AUDIT-01 — independent adversarial core audit (implementer ≠ sole auditor)
- 24-PKG-01 — installed-package boundary audit (which LOC belongs in the runtime distribution; executable protocol/research code destination: tests/benchmarks/examples, never artifacts/ for aesthetics; alongside kernel-count and test-minimization reviews; not before v0.4.23 closes)
- 24-EXIT-01 — verify programme exit criteria (roadmap §5.2)
- 24-VERIFY-01 — full 0.4.24 RC + CI + seal + publish chain
- 24-STACK-01 — seal v0.4.24; archive or clear this stack
