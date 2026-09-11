# Remaining work — v0.4.23 → v0.4.24

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.22 shipped** (tag `v0.4.22`, PyPI, GitHub release, RTD stable).
Published predecessor: `v0.4.21` (immutable).

---

# v0.4.23

Entry: v0.4.22 shipped; W3/W7 classifications hold zero `UNKNOWN`; W8 baseline current.

- 23-DELAY-01 (remainder) — delay/history compaction only (delay classes; dense ring already bit-exact); delayed registrable HDP probe `event_{t-d} → H_t → P → Θ_t → I_t` CLOSED per `artifacts/programme/v0_4_23_delay01_registrable_receipt.md`
- 23-SIMP-01 — delay helper factorization (L-01): internal metadata snapshot only; preserve traced/host/JIT asymmetry; gate `test_compat_jom01_regressions`
- 23-SIMP-02 — JIT dispatch consolidation (L-02): `_dispatch_jit_cached` in `_model_simulate.py`; preserve cache keys and guard names
- 23-SIMP-03 — scan recording audit (G): measure `T×N`/`T×E` stacks when `record_*` false; smallest static specialization without compile explosion
- 23-SIMP-04 — compact tau decay (L-05): benchmark only on `sign_from_receptor` / `from_mechanism_table` paths (per-edge random tau: gather **slower**, R-01)
- 23-SIMP-05 — HDP kernel unification: defer until 23-DELAY-01 delayed registrable semantics fixed
- 23-STOCH-01 — continuation-owned stochastic inputs where required (per-path verification)
- 23-REC-01 — streaming/decimated recording + memory preflight (W17.2)
- 23-EDGE-01 — edge selectors + immutable transforms pipeline (align W16 structural G)
- 23-LAW-01 — structured state-dependent plasticity / RBD law hook (gated on closed 23-HDP-01 registrable primitive; no Jomission-specific controller)
- 23-FIN-01 — long-run transactional finalization (W17.4) if evidence justifies
- 23-SYSID-01 — operating-point system-ID utility (W17.3) if evidenced
- 23-JDNA-01 — minimal `evolve()` runtime substrate only if semantics + tests complete before public claims
- 23-VERIFY-01 — full 0.4.23 RC + CI + seal + publish chain
- 23-STACK-01 — review and rewrite this file from sealed v0.4.23 before v0.4.24 work

**Ordered dependencies (remaining):** 23-DELAY-01 remainder (compaction only) decoupled from 23-SIMP-05 per receipt; delayed registrable HDP closed unblocking 23-SIMP-05 and full LAW-01 exercise; REP-03 gate retains `_SPARSE_DIRECT_N=5000` until bit-exact equivalence (`artifacts/fact_stack.md`). Jomission downstream migration remains **unverified** post-COMPAT-JOM-01. Pre-existing closure failures (HP-05 storage setup, population h_dim, disconnected-null diag) recorded in DELAY-01 receipt, not introduced here.

---

# v0.4.24

Entry: v0.4.23 shipped; W8 shows cumulative programme receipts from `8823520` baseline.

- 24-ENT-01 — confirm 0.4.24 entry criteria (after 23-STACK-01)
- 24-W16-6 — neurobiophysical geometry / bounded augmentation (W16.6); named observable, metric, ε
- 24-FIELD-01 — field/LFP proxy computational audit with declared error bounds
- 24-JAX-01 — JAX execution profile programme (measure before JIT changes)
- 24-EQUIV-01 — per-observable equivalence acceptance (exact or `d≤ε` pre-declared)
- 24-AUDIT-01 — independent adversarial core audit (implementer ≠ sole auditor)
- 24-EXIT-01 — verify programme exit criteria (roadmap §5.2)
- 24-VERIFY-01 — full 0.4.24 RC + CI + seal + publish chain
- 24-STACK-01 — seal v0.4.24; archive or clear this stack
