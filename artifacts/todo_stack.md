# Remaining work — v0.4.24 (v0.4.23 sealed)

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.23 sealed** (tag `v0.4.23` @ `7e91da3`, peel verified; origin/main @ `7e91da3`; GitHub release with retained artifacts).
**v0.4.22** remains the published PyPI release until 23-PUB-01 executes.
Published predecessor: `v0.4.21` (immutable).

---

# v0.4.23 (sealed — record only, no executable work)

- 23-PUB-01 — OPEN (sole 23 leftover): upload retained `jaxfne-0.4.23` wheel/sdist bytes to PyPI (hashes in `artifacts/release/v0_4_23_release_receipt.json`; no rebuild); then advance published-version surfaces (`PUBLISHED_PYPI_VERSION` → `0.4.23`, `PREVIOUS` → `0.4.22`, install/colab published statements, drop RC wording); verify RTD stable build for the tag. Blocked on: PyPI credential/token in workspace.
- Closed: DELAY-01 (delayed HDP + compaction NO_CHANGE), SIMP-01 (kept), SIMP-02/03/04/05 (NO_CHANGE), STOCH-01 (NO_CHANGE), REC-01, EDGE-01, LAW-01 (improvements), FIN-01/SYSID-01/JDNA-01 (NOT_REQUIRED), HDP-02 (authorized re-baseline), VERIFY-01 (gates green; manifest + closure repaired). Receipts: `artifacts/programme/v0_4_23_*_receipt.md`.

---

# v0.4.24 (active)

Entry: v0.4.23 sealed; W8 shows cumulative programme receipts from `8823520` baseline.

- 24-ENT-01 — confirm 0.4.24 entry criteria
- 24-W16-6 — neurobiophysical geometry / bounded augmentation (W16.6); named observable, metric, ε
- 24-FIELD-01 — field/LFP proxy computational audit with declared error bounds
- 24-JAX-01 — JAX execution profile programme (measure before JIT changes)
- 24-EQUIV-01 — per-observable equivalence acceptance (exact or `d≤ε` pre-declared)
- 24-AUDIT-01 — independent adversarial core audit (implementer ≠ sole auditor)
- 24-PKG-01 — installed-package boundary audit (which LOC belongs in the runtime distribution; executable protocol/research code destination: tests/benchmarks/examples, never artifacts/ for aesthetics; alongside kernel-count and test-minimization reviews)
- 24-EXIT-01 — verify programme exit criteria (roadmap §5.2)
- 24-VERIFY-01 — full 0.4.24 RC + CI + seal + publish chain
- 24-STACK-01 — seal v0.4.24; archive or clear this stack

**Ordered dependencies:** 23-PUB-01 needs a PyPI credential only (no code); REP-03 gate retains `_SPARSE_DIRECT_N=5000` until bit-exact equivalence (`artifacts/fact_stack.md`); Jomission downstream migration remains **unverified** post-COMPAT-JOM-01; `diag=None`-for-identity stands per sealed HDP-01.
