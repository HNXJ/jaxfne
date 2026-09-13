# Remaining work — v0.4.24 (v0.4.23 published)

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.23 published** (tag `v0.4.23` @ `7e91da3`, peel verified; origin/main @ `7e91da3`; GitHub release with CI bytes; PyPI `0.4.23` via trusted publishing run `34703754053`; RTD stable build `34526536` @ `7e91da3`).
**v0.4.22** is now the published predecessor (tag, PyPI, GitHub release, RTD history).
Published predecessor: `v0.4.21` (immutable).

---

# v0.4.23 (published — record only, no work)

- Closed: DELAY-01 (delayed HDP + compaction NO_CHANGE), SIMP-01 (kept), SIMP-02/03/04/05 (NO_CHANGE), STOCH-01 (NO_CHANGE), REC-01, EDGE-01, LAW-01 (improvements), FIN-01/SYSID-01/JDNA-01 (NOT_REQUIRED), HDP-02 (authorized re-baseline), VERIFY-01 (gates green; manifest + closure repaired), PUB-01 (Antigravity trusted publishing; hash discrepancy reconciled to CI bytes). Receipts: `artifacts/programme/v0_4_23_*_receipt.md`.

---

# v0.4.24 (active)

Entry: v0.4.23 shipped; W8 shows cumulative programme receipts from `8823520` baseline.

- AUDIT-01 — OPEN: adversarial self-review battery recorded per `artifacts/programme/v0_4_24_audit01_selfreview_receipt.md` (no defects; identity qualified as routing-not-kernel property); still requires implementer ≠ sole auditor (external party)
- PKG-01 — IN PROGRESS: installed-package boundary audit (which LOC belongs in the runtime distribution; executable protocol/research code destination: tests/benchmarks/examples, never artifacts/ for aesthetics; alongside kernel-count and test-minimization reviews)
- API-01 — public-surface minimization under API constraints (describe implemented behavior only)
- PRO-01 — protocol/research code placement + minimization (destination per PKG-01)
- HDP-GEN-01 — general finite-state HDP expressivity (spec: `artifacts/programme/v0_4_24_hdp_gen01_spec.md`); known gaps vs current surface: vector-H rules (E), saturating parameterizations (F), global/population scopes, non-weight Theta targets; acceptance is critic-driven (EXPRESSIBLE/OUTSIDE/ENGINE_GAP)
- EXIT-01 — verify programme exit criteria (roadmap §5.2)
- VERIFY-01 — full 0.4.24 RC + CI + seal + publish chain
- STACK-01 — seal v0.4.24; archive or clear this stack

## Minimization charter (applies to MIN-01/DOC-01/CODEMIN-01/TESTMIN-01/PKG-01/API-01/PRO-01)

- Vocabulary: minimize unnecessary jargon and synonyms; preserve technical terms carrying distinct meaning. Lint prevents demonstrated stale vocabulary from returning — no blind global bans.
- Preserved terms (exact meanings required): RBS, RBD, HDP, continuation, proxy, configuration, JDNA, receipt.
- Target: minimum complexity subject to complete required semantics.
- 100/100: no known material defect under the required deterministic/adversarial gates; no known strictly superior simplification under identical required semantics, API constraints, numerical behavior, and scientific meaning.

**Ordered dependencies:** REP-03 gate retains `_SPARSE_DIRECT_N=5000` until bit-exact equivalence (`artifacts/fact_stack.md`); Jomission downstream migration remains **unverified** post-COMPAT-JOM-01; `diag=None`-for-identity stands per sealed HDP-01.
