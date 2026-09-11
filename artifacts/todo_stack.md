# Remaining work — v0.4.23 → v0.4.24

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.22 shipped** (tag `v0.4.22`, PyPI, GitHub release, RTD stable).
Published predecessor: `v0.4.21` (immutable).

---

# v0.4.23

Entry: v0.4.22 shipped; W3/W7 classifications hold zero `UNKNOWN`; W8 baseline current.

- **23-REP-03** — sparsity routing: honour `p_connect` or refuse; owns dense vs sparse-direct equivalence at `_SPARSE_DIRECT_N` (test at N=5000: **FAIL**, not bit-exact — retain threshold 5000 unless exact equivalence proven; no ε-equivalence; optimization not mandatory)
- 23-H-01 — H/RBS/RBD representation reduction (observable-specific; written per-coordinate argument)
- **23-HDP-AUDIT-01** — HDP public/runtime expressivity audit (downstream capability hypothesis is evidence to test, not established JaxFNE fact): inspect whether current HDP supports activity/events → H/B rule state → `P(X,H,B,Θ)` → effective parameter → realized event/current; test rule-specific state, neuron/synapse/edge scope, event/time updates, custom rule dispatch, dynamic effective synaptic efficacy, bounds/saturation, observation, checkpoint/continuation, disabled-rule static identity, determinism/RNG; STDP/STP-style probes where justified; classify exactly one of `EXISTING_HDP_SURFACE_SUFFICIENT` | `HDP_EXTENSION_REQUIRED` | `ENGINE_CAPABILITY_GAP` (only the latter two spawn implementation TODOs); gated after **23-COMPAT-JOM-01**; Jomission-specific Hill dynamics are not a JaxFNE requirement
- 23-HDP-01 — disabled HDP identity: bit-exact null plasticity (`Ẇ = 0`); no dense mutable `W` when HDP off (gated on 23-HDP-AUDIT-01; if extension required: generic HDP extension only — qualify first with simple synthetic rule with analytically predictable state-dependent efficacy; require configured → realized → current causality, checkpoint identity, disabled identity; downstream Jomission qualification only after generic qualification)
- 23-DELAY-01 — delay/history compaction (delay classes, ring buffers; bit-exact continuation)
- 23-STOCH-01 — continuation-owned stochastic inputs where required (per-path verification)
- 23-REC-01 — streaming/decimated recording + memory preflight (W17.2)
- 23-EDGE-01 — edge selectors + immutable transforms pipeline (align W16 structural G)
- 23-LAW-01 — structured state-dependent plasticity / RBD law hook (gated on 23-HDP-AUDIT-01; implementation only if `HDP_EXTENSION_REQUIRED` or `ENGINE_CAPABILITY_GAP`; no Jomission-specific controller)
- 23-FIN-01 — long-run transactional finalization (W17.4) if evidence justifies
- 23-SYSID-01 — operating-point system-ID utility (W17.3) if evidenced
- 23-JDNA-01 — minimal `evolve()` runtime substrate only if semantics + tests complete before public claims
- 23-VERIFY-01 — full 0.4.23 RC + CI + seal + publish chain
- 23-STACK-01 — review and rewrite this file from sealed v0.4.23 before v0.4.24 work

**Ordered dependency (HDP / downstream):** compatibility repair → downstream migration qualification (within COMPAT-JOM-01 acceptance) → HDP audit/extension → generic HDP qualification → downstream Jomission qualification.

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
