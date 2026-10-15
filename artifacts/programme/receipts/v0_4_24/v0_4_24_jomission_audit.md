# JOMISSION FINAL CAPABILITY AUDIT — published JaxFNE v0.4.24

**Target:** tag `v0.4.24` (peel `7f89eff`), PyPI 0.4.24 hashes verified
equal to manifest. Engine tree `jaxfne/` identical tag↔dev HEAD at audit
time (only `tests/test_docs_version_alignment.py` differs, test-only), so
cited test evidence transfers to the immutable release. Phase 1 was READ
ONLY (transfer probe ran uncommitted in scratch against the installed
PyPI artifact). Phase 2 (post-audit closure, human-authorized): items #3
(ADVANCED listing + docs + tier test) and #8 (bulk aux/bias forwarding +
4 tests) repaired on dev with broad green; v0.4.24 release identity
untouched.

## 1–14 table

| # | Item | Verdict | Evidence (API / impl / tests / docs / limit) |
|---|------|---------|-----------------------------------------------|
| 1 | HDP semantics | YES | `Delta(H,Theta)=P(X,H,B,Theta,events,params)` is `HDPRuleContext→HDPRuleUpdate` (`jaxfne/hdp_rule.py`); ownership: X in `DynamicState`/Model carry, B in delay ring, W in edges, control in H/aux/b, rule_params static; doctrine `docs/doctrine/rbs_rbd_hdp.md` |
| 2 | General rule state | YES | scalar/vector H, per-neuron/per-edge/scalar/multi aux, pre/post, events, decay, mixed — all analytic in `tests/test_hdp_gen01_expressivity.py` (20 tests) + prior suites |
| 3 | Generic registration | YES (post-audit closure) | `register_hdp_rule` + `HDPRuleDescriptor/Update/Context` in ADVANCED tier (`public_surface.py`, `ADVANCED_NAMESPACE`, root attrs outside 190-name `__all__`), documented in `docs/guides/hdp.md`, tier-tested (`test_registrable_hdp_surface_is_advanced_not_public`); configured via public `hdp_params["hdp_rule"]` |
| 4 | Dynamic efficacy | YES | H→P→Theta→I proven; P1≠P2→Theta1≠Theta2→I1≠I2 (`test_p1_ne_p2…`); facilitation (gain sweeps), depression (BCM dw<0 measured, w 0.2→0.1958), bounds + saturation (logistic/Hill monotone, ceiling-exact) |
| 5 | STDP | generic YES / named YES | Generic: eligibility pre/post traces + decay + weight mod, analytic. Named: `jaxfne.plasticity` (STDPPlasticityConfig/State/kernel, ADVANCED root attrs, `docs/api/plasticity.md`) wired into `jaxfne.streaming.run_stdp_stream`, tested (`tests/test_streaming.py`, `test_v0341_kernels.py`). Limitation: named surface is standalone/streaming, not an edge-kernel-native rule (covered generically instead). |
| 6 | Short-term efficacy | YES | Logistic probe (weak→strong→strictly-bounded) + Hill rise-and-fall transfer probe on PyPI bytes, both analytic; Jomission Hill curve itself not implemented per instruction |
| 7 | HDP vs homeostasis | YES | All GEN probes run with homeostasis unset; registered path independent of `enable_homeostasis`/history controller; explicit rejection of unconsumed drive on the EI path (`test_phaseD_source_schema.py`) |
| 8 | Observability | YES (post-audit closure) | Bulk `last_hdp_diagnostics` forwards aux/aux_trace/bias finals+traces via `_hdp_packed` tuple + `diag_store` (dynamics unchanged); identity, record-off, and chunk-agreement tested; kernel + continuation paths already carried them |
| 9 | Continuation | YES | X/H/W/B/aux/bias bit-exact chunked (kernel + Model, incl. in-flight delay, stochastic, vector-H, edge-aux); K ≡ theta_S population-controller coords carried where declared (registered control lives in H/aux/b instead). Bounds: EQUIV-01. |
| 10 | Disabled identity | YES | Null-HDP bit-exact vs baseline; registered zero-gain Model-level identity; null stochastic rule leaves membrane V bit-identical while rule state varies |
| 11 | JAX execution | PARTIAL (evidence boundary, no code change per authorization) | jit/tracing/deterministic replay/rule+continuation RNG/no-callback: YES (tests). Bounds: EQUIV-01 (registered d≤1e-4). Recorded unverified boundary: GPU execution and vmap-over-rules have no execution evidence (CPU-only CI); code is device-agnostic lax.scan. No GPU claim made; no vectorization architecture added. |
| 12 | Old regressions | YES | `tests/test_compat_jom01_regressions.py` green on published-equivalent code (tracer conversion + compact tau_ms round-trip) |
| 13 | Counterexample challenge | YES | Fresh declared-grammar round: 2 EXPRESSIBLE, 4 OUTSIDE (exogenous ports, opaque-step reads, theta_S reads, sign flips — no grammar redefinition), ENGINE_GAP = 0 |
| 14 | Transfer test | YES | Hill rise-and-fall short-term efficacy as one ordinary registered rule on installed PyPI 0.4.24: analytic match, bounded. JOMISSION_ENGINE_EXTENSION_REQUIRED = NO (scratch probe, uncommitted; phenotype expressly out of scope) |

## Verdict

JAXFNE_0_4_24_FULLY_SATISFIES_JOMISSION_ENGINE_REQUIREMENTS

Qualified explicitly: thirteen YES, one PARTIAL on evidence-only grounds
(item 11 GPU/vmap execution evidence — recorded boundary, no code change
authorized or made). Zero NO, zero ENGINE_GAP. Engine capability within
the declared grammar is complete; the bounded closure claim stands.

JOMISSION_MIGRATION_VERIFIED = NO (remains NO until Jomission runs its
migration/scientific-identity suite against the published release).
