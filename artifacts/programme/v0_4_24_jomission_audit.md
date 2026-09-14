# JOMISSION FINAL CAPABILITY AUDIT — published JaxFNE v0.4.24

**Target:** tag `v0.4.24` (peel `7f89eff`), PyPI 0.4.24 hashes verified
equal to manifest. Engine tree `jaxfne/` identical tag↔dev HEAD at audit
time (only `tests/test_docs_version_alignment.py` differs, test-only), so
cited test evidence transfers to the immutable release. READ ONLY: no
implementation performed; transfer probe ran uncommitted in scratch
against the installed PyPI artifact.

## 1–14 table

| # | Item | Verdict | Evidence (API / impl / tests / docs / limit) |
|---|------|---------|-----------------------------------------------|
| 1 | HDP semantics | YES | `Delta(H,Theta)=P(X,H,B,Theta,events,params)` is `HDPRuleContext→HDPRuleUpdate` (`jaxfne/hdp_rule.py`); ownership: X in `DynamicState`/Model carry, B in delay ring, W in edges, control in H/aux/b, rule_params static; doctrine `docs/doctrine/rbs_rbd_hdp.md` |
| 2 | General rule state | YES | scalar/vector H, per-neuron/per-edge/scalar/multi aux, pre/post, events, decay, mixed — all analytic in `tests/test_hdp_gen01_expressivity.py` (20 tests) + prior suites |
| 3 | Generic registration | PARTIAL | Mechanism + `hdp_params["hdp_rule"]` config: YES (20 GEN tests, shipped rules). Limitation: `register_hdp_rule`/`HDPRuleDescriptor` importable only via unlisted `jaxfne.hdp_rule` (absent from `__all__` and `ADVANCED_NAMESPACE`, undocumented). Smallest missing: ADVANCED listing + docs section. Engine requirement met; surface-listing convenience missing. |
| 4 | Dynamic efficacy | YES | H→P→Theta→I proven; P1≠P2→Theta1≠Theta2→I1≠I2 (`test_p1_ne_p2…`); facilitation (gain sweeps), depression (BCM dw<0 measured, w 0.2→0.1958), bounds + saturation (logistic/Hill monotone, ceiling-exact) |
| 5 | STDP | generic YES / named YES | Generic: eligibility pre/post traces + decay + weight mod, analytic. Named: `jaxfne.plasticity` (STDPPlasticityConfig/State/kernel, ADVANCED root attrs, `docs/api/plasticity.md`) wired into `jaxfne.streaming.run_stdp_stream`, tested (`tests/test_streaming.py`, `test_v0341_kernels.py`). Limitation: named surface is standalone/streaming, not an edge-kernel-native rule (covered generically instead). |
| 6 | Short-term efficacy | YES | Logistic probe (weak→strong→strictly-bounded) + Hill rise-and-fall transfer probe on PyPI bytes, both analytic; Jomission Hill curve itself not implemented per instruction |
| 7 | HDP vs homeostasis | YES | All GEN probes run with homeostasis unset; registered path independent of `enable_homeostasis`/history controller; explicit rejection of unconsumed drive on the EI path (`test_phaseD_source_schema.py`) |
| 8 | Observability | PARTIAL | H/w/traces/rule name via public `last_hdp_diagnostics`; aux/b finals via continuation path + kernel diag. Limitation: bulk `diag_store` drops aux/aux_trace/b (`_model_simulate.py:462-480`); continuation diag keeps finals only. Smallest missing: forward aux/b through `_hdp_packed` tuple + store. Nonessential diagnostics internal; causal chain fully establishable through public
`last_hdp_diagnostics` + continuation state + kernel diagnostics. |
| 9 | Continuation | YES | X/H/W/B/aux/bias bit-exact chunked (kernel + Model, incl. in-flight delay, stochastic, vector-H, edge-aux); K ≡ theta_S population-controller coords carried where declared (registered control lives in H/aux/b instead). Bounds: EQUIV-01. |
| 10 | Disabled identity | YES | Null-HDP bit-exact vs baseline; registered zero-gain Model-level identity; null stochastic rule leaves membrane V bit-identical while rule state varies |
| 11 | JAX execution | PARTIAL | jit/tracing/deterministic replay/rule+continuation RNG/no-callback: YES (tests). Bounds: EQUIV-01 (registered d≤1e-4). Limitation: GPU execution and vmap-over-rules have no execution evidence (CPU-only CI); code is device-agnostic lax.scan. |
| 12 | Old regressions | YES | `tests/test_compat_jom01_regressions.py` green on published-equivalent code (tracer conversion + compact tau_ms round-trip) |
| 13 | Counterexample challenge | YES | Fresh declared-grammar round: 2 EXPRESSIBLE, 4 OUTSIDE (exogenous ports, opaque-step reads, theta_S reads, sign flips — no grammar redefinition), ENGINE_GAP = 0 |
| 14 | Transfer test | YES | Hill rise-and-fall short-term efficacy as one ordinary registered rule on installed PyPI 0.4.24: analytic match, bounded. JOMISSION_ENGINE_EXTENSION_REQUIRED = NO (scratch probe, uncommitted; phenotype expressly out of scope) |

## Verdict

JAXFNE_0_4_24_FULLY_SATISFIES_JOMISSION_ENGINE_REQUIREMENTS

Qualified explicitly: three items are PARTIAL on non-engine grounds
(item 3 surface listing, item 8 bulk-diagnostic plumbing, item 11
GPU/vmap execution evidence), each with its smallest exact missing
capability stated above — none is a missing engine capability, so none
is declared a remaining engine gap. If PARTIALs are read against the
verdict name rather than against engine capability, the Valtorian
translation is: zero engine gaps, three listed non-engine limitations.
Engine capability within the declared grammar is complete; the bounded
closure claim stands.

JOMISSION_MIGRATION_VERIFIED = NO (remains NO until Jomission runs its
migration/scientific-identity suite against the published release).
