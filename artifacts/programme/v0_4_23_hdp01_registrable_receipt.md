# 23-HDP-01 receipt — CLOSED (first increment)

**Branch:** `dev`  
**Classification input:** `HDP_EXTENSION_REQUIRED` (audit 23-HDP-AUDIT-01)

## Delivered

### Registrable primitive

- `jaxfne/hdp_rule.py`: `HDPRuleDescriptor`, `HDPRuleContext`, `HDPRuleUpdate`, `register_hdp_rule`, `get_hdp_rule`, `hdp_params_are_identity`
- Operator contract: `(H, X, B, Θ) → (ΔH, ΔΘ)` via per-rule `step_fn`; rules declare `h_coords`, `theta_targets`, `aux_coords`, bounds
- Qualification rule: `synthetic_presyn_gain` (event-coupled H + edge-weight drive)

### Kernel + dispatch

- `jaxfne/_hdp_registrable_kernel.py`: `simulate_edge_recurrent_izhikevich_hdp_registered` (zero-delay path)
- `Model.simulate` routes `hdp_rule=<registered>` through registrable kernel; builtin rules unchanged on legacy kernel

### Disabled identity

- Documented null builtin `hdp_params` route to **baseline** kernel → **bit-exact** `V_m` vs `enable_hdp=False`
- `enable_hdp=True` + `receptor_exponential` still rejected before identity bypass

### Continuation carrier

- `DynamicState` extended: `theta_S`, `aux` (shape `(0,)` when unused)
- Population cold-start + `ContinuationState` carry `theta_S`; legacy population continuation rejection removed
- Registered-rule chunk continuation verified

## Qualification evidence

| Probe | Result |
| --- | --- |
| `P1 ≠ P2 ⇒ Θ(t) ≠` | PASS (`k_w=0.05` vs `0.20`) |
| `⇒ I(t) ≠` | PASS (sources differ) |
| Analytic H, w discrete rule | PASS vs manual reference |
| JIT deterministic replay | PASS (model dispatch) |
| Chunk continuation | PASS |
| Disabled identity bit-exact | PASS |

Tests: `tests/test_hdp01_registrable_qualification.py` (6), updated `test_hdp_dispatch`, `test_hdp_audit01`, `test_continuation_contract`

## Not in this increment

- STDP/STP/homeostasis as registered rules (deferred; expressivity via same primitive)
- Registrable kernel with finite edge delays
- Full population-controller continuation simulate without `m_ei_edge_mask` fixtures (cold-start carrier only on suite2)
- Jomission migration (still unverified downstream)

## Spawned

None new — **23-LAW-01** remains gated on exercising the primitive for structured laws.
