# 23-HDP-AUDIT-01 receipt — CLOSED

**Branch:** `dev`  
**Method:** test the current engine; no new HDP features implemented.

## Classification

$$
\boxed{\textbf{HDP\_EXTENSION\_REQUIRED}}
$$

Not `EXISTING_HDP_SURFACE_SUFFICIENT` (custom rules impossible; STDP/STP not on simulate path).  
Not `ENGINE_CAPABILITY_GAP` (partial causal chain, continuation, bounds, and diagnostics already exist).

## Causal chain (current engine)

| Stage | Status | Evidence |
| --- | --- | --- |
| event/activity | **present** | presyn spikes → `syn_state`; `I_syn` in H income term |
| → H/B | **present (fixed forms)** | node `dH/dt` ODE; population 2D restoring `H` |
| → P(X,H,B,Θ) | **closed list** | `hdp_rule` ∈ {signed_linear, signed_quadratic, hebbian_product}; population bypasses `hdp_rule` |
| → Θ_eff | **present** | plastic `w` (and population `theta_S` → effective edge/intrinsic params) |
| → current/event | **present** | `w` × presyn → synaptic drive → Izhikevich step |

## Probe matrix

| Probe | Result |
| --- | --- |
| State scope (neuron H, edge w) | pass — no per-edge H beyond `w` |
| Custom rule definition | **fail** — `ValueError: Unknown hdp_rule` |
| Update timing | per-step: H → dw → neuron |
| Effective synaptic gain | `w` multiplicative update + clip |
| Bounds/saturation | H_min/max, w_floor/ceiling, barrier |
| Observation | `last_hdp_diagnostics`, metadata |
| Checkpoint/continuation | node H/w yes; population H/Θ **no** |
| Disabled identity | H/w frozen at null gains; **V_m not baseline bit-exact** |
| JIT | deterministic replay |
| RNG/continuation | covered by `test_continuation_contract.py` |
| STDP/STP | `plasticity.py` dense-W STDP **off** simulate-HDP path |

Machine-readable: `artifacts/audit/hdp_audit_01_probe_results.json`  
Tests: `tests/test_hdp_audit01_expressivity_probes.py`

## Spawned work (implementation only if extension required)

- **23-HDP-01** — generic extension + disabled identity (bit-exact null plasticity where promised)
- **23-LAW-01** — structured law hook on smallest generic primitive (not STP/STDP as separate engines)

Target primitive: one registrable finite-state plasticity operator inside the existing HDP step + widened continuation carrier for auxiliary coordinates (e.g. `theta_S`, eligibility traces).

## Constraints preserved

- Jomission Hill dynamics: not a JaxFNE requirement
- Jomission migration post-COMPAT: **unverified** — downstream must rerun migration suite independently
- `fact_stack.md`: unchanged
