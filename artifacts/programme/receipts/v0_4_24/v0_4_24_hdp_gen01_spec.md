# 24-HDP-GEN-01 — general finite-state HDP expressivity (specification)

Status: **SPEC** — human-authorized goal; not started.

## Goal

JaxFNE must support any plasticity rule representable as a finite-dimensional,
JAX-compatible state update using the declared simulation state.

General form:

    H_{t+1}, Theta_{t+1}
        = P(H_t, X_t, B_t, Theta_t, events_t, parameters)

The user defines: required H coordinates; state shape/scope; state dynamics;
event/time dependence; plastic target Theta; bounds/saturation when required.

The engine must not require a new hard-coded simulation branch for each named
plasticity mechanism.

## Required expressivity

1. H must support arbitrary finite coordinate dimension d_H >= 0.
2. Rule state must support, where mathematically meaningful: global;
   population; neuron; edge/connection; pre/post-dependent state.
3. Rules must support: continuous/discrete time updates; spike/event-triggered
   updates; delayed events through B; combinations of event and continuous
   dynamics.
4. Theta targets must include at least: synaptic/effective efficacy; other
   declared mutable model parameters where ownership permits.
5. A rule must be able to causally alter executed efficacy:
   H -> P -> Theta_eff -> event/current.
6. Bounds must be general enough for: clipping; saturating
   parameterizations; bounded state/rule dynamics — without
   mechanism-specific engine branches.
7. Complete continuation must preserve every state required for future
   evolution: C_t = (X_t, H_t, W_t, B_t, K_t, rule_state_t), with no hidden
   state outside the continuation carrier.
8. Observability must expose enough state to establish:
   configured -> realized -> executed -> effective.
9. Disabled identity: no rule / disabled rule == ordinary static execution,
   under the strongest declared equivalence class.
10. JAX: jit-safe; tracer-safe; deterministic under declared RNG semantics;
    compatible with vectorization where state shapes permit; no Python
    per-event callback.

## Qualification (minimal basis spanning distinct failure classes)

A. continuous scalar state -> efficacy;
B. event-driven pre/post trace -> weight;
C. per-edge auxiliary state -> efficacy;
D. delayed event -> state -> efficacy;
E. multidimensional H rule;
F. bounded/saturating efficacy rule.

At least one probe must have analytically predictable trajectories.
Current coverage: A (synthetic_presyn_gain), B/C (eligibility_trace_gain),
D (delayed probes) exist; E (vector-H rules) and F (saturating
parameterizations) are the known gaps, as are global/population rule
scopes and non-weight Theta targets.

Named STDP/STP implementations are NOT the goal. They may be used as
probes only if they expose a missing dimension of the generic interface.

## Acceptance

Passes only when an independent critic cannot produce a reasonable
finite-state plasticity rule, within JaxFNE's declared mathematical scope,
that requires modifying the simulation engine rather than defining state +
rule + target through the generic HDP surface. Classify every attempted
counterexample: EXPRESSIBLE / OUTSIDE_DECLARED_MODEL / ENGINE_GAP. Any
ENGINE_GAP keeps this item open.

No Jomission/Hill-specific implementation is permitted.
