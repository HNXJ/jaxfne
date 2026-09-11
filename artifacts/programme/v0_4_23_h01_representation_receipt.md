# 23-H-01 receipt — CLOSED (audit increment)

**Branch:** `dev`  
**Method:** coordinate inventory + observable-specific knockout probes; no carrier refactor yet.

## Classification

$$
\boxed{\textbf{REDUCTION\_CANDIDATES\_IDENTIFIED}}
$$

Per-coordinate arguments are recorded in `artifacts/audit/h_representation_probe_results.json`.

## Knockout probes

| Coordinate | Path | Observable | Knockout | Result |
| --- | --- | --- | --- | --- |
| `dynamic.H` | baseline continuation | `V_m`, spikes | perturb H ∈ {1, 3} | **inert** |
| `dynamic.w` | baseline continuation | `V_m`, spikes | scale w × 1.75 | **inert** |
| `dynamic.H` | RBD kernel | `V_m` | `beta_h=0` vs `beta_h>0`, perturbed H₀ | **required** (ΔV > 0) |
| `dynamic.w` | HDP simulate | `V_m` | `enable_hdp=False` vs active | **required** |
| `dynamic.theta_S`, `aux` | cold-start carrier | — | unused shapes | `(0,)` default |

## Per-coordinate reduction status (summary)

| ID | Status | Argument |
| --- | --- | --- |
| `prev_spikes` | contract passthrough, dead-by-design on zero-delay baseline | documented in emitter kernel; kept for checkpoint/continuation parity |
| `H`, `w` on baseline carrier | carrier passthrough redundant | unified eight-field `DynamicState` for HDP/baseline `compile_step_fn`; observably inert on baseline-only runs |
| `H` on RBD | retain | couples to `I_drive` via `G_H(H; beta_H)` and to `F_H` |
| `w` on HDP | retain | plastic synaptic state drives recurrent current |
| `theta_S`, `aux` | retain with zero default | population/rule coordinates; shape `(0,)` when unused |
| `delay_state` | defer **23-DELAY-01** | required when delays positive; compaction not audited here |

## Tests

`tests/test_h01_representation_probes.py` (5 probes)

## Not in this increment

- Slim baseline-only continuation carrier (optional follow-up; needs contract gate)
- Memory measurement of carrier slots at scale
- Homeostasis rate-integrator coordinate (separate kernel; not in unified `DynamicState` today)
- STDP/STP representation (explicitly deferred)

## Spawned

None new — reduction implementation items remain under this ID only if a slim carrier is pursued; delay compaction under **23-DELAY-01**.
