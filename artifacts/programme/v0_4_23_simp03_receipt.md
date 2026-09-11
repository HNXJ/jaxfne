# 23-SIMP-03 receipt — NO_CHANGE

**Branch:** `dev` (on top of `e1e5452`)
**Verdict:** no new specialization. No code changed.

## Scan-output classification (main `lax.scan` kernels)

| Output | Shape | REQUIRED_FOR |
| --- | --- | --- |
| `V` (voltages) | T×N | requested recording (`Signals.V_m` always present) |
| `spikes` | T×N | requested recording + continuation (`prev_spikes`) |
| `sources` | T×N | requested recording; downstream field calc when `record_fields` |
| `H_trace` (HDP) | T×N[×d] | diagnostics (`last_hdp_diagnostics`) |
| `w_trace` (HDP, iff `record_weight_trace`) | T×E | diagnostics only; `None` when False |
| `aux_trace` (registrable) | T×A | diagnostics only |
| `edge_current_trace` (iff `record_edge_current`) | T×E | diagnostics only |
| `current_trace`, `u_trace` (iff bundled flag) | T×N | diagnostics only |
| `r_bar`/`I_H` (boundary path) | T×N | diagnostics of active controller |
| `S_L/S_H/dh`, `dH_*` (iff `record_*_components`) | T×N | diagnostics only |

Static specialization already exists and is non-combinatorial: each
optional stack is gated by a static bool selecting between two step
closures (baseline bundles its triple into one flag — 2 variants, not
8 — deliberately).

## Measured (T=500, N=200, E=1600, float32)
- Always-stacked: V/S/src 0.40MB each; H 0.40MB; w 3.20MB iff enabled.
- `record_weight_trace=False` → `w_trace is None`, V/S bit-exact, H exact,
  wall 0.72s → 0.32s (dominant-term mitigation already documented).
- Optional baseline triple off → dynamics bit-exact (`test_recording_invariance`
  covers HDP delayed; dispatch test covers `record_weight_trace=False`).
- Confirmed unrequested materialization: kernel stacks `sources` T×N even
  when `record_sources=False` (Model discards); batch stacks `H_trace`
  T×N[×B] then discards (`record_weight_trace=False` drops only w).

## Why NO_CHANGE
Removing the residual single-T×N stacks needs return-arity flags through
~6 kernels + 3 dispatch paths + continuation (and `sources` must stay
when `record_fields=True`), doubling compile variants per kernel for one
T×N saving against already-gated T×E dominants. Benefit < concepts.
Per-flag unbundling of the baseline triple = 8 variants: compile
explosion, explicitly out of scope.
