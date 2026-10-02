# Structural HDP: design note

Status: design only, 2026-09-29. Nothing here is implemented. Authority: the
rule-grammar fact in `artifacts/fact_stack.md` (structural HDP is a special
case over a declared candidate set). Build after the 0.5.5 seal, additive and
opt-in; existing outputs stay bit-identical.

## Definition

Structural HDP is HDP whose modified parameter is the **existence** of a
connection, over a finite candidate set declared before the run.

| Object | Meaning |
|---|---|
| Candidate set `C` | declared list of `(pre, post, receptor)` triples; finite, fixed at construct |
| Existence state `e_c` | one variable per candidate, `e_c ∈ {0,1}` (hard) or `[0,1]` (graded) |
| Weight `w_c` | the connection weight when present; may be fixed or itself plastic |
| Effective weight | `w_c · e_c` (graded) or `w_c` where `e_c = 1` (hard) |
| Rule | `ė_c = F_e(X, H, B, Θ, ...)`, same form as `Ẇ = F_W(H, ...)`: a declared state, law and target set |

Growth and pruning are laws on `e_c` (for example a slow drive toward a
target rate that raises `e_c` on under-active targets and lowers it on
over-active ones). A synapse that does not appear in `C` cannot be created:
that stays outside the grammar and is refused.

Ordinary HDP is the case where `e_c` is constant (the candidate set is the
edge list, `ė_c = 0`). So this generalizes, and the off arm of any étude is
this case.

## Why this shape

- Static shapes. The edge arrays keep their length; jit and `scan` are
  unchanged. Existence is state, not topology.
- Finite declared target set. The grammar's own requirement holds.
- Continuation. `e_c` belongs to the carrier; a continued run is bit-exact
  like any other declared state.
- Compact storage. Sign and receptor come from declared metadata (fact); `e_c`
  is per-candidate state and does not enter compaction rules.

## Semantics to settle before building

| Question | Options | Lean |
|---|---|---|
| Hard or graded existence | hard needs a threshold and a deterministic rule; graded is differentiable | graded first; hard as a thresholded readout |
| Stochastic growth | needs RNG keys | explicit keys per the seed fact; deterministic default |
| Delay of a new edge | delay ring `B` is per edge | candidate carries its delay; ring size fixed by `C` |
| Cost | state and update scale with `|C|`, not with realized edges | declare `|C|`, refuse above a memory bound |
| Sign and receptor | declared per projection | fixed per candidate; a rule cannot flip sign |
| Readout | which arrays report existence | expose `e_c` and realized edge count in `Signals` and the manifest |
| Bit-identity | existing configs | no candidate set declared means the current path, byte for byte |

## Refusals (construct time)

- A structural rule with no candidate set.
- A candidate outside the declared populations, or a duplicate triple.
- A rule targeting an edge not in `C`.
- Candidate count above the declared memory bound.
- Structural rule on a route that cannot carry the state (named per route).

## Tests it needs

1. `ė = 0` reproduces ordinary HDP and the no-HDP run, bit for bit.
2. Perturbation at 0, None and boundary of each structural parameter changes
   the realized existence or raises (consumption gate).
3. Continuation split at `t` equals the uninterrupted run, with `e_c` in the
   carrier.
4. Same seed gives the same existence trajectory; a different seed changes it
   where the rule is stochastic.
5. Sparse-direct equals dense with `e_c` present (REP-03).
6. Sign, receptor and delay identity are preserved through growth and pruning.

## Use

- Literature étude row 12 (structural plasticity plus synaptic scaling, eLife
  88376 as returned by search, unread) becomes a possible target: the
  structural half maps to `e_c`, the scaling half to `w_c`. Stays "hold" until
  P1 shows what the result depends on.
- Étude arms: off = `e_c` fixed; on = one declared structural rule chosen at
  preregistration.

## Not decided

Names, declaration API, and whether hard existence is worth the extra
semantics. None of them is designed here on purpose; the note fixes the
definition and the constraints so the API can follow from them.
