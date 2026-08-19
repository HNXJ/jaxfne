# Relative-Quantity Numerical Grammar

This page is the single authoritative statement of how jaxfne distinguishes
**base**, **relative**, and **effective** quantities, and why physical time is
treated differently from the relative-value convention. It is a *semantic
contract*: equations, code, tests, and documentation must describe the same
system under these rules.

## 1. Three quantity kinds

Every numerical quantity in a jaxfne model belongs to exactly one of these
kinds. The distinction is semantic, not a Python wrapper — inside JAX hot
kernels the values are plain arrays; the grammar governs *what the numbers
mean* and *how they combine*.

| Kind | Meaning | Units | Example |
|---|---|---|---|
| **BaseValue** | Physical/model baseline. Retains its actual units and semantics. Static/shared baseline that relative state modulates around. | physical (or declared model) units | `edges.weight` baseline `m0`; base drive `b`; base intrinsic `a_base`; `dt_ms`, `duration_ms` |
| **RelativeValue** | Dimensionless internal dynamic state or modulation. Zero (or another declared reference) represents the reference/base state where applicable. Its admissible domain is fixed by its geometry. | dimensionless | `H_i` (RBS coordinate), `r_p` relative gain modulation, `fraction` cell-type fractions |
| **EffectiveValue** | The quantity actually used in dynamics, produced from a base and a relative modulation through an explicit mapping. | inherits base units | `p_eff = C_p(p0, r_p)` |

Effective physical values must be produced through **explicit mappings**; a
relative value must never be silently treated as a physical quantity.

## 2. Admissible relative domains

A relative quantity is not blindly forced into `[-1, +1]`. Its canonical
relative domain must preserve its admissible geometry. The supported domain
shapes:

| Domain | Meaning | jaxfne example |
|---|---|---|
| `[-1, +1]` | finite signed | signed correlation / contrast proxies |
| `[0, +1]` | finite nonnegative | `fraction` cell-type proportions (sum to 1 over types) |
| `[-1, 0]` | finite nonpositive | — |
| `[-1, +inf)` | upper-unbounded signed | — |
| `(-inf, +1]` | lower-unbounded signed | — |
| `[0, +inf)` | nonnegative unbounded | weight magnitude `m \in [w_floor, w_ceiling]` (bounded in practice by explicit clamps) |
| `(-inf, 0]` | nonpositive unbounded | population-H deviation coordinate (signed, unnormalized controller state) |
| `(-inf, +inf)` | fully unbounded signed | — |

Rules:

- **Audit before assigning.** Domains are derived from the actual variables
  and their clamps/equations, never imposed retroactively. Do not
  retroactively reinterpret quantities whose established semantics are
  genuinely dimensional.
- **Zero = reference where applicable.** A relative value of zero represents
  the reference/base state in coordinates that declare `H* = 0` (deviation,
  trace, activity-history). Coordinates that declare `H* = 1` (multiplicative
  availability gains) use one as their reference instead. Each coordinate must
  declare its reference convention (see §4).
- **Domain invariants must not silently escape.** Relative-domain invariants
  must either be preserved by the dynamics or explicitly constrained by
  documented clamps (e.g. node H `[H_min, H_max]`).

## 3. Physical time exception

**Time remains physical and dimensional.** It is never normalized into the
relative-value convention.

- `dt_ms`, `duration_ms`, delays (`delay_steps`, delay_ms), time constants
  (`tau_0_ms`, `tau_syn_ms`, `tau_r_ms`, `tau_H_s`, `tau_theta_s`),
  timestamps, integration windows, and adaptation/growth timescales all carry
  physical time units (milliseconds, or seconds where the controller law
  declares `s`).
- The causal invariant is `t[n+1] > t[n]` — the simulation clock is
  physical, monotonic, and forward-causal. Constant `dt` is a convenience, not
  a requirement.
- Dimensionless state dynamics may use dimensional timescales:
  `tau_H * dH/dt = F(H, ...)` where `H` and `F` are dimensionless and `tau_H`
  carries time units.
- Adaptive timescales may themselves carry relative modulation **only when
  explicitly modeled**; the simulation clock itself is never relative.

## 4. H / RBS coordinates

`H` is a finite-dimensional **relative** hidden/biophysical state. A
coordinate exists only if it has all four attributes:

```text
h_k = (meaning, domain, R_k, Gamma_k)

    dh_k/dt = R_k(H, X, U, ...)     update rule
    Gamma_k(H) -> relative gains    mapping onto gain parameters
```

- Each coordinate declares: semantic meaning; admissible relative domain;
  update rule `R_k`; and the mapping `Gamma_k` from the coordinate to one or
  more relative gain parameters.
- Static/shared **base** parameters remain the physical/model baseline;
  neuron individuality and adaptation are expressed as dynamic **relative
  gains** around those bases.
- Do **not** create a generic "excitability" coordinate when excitability is
  better expressed as an emergent phenotype of several parameter gains.
- `H` is **not** intrinsically synonymous with homeostasis. A particular
  realization may contain homeostatic/restoring terms, but that is a
  specialization, not the definition.

The concrete per-coordinate table for the shipped HDP kernels lives in
[`docs/guides/hdp.md`](../guides/hdp.md) and
[`docs/doctrine/rbs_rbd_hdp_inventory.md`](rbs_rbd_hdp_inventory.md).

## 5. Effective-parameter mappings

Where a base parameter `p0` has a relative modulation `r_p`, the effective
parameter follows an explicit documented mapping:

```text
p_eff = C_p(p0, r_p)
```

The mapping form depends on the parameter's semantics — additive,
multiplicative, bounded, signed, etc. **Do not impose one mapping universally
where it is mathematically inappropriate.**

The concrete HDP mappings:

| Parameter | Base `p0` | Relative `r_p` | `p_eff = C_p(p0, r_p)` |
|---|---|---|---|
| Synaptic weight magnitude (node HDP) | `m0 = \|edges.weight\|` | H-derived basis `basis(H_post − H_pre)` | weight is an **integrated ODE state**: `dm/dt = ±K_HDP·basis·m + K_w_ctrl·(m0 − m)`; `K_HDP=0 ∧ K_w_ctrl=0 ⇒ w = w0` (exact null recovery) |
| Edge weight (population restoring) | sign of `w_baseline` (base magnitude not carried) | `θ_0 ∈ [0.1, 5.0]` | `w_eff = sign(w_baseline)·θ_0` — **magnitude-replacing**, not multiplicative on base magnitude; θ=1 gives unit magnitude, so the base magnitude is not recovered. Documented limitation of this controller. |
| Intrinsic `a` (population restoring) | `a_base` | `θ_1 ∈ [0.25, 4.0]` | `a_eff = a_base·θ_1` — multiplicative; θ=1 ⇒ `a_eff = a_base` (exact recovery) |

## 6. Language and claim gates

Documentation and publication prose must not silently destroy these
semantics. The package's docs-language gate distinguishes at least:

- **base** — physical/model baseline (retains units);
- **relative** — dimensionless dynamic state/modulation;
- **normalized** — a dimensionless rescaling (not automatically synonymous
  with "relative");
- **effective** — the mapped output `C_p(p0, r_p)` (does **not** imply
  empirically calibrated);
- **calibrated** — an explicit calibration transformation has been applied;
- **physical** — a measured/absolute physical quantity.

Semantic escalation is a defect: a relative/proxy output must not become an
unqualified physical measurement; a relative parameter must not be described
as though its numerical value carries the base unit; "effective" must not
automatically imply calibration. See
[`scripts/audit_public_docs_language.py`](https://github.com/HNXJ/jaxfne/blob/main/scripts/audit_public_docs_language.py).
