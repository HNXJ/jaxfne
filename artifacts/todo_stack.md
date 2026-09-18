# Remaining work

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.24 published** (tag `v0.4.24`, peel `7f89eff`; origin/main @ `7f89eff`; GitHub release with CI bytes; PyPI `0.4.24` via trusted publishing run `34839857303`; RTD stable build `34547734` @ `7f89eff`).
Published predecessors: `v0.4.23`, `v0.4.22` (immutable).

---

MODE = TFNE/2 ADOPTION

`tfne/2` (`SEALED LANGUAGE`) replaced project source 7. The language is
authority; `jaxfne.tfne` is a compiler against it and conforms partially.
Measured delta and method:
`artifacts/programme/tfne2_language_supersession_receipt.md`; current standing
of every clause in `docs/doctrine/tfne_algebra.md` under "Compiler
conformance".

Authorized priority order:

```
TFNE2-08 (done) -> TFNE-EXEC-01 (done) -> TFNE-IMPORT-01 (done)
  -> TFNE-PARAM-01 (done) -> TFNE2-03 (done) -> remaining conformance gaps
```

The three clauses whose non-conformance changed realized biology are closed:
TFNE2-01 (S14 exclusions), TFNE2-02 (S6/S7 instance addressing), TFNE2-08 (S10
ordered adjacency). Receipts `tfne2_conformance_0102_receipt.md` and
`tfne2_conformance_08_receipt.md`. Execution qualification is next; do not
resume the conformance list ahead of it.

## Next

- **TFNE2-04** — declared frontiers; the one that unblocks `CTX-01`. First of
  the remaining conformance gaps below.
- **TFNE-PARAM-02** — declared `delay` is refused, not carried.
  `E_PARAM_UNSUPPORTED` fails closed because no execution path consumes it:
  `Configuration.connections()` has no delay field, `compile_connection_rules`
  has no delay support, and `InterConnection` carries none. `EdgeList` does
  have `delay_steps`, so the gap is the compiler chain, not the kernel.
  Supporting it means a delay field on the connection-rule surface plus the
  configured/realized/executed identity that TFNE-PARAM-01 established for
  weight, probability and mechanism.
- **TFNE-PARAM-03** — `mechanism identity != mechanism kinetics`. A declared
  `AMPA` edge from a TFNE spec executes with `tau_ms = 0.1`, not AMPA's 2.0 ms.
  Identity (name, receptor index, E/I split) does transfer. **Do not repair
  opportunistically:** 0.1 -> 2.0 changes the trajectories of every existing
  TFNE-derived simulation.

  *Kinetics authority, determined (the prerequisite the repair needs):*

  | Candidate | Standing |
  | --- | --- |
  | `StaticParams.dT_ms` | **Owner** of per-connection synaptic tau. `neuronal_tensor.py:868` sets the mechanism's `tau_ms` from it. Despite the "dT" name it is a time constant, not a timestep. |
  | `standard_receptor_specs()` | **Canonical per-receptor values.** AMPA 2.0, GABA_A 5.0, NMDA 100.0, GABA_B 150.0. Declarative metadata, "no biological claim". |
  | `sign_only_tau_exc_ms` / `_inh_ms` | 2.0 / 5.0 for the compact `sign_from_receptor` storage mode. Agrees with the above. |
  | TFNE rule/model declaration | Cannot express a tau. Realized `mechanism_table` carries `tau_ms: None`, `declared_not_simulated`. S12 rule bodies (TFNE2-05) would be where it becomes expressible. |

  The authorities do **not** conflict: every hand-written caller copies the
  canonical value into `dT_ms` — `dT_ms=AMPA_TAU_MS` where `AMPA_TAU_MS = 2.0`
  in `scripts/hdp_1000_neuronal_tensor_column.py:74,101`, and `dT_ms=2.0` /
  `5.0` in `examples/08_neuronal_tensor_first.py` and
  `scripts/build_canonical_neuronal_tensor_configs.py`. The `0.1` default is a
  placeholder, not a competing definition.

  So the real gap is narrower than "kinetics is undefined": TFNE's
  `to_neuronal_tensor` never constructs `StaticParams`, so a TFNE spec inherits
  the placeholder while an equivalent hand-written tensor gets 2.0/5.0. The
  repair is to populate `dT_ms` from the declared mechanism's canonical spec.

  *What depends on the current 0.1:* no test asserts it. The TFNE tau test
  asserts agreement with the bridge and invariance across `dt`, not the value,
  so it survives the repair. `tests/test_synaptic_kernel_v011.py:57-60` asserts
  the mechanism tau table equals `standard_receptor_specs()`, which is evidence
  for the canonical values rather than against the change.

  **BLOCKED — the repair is not bounded. Do not attempt it as a one-line
  change.** The construction path is proven: `to_neuronal_tensor` builds
  `InterConnection(...)` and `AreaConnection(...)` (`jaxfne/tfne.py:1706,1724`)
  with no `static=` argument, so `StaticParams()` supplies `dT_ms = 0.1`.
  Setting it is mechanically trivial. Deciding *what to set it to* is not,
  because the name space does not resolve:

  - TFNE mechanism names are arbitrary strings. `_relation_mechanism`
    (`jaxfne/tfne.py:1859`) returns whatever the rule declared, defaulting to
    `DIRECT_MECHANISM = "tfne_direct"` or `tfne_<rule>`. There is no permitted
    vocabulary.
  - Nothing resolves a name to a canonical receptor. In
    `compile_connection_rules`, `receptor_index` is just the declaration order
    and `tau_ms` comes from the declared params; `standard_receptor_specs()` is
    never consulted on this path.
  - The canonical table is keyed `AMPA`, `GABA_A`, `NMDA`, `GABA_B`. TFNE specs
    in this repository use `AMPA` (27 occurrences) and **`GABA`** (5) — and
    `GABA` is not a key.

  So an exact-match lookup would set AMPA to 2.0 while leaving `GABA` and
  `tfne_direct` at 0.1: excitatory kinetics 20x slower, inhibitory unchanged.
  That asymmetry changes E/I balance in every affected model and is worse than
  the uniform placeholder. Treating `GABA` as `GABA_A` is an aliasing decision
  no source authorizes.

  The prerequisite is the mechanism vocabulary itself — S25's
  `E_MECHANISM_UNRESOLVED` / `E_MECHANISM_NOT_PERMITTED` (TFNE2-07) and S12
  rule bodies (TFNE2-05), both open. **Surface before choosing:** (a) define
  the permitted mechanism vocabulary under TFNE2-07 and resolve kinetics from
  it, (b) let S12 rule bodies declare tau explicitly and refuse an
  unresolvable mechanism, or (c) leave the placeholder until one of those
  lands. Whichever is chosen, all mechanisms must move together.
- **TFNE-PARAM-04** — declared geometry is realized but not executed. `G =
  [z0 = ...; z1 = ...]` is recorded faithfully in `s["geometry"]`, but at equal
  seed the executed positions are bit-identical whatever range is declared:
  `(0,1)`, `(10,20)` and `(-5,-4)` all sample the same z in the unit interval.
  TFNE's `to_neuronal_tensor` builds `Geometry3D(value_tag="relative")`, and a
  relative normalization would still let the declared *extent* matter, so
  identical output means the declaration is inert rather than rescaled. This is
  the same class as TFNE-PARAM-01 — configured and realized, not executed —
  and it matters because geometry is what field observables are computed
  against, so an LFP-style claim would rest on coordinates the specification
  did not choose. Pinned as the current state by
  `test_declared_geometry_does_not_reach_the_executed_positions`; the test
  fails when repaired and must then be inverted. Before repairing, determine
  whether absolute or relative coordinates are the intended semantics, the
  same authority question as TFNE-PARAM-03.

  Stays separate until its coordinate semantics are established. The supported
  claim today is exactly `declared G != executed G`. Whether the eventual
  repair preserves absolute coordinates, normalized coordinates, or an explicit
  declared transformation requires its own authority and must not be settled as
  a side effect of TFNE-PARAM-03.

### Parameter ownership, as measured (TFNE-PARAM-01)

| Parameter | Configured | Realized | Executed |
| --- | --- | --- | --- |
| `weight` | yes | yes | yes — kernel-resolved, and it moves the trajectory |
| `probability` | yes | yes | yes — thins the executed edge set |
| `mechanism` identity | yes | yes | yes — receptor index and kind |
| `direction` | yes | yes | yes — as realized pre/post |
| `plasticity` | yes | provenance only | no — declared rule identity, not an edge parameter |
| `delay` | refused | refused | refused (`E_PARAM_UNSUPPORTED`) |
| mechanism kinetics (`tau_ms`) | not expressible | `None`, `declared_not_simulated` | bridge `dT_ms`, not the receptor's own tau (TFNE-PARAM-03) |
| fixed mechanism params (`g_mech`, reversal potentials) | not expressible | — | bridge defaults; reversal potentials metadata only |
| geometry (`G`) | yes | `s["geometry"]` | **no** — declared range is inert (TFNE-PARAM-04) |

Receipt: `artifacts/programme/tfne_param01_receipt.md`. The equivalence is a
dev-gate module, not prose: `tests/test_tfne_parameter_transfer.py`.

Extend that coverage **by semantic class**, not by accumulating fields. Each
class keeps its own independent check; equality of counts, names, shapes or
default-valued outputs is too weak to establish semantic preservation, and
perturbations must be asymmetric and non-neutral. Current standing:

| Class | Covered |
| --- | --- |
| topology | yes — multiset of `(pre, post)` |
| mechanism identity | yes |
| mechanism kinetics | yes — dt-invariance and bridge agreement (value itself is TFNE-PARAM-03) |
| weight | yes — kernel resolver plus trajectory change |
| delay | yes — as a refusal (TFNE-PARAM-02) |
| probability / realization | yes — thinning at 0.5, not 1.0 |
| mutable / plastic rule identity | yes — provenance, and `h0["w"]` separate from `s["edge_weight"]` |
| geometry-dependent parameters | yes — as a pinned divergence. The declared range does not reach executed positions (TFNE-PARAM-04). |

## Remaining measured conformance gaps

Each item conforms one clause. Standing of every clause is in the doctrine
page; none of the items below changes realized biology — the compiler rejects
what it cannot express rather than realizing a different nervous system.

- **TFNE2-03** — S20 canonical ordering. **Typed natural ordering is done**
  (`L1<L2<L10`, `SEG.2<SEG.10`, declaration-independent), applied at the single
  point where `ExplicitModel.order` is built so realization and
  `to_neuronal_tensor` cannot drift apart on it. Receipt:
  `artifacts/programme/tfne2_conformance_03_receipt.md`.

  **The override is done too.** The language gap was ruled on by Hamm —
  dedicated syntax, and ordinary enumeration never implies order — and project
  source 7 now carries S20.1 defining `order[A] := [m1, ..., mk]`, with the
  amendment recorded in that source's status header. The compiler implements
  it and refuses any declaration it cannot honour exactly
  (`E_ORDER_INCOMPLETE`, `E_ORDER_DUPLICATE_MEMBER`, `E_ORDER_MEMBER_UNKNOWN`,
  `E_ORDER_NOT_IMMEDIATE`, `E_ORDER_SCOPE_UNKNOWN`, `E_ORDER_SCOPE_AMBIGUOUS`,
  `E_ORDER_DUPLICATE`).

  *One known exception remains:* cell types keep their `C = {...}` enumeration
  order. S20.1 states that implicit source order never carries scientific
  semantics and that this is temporary compatibility behaviour, so bringing
  cell types under the same rule is outstanding work, not settled behaviour.
- **TFNE2-04** — S9 frontiers beyond the derived ordered defaults that S10
  already uses: declared `in := [...]` / `out := [...]` bodies, the
  X-composite union default, `E_FRONTIER_UNRESOLVED`. Carries the rest of S8
  — only a declared frontier can make `{A O B} O C` differ in edge set from
  `A O B O C`.
- **TFNE2-05** — S12 `$L` / `$R` rule metavariables and rule bodies that
  specify topology, mechanism, parameters, geometry and delay independently.
  Also gives `X[k]` a way to select endpoints, which it currently lacks.
- **TFNE2-06** — S14/S25 statement atomicity: `;`-separated statements inside a
  composite, each atomic, a statement with an invalid resolved projection
  contributing nothing. Carries S6 prefix rule application `O[k](SEG^n)`.
- **TFNE2-07** — S25 semantic failure vocabulary (`E_ADDRESS_UNKNOWN`,
  `E_FRONTIER_UNRESOLVED`, `E_MECHANISM_UNRESOLVED`,
  `E_MECHANISM_NOT_PERMITTED`, `E_PROJECTION_REDUNDANT`,
  `E_EXCLUSION_UNKNOWN`, missing realization policy, ambiguous expansion,
  invalid proportion) as typed classes rather than prose `TFNEError`. Carries
  S13 projection-identity redundancy and S11 ungrouped-`X` rejection.

  **Blocks TFNE-PARAM-03.** `E_MECHANISM_UNRESOLVED` /
  `E_MECHANISM_NOT_PERMITTED` are not merely error classes here: they are the
  mechanism vocabulary that kinetics resolution presupposes. The dependency is
  `mechanism vocabulary -> mechanism resolution -> kinetics -> PARAM-03`, so
  PARAM-03 stays blocked behind this item rather than acquiring an isolated
  patch.

  *Acceptance for the mechanism-resolution work* (authorized, stronger than
  merely resolving `GABA`). For every executable connection mechanism, the
  chain

  ```text
  declared mechanism -> resolved mechanism identity
                     -> mechanism parameters / kinetics
                     -> realized connection
                     -> kernel-consumed mechanism
  ```

  must be deterministic and inspectable. Unknown or unpermitted mechanism names
  fail closed. Aliases such as `GABA -> GABA_A` require an explicit
  authoritative declaration and must never arise from heuristic name matching.

  Once mechanism resolution exists, return to PARAM-03 and test: `AMPA`;
  `GABA_A`; `NMDA`; `GABA_B` where supported; a custom permitted mechanism;
  rejection of an unresolved mechanism; several mechanisms with asymmetric tau
  values; `dt` invariance; and a TFNE construction against an equivalent
  hand-written JaxFNE construction.

## Definition layer

- **CTX-01** — `tfne/2` S27 names `CTX` as the next definition candidate:
  populations and `P_{l,c}` including absent populations, `N`-scaling and
  allocation, `in`/`out` interfaces, local connectivity, biological identity
  vs dynamical realization, geometry required for observables. S28 keeps
  spectrolaminar qualification (SL0/SL1/SL2) outside the algebra, and forbids
  phenomenological visualization proxies as the objective defining `CTX`.
  Depends on TFNE2-04 for declared `in`/`out`.

---

## Long-term goal — agent-native JaxFNE

The durable goal statement is **adopted** and lives in project source 6,
`artifacts/project_sources/6_other_important_notes.md`, section 6 "Long-term
plan", as the cross-cutting objective "Agent-native JaxFNE". That source is
the authority; this section keeps only the executable work and must not
restate the rationale. Do not start it ahead of the TFNE conformance and identity
work above — step 1 of its own sequence *is* that work.

Shape of the thing being built:

```
scientific intent -> TFNE -> verified JaxFNE operations -> simulation -> evidence
q -> skill -> A_TFNE -> NF(A) -> JaxFNE
```

The load-bearing constraint is that natural language does not map to arbitrary
generated simulator code. The agent constructs, parameterizes, composes,
executes, inspects and tests existing typed scientific objects. The failure
class it is built against is
`researcher's intended model != agent's executable interpretation`, which is
exactly TFNE-PARAM-01 and TFNE-PARAM-03.

### Sequence

1. Finish TFNE configured/realized/executed identity, including TFNE-PARAM-01.
   *(done for weight, probability, mechanism identity, direction; PARAM-02 and
   PARAM-03 open)*
2. Establish canonical TFNE -> JaxFNE compilation and `I`.
3. Inventory existing JaxFNE capabilities as agent-safe operations.
4. Per major capability, align equation, code, docs, tests and inspection into
   one auditable capability record. `delay` is the worked example: TFNE says
   how it is specified, code how it is executed, docs the equation, units and
   limitations, the skill when and how to configure and test it, tests the
   zero-delay limit, arrival timing, continuation and composition, and
   inspection the configured, realized and executed value.
5. Build a small agent-facing tool surface of scientific operations, not thin
   wrappers over every Python function: `realize(TFNE)`, `inspect(model)`,
   `simulate(model, T, dt)`, `compare(configured, realized, executed)`,
   `observe(source/field/probe)`, `verify(property)`. An agent should not need
   the 266 classified root symbols to build a standard experiment.
6. Build composable skills around that surface, not one large agent skill:
   `jaxfne-model`, `jaxfne-network`, `jaxfne-state`, `jaxfne-plasticity`,
   `jaxfne-fields`, `jaxfne-simulate`, `jaxfne-verify`, `jaxfne-inspect`. A
   general skill may route among them; the procedures stay small and testable.
7. Canonical end-to-end modeling tasks with frozen expected properties.
8. Adversarial semantic-substitution tests — stale docs, renamed APIs,
   parameter substitution, wrong units or types.
9. Benchmark skill+tools against direct repository/API use on a frozen task
   set: model fidelity (intended vs configured), realization fidelity,
   execution fidelity, numerical validity (`dt`, continuation, convergence),
   scientific validity of the claim/evidence relation, unsupported rejection,
   reproducibility under declared stochastic semantics, efficiency,
   inspectability, and repair under injected defects. Primary metric is
   scientific-model fidelity, not question-answer accuracy.
10. Only then consider external MCP or other transport. MCP is an
    implementation option, not the scientific architecture.

### Design constraints to honour when building the above

- Verification is claim-conditioned, `V = V(claim)`: the verification plan is
  generated from the scientific claim, not from the API calls made. "Runs"
  needs execution; "weight is 0.5" needs executed parameter inspection;
  "oscillates at 40 Hz" needs signal analysis plus numerical adequacy;
  "PING-like" needs E/I timing, participation, loop dependence and
  perturbation; "traveling wave" needs spatial phase progression, not delay
  alone; "plasticity caused effect" needs an intervention on the mutable rule.
- Each boundary in
  `intent -> specification -> realization -> execution -> observation -> claim`
  needs its own test; they are different questions.
- `TFNE <-> code <-> docs <-> skills <-> tests` is one semantic source with
  several projections — not literal duplication, and not five descriptions
  that can drift.
- Unsupported capabilities fail closed rather than inviting improvisation.
- Reference for design lessons only, not a template: Paper2Agent (validated
  tools checked against reference outputs before exposure, separated
  tools/resources/workflow prompts, exclusion of failing tools, explicit
  out-of-scope rejection, injected repository-drift defects, benchmarking
  against direct repository access). Numerical reproduction alone is
  insufficient for mechanistic modeling, and JaxFNE additionally owns its
  specification language.
