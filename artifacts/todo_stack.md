# Remaining work

> **Concurrent editors (2026-09-23).** The human owner and a Claude Code
> session also edit this file and other files under `artifacts/`. We change
> them by small, incremental adjustments: amend, reorder or annotate existing
> items. We make no wholesale rewrites and drop no item without a note.
> In progress: 0.5.x re-scoped to add an Atlas track (AT-01…AT-10, 1N → 2N →
> population → 2 areas → 20 areas). Release map and 0.5.1–0.5.2 stacks are in the
> 0.5.x programme section; 0.5.3–0.5.5 stacks follow one at a time. Before editing, re-read from
> disk. A diff you did not make is a concurrent edit (H12): keep it and do not
> revert it.

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
- **TFNE-PARAM-03** — `mechanism identity != mechanism kinetics`. **Done**
  via TFNE2-07's vocabulary, choosing (a)+(b) together: the permitted
  vocabulary is defined (`resolve_mechanism`: CANONICAL exact,
  CUSTOM_DEFINED via rule `tau_ms`, no aliases) and kinetics resolve from
  it at the tensor bridge into `StaticParams(dT_ms)` (+ reversal
  metadata); unresolvable refused at execution. All mechanisms moved
  together (AMPA 2.0, GABA_A 5.0, NMDA 100.0, GABA_B 150.0); `GABA` is
  UNRESOLVED (ambiguous A/B, no alias) with the two executing fixtures
  moved to GABA_A; absent mechanism stays `tfne_direct` 0.1 placeholder
  (unchanged trajectories). Receipt:
  `artifacts/programme/tfne2_conformance_07_mech_receipt.md`. The analysis
  below is retained as the recorded rationale for these choices.

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

  *Traced (inspect-only):* identity survives TFNE `G` → `s["geometry"]` →
  JDNA completion (declared domain sampled under `K_D` with origins) →
  `to_neuronal_tensor` `Geometry3D`, and is first lost in
  `neuronal_tensor_to_configuration`, which drops `Layer.geometry` for the
  Configuration default column (stated in its own docstring). The live
  `construct(tensor, ...)` path honors declared domains via
  `_construct_neuronal_tensor_impl`. Open before any repair: field/cable
  consumers assume unit-relative depth (contacts `linspace(0,1)`), the
  transfer seed ownership (`K_D` needs an explicit parameter), and the
  pin-test inversion above.

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
- **TFNE-JDNA boundary (done)** — TFNE constrains (possibly underdetermined),
  JDNA completes under `D + K_D`, Model realizes, Simulation runs. Doctrine
  `docs/doctrine/tfne_jdna_boundary.md`; source 7 S6.1 (`A^{nX}`/`A^{nO}`,
  compiler conforms) and S29.1 (underdetermination, defaults table, value
  origins, geometry owned by JDNA). JDNA completion layer
  (`jaxfne/jdna/completion.py`: `resolve`, `realize_geometry`,
  `complete_tfne`; `develop` provenance gains additive `value_origins`).
  Receipts `tfne_replication_relation_receipt.md`,
  `tfne_jdna_completion_receipt.md`. PARAM-04 stays open under JDNA
  ownership (bridge pin test intact, never patched); PARAM-03 refusal
  recorded in code as `mechanism_tau_ms: required`.
- **TFNE2-04** — declared frontiers. **Done.** `in[A] := [...]` /
  `out[A] := [...]` name immediate-member subsets (replica-aware); a declared
  side overrides the derived default for that side only; anything
  unhonourable is `E_FRONTIER_UNRESOLVED`; bare scopes resolve only when
  unambiguous; frontier metadata is canonical (digest-sensitive). Only a
  declared frontier can make `{A O B} O C` differ in edge set from
  `A O B O C`. Receipt:
  `artifacts/programme/tfne2_conformance_04_receipt.md`. CTX-01's `in`/`out`
  dependency is unblocked. Carried, not added: X[k]-rule frontier override
  (bodies now exist via TFNE2-05; declaring a composite interface from a
  rule still needs a language decision).
- **TFNE2-05** — S12 `$L` / `$R` rule bodies. **Done.** Bodies are
  projection statements over `$L`/`$R` (whole operands), `.out`/`.in`
  (resolved interfaces), member refs, and `{a, b}` collections, with
  per-statement direction and `[mech=...]` override; names resolve within
  their own side, absences and out-of-operand paths are
  `E_ADDRESS_UNKNOWN`; flat rules keep exact legacy behavior. Also gives
  `X[k]` endpoint selection, which it previously lacked. Receipt:
  `artifacts/programme/tfne2_conformance_05_receipt.md`. Carried, not
  added: `X[k]`-rule frontier override (needs a language decision on
  interface-declaring rule syntax) and per-statement geometry/delay.
- **TFNE2-06** — S14/S25 statement atomicity. **Done.** Atomicity is
  contextual (`internal != exposed != atomic`): `{s1; s2}` expands each
  statement independently; an invalid resolved projection contributes
  nothing (no nodes/relations/exclusions) while siblings realize fully;
  top-level invalid projections still abort and contradictory selections
  still raise. Carried S6 `O[k](SEG^n)` prefix chains rule `k` over
  instances with head/tail frontiers (only `O[k]` over plain `A^n`;
  `(`/`)` lexed, previously hard errors). Receipt:
  `artifacts/programme/tfne2_conformance_06_receipt.md`.
- **TFNE2-07** — S25 semantic failure vocabulary. **Mechanism subset
  done:** `resolve_mechanism` classifies every name
  (CANONICAL/EXPLICIT_ALIAS-uninhabited/CUSTOM_DEFINED/UNRESOLVED/
  NOT_PERMITTED) with no heuristic aliasing; kinetics resolve at the
  tensor bridge into `StaticParams`; unresolvable refused at execution.
  Receipt: `artifacts/programme/tfne2_conformance_07_mech_receipt.md`.
  Remaining: none on vocabulary — full S25 taxonomy implemented as
  `TFNEError` subclasses (`TFNEAddressUnknown`, `TFNEAmbiguousExpansion`,
  `TFNEExclusionUnknown`, `TFNEFrontierUnresolved`,
  `TFNEInvalidProportion`, `TFNEMechanism*`, `TFNEMissingPolicy`,
  `TFNEOrderViolation`, `TFNEProjectionRedundant`); S13 redundancy refused
  at leaf identity; S11 ungrouped same-rule X refused without grouping or
  `associative = true`. Receipt:
  `artifacts/programme/tfne2_conformance_07_receipt.md`.

  **Unblocked and closed TFNE-PARAM-03** via the mechanism subset above
  (vocabulary → resolution → kinetics → PARAM-03 in one sequence, as
  required). The chain is deterministic and inspectable per the acceptance
  below; `GABA -> GABA_A` was refused as heuristic aliasing (GABA is
  ambiguous, not unaliased).

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

- **CTX-01** — `tfne/2` S27 names `CTX` as the next definition candidate.
  **First integrated model done:** `tests/test_tfne_ctx01.py` exercises
  the conformant algebra end to end (named areas, N/P, O/X bodies,
  frontiers, kinetics, replication, order, atomicity, JDNA completion
  with explicit `K_D`, construction, simulation) with no new grammar.
  Receipt: `artifacts/programme/tfne_ctx01_receipt.md`. S27's
  population/`P_{l,c}` definition family remains future work; the
  `in`/`out` dependency from TFNE2-04 is satisfied.

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

---

---

# 0.5.x programme — optimization + integration family (authorized 2026-09-20)

Governing objective: 0.5.x = faster simulation + faster verification +
better agent use + cleaner integration, subject to Δscientific semantics=0
unless an individually authorized correctness repair requires otherwise.
0.4.x established the semantics and harness; optimize against that stable
baseline and measure equivalence.
*Amended 2026-09-23 (human-authorized):* Δscientific semantics=0 means
existing canonical configurations stay bit-identical to the frozen baseline;
0.5.x is additive-only. New scientific capability is opt-in behind an
explicit declaration, default behaviour unchanged, and enters only via
research → equations/units → isolated validation → generic test → human
authorization → versioned capability.

Acceptance for the whole programme (deltas from frozen v0.4.25 baseline):
T_simulation↓, M_peak↓, T_test↓, T_agent_task↓, E_semantic=0 (no known
unintended semantic changes), C_scientific ≥ C_0.4.25 (retained coverage).

Release sequence (one objective each; NOT all in 0.5.0; amended
2026-09-23, human-authorized). Each release has two lanes — ENGINE ∥ ATLAS —
with one shared acceptance boundary. Atlas simulations AT-01…AT-10
(1N → 2N → population → 2 areas → 20 areas) are consumers/tests of the
engine, not authorities over it. (IDs are `AT-`, not `S`, because `S<n>`
already names tfne/2 sections.)
0.5.0 measurement baseline + integration + skills foundation (sealed);
0.5.1 execute efficiently (runtime/recording + test/gate acceleration,
moved from old 0.5.2) + AT first pass and gap matrix;
0.5.2 source/field: X→Q→Φ→Y, units, geometry (PARAM-04), delay (PARAM-02),
probes, configured→realized→executed inspection (moved from old 0.5.3);
AT-01…AT-06;
0.5.3 state/plasticity/long time, causal clamps; AT-07;
0.5.4 composition, two areas; AT-08, AT-09;
0.5.5 synthesis: AT-10 (20-area JDNA), frozen measurement vector, Atlas
generator, reduction matrix, skill/tool benchmark on AT-01…AT-10 (moved
from old 0.5.4);
later: architecture simplification only with accumulated evidence.
Standing decisions: AT-01 physical anchor starts on the optional Jaxley HH
bridge (`bridges.py`); a native HH emitter would be new semantics (promotion
path above). `jaxfne.vis.atlas_suite` stays the view layer: one simulation,
many panels, no simulation inside visualization.

## 0.5.0 stack (in order; measurement before optimization)

1. Benchmark current v0.4.25/dev performance (DONE 2026-09-20:
   `scripts/benchmark_050_baseline.py` + `artifacts/perf/baseline_050.json`;
   1n/10n-HDP/100n/1000n with construct/sim1-compile/sim2-run/probe/
   manifest phases; first-vs-second-call separates JIT compile).
2. Canonical benchmark models + frozen outputs (DONE: same receipt; rerun
   with `python scripts/benchmark_050_baseline.py`).
3. Profile construct/compile/simulate/record/observe + memory independently
   (DONE 2026-09-20: `scripts/profile_050_phases.py` +
   `artifacts/perf/profile_050.json`; N/E, RSS deltas, HDP recording
   volumes; first-call vs warm separated).
4. Profile test/gate runtime; defect→gate matrix draft (DONE 2026-09-20:
   `artifacts/perf/test_profile_050.md`; broad 2216 s dominated by
   equivalence/construct/generator tests; 5 proposals, no actions).
5. Audit TFNE→JDNA→Model integration for duplicate construction paths
   (DONE 2026-09-20, delegated analysis: 41 routes classified canonical /
   thin-adapter / independent-primitive / stub; redundant candidates listed,
   no action; 6 open questions incl. tensor-connection scaffolding and
   tutorial dict-engine status).
6. Audit/refine current skills against the lifecycle (DONE 2026-09-20:
   coverage map — TFNE/JDNA: jaxfne-core routing only (no dedicated
   skill; gap for item 7 workflow); Model/Simulation: jaxfne-repo +
   jaxfne-science; Observation/Evidence: jaxfne-science + jaxfne-audit;
   release/seal/vocab covered. Refines applied: canonical-entry pinning
   rule in jaxfne-core; claim-conditioned verification in jaxfne-science.
   Remaining gap: task-shaped skills (model/develop/simulate/inspect/
   fields/plasticity/validate/optimize) deferred to item 7 workflow).
7. One canonical question→TFNE→JDNA→simulation→verification agent workflow (DONE 2026-09-20: `artifacts/skills/jaxfne-workflow/SKILL.md` — routing+gates table invoking existing skills, no implementation duplication; harness manifest synced, 8 skills).
8. Benchmark that workflow against raw-repository agent use (DONE 2026-09-20:
   frozen packet + 10 independent runs; F̄=1.00 both arms, ΔF=0, ΔH=0;
   friction unmeasured except files-consulted proxy; results in
   `artifacts/perf/workflow_benchmark_results_050.md`; no optimization
   follows).
9. Rank actual bottlenecks (DONE 2026-09-22: two-lane Rc/Rp(W) evidence matrix; Item 9 PASS).
   Rc repair checkpoint (DONE 2026-09-22, pre-Item-10, Δsimulation=0: P1
   non-vacuous sync check; P5 shared rate/band contracts + parity tests;
   P2 rgba rate-alpha; P3/P7 honest time labels + square-axis refusal; P4
   synthetic-geometry provenance; P8 route distinction; plot_depth_profile
   backend divergence declared-STOP, not unified).
10. Only then authorize the first optimization batch (DONE 2026-09-23:
   manifest conservation-diagnostics jnp→numpy; frozen gate FAILED strict
   abs leg as recorded, human-authorized allclose correction, re-evaluated
   PASS: 19 floats 0 failures, cold 379→1.7 ms; receipt
   `artifacts/programme/item10_adjudication_receipt.md`; timing
   `artifacts/perf/item10_manifest_timing_050.json`; frozen baseline_050
   untouched).
   0.5.0 SEALED+published 2026-09-23: tag v0.5.0 (peel 499c54f), origin/main
   @ 499c54f, GitHub release, PyPI 0.5.0 (hashes == CI manifest; clean-install
   smoke verified), RTD auto-registered; receipt
   `artifacts/release/v0_5_0_release_receipt.json`; post-release sync done.

## 0.5.1 stack — execute efficiently (ENGINE ∥ ATLAS)

Question: can the spacetime scales the Atlas needs run without recording
or construction dominating? (Δt,T,Δr) = f(phenomenon), so cost is measured
as C(N,E,T,Δt,N_H,N_W,N_recorded,mechanism), not at one resolution.

0. Prerequisite (human): add the Atlas source document to
   `artifacts/project_sources/`; AT-01…AT-10 definitions reference it, not
   copies of it. ATLAS items 5–7 block on this; ENGINE items do not.

ENGINE
1. Extend the benchmark matrix (reuse `scripts/benchmark_050_baseline.py`,
   `scripts/profile_050_phases.py`; frozen `artifacts/perf/baseline_050.json`
   untouched): warm-step scaling in N, E, T; recording off/minimal/full for
   H, W, Q, Φ where the path records them; chunked/continuation cost;
   construction vs long-run amortization; one point at projected AT-10
   (20-area) size so 0.5.5 is not the first contact with that scale.
2. Rank bottlenecks from item 1. If recording dominates: selective /
   downsampled recording as an opt-in API; default recording unchanged.
3. Optimizations one at a time, each with a predeclared equivalence gate
   (item-10 pattern: freeze gate → run → adjudicate → receipt).
4. Test/gate acceleration (moved from old 0.5.2): act on
   `artifacts/perf/test_profile_050.md` proposals 1 (narrow equivalence
   gate), 3 (cache immutable evidence), 4 (parallelize families; measure
   before/after). Proposals 2 and 5 are invariants: keep adversarial
   TFNE/JDNA tests in dev; every historical defect keeps a cheap detector.

ATLAS
5. First pass: AT-01…AT-10 at toy size (smallest N, short T), current
   public features only, defined as TFNE/JDNA data. AT-01 via the Jaxley
   bridge; if Jaxley is absent the run records REFUSED, not a substitute.
6. Measurement vector schema v0: Y = {X, H, W, Q, Φ_E, Φ_B, SPK, PSD, C, φ,
   E_reduction, T_compute, M_compute}; absent quantity = `OMITTED`, refused
   capability = `REFUSED`, never synthesized. Evolves 0.5.2–0.5.4, frozen
   in 0.5.5.
7. Gap matrix Y × AT (IMPLEMENTED / OMITTED / REFUSED) from item 5, stored
   as an artifact. It is the burndown for 0.5.2–0.5.5: each release names
   the cells it moves.
8. Firewall gate: AT scenarios import only the public surface
   (`jaxfne/public_surface.py`); a gate refuses private-module imports and
   any AT-specific branch in `jaxfne/`.

ACCEPTANCE (0.5.1 seal)
- Existing canonical configurations bit-identical to the frozen baseline.
- AT workloads and the 20-area point are in the benchmark matrix, measured.
- Every optimization has a PASS equivalence receipt; failures recorded.
- Gate wall time measured before/after for each test/gate change.
- Schema v0, gap matrix and firewall gate committed; gate active in CI.

## 0.5.2 stack — source/field (ENGINE ∥ ATLAS)

Question: does X → Q → Φ(r,t) → Y hold with each output's epistemic level
explicit and tested? Levels: CALIBRATED ≠ REDUCED_PHYSICAL ≠ RELATIVE_PROXY.
Full electrodiffusion stays out of scope.

0. Decisions (human), before items 1–2:
   a. TFNE-PARAM-04 coordinate semantics: absolute or relative declared
      geometry (same authority question as TFNE-PARAM-03). The repair
      changes executed positions for TFNE specs whose declared range is not
      the unit interval, so it is a correctness repair needing individual
      authorization under the programme rule.
   b. Delay declaration unit: ms (realized as `delay_steps` via dt, with a
      declared rounding/refusal rule) or steps.

ENGINE
1. TFNE-PARAM-04 repair: declared geometry reaches executed positions;
   invert `test_declared_geometry_does_not_reach_the_executed_positions`.
2. TFNE-PARAM-02 repair: delay on the connection-rule surface →
   `compile_connection_rules` → `InterConnection` → `EdgeList.delay_steps`.
   Absent delay = current behaviour. Tests: zero-delay limit bit-identical,
   arrival timing, continuation across a chunk boundary with spikes in
   flight, composition hook for 0.5.4.
3. One source representation Q consumed by every probe in `fields/probes.py`
   and `fields/proxy.py`; existing proxy probe outputs bit-identical.
4. Epistemic level carried on every Φ/Y output (`FieldOutput`,
   `ProbeReadout`, manifest). A proxy becomes CALIBRATED only through an
   explicit calibration transform declaring units, conductivity and
   distance; a gate refuses the relabel otherwise.
5. Probe/electrode semantics: position, reference and filter declared; no
   invented contacts or positions on scientific paths (extends the Rc P4
   synthetic-geometry provenance).
6. Configured → realized → executed inspection per capability (moved from
   old 0.5.3), first for geometry, delay, source and probe; the table under
   "Parameter ownership, as measured (TFNE-PARAM-01)" is the template.
7. Field-path cost (Q and Φ recording at AT sizes) added to the 0.5.1
   benchmark matrix.
8. Field visualization consumes canonical Q/Φ only; resolve or keep the
   declared-STOP `plot_depth_profile` backend divergence, with the choice
   recorded.

ATLAS
9. AT-01 physical anchor: Jaxley HH reference (REFUSED if Jaxley absent)
   against the reduced neuron; spike-time, rate and Vm-feature tolerances
   declared before the run.
10. AT-02 pair transmission and AT-03 recurrent pair, with declared delay.
11. AT-04 source superposition. Φ → X feedback fails closed unless
    independently justified; causal distinctions use supported
    perturbations only.
12. AT-05 population field emergence; AT-06 geometry/electrode locality
    (needs items 1 and 5).
13. Reduction row HH → reduced neuron → population source, against the
    predeclared tolerances; failures recorded. Measurement schema v0 → v1.
14. Gap matrix updated; the seal note names each cell moved.

ACCEPTANCE (0.5.2 seal)
- Existing canonical configurations bit-identical, except outputs changed by
  the authorized PARAM-04 repair, each listed with its receipt.
- Zero-delay limit bit-identical to pre-0.5.2 execution.
- Epistemic-level relabel refusal tested adversarially (H5).
- Firewall gate passes: AT-01…AT-06 use the public surface only.
- Reduction tolerances and results recorded, PASS or FAIL.

## Tracks (parallel after the baseline)

- P performance: benchmark matrix (1n dispatch; 10n HDP; 100n sparse;
  1000n column; 10k sparse; high-density; recording-heavy;
  continuation/chunked; TFNE→JDNA→Model construction); T_total =
  construct+compile+simulate+record+observe; M_peak =
  persistent+construction+state+recording+temporary. Optimize only
  profiled components.
- T testing: micro⊂dev⊂broad⊂release hierarchy by defect class; maximize
  defects/(wall+compute); defect→cheapest-gate matrix; keep adversarial
  semantic tests; cache immutable evidence where valid; parallelize
  families; every historical defect keeps a cheap local detector.
- S skills: task-shaped skills (model/develop/simulate/inspect/fields/
  plasticity/validate/optimize) around question→skill→TFNE→JDNA→Model→
  simulation→verification; benchmark vs agent+raw-repo on semantic model
  fidelity (not answer accuracy).
- I integration: converge on TFNE→JDNA→Model→Simulation→Observation; every
  alternate path classified as thin adapter / independent primitive /
  deprecated redundancy. One semantic implementation, many entrances.
- A architecture: god-modules are candidates, not tasks; factor only with
  measured benefit under old≡new output (bit-exact or predeclared
  tolerance).
- V validation: generalize configured→completed→realized→executed→observed
  per capability (identity/sign/scale/shape/units/geometry/tau/delay/
  stochastic/mutable/source-field); claim-conditioned V=V(claim).
- U inspection: cheap origin/provenance queries (summary/inspect/trace/
  explain/compare); index/provenance map I answers specified→executed.

## Carried staging (prior candidates, not started)

- A8 gh-pages publishing policy (versioned paths, RTD story, audit
  handling) — decide before any asset migration.
- C5–C7 deferrals: matmul reassociation, div-to-mul rewrites, null-term
  elision absent a signed-zero/NaN exactness contract; kernel memoization;
  `record_weight_trace` default/layout changes; `[T,N]`/`[T,X]` layout
  changes.
- P-001: repo-wide `scripts/` legacy lint bulk cleanup.
- TFNE-PARAM-02 (delay refused) and TFNE-PARAM-04 (geometry inert) remain
  open under JDNA ownership, per the standing sections above.
- Frozen protocols must record JAX/lib versions in future bundles (P-003
  provenance rule).
- Deep audit 2026-09-20 (verified, not trusted): tune() now warns on
  silently-dropped mixed args + 3 tests; validate_configuration docstring
  states FAIL-report (not raise); completion.py comment matches landed
  TFNE2-07 vocabulary. Deferred owner-decisions: units.py + pynwb_compat
  unwired (wire or remove); emitters.py variant split (10 parallel
  simulate_edge_recurrent_izhikevich_*, 1052L HDP fn); entry fragmentation
  (≥8 simulate forms, ≥10 config builders, RuntimeConfig vs
  RuntimeConfiguration, compute_fields accessor, dual manifests, two
  same-named build_laminar_column/export_tutorial_artifacts);
  compile_step_fn **hdp_kwargs unknown-key policy; validate_hdp_params
  non-dict non-strict silent-pass; broad UNTESTED-exact refusal tail;
  PLACEHOLDER_NOTEBOOKS + artifact-gated skips;
  post-0.4.14 compat aliases. Sealed: stale v0.3.4 xfails removed (4 dead
  tests, suites green). Two worker false alarms (pool path, R4/R6
  coverage) corrected in review — reviewers≠workers stays mandatory.
- Integrity round (P-006): 13 api signature drifts fixed + verified,
  `audit_doc_code_integrity.py` gate added (fences/refs/symbols); reviewer
  caught 1 worker false claim before application — reviewers≠workers stays
  mandatory. Gate is permanent harness: `doc_code_integrity_audit` family in
  broad/release/rc + explicit ci.yml step.
