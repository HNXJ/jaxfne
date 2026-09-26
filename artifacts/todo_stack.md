# Remaining work

> **Concurrent editors (2026-09-23).** The human owner and a Claude Code
> session also edit this file and other files under `artifacts/`. We change
> them by small, incremental adjustments: amend, reorder or annotate existing
> items. We make no wholesale rewrites and drop no item without a note.
> In progress: 0.5.x re-scoped to add an Atlas track (AT-01…AT-10, 1N → 2N →
> population → 2 areas → 20 areas). Release map and 0.5.1–0.5.5 stacks are in the
> 0.5.x programme section. Atlas source: `artifacts/project_sources/8_atlas.md`;
> requirements and coverage state: `artifacts/programme/atlas_coverage.json`.
> **Handoff (human-approved plan, 2026-09-23):** the planning session is
> archived; the opencode agent executes from here. Start at 0.5.1 ENGINE
> item 1a. Read "Goals in plain words" first. Human decisions listed in the
> stacks (items marked "Decisions (human)") block only the items they name. Before editing, re-read from
> disk. A diff you did not make is a concurrent edit (H12): keep it and do not
> revert it.

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**Current:** v0.5.0 published (see "0.5.0 (sealed)"); work proceeds in the
0.5.x programme below. The sections from "MODE = TFNE/2 ADOPTION" to
"Long-term goal" are kept as rationale; their open items are scheduled in
the 0.5.x stacks — see "Where older open items went".

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

*Superseded 2026-09-23:* TFNE2-04 is done (see below), PARAM-03 is done,
PARAM-02 is 0.5.2 item 2, PARAM-04 is 0.5.2 item 1. Kept for rationale.

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
  same authority question as TFNE-PARAM-03. *Decided 2026-09-23: relative,
  fractions in [0,1]; repair scheduled as 0.5.2 item 1.*

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

## Goals in plain words (0.5.1–0.5.5)

Whole programme — **want:** a faster JaxFNE that can run the full Atlas,
from one neuron to twenty brain areas, and a manuscript built on it.
**Need:** old results never change; new abilities are switched on
explicitly; every claim has evidence.

| Release | What we want | What we need |
|---|---|---|
| 0.5.1 | Atlas-sized runs and our own checks finish fast enough. | Measure where time and memory go; speed up only the measured slow parts; show results stay identical. |
| 0.5.2 | Go from neuron activity to what an electrode would record. | Positions and delays that take effect; every signal labelled calibrated, approximated or relative; Atlas simulations 1–6. |
| 0.5.3 | Long runs where the network changes itself, and experiments that show cause. | Every changing quantity visible, replayable and switchable (on/off/clamp); split runs match unsplit runs; Atlas simulation 7. |
| 0.5.4 | Connect brain areas. | Joining areas leaves each area unchanged inside; delays and learning between areas; Atlas simulations 8–9. |
| 0.5.5 | One Atlas showing the same model language works from 1 neuron to 20 areas, and the manuscript. | The 20-area simulation; one shared list of measurements; what survives each simplification; generated figures; every manuscript claim tied to evidence. |

Every release seal also requires: the Atlas coverage rows it owns
(`artifacts/programme/atlas_coverage.json`) moved past PLANNED with an
evidence path, or deferred by the human with a reason; and each
`candidate: true` row it owns either promoted through the path in the
programme rule or marked OUT_OF_SCOPE by the human.

## Where older open items went (reconciled 2026-09-23)

| Open item (older section) | Scheduled |
|---|---|
| TFNE-PARAM-04 geometry inert | 0.5.2 item 1 |
| TFNE-PARAM-02 delay refused | 0.5.2 item 2 |
| Agent-native step 3 (capability inventory) and step 4 (capability records; delay is the worked example) | 0.5.2 item 6, then each release for its capabilities |
| PARAM-03 check "TFNE construction vs equivalent hand-written construction" (not in the mech receipt) | 0.5.2 item 6 |
| TFNE2-03 exception: cell types still ordered by `C = {...}` enumeration | 0.5.4 item 1b |
| TFNE2-05 carried: per-statement delay/geometry in rule bodies | 0.5.4 item 1c |
| TFNE2-04/05 carried: `X[k]`-rule frontier override (language decision) | 0.5.4 item 0 |
| Agent-native step 2 (canonical TFNE → JaxFNE compilation and `I`) | 0.5.4 item 2 |
| Agent-native steps 5, 6, 8 (tool surface, task-shaped skills, adversarial semantic-substitution tests) | 0.5.5 items 5a–5c |
| Agent-native steps 7, 9 (frozen end-to-end tasks, benchmark) | 0.5.5 item 5d |
| Everything in "Carried staging", S27 `P_{l,c}` definitions, agent-native step 10 (MCP) | after 0.5.5 — "Open work outside the release stacks" |

## 0.5.0 (sealed)

Sealed and published 2026-09-23. Release receipt
`artifacts/release/v0_5_0_release_receipt.json`; item evidence in
`artifacts/perf/*_050*` and `artifacts/programme/item10_adjudication_receipt.md`;
Rc repair checkpoint in commit `bfc88a7`. Frozen `artifacts/perf/baseline_050.json`
stays the equivalence reference.

## 0.5.1 stack — execute efficiently (ENGINE ∥ ATLAS)

Question: can the spacetime scales the Atlas needs run without recording
or construction dominating? (Δt,T,Δr) = f(phenomenon), so cost is measured
as C(N,E,T,Δt,N_H,N_W,N_recorded,mechanism), not at one resolution.

Atlas routing: source `artifacts/project_sources/8_atlas.md`; requirement
rows `AT-0n-R<k>` with release, candidate flag and state in
`artifacts/programme/atlas_coverage.json`. Each release's ATLAS lane owns
every row whose `release` names it. `S<n>` inside `8_atlas.md` ≡ `AT-0n`;
elsewhere `S<n>` stays a tfne/2 section. `B` = magnetic field throughout.
Rows with `candidate: true` (calibrated HH/field, Φ_B beyond proxy,
Φ → X, B as input to dynamics) need independent evidence and human
authorization before any engine work.

ENGINE — sealed 2026-09-24 (commits on dev via merges; receipts in
`artifacts/perf/` + `artifacts/programme/opt051_*_receipt.md`):
- Item 1 (matrix): spec frozen before measuring (`matrix_051_spec.json`,
  14 cells), harness `scripts/benchmark_051_matrix.py`, results
  `matrix_051.json` 14/14 MEASURED incl. coupled AT-10 (20k neurons, 4.7M
  edges, 9.1% inter-area); `baseline_050.json` untouched.
- Item 2 (ranking): `bottlenecks_051.md`; only B1/B2 entered item 3, rest
  carried per stop rule 3d (no remaining change ≥10% of its cell).
- Item 3 (optimize): two landed, both bit-exact with PASS receipts —
  opt051_1 jit-auto default (warm −58..−98%), opt051_2 rule-compile
  selection index (AT-10 construct −48%, edges identical).
- Item 4 (test/gate speed): equivalence narrowing, fig06 evidence cache,
  xdist dev/broad/slow sweeps; before/after in `test_timing_051.md`
  (broad 31→6 min); defect table fully covered after each change.

ATLAS — sealed 2026-09-24:
- Item 5: toy pass AT-01…AT-10, 10/10 OK in ~27s
  (`artifacts/atlas/at01_at10_toy.py`; AT-01 via real Jaxley bridge).
- Items 6+7: schema v0 + gap matrix `artifacts/programme/atlas_gap_051.md`
  (20 IMPLEMENTED / 10 REFUSED / 100 OMITTED).
- Items 8/8b: firewall gate (`tests/test_atlas_firewall.py`) + coverage
  check (`scripts/check_atlas_coverage.py`), both wired into CI.

Failures recorded (open, owned): B8 chunked-vs-single spike divergence
(173 vs 174, identical seeds; fixed in 0.5.3 item 3, P-010);
`test_source_generation_vs_projection_split` wall-time flake (fails
identically on pristine fd46e1c; P-009, closed 2026-09-24 by warmup). AT-10 benchmark wiring is a
full-bipartite placeholder (real wiring is G_20, 0.5.5).

ACCEPTANCE (0.5.1 seal) — met: canonical configs bit-identical (broad
4104 green on merged tree); matrix + 20-area point measured; every landed
optimization has a PASS receipt (no landed-and-reverted attempts; excluded
candidates listed in `bottlenecks_051.md`); gate timings before/after
recorded; schema/gap/firewall committed and active in CI.

## 0.5.2 stack — source/field (ENGINE ∥ ATLAS)

Question: does X → Q → Φ(r,t) → Y hold with each output's epistemic level
explicit and tested? Levels: CALIBRATED ≠ REDUCED_PHYSICAL ≠ RELATIVE_PROXY.
Full electrodiffusion stays out of scope.

0. Decisions (human, DECIDED 2026-09-23):
   a. TFNE-PARAM-04: relative coordinates. A declared range is a pair of
      fractions of the area's extent, within [0,1]; a range outside [0,1]
      is refused. The repair is authorized as a correctness repair: specs
      declaring a sub-range of [0,1] change executed positions, and the
      pinned test's `(10,20)` and `(-5,-4)` cases become refusals. List
      every changed canonical output in the seal receipt.
   b. Delay: declared in ms. Realized as `delay_steps = round(delay_ms /
      dt_ms)`; configured ms and realized steps both recorded in the
      manifest; a positive delay that rounds to 0 steps is refused.

ENGINE — sealed 2026-09-24 (commits e91adee + d872816 on dev; receipts in
`artifacts/programme/`):
- Items 1+2 (PARAM-04/PARAM-02): relative geometry reaches execution
  (outside-[0,1]/half/degenerate refused); delay ms → steps chain with
  manifest recording; zero-delay bit-identical; changed canonical outputs:
  none. Receipts `tfne_param04_receipt.md`, `tfne_param02_receipt.md`.
- Items 3+4+5 (Q, epistemic, probes): `CanonicalSource` consumed by every
  probe (bare-array outputs bit-identical); levels + sealed calibration +
  refusal gate; electrode semantics declared, invented contacts refused
  unless explicitly opted in. Manifest carries `field_epistemic`
  (dispatcher follow-up, tested).
- Item 6 (inspection + records): `capability_inventory.md`,
  `capability_geometry.md`, `capability_delay.md`; PARAM-03
  TFNE-vs-handwritten check found existing
  (`test_tfne_matches_handwritten_canonical_construction`), not duplicated.
- Item 7 (field cost): 3 cells appended to the 0.5.1 matrix (prior cells
  byte-identical).
- Item 8 (field vis): divergence KEPT (doc-only, zero outputs changed);
  Plotly refuses source-only signals, matplotlib draws declared Φ.

ATLAS — sealed 2026-09-24 (`artifacts/atlas/at01_at06_052.py`, 6 commits):
- Items 9–12: AT-01 (Jaxley HH anchor, tolerances predeclared; v_peak +
  spike-time FAILs recorded, not tuned), AT-02/AT-03 (declared delay),
  AT-04 geometry arm (Φ→X refused), AT-05/AT-06 (C(R,f) proxy; inequality
  not reached at toy size, owned by 0.5.5 scale matrix). 29 tests green.
- Item 13: reduction row with predeclared tolerances; schema v0→v1
  (v0 names stable).
- Item 14: gap v1 (`atlas_gap_052.md`): 41 IMPLEMENTED / 6 REFUSED /
  31 OMITTED over AT-01…AT-06; 29 cells moved, named.
- Candidate rows AT-01-R4/R5, AT-04-R3 marked OUT_OF_SCOPE per human
  decision 2026-09-24 (no independent evidence yet); scenarios fail
  closed where they apply. Open deviations recorded: TFNE 16-vs-4
  contacts (declared-vs-realized); AT-03 single-mechanism bidirectional
  rule (S27 P post-0.5.5).

ACCEPTANCE (0.5.2 seal) — met: canonical configs bit-identical except
authorized PARAM-04 outputs (none changed); zero-delay bit-identical;
relabel refusal adversarially tested; firewall passes for AT-01…AT-06;
reduction tolerances/results recorded PASS and FAIL.

## 0.5.3 stack — state, plasticity, causality, long time (ENGINE ∥ ATLAS)

Question: can (X, H, B, W) → Q → Φ run for long horizons with each mutable
state owned, inspectable, replayable and causally intervenable?
Separations kept explicit: H ≠ HDP; attenuation ≠ adaptation;
bounded ≠ stable. Symbol definitions come from the Atlas source and
`docs/doctrine/rbs_rbd_hdp.md`, not from this file. B is the magnetic field
(an observation); B as an input to dynamics is a candidate (AT-07-R4).

ENGINE — sealed 2026-09-24 (merges `76fa9bb`, `4f4d88d`; receipts
`artifacts/programme/*_053.md`):
- Items 1–4 (ownership, budgets, continuation, replay): H/W/K
  configured→realized→executed table; H/W stride/subset budgets (full
  recording stays default); chunked ≡ continuous bit-exact for all
  mutable state incl. delays in flight; replay bit-identical with
  declared RNG domains. P-010 (173 v 174) root-caused and fixed in
  `865e74b`: plain path drew bulk noise, chained path per-step.
- Items 5, 6, 7, 7b: per-rule/per-projection enable/disable/clamp
  (adversarial); `jaxfne/intervene.py` intervention object with manifest
  roundtrip; boundedness and stability as separate measurements; HDP
  unknown/malformed parameters fail closed (H7 table).
- Decisions (human, 2026-09-24): (a) APPROVED — Model-level stochastic
  runs draw the new chain-consistent noise stream and honor declared
  `noise_scale`; seeded stochastic outputs change (changelog
  [Unreleased]). (b) ACCEPTED — baseline jit-vs-eager V/sources move from
  bit-exact to ≤ EPS_V (observed 3.9e-5; spikes exact).

ATLAS — sealed 2026-09-24 (merge `e9f12b8`; homeostasis fix `edadb87`;
broad 4437 cases / 0 fail on HEAD; `atlas_coverage VALID`):
- Items 8/9/10 (AT-07 fixed-W vs HDP, AT-04 H-perturbation arm, schema
  v1 → v2, gap matrix v2).
- P-010 follow-up: homeostatic kernel accepts the Model-level chain
  noise schedule, so the k_gain=0 null matches the edge_list baseline
  exactly (`test_kgain_zero_null_matches_baseline_edge_list` green in
  the full suite); legacy None path bit-identical.

ACCEPTANCE (0.5.3 seal)
- Existing canonical configurations bit-identical, except the approved
  stochastic noise-stream change (decision a above); plasticity-off equals
  the pre-0.5.3 fixed-W path bit-for-bit.
- Chunked ≡ continuous for every mutable state, delays included.
- Replay bit-identical; RNG-domain isolation tested.
- Clamp and disable tested adversarially; interventions serialize and
  roundtrip (H4).
- Long-horizon AT runs fit within the 0.5.1 performance envelope, or the
  overrun is recorded with its cause.

## 0.5.4 stack — composition and two areas (ENGINE ∥ ATLAS)

Question: does N_A ⊕_C N_B → (s, h_0, I) preserve everything each area
had alone? Acceptance is capability and identity, not a phenotype.

ENGINE — sealed (commits `cd7dbca`, `6792bc7`, `dcec360`, `e4bd459`,
`4bab2f1`, `04bbd26`, `c15cd07`; receipt
`artifacts/programme/composition_054.md`):
- Item 1 (`connect()` + CTX-01 path): member RNG domains
  (`ensemble_member_seed`, solo-identity eager+jit); continuation member
  streams (chunked == continuous == solo); cross-edge dt_ms (delay_ms now
  compiles, recorded in ensemble metadata); S12 per-statement
  `[delay=MS]` (parse→relation→tensor→steps). Geometry/depth preserved,
  x disjoint; probes/fields finite; HDP engages; batch deterministic.
- Item 2: hierarchical-vs-flat TFNE programs, same realized values and
  bit-identical runs (test-only lock-in; compilers intentionally
  distinct, not unified).
- Item 3: `population_rate` + `cross_area_coherence` in `jaxfne.fields`
  (coherence ~1 on identical sines, delay reads as phase, silence
  declared valid=False, ensemble per-area + cross verified).
- Item 4: `ensemble_edge_ownership` (member + per-rule cross ranges,
  W_12/W_21 separate); mask scoping freezes exactly, moves the rest;
  zero mask disables; replay bit-exact.
- Item 5: k=1..4 scaling probe (`scaling_054.json`, all finite; frozen
  0.5.1 matrix untouched).
- Deferred: 1b (S20.1 ordering — individual human authorization
  required); item 0 (X[k] frontier — not needed for AT-08/AT-09, stays
  after 0.5.5).

ATLAS — sealed (commit `991a00a`; assay
`artifacts/atlas/at08_at09_054.py`, tests
`tests/test_atlas_at0809_054.py`, `atlas_gap_054.md`, coverage
AT-08/AT-09 VALIDATED):
- Items 6/7 (AT-08 fixed/adapt arms with R4 decomposition; AT-09
  cross/member/frozen arms with R2 separation and R3 comparison).
- Item 8 (schema v2 → v3: 13 stable names + 10 area/cross cells; gap
  matrix updated).

ACCEPTANCE (0.5.4 seal)
- Existing canonical configurations bit-identical.
- Each area run alone ≡ that area inside the composition with cross-area
  edges set to zero.
- Hierarchical ≡ flattened, bit-identical.
- Composition survives serialization roundtrip and chunked continuation.
- Firewall gate passes for AT-07…AT-09.

## 0.5.5 stack — Atlas synthesis, scaling, manuscript (ENGINE ∥ ATLAS)

Question: does one small grammar survive changes in neural scale, physical
scale and timescale, 1N → 2N → population → 2A → 20A? End of 0.5.5 = the
whole Atlas manuscript plan complete.

0. Decisions (human, DECIDED 2026-09-23): the Atlas manuscript is a new
   manuscript in `artifacts/publication/atlas/`, write-once per figure.
   The frozen Figure 1–7 snapshot
   (`artifacts/publication/frozen_manifest.json`) stays untouched and
   citable; Atlas figure numbering is independent of it and of the E2
   Figure 8–9 plan.
   Decisions (human, DECIDED 2026-09-25): (a) `artifacts/atlas/y_schema.py`
   stays an artifact module through 0.5.5 (no public API move).
   (b) `build_atlas` keeps its simulate fallback unchanged; the Atlas
   generator uses a new view-only entry that takes data and refuses to
   simulate. (c) Per-simulation data bundles are in memory: runners return
   the bundle beside their summary and the generator re-runs from the
   manifest; nothing large is persisted. (d) AT-07-R4 (B as input to
   dynamics) is OUT_OF_SCOPE: no independent evidence or calibration
   anchor, same class as AT-01-R4/R5.
   Decision (agent, 2026-09-25, recommended default): `build_atlas` given
   signals records seed/duration only when passed, null otherwise; it no
   longer stamps its own defaults as the run identity of foreign signals.

ENGINE
3. Remainder of the manifest item (spec registry, digest, manifest and
   inheritance check landed in `artifacts/atlas/at_manifest.py`): runners
   still hard-code the inputs each spec lists under `transcribed`
   (file:line per field), so the spec describes the run but does not drive
   it. Move those literals to module constants or have runners read
   `at_spec`, until every `transcribed` list is empty; then add a
   regeneration test per AT (only AT-02 has one, marked slow). The three
   `_spec_digest` variants (053, 054, `intervene.network_spec_digest`)
   stay separate until then.
5a. Agent tool surface (agent-native step 5): `realize`, `inspect`,
   `simulate`, `compare`, `observe`, `verify` over the typed objects, not
   wrappers over every function.
5b. Task-shaped skills around it (step 6): model, network, state,
   plasticity, fields, simulate, verify, inspect; harness manifest synced.
5c. Adversarial semantic-substitution tests (step 8): stale docs, renamed
   APIs, parameter substitution, wrong units or types.
5d. Skill/tool benchmark (moved from old 0.5.4), after 5a–5c: AT-01…AT-10
   with frozen expected properties as the task set (step 7); skill+tools vs
   direct repository use on scientific-model fidelity (step 9 metrics).

ATLAS
6. AT-10: G_20 →D(K_D)→ N_20 via JDNA; baseline, plastic and
   stochastic-plastic phases on the same realized system. Bounded
   trajectories count as evidence of active stabilization only with a
   perturbation/control assay (0.5.3 items 6–7) that separates
   bounded ≠ returning ≠ homeostatically stabilized.
   State: JDNA `develop(G, seed)` exists with one single-area genome
   (`canonical-v1-column-1000n.json`); no G_20 genome, no N_20 builder,
   no declared cross-area delays (the 0.5.1 `at10_20area` benchmark cell
   is a full-bipartite ring placeholder). AT-10 exists only as the
   3-area toy. The same assay also moves AT-07-R3 (now SUPPORTED) to
   VALIDATED.
7. Reduction/scale matrix: for each transition M_i → M_(i+1), which
   observations survive within the predeclared tolerance and which do not;
   failures stay in the matrix.
8. Performance/reduction map: T_compute and M_compute per AT beside
   E_reduction.

MANUSCRIPT (ends 0.5.5)
9. Atlas coverage matrix: every section, figure, simulation and claim in the
   Atlas source → todo item → produced artifact. 100% mapped or marked out
   of scope by the human. Runs at every release seal and at 0.5.5 seal. The check reads `artifacts/programme/atlas_coverage.json`
   mechanically, not prose; seal needs every requirement ID VALIDATED or
   CANONICAL, or marked out of scope by the human.
10. Figures via the existing generator seam (`scripts/publication_figures/`
    pattern: `build_figure*()`, semantic spec, semantic audit, generation
    receipt, equivalence gate), consuming generated Atlas data only. Every
    figure uses the columns structure → dynamics → state/plasticity →
    source → field → observation → computation (AT-00-R5).
11. Claim ledger for the Atlas manuscript: each claim with its
    claim-conditioned verification V(claim) and evidence path; package
    capability claims kept apart from scientific-result claims.
12. Methods from manifests: units, calibration level, tolerances, seeds and
    versions quoted from generated artifacts, not retyped.
13. Citations primary-verified (H10) before inclusion.
14. Manuscript text complete and consistent with the claim ledger;
    submission is a separate human action.

ACCEPTANCE (0.5.5 seal = end of programme)
- Existing canonical configurations bit-identical to the frozen baseline.
- AT-01…AT-10 regenerate from manifests; Atlas and figures reproduce.
- Coverage matrix 100% (item 9).
- Programme deltas vs v0.4.25 measured: T_simulation, M_peak, T_test,
  T_agent_task, E_semantic = 0, C_scientific retained.
- Every manuscript claim traces to a ledger row with PASS evidence; negative
  and failed results reported as such.

## Open work outside the release stacks (after 0.5.5 unless a trigger fires)

Details stay in "Carried staging" below. In order:
1. A8 gh-pages publishing policy. Trigger: before any Atlas HTML or figure
   is published to public docs (0.5.5 items 2 and 10).
2. `units.py` and `pynwb_compat` unwired: wire or remove (owner decision).
3. Architecture candidates (`emitters.py` variant split, entry
   fragmentation, dual manifests, same-named builders). Trigger: 0.5.1
   item 2 ranks one as a measured bottleneck; then it enters 0.5.1 item 3.
4. C5–C7 numerical deferrals; need a signed-zero/NaN exactness contract
   first.
5. UNTESTED-exact refusal tail; PLACEHOLDER_NOTEBOOKS and artifact-gated
   skips; post-0.4.14 compatibility aliases.
6. P-001 `scripts/` legacy lint cleanup.
7. S27 population/`P_{l,c}` definition family. Trigger: AT-06 or AT-10
   needs population definitions that CTX-01 cannot express.
8. Agent-native step 10: MCP or other transport.

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
