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

# 0.4.25 — TFNE-algebra + visualization sweep (in progress; v2 plan)

Scope: (a) vis drop-in sealed+pushed (`9f79daa`); (b) seven-panel dark Plotly
atlas per model, TFNE→JDNA→Model teaching lifecycle everywhere, simplified
language; (c) full 0.4.25 release. This v2 plan supersedes the v1 batch
sequence below except where re-referenced: v1 slugs/configs/run-labels are
retained assets; v1 six-panel outputs will be regenerated under the 7-panel
spec. Target end state:

$$
\boxed{
\text{one scientific model}
\rightarrow
\text{one provenance-bearing simulation}
\rightarrow
\text{many documentation views}
}
$$

Locked authority decisions: stable TFNE subset is narrow (ordering,
frontiers, rule bodies, atomicity, mechanism resolution, replication, JDNA
completion, CTX-01; delay/geometry shown as limitations); every atlas panel
is defined by its data contract (§Batch A2), not its title; page archetype is
`question -> minimal algebra -> run -> result/figure -> interpretation/limits
-> links`; gallery is generated from manifests (never hand-maintained);
H-SPICE is native Plotly (no PNG-in-chrome); recording policy is budget-based
(§Batch A3), never neuron-count-based; `T_atlas = 1000 ms` is a
visualization/example standard, NOT scientific identity ("canonical
general-purpose atlas examples use 1000 ms unless the scientific example
declares another duration").

## Release goal (authorized 2026-09-20)

Single full 0.4.25 release (ultimate 0.4.x; 0.5.x chapter after) once ALL hold:
(1) Docs: low-verbosity, simple, smooth; tables/lists/paragraphs/HTML
figures interleaved; theme matched; TFNE grammar + terms consistent; left
menu (`mkdocs.yml` nav) organized.
(2) Code: Batch F acceptance — low complexity, optimal operation order,
canonical flattening; JAX switches (float32/64, cuda/cpu/parallel-cpu/metal)
via official mechanisms; official-doc conformance suffices, no bespoke
JAX-plumbing tests.
(3) Stacks empty: this 0.4.25 section fully removed item-by-item as sealed
AND issue-log Open section empty with resolutions recorded (history
preserved, never rewritten; new problems detected mid-sweep get IDs and are
drained the same way).
Authorization: the full release sequence (tag/main/GitHub/PyPI/RTD) is
pre-authorized to execute ONCE, only when (1)–(3) verify green on the sealed
commit. No partial releases.

## Starting state (v1 baseline sealed as `82f30f0`; HEAD == origin/dev there,
verified at seal; re-verify with `git rev-parse HEAD` / `git rev-parse
origin/dev`)

- Sealed v1 tree: dark `_emit` + stills theming, 23-spec
  `scripts/generate_doc_page_atlases.py`, regenerated canonical (config_hash
  `4b0d96456d56bc1a` unchanged — style only) + three-area atlases, 22 new
  per-page atlas dirs, `jaxley_interop` panel, atlas-link edits in 8 guides
  + 25 tutorial pages, `jaxfne-developer` agent + pool row, v2 plan.
- B8 re-pin executed: canonical atlas is now 1000 ms / 2000 steps,
  `config_hash e701098092814baa` (duration is hash-covered — deliberate,
  gated). Per-page "pinned 200 ms" labels updated to the new pin.
- Untouched (Batch 1 remainder): `docs/guides/{jdna,output_bundles,
  configuration_grammar,model_inspection,atlas_suite,showcases}.md`,
  `docs/{quickstart,index,colab}.md`.
- Canonical atlas regenerated with dark figures: `config_hash` gate passed
  (`4b0d96456d56bc1a` unchanged — data identical, style only); manifest `sha256`
  changed as expected.
- `artifacts/todo_stack.md` itself is a working file: keep planned content,
  drop superseded lines as batches seal (done → remove).

## Standing facts workers must not re-decide

1. `dt_ms=0.1` float32 time grids drift non-uniform on long runs and are
   rejected by the atlas gate (`INVALID_TIME_GRID`); the generator uses
   `dt_ms=0.5` (binary-exact) wherever page runs used 0.1, and every run
   label says so.
2. `docs/_static/atlas/eeg_meg_emm_100/` was removed (byte-identical to
   `ei_population_100` — same dynamics, extra probe modes change no panel);
   tutorial 09 links the shared slug.
3. `Configuration` has no `.paradigm()` method; paradigms ride on
   `simulate(..., paradigm=...)`. The evoked tutorial was already fixed to
   the supported path.
4. The Jaxley bridge returns `Signals` without a `Model`, so no 6-panel
   atlas exists for it — only the standalone dark `vm` panel.
5. `docs/etudes/**` are frozen protocols: link-only or skip, never regenerate.

## Slug table (normative for every page edit)

`P` = `../_static/atlas/<slug>/` under `docs/tutorials/` and `docs/guides/`,
`_static/atlas/<slug>/` under `docs/` root. Regen = full command after `P`.

| Doc page | Slug (or share) | Run label (exact sentence) | Status |
|---|---|---|---|
| tutorials/01_single_neuron_multimodal.md | `single_neuron` | 100 ms, dt 0.1 ms, seed 0 | done |
| tutorials/02_two_neuron_ei.md | `two_neuron_ei` | 500 ms, dt 0.5 ms, seed 0; the page shows dt 0.1 ms | done |
| tutorials/03_network_100_ei.md | `network_100_ei` | 100 ms, dt 0.1 ms, seed 42 (page run) | done |
| tutorials/04_v1_column.md | `v1_column` | 1000 ms, dt 0.5 ms, seed 0; the page shows dt 0.1 ms | done |
| tutorials/05_v1_pfc_dual_column.md | `v1_pfc_dual` | single 1000 ms AAAB trial, dt 0.5 ms, no HDP trial-to-trial carryover; the page chains trials with carryover | done |
| tutorials/06_v036_100_neuron_ei_population.md | `ei_population_100` | 1000 ms, dt 0.5 ms, seed 42; the page shows dt 0.1 ms | done |
| tutorials/06_jaxfne_suite_no_1_computational_biophysics.md | `suite1_column` | smoke-scale 1000 ms, dt 0.5 ms, seed 44, of the notebook's 48-neuron column (notebook: 5000 ms, dt 0.1 ms) | done |
| tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md | `suite2_net1` | `suite2_net1_config`, 100 neurons, 1000 ms, dt 0.5 ms, seed 7; the page shows dt 0.1 ms | done |
| tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md | `evoked_l4` | evoked condition, 1500 ms, dt 0.5 ms, seed 7; the page shows dt 0.1 ms | done |
| tutorials/08_jaxfne_suite_no_3_low_frequency_scaling.md | `scale_100` | N=100 scale, 1000 ms, dt 0.5 ms, seed 2303, with the notebook's async patch (notebook: N=10/50/100/500, dt 0.1 ms) | done |
| tutorials/07_v037_source_bookkeeping.md | `source_column_48` | 48-neuron column, 1000 ms, dt 0.5 ms, seed 42; the page shows dt 0.1 ms | done |
| tutorials/08_v038_lfp_csd_readout.md | `lfp_csd_12` | 12-neuron laminar example, 1000 ms, dt 0.5 ms, seed 42; the page shows dt 0.1 ms | done |
| tutorials/09_v0310_eeg_meg_emm_proxy_bundle.md | share `ei_population_100` | same 100-neuron dynamics; EEG/MEG/EMM are additional probe modes on those signals | done |
| tutorials/10_v0313_omission_oddball.md | `omission_60` | plain drive, 1000 ms, dt 0.5 ms, seed 42; conditions declared via `omission_oddball_paradigm` | done |
| tutorials/11_multi_laminar_cortical_agsdr.md | `v1v4_80` | closest `Model` equivalent (`suite2_v1_v4_config`, 80/area, 1000 ms, dt 0.5 ms, seed 42); the page flow is `tutorial_utils` dict-based | done |
| tutorials/13_canonical_column_etude.md | `canonical_etude_1000` | uniform-drive run, 1000 ms, dt 0.5 ms, seed 0 (page run) | done |
| tutorials/01_define..08_compare + index primer | canonical `docs/_static/atlas/` | pinned 200 ms reference run (`generate_readme_atlas.py --html-only`) | done |
| guides/hdp.md | `hdp_1000` | short 200 ms HDP run, dt 0.5 ms, weight-trace recording off | done |
| guides/homeostasis.md | `homeostasis_1000` | 1000 ms, dt 0.5 ms, seed 0 (page run) | done |
| guides/calibration.md | `calibration_100` | minimal 200 ms completion, dt 0.5 ms, seed 0; the page simulate call is a placeholder | done |
| guides/tensor_field_workflows.md | `single_neuron` | single-neuron example scale | done |
| guides/probe_operators.md | `probe_32` | minimal 200 ms completion, 32 neurons, dt 0.5 ms, seed 0; the snippet is aspirational | done |
| guides/objective_grammar.md | `objective_60` | pre-tune run, 200 ms, dt 0.5 ms, seed 1, with the page's paradigm | done |
| guides/operator_composition.md | `operator_chain_40` | 40-neuron run, 100 ms, dt 0.5 ms, seed 3 (page run) | done |
| guides/jaxley_interop.md | `jaxley_interop/vm_dark.html` iframe | standalone dark `V_m` panel (no `Model`, no atlas) | done |
| guides/jdna.md | canonical `docs/_static/atlas/` | pinned 200 ms reference; the develop path matches §Example | TODO (Batch 1A) |
| guides/output_bundles.md | `probe_32` | illustrative 32-neuron run for bundle export | TODO (Batch 1A) |
| guides/configuration_grammar.md | `config_grammar_1000` | smoke-scale 200 ms run of the 1000-neuron head config (guide shows 1000 ms) | TODO (Batch 1A) |
| guides/model_inspection.md | none (toy example; cross-link atlas_suite) | fix `jaxfne.build(...)` → `jaxfne.construct(...)`; fix `build_atlas(..., path=)` → `out_dir=` | TODO (Batch 1A) |
| guides/atlas_suite.md | — | quickstart: `out_dir` → `docs/_static/atlas`, runtime 500.0 → 200.0 ms (pinned table) | TODO (Batch 1A) |
| guides/showcases.md | `atlas_three_area` (analogue) | 100/area reference for the page's 300/area hierarchy | TODO (Batch 1A) |
| quickstart.md | canonical `docs/_static/atlas/` | tensor path identical (`load_canonical_neuronal_tensor`) | TODO (Batch 1B) |
| index.md | canonical `docs/_static/atlas/` direct links | reference panels beside the three-area set | TODO (Batch 1B) |
| colab.md | `single_neuron`, `two_neuron_ei` | rewrite Cell 2/3 to the Configuration API (current `construct(config)` takes no `emitters=`; `IzhikevichEmitter` takes no `v_init`/`u_init`) + atlas links | TODO (Batch 1B) |

Skipped with reason (do not pick up): `12_izhikevich_*` (browser-only,
has iframe); `notebook_standard.md`, `tutorial_outputs.md` (process docs, no
runs); `api/*` (reference fragments); `etudes/*` (frozen); theory/appendix/
changelog/releases/migration/performance (no runnable sims).

## Snippet template (verbatim; fill `<RUN>`, `<P>`, `<REGEN>` from the table)

```markdown
## Interactive atlas (dark)

Dark-theme Plotly panels from <RUN>: [index](<P>index.html) · [3D](<P>network_3d.html) · [connectivity](<P>connectivity.html) · [raster](<P>raster.html) · [traces](<P>traces.html) · [spectral](<P>spectral.html) · [state summary](<P>state_summary.html).

Regenerate: `<REGEN>`.
```

Placement: append at page end as a new H2, except `13_canonical_column_etude`
(before `## Notes on scale and claims`), `07_add_dynamics` (HDP variant text,
done), `07_suite2` (net1 + pointer, done). If an assigned oldString does not
match, stop that item and report — never fuzzy-match (a prior pass mis-applied
one edit; every edit must be followed by reading the edited region).

## Batch A — authority & contracts (surface decisions, don't improvise)

Pinned TFNE doc subset (A1 sealed 2026-09-20; docs cite only these):
ordering S20/S20.1 (typed natural + `order[A]` override), frontiers
(`in`/`out`, `E_FRONTIER_UNRESOLVED`), rule bodies S12 (`$L`/`$R`,
`X[k]` selection), atomicity S14/S25 (contextual), mechanism resolution
S25 (`resolve_mechanism`; kinetics resolve at the tensor bridge),
replication, JDNA completion (TFNE→JDNA boundary, `K_D`, value origins),
CTX-01 first integrated model. Shown as limitations, never hidden: delay
refused (`E_PARAM_UNSUPPORTED`), declared geometry inert at execution
(pinned by `test_declared_geometry_does_not_reach_the_executed_positions`).
- A2. Atlas data contract (normative; a panel that cannot meet its contract
  fails/omits explicitly, never substitutes): H-SPICE schema ← realized
  TFNE/JDNA/model metadata (else fail/omit explicitly); 3D network ←
  executed geometry (qualify under PARAM-04); raster ← executed spikes
  (else fail); LFP ← declared source/field/probe output (never substitute
  population activity); H dynamics ← recorded H (omit if unrecorded); HDP ←
  actual mutable target/rule diagnostics (never infer from changing
  activity); oscillatory response ← executed observation over declared
  window (mechanism qualified separately).
- A3. Recording-budget policy (replaces neuron-count rules). Limiting
  object `M ∝ T(N_H + N_W + N_recorded)`. Classes: small mechanistic
  circuit → full H, HDP targets, representative W, diagnostics;
  medium/large network → population H, selected traces, summaries; large
  edge-plastic network → sparse/sample/aggregate plastic state, never full
  `T×E` unless explicitly budgeted. Every atlas declares class + budget.
- A4. Provenance manifest schema (emitted by `build_atlas`): TFNE digest,
  `K_D`, model/config identity, simulation identity, panel inputs,
  code/version; plus per-panel lineage `panel -> source artifact ->
  variable/path -> transform -> units/relative status -> sampling/window`.
  The gallery is an evidence index, not an image index.
- A5. Figure states `GENERATED ≠ VALIDATED ≠ CANONICAL`. Only VALIDATED
  enters galleries; only deliberately selected VALIDATED outputs become
  canonical doc assets. Promotion is a gated batch item, never a hand-edit.
- A6. Canonical teaching lifecycle `question → TFNE → JDNA → JaxFNE Model →
  Simulation → Observation`, plus the standing audit rule: audit every
  construction diagram and workflow for TFNE/JDNA/Model ownership (the old
  direct-hierarchy-into-`(s,h0,I)` boundary is not preserved).
- A7. `T_atlas = 1000 ms` is a visualization/example standard, not
  scientific identity: canonical general-purpose atlas examples use 1000 ms
  unless the scientific example declares another duration.
- A8. gh-pages publishing policy (user direction 2026-09-20; details are a
  human decision recorded here before B6/E3 execute). Measured v1 payload is
  ~21.2 MB committed HTML — accepted, forward cap only (v1 blobs stay in
  history; no rewrite). Precedent: `origin/gh-pages` already holds a
  deployed built site (v0.4.20 RC). Migration shape: generated atlas HTML
  leaves dev/main (gitignored; generator still reproduces it locally so
  `mkdocs serve` is intact); panels publish to `gh-pages` at versioned
  paths; docs link absolute URLs. Open sub-decisions: (a) regenerate at
  publish time (slow) vs publish-from-local-build; (b) RTD-stable story —
  recommendation is keep RTD for versioned text (release receipts cite RTD
  stable @ SHA; mid-0.4.x is not the time to rewire hosts) with panels
  resolving to versioned gh-pages URLs; (c) orphan-audit blindness to
  absolute URLs is covered by the E3 post-publish link check, not by
  pretending local build verifies them.
- A9. Freeze scope for 0.4.25 code touches (rules I-008): the v0.4.17-era
  frozen-use narrative is superseded for this sweep — kernel numerics stay
  frozen (REP-03 and equivalence bounds stand), while presentation-only vis
  changes and trajectory-preserving Batch F touches (gated bit-exact) are
  allowed. Anything altering trajectories is a 0.5.x item by default.

## Batch B — code (parallel after A; disjoint files)

- B1. Native Plotly H-SPICE panel (dark, interactive/vector, testable;
  theme presentation-only per standing rule).
- B2. Atlas 7-panel update in `atlas_suite` (hspice, network3d, raster, lfp,
  h_dynamics, hdp, oscillatory — each enforcing its A2 contract incl.
  explicit fail/omit paths) + tests per contract branch.
- B3. Provenance manifest emission (A4 schema) in `build_atlas`.
- B4. Figure-state field + promotion tooling (VALIDATED / CANONICAL behind
  Batch E gates).
- B5. Frozen-bundle no-resimulation gate: étude figure generation ⇒
  `Δsimulation = 0`; generators consume bundle dirs only; AST gate asserts
  no simulate/construct import path in étude figure scripts.
- B6. Gallery-from-manifests generator (lists only VALIDATED+ atlases with
  provenance; `gallery.md` becomes output, not source). Emits absolute
  gh-pages URLs per the A8 versioned layout (relative links only where the
  asset is committed alongside the page).
- B7. Semantic-negative docs gate for the 7 forbidden conflations (delay
  supported by TFNE; declared geometry == executed geometry; GABA ==
  GABA_A; H == HDP; proxy == physical LFP; source order == realization
  order; TFNE directly fills unspecified developmental choices), with an
  allowlist for doctrine pages stating the limitation itself.
- B8. Canonical 1000 ms re-pin (`generate_readme_atlas` EXPECTED +
  duration; deliberate, hash-gated).

## Batch F — code sweep (after B, before C; trajectory-preserving)

- F1. Hot-path complexity/order/flattening pass (simulate kernels, field
  projections, probe operators): linear-scan order, minimal temporaries,
  canonical `[T,N]`/`[T,X]` layouts; no semantic change.
- F2. JAX switches audit + doc: float32/64 via stock `jax_enable_x64`
  (central policy in `runtime.py`; close divergences); backend = stock JAX
  (`jax.devices`/default backend; cuda/cpu/parallel-cpu via official
  mechanisms; Apple Metal via external `jax-metal` plugin — documented, not
  coded). Official-doc conformance is the acceptance (user-authorized).
- F3. I-001 one-line fix (stale 259-symbols comment,
  `jaxfne/public_surface.py:118`) + I-006 lesson enforced (behavioral
  suites gate every lint-driven rename).
- F4. Verify: canonical `config_hash` gates + `run_test_gate.py dev` +
  targeted suites, bit-exact. Any numerical delta fails closed → the item
  moves to 0.5.x planning, sweep continues without it.
- Acceptance: Batch C regen runs on post-F code; docs describe post-F
  behavior.

## Batch C — regenerate (after B; record generator command per atlas)

- C1. Canonical + three-area (7-panel, `T_atlas` standard).
- C2. Small mechanistic circuits (full-recording class incl. an HDP
  showcase, e.g. MCC-3-style 10n).
- C3. Suite/tutorial set (reuse v1 slugs from the slug table).
- C4. Guide set (reuse v1 slugs).
- C5. Étude post-hoc set (frozen bundles only; B5 gate enforced) + Jaxley
  standalone panel as-is.
- C6. Promotion pass: mark regenerated outputs VALIDATED per A5
  (gate-checked), select canonical assets deliberately.

## Batch D — rewrite (parallel groups after A; may overlap C)

- D1. Lifecycle/ownership audit across every construction diagram and
  workflow page (A6 rule; not a wording pass).
- D2–D7. Archetype pass per group (`question → minimal algebra → run →
  result/figure → interpretation/limits → links`; dedupe config dumps —
  paste once, link): cumulative tutorials; archive/suite tutorials; chain
  guides; state guides (HDP/homeostasis); root pages
  (quickstart/index/colab); atlas/model_inspection/showcases/gallery pages.
- D8. Fold in the 9 remaining v1 page files (old Batch 1A/1B list) under the
  new archetype where not yet covered.
- D9. Simplification check: derivations live in doctrine; pages carry
  operational summaries + links (spot-check per group).

## Batch E — seal (after C + D)

- E1. Language + vocabulary + orphan audits green.
- E2. Semantic-negative gate (B7) green.
- E3. Atlas provenance/completeness validation: every manifest carries the
  A4 fields; gallery lists only VALIDATED with resolving links. Absolute
  gh-pages URLs are validated by a post-publish remote link check (its own
  step after deploy) — local `mkdocs build --strict` cannot verify them
  and must not be claimed to.
- E4. Gates dev → broad → rc + `mkdocs build --strict` (H2: no
  release-ready claim without the exact gate on the sealed state) + green
  remote CI on the exact sealed commit (I-005 standing rule).
- E5. Changelog/version/release prep per old Batch 4 items 1–4 (bump list,
  v0.4.25 entry, authority rollover draft, validators), then STOP +
  surface: tag, main merge, GitHub release, PyPI, RTD need explicit user
  authorization. On seal: remove completed items (done → remove; git keeps
  history).
- E6. Drain the problem stack (grounded 2026-09-20; entries preserved
  verbatim, resolutions appended, then Open section cleared): I-001 fix in
  F3; I-002/I-007 superseded (immutable receipts, v0.4.24 chain rules);
  I-003 superseded (`dist/` absent; clean-room build at release); I-004
  standing guidance → Batch D acceptance (observed values cite exact
  artifacts); I-005 standing rule → E4; I-006 lesson → F3 + developer
  traps; I-008 ruled in A9; I-009 verified-fixed (routes absent from
  `for_ai_agents.md`) → close with evidence; I-010 fix in Batch D
  (`contributing.md:34` → `artifacts/skills/`); I-011 superseded (scripts
  absent) → close; I-012 receipt refs → authority rollover note in E5;
  I-013 workbench index → banner-or-update in Batch D; I-014 scope
  documentation → E5. New problems found mid-sweep get IDs and drain here.
