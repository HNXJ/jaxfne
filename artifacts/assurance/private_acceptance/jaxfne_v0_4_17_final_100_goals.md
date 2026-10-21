# JaxFNE v0.4.17 --- Final 100/100 Goals

> **SUPERSEDED AS EVALUATION TARGET (2026-08-21).** The current project-goal
> source is the supplied TFNE submission project-source set v2, whose final
> acceptance list contains goals 1–100. This checkout-local copy enumerates
> goals 1–95 only and is retained verbatim as a historical snapshot of the
> seal-review instrument at the time of the PEC checkpoint; rows 96–100 must
> come from the authoritative v2 source, not from this file. Precedence rule
> unchanged: frozen executable/evidence truth > matching definitions >
> current private project goals.

## Purpose

This file is the final acceptance set for JaxFNE v0.4.17. It replaces
the earlier 69-goal list for the final seal review. Each goal is one
auditable sentence.

The target is not maximum feature count. The target is a small, fast,
mathematically coherent, biophysically explicit, fully documented,
cleanly packaged JaxFNE that can remain scientifically frozen for about
40 days while it is used heavily.

The public project should describe **JaxFNE itself**. Harness state,
agent operation, private publication work, plans, audits, release
deliberation, and temporary logs are private-like work and should be
isolated so they can be removed without affecting the package or public
documentation.

The acceptance equation is:

$$
100/100 = S_{science}+S_{code}+S_{tests}+S_{docs}+S_{public/private}+S_{package}+S_{seal}
$$

where every required term must pass; the sum is conceptual, not an
averaging rule. A P0/P1 failure in any required term blocks 100/100.

## Goals

1.  **TFNE equation set.** Define one compact TFNE equation set
    connecting emitter state, source, field, probe, objective, and
    optimization without duplicating the same scientific relation in
    separate subsystems.

2.  **Single executable path.** Provide one canonical public path from
    specification to construction, simulation, signals, analysis,
    optimization, visualization, and export, with compatibility paths
    clearly secondary.

3.  **Geometry as data.** Represent neuron, population, layer, area,
    source, probe, and sensor geometry explicitly and reuse the same
    geometry for connectivity, delays, fields, probes, figures, and
    saved results.

4.  **Canonical tensor meaning.** Define the minimal tensor data needed
    for areas, layers, neuron types, topology, geometry, base
    parameters, relative modifiers, and metadata without storing the
    same fact in competing forms.

5.  **Configuration and NeuronalTensor relation.** Document exactly how
    Configuration and NeuronalTensor relate to the executable model and
    avoid claiming a reversible mapping where information is
    intentionally lost.

6.  **JDNA build-time generation.** Keep JDNA/pseudogenome generation a
    deterministic build-time map into standard JaxFNE tensors for
    v0.4.17, with runtime H-triggered structural growth explicitly
    outside the supported release surface unless already fully
    validated.

7.  **Invariant and variant quantities.** Classify shared/invariant
    quantities separately from neuron-, synapse-, layer-, trial-, seed-,
    and time-varying quantities so shared values are stored once and
    variation remains compact.

8.  **Static, dynamic, plastic, structural types.** Classify model
    quantities as static, dynamic, plastic, or structural and state
    their update and persistence meaning explicitly.

9.  **Base-relative-effective rule.** Represent adaptive quantities by a
    physical/model base, a dimensionless relative value, and an explicit
    map to the effective value used numerically.

10. **Relative domains.** Give each relative quantity an admissible
    domain matching its geometry, including finite signed, sign-limited,
    one-sided unbounded, and fully unbounded cases.

11. **Reference state.** Use relative zero as the base/reference state
    wherever mathematically valid and test exact recovery of the base at
    that state.

12. **Physical units.** Keep physical/model units attached to bases and
    explicit calibration/effective maps so internal relative
    calculations remain dimensionless without losing dimensional
    interpretation.

13. **Physical time.** Keep dt, duration, delays, time constants,
    windows, and timestamps dimensional and forward-causal with t\[n+1\]
    \> t\[n\].

14. **H state.** Define H as a finite-dimensional relative
    hidden/biophysical state and never equate H with homeostasis by
    definition.

15. **H coordinate completeness.** Require each H coordinate to have a
    meaning, domain, update equation, timescale, reference/initial
    state, and map to relative parameter effects.

16. **RBD dynamics.** Express H-dependent dynamics through explicit
    equations that preserve causal ordering, supported bounds, and the
    declared base-relative-effective rule.

17. **H-to-gain map.** Map H into relative gains or offsets through
    explicit functions rather than silently overwriting shared static
    bases.

18. **Izhikevich base-plus-relative form.** Keep shared Izhikevich bases
    fixed and express neuron-specific or time-dependent phenotype
    through declared relative maps appropriate to each parameter.

19. **HDP integration.** Express HDP through the same H/state/relative
    grammar used by the rest of TFNE rather than through an isolated
    mechanism-specific representation.

20. **Adaptation.** Define adaptation as measurable history-dependent
    response change with explicit timescale, recovery, controls, and H
    dependence.

21. **Habituation.** Define habituation with repeated-stimulus,
    recovery, and control conditions that separate it from fatigue,
    instability, or normalization artifacts.

22. **Plasticity.** Define plasticity as persistent history-dependent
    parameter or connectivity change with explicit update rules, bounds,
    nulls, and persistence tests.

23. **Memory.** Define memory operationally through later response
    dependence on prior stimulation and distinguish H, synaptic, and
    structural contributions where supported.

24. **Full-state continuation.** Define the complete Markov state for
    each supported kernel and verify segmented continuation against
    uninterrupted simulation.

25. **Connectivity forms.** Support generalized dense, sparse, edge, or
    structured connectivity with equivalent scientific meaning where
    equivalence is claimed.

26. **Delay causality.** Represent delays in physical time and preserve
    deterministic causal propagation with clear links to topology and
    geometry.

27. **Source operator.** Define source construction with explicit mode,
    sign, support, gain, normalization, relative/calibrated status, and
    no silent double counting.

28. **Field operator.** Separate linear proxy projection from numerical
    field/PDE solving and record operator type, assumptions, geometry,
    and validation state independently.

29. **Probe operator.** Treat probe/sensor position, orientation,
    spacing, reference, and normalization as explicit scientific inputs
    rather than plotting details.

30. **Raster output.** Provide canonical spike/raster outputs with
    explicit axes, timing, population identity, and reproducible
    provenance.

31. **Activity and rate output.** Provide canonical activity/rate
    transforms with explicit windows, normalization, axes, and
    units/status.

32. **LFP-like output.** Provide LFP-like proxy output only through
    declared source/field/probe maps and never label uncalibrated
    relative output as physical LFP.

33. **CSD-like output.** Provide CSD-like output with explicit contact
    geometry, spatial derivative rule, boundary handling, normalization,
    and proxy/calibration state.

34. **PSD output.** Provide package-native PSD/spectral analysis with
    explicit sampling, windows, normalization, frequency range, and
    trial/seed aggregation.

35. **EEG-like and MEG-like output.** Provide EEG-like and MEG-like
    paths only with explicit geometry/source/field/probe meaning and
    preserved proxy/calibration state.

36. **Visualization from package data.** Make figures consume
    package-generated arrays and metadata without reimplementing
    scientific calculations in plotting code.

37. **Visualization truth.** Require figures to expose invalid,
    negative, or unresolved outcomes rather than hide them through
    clipping, smoothing, normalization, selection, or decorative
    replacement.

38. **Objective representation.** Represent objectives with component
    metrics, targets, conditions, null/control state, rejection reasons,
    and provenance rather than only a scalar loss.

39. **Optimizer-independent objectives.** Keep objectives independent of
    optimizer choice so AGSDR, GSDR, Adam, and other supported methods
    can use the same objective definition.

40. **AGSDR and GSDR.** Provide stable AGSDR/GSDR use with explicit
    search spaces, bounds, seeds, budgets, stopping conditions,
    diagnostics, and reproducible outputs.

41. **Adam and gradient methods.** Provide Adam or related gradient
    optimization only on declared differentiable paths with small
    deterministic gradient checks.

42. **Differentiability limits.** State exactly which
    emitter/source/field/probe/objective compositions are differentiable
    and how spike/reset discontinuities affect gradients.

43. **Shared-base computation.** Exploit shared bases plus relative
    tensors to reduce parameter duplication and make broadcasting,
    batching, serialization, and sensitivity analysis simpler.

44. **JAX execution.** Use pure JAX kernels, explicit PRNG keys,
    lax.scan, vmap, JIT, and sparse/edge forms where they measurably
    improve execution while preserving CPU-correct reference behavior.

45. **Performance evidence.** Measure runtime, memory, compile cost, and
    scaling for representative paths instead of asserting computational
    advantage without benchmarks.

46. **Generalized code.** Reduce the core toward fewer orthogonal
    functions that compose into more behavior while removing duplicated
    constants, transforms, special cases, and obsolete compatibility
    code.

47. **Short code without cleverness.** Reduce code size only through
    real abstraction or deletion and reject compression that harms
    readability, numerical transparency, tests, or auditability.

48. **Stable public API.** Give each public concept one canonical owner,
    keep compatibility adapters thin, keep optional dependencies lazy,
    and remove stable-looking dead or incomplete paths.

49. **Explicit failure.** Fail clearly on NaN/Inf, invalid domains,
    impossible geometry, unsupported calibration, continuation mismatch,
    solver failure, invalid objectives, and invalid public use.

50. **Serialization.** Roundtrip supported specifications, states,
    manifests, and results without losing geometry, domains,
    relative/calibrated state, or scientific meaning.

51. **Reproducibility.** Record sufficient version, SHA, configuration,
    tensor, geometry, bases, state, runtime, PRNG, analysis settings,
    and hashes to reproduce release-facing results.

52. **Fast invariant tests.** Maintain small mathematical tests for
    determinism, finite values, shapes, linearity where declared,
    zero-source behavior, equivalence, continuation, nulls, bounds,
    relative domains, mappings, gradients, and serialization.

53. **Ultrafast default test gate.** Make the default local gate
    complete enough for ordinary development yet fast enough for
    frequent use by removing redundant simulations, duplicate
    parameterizations, repeated builds, oversized fixtures, and
    unnecessary subprocesses.

54. **Tiered test execution.** Separate tiny smoke/invariant tests,
    normal fast tests, targeted integration tests, and slow/release
    tests so agents run the minimum sufficient set before the full seal
    gate.

55. **Test meaning preserved.** Any test-load reduction must preserve or
    strengthen the scientific/API conditions covered and must never
    loosen tolerances or delete independent evidence merely to reduce
    runtime.

56. **Quiet test output.** Keep successful test output compact and make
    failures report only the condition, expected value, actual value,
    and minimal reproduction needed for diagnosis.

57. **Truth labels.** Keep relative, normalized, effective, calibrated,
    physical, proxy, experimental, negative, positive, and unresolved
    meanings distinct in code, metadata, figures, and public
    documentation.

58. **Language checks.** Use automated public-language checks to detect
    semantic escalation and inconsistent terms while preserving real
    mathematical distinctions.

59. **Public package purity.** Remove harness, agent,
    publication-workflow, release-planning, private-review, and internal
    project-management concepts from the public JaxFNE package surface
    unless they are genuinely required runtime functionality.

60. **Public documentation purity.** Keep public docs about JaxFNE code,
    mathematics, biophysics, usage, outputs, limits, and examples rather
    than exposing agent harnesses, private publication work, internal
    plans, release deliberation, or temporary review machinery.

61. **Private work isolation.** Place harness state, agent notes,
    publication work, plans, audits, temporary receipts, and other
    private-like work under ignored/local storage or clearly private
    artifact areas so it can be removed without changing the package or
    public docs.

62. **No public harness vocabulary.** Ensure README, package docstrings,
    public API docs, tutorials, and public guides do not mention agent
    harness operation, CURRENT_TASK, internal auditor roles, private
    release state, or similar work machinery.

63. **No public publication machinery.** Ensure publication-specific
    generators, claim maps, manuscript reconstruction code, and private
    figure-review machinery are not exposed as ordinary public JaxFNE
    APIs or public user documentation unless intentionally generalized
    into a real package function.

64. **Artifact separation.** Separate public scientific examples/results
    from private logs, audits, release receipts, publication evidence,
    and agent state with clear storage and packaging rules.

65. **Distribution hygiene.** Ensure wheel and sdist contain only
    intended package/public assets and exclude private harness, agent,
    publication-plan, scratch, receipt, cache, and local-work files.

66. **Every public doc page current.** Review every page included in the
    public documentation build against v0.4.17 code and remove, update,
    merge, or explicitly retire any stale page.

67. **Every public doc symbol valid.** Verify every public documentation
    code symbol, signature, import path, configuration field, and
    command against the actual v0.4.17 public API.

68. **Every public doc equation valid.** Verify every public equation
    and mathematical statement against the implemented v0.4.17 meaning
    and current scientific rules.

69. **Every public doc example executable.** Run or mechanically
    validate every public code example that is intended to execute and
    ensure outputs/labels match current API and meaning.

70. **Every public doc link valid.** Build docs strictly with zero
    broken internal links, missing assets, stale navigation entries, or
    orphaned release-facing pages.

71. **Docs concise and dry.** Rewrite public docs toward direct
    descriptions of code, mathematics, biophysics, inputs, outputs,
    units, limits, and minimal use with little repeated or historical
    prose.

72. **Docs vocabulary small.** Use a small stable vocabulary with one
    preferred simple word per concept and avoid unnecessary process/meta
    terms when a direct mathematical word works.

73. **README minimal.** Keep README as a compact entry point containing
    what JaxFNE is, installation, minimal use, main scientific ideas,
    documentation links, and a small visual section.

74. **README visual evidence.** Use a small set of real
    package-generated figures near the README bottom to show
    geometry/circuit, activity, signals/readouts, and
    short/long-timescale behavior without turning README into a gallery.

75. **Documentation figures.** Populate docs with package-generated
    figures for circuits, geometry, raster/activity, H trajectories,
    gains, weights, source/readout signals, PSD, adaptation, recovery,
    and supported optimization paths.

76. **Multiscale figures.** Show short- and long-timescale behavior
    explicitly where relevant so millisecond neural dynamics and slower
    H/plastic changes are visually distinguishable.

77. **Figure reproducibility.** Generate documentation figures from
    reproducible scripts or notebooks with declared seed, runtime,
    analysis settings, and source data/status.

78. **Tutorials.** Keep tutorials thin, executable, deterministic,
    API-native, quantitatively checked, and free of notebook-local
    copies of reusable scientific engines.

79. **Etudes.** Keep a small number of complete scientific Etudes
    connecting question, equations, configuration, simulation,
    controls/nulls, analysis, figures, interpretation, and reproducible
    outputs.

80. **Independent workbench.** Allow a user to construct, simulate,
    continue, analyze, visualize, optimize, export, restore, and
    reproduce intended v0.4.17 work entirely through supported public
    interfaces.

81. **Forty-day core freeze.** After release, keep the v0.4.17
    scientific core and numerical meaning unchanged for approximately 40
    days while using it as a stable scientific tool.

82. **Freeze-period editable areas.** During the 40-day period allow
    documentation, documentation figures, private publication work,
    skills, simulations, and new artifacts/logs to evolve without
    changing core scientific behavior.

83. **Freeze-period issue logging.** During the freeze route core bugs,
    friction, performance limits, scientific questions, and ideas
    through observe -\> reproduce -\> log rather than immediate core
    repair.

84. **Release identity separation.** Keep C_core, C_release, C_receipt,
    and current repository head as distinct identities so evidence
    commits and documentation changes cannot be confused with the
    packaged scientific source.

85. **Release package identity.** Make the v0.4.17 tag, GitHub release,
    and PyPI source/artifacts identify the same intended release source
    and record exact artifact hashes.

86. **Clean-room package test.** Install wheel and sdist in clean
    environments and verify import, canonical construction/simulation
    smoke paths, manifest/serialization behavior, and absence of
    private-file leaks.

87. **Remote CI green.** Require the exact final release source to
    satisfy the required remote CI/release checks rather than relying
    only on local tests or nearby commits.

88. **Independent final seal.** Require a fresh read-only auditor to
    reconstruct Git state, inspect the immutable candidate, reproduce
    critical evidence, and return GO/NO-GO without repairing its own
    candidate.

89. **No self-awarded 100.** Do not accept 100/100 from the agent that
    made the final changes unless an independent fresh review verifies
    every release-required condition on the exact immutable state.

90. **Public/private removal test.** Verify that deleting private/local
    harness, publication, plan, audit, and temporary artifact areas
    would not break JaxFNE import, public docs, supported examples,
    tests required by users, or packaged runtime behavior.

91. **Public tree understandable alone.** Ensure a new user can
    understand the public repository/package from README, public docs,
    examples, and API references without needing private agent notes,
    publication drafts, or release logs.

92. **No stale public pages.** Require a page-by-page inventory showing
    every public documentation page as KEEP, UPDATE, MERGE, or REMOVE,
    with no unreviewed page remaining at seal.

93. **No stale public code paths.** Require a public-surface inventory
    showing every exported or documented callable as supported,
    compatibility-only, experimental, or removed, with no ambiguous
    stable-looking path.

94. **No unnecessary repository weight.** Remove generated caches,
    obsolete reports, redundant artifacts, duplicate docs, and dead
    scripts from the public/release tree when they provide no current
    scientific or user value.

95. **Final 100/100 state.** Declare v0.4.17 complete only when
    scientific meaning, code, fast and release tests, public API, every
    public doc page, figures, package contents, remote CI, release
    identities, and independent seal all agree with no known correctable
    release defect.

## Required final audit

For every goal, return exactly one of `PASS`, `PARTIAL`, `FAIL`, or
`DEFER`, with direct file/test/command evidence; `DEFER` is acceptable
only for work explicitly outside the supported v0.4.17 surface and must
not leave a misleading public API or document behind.

Classify every non-PASS item as `P0`, `P1`, or `P2`; P0/P1 blocks the
final seal, while P2 may remain only when it does not contradict the
stated v0.4.17 public meaning or 40-day use goal.

Before scoring, audit the public/private boundary explicitly: inspect
package exports, `jaxfne/`, README, MkDocs navigation, public docs,
examples, tutorials, wheel, and sdist for
harness/publication/plan/audit/release-process leakage. Private-like
work may remain locally or under private artifact areas, but public
JaxFNE must not depend on it.

Before scoring documentation, produce a page inventory for every page in
the public MkDocs build and mark it `KEEP`, `UPDATE`, `MERGE`, or
`REMOVE`; 100/100 requires zero unreviewed public pages and all kept
pages to match v0.4.17 code, equations, units/status, and examples.

Before scoring tests, report the default fast-gate runtime, full
non-slow runtime, slow/release runtime, the dominant expensive tests,
and why each remaining expensive test provides independent evidence;
test reduction must preserve scientific/API meaning.

Before scoring release state, distinguish `C_core`, `C_release`,
`C_receipt`, and `C_head`, verify remote state after fetch, and prove
the tag/GitHub/PyPI identity from the exact release source and artifact
hashes.

## Final definition of 100/100

100/100 means there is no known correctable release defect; the
scientific core is coherent and frozen; the default tests are fast
enough for frequent use while the full seal tests remain complete; every
public documentation page is current; public package/docs contain no
unnecessary harness, private publication, planning, audit, or temporary
release machinery; private-like work can be removed without breaking
public JaxFNE; wheel/sdist are clean; the exact release source is green
remotely; and a fresh independent seal agent returns `SEAL_GO` without
needing a repair.
