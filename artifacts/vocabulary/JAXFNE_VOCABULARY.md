# JaxFNE canonical vocabulary

Status: **ACTIVE** project source. Audience: humans and agents writing JaxFNE
prose. External reference for standard biophysics terms: BMTK/SONATA and Jaxley
glossaries — normalize to established computational-neuroscience usage; do not
copy those glossaries wholesale.

Target: **~70 controlled terms** + ordinary scientific English elsewhere.

Deterministic audit: `python scripts/audit_vocabulary.py --check` (high-confidence
drift only). Semantic review: `artifacts/subagents/vocabulary_critic.md`.

**Public entry pages** (`README.md`, `docs/index.md`): minimum words; no named-tool
comparisons; state what JaxFNE is, why flexibility matters, and what users can
change and measure. See `artifacts/context.md`.

---

## 1. Canonical terms

| Canonical term | Meaning | Use instead of | Scope / rule |
| --- | --- | --- | --- |
| model | A specified neural/biophysical system | framework, architecture when generic | Default general term |
| model specification | Description used to construct a model | grammar, schema when generic | `JDNA` is a specific specification form |
| model resolution | Level of biological detail | abstraction level | Standard modeling term |
| cell | A modeled neural unit in the realized network | neuron when generic prose suffices | Use `neuron` when biology is explicit |
| compartment | A spatial subdivision of a cell (e.g. Jaxley) | segment when mechanism-specific | Interop / multicompartment contexts |
| population | A grouped set of cells sharing metadata | ensemble when JaxFNE population API intended | Typed population in `NeuronalTensor` |
| state variable | Time-dependent modeled quantity | coordinate when unnecessary | Standard mathematical term |
| neural state \(X\) | Fast neural / membrane / spike state | activity state, emitter state | JaxFNE mathematical term |
| biophysical state \(H\) | Additional time-dependent biological state | hidden state, homeostatic state | May include ions, energy, resources, synaptic state |
| parameter \(W\) | Fixed or mutable model parameter | weight state when generic | State mutability explicit in HDP |
| RBS | Representation of \(H\) | expanded synonyms | Define once; link to HDP guide |
| RBD | Dynamics of \(H\): \(\dot H = F_H(\cdot)\) | H dynamics grammar | Define once |
| HDP | State-dependent parameter dynamics: \(\dot W = F_W(\cdot)\) | plasticity framework | Distinct from \(H\) itself |
| plasticity | Activity/state-dependent **parameter** change | adaptation when parameter change intended | HDP family |
| adaptation | Change in neural/biophysical **state** or dynamics | attenuation | Distinct from plasticity |
| geometry | Spatial position, extent, orientation | spatial grammar | Standard term |
| structure | Network / topological organization | structural grammar | Includes realized connectivity |
| connection | Directed neural relation (scientific) | edge when biology intended | General scientific prose |
| edge | Stored graph connection record | connection when storage intended | Computational representation |
| synapse | Transmission mechanism at a connection | edge | Use when mechanism matters |
| source | Quantity generating a field/readout (\(Q\)) | source operator when unnecessary | Mathematical: \(Q = F_Q(\cdot)\) |
| field | Spatial quantity derived from sources (\(\Phi\)) | field operator when unnecessary | Mathematical: \(\Phi = F_\Phi(Q,G)\) |
| probe | Defined measurement / readout operator | observation operator | JaxFNE API concept |
| observation | Result being measured or compared | observable when unnecessary | General prose |
| simulation | Numerical evolution of a model | execution when generic | Standard term |
| recording | State or output retained during simulation | capture | Standard term |
| realized network | Constructed cells and connections | realized topology when broader unnecessary | Inspectable `Model` |
| configured | Requested in specification | declared | Only when distinction matters |
| realized | Present after construction | instantiated when unnecessary | vs configured |
| executed | Numerically evaluated during simulation | run-time realized | vs configured/realized |
| effective | Demonstrably changes a target result under intervention | causal, effectual | **Only with evidence** |
| relative | Defined relative to a reference within the model | normalized when not mathematically equivalent | Default output status |
| calibrated | Mapped to declared physical units/reference | physical when insufficient | Requires explicit receipt |
| proxy | Approximate / readout quantity without calibration | physical measurement | Never silently equate |
| JDNA | JaxFNE model/development specification (`PseudoGenome` → `develop`) | pseudo-genome in formal docs | Explain pseudo-genomic idea separately |
| development | Change in model structure or specification | evolution when generic | Runtime structural change: **planned** |
| perturbation | Controlled change to model or input | intervention when generic | System-ID / mechanism studies |
| parameter fitting | Estimation of parameters from data | calibration unless calibration intended | Distinct from output calibration |
| model reduction | Removing/compressing detail under stated criteria | simplification | Explicit criteria required |
| continuation | Resuming simulation from complete required state | restart when semantics differ | JaxFNE-specific |
| checkpoint | Persisted simulation/model state | receipt | Distinct from manifest receipt |
| test | Executable correctness check | gate when generic | Literal gate names unchanged |
| required test | Test that must pass for a task/release | gate in prose | Named gates: `run_test_gate.py` |
| evidence | Observed support for a claim | truth, receipt | Prefer over "truth" in prose |
| result | Output of test or analysis | truth | |
| limitation | Known supported boundary | exclusion, negative capability | |
| public API | Supported user-facing software interface | contract in prose | Frozen artifact may say "contract" |
| pipeline | Ordered computational steps | grammar when generic | e.g. source → field → probe |
| rule | Required invariant or condition | doctrine, governance | |
| membrane potential | Transmembrane voltage \(V_m\) | voltage when ambiguous | Standard biophysics |
| membrane current | Current across membrane | | Standard biophysics |
| conductance | Membrane or synaptic conductance | | Standard biophysics |
| reversal potential | Equilibrium potential for an ion/channel | | Standard biophysics |
| time step | Integration step \(\Delta t\) | dt when code context | |
| simulation duration | Modeled biological time span | | Distinct from wall-clock runtime |
| wall-clock runtime | Elapsed real time | | Distinct from simulation duration |
| parameter sharing | One stored value referenced by many edges/cells | | W10/W11 representation topic |
| connection probability | Probability of creating a connection in a rule | p_connect in prose when API named | |
| presynaptic / postsynaptic | Source / target side of a synapse | pre/post in prose when API named | |
| manifest | Structured provenance record for a run or artifact | | Distinct from checkpoint |
| mechanism | Named synaptic or biophysical process | | Keep when scientifically precise |
| delay | Finite transmission or history delay | wave | Distinct from propagating waves |
| attenuation | Decrease in amplitude (e.g. signal) | adaptation | Distinct from adaptation |
| emitter | Neural dynamics implementation at a cell | | JaxFNE API role |
| signals | Bundled simulation outputs (`Signals`) | | API type name — preserve |

---

## 2. Controlled words (valid but rare)

| Term | Use only when |
| --- | --- |
| grammar | Referring to a **formally defined** compositional syntax (e.g. configuration grammar doc title) |
| protocol | A **defined** experimental or test protocol (`Protocol D`, `protocol_h`, file paths) |
| contract | A **literal** software/API compatibility artifact (`public_surface_contract`, continuation contract table) |
| gate | A **literal** named release/test gate or technically necessary gate semantics |
| truth | Discussing epistemology specifically; otherwise use evidence, state, result, verified behavior |
| architecture | Actual software/system architecture (e.g. BMTK build/simulate split) |
| framework | Referring to an **established external** framework (JAX, Jaxley), not as filler for "model" |
| ontology | A genuine formal ontology or fixed metadata taxonomy |
| doctrine | Repository-internal doc titles only; **avoid in public nav prose** |
| coordinate | Mathematical coordinate or state coordinate specifically |
| operator | A mathematical map/operator specifically (\(F_X\), probe operator) |
| schema | Data layout or JSON schema specifically |
| invariant | Mathematical or test invariant specifically |

---

## 3. Strict distinctions (never silently equate)

| Never silently equate |
| --- |
| \(H\) = HDP |
| state = parameter |
| source = field |
| field = probe |
| proxy = calibrated measurement |
| relative = calibrated |
| configured = realized |
| realized = executed |
| executed = effective |
| connection = synapse = edge (context-dependent; see §1) |
| delay = wave |
| attenuation = adaptation |
| biophysical state = structural development |
| model specification = realized model |
| plasticity = adaptation |
| simulation duration = wall-clock runtime |

---

## 4. Audit procedure (semantic)

1. Read this file.
2. For each relevant noncanonical occurrence: classify meaning.
3. Replace **only** if a canonical term preserves meaning.
4. Preserve: API identifiers, equations, citations, historical changelog text, technically precise uses in §2.
5. Report `UNCERTAIN` instead of guessing.
6. **Never** blind global find-and-replace.

Classification labels: `REPLACE` | `TECHNICALLY_REQUIRED` | `API_IDENTIFIER` |
`MATHEMATICAL_TERM` | `HISTORICAL` | `CITATION` | `UNCERTAIN`.
