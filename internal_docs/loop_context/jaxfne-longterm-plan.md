<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# JAXFNE Long-Term Plan

## Thesis

`jaxfne` stays compact: a JAX-native bridge from emitters to source/field/probe readouts, objectives, optimization, visualization, and evidence reports.

## Version-line roles

| Line | Role | Entry gate |
|---|---|---|
| v0.3.x | tutorial atlas, Etudes, proxy readout hardening, evidence stack | ED9/ED10, full manifests, strict JSON, clean install smoke |
| v0.4.x | experimental physical field solvers | stable source schema, field schema, boundary/gauge doctrine, residual tests |
| v0.5.x+ | external comparison, calibration workflows, inverse modeling, uncertainty | external reference/empirical comparison, held-out tests, uncertainty reports |

## Evidence ladder

```text
mathematical consistency
-> proxy/operator contract tests
-> electromagnetic admissibility metadata
-> numerical convergence/residuals
-> external-tool or empirical comparison
-> mechanism support through perturbation/model comparison
```

## v0.3 publication closure

Before a high-journal package paper is submitted:

- 8/8 main figures regenerated from scripts.
- ED1-ED10 present with scripts, PNGs, manifests, and receipts.
- All `outputs/evidence/*` regenerated from a clean live checkout.
- `evidence_checklist.json` and `inventory.json` strict JSON pass.
- All root import/optional dependency gates pass in a clean venv.
- Notebook smoke/full receipts exist or are explicitly out of scope.
- No proxy readout is described as physical measurement.
- Release/archive status is exact: tag, SHA, wheel/sdist, archive/DOI, or explicitly pending.

## Solver entry criteria

Open v0.4 solver implementation only after v0.3.x has:

```text
stable source schema
stable field metadata schema
boundary/gauge doctrine
source conservation tests
manifest validators
proxy-vs-solver API separation
notebook evidence receipts
```

## Likely consolidation areas

```text
metric registry shared by objectives and tutorials
JSON-safe export helpers
truth-gate validation helpers
visualization input coercion
optimizer report schemas
probe/readout contract schemas
source calibration reports
```

## Ecosystem position

- Jaxley: detailed differentiable cell/compartment modeling.
- PyNWB: optional export/archive format with schema and provenance.
- LFPy and EEG/MEG tools: detailed external forward modeling.
- `jaxfne`: compact composition, source/field/probe scaffolding, tutorial evidence, optimizer workflows, and reproducible manifests.

## Future physical solver namespace

Use experimental namespace until P3/P4/P5 evidence exists:

```python
jtfne.fields.experimental.solve_volume_conductor_smoke(...)
```

Do not promote a physical solver to stable API until boundary, gauge, residual, convergence, units, calibration, and validation reports are part of the manifest.
