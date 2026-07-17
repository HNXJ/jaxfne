---
name: jaxfne-objective-grammar
description: >-
  The mandatory top-level object-transform grammar for jaxfne: the verified
  public chain Configuration/NeuronalTensor → construct → simulate → Signals →
  probe/objective/tune → manifest. Use first when structuring any jaxfne
  script, notebook, pipeline, or new API to pick the right stage, then route to
  the detailed skill for that stage (jaxfne-config, jaxfne-neural-tensor,
  jaxfne-neural-network, jaxfne-vis-modules). Flags invented
  signals.rate()/jtfne.optimize()/jtfne.weld() patterns before they reach code.
---

# jaxfne Objective Grammar

USE FIRST: `catalog-glossary-jaxfne`, then the detailed skill for your stage.

## Mandatory chain (verified public surface)

```text
Configuration/NeuronalTensor → construct → simulate → Signals → (vis | probe | objective | manifest)
                                                              ↘ Model.tune → TuneResult
```

Legacy aliases still import: `Configuration`/`Config`, `Model`/`Net`.

```python
import jaxfne as jtfne
jtfne.enable_x64()  # before array construction if x64 needed
```

## Route by stage

| Stage | Skill |
|-------|-------|
| Declaring static circuit structure via the fluent `Configuration` builder | `jaxfne-config` |
| Declaring static circuit structure via `NeuronalTensor` (Areas × Layers × NeuronTypes), or enabling HDP | `jaxfne-neural-tensor` |
| `construct → simulate → Signals`, reading signals, probe/objective/tune, manifest/receipt | `jaxfne-neural-network` |
| Any plotting/figure work on a `Signals` object | `jaxfne-vis-modules` |
| Deep dataclass/truth-gate schema reference | `jaxfne-modeling-optimization-schema` |

## Violations (rewrite if you see these)

- Hand-rolled PSD/raster when `jtfne.vis.*` or `tutorial_utils` pipeline exists
- `signals.rate()` / `signals.probe()` / `jtfne.optimize()` / `jtfne.weld()` invented API
- Global `cell_types=` for laminar E:I gradient (use `.area_layer_cell_types`, see `jaxfne-config`)
- Skipping manifest/receipt on release-facing runs without explicit reason

## Related

- Multi-trial spectrolaminar: `tutorial_utils` path (`catalog-glossary-jaxfne` §2)
- Open contradictions: `skills/FRICTIONS_STACK.md`
