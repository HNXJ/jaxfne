---
name: jaxfne-model
description: Choose and declare the neuron model (emitter, cell types, drives) through a Configuration or preset.
metadata:
  audience: agents
---
# jaxfne model

## WHEN
Picking the single-neuron dynamics, cell-type mix or drive for a circuit.

## AUTHORITIES
1. TFNE theory: `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md`.
2. Live code: `jaxfne/_config.py` (`Configuration.set_emitter`, `cell_types`,
   `drive(baseline_drive_by_cell_type=...)`; `cell_type_drives` is refused), presets in `jaxfne/_construct_presets.py`.

## RULES
- Dynamics are native and uncalibrated unless a calibration transform is declared;
  never report emitter voltages as physical units.
- Use a named preset (for example `suite2_single_neuron_config`) or a declared
  `Configuration`; never a notebook-local engine.
- A preset's defaults (probes, drives) are part of the model: read them, do not assume.

## STEPS
1. Start from a preset or `jaxfne.Configuration()`; set `set_emitter(family, preset)`.
2. Declare `cell_types({...})` fractions and drives.
3. `jaxfne.construct(cfg)`; read `model.neuron_table()`.
4. Route runs to `jaxfne-simulate`, readouts to `jaxfne-inspect`.

## STOP
- Unknown emitter family or preset; `ValueError` from `cell_types` (empty, negative or
  zero-mass fractions).

## VERIFY
- `jaxfne.agent.inspect(model)["emitter"]` and neuron counts match the declaration.

## DONE
- Constructed model whose emitter, cell types and counts match what was declared.
