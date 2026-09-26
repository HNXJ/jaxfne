---
name: jaxfne-fields
description: Declare field domains and probes and read source/field proxies at their epistemic level.
metadata:
  audience: agents
---
# jaxfne fields

## WHEN
LFP, CSD, phi_e or source readouts; electrode/contact declarations.

## AUTHORITIES
1. Visualization and observation rules: `artifacts/project_sources/3_jaxfne_visualization_rules.md`.
2. Live code: `Configuration.field`, `Configuration.probe`, `Configuration.set_probes`,
   the declared list `Configuration.probes` (`jaxfne/_config.py`); `jaxfne/fields/` (`project_laminar_sources`).

## RULES
- Every field readout is a RELATIVE_PROXY; no calibration transform exists.
- Source, field, probe and calibration are distinct; a projection is not a measurement.
- A readout that was not recorded is refused, never synthesized from V_m or spikes.
- Presets may declare probes by default; check before adding a second probe.

## STEPS
1. Read `cfg.probes` (presets may already declare one); then `cfg.field(...)` and
   `cfg.probe(...)` or `cfg.set_probes(modes, ...)`, keywords as in live presets.
2. Simulate (`jaxfne-simulate`).
3. `jaxfne.agent.observe(run, "lfp")` returns values plus level `RELATIVE_PROXY`.

## STOP
- `observe` raises "not recorded"; a request to report proxy amplitudes as physical units.

## VERIFY
- `jaxfne.agent.inspect(run)["recorded"]` lists the readout; values finite.

## DONE
- Readout with its level and contact count, from a declared probe.
