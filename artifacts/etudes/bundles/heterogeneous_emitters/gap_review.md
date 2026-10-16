# Gap review — Heterogeneous emitters Etude

- Gates: `{"A": true, "B": true, "C": true, "D": true, "E": true}`
- Rates (Hz): `{"izh": 25.80000114440918, "hei": 1.0}`
- N: `{"izh": 10, "hei": 2}`
- Similar-rate / different-Q control (null if False): `False`
- v0415b revision: finite/stable declared Q, not figure appearance.

| item | class | note |
|------|-------|------|
| Two mature distinct F_X (Izhikevich vs homeostatic_ei) | NO_GAP | Smallest implemented pair. |
| LIF/GLIF unused | NO_GAP | Placeholders; not silent Izhikevich substitutes. |
| HEI Model.simulate ignores StimulusSchedule | ANALYSIS_GAP | Live Model.simulate docstring is the Izhikevich vertical slice; HEI returns before paradigm resolution. Supported HEI drive is simulate_homeostatic_ei(..., drive_schedule=). docs/api/core.md overgeneralizes injection without an HEI caveat (closure docs, not this étude). Silent ignore is an interface limitation, not CORRECTNESS_DEFECT of the Izhikevich-scoped method. |
| Jaxley omitted | NO_GAP | Not a capability claim; deferred to avoid an integration étude. |
| AGSDR omitted | NO_GAP | Not a capability claim; would need a cross-family observable contract. |
| Family-native U scales | NO_GAP | Common shape, not calibrated current equivalence. |
| Q not physically equated | NO_GAP | Declared source semantics preserved. |
| HEI n=10 minimal extra-drive divergence | NO_GAP | Preserved as failed v0415; not a missing observation operator. |

