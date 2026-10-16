# Gap review — Heterogeneous emitters Etude

- Gates: `{"A": true, "B": false, "C": true, "D": true, "E": false}`
- Rates (Hz): `{"izh": 25.80000114440918, "hei": 8.300000190734863}`
- Similar-rate / different-Q: `False`

| item | class | note |
|------|-------|------|
| Two mature distinct F_X (Izhikevich vs homeostatic_ei) | NO_GAP | Smallest implemented pair. |
| LIF/GLIF unused | NO_GAP | Placeholders; not silent Izhikevich substitutes. |
| Jaxley not a primary row | PHYSICAL_MODEL_GAP | Deferred; distinct stimulus API and source_mode. |
| HEI Model.simulate ignores StimulusSchedule | ANALYSIS_GAP | Kernel already accepts drive_schedule; unifying Model.simulate is not required for composition. |
| Family-native U scales | NO_GAP | Common shape, not calibrated current equivalence. |
| Q not physically equated | NO_GAP | Declared source semantics preserved. |
| AGSDR omitted | NO_GAP | Would need a cross-family observable contract. |
| GENERAL_OPERATOR_GAP | NO_GAP | Empty; no package mutation. |

