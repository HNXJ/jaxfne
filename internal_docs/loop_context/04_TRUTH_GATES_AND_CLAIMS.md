# 04_TRUTH_GATES_AND_CLAIMS

Default status gates: `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`, and `physical_amplitude_claim_allowed=False`. Technical report context states physical amplitude/sensor-level/mechanism claims require calibration, geometry, solver evidence, null tests, and empirical comparison (`2026_jtfne_arxiv.txt:L89`).

## Gate/readout evidence from ZIP grep

|path|line|text|
|---|---|---|


## Claim-language contract

| Claim class | Allowed wording | Forbidden without evidence | Evidence required |
|---|---|---|---|
| Field | `laminar proxy`, `proxy transform`, `field/status metadata` | solved PDE / Maxwell / Poisson field | solver, geometry, boundary, gauge, residual, convergence |
| EEG/MEG | `EEG-like`, `MEG-like`, proxy projection | real sensor-level EEG/MEG simulator | source orientation, head/volume model, leadfield, sensor geometry, units, validation |
| Amplitude | native/proxy amplitude | calibrated amperes/volts/tesla | source calibration, units, physical field/probe metadata |
| EMM | normalized EMM-proxy | biological metabolism / stress-energy proof | calibrated currents/fields/conductivity/geometry/energy accounting |
| Objective | computational criterion | mechanism proof | nulls, ablations, repeated seeds, perturbation/model comparison, empirical comparison |
