# HDP + jaxfne Unified Publication Plan

## Title (decided 2026-08-04)

**Primary:**
> Omission Responses in a Multi-Area Laminar Cortical Hierarchy Emerge from Simple Firing-Rate Homeostasis

**Alternative (jaxfne-explicit):**
> Omission Responses from Firing-Rate Homeostasis in a Laminar Cortical Hierarchy Simulated with jaxfne

---

## Primary Story

We implemented a multi-area dense laminar cortical hierarchy (TFNE-Izhikevich) in jaxfne.
We simulated the exact omission paradigm used in our in-vivo recordings.
We observed that omission responses emerge from a single slow homeostatic state variable H(t) —
no prediction-error circuitry, no short-term synaptic depression, no additional mechanisms required.

The omission response character (sparse/dense, strong/weak, feedforward/feedback,
laminar profile) is an **empirical result** of the simulation — not a presupposition.
It does not change the plan structure.

---

## Secondary Story

Present **jaxfne** as a standard, reproducible open package for emitter→source→field biophysics.
Every result is produced by the same operator grammar and exported as deterministic artifacts
(manifest.json, metrics.json, PNG figures, validation_report.json).

---

## H(t) Formulation (decided 2026-08-04)

The specific H(t) used in this paper is the minimal slow firing-rate tracking variable:

    τ_H · dH/dt = r_target - r(t)

where:
  r(t)       = instantaneous population firing rate
  r_target   = homeostatic set-point (free parameter, matched to in-vivo baseline)
  τ_H        = slow timescale (seconds to tens of seconds; to be calibrated against data)
  H(t)       = gates intrinsic excitability as additive offset on Izhikevich threshold b
               OR multiplicative gain on input current (ablation distinguishes these two modes)

HDP is general — H(t,x,y,z) extensions remain downstream options.
This paper claims this specific minimal H(t) is *sufficient*, not that it is the only form.

---

## Anchoring References (decided 2026-08-04)

| Reference | Role in paper |
|---|---|
| Turrigiano & Nelson, Nat Rev Neurosci 2004 | Foundational: neurons regulate firing toward a set-point via intrinsic/synaptic homeostasis |
| Cannon & Miller, PLOS Comp Biol 2016 | Formal closest match to H(t) equation (slow threshold/gain variable, τ_θ ~30 s) |
| Abbott & Varela, Science 1997 | Fast-mechanism foil: SSA via synaptic depression (mechanistically distinct from H(t)) |
| Yaron et al., Neuron 2025 | Empirical target: neurons encoding negative prediction errors for omissions |

---

## Central Scientific Question

Does a single slow homeostatic variable H(t) — tracking population firing rate toward
a set-point — suffice to produce the omission response observed in multi-area laminar
in-vivo recordings, without predictive coding circuits or synaptic depression?

Secondary: Does H(t) also explain short-term adaptation, long-term adaptation,
local oddball, global-local oddball, recovery dynamics, and generalize across
neuron models (Izhikevich, LIF, HH/Jaxley)?

---

## Claim Level

Consistent with jaxfne documented status:
  claim_level: computational_scaffold
  field_claim_level: proxy_readout
  physical_amplitude_calibrated: false

All field outputs are Relative unless an explicit calibration step is documented.

---

## Repository Positioning

Consistent with jaxfne emitter→source→field→probe operator grammar.
No new APIs created for this paper without first searching existing APIs.
