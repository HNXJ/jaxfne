# Title and Abstract Decisions

Decision date: 2026-08-04

---

## Title (locked)

Primary:
  Omission Responses in a Multi-Area Laminar Cortical Hierarchy Emerge from
  Simple Firing-Rate Homeostasis

Alternative (jaxfne-explicit):
  Omission Responses from Firing-Rate Homeostasis in a Laminar Cortical
  Hierarchy Simulated with jaxfne

---

## Five-Sentence Summary

Cortical adaptation — spanning short-term suppression, omission responses,
and hierarchical mismatch — has been attributed to multiple mechanisms
including synaptic depression, intrinsic excitability changes, and predictive
coding circuits, yet no single computational principle has been shown to unify
these phenomena within one tractable model.

Here we introduce a minimal homeostatic state variable H(t), governed by
τ_H · dH/dt = r_target − r(t), that modulates intrinsic excitability of
Izhikevich neurons in a multi-area dense laminar cortical hierarchy (TFNE).

We simulate the exact omission paradigm from our in-vivo multi-area laminar
recordings and show that an omission response — matching the laminar and
spectrolaminar profile of the in-vivo data — emerges directly from H(t)
without invoking prediction-error circuits or short-term synaptic depression.

Systematic ablations removing H(t), isolating intrinsic-only or synaptic-only
coupling, contrasting H(t) against Abbott & Varela (1997) short-term depression,
and sweeping τ_H, r_target, noise, and connectivity establish both the
necessity of the slow homeostatic mechanism and the boundary of its generality
across neuron models (Izhikevich, LIF, HH/Jaxley).

These results position simple firing-rate homeostasis as a parsimonious
mechanism for the omission response and jaxfne as a reproducible open platform
for hypothesis-driven, pipeline-locked laminar electrophysiology simulation.

---

## Three-Sentence Novelties + Advantages

Novelty 1 — Mechanistic parsimony:
  Prior accounts of omission responses invoke either predictive coding
  circuits (Friston-style) or short-term synaptic depression (Abbott/Varela);
  we show that a single scalar H(t) tracking firing-rate homeostasis on a
  slow timescale (τ_H seconds–tens of seconds) is sufficient to reproduce the
  omission response in a biophysically realistic multi-area laminar hierarchy,
  providing the most parsimonious account to date.

Novelty 2 — Pipeline-locked reproducibility:
  Every result is produced by jaxfne's deterministic operator grammar
  (Configuration→Construct→Simulate→Source→Field→Probe→Objective→Manifest),
  exporting manifest.json, validation_report.json, and metrics.json under
  fixed seeds, making all figures directly reproducible artifacts rather than
  one-off notebook outputs.

Novelty 3 — Cross-model generalization with rigorous ablation:
  The H(t) claim is stress-tested by five ablation conditions (H(t)-removed,
  intrinsic-only, synaptic-only, fast-timescale, and Abbott/Varela synaptic-
  depression controls) validated across three neuron model classes, with every
  result reported with ≥5 seeds and confidence intervals.

---

## H(t) Anchoring References

  Turrigiano & Nelson (2004) Nat Rev Neurosci   — foundational motivation
  Cannon & Miller (2016) PLOS Comp Biol         — formal closest match to H(t)
  Abbott & Varela (1997) Science                — fast-mechanism foil
  Yaron et al. (2025) Neuron                    — empirical omission target

---

## Open Questions (empirical — do not change plan)

These will be answered by Stage 8 results, not by assumption:
  - Is the omission response sparse or dense in spiking?
  - Is it strong or weak in LFP amplitude?
  - Is the laminar signature feedforward (γ superficial) or feedback (α/β deep)?
  - Is there more or less activity at omission time vs. baseline?
  - What is τ_H calibrated to from in-vivo firing rate data?
