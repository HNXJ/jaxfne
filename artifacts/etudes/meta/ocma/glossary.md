# OCMA-J24 glossary: replication evidence for one reference paper

Ontology for the ontology-constrained multi-agent analysis (OCMA) of jaxfne's reference list, after the HPC-36
design of Nejat et al., Sci Rep 2026 (doi:10.1038/s41598-026-77038-y; code `hnxj/mllm-public`). Unit of analysis:
one row of `docs/reference/references.md`. Every non-null score cites `path:line` in this repository (or the
paper's page and figure for factors 17-20).

Scale (differs from HPC-36 because the evidence is a repository, which can be searched exhaustively):

| Score | Meaning |
|---|---|
| +1.0 | present and verified: the cited file shows it, for this paper |
| +0.6 | present, partial (a subset, a qualitative match, or a value without tolerance) |
| +0.2 | indirect (a generic capability that would serve, no paper-specific use) |
| 0.0 | searched and absent |
| -0.6 | present but disagrees (a test or result departs from the paper; a documented departure) |
| -1.0 | refused: the core rejects the mechanism, or a scored result failed on the paper's claim |
| null | not applicable to this paper (the paper has no such element) |

## H1 Mechanism realized in the core (IDs 1-8)

| ID | Key | Definition |
|---|---|---|
| 1 | Emitter class | the paper's cell or population model exists as a jaxfne emitter (`jaxfne/emitters*.py`) |
| 2 | Synapse kinetics | its synapse or receptor model (current/conductance, rise/decay, delays, STP) exists in the core |
| 3 | Plasticity rule | its plasticity or homeostasis rule exists as an HDP or RBD realization |
| 4 | Network topology | its connectivity statistics (sizes, p, in-degree, areas, layers) are constructible from a `CircuitSpec` |
| 5 | Drive protocol | its external input (Poisson, OU, step, pulses, stimuli) is expressible with core drives |
| 6 | Observable readout | its measured quantity (rate, spectrum, LFP, CSD, EEG, weights) is a core readout or probe |
| 7 | Parameters transcribed | the paper's parameter values appear in repo code or a source record, cited to the paper |
| 8 | Core gap | a needed piece is missing or refused (score -1 refused, 0 missing nothing, null n/a; cite the refusal) |

## H2 Reproduction evidence in the repository (IDs 9-16)

| ID | Key | Definition |
|---|---|---|
| 9 | Value pinned | a test pins a number from the paper (equation, table, stated value) |
| 10 | Regime reproduced | a test or etude reproduces a qualitative regime the paper reports |
| 11 | Panel remade | a figure panel of the paper is re-made (etude, atlas, notebook, dashboard figure) |
| 12 | Quantitative match | a paper number is compared with a declared tolerance and the comparison is recorded |
| 13 | Preregistered | a signed preregistration exists for this paper |
| 14 | Scored | a scored result exists: +1 whatever its outcome (pass, pending, partial); the outcome is scored in factor 12; -1 only when the lane was refused or withdrawn |
| 15 | Oracle comparison | an independent implementation (author code, Brian2, NEST, Jaxley) was compared |
| 16 | No target fitting | the reproduction did not tune parameters to the target figure (or tuning is declared) |

## H3 Dashboard readiness (IDs 17-24)

| ID | Key | Definition |
|---|---|---|
| 17 | Named target | a specific figure, table or stated number is named as the target |
| 18 | Estimand from core | the target is computable from core readouts without new mechanisms |
| 19 | Criterion from paper | a pass criterion follows from numbers the paper states |
| 20 | Oracle available | author code or data is public (or not needed) |
| 21 | Cheap run | the target runs in under 10 min on one CPU core (inferred from sizes and durations) |
| 22 | Scorer exists | a script writes a score file the dashboard can read (`files` plus `key`) |
| 23 | Publishable | no restricted data and no local-only rule (e.g. the Mulvey/LANSA lane stays local) |
| 24 | Core-check fit | +1 when one mechanism-level target (an equation, closed form, single-cell or single-mechanism regime) can run as a test-sized check; +0.6 when such a target exists inside a larger paper; 0 when only a full network lane reproduces the paper |

## Derived scores

H1 is the mean of non-null factors 1-7, with factor 8 included only when negative (a gap is a penalty;
"nothing missing" adds nothing). H2 and H3 are the means of their non-null factors. Dashboard action, first match wins:

1. the row already has a dashboard entry: `exists` (update it if the council found new evidence);
2. H2 >= 0.6 and H3 >= 0.5: `core-check` from existing evidence;
3. H1 >= 0.6 and H3 >= 0.5: `core-check (build)`: a new scorer for the named target;
4. H1 >= 0.6: `roadmap` row with its first target;
5. otherwise the row stays on the study list.

## Versions

- J24 (2026-11-08): pilot, 5 rows (HNXJ/ocma `cases/jaxfne/council/pilot/`).
- J24.1 (2026-11-08, after the pilot, before the full council): factor 14 scores existence, outcome moves to
  factor 12 (a rater scored Vogels 2011 -1 for its pending R-E criterion); factor 8 is a penalty only (0 pulled
  complete mechanisms down); factor 24 graded (raters split 0 to +1 on paper lanes); action rule adds `exists`
  and `core-check (build)` (under J24 no pilot row, including the signed and scored Vogels lane, reached any
  dashboard action); citations must be repo-root paths (`PILOT.md`).
