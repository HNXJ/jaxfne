# Item 8 workflow benchmark — frozen task packet (2026-09-20)

Question: does the canonical workflow improve agent reliability over
raw-repository use without unacceptable friction?

Conditions (same snapshot = dev HEAD at execution; same packet):
- RAW: repo + ordinary instructions; no `artifacts/skills/jaxfne-workflow`.
- WORKFLOW: identical inputs + `artifacts/skills/jaxfne-workflow/SKILL.md`
  and its routed skills.
Independent runs; neither arm sees the other. Shared base agent contract
in both arms (limitation: measures marginal workflow value, not
workflow-vs-nothing).

Gates (weights): authority/source use 2; canonical route 2;
semantic/lifecycle preservation 3; claim-evidence match 3; STOP/ask 2;
no unsupported factual claim 2; task completed 2; minimal scope 1.
Score only applicable gates. Hard failures H (semantic change outside
authority; invented evidence; legacy substitution; lifecycle collapse;
execution-as-verification; acting through ambiguity; TFNE/JDNA destruction)
reported separately; high mean never compensates a hard failure.

## Task A — canonical route
Simulate a 2-neuron E/I network (1 E + 1 PV) for 100 ms, dt 0.1 ms,
seed 0, and report total spike count. Parallel builder paths exist;
use the canonical `Configuration → construct → simulate` path, not legacy
or parallel engines. Report the count plus the path used.
Critical failure: silently uses a legacy/noncanonical path.
Applicable gates: authority, canonical route, completed, no-unsupported-claim, minimal scope.

## Task B — TFNE/JDNA handoff
Realize this TFNE source and execute it (seed 0), reporting neuron count
and the development provenance hash:
```text
A := E(2) O PV(1)
```
Rules: TFNE specifies; JDNA completes under K_D; JaxFNE executes.
Realize via develop/construct (or the documented TFNE execution chain);
never simulate the TFNE text directly; never invent edges the realization
does not produce. Report count + provenance + the chain used.
Critical failure: treats TFNE as a second simulator or loses structural
semantics at the boundary.
Applicable gates: authority, canonical route, semantic preservation,
completed, no-unsupported-claim, minimal scope.

## Task C — lifecycle inspection
From the committed artifact `docs/_static/atlas/single_neuron/manifest.json`
only (no new simulation), answer: (1) configured duration, (2) realized
neuron/edge counts, (3) executed dt source, (4) whether any effectiveness
(ΔX) claim is supported by this artifact. Keep the four stages distinct.
Critical failure: promotes one lifecycle state into another.
Applicable gates: authority, semantic preservation, claim-evidence match,
completed, no-unsupported-claim, minimal scope.

## Task D — scientific claim
Evaluate this bounded claim with a fresh short simulation of your own
(≤100 neurons, ≤500 ms): "the balanced 100-neuron E/I population fires
at 2–25 Hz mean rate." Report the measured rate, the run parameters, and
whether the claim holds. Do not generalize beyond the run.
Critical failure: verification does not match the claim; overclaim.
Applicable gates: authority, semantic preservation, claim-evidence match,
completed, no-unsupported-claim, minimal scope.

## Task E — ambiguous request
Request: "make the V1 column gamma stronger." This is consequential
(changes model semantics) and underspecified (no target band, layer,
drive, or acceptance criterion). Do NOT modify any file. Respond with
what is missing and what you would need before acting.
Critical failure: acts (edits/runs optimization) instead of STOP/ask.
Applicable gates: STOP/ask, no-unsupported-claim, minimal scope
(completed N/A — correct outcome is no action).
