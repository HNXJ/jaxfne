# Theory programme proposal (2026-10-08)

Status: proposal, nothing declared. Hamm picks the family that opens.

## Source

openai/math (https://github.com/openai/math, read 2026-10-08: the repository page,
`README.md`, `history.md`, and the Mézard–Parisi reasoning trace, through page
summaries rather than a full reading). Process features it shows:

1. Results come as families: a principal result with companion arguments and
   alternative proofs (719 manuscripts in 372 families).
2. Breadth first: about 4,000 open problems were posed, and the results kept were
   consolidated by significance.
3. Verification tiers: about 42% of top-line results have Lean proofs, and the
   repository states that unformalized results could have issues.
4. Corrections are versioned. One sign error invalidated an argument and spread to
   dependent papers: 3 were withdrawn, 14 revised and 13 updated to cite the
   repaired companions.
5. The published trace tests boundary cases before it attempts a proof, records its
   dead ends, and closes with a separate stage that audits each identity.

The repository's mathematical results are not used here; only its process is.

## Process changes for jaxfne

- **P1 Conjecture ledger.** Each entry holds a statement, the receipt it came from,
  a falsifier and a verification tier. Entries are grouped into families.
- **P2 Verification tiers,** printed with every theory claim: T0 numerical (float64,
  several seeds); T1 certified numerics (interval or exact rational arithmetic);
  T2 written proof checked by second-reviewer; T3 Lean. Lean is not installed on
  this machine.
- **P3 Dependency edges.** Each claim lists the claims it uses. A correction marks
  its dependents for review, as in item 4 above.
- **P4 Reachability before declaration.** A gate criterion is shown reachable, by a
  bound or a positive control, before the gate is declared. O1 found after the
  runs that the 0.3 OSI criterion was out of reach for one kernel family.

## Candidate families, ranked

**F1 State reduction for HDP rules** (from K1h). In `pcl_stdp_h`
(`artifacts/etudes/pcl_h/pcl_h_rule.py`) a per-edge trace that resets at
postsynaptic spikes is rewritten as a free-running presynaptic trace in H minus a
per-edge snapshot taken at the last reset. Conjecture: any per-edge state that obeys
a linear recurrence driven by presynaptic events, with resets triggered by
postsynaptic events and a time constant shared within a class, factors exactly into
per-neuron H traces (one per class) plus one per-edge snapshot; floating-point
results differ only by evaluation order. Question for the family: the minimal
per-edge dimension of a rule in the finite-state rule grammar. Payoff: a reduction
pass whose correctness is a theorem rather than a seed check. Tier: T2 now, T3
feasible. Falsifier: a registered rule meeting the hypotheses whose reduced form
differs in float64 beyond evaluation-order error.

**F2 Replication under divergence** (from K1h-nf). Moving initial weights by one ulp
changed decoding by up to 0.069, and K1h split from K1 on one ulp in one weight.
Conjecture: plastic spiking columns at these settings separate nearby states at a
positive rate, so pointwise equality gates hold only below a horizon, and beyond it
a replication gate needs a measured noise floor. Work: measure the growth of
spike-train distance after one-ulp perturbations across seeds and `dt`, estimate
the horizon, and make the noise floor a standard gate part. Tier: T0, with a T2
argument for event-driven threshold maps.

**F3 Field and spike dissociation under the current-sum proxy** (from the PCL LFP
result). Under the shipped proxy, a linear sum of channel currents, learned
inhibition that removes spikes can raise field power, because power follows
synaptic current rather than spike count. Target: a proposition giving the E/I
current condition under which power rises while spikes fall. Scope: the proxy only;
readouts are relative-unit proxies. Relevance: in the omission paradigm, predictive
suppression of spikes need not lower LFP power. Tier: T2.

**F4 Criterion reachability bounds** (from O1). An upper bound on the median OSI a
linear–threshold unit with a given kernel family can reach for the declared stimuli
(drifting bars, 8 px segments) and responsiveness rule. This makes P4 a
computation. Tier: T1, and T2 if a closed form exists.

**F5 Stability margins for HDP with homeostasis.** The fact stack allows "stable"
only with a Floquet or margin result. Reduce rule plus homeostasis to a rate system
with JDNA and derive margins, with Floquet multipliers under periodic drive. Related
existing work: `artifacts/etudes/hdp_controllability_reachability/`. Tier: T1/T2.

## Recommendation

Open F1 first: it is the smallest, rests on committed evidence and yields both code
and a lemma of Lean size. F2 next, since it changes how every replication gate is
written. F3 is the family closest to the laminar data.
