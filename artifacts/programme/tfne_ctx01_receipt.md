# CTX-01 — CLOSED. One integrated cortical model across the algebra.

**Branch:** `dev`
**Scope:** a single compact TFNE source plus its verification battery.
No grammar added, no existing behavior changed. PARAM-02/PARAM-04
pinned as bounded, not repaired.

## The model

`tests/test_tfne_ctx01.py`, `CTX` — two reusable laminar areas sharing
layer definitions, one three-statement rule, declared frontiers, an
order override, replication, and an atomic corner:

```text
O[both] := [mechanism = AMPA; weight = 0.625; plasticity = stdp;
            $L.out>$R.in [mech=AMPA]; {L4,L23}<L5 [mech=GABA_A];
            {L23}<>{L23} [mech=AMPA]];
O[aux] := [direction = >; mechanism = AMPA; weight = 0.375];
out[V1] := [L23]; in[V2] := [L4];
order[V1] := [L5, L23, L4, SEG.2, SEG.1];
L4 := [C = {E}; N = 2; G = [z0 = 0.0; z1 = 4.0]];
L23 := [C = {E, PV}; P = {E: 0.667, PV: 0.333}; N = 3];
L5 := [C = {E}; N = 1]; SEG := [C = {E}; N = 1];
V1 := L4 O L23 O L5 O SEG^2; V2 := L4 O L23 O L5;
AUX := {L5 O[aux] L4; Z.Q > L4};
V := AUX O V1 O[both] V2;
```

17 neurons, 31 edges (6 + 5 + 9 + 9 + 2), mechanisms AMPA/GABA_A at
canonical 2.0/5.0 ms. Every value asymmetric (0.625/0.375 weights,
2/3/1 counts, 2.0/5.0 taus) so no default passes silently.

## Verified by semantic class

| Class | Evidence |
|---|---|
| source / NF / digest | replay-stable; all declarations in NF |
| object identity | every neuron addressable; slices exact |
| cardinality / proportion | L23 = 2E+1PV both areas; 17 total |
| ordering | V1 override incl. replicas; V2 natural |
| topology | 31 edges from named statements |
| frontier identity | r1 through declared out/in; inner intact |
| mechanism identity | AMPA/GABA_A through to tensor |
| mechanism kinetics | executed {2.0, 5.0} |
| fixed weights | {0.625, 0.375}, non-default |
| developmental provenance | per-leaf origins; K_D determinism |
| geometry | JDNA honors G; executed ignorant at equal seed (PARAM-04 bounded) |
| plastic provenance | `stdp` recorded; edge count identical with/without |
| atomicity | dropped statement traceless in full model |
| constructed == realized | neuron table follows axis; edges subset |
| simulation | finite V_m/spikes, 17 neurons |
| refused | GABA statement → UNRESOLVED; delay → UNSUPPORTED |

## Newly exposed defect

None in the algebra. Two probe bugs (both mine, both instructive):
explicit `seed=` is required when comparing textually-differing specs
(digest-derived seed moves RNG with the text), and statement `[mech=]`
wins over the rule default (so refusal probes must target statements).

## Evidence

```
python -m pytest tests/test_tfne_ctx01.py -q
    -> 15 passed

python scripts/run_test_gate.py dev
    -> 282 passed, 1 skipped, 2 deselected (ctx01 module runs in
       broad/targeted, not in the dev module list)
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 4055 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+15 against the prior 4040 outcomes: the ctx01 module; the
       unrelated field01 timing failure from that run passed here,
       confirming pre-existing flakiness, not a regression)
```

The receipt is necessarily edited after the gate it reports.

## Status

- CTX-01: **CLOSED**. No new grammar needed: the integrated model stayed
  inside the conformant algebra throughout.
- Next in stack order: TFNE-PARAM-02 (delay), TFNE-PARAM-04 (geometry
  via JDNA).
