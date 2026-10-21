# TFNE/2 language supersession — authority adopted, compiler delta measured

**Branch:** `dev`
**Predecessor state:** `518cd5e` (adopted `tfne/1` as project source 7, added `jaxfne/tfne.py`, `tests/test_tfne_algebra.py`)
**Authority input:** `tfne/2`, status `SEALED LANGUAGE`, supplied by Hamm as a corrected replacement for project source 7
**Scope:** authority replacement and measurement only. No change to `jaxfne/tfne.py` or its tests.

## Why this is a supersession and not an edit

`tfne/2` is not a clarification of the text committed at `518cd5e`. It changes
realized meaning in two places (exclusion direction, instance addressing) and
adds five language features the compiler has no representation for. The
predecessor text was one week old in the repository; it is superseded, not
revised.

`tfne/2` S31 states that the language seal implies no parser or compiler.
Adopting the authority therefore does not by itself make `jaxfne.tfne`
defective-and-blocking; it makes the gap between them a measured, named
quantity instead of an unstated one.

## Delivered

- `artifacts/project_sources/7_tfne_algebra.md` — replaced with `tfne/2` verbatim under the project-source status header convention. Header records that the language seal implies no compiler and that the source outranks `jaxfne.tfne` on disagreement.
- `artifacts/project_sources/README.md` — entry 7 updated to `tfne/2` / `SEALED LANGUAGE`, pointing at the doctrine page for the conformance boundary.
- `docs/doctrine/tfne_algebra.md` — the `tfne/1` "Deterministic rules" section asserted superseded semantics as current fact. Replaced with a `Compiler conformance` section (satisfied clauses, non-conformant clauses, compiler-fixed choices where the language leaves room). Mapping-table row for exclusions corrected from "realization vetoes" to subtraction.

## Method

Conformance measured by direct execution against the checkout, not by reading
docstrings. Two probe scripts drive `parse -> resolve -> realize` per clause and
classify each as CONFORMS / CONTRADICTS.

```
python scratchpad/tfne2_probe.py     # S6 S7 S9 S12 S14 S20 S25
python scratchpad/tfne2_probe2.py    # S6 S8 S11 S13 S20
```

Probes are scratch, not repository artifacts. The verdicts they produced are
reproduced below; the durable statement of the delta lives in
`docs/doctrine/tfne_algebra.md`.

First run of probe 1 resolved `jaxfne` from `C:\Python314\Lib\site-packages`
(installed 0.4.24, which has no `tfne` module) rather than the checkout, and
failed on import. Re-run with the checkout pinned on `sys.path`. Recorded
because it means a bare `python script.py` outside the repository root does not
see this work; `pytest` passes only because it inserts rootdir.

## Measured delta (OBSERVED)

| Clause | `tfne/2` requires | Observed at `518cd5e` |
| --- | --- | --- |
| S6, S7 replication addressing | `A^n = {A.1 ... A.n}` | `['E.0','E.1','E.2','E.3']` |
| S6 prefix rule application | `O[k](SEG^8)` -> 7 ordered adjacencies | `TFNEError: unexpected character '('` |
| S8 group sensitivity | `{A O B} O C` composes via composite frontier; not generally equivalent to `A O B O C` | both 12 edges, identical edge set |
| S9 frontiers | `in`/`out` reserved, derived defaults, `E_FRONTIER_UNRESOLVED` | `A.out` -> "matches no realized object" |
| S11 cross associativity | ungrouped `A X[k] B X[k] C` requires grouping | accepted ungrouped, 24 edges |
| S12 rule binding | `$L` / `$R` metavariables | `TFNEError: unexpected character '$'` |
| S13 projection identity | redundant explicit projection invalid | no check over `(src,dst,mechanism)`; `duplicate` guards cover rule/def/path names only |
| S14 exclusions | `E_- subset G_0`, `G = G_0 \ E_-`; unmatched exclusion invalid | **inverted** — matched exclusion raises ("violated by 4 realized edge(s)"); unmatched exclusion silently accepted |
| S20 canonical ordering | typed natural, declaration-independent (`L1<L2<L10`) | `['V.L10','V.L2','V.L1']` (declaration order) |
| S14, S25 statement atomicity | `;`-separated atomic statements inside a composite | `;` inside `{...}` is a parse error |
| S25 failure vocabulary | six named semantic classes | none present in `jaxfne/tfne.py` |

Clauses measured as satisfied: core form and normal form, recursive scale to
one cell, identity/`model` separation, `sum_c N[A.c] == N[A]`, replication
without implied connectivity, geometry into `s`, `H` reserved, typed `x`/`y`,
bidirectional index map, idempotent normalization seeding realization, hashes
as receipts.

## Consequence

Two entries change realized biology rather than syntax:

- **S14** is inverted. A specification `tfne/2` reads as "expand the rules, then
  remove this projection" is currently read as "fail if this projection exists",
  and a stale exclusion that `tfne/2` requires to fail is currently a silent
  no-op. Both directions of the guarantee are wrong, not weak.
- **S6/S7** is off by one. Every replicated instance path in `I` names a
  different instance than the language does.

`tests/test_tfne_algebra.py` encodes the superseded semantics — the test named
`test_exclusion_passes_when_absent_violates_when_present` asserts the inverted
rule as intended behaviour. Conforming the compiler therefore rewrites tests,
it does not merely extend them.

## Status

- `tfne/2` adopted as project source 7: **DONE**
- Compiler conformance: **NOT STARTED**, queued on `artifacts/todo_stack.md`
- `jaxfne/tfne.py` and `tests/test_tfne_algebra.py`: **UNCHANGED** (24 passed, 1.38s, after the documentation changes)

Nothing in this receipt claims `jaxfne.tfne` executes a TFNE specification.
That claim appears in the module docstring and the doctrine pipeline table and
remains untested: `tests/test_tfne_algebra.py` states "no simulation kernels are
touched", and no test constructs or simulates a model from `to_neuronal_tensor`
output. Queued separately.
