# TFNE-IMPORT-01 — CLOSED. `jaxfne.tfne` registered ADVANCED.

**Branch:** `dev`
**Entry state:** `d2c0d2d` (execution qualified)
**Tier decision:** ADVANCED, authorized by Hamm.
**Scope:** reachability and its tier. No compiler semantics changed.

## Defect

`artifacts/context.md` declares `jaxfne` the only supported public entry point,
and `docs/doctrine/tfne_algebra.md` names `jaxfne.tfne` as the implementation of
the canonical language. Neither was true together:

```
import jaxfne; hasattr(jaxfne, "tfne")   ->  False
```

Only `from jaxfne import tfne` worked. Every sibling submodule — `emitters`,
`plasticity`, `hdp_rule`, `bridges`, `connectivity`, `neuronal_tensor` — was
already attribute-reachable, so `tfne` was the exception rather than a
deliberate boundary.

The test suite could not have caught this. `pytest` inserts rootdir on
`sys.path`, so `tests/test_tfne_algebra.py` passed while a user outside the
repository root would resolve `jaxfne` from site-packages, where version 0.4.24
has no `tfne` module at all.

## Delivered

- `jaxfne/__init__.py` imports the module, so `jaxfne.tfne` resolves. Module-level imports are stdlib plus numpy (`neuronal_tensor` and `connectivity` are imported inside functions), so root import cost is unchanged and `tests/test_root_import_lightweight.py` is unaffected.
- `jaxfne/public_surface.py`: `tfne` added to `_ADVANCED` and `ADVANCED_NAMESPACE["tfne"] = "jaxfne.tfne"`.

The registered symbol is the **module**, not its contents. `parse`, `resolve`,
`realize`, `normalize` and `flatten` are generic names that would collide
conceptually at the root; they stay reachable as `jaxfne.tfne.parse` and so on.
This follows the established contract shape — ADVANCED means reachable but
outside `__all__` — while keeping the root namespace clean.

ADVANCED rather than CANONICAL is the honest tier today: the compiler conforms
partially to a sealed language (S9 declared frontiers, S11, S12, S13, S20, S25
and statement atomicity remain open) and TFNE-PARAM-01 means declared rule
parameters do not reach the executed model.

## Tests

Added to `tests/test_tfne_execution.py`:

- `test_tfne_is_reachable_from_the_documented_entry_point` — `hasattr`, absent from `__all__`, `symbol_tier("tfne") == "ADVANCED"`, namespace maps to `jaxfne.tfne`.
- `test_tfne_imports_from_outside_the_repository_root` — a clean interpreter launched from a scratch directory with only the checkout on `PYTHONPATH`, asserting the resolved `jaxfne.__file__` is the checkout, that `jaxfne.tfne` is reachable, and that `flatten` realizes 3 neurons. An in-process check would have passed on rootdir insertion alone, which is exactly the failure mode that hid this.

## Public surface contract

The surface grew by one classified symbol, 265 -> 266. Handled as the file's own
convention requires rather than by loosening the gate:

- `artifacts/public_surface_contract_v0413.json` regenerated with its owning generator, `scripts/generate_public_surface_contract.py`.
- `test_public_symbol_count_contraction_from_baseline` updated to 266 with the reason recorded in its docstring, alongside the existing JDNA, SurrogateConfig and registrable-HDP entries.
- `public_exports` stays 190, `compatibility` 13, `experimental_internal` 13 — the growth is entirely ADVANCED, so the canonical contract is unchanged.

## Evidence

```
python -m pytest tests/test_public_surface_contract_v0413.py \
                 tests/test_tfne_execution.py \
                 tests/test_tfne_algebra.py -q   ->  54 passed in 7.82s
python scripts/run_test_gate.py dev              ->  138 passed, 1 skipped, 2 deselected
                                                     docs language audit: pass
                                                     vocabulary check: pass
python scripts/run_test_gate.py broad            ->  3917 passed, 75 skipped, 37 deselected,
                                                     4 xfailed in 1650.10s; All checks passed!
```

`broad` was run because this change touches `jaxfne/__init__.py` and the public
surface, which the curated dev gate does not cover on its own.

## Status

- TFNE-IMPORT-01: **CLOSED**
- Next: `TFNE-PARAM-01` (opened by TFNE-EXEC-01, pinned by test), then the remaining measured conformance gaps TFNE2-03 through TFNE2-07, then `CTX-01`.
