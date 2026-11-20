# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

- O-FIG05 equivalence-gate fig05 frozen PNG stale after a disclosed noise-stream change (865e74bb, 2026-09-24; inferred from the C3 estimator bisect); CI Linux runs the gate informationally and never sees it | Hamm (frozen-evidence change control) | since 2026-11-15 | closes when `tests/test_equivalence_gate_v20260815.py` passes on darwin after an owner-authorized fig05 re-freeze, or panel C renders from frozen V_m (T-FIG05)
- O-NUM bit-exact float32 assertions across differently compiled or shaped XLA graphs (#1–#5 of `artifacts/assurance/mac_gate_attribution.md`, incl. former O-97 `test_composition_identity_054.py` 2-ulp drift at n=8, 12) are platform-dependent; fixing requires a tolerance or O0 compile | Hamm | since 2026-11-15 | closes when Hamm rules the remedy and the tests are updated and pass (T-NUM)
- O-73 `population_rates` takes `dt` in seconds positionally (`jaxfne/analysis/metrics.py:1139`); a caller passing ms positionally gets rates 1000x wrong with only a DeprecationWarning | Hamm (release) | since 2026-11-14 | closes when a positional-ms test fails on the old form and the fix or hard error lands (T-26)
- O-74 `_bin_counts`, `fleiss_kappa_binary`, `fano_factor`, `pairwise_cc_binned_disjoint` raise ValueError when `bin_ms/dt_ms` is not whole (0.5.8 rounded) with no deprecation path (`jaxfne/analysis/metrics.py`) | Hamm (release) | since 2026-11-14 | closes when a deprecation path or changelog entry is pinned by a test (T-26)
- O-75 `fleiss_kappa_binary` changed definition at the same name and signature (`jaxfne/analysis/metrics.py:324`: 0.5.8 custom kappa to textbook Fleiss; unverified against 0.5.8) | Hamm | since 2026-11-14 | closes when verified against source and recorded in the changelog or reverted (T-26)
- O-76 Six bitwise tests are `xfail(strict=False)` and cannot fail (`tests/test_adex_route.py:1060,1086,1125`; `tests/test_lif_tensor_route.py:705,731,776`) | agent | since 2026-11-14 | closes when each is a strict xfail or passing test (T-NUM)
- O-77 Data-gated tests skip silently when data is absent (`tests/test_experiment_a_b1_canonical.py`, `tests/test_fig02_04_experiment_a.py`); `data_gated` marker coverage unverified | agent | since 2026-11-14 | closes when a data-free clone reports every skip with its marker (T-92)
