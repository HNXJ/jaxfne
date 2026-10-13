# Archived scripts

Scripts with no reference in the tree, archived 2026-10-13 (T-17). History: git.

| script | lines | last commit touching it | what it did |
|---|---|---|---|
| audit_semantic_negatives.py | 114 | 19aef19a 2026-09-19 | Semantic-negative docs gate (Batch B7): fails on forbidden conflations unless the line carries an explicit negation marker. |
| generate_mechanism_tutorials.py | 524 | 9958a725 2026-09-28 | Generates 3 progressive mechanism notebooks (01, 02, 03) reusing one conceptual model. |
| make_delta_test_01_report.py | 629 | baf43d49 2026-09-30 | Builds the v0.3.31 Delta-Test 01 companion bundle (PDF report, PNG zip, manifest) from existing receipts; does not re-run the notebook. |
| repair_notebooks.py | 198 | 0635f3a8 2026-08-26 | Splits long code cells into chunks of at most 10 logical lines (`split_code_lines`) and rewrites notebooks under `artifacts/tutorials` and `notebooks/`. |
