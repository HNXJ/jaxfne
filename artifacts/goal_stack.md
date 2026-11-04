# goal stack

Top = current focus. Goals here are what Hamm said; agents propose edits in `p-jaxfne`. The
0.5.x programme goals stay in `todo_stack.md` ("Goals in plain words").

- G-0 be able to submit the jaxfne paper to arXiv: 8 main and 16 supplementary figures, at most 80 references, checklist C1-C12 at 100/100 (Hamm 2026-10-25). Held by: `artifacts/programme/design/plans/arxiv_paper_plan.md`; every paper task names its figure or checklist item.
- G-1 be able to read every jaxfne release as tables generated from changelog fragments, claim-checked (Hamm 2026-10-12). Held by: `scripts/release_body.py --tag <next>` passes its claim check (script not yet ported).
- G-2 be able to run CAMINOS Fig. 1 through the jaxfne API within the port's tolerance (Hamm 2026-10-12: build the features, then P6). Held by: P6 v2 PASS (`papers/caminos/RESULT_P6.md`).
- G-3 be able to hand jnwb a jaxfne simulation with known lag and spectrum as a test fixture (Hamm 2026-10-12, meet 204dfdaa). Held by: the fixture's recovery test in jnwb `tests/`.
- G-4 be able to see every scored replication at https://hnxj.github.io/jaxfne/dashboard/, published on dev push (Hamm 2026-10-12). Held by: `python scripts/build_dashboard_data.py --check` and the `Dashboard` workflow.
