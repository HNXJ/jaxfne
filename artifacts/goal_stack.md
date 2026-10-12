# goal stack

Top = current focus. One line per goal: "be able to <do X>", then the check that holds it. Goals
here are what Hamm said; agents propose edits in `p-jaxfne`, never write them. The 0.5.x programme
goals stay in `todo_stack.md` ("Goals in plain words").

- G-1 be able to read every jaxfne release as tables generated from changelog fragments, with
  jnwb's claim check (Hamm 2026-10-12; meet 204dfdaa item 6). Held by: `scripts/release_body.py
  --tag <next>` renders the tables and the claim check passes (not yet in jaxfne).
- G-2 be able to run CAMINOS Fig. 1 through the jaxfne API and match the port within a declared
  tolerance (Hamm 2026-10-12: build the features, then P6). Held by: the signed P6 prereg's
  equivalence test (not yet written).
- G-3 be able to hand jnwb a jaxfne simulation with known lag and spectrum as a test fixture
  (Hamm 2026-10-12, meet 204dfdaa item 8). Held by: the fixture's recovery test in jnwb `tests/`.
- G-4 be able to see every scored replication on https://hnxj.github.io/jaxfne/dashboard/,
  published by Actions on dev push (Hamm 2026-10-12). Held by: `python
  scripts/build_dashboard_data.py --check` and the `Dashboard` workflow on dev.
