---
name: jaxfne-repo
description: Bounded implementation, debugging, testing, API changes, and repository verification for jaxfne.
---
# jaxfne repository procedure

Before mutation:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

1. Preserve pre-existing user changes.
2. Identify the executable parameter/storage/path actually consumed by the selected backend before modifying optimization or configuration code.
3. Patch the smallest semantic boundary. Preserve compatibility unless the task explicitly authorizes contraction.
4. Add cheap property/unit tests first; attach many assertions to shared minimal-circuit executions rather than multiplying simulations.
5. Verify relevant nulls, negative controls, shape rejection, finiteness, deterministic/PRNG semantics, and continuation when affected.
6. Run targeted tests and exact receipts. Run broader gates only when proportional to the checkpoint.
7. If tests become slow, distinguish test-selection drift, repeated execution, compilation/runtime regression, and environment variance before optimizing.
8. Update directly affected docs only. Public docs describe equations/contracts, not debugging history.
9. Report API, mathematical, numerical, evidence, documentation, and compatibility deltas when material.
10. Git remote mutation requires explicit authorization.
