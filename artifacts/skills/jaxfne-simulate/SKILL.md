---
name: jaxfne-simulate
description: Run a realization, Configuration or Model with explicit duration, dt and seed; continue across chunks.
metadata:
  audience: agents
---
# jaxfne simulate

## WHEN
Executing any model.

## AUTHORITIES
1. Capability rows: `artifacts/programme/capability_inventory.md` (Simulate).
2. Live code: `jaxfne.agent.simulate`, `jaxfne.simulate` (`continuation=`,
   `return_state=True`), `Model.simulate_batch`.

## RULES
- `duration_ms`, `dt_ms` and `seed` are always explicit; never rely on defaults.
- A realization is compiled at the run's `dt_ms` (delay steps depend on it).
- Nonzero delays need the edge_list backend; the dense backend rejects them.
- Chunked runs pass the full returned state (including `delay_state`) as `continuation`.

## STEPS
1. `run = jaxfne.agent.simulate(obj, duration_ms=..., dt_ms=..., seed=...)`.
2. Chunks: `sig, state = jaxfne.simulate(model, ..., return_state=True)`, then
   `jaxfne.simulate(model, continuation=state, ..., return_state=True)`.
3. Hand the `Run` to `jaxfne-inspect` / `jaxfne-verify`.

## STOP
- `TypeError` from `agent.simulate` (unsupported object); non-finite state.

## VERIFY
- `jaxfne.agent.verify(run, "time_identity")` and `"finite"` PASS; segmented equals
  continuous bit-exactly when chunking.

## DONE
- A `Run` with its request (duration, dt, seed) and its source objects.
