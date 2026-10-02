---
name: jaxfne-plasticity
description: Enable, disable, clamp or mask HDP plasticity (dW/dt = F_W(H, ...)) with explicit controls.
metadata:
  audience: agents
---
# jaxfne plasticity

## WHEN
Any run where weights change: HDP rules, masks, frozen controls, clamps.

## AUTHORITIES
1. `docs/doctrine/rbs_rbd_hdp.md` (HDP is dW/dt = F_W(H, ...)).
2. Live code: `jaxfne/hdp_network.py` (`projection_mask`, `enable_plasticity`,
   `disable_plasticity`, `clamp_plasticity`, `pin_projection_weights`), `Configuration.hdp`.

## RULES
- Every plastic result needs a fixed-W control on the identical realized network.
- Masks select edges by realized ownership (`jaxfne.ensemble_edge_ownership` for ensembles);
  never by guessed index ranges.
- `hdp_params` is a compatibility transport; name the semantic group you change.
- Stored != plastic != free != optimizer-exposed parameters.

## STEPS
1. Build one realized network; copy it per arm (same `W0`).
2. Plastic arm: `hdp_params`; control: `disable_plasticity(hp)` (a rule without a
   `k_w` default raises; use an all-zero mask instead).
3. Partial: `m = projection_mask(n_edges, values=...)`, then `disable_plasticity(hp, mask=m)`
   or `clamp_plasticity(hp, mask=m, value=v)` with `pin_projection_weights`.
4. Capture `w_final`/`w_trace` per arm (`jaxfne-state` rule on diagnostics).

## STOP
- Arms built from different realizations; a mask whose length is not `n_edges`.

## VERIFY
- Control arm `W` bit-equal to `W0`; plastic arm `W` differs; masked edges unchanged.

## DONE
- Per-arm weight change with its control, on one realized network.
