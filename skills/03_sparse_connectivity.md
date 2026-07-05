# Sparse Connectivity Skill

## Purpose
Replace dense O(N^2) connectivity construction with sparse or block-sparse construction.

## Rules
- Do not materialize dense `n x n` matrices when the actual graph is sparse or local.
- Generate edge lists directly from layer, area, or block rules.
- Prefer block-sparse, CSR-like, or edge-list representations for large networks.
- Preserve semantics; only the representation should change.
- Use dense construction only for small tests or explicit debugging.

## Acceptance checks
- Large-N builds do not scale quadratically in memory by default.
- Connectivity code has a sparse path that is the default for large models.
- Unit tests cover identical connectivity semantics between dense and sparse representations on small cases.

## Verified case in this repo (2026-07-05)

`jaxfne/connectivity.py::compile_connection_rules_jax` (a `jax.jit`-wrapped,
public, `probability`-rule connectivity compiler) materialized two dense
`(n_pre, n_post)` index grids via `jnp.tile` before sampling — a direct
violation of this rule and of the module's own docstring ("sparse-first ...
never materializes a dense mask"). Fixed by deriving `(pre, post)` for the
selected `max_edges` candidates via flat-index arithmetic
(`selected_idx // n_post`, `selected_idx % n_post`) on only the selected
subset, instead of building full `(n_pre, n_post)` grids. Verified
bit-identical output vs. the old implementation for the same PRNG key/inputs
at a non-trivial scale (13×9 pre/post, `max_edges=20`) — a true
behavior-preserving refactor, not just a shape-compatible rewrite.

Caveat worth knowing before assuming full sparsity: this jitted variant still
touches one `O(n_pre*n_post)` array of per-pair random scores, since that's
inherent to preserving the exact independent-Bernoulli-then-cap-at-`max_edges`
semantics under a single static-shape `jax.jit` call — a genuinely
`O(max_edges)`-only implementation would need different semantics (e.g. the
host-side `compile_connection_rules`'s rejection-sampling approach, which
directly draws `round(p * n_pre * n_post)` candidates) or a `lax.while_loop`
rejection scheme. Removing the redundant *index* grids (this fix) and
removing the *scoring* array (a bigger, separate change) are not the same
claim — don't conflate "no dense grid" with "O(max_edges) memory" when
reviewing similar code.
