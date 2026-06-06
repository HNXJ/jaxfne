# 03_JAX_RUNTIME_CONTRACT

Doctrine: numerical kernels should use `jax.numpy`, explicit random keys, `lax.scan` for time, `vmap` for batching, JIT only for hot numerical paths, and no serialization/plotting inside JIT. Manuscript JAX backend text is at `2026_jtfne_arxiv.txt:L243`; the JAX doctrine file in uploaded context adds explicit PRNG, no hidden global random state, scan/vmap/JIT constraints, float32 default, and x64 opt-in.

Live audit highlights: JAX kernels use explicit PRNGKey at 47 sites, `lax.scan` at 12 sites, `vmap` at 34 sites, JIT at 10 sites, and the main gap was host-side unseeded `np.random` in objectives nulls (`Pasted markdown.md:L95`).

## Grep evidence (live audit @ fab4c9c, `jaxfne/**/*.py`)

The original ZIP grep tables came back empty; these are the live-repo counts from the deep audit. Re-run the greps to refresh before any patch.

| category | live count | reproduce command |
|---|---|---|
| explicit PRNG (`PRNGKey`/`jax.random`/`fold_in`/`split`) | 47 | `grep -RnE "PRNGKey\|jax\.random\|fold_in\|random\.split" jaxfne --include='*.py'` |
| `lax.scan` (time evolution) | 12 | `grep -RnE "lax\.scan" jaxfne --include='*.py'` |
| `vmap` (batch/seed/candidate) | 34 | `grep -RnE "vmap" jaxfne --include='*.py'` |
| `jit` (opt-in numeric hot paths) | 10 | `grep -RnE "jax\.jit\|@jit\|[^a-z]jit\(" jaxfne --include='*.py'` |
| pytree (`tree_flatten`/`register_pytree`/FlatNet) | 23 | `grep -RnE "register_pytree\|tree_flatten\|tree_unflatten\|FlatNet\|simulate_flat" jaxfne --include='*.py'` |
| `enable_x64` references | 20 | `grep -RnE "enable_x64" jaxfne --include='*.py'` |
| host-side `np.random` (non-kernel) | 16 | `grep -RnE "np\.random\|numpy\.random\|random\.seed" jaxfne --include='*.py'` |

Quote the `--include='*.py'` glob — an unquoted `--include=*.py` triggers a zsh `nomatch` that silently aborts the whole `grep` line (same failure class as B02's dist-clean glob).

### Known host-side `np.random` sites (the F21 surface)

- `objectives.py:118,133,141,149` — **RESOLVED by B01 / PR #22** (now threaded through explicit `rng`/`null_seed`).
- `tutorial_utils.py` (5 sites), `fields/proxy.py:575` — seeded `RandomState`/`default_rng(seed)`, reproducible host-side, acceptable.
- `vis/*` (4 sites) — fixed-seed cosmetic layout jitter, acceptable.

### Known recompilation hot spots (B05)

- `validation.py:1251` emits `N_compile=2` for the jit opt-in paths `simulate` `(1,16,5,30)` and `simulate_batch` `(3,16,4,20)` — stabilize static-arg signatures; assert `N_compile <= 1`; prove traces unchanged.



## Contract for future patches

| Area | Contract | Stop rule |
|---|---|---|
| PRNG | Every stochastic simulation/JAX path accepts explicit PRNG keys; host-side statistical nulls must accept `seed` or `rng`. | Stop on hidden global random state in reproducible reports. |
| scan | Time stepping kernels use `jax.lax.scan` where feasible. | Stop on Python time-loop in public simulation hot path unless profiled/justified. |
| vmap | Batch/seeds/candidates use `vmap` when shape-stable. | Stop on copy-pasted candidate loops in numerical kernels. |
| jit | JIT only numeric hot paths. | Stop on JSON, plotting, file I/O, or Python mutation inside JIT. |
| dtype | Default float32; x64 opt-in before array creation; report dtype. | Stop on accidental float64 promotion in kernels. |
| PyTree | FlatNet/PyTree must keep Python objects out of traced kernels. | Stop if FlatNet carries mutable Python objects into JIT. |
