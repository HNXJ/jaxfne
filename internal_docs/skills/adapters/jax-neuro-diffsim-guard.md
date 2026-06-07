# jax-neuro-diffsim-guard

**Triggers:** jaxley, integrate, neuron simulation, differentiable, dt, solver, BPTT, spike reset, jaxfne tune.

**Purpose:** Keep differentiable neuron simulations correct and memory-feasible across jaxley, jaxfne, and custom JAX ODEs.

**Gates:**

- Enable x64 before building stiff biophysical arrays when scientifically needed.
- Verify API names against the installed package — never from memory.
- `stimulate()` / `record()` are not jit/grad-safe; use `data_stimuli` / `data_clamps` in `integrate`.
- Hard spike reset is non-differentiable; use `stop_gradient` or surrogate gradients.
- Long backprop: use `checkpoint_lengths` (jaxley) or `jax.checkpoint` / `remat` on scan bodies.

**Full skill:** user-installed `jax-neuro-diffsim-guard`. Pair with `neuro-biophysics-units-sanity` on NaN or implausible Vm.
