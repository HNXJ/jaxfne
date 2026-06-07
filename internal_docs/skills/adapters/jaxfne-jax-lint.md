# jaxfne-jax-lint

**Triggers:** JAX compatibility, JIT, vmap, scan, PRNG, dtype, kernel lint.

**Purpose:** Catch JAX semantic drift — numpy in hot paths, global randomness, missing `lax.scan`, float32 discipline.

**Checks:**

```bash
grep -rn "random.seed\|np\.random" jaxfne/ | head
grep -rn "jax.random.PRNGKey\|lax.scan" jaxfne/ | head
```

Report PASS/FAIL/WARN with `file:line`.

**Full skill:** user-installed `jaxfne-jax-lint`.
