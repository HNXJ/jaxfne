# Solvers API

ODE solver backends for jaxfne neural simulations.

```python
from jaxfne.solvers import (
    SolverConfig,
    EulerSolver,
    DiffraxSolver,
    solve_ode,
    euler_step,
    euler_scan,
    solve_volume_conductor_experimental,
)
```

---

## `SolverConfig`

```python
@dataclass
class SolverConfig:
    ...
```

Configuration class for ODE solvers. Selects the solver backend and its
parameters.

**Key fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | `"euler"` | Solver backend: `"euler"` or `"diffrax"`. |
| `dt` | `float` | `0.1` | Integration timestep in milliseconds. |
| `rtol` | `float` | `1e-3` | Relative tolerance (diffrax only). |
| `atol` | `float` | `1e-6` | Absolute tolerance (diffrax only). |
| `solver_type` | `str \| None` | `None` | Diffrax sub-solver: `"dopri5"` \| `"tsit5"` (defaults to Tsit5). |

---

## `EulerSolver`

```python
class EulerSolver:
    ...
```

Forward Euler integrator using `JAX` and `jax.lax.scan`. The default and
recommended backend for most jaxfne simulations.

**Method:**

```python
def solve(self, dydt_fn, y_init, t_start, t_end) -> tuple[jax.Array, jax.Array]
```

Integrates `dydt_fn` from `t_start` to `t_end` using forward Euler stepping
compiled via `lax.scan` — no Python loop overhead at runtime. Returns
`(y_final, trajectory)`.

---

## `DiffraxSolver`

```python
class DiffraxSolver:
    ...
```

Optional Runge-Kutta solver using [diffrax](https://github.com/patrick-kidger/diffrax)
(lazily imported). Provides higher-order integration at the cost of more
function evaluations per step.

> **Requires:** `pip install diffrax`. Raises `ImportError` at construction
> time if diffrax is not installed.

**Method:**

```python
def solve(self, dydt_fn, y_init, t_start, t_end) -> tuple[jax.Array, jax.Array]
```

Integrates `dydt_fn` from `t_start` to `t_end` using diffrax. Returns
`(y_final, trajectory)`.

---

## `solve_ode`

```python
def solve_ode(
    config: SolverConfig,
    dydt_fn,
    y_init: jax.Array,
    t_start: float,
    t_end: float,
) -> tuple[jax.Array, jax.Array]
```

**Public ODE solver entrypoint.** Routes to `EulerSolver` or `DiffraxSolver`
based on `config.method`. Note `config` is the first positional argument.

```python
cfg = SolverConfig(method="euler", dt=0.1)
y_final, trajectory = solve_ode(cfg, my_dydt, y_init, t_start=0.0, t_end=1000.0)
```

---

## `euler_step` *(backward-compat)*

```python
def euler_step(y, t, dt, dydt_fn) -> jax.Array
```

Single forward Euler step. Kept for backward compatibility; prefer
`solve_ode` with `SolverConfig(method="euler")` in new code.

---

## `euler_scan` *(backward-compat)*

```python
def euler_scan(y_init, t_start, dt, n_steps, dydt_fn) -> tuple[jax.Array, jax.Array]
```

Forward Euler integration over a full time range via `lax.scan`. Note the
range is expressed as `n_steps` (not `t_end`); returns `(y_final, trajectory)`.
Kept for backward compatibility; prefer `solve_ode` in new code.

---

## `solve_volume_conductor_experimental`

```python
def solve_volume_conductor_experimental(...) -> None
```

Experimental volume conductor solver skeleton.

> **Always raises `NotImplementedError`.** Requires boundary/gauge/residual/
> convergence validation before it can be used. Do not call in production.
