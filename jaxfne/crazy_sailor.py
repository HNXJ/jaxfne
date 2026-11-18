"""Crazy Sailor M1: N-dimensional point-object active-fixation environment.

Pure JAX. Specification: ``artifacts/programme/design/plans/crazy_sailor/m0_contract.md``.
Internal until M5: not exported from the package root.

The environment owns dynamics, disturbances and the fixation error. It knows nothing
about controllers, observations or learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

DTYPE = jnp.float32


@dataclass(frozen=True)
class Calm:
    """No disturbance: returns a zero velocity increment."""

    def increment(self, key, t, p, dt, n):
        del key, t, p, dt
        return jnp.zeros((n,), DTYPE)


@dataclass(frozen=True)
class WhiteNoise:
    """Independent Gaussian velocity increments: dv^w = sigma * sqrt(dt) * eps, eps ~ N(0, I)."""

    sigma: float

    def __post_init__(self):
        if not float(self.sigma) >= 0.0:
            raise ValueError(f"WhiteNoise.sigma must be >= 0, got {self.sigma}")

    def increment(self, key, t, p, dt, n):
        del t, p
        eps = jax.random.normal(key, (n,), DTYPE)
        return jnp.asarray(self.sigma, DTYPE) * jnp.sqrt(jnp.asarray(dt, DTYPE)) * eps


@dataclass(frozen=True, eq=False)
class SailorConfig:
    """Environment parameters. Only this module reads them.

    ``actuator`` is B (N, D), default identity when D = N. ``target`` is p*, default zero.
    """

    n_dim: int
    dt: float
    damping: float = 0.0
    n_act: Optional[int] = None
    actuator: Optional[object] = None
    target: Optional[object] = None
    disturbance: object = Calm()

    def __post_init__(self):
        n = int(self.n_dim)
        if n < 1:
            raise ValueError(f"n_dim must be >= 1, got {n}")
        d = n if self.n_act is None else int(self.n_act)
        if d < 1:
            raise ValueError(f"n_act must be >= 1, got {d}")
        if not float(self.dt) > 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if not float(self.damping) >= 0.0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")
        b = jnp.eye(n, d, dtype=DTYPE) if self.actuator is None else jnp.asarray(self.actuator, DTYPE)
        if b.shape != (n, d):
            raise ValueError(f"actuator must have shape {(n, d)}, got {b.shape}")
        tgt = jnp.zeros((n,), DTYPE) if self.target is None else jnp.asarray(self.target, DTYPE)
        if tgt.shape != (n,):
            raise ValueError(f"target must have shape {(n,)}, got {tgt.shape}")
        object.__setattr__(self, "n_dim", n)
        object.__setattr__(self, "n_act", d)
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "damping", float(self.damping))
        object.__setattr__(self, "actuator", b)
        object.__setattr__(self, "target", tgt)


class State(NamedTuple):
    """Complete environment state. Nothing else is carried between steps."""

    p: jax.Array
    v: jax.Array
    key: jax.Array
    t_step: jax.Array


class Info(NamedTuple):
    """Per-step quantities computed by the environment at the new position."""

    error: jax.Array
    sq_error: jax.Array
    force: jax.Array


def reset(cfg: SailorConfig, key, p0=None, v0=None) -> State:
    """Start an episode. ``key`` is the episode seed; default p0 = v0 = 0."""
    p = jnp.zeros((cfg.n_dim,), DTYPE) if p0 is None else jnp.asarray(p0, DTYPE)
    v = jnp.zeros((cfg.n_dim,), DTYPE) if v0 is None else jnp.asarray(v0, DTYPE)
    if p.shape != (cfg.n_dim,) or v.shape != (cfg.n_dim,):
        raise ValueError(f"p0 and v0 must have shape {(cfg.n_dim,)}")
    return State(p, v, key, jnp.zeros((), jnp.int32))


def step(cfg: SailorConfig, state: State, action) -> tuple[State, Info]:
    """Advance one physics step of length cfg.dt.

    Order: split key; v' = v + dt(-beta v + B a) + dv^w; p' = p + dt (v + v') / 2.
    """
    a = jnp.asarray(action, DTYPE)
    if a.shape != (cfg.n_act,):
        raise ValueError(f"action must have shape {(cfg.n_act,)}, got {a.shape}")
    key, sub = jax.random.split(state.key)
    t = state.t_step.astype(DTYPE) * cfg.dt
    dv_w = cfg.disturbance.increment(sub, t, state.p, cfg.dt, cfg.n_dim)
    force = cfg.actuator @ a
    v_new = state.v + cfg.dt * (-cfg.damping * state.v + force) + dv_w
    p_new = state.p + 0.5 * cfg.dt * (state.v + v_new)
    err = jnp.linalg.norm(p_new - cfg.target)
    new_state = State(p_new, v_new, key, state.t_step + 1)
    info = Info(error=err, sq_error=err * err, force=force + dv_w / cfg.dt)
    return new_state, info


def rollout(cfg: SailorConfig, state: State, actions) -> tuple[State, Info]:
    """Scan ``step`` over actions of shape (T, n_act). Returns final state and stacked Info."""
    actions = jnp.asarray(actions, DTYPE)
    if actions.ndim != 2 or actions.shape[1] != cfg.n_act:
        raise ValueError(f"actions must have shape (T, {cfg.n_act}), got {actions.shape}")

    def body(s, a):
        return step(cfg, s, a)

    return jax.lax.scan(body, state, actions)
