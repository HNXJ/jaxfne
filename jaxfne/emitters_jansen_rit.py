"""Jansen-Rit 1995 neural-mass column emitter (function-level only).

One cortical column of Jansen and Rit (1995, Biol Cybern 73:357-366,
doi:10.1007/BF00199471): three populations (pyramidal, excitatory and
inhibitory interneurons), each synapse a second-order linear kernel

    h_e(t) = A a t e^{-a t},    h_i(t) = B b t e^{-b t},

written as the ODE pair y'' = G S(.) - 2 c y' - c^2 y with (G, c) = (A a, a)
or (B b, b). With y0..y2 the postsynaptic potentials and y3..y5 their
velocities::

    dy0/dt = y3
    dy3/dt = A a S(y1 - y2) - 2 a y3 - a^2 y0
    dy1/dt = y4
    dy4/dt = A a (p + C2 S(C1 y0)) - 2 a y4 - a^2 y1
    dy2/dt = y5
    dy5/dt = B b C4 S(C3 y0) - 2 b y5 - b^2 y2

with the sigmoid S(v) = 2 e0 / (1 + exp(r (v0 - v))). C1..C4 are
connectivity constants; p(t) is the external input onto the excitatory
branch. The EEG-like output is y1 - y2.

Units: dt is in seconds, a and b in 1/s, A and B in mV, e0 in spikes/s,
v0 in mV, r in 1/mV. Any consistent time unit works as long as dt and 1/a
share it. Integration is fixed-step classical RK4 with p held constant
over each step (p is the per-step sample array ``drive``). The module
holds no random state: the drive array fully determines the trajectory,
so one drive array gives one bit-identical output.

Function-level only; Configuration/construct wiring lives elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class JansenRitParams:
    """Jansen-Rit column parameters (drive array excluded: it varies per run).

    All fields are required. ``__post_init__`` runs the same scalar checks
    as :func:`check_jansen_rit_params`; ``drive`` is checked there because
    its length must equal ``n_steps``.
    """

    A: float
    B: float
    a: float
    b: float
    C1: float
    C2: float
    C3: float
    C4: float
    e0: float
    v0: float
    r: float
    dt: float
    n_steps: int

    def __post_init__(self) -> None:
        """Validate every scalar field, refusing non-finite or out-of-range values."""
        _check_jansen_rit_scalars(
            self.A, self.B, self.a, self.b,
            self.C1, self.C2, self.C3, self.C4,
            self.e0, self.v0, self.r, self.dt, self.n_steps,
        )


def _checked_positive(name: str, value: float) -> float:
    """Finite value strictly above zero, refusing the rest by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite real number > 0 (bools refused, never coerced)"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite value > 0"
        )
    if float(value) <= 0.0:
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a value > 0"
        )
    return float(value)


def _checked_nonnegative(name: str, value: float) -> float:
    """Finite value at or above zero, refusing the rest by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite real number >= 0 (bools refused, never coerced)"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite value >= 0"
        )
    if float(value) < 0.0:
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a value >= 0"
        )
    return float(value)


def _checked_threshold(name: str, value: float) -> float:
    """Finite real number of any sign, refusing the rest by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite real number (bools refused, never coerced)"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"jansen_rit param {name}={value!r} is not realized: it takes "
            "a finite value"
        )
    return float(value)


def _check_jansen_rit_scalars(
    A: float,
    B: float,
    a: float,
    b: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    n_steps: int,
) -> dict[str, float]:
    """Validate the scalar parameters, returning them as floats."""
    out: dict[str, float] = {
        "A": _checked_positive("A", A),
        "B": _checked_positive("B", B),
        "a": _checked_positive("a", a),
        "b": _checked_positive("b", b),
        "C1": _checked_nonnegative("C1", C1),
        "C2": _checked_nonnegative("C2", C2),
        "C3": _checked_nonnegative("C3", C3),
        "C4": _checked_nonnegative("C4", C4),
        "e0": _checked_positive("e0", e0),
        "v0": _checked_threshold("v0", v0),
        "r": _checked_positive("r", r),
        "dt": _checked_positive("dt", dt),
    }
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError(
            f"jansen_rit param n_steps={n_steps!r} is not realized: it takes "
            "an integer >= 1 (bools refused, never coerced)"
        )
    if int(n_steps) < 1:
        raise ValueError(
            f"jansen_rit param n_steps={n_steps!r} is not realized: it takes "
            "an integer >= 1"
        )
    out["n_steps"] = float(int(n_steps))
    return out


def check_jansen_rit_params(
    A: float,
    B: float,
    a: float,
    b: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    n_steps: int,
    drive: np.ndarray,
) -> dict[str, float]:
    """Validate a full Jansen-Rit run specification.

    Gains and rate constants (``A``, ``B``, ``a``, ``b``, ``e0``, ``r``,
    ``dt``) must be finite and > 0; couplings (``C1``..``C4``) finite and
    >= 0; ``v0`` finite; ``n_steps`` an integer >= 1. ``drive`` is the
    per-step p(t) array of shape ``(n_steps,)`` with all samples finite.
    Returns the scalars as floats.
    """
    out = _check_jansen_rit_scalars(
        A, B, a, b, C1, C2, C3, C4, e0, v0, r, dt, n_steps
    )
    steps = int(n_steps) if not isinstance(n_steps, bool) else -1
    arr = np.asarray(drive, dtype=np.float64)
    if arr.shape != (steps,):
        raise ValueError(
            f"jansen_rit param drive has shape {arr.shape}; it takes "
            f"shape ({steps},) matching n_steps"
        )
    if not bool(np.isfinite(arr).all()):
        raise ValueError(
            "jansen_rit param drive is not realized: it takes finite samples"
        )
    return out


def jansen_rit_sigmoid(
    v: float,
    e0: float,
    v0: float,
    r: float,
) -> float:
    """S(v) = 2 e0 / (1 + exp(r (v0 - v))), the Jansen-Rit firing-rate curve."""
    return 2.0 * e0 / (1.0 + jnp.exp(r * (v0 - v)))


def jansen_rit_rhs(
    state: np.ndarray,
    p: float,
    A: float,
    B: float,
    a: float,
    b: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    e0: float,
    v0: float,
    r: float,
) -> np.ndarray:
    """Right-hand side of the six-state Jansen-Rit column at input level ``p``.

    ``state`` is ``[y0, y1, y2, d0, d1, d2]`` (potentials, then velocities).
    """
    y0, y1, y2, d0, d1, d2 = state
    drive_pyramidal = jansen_rit_sigmoid(y1 - y2, e0, v0, r)
    drive_exc = jansen_rit_sigmoid(C1 * y0, e0, v0, r)
    drive_inh = jansen_rit_sigmoid(C3 * y0, e0, v0, r)
    return jnp.stack(
        [
            d0,
            d1,
            d2,
            A * a * drive_pyramidal - 2.0 * a * d0 - a * a * y0,
            A * a * (p + C2 * drive_exc) - 2.0 * a * d1 - a * a * y1,
            B * b * C4 * drive_inh - 2.0 * b * d2 - b * b * y2,
        ]
    )


def _rk4_matrix_step(
    state: np.ndarray,
    p: float,
    dt: float,
    A: float,
    B: float,
    a: float,
    b: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    e0: float,
    v0: float,
    r: float,
) -> np.ndarray:
    """One classical RK4 step of the column with input ``p`` held constant."""

    def rhs(s: np.ndarray) -> np.ndarray:
        return jansen_rit_rhs(
            s, p, A, B, a, b, C1, C2, C3, C4, e0, v0, r
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_jansen_rit(
    A: float,
    B: float,
    a: float,
    b: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    n_steps: int,
    drive: np.ndarray,
) -> dict[str, np.ndarray]:
    """Simulate one Jansen-Rit column from rest with fixed-step RK4.

    ``drive`` is the per-step p(t) array of shape ``(n_steps,)``. The run
    starts at the zero state and steps through ``drive`` with
    :func:`jax.lax.scan` under :func:`jax.jit`. Returns ``{"y": (n_steps, 6)
    float64, "eeg": (n_steps,) float64}`` with ``eeg = y1 - y2``.
    """
    spec = check_jansen_rit_params(
        A, B, a, b, C1, C2, C3, C4, e0, v0, r, dt, n_steps, drive
    )
    step_dt = spec["dt"]
    spec_tuple = (
        spec["A"], spec["B"], spec["a"], spec["b"],
        spec["C1"], spec["C2"], spec["C3"], spec["C4"],
        spec["e0"], spec["v0"], spec["r"],
    )

    @jax.jit
    def _loop(
        y_init: np.ndarray, p_arr: np.ndarray
    ) -> np.ndarray:
        def body(
            state: np.ndarray, p: float
        ) -> tuple[np.ndarray, np.ndarray]:
            nxt = _rk4_matrix_step(state, p, step_dt, *spec_tuple)
            return nxt, nxt

        _, ys = jax.lax.scan(body, y_init, p_arr)
        return ys

    y_init = jnp.zeros((6,), dtype=jnp.float64)
    p_arr = jnp.asarray(np.asarray(drive, dtype=np.float64), dtype=jnp.float64)
    y = np.asarray(_loop(y_init, p_arr), dtype=np.float64)
    return {"y": y, "eeg": y[:, 1] - y[:, 2]}


def linear_kernel_rhs(
    state: np.ndarray,
    u: float,
    A: float,
    a: float,
) -> np.ndarray:
    """Right-hand side of one excitatory synaptic kernel from rest.

    ``state`` is ``[y, v]``; the pair reads y'' = A a u - 2 a v - a^2 y,
    whose impulse response is h(t) = A a t e^{-a t}.
    """
    y, v = state
    return jnp.stack([v, A * a * u - 2.0 * a * v - a * a * y])


def simulate_linear_kernel(
    A: float,
    a: float,
    dt: float,
    n_steps: int,
    drive: np.ndarray,
) -> dict[str, np.ndarray]:
    """Simulate one linear synaptic kernel from rest with fixed-step RK4.

    ``drive`` is the per-step input array of shape ``(n_steps,)``; an
    approximate impulse (``drive[0] = 1 / dt``, zero elsewhere) traces
    h(t) = A a t e^{-a t} on the grid. Returns ``{"y": (n_steps,)
    float64}``.
    """
    gain = _checked_positive("A", A)
    rate = _checked_positive("a", a)
    step = _checked_positive("dt", dt)
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError(
            f"jansen_rit param n_steps={n_steps!r} is not realized: it takes "
            "an integer >= 1 (bools refused, never coerced)"
        )
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(
            f"jansen_rit param n_steps={n_steps!r} is not realized: it takes "
            "an integer >= 1"
        )
    arr = np.asarray(drive, dtype=np.float64)
    if arr.shape != (steps,):
        raise ValueError(
            f"jansen_rit param drive has shape {arr.shape}; it takes "
            f"shape ({steps},) matching n_steps"
        )
    if not bool(np.isfinite(arr).all()):
        raise ValueError(
            "jansen_rit param drive is not realized: it takes finite samples"
        )

    @jax.jit
    def _loop(
        s_init: np.ndarray, u_arr: np.ndarray
    ) -> np.ndarray:
        def body(
            state: np.ndarray, u: float
        ) -> tuple[np.ndarray, np.ndarray]:
            k1 = linear_kernel_rhs(state, u, gain, rate)
            k2 = linear_kernel_rhs(state + 0.5 * step * k1, u, gain, rate)
            k3 = linear_kernel_rhs(state + 0.5 * step * k2, u, gain, rate)
            k4 = linear_kernel_rhs(state + step * k3, u, gain, rate)
            nxt = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return nxt, nxt[0]

        _, ys = jax.lax.scan(body, s_init, u_arr)
        return ys

    s_init = jnp.zeros((2,), dtype=jnp.float64)
    u_arr = jnp.asarray(arr, dtype=jnp.float64)
    return {"y": np.asarray(_loop(s_init, u_arr), dtype=np.float64)}
