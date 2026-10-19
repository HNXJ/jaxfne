"""Column tiling and spherical-head MEG and EEG forward models.

One simulated column becomes a cortex: :func:`column_dipole` reduces the column's
sources to a current dipole along its axis, :func:`folded_sphere_surface` places
column copies on a folded surface with outward normals, and
:func:`tile_column_dipoles` copies simulated realizations onto those columns, and
:func:`signals_dipole` takes the dipole straight from a simulation's ``Signals``.
:func:`meg_sphere_leadfield` (Sarvas 1987) and :func:`eeg_sphere_leadfield`
(homogeneous sphere, Frank 1952) map oriented dipoles to sensors.

The geometry and leadfield functions check their inputs eagerly, so call them outside
`jax.jit` (their outputs are plain arrays that jitted code can use).

Units: positions in metres, dipoles in A·m, MEG leadfields in T/(A·m), EEG
leadfields in V/(A·m). :func:`column_dipole` returns the source's relative
units; an absolute dipole needs an explicit, declared scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

MU0 = 4e-7 * np.pi


def column_dipole(source: jax.Array, positions: jax.Array, axis: int = 2) -> jax.Array:
    """Current dipole of one column along its axis: ``p(t) = Σ_i Q_i(t) (z_i − z̄)``.

    ``source`` is ``[T, N]`` (per-neuron sources), ``positions`` ``[N, 3]``; ``z``
    is ``positions[:, axis]``. The first moment about the mean depth cancels the
    column's common-mode current. Output ``[T]`` in the source's relative units.
    """
    q = jnp.asarray(source)
    pos = jnp.asarray(positions)
    if q.ndim != 2 or pos.ndim != 2 or pos.shape[0] != q.shape[1]:
        raise ValueError(f"source [T, N] and positions [N, 3] required, got {q.shape} and {pos.shape}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    z = pos[:, axis]
    return q @ (z - z.mean())


def signals_dipole(signals: Any, model: Any, axis: int = 2) -> jax.Array:
    """Column dipole ``[T]`` of a simulation: :func:`column_dipole` of its sources at the model's positions.

    The bridge from a laminar run to MEG and EEG: the sources that make the
    run's LFP also give its dipole, which :class:`SphereForward` maps to sensors
    once scaled to A·m. ``signals`` must carry sources (``record_sources=True``).
    """
    sources = getattr(signals, "sources", None)
    if sources is None:
        raise ValueError("signals carry no sources; simulate with record_sources=True")
    params = getattr(model, "params", None)
    if not isinstance(params, dict) or "positions" not in params:
        raise ValueError("model must carry params['positions'] (a constructed jaxfne Model)")
    return column_dipole(sources, params["positions"], axis=axis)


def _unit(v: jax.Array) -> jax.Array:
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def folded_sphere_surface(
    n_columns: int,
    *,
    radius_m: float = 0.068,
    fold_amplitude_m: float = 0.009,
    fold_frequencies: tuple[int, int] = (12, 16),
    max_polar_deg: float = 110.0,
) -> tuple[jax.Array, jax.Array]:
    """Column positions and outward normals on a folded sphere ``[M, 3]`` each.

    Radius ``r(θ, φ) = radius_m + fold_amplitude_m · cos(kφ φ + 2.5 sin 3θ) · sin(kθ θ)``
    with ``(kφ, kθ) = fold_frequencies``; columns sit on a Fibonacci lattice from
    the pole down to ``max_polar_deg``. A normal points from deep to superficial
    layers, so folding sets each column's orientation (radial on crowns,
    tangential on sulcal walls). A synthetic cortex, not an individual anatomy.
    """
    if n_columns < 1:
        raise ValueError(f"n_columns must be >= 1, got {n_columns}")
    if radius_m <= 0 or fold_amplitude_m < 0 or fold_amplitude_m >= radius_m:
        raise ValueError("need radius_m > 0 and 0 <= fold_amplitude_m < radius_m")
    if not 0.0 < max_polar_deg <= 180.0:
        raise ValueError(f"max_polar_deg must be in (0, 180], got {max_polar_deg}")
    k_phi, k_theta = (int(k) for k in fold_frequencies)

    def surf(th, ph):
        r = radius_m + fold_amplitude_m * jnp.cos(k_phi * ph + 2.5 * jnp.sin(3 * th)) * jnp.sin(k_theta * th)
        return jnp.stack([r * jnp.sin(th) * jnp.cos(ph), r * jnp.sin(th) * jnp.sin(ph), r * jnp.cos(th)], -1)

    i = jnp.arange(n_columns) + 0.5
    cos_max = float(np.cos(np.radians(max_polar_deg)))
    th = jnp.arccos(1.0 - i / n_columns * (1.0 - cos_max))
    ph = jnp.pi * (1.0 + 5.0**0.5) * i
    pos = surf(th, ph)
    d_th = jax.vmap(jax.jacfwd(surf, argnums=0))(th, ph)
    d_ph = jax.vmap(jax.jacfwd(surf, argnums=1))(th, ph)
    nrm = _unit(jnp.cross(d_th, d_ph))
    nrm = nrm * jnp.sign(jnp.sum(nrm * pos, axis=1))[:, None]
    if not bool(jnp.all(jnp.isfinite(nrm))) or not bool(jnp.all(jnp.abs(jnp.linalg.norm(nrm, axis=1) - 1) < 1e-3)):
        raise ValueError("degenerate surface normal; change n_columns or the fold parameters")
    return pos, nrm


def tile_column_dipoles(
    realizations: jax.Array,
    n_columns: int,
    key: jax.Array,
    *,
    in_phase: jax.Array | None = None,
    in_phase_realization: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Copy simulated column dipoles onto ``n_columns`` columns.

    ``realizations`` is ``[K, T]`` (one dipole time course per simulated seed).
    Each column draws a realization and a circular time shift from ``key``;
    columns where the boolean ``in_phase`` ``[M]`` is set share realization
    ``in_phase_realization`` with zero shift (a coherent patch). Returns
    ``(dipoles [T, M], assignment [M], shift [M])``.
    """
    r = jnp.asarray(realizations)
    if r.ndim != 2:
        raise ValueError(f"realizations must be [K, T], got {r.shape}")
    k_real, t_len = r.shape
    if n_columns < 1:
        raise ValueError(f"n_columns must be >= 1, got {n_columns}")
    if not 0 <= in_phase_realization < k_real:
        raise ValueError(f"in_phase_realization must index one of {k_real} realizations")
    if in_phase is None and in_phase_realization != 0:
        raise ValueError("in_phase_realization needs an in_phase mask")
    k_a, k_s = jax.random.split(key)
    assign = jax.random.randint(k_a, (n_columns,), 0, k_real)
    shift = jax.random.randint(k_s, (n_columns,), 0, t_len)
    if in_phase is not None:
        mask = jnp.asarray(in_phase, dtype=bool)
        if mask.shape != (n_columns,):
            raise ValueError(f"in_phase must be [{n_columns}], got {mask.shape}")
        assign = jnp.where(mask, in_phase_realization, assign)
        shift = jnp.where(mask, 0, shift)
    t_idx = (jnp.arange(t_len)[:, None] - shift[None, :]) % t_len
    return r[assign[None, :], t_idx], assign, shift


def sphere_sensor_cap(n: int, radius_m: float, max_polar_deg: float = 98.6) -> tuple[jax.Array, jax.Array]:
    """``n`` sensors on a spherical cap (Fibonacci lattice) and their radial normals."""
    if n < 1 or radius_m <= 0:
        raise ValueError("need n >= 1 and radius_m > 0")
    j = jnp.arange(n) + 0.5
    cos_max = float(np.cos(np.radians(max_polar_deg)))
    th = jnp.arccos(1.0 - j / n * (1.0 - cos_max))
    ph = jnp.pi * (1.0 + 5.0**0.5) * j
    nrm = jnp.stack([jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)], -1)
    return radius_m * nrm, nrm


def _check_dipoles(dipole_pos: jax.Array, dipole_ori: jax.Array) -> tuple[jax.Array, jax.Array]:
    pos, ori = jnp.asarray(dipole_pos), jnp.asarray(dipole_ori)
    if pos.ndim != 2 or pos.shape[1] != 3 or ori.shape != pos.shape:
        raise ValueError(f"dipole_pos and dipole_ori must both be [M, 3], got {pos.shape} and {ori.shape}")
    return pos, ori


def meg_sphere_leadfield(
    sensor_pos: jax.Array, sensor_ori: jax.Array, dipole_pos: jax.Array, dipole_ori: jax.Array
) -> jax.Array:
    """MEG leadfield ``[S, M]`` in T/(A·m) for a conducting sphere centred at the origin.

    Sarvas (1987), doi:10.1088/0031-9155/32/1/004, closed form, which includes the volume currents and does not
    depend on the conductivity profile: ``B · sensor_ori`` per unit dipole along
    ``dipole_ori``. Radial dipoles give zero field. Sensors must lie outside the
    dipoles' radius.
    """
    rs, ns = jnp.asarray(sensor_pos), jnp.asarray(sensor_ori)
    if rs.ndim != 2 or rs.shape[1] != 3 or ns.shape != rs.shape:
        raise ValueError(f"sensor_pos and sensor_ori must both be [S, 3], got {rs.shape} and {ns.shape}")
    r0, q = _check_dipoles(dipole_pos, dipole_ori)
    if float(jnp.min(jnp.linalg.norm(rs, axis=1))) <= float(jnp.max(jnp.linalg.norm(r0, axis=1))):
        raise ValueError("every sensor must lie farther from the centre than every dipole")

    def one(r0_m, q_m):
        a_vec = rs - r0_m
        a = jnp.linalg.norm(a_vec, axis=1)
        r = jnp.linalg.norm(rs, axis=1)
        adotr = jnp.sum(a_vec * rs, axis=1)
        f = a * (r * a + r**2 - rs @ r0_m)
        grad_f = (a**2 / r + adotr / a + 2 * a + 2 * r)[:, None] * rs - (a + 2 * r + adotr / a)[:, None] * r0_m
        qxr0 = jnp.cross(q_m, r0_m)
        b = MU0 / (4 * jnp.pi * f[:, None] ** 2) * (f[:, None] * qxr0 - (rs @ qxr0)[:, None] * grad_f)
        return jnp.sum(b * ns, axis=1)

    return jax.vmap(one, out_axes=1)(r0, q)


def eeg_sphere_leadfield(
    electrode_pos: jax.Array,
    dipole_pos: jax.Array,
    dipole_ori: jax.Array,
    *,
    conductivity_s_per_m: float,
    average_reference: bool = True,
) -> jax.Array:
    """EEG leadfield ``[E, M]`` in V/(A·m) for a homogeneous sphere centred at the origin.

    Electrodes must lie on one sphere (radius ``R``, 0.1% tolerance) enclosing
    the dipoles. Frank (1952), doi:10.1063/1.1702037, closed form, the sum of the Neumann series
    ``Σ (2n+1)/n (b/R)^n P_n(cos γ)``::

        V = q · [2 d/|d|³ + (r + R d/|d|) / (R (R² − r·r₀ + R|d|))] / (4π σ),  d = r − r₀.

    ``average_reference`` subtracts the electrode mean. A single shell: skull
    and scalp layers are not modelled.
    """
    if conductivity_s_per_m <= 0:
        raise ValueError(f"conductivity_s_per_m must be > 0, got {conductivity_s_per_m}")
    re = jnp.asarray(electrode_pos)
    if re.ndim != 2 or re.shape[1] != 3:
        raise ValueError(f"electrode_pos must be [E, 3], got {re.shape}")
    r0, q = _check_dipoles(dipole_pos, dipole_ori)
    radii = jnp.linalg.norm(re, axis=1)
    big_r = float(jnp.mean(radii))
    if float(jnp.max(jnp.abs(radii - big_r))) > 1e-3 * big_r:
        raise ValueError("electrodes must lie on one sphere centred at the origin")
    if float(jnp.max(jnp.linalg.norm(r0, axis=1))) >= big_r:
        raise ValueError("every dipole must lie inside the electrode sphere")

    def one(r0_m, q_m):
        d_vec = re - r0_m
        d = jnp.linalg.norm(d_vec, axis=1)
        den = big_r * (big_r**2 - re @ r0_m + big_r * d)
        g = 2 * d_vec / d[:, None] ** 3 + (re + big_r * d_vec / d[:, None]) / den[:, None]
        return (g @ q_m) / (4 * jnp.pi * conductivity_s_per_m)

    lead = jax.vmap(one, out_axes=1)(r0, q)
    if average_reference:
        lead = lead - lead.mean(axis=0, keepdims=True)
    return lead


@dataclass(frozen=True)
class SphereForward:
    """An oriented-dipole forward map ``y = dipoles @ Wᵀ`` with its model and units.

    ``W`` comes from :func:`meg_sphere_leadfield` or :func:`eeg_sphere_leadfield`;
    ``modality`` is ``"meg"`` or ``"eeg"``. Output amplitudes are physical only
    when the dipoles are in calibrated A·m.
    """

    modality: str
    W: jax.Array
    model: str
    units: str

    def __post_init__(self) -> None:
        if self.modality not in ("meg", "eeg"):
            raise ValueError(f"modality must be 'meg' or 'eeg', got {self.modality!r}")
        if jnp.asarray(self.W).ndim != 2:
            raise ValueError("W must be [n_sensors, n_dipoles]")

    def apply(self, dipoles: jax.Array) -> jax.Array:
        """Sensor signals ``[T, n_sensors]`` from dipole moments ``[T, n_dipoles]`` (A·m)."""
        p = jnp.asarray(dipoles)
        if p.ndim != 2 or p.shape[1] != self.W.shape[1]:
            raise ValueError(f"dipoles must be [T, {self.W.shape[1]}], got {p.shape}")
        return p @ jnp.asarray(self.W).T

    def report(self) -> dict[str, Any]:
        """Model, units and claim level of this forward map."""
        return {
            "modality": self.modality,
            "model": self.model,
            "leadfield_units": self.units,
            "orientation_claim": "dipole_along_column_normal",
            "matrix_shape": [int(self.W.shape[0]), int(self.W.shape[1])],
            "head_model": "sphere",
            "physical_amplitude_calibrated": False,
        }


__all__ = [
    "MU0",
    "SphereForward",
    "column_dipole",
    "eeg_sphere_leadfield",
    "folded_sphere_surface",
    "meg_sphere_leadfield",
    "signals_dipole",
    "sphere_sensor_cap",
    "tile_column_dipoles",
]
