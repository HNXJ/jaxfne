"""Hybrid kernel LFP: the laminar LFP of passive multicompartment cells driven by point-network spikes.

The network runs on point neurons; the LFP is computed separately from multicompartment cells
(the hybrid scheme of Hagen et al. 2016, Cereb. Cortex 26:4461). Each connection class (presynaptic
population onto a postsynaptic cell type, synapses on the compartments within a depth range) has a
kernel: the extracellular potential at each contact after one presynaptic spike. The LFP is the sum
over classes of the presynaptic spike count convolved with that class's kernel.

Cells are vertical cables (compartments are segments along the depth axis z, in um below the pia).
A synapse injects a current-based exponential input ``exp(-t / tau_syn)`` (nA; positive is inward,
depolarizing) into a passive cable:

    C_i dV_i/dt = -gL_i V_i + sum_j g_ij (V_j - V_i) + I_i(t)

solved exactly on the time grid (matrix exponential of the system augmented with the synaptic
trace). The transmembrane current of compartment i (capacitive + leak + synaptic, outward) equals
the axial current flowing into it, ``i_m = sum_j g_ij (V_j - V_i)``, and sums to zero over the cell.
Each compartment's current is spread uniformly along its segment (line source) in an infinite
homogeneous medium of conductivity sigma:

    phi_c = 1 / (4 pi sigma) * sum_k i_k / L_k * [asinh((z_k1 - z_c) / rho) - asinh((z_k0 - z_c) / rho)]

averaged over the lateral distances rho of the cells from the electrode axis.

Units: um, ms, nA, uF/cm^2, ohm cm^2, ohm cm, S/m; potentials in mV
(1 nA / (4 pi * 1 S/m * 1 um) = 1/(4 pi) mV). Every biophysical parameter is a required argument;
this module holds no biological defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

__all__ = [
    "CableCell",
    "cable_cell",
    "profile_cell",
    "synapse_weights_in_depth_range",
    "passive_membrane_currents",
    "line_source_matrix",
    "synapse_kernel",
    "kernel_lfp",
]


@dataclass(frozen=True)
class CableCell:
    """Vertical multicompartment cell.

    ``z0``/``z1`` are each compartment's end depths (um below the pia), ``diam`` its diameter (um),
    ``parent`` the index of the compartment it attaches to (-1 for the root, parents precede
    children). ``area_factor`` (default 1) multiplies the membrane area ``pi * diam * length`` while
    ``diam`` alone sets the axial resistance: a compartment can stand for several branches at one depth
    (or for spines).
    """

    z0: np.ndarray
    z1: np.ndarray
    diam: np.ndarray
    parent: tuple[int, ...]
    area_factor: np.ndarray | None = None

    def __post_init__(self) -> None:
        z0, z1, d = (np.asarray(a, np.float64) for a in (self.z0, self.z1, self.diam))
        n = len(self.parent)
        if not (z0.shape == z1.shape == d.shape == (n,)) or n == 0:
            raise ValueError("z0, z1, diam and parent must have one entry per compartment")
        af = np.ones(n) if self.area_factor is None else np.asarray(self.area_factor, np.float64)
        if af.shape != (n,) or not np.all(np.isfinite(af)) or not np.all(af > 0):
            raise ValueError("area_factor must be finite and positive, one per compartment")
        object.__setattr__(self, "area_factor", af)
        if self.parent[0] != -1 or any(not (0 <= p < k) for k, p in enumerate(self.parent) if k > 0):
            raise ValueError("parent[0] must be -1 and every other parent must precede its child")
        if not (np.all(np.isfinite(z0)) and np.all(np.isfinite(z1))) or np.any(np.abs(z1 - z0) <= 0):
            raise ValueError("every compartment needs finite depths and a nonzero length")
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError("every compartment diameter must be finite and positive")
        object.__setattr__(self, "z0", z0)
        object.__setattr__(self, "z1", z1)
        object.__setattr__(self, "diam", d)

    @property
    def n_comp(self) -> int:
        return len(self.parent)

    @property
    def length(self) -> np.ndarray:
        return np.abs(self.z1 - self.z0)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.z0 + self.z1)

    @property
    def area_cm2(self) -> np.ndarray:
        return np.pi * self.diam * self.length * self.area_factor * 1e-8


def cable_cell(sections: Sequence[Mapping]) -> CableCell:
    """Build a :class:`CableCell` from sections, each split into equal compartments.

    Each section is a mapping with ``z0``, ``z1`` (um), ``diam`` (um), ``n`` (compartments),
    ``parent`` (index of an earlier section, or -1 for the first) and optionally ``area_factor``.
    A section attaches to the last compartment of its parent section.
    """
    z0, z1, diam, factor, parent, last = [], [], [], [], [], []
    for s_idx, s in enumerate(sections):
        n = int(s["n"])
        p = int(s["parent"])
        if n < 1 or (s_idx == 0) != (p == -1) or p >= s_idx:
            raise ValueError(f"section {s_idx}: n must be >= 1, only section 0 has parent -1, parents precede")
        edges = np.linspace(float(s["z0"]), float(s["z1"]), n + 1)
        for k in range(n):
            parent.append(-1 if (p == -1 and k == 0) else (len(z0) - 1 if k > 0 else last[p]))
            z0.append(edges[k])
            z1.append(edges[k + 1])
            diam.append(float(s["diam"]))
            factor.append(float(s.get("area_factor", 1.0)))
        last.append(len(z0) - 1)
    return CableCell(np.array(z0), np.array(z1), np.array(diam), tuple(parent), np.array(factor))


def profile_cell(soma_z_um: float, soma_diam_um: float, soma_length_um: float,
                 branches: Sequence[Mapping]) -> CableCell:
    """Build a :class:`CableCell` from depth profiles: a soma compartment centred at ``soma_z_um`` and
    one chain per branch (e.g. up and down), each attached to the soma.

    A branch maps ``edges_um`` (n + 1 depths, strictly monotone, the first on the soma's surface) to
    ``area_um2`` (membrane area per bin, > 0) and ``diam_um`` (axial-equivalent diameter per bin, > 0,
    e.g. Rall's ``(sum d^1.5)^(2/3)`` over the branches crossing the bin). Each bin becomes a compartment
    with that diameter and the area factor that restores its membrane area.
    """
    half = 0.5 * _positive("soma_length_um", soma_length_um)
    z0, z1, diam, factor = [soma_z_um - half], [soma_z_um + half], [float(soma_diam_um)], [1.0]
    parent = [-1]
    for b in branches:
        e = np.asarray(b["edges_um"], np.float64)
        area = np.asarray(b["area_um2"], np.float64)
        d = np.asarray(b["diam_um"], np.float64)
        if e.ndim != 1 or len(e) < 2 or area.shape != (len(e) - 1,) or d.shape != (len(e) - 1,):
            raise ValueError("a branch needs n + 1 edges and n areas and diameters")
        step = np.diff(e)
        if not (np.all(step > 0) or np.all(step < 0)):
            raise ValueError("branch edges must be strictly monotone")
        if not (np.isclose(abs(e[0] - soma_z_um), half, rtol=0, atol=1e-9)):
            raise ValueError("a branch must start on the soma's surface")
        if not (np.all(area > 0) and np.all(d > 0)):
            raise ValueError("branch areas and diameters must be positive")
        for k in range(len(area)):
            parent.append(0 if k == 0 else len(z0) - 1)
            z0.append(e[k])
            z1.append(e[k + 1])
            diam.append(d[k])
            factor.append(area[k] / (np.pi * d[k] * abs(step[k])))
    return CableCell(np.array(z0), np.array(z1), np.array(diam), tuple(parent), np.array(factor))


def synapse_weights_in_depth_range(cell: CableCell, z_min: float, z_max: float) -> np.ndarray:
    """Synapse fraction per compartment: proportional to membrane area over the compartments whose
    midpoint depth lies in ``[z_min, z_max]`` (um); sums to 1. A range holding no compartment is
    refused."""
    m = (cell.mid >= z_min) & (cell.mid <= z_max)
    if not m.any():
        raise ValueError(f"no compartment midpoint in depth range [{z_min}, {z_max}] um")
    w = np.where(m, cell.area_cm2, 0.0)
    return w / w.sum()


def _positive(name: str, v: float) -> float:
    v = float(v)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and positive; got {v!r}")
    return v


def passive_membrane_currents(
    cell: CableCell,
    syn_weights: np.ndarray,
    *,
    tau_syn_ms: float,
    dt_ms: float,
    n_steps: int,
    cm_uF_cm2: float,
    rm_ohm_cm2: float,
    ra_ohm_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Membrane potential (mV, from rest) and transmembrane current (nA, outward) per compartment,
    ``[n_steps, n_comp]`` at t = 0, dt, ..., after a synaptic current ``exp(-t / tau_syn)`` nA starts
    at t = 0, split over compartments by ``syn_weights`` (non-negative, summing to 1)."""
    tau, dt = _positive("tau_syn_ms", tau_syn_ms), _positive("dt_ms", dt_ms)
    cm, rm, ra = _positive("cm_uF_cm2", cm_uF_cm2), _positive("rm_ohm_cm2", rm_ohm_cm2), _positive("ra_ohm_cm", ra_ohm_cm)
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    w = np.asarray(syn_weights, np.float64)
    if w.shape != (cell.n_comp,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("syn_weights must be non-negative, one per compartment, summing to 1")
    n = cell.n_comp
    c_nF = cm * cell.area_cm2 * 1e3                     # uF/cm2 * cm2 = uF -> nF
    gl_uS = cell.area_cm2 / rm * 1e6                    # S -> uS
    r_half = ra * (cell.length * 1e-4 / 2.0) / (np.pi * (cell.diam * 1e-4 / 2.0) ** 2)   # ohm
    lap = np.zeros((n, n))
    for k, p in enumerate(cell.parent):
        if p >= 0:
            g = 1e6 / (r_half[k] + r_half[p])           # uS
            lap[k, k] += g
            lap[p, p] += g
            lap[k, p] -= g
            lap[p, k] -= g
    m = np.zeros((n + 1, n + 1))
    m[:n, :n] = -(np.diag(gl_uS) + lap) / c_nF[:, None]
    m[:n, n] = w / c_nF
    m[n, n] = -1.0 / tau
    phi = scipy.linalg.expm(m * dt)
    x = np.zeros(n + 1)
    x[n] = 1.0
    v = np.empty((n_steps, n))
    for t in range(n_steps):
        v[t] = x[:n]
        x = phi @ x
    return v, -(v @ lap.T)


def line_source_matrix(
    cell: CableCell,
    contact_depths_um: np.ndarray,
    *,
    radii_um: np.ndarray,
    sigma_S_m: float,
    min_distance_um: float,
) -> np.ndarray:
    """Potential (mV) at each contact per 1 nA of transmembrane current in each compartment,
    ``[n_contacts, n_comp]``, averaged with equal weight over cells at lateral distances ``radii_um``
    from the electrode axis (each floored at ``min_distance_um``). For cells uniform in a disk of radius
    R, pass equal-area radii ``R * sqrt((k + 0.5) / n)``."""
    sigma = _positive("sigma_S_m", sigma_S_m)
    dmin = _positive("min_distance_um", min_distance_um)
    zc = np.asarray(contact_depths_um, np.float64).reshape(-1, 1, 1)
    if zc.size == 0 or not np.all(np.isfinite(zc)):
        raise ValueError("contact_depths_um must be a non-empty finite array")
    r = np.asarray(radii_um, np.float64)
    if r.size == 0 or not (np.all(np.isfinite(r)) and np.all(r >= 0)):
        raise ValueError("radii_um must be a non-empty finite array of non-negative distances")
    rho = np.maximum(r.reshape(1, -1, 1), dmin)
    lo, hi = np.minimum(cell.z0, cell.z1), np.maximum(cell.z0, cell.z1)
    f = (np.arcsinh((hi - zc) / rho) - np.arcsinh((lo - zc) / rho)) / cell.length
    return f.mean(1) / (4.0 * np.pi * sigma)


def synapse_kernel(
    cell: CableCell,
    syn_weights: np.ndarray,
    contact_depths_um: np.ndarray,
    *,
    tau_syn_ms: float,
    dt_ms: float,
    n_steps: int,
    cm_uF_cm2: float,
    rm_ohm_cm2: float,
    ra_ohm_cm: float,
    radii_um: np.ndarray,
    sigma_S_m: float,
    min_distance_um: float,
) -> np.ndarray:
    """LFP kernel ``[n_contacts, n_steps]`` (mV) of one synapse of 1 nA peak current: the line-source
    potential of the cell's transmembrane currents after one presynaptic spike."""
    _, i_m = passive_membrane_currents(cell, syn_weights, tau_syn_ms=tau_syn_ms, dt_ms=dt_ms, n_steps=n_steps,
                                       cm_uF_cm2=cm_uF_cm2, rm_ohm_cm2=rm_ohm_cm2, ra_ohm_cm=ra_ohm_cm)
    mat = line_source_matrix(cell, contact_depths_um, radii_um=radii_um, sigma_S_m=sigma_S_m,
                             min_distance_um=min_distance_um)
    return mat @ i_m.T


@jax.jit
def kernel_lfp(spike_counts: jax.Array, kernels: jax.Array) -> jax.Array:
    """LFP ``[T, n_contacts]`` from presynaptic spike counts ``[T, n_classes]`` (spikes per step of each
    connection class, already multiplied by its synapse count and signed peak current) and kernels
    ``[n_classes, n_contacts, K]``: the causal convolution summed over classes, truncated to T,
    computed by FFT over T + K - 1 points."""
    t, k = spike_counts.shape[0], kernels.shape[-1]
    n = t + k - 1
    s = jnp.fft.rfft(spike_counts, n=n, axis=0)                 # [F, C]
    h = jnp.fft.rfft(kernels, n=n, axis=-1)                     # [C, P, F]
    return jnp.fft.irfft(jnp.einsum("fc,cpf->fp", s, h), n=n, axis=0)[:t]
