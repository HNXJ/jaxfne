"""Hodgkin-Huxley network kernel: OU noise and graded tanh synapses on an edge list.

Reference: Nejat, Sherfey, Bastos 2026, PLoS One 21(8): e0354021 (GSDR),
Methods Eq. 1; the code that produced Fig. 6 (``HNXJ/GSDR`` ``misc/main.py``
run, read as data): stock Jaxley ``HH()`` plus one per-cell ``Inoise`` OU
channel, ``GradedAMPA``/``GradedGABAa`` with tanh activation. The HH kinetic
parameters below are the classic values assumed in
``artifacts/etudes/gsdr2026/PORT_SPEC.md`` A1 (the reference names ``C_m``
only); a fidelity fix can replace them.

Units are mV, ms, mS/cm2, uF/cm2, uA/cm2 (mS * mV = uA; uA / uF = mV / ms).
The paper's Eq. 1 sign convention holds for intrinsic and synaptic currents
(positive outward hyperpolarizes); the injected drive ``I_inj`` is
depolarizing-positive (inward), matching how the reference adds its stimulus.

Numerics follow the reference solver (jaxley 0.14.0, read as data):
gates advance by exact exponential Euler (``solver_gate``), voltages by
backward Euler with the ionic and synaptic currents linearized at frozen
gates (the reference ``bwd_euler`` split: channels stepped before voltage),
compartments init at ``v = -70`` mV (single-compartment gates at their rate
fixed point; multi-compartment gates at the reference default 0.2, since
the reference never calls ``init_state``), and the ``exp`` arguments are
clipped at 20
(``save_exp``). The HH rate shapes are the classic ones; ``eLeak = -54.3``
is the reference value (``channels/hh.py``).

Function-level only. Configuration, construct and simulate wiring is not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.errors
import jax.numpy as jnp
import numpy as np

# PORT_SPEC.md A1: reference HH values (jaxley channels/hh.py, read as data).
HH_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "E": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
    "Ig": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
    "Il": {
        "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0, "gL_mS_cm2": 0.3,
        "ENa_mV": 50.0, "EK_mV": -77.0, "EL_mV": -54.3, "Cm_uF_cm2": 1.0,
    },
}

_PARAM_FIELDS = (
    "gNa_mS_cm2", "gK_mS_cm2", "gL_mS_cm2",
    "ENa_mV", "EK_mV", "EL_mV", "Cm_uF_cm2",
)

# PORT_SPEC.md section 3, Fig. 6 run values (the builder defaults differ;
# the run values win): per-cell tau ~ U(10, 50), sigma = clip(U(-0.1, 0.1)).
# No in-tree code imports these three names (the GSDR runner draws from the
# same ranges via literals in run_gsdr.py:ou_hyperparams); they are kept as
# the documented run values, not wired defaults.
HH_OU_TAU_LO = 10.0
HH_OU_TAU_HI = 50.0
HH_OU_SIGMA_CLIP = 0.05

# PORT_SPEC.md A5: spike readout threshold of the reference loss.
HH_SPIKE_THR_MV = -20.0

# Resting voltage used for V0 when the caller passes none (reference v_init).
HH_V_REST_MV = -70.0

# Reference gate default state (jaxley 0.14.0 ``channels/hh.py`` default
# ``m = h = n = 0.2``). The reference integrate path never calls
# ``init_state``, so gates start here, not at their rate fixed point. The
# multi-compartment path uses this when ``gate_init`` is None; the
# single-compartment path keeps its rate-fixed-point init unless
# ``gate_init`` is given explicitly.
HH_GATE_DEFAULT = 0.2

# Clip of exp() arguments (the reference save_exp(max_value=20.0)).
_SAVE_EXP_MAX = 20.0


# Declared entry-point defaults with PORT_SPEC.md reference values (finding 3).
# Section 4, Fig. 6 run values (code-observed, all three graded classes):
# tanh slope 5.0 mV, threshold -20.0 mV; s(0) 0.1 for the connected classes
# (AMPA, GABAa; GABAb s(0) 0.01 is defined but never connected).
HH_GRADED_SLOPE_DEFAULT = 5.0
HH_GRADED_VTH_DEFAULT = -20.0
HH_GRADED_S0_DEFAULT = 0.1

# Section 3, Fig. 6 run value: OU sigma clip upper (clip(U(-0.1, 0.1), 0, 0.05));
# the lane works in uA/cm2, so the GSDR runner scales it by 1000 (mA/cm2 -> uA/cm2).
HH_OU_SIGMA_DEFAULT = 0.05

# Lane-local OU tau fallback (ms): used when ou_tau is absent and noise is
# not live (no ou_key, or effective sigma all zero). PORT_SPEC.md section 3
# gives a per-cell U(10, 50) range, not a single value, so a live noise run
# must pass ou_tau explicitly (refused when absent). Note the fallback is
# consumed when ou_key is present with zero effective sigma: the channel
# still steps, holding ou at 0.
HH_OU_TAU_FALLBACK_MS = 20.0


def _save_exp(x):
    """``exp(min(x, 20))``: the reference overflow guard (solver_gate)."""
    return jnp.exp(jnp.minimum(x, _SAVE_EXP_MAX))


def _hh_rate_coeffs(v):
    """Classic HH rate coefficients at absolute voltage ``v`` (elementwise).

    Same kinetics as the reference ``HH.m/h/n_gate`` (absolute-voltage form);
    ``vv = v + 65`` is the rest-relative frame.
    """
    vv = v + 65.0
    am = jnp.where(jnp.abs(25.0 - vv) < 1e-7, 1.0,
                   0.1 * (25.0 - vv) / (_save_exp((25.0 - vv) / 10.0) - 1.0))
    bm = 4.0 * _save_exp(-vv / 18.0)
    ah = 0.07 * _save_exp(-vv / 20.0)
    bh = 1.0 / (_save_exp((30.0 - vv) / 10.0) + 1.0)
    an = jnp.where(jnp.abs(10.0 - vv) < 1e-7, 0.1,
                   0.01 * (10.0 - vv) / (_save_exp((10.0 - vv) / 10.0) - 1.0))
    bn = 0.125 * _save_exp(-vv / 80.0)
    return am, bm, ah, bh, an, bn


def _exp_euler_gate(x, dt_ms, alpha, beta):
    """Exact exponential-Euler step for ``dx/dt = alpha(1-x) - beta x``."""
    tau = 1.0 / (alpha + beta)
    x_inf = alpha * tau
    e = _save_exp(-dt_ms / tau)
    return x * e + x_inf * (1.0 - e)


def _require_finite_dt(value: Any, name: str) -> None:
    """Refuse ``value`` unless it is finite and > 0 (eager calls only).

    Under ``jit`` the ``float()`` conversion fails loudly instead of
    validating; jitted use validates ``dt`` once at the entry point
    (:func:`simulate_hh` refuses a non-finite or non-positive ``dt_ms``).
    A ``str`` or ``bytes`` value is refused with ``TypeError``, since
    ``float()`` would parse ``"0.1"`` silently.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    try:
        dt = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


def hh_gate_inf(v):
    """Rate fixed point ``(m, h, n)`` at voltage ``v`` (the reference init)."""
    am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
    return (am / (am + bm), ah / (ah + bh), an / (an + bn))


def _check_gate_init(value: Any) -> float | None:
    """Validate the ``gate_init`` kernel argument; return a float or None."""
    if value is None:
        return None
    try:
        g = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"gate_init must be a finite number or None; got {value!r}"
        )
    if not math.isfinite(g):
        raise ValueError(
            f"gate_init must be a finite number or None; got {value!r}"
        )
    return g


class HHParams:
    """Per-neuron HH parameters: float arrays of shape [N], with cell labels."""

    def __init__(self, labels: tuple[str, ...], **columns: jax.Array) -> None:
        self.labels = labels
        for field in _PARAM_FIELDS:
            setattr(self, field, columns[field])

    _fields = _PARAM_FIELDS

    def _replace(self, **kwargs: jax.Array) -> "HHParams":
        columns = {f: kwargs.get(f, getattr(self, f)) for f in _PARAM_FIELDS}
        return HHParams(labels=self.labels, **columns)


def make_hh_params(
    labels: Sequence[str],
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> HHParams:
    """Build :class:`HHParams` for one cell type per neuron.

    Defaults: classic HH values (``HH_CELL_TYPE_DEFAULTS``; PORT_SPEC.md A1).

    ``overrides`` maps a cell type to a dict of field overrides (field names as
    in :class:`HHParams`). Unknown cell types, unknown fields, and overrides
    for a cell type absent from ``labels`` raise ``ValueError``.
    """
    labels = tuple(str(lab) for lab in labels)
    for lab in labels:
        if lab not in HH_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"unknown HH cell type {lab!r}; expected one of {sorted(HH_CELL_TYPE_DEFAULTS)}"
            )
    overrides = {} if overrides is None else overrides
    for ctype, fields in overrides.items():
        if ctype not in HH_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"overrides name unknown HH cell type {ctype!r}; "
                f"expected one of {sorted(HH_CELL_TYPE_DEFAULTS)}"
            )
        if ctype not in labels:
            raise ValueError(f"overrides name cell type {ctype!r}, which is not in labels {labels}")
        for field in fields:
            if field not in _PARAM_FIELDS:
                raise ValueError(
                    f"unknown HH parameter field {field!r} for cell type {ctype!r}; "
                    f"expected one of {_PARAM_FIELDS}"
                )
    columns = {}
    for field in _PARAM_FIELDS:
        values = [
            float(overrides.get(lab, {}).get(field, HH_CELL_TYPE_DEFAULTS[lab][field]))
            for lab in labels
        ]
        columns[field] = jnp.asarray(np.asarray(values, dtype=np.float64))
    return HHParams(labels=labels, **columns)


def hh_step(state, I_inj, params: HHParams, dt_ms: float):
    """One split step of the classic HH equations for a vector of cells.

    Gates advance by exact exponential Euler at frozen voltage; then voltage
    advances by backward Euler with the ionic current linearized at the new
    gates (``(v + dt*c/Cm) / (1 + dt*G/Cm)``): the reference operator split
    (channels stepped before voltage). ``state`` is ``(v, m, h, n)``:
    membrane voltage (mV, absolute) and gating variables. ``I_inj`` is
    depolarizing-positive injected current (inward). Returns ``(v, m, h, n)``.
    Pure ``jax.numpy``; dtype follows the inputs. ``dt_ms`` is validated by
    the caller (:func:`simulate_hh` and
    :func:`jaxfne._construct_hh.hh_kernel_args` refuse a non-finite or
    non-positive ``dt``); the stepper itself stays jittable.
    """
    v, m, h, n = state
    p = params
    am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
    m = _exp_euler_gate(m, dt_ms, am, bm)
    h = _exp_euler_gate(h, dt_ms, ah, bh)
    n = _exp_euler_gate(n, dt_ms, an, bn)
    g_na = p.gNa_mS_cm2 * m ** 3 * h
    g_k = p.gK_mS_cm2 * n ** 4
    g_l = p.gL_mS_cm2
    g_tot = g_na + g_k + g_l
    c_tot = g_na * p.ENa_mV + g_k * p.EK_mV + g_l * p.EL_mV + I_inj
    v = (v + c_tot / p.Cm_uF_cm2 * dt_ms) / (1.0 + g_tot / p.Cm_uF_cm2 * dt_ms)
    return (v, m, h, n)


def graded_activation(v_pre, v_th: float = HH_GRADED_VTH_DEFAULT,
                      slope: float = HH_GRADED_SLOPE_DEFAULT):
    """Tanh activation of the graded synapses: ``0.5 * (1 + tanh((Vpre-Vth)/slope))``.

    Defaults are the Fig. 6 run values (PORT_SPEC.md section 4).
    """
    return 0.5 * (1.0 + jnp.tanh((v_pre - v_th) / slope))


def graded_syn_step(s, v_pre, tau_d: float, tau_r: float,
                    slope: float = HH_GRADED_SLOPE_DEFAULT,
                    v_th: float = HH_GRADED_VTH_DEFAULT, dt: float = 0.1):
    """One graded-synapse gating step: ``ds/dt = -s/tauD + act*(1-s)/tauR``.

    ``slope``/``v_th`` default to the Fig. 6 run values (PORT_SPEC.md
    section 4). A non-finite update keeps the old ``s`` (the reference
    float32 barrier). ``dt`` is validated by the caller (:func:`simulate_hh`
    refuses a non-finite or non-positive ``dt``); the stepper itself stays
    pure ``jax.numpy`` (jittable and differentiable).
    """
    act = graded_activation(v_pre, v_th, slope)
    new_s = s + ((-s / tau_d) + act * (1.0 - s) / tau_r) * dt
    return jnp.where(jnp.isfinite(new_s), new_s, s)


def graded_syn_current(g, s, v_post, e_rev):
    """``I_syn = g * s * (Vpost - E_rev)``, outward-positive (paper Eq. 1)."""
    return g * s * (v_post - e_rev)


def ou_step(n, key, mean: float = 0.0, sigma: float = HH_OU_SIGMA_DEFAULT,
            tau: float = HH_OU_TAU_FALLBACK_MS, dt: float = 0.1):
    """One Ornstein-Uhlenbeck step (the reference ``Inoise.update_states``).

    ``n += (mean - n)/tau * dt + sigma * sqrt(2/tau) * xi * sqrt(dt)`` with
    ``xi ~ N(0, 1)`` drawn from ``key``; the noise current is ``-n``
    (outward-positive). Shape follows ``n``. ``sigma`` defaults to the Fig. 6
    run clip upper (PORT_SPEC.md section 3); ``tau`` defaults to the
    lane-local 20 ms fallback (PORT_SPEC.md gives a U(10, 50) range, not a
    single value). A non-finite update keeps the old ``n`` (the reference
    float32 barrier). ``dt`` is validated by the caller (:func:`simulate_hh`
    refuses a non-finite or non-positive ``dt``); the stepper itself stays
    pure ``jax.numpy`` (jittable). The reference noise channel
    current is in mA/cm2; callers working in uA/cm2 scale ``sigma`` by 1000.
    """
    xi = jax.random.normal(key, shape=jnp.shape(n), dtype=jnp.asarray(n).dtype)
    new_n = n + (mean - n) / tau * dt + sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt)
    return jnp.where(jnp.isfinite(new_n), new_n, n)


_EDGE_ARGS = ("edge_pre", "edge_post", "edge_weight", "edge_receptor")


def _check_edges(N, R, given):
    """Validate the four edge arrays; return numpy arrays, or None when all are absent.

    ``edge_weight`` may be a JAX tracer (``jax.jit``/``jax.grad`` through the
    kernel): only its static checks (ndim, length) apply then, since value
    checks need concrete values. Every concrete input takes the strict path
    exactly as before.
    """
    missing = [k for k in _EDGE_ARGS if given[k] is None]
    if missing and len(missing) < len(_EDGE_ARGS):
        raise ValueError(f"edge arrays must be all given or all None; missing {missing}")
    if missing:
        return None
    arrs: dict[str, Any] = {}
    abstract_weight = False
    for k in _EDGE_ARGS:
        v = given[k]
        if k == "edge_weight" and isinstance(v, jax.Array):
            try:
                arrs[k] = np.asarray(v)
            except Exception:
                # A tracer under jit/grad (a concrete JAX array always
                # converts): keep it in JAX land, check values never.
                arrs[k] = jnp.asarray(v)
                abstract_weight = True
        else:
            arrs[k] = np.asarray(v)
        if arrs[k].ndim != 1:
            raise ValueError(f"{k} must be one-dimensional, got shape {arrs[k].shape}")
    lengths = {k: a.shape[0] for k, a in arrs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"edge arrays must have equal lengths; got {lengths}")

    def _ints(name):
        a = arrs[name].astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integers")
        return a.astype(np.int64)

    pre = _ints("edge_pre")
    post = _ints("edge_post")
    rec = _ints("edge_receptor")
    if pre.size and (pre.min() < 0 or pre.max() >= N):
        raise ValueError(f"edge_pre must index cells in [0, {N}); got min {pre.min()}, max {pre.max()}")
    if post.size and (post.min() < 0 or post.max() >= N):
        raise ValueError(f"edge_post must index cells in [0, {N}); got min {post.min()}, max {post.max()}")
    if rec.size and (rec.min() < 0 or rec.max() >= R):
        raise ValueError(f"edge_receptor must index receptors in [0, {R}); got min {rec.min()}, max {rec.max()}")
    weight = arrs["edge_weight"]
    if abstract_weight:
        # Static checks only; finiteness and sign need concrete values.
        return {"pre": pre, "post": post, "weight": weight, "receptor": rec}
    weight = weight.astype(np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError(f"edge_weight must be finite; got {weight}")
    if np.any(weight < 0.0):
        raise ValueError(f"edge_weight must be >= 0 (the reversal sets the sign); got min {weight.min()}")
    return {"pre": pre, "post": post, "weight": weight, "receptor": rec}


# ---------------------------------------------------------------------------
# Optional per-cell compartment morphology (multi-compartment HH route).
#
# The default kernel path is one compartment per cell. The reference GSDR
# cell instead has 4 compartments (2 branches x ncomp=2, jaxley defaults
# r = 1 um, l = 10 um, Ra = 5000 ohm cm, HH in every compartment;
# ``artifacts/etudes/papers/gsdr2026/CALIBRATION.md`` section 6.10):
# presynaptic V and recording at branch 0 loc 0.0, synapses onto branch 1
# loc 1.0, stimulus onto branch 1 loc 0.0. The mapping to compartment
# indices is branch 0 loc 0.0 -> 0, branch 0 loc 1.0 -> 1,
# branch 1 loc 0.0 -> 2, branch 1 loc 1.0 -> 3
# (:func:`reference_4comp_morphology`).
#
# Axial coupling follows the reference discretization (jaxley 0.14.0, read
# as data): ``compute_axial_conductances`` in
# ``site-packages/jaxley/utils/morph_attributes.py`` (half-segment
# resistive loads ``(l/2)/r^2/pi``, series resistance over the two halves,
# divided by the sink compartment area and capacitance, ``1e7`` for
# ``S/cm/um`` -> ``mS/cm2``) and the implicit backward-Euler voltage update
# ``step_voltage_implicit_with_dhs_solve`` in
# ``site-packages/jaxley/solver_voltage.py`` (diagonals ``1 + dt*g/Cm``
# plus the axial terms, right-hand side ``v + dt*c/Cm``; the reference
# branchpoint node is eliminated by substitution, which for a pairwise
# junction equals the two-half series formula used here). Per-compartment
# HH gates step exactly as the single-compartment path
# (:func:`_exp_euler_gate` at frozen voltage, then the linearized
# backward-Euler voltage step). Point-process inputs convert by the target
# compartment area (``convert_point_process_to_distributed`` in
# ``site-packages/jaxley/utils/cell_utils.py``: ``nA / area_um2 * 1e5``).
# ---------------------------------------------------------------------------


def point_process_to_distributed(current_nA, area_um2):
    """Convert a point-process current (nA) to a density (uA/cm2).

    ``current_nA / area_um2 * 1e5``: the reference
    ``convert_point_process_to_distributed`` (jaxley 0.14.0,
    ``site-packages/jaxley/utils/cell_utils.py``), applied with the target
    compartment area for synapse conductances (uS -> mS/cm2, same factor)
    and stimulus currents (nA -> uA/cm2).
    """
    return np.asarray(current_nA, dtype=np.float64) / np.asarray(area_um2, dtype=np.float64) * 1e5


class HHMorphology:
    """Per-cell compartment geometry shared by every cell of a population.

    ``parents[k]`` is the index of compartment ``k``'s parent, ``-1`` for
    the root; ``length_um[k]`` / ``radius_um[k]`` are the per-compartment
    cylinder length and radius (um); ``ra_ohmcm`` is the scalar axial
    resistivity (ohm cm). Areas (um2) are ``2*pi*r*l`` per compartment.
    """

    def __init__(self, parents, length_um, radius_um, ra_ohmcm=5000.0) -> None:
        self.parents = tuple(int(p) for p in parents)
        self.length_um = tuple(float(v) for v in length_um)
        self.radius_um = tuple(float(v) for v in radius_um)
        self.ra_ohmcm = float(ra_ohmcm)
        self.area_um2 = tuple(
            2.0 * math.pi * r * le
            for r, le in zip(self.radius_um, self.length_um)
        )

    @property
    def n_comp(self) -> int:
        """Number of compartments."""
        return len(self.parents)


def reference_4comp_morphology() -> HHMorphology:
    """The GSDR reference geometry: 2 branches x 2 compartments (4 total).

    jaxley defaults (``site-packages/jaxley/modules/compartment.py``:
    ``Compartment.compartment_params``): length 10 um, radius 1 um, axial
    resistivity 5000 ohm cm, capacitance 1 uF/cm2. Compartments form the
    chain 0 - 1 - 2 - 3 (branch 0 loc 0.0 -> 0 through branch 1 loc 1.0
    -> 3), so ``parents == (-1, 0, 1, 2)``.
    """
    return HHMorphology(
        parents=(-1, 0, 1, 2),
        length_um=(10.0, 10.0, 10.0, 10.0),
        radius_um=(1.0, 1.0, 1.0, 1.0),
        ra_ohmcm=5000.0,
    )


def _check_morphology(morphology) -> dict[str, Any]:
    """Validate ``morphology``; return arrays, links and base axial conductances."""
    if not isinstance(morphology, HHMorphology):
        raise ValueError(
            f"morphology must be an HHMorphology; got {morphology!r}"
        )
    parents = np.asarray(morphology.parents)
    length = np.asarray(morphology.length_um, dtype=np.float64)
    radius = np.asarray(morphology.radius_um, dtype=np.float64)
    try:
        ra = float(morphology.ra_ohmcm)
    except (TypeError, ValueError):
        raise ValueError(f"morphology ra_ohmcm must be a number; got {morphology.ra_ohmcm!r}")
    area = np.asarray(morphology.area_um2, dtype=np.float64)
    n_comp = int(parents.shape[0])
    if not (length.shape == (n_comp,) and radius.shape == (n_comp,) and area.shape == (n_comp,)):
        raise ValueError(
            "morphology parents, length_um, radius_um and area_um2 must have "
            f"equal lengths; got {parents.shape}, {length.shape}, {radius.shape}, {area.shape}"
        )
    if n_comp < 1:
        raise ValueError(f"morphology must hold at least one compartment; got {n_comp}")
    if not (np.all(np.isfinite(length)) and np.all(length > 0.0)):
        raise ValueError(f"morphology length_um must be finite and > 0; got {length}")
    if not (np.all(np.isfinite(radius)) and np.all(radius > 0.0)):
        raise ValueError(f"morphology radius_um must be finite and > 0; got {radius}")
    if not (math.isfinite(ra) and ra > 0.0):
        raise ValueError(f"morphology ra_ohmcm must be finite and > 0; got {morphology.ra_ohmcm!r}")
    if not (np.all(np.isfinite(area)) and np.all(area > 0.0)):
        raise ValueError(f"morphology area_um2 must be finite and > 0; got {area}")
    if not np.all(np.isfinite(parents)) or np.any(parents != np.round(parents)):
        raise ValueError(f"morphology parents must hold integers; got {parents}")
    parents = parents.astype(np.int64)
    if int(np.sum(parents == -1)) != 1:
        raise ValueError(
            f"morphology parents must hold exactly one root (-1); got {parents.tolist()}"
        )
    if np.any(parents < -1) or np.any(parents >= n_comp):
        raise ValueError(
            f"morphology parents must index compartments in [0, {n_comp}) or be -1; "
            f"got {parents.tolist()}"
        )
    if np.any(parents == np.arange(n_comp)):
        raise ValueError(
            f"morphology parents must not self-parent; got {parents.tolist()}"
        )
    for k in range(n_comp):
        seen = 0
        node = k
        while parents[node] != -1:
            node = int(parents[node])
            seen += 1
            if seen > n_comp:
                raise ValueError(
                    f"morphology parents must be acyclic; got {parents.tolist()}"
                )
    link_child = np.asarray(
        [k for k in range(n_comp) if parents[k] != -1], dtype=np.int64
    )
    link_parent = parents[link_child]
    # Base axial conductances (mS/cm2 at Cm = 1) per link and direction,
    # sink side first: A[link, 0] sinks in link_child, A[link, 1] in
    # link_parent (the reference ``compute_axial_conductances`` c2c form).
    load = (length / 2.0) / radius ** 2 / math.pi
    rsum = (load[link_child] + load[link_parent]) * ra
    axial = np.zeros((link_child.shape[0], 2), dtype=np.float64)
    axial[:, 0] = 1.0 / rsum / area[link_child] * 1e7
    axial[:, 1] = 1.0 / rsum / area[link_parent] * 1e7
    if not np.all(np.isfinite(axial)) or np.any(axial <= 0.0):
        raise ValueError(f"morphology yields non-finite axial conductances; got {axial}")
    return {
        "n_comp": n_comp,
        "parents": parents,
        "area": area,
        "link_child": link_child,
        "link_parent": link_parent,
        "axial_base": axial,
    }


def _comp_index_array(values, n, lo_name: str) -> np.ndarray:
    """Broadcast-or-check an integer compartment selection of length ``n``."""
    if values is None:
        return np.zeros((n,), dtype=np.int64)
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 0:
        a = np.full((n,), float(a))
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(
            f"{lo_name} must be a scalar or shape ({n},); got shape {a.shape}"
        )
    if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
        raise ValueError(f"{lo_name} must hold integers; got {a}")
    return a.astype(np.int64)


def _check_comp_range(comp, n_comp: int, lo_name: str) -> None:
    if comp.size and (comp.min() < 0 or comp.max() >= n_comp):
        raise ValueError(
            f"{lo_name} must index compartments in [0, {n_comp}); "
            f"got min {comp.min()}, max {comp.max()}"
        )


def _tolerant_asarray(value, dtype):
    """``np.asarray`` that passes tracers through as JAX arrays.

    Batch-only: called only with ``_prevalidated=True``, after
    :func:`simulate_hh_batch` validated every batch element's values on the
    host. Shapes stay static under tracing.
    """
    try:
        return np.asarray(value, dtype=dtype)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return jnp.asarray(value, dtype=dtype)


def simulate_hh(
    params: HHParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight=None,
    edge_receptor=None,
    receptor_e_rev=(),
    receptor_tau_r=(),
    receptor_tau_d=(),
    receptor_slope=(),
    receptor_vth=(),
    receptor_s0: float = HH_GRADED_S0_DEFAULT,
    ou_tau=None,
    ou_sigma=None,
    ou_key=None,
    noise_gain: float = 1.0,
    I_ext=None,
    I_ext_nA=None,
    V0_mV=None,
    gate_init=None,
    record_v: bool = True,
    morphology=None,
    edge_pre_comp=None,
    edge_post_comp=None,
    edge_weight_uS=None,
    stim_comp=None,
    record_comp=None,
    _prevalidated: bool = False,
) -> dict[str, Any]:
    """Simulate HH cells with OU noise and graded tanh synapses on an edge list.

    Cells: classic Hodgkin-Huxley (:func:`hh_step`). Synapses: the reference
    graded form (PORT_SPEC.md section 4): each edge ``e`` runs from
    ``edge_pre[e]`` to ``edge_post[e]`` through receptor ``edge_receptor[e]``
    with conductance ``edge_weight[e]`` (mS/cm2, >= 0); the gating variable of
    edge ``e`` evolves as ``ds/dt = -s/tauD + act(Vpre)*(1-s)/tauR`` with
    ``act`` the tanh activation, and contributes
    ``g * s * (Vpost - E_rev)`` (outward-positive) to its post cell. The
    presynaptic voltage is read at the current step: the reference has no
    delay ring, so this kernel takes no ``edge_delay_steps`` argument.

    Noise: per-cell OU channel (:func:`ou_step`) with time constants
    ``ou_tau`` ([N], ms), amplitudes ``ou_sigma`` ([N]) and key ``ou_key``;
    the noise current is ``-ou``. Noise is live only when a key is given and
    the effective sigma (``ou_sigma`` times ``noise_gain``) is nonzero; only
    a live run requires ``ou_tau``. ``noise_gain`` is a nonnegative scalar
    multiplier on ``ou_sigma`` (lane-4e residual X: the noise level; the
    default 1.0 reproduces the unscaled channel bit-exactly).

    ``I_ext`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``:
    depolarizing-positive injected current. ``V0_mV`` is None (rest at -70 mV)
    or shape ``[N]``. ``gate_init`` is None (single-compartment gates start at
    their rate fixed point) or an explicit gating value for every m/h/n gate.

    Multi-compartment form: pass ``morphology`` (an :class:`HHMorphology`;
    every cell shares it) with per-edge ``edge_pre_comp``/``edge_post_comp``
    (presynaptic-V and synapse compartments, default 0), per-cell
    ``stim_comp``/``record_comp`` (stimulus and spike-readout compartments,
    default 0), and point-process inputs ``edge_weight_uS`` (uS, converted
    by the post compartment area) and ``I_ext_nA`` (nA, converted by the
    stimulus compartment area) instead of the density forms ``edge_weight``
    (mS/cm2) and ``I_ext`` (uA/cm2). Mixing the two forms of either input
    is refused, as is any out-of-range compartment. ``V_mV`` is then
    ``[n_steps, N, C]`` and the final states are ``[N, C]``; spikes are
    upward crossings of each cell's ``record_comp``. Gates init at the
    reference default state 0.2 (``channels/hh.py``; the reference never
    calls ``init_state``), unless ``gate_init`` gives an explicit value.
    Without ``morphology`` (and no other new argument) the call runs the
    single-compartment path bit-identically to before.

    Per step, in order: synapse gates update from the current voltages
    (a non-finite update keeps the old gating: the reference float32
    barrier); synaptic conductance and offset terms accumulate per post
    cell; the OU channel updates (same hold barrier); gates advance by
    exact exponential Euler and voltage by backward Euler at frozen gates
    (the :func:`hh_step` split); upward crossings of -20 mV are spikes.
    Held (non-finite) updates are counted in the scan carry and reported as
    ``held_steps`` (``{"s": n, "ou": n}``): ``s`` is one per edge per held
    step, ``ou`` one per OU element per held step, so each count is a count
    of held element updates summed over the run, not of steps. The GSDR fit
    maps a NaN loss to inf and reverts the generation (PORT_SPEC.md section 10).

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV``
    (``[n_steps, N]``, after each step's update, only if ``record_v``), the
    final ``v``, ``m``, ``h``, ``n``, ``ou`` and ``s``, and ``held_steps``.
    Implemented with ``jax.lax.scan``; no dense ``N x N`` matrix. dtype
    follows ``params``.

    Gradients: ``edge_weight`` is read as a JAX array and carries a gradient
    through the full scan (``tests/test_hh_full_kernel_grad.py``). The
    receptor arrays and ``receptor_s0`` are read as host numbers for
    validation, as is ``noise_gain`` here (:func:`simulate_hh_batch` takes it
    as a traced value), so none of them carries a gradient.

    Raises ``ValueError`` naming the argument for every refused input
    (``dt_ms`` must be finite and > 0; ``ou_tau`` is required when noise is
    live and must be finite and > 0; ``ou_sigma`` must be finite).
    """
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    _require_finite_dt(dt_ms, "dt_ms")

    if (
        morphology is not None
        or edge_pre_comp is not None
        or edge_post_comp is not None
        or edge_weight_uS is not None
        or stim_comp is not None
        or record_comp is not None
        or I_ext_nA is not None
    ):
        return _simulate_hh_multi(
            params, n_steps=n_steps, dt_ms=dt_ms,
            edge_pre=edge_pre, edge_post=edge_post, edge_weight=edge_weight,
            edge_receptor=edge_receptor, receptor_e_rev=receptor_e_rev,
            receptor_tau_r=receptor_tau_r, receptor_tau_d=receptor_tau_d,
            receptor_slope=receptor_slope, receptor_vth=receptor_vth,
            receptor_s0=receptor_s0, ou_tau=ou_tau, ou_sigma=ou_sigma,
            ou_key=ou_key, noise_gain=noise_gain, I_ext=I_ext,
            I_ext_nA=I_ext_nA, V0_mV=V0_mV, gate_init=gate_init,
            record_v=record_v, morphology=morphology,
            edge_pre_comp=edge_pre_comp, edge_post_comp=edge_post_comp,
            edge_weight_uS=edge_weight_uS, stim_comp=stim_comp,
            record_comp=record_comp, _prevalidated=_prevalidated,
        )

    N = int(jnp.shape(params.gNa_mS_cm2)[0])
    dtype = jnp.asarray(params.gNa_mS_cm2).dtype
    e_rev = np.asarray(receptor_e_rev, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(receptor_tau_r, dtype=np.float64).reshape(-1)
    tau_d = np.asarray(receptor_tau_d, dtype=np.float64).reshape(-1)
    slope = np.asarray(receptor_slope, dtype=np.float64).reshape(-1)
    vth = np.asarray(receptor_vth, dtype=np.float64).reshape(-1)
    if not (e_rev.shape == tau_r.shape == tau_d.shape == slope.shape == vth.shape):
        raise ValueError(
            "receptor_e_rev, receptor_tau_r, receptor_tau_d, receptor_slope and "
            f"receptor_vth must have the same length; got {[a.size for a in (e_rev, tau_r, tau_d, slope, vth)]}"
        )
    if np.any(tau_r <= 0):
        raise ValueError(f"receptor_tau_r must be > 0 for every receptor; got {tau_r.tolist()}")
    if np.any(tau_d <= 0):
        raise ValueError(f"receptor_tau_d must be > 0 for every receptor; got {tau_d.tolist()}")
    if np.any(slope <= 0):
        raise ValueError(f"receptor_slope must be > 0 for every receptor; got {slope.tolist()}")
    try:
        s0 = float(receptor_s0)
    except (TypeError, ValueError):
        raise ValueError(f"receptor_s0 must be a number; got {receptor_s0!r}")
    if not math.isfinite(s0):
        raise ValueError(f"receptor_s0 must be finite; got {receptor_s0!r}")
    R = int(e_rev.size)

    edges = _check_edges(
        N, R,
        {"edge_pre": edge_pre, "edge_post": edge_post,
         "edge_weight": edge_weight, "edge_receptor": edge_receptor},
    )
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "receptor": np.zeros((0,), np.int64),
        }
    n_edges = int(edges["pre"].shape[0])

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    post_j = jnp.asarray(edges["post"], jnp.int32)
    w_j = jnp.asarray(edges["weight"], dtype)
    rec_j = jnp.asarray(edges["receptor"], jnp.int32)
    erev_j = jnp.asarray(e_rev, dtype)
    taur_j = jnp.asarray(tau_r, dtype)
    taud_j = jnp.asarray(tau_d, dtype)
    slope_j = jnp.asarray(slope, dtype)
    vth_j = jnp.asarray(vth, dtype)

    if ou_sigma is None:
        sig_j = jnp.zeros((N,), dtype)
    else:
        sig_j = jnp.asarray(ou_sigma, dtype)
        if sig_j.shape != (N,):
            raise ValueError(f"ou_sigma must be None or shape ({N},); got {sig_j.shape}")
        # Shared input: with _prevalidated its finiteness was checked on the
        # host (element 0 runs the full serial validation), and the
        # jnp.asarray above is a tracer under the batch jit.
        if not _prevalidated and not bool(jnp.all(jnp.isfinite(sig_j))):
            raise ValueError(f"ou_sigma must be finite; got {ou_sigma!r}")
    if _prevalidated:
        # Batch path: this element's gain was validated on the host and the
        # slice may be a tracer, so take it as-is without value judgments.
        # ``eff_nonzero`` stays None (falsy below): the per-element liveness
        # raises already ran in :func:`simulate_hh_batch`.
        gain_arr = jnp.asarray(noise_gain)
        sig_j = sig_j * gain_arr
        eff_nonzero = None
    else:
        try:
            gain = float(noise_gain)
        except (TypeError, ValueError):
            raise ValueError(f"noise_gain must be a number; got {noise_gain!r}")
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError(f"noise_gain must be finite and >= 0; got {noise_gain!r}")
        sig_j = sig_j * jnp.asarray(gain, dtype)
        # Noise is live only when a key is given and the effective sigma
        # (ou_sigma times noise_gain) is nonzero; only a live run needs ou_tau.
        eff_nonzero = bool(jnp.any(sig_j != 0.0))
    if ou_tau is None:
        if ou_key is not None and eff_nonzero:
            raise ValueError(
                "ou_tau is required when OU noise is live (ou_key given and "
                "ou_sigma * noise_gain nonzero); got None "
                "(PORT_SPEC.md section 3 gives a per-cell U(10, 50) range, not a default)"
            )
        tau_j = jnp.full((N,), HH_OU_TAU_FALLBACK_MS, dtype=dtype)
    else:
        tau_j = jnp.asarray(ou_tau, dtype)
        if tau_j.shape != (N,):
            raise ValueError(f"ou_tau must be None or shape ({N},); got {tau_j.shape}")
        # Shared input, same batch rationale as ou_sigma above.
        if not _prevalidated and (not bool(jnp.all(jnp.isfinite(tau_j))) or bool(jnp.any(tau_j <= 0.0))):
            raise ValueError(f"ou_tau must be finite and > 0; got {ou_tau!r}")
    if eff_nonzero and ou_key is None:
        raise ValueError("ou_key is required when ou_sigma * noise_gain is nonzero")
    noise_on = ou_key is not None
    if ou_key is not None:
        try:
            jax.random.split(ou_key)
        except (TypeError, ValueError):
            # Probed on this tree's JAX: split raises TypeError for every
            # malformed key tried (str, None, int, float, list, wrong dtype
            # or shape); ValueError is caught for other JAX versions.
            raise ValueError(f"ou_key must be a JAX PRNG key; got {ou_key!r}")

    I_const = jnp.zeros((N,), dtype)
    I_all = None
    if I_ext is not None:
        I_arr = jnp.asarray(I_ext, dtype)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )

    if V0_mV is None:
        V0 = jnp.full((N,), HH_V_REST_MV, dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape != (N,):
            raise ValueError(f"V0_mV must be None or shape ({N},); got {V0.shape}")

    thr = jnp.asarray(HH_SPIKE_THR_MV, dtype)

    def body(carry, I_t):
        v, m, h, n, ou, s, prev_v, rkey, n_held_s, n_held_ou = carry
        if noise_on:
            rkey, uk = jax.random.split(rkey)
        I_t = I_const if I_t is None else I_t
        g_syn = jnp.zeros((N,), dtype)
        c_syn = jnp.zeros((N,), dtype)
        if n_edges:
            v_pre = v[pre_j]
            act = 0.5 * (1.0 + jnp.tanh((v_pre - vth_j[rec_j]) / slope_j[rec_j]))
            new_s = s + ((-s / taud_j[rec_j]) + act * (1.0 - s) / taur_j[rec_j]) * dt_ms
            # The reference float32 barrier: a non-finite update keeps the
            # old gating; held steps are counted in the scan carry and
            # reported as held_steps (the GSDR fit maps a NaN loss to inf
            # and reverts the generation: PORT_SPEC.md section 10).
            n_held_s = n_held_s + jnp.sum((~jnp.isfinite(new_s)).astype(jnp.int32)).astype(jnp.int32)
            s = jnp.where(jnp.isfinite(new_s), new_s, s)
            gs = w_j * s
            g_syn = jax.ops.segment_sum(gs, post_j, num_segments=N)
            c_syn = jax.ops.segment_sum(gs * erev_j[rec_j], post_j, num_segments=N)
        if noise_on:
            xi = jax.random.normal(uk, shape=(N,), dtype=dtype)
            new_ou = ou + (0.0 - ou) / tau_j * dt_ms + sig_j * jnp.sqrt(2.0 / tau_j) * xi * jnp.sqrt(dt_ms)
            # As above: hold the old noise, count the held update.
            n_held_ou = n_held_ou + jnp.sum((~jnp.isfinite(new_ou)).astype(jnp.int32)).astype(jnp.int32)
            ou = jnp.where(jnp.isfinite(new_ou), new_ou, ou)
        i_noise = -ou
        am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
        m = _exp_euler_gate(m, dt_ms, am, bm)
        h = _exp_euler_gate(h, dt_ms, ah, bh)
        n = _exp_euler_gate(n, dt_ms, an, bn)
        g_na = jnp.asarray(params.gNa_mS_cm2, dtype) * m ** 3 * h
        g_k = jnp.asarray(params.gK_mS_cm2, dtype) * n ** 4
        g_l = jnp.asarray(params.gL_mS_cm2, dtype)
        g_tot = g_na + g_k + g_l + g_syn
        c_tot = (g_na * jnp.asarray(params.ENa_mV, dtype)
                 + g_k * jnp.asarray(params.EK_mV, dtype)
                 + g_l * jnp.asarray(params.EL_mV, dtype)
                 + c_syn - i_noise + I_t)
        cm = jnp.asarray(params.Cm_uF_cm2, dtype)
        v_new = (v + c_tot / cm * dt_ms) / (1.0 + g_tot / cm * dt_ms)
        spk = (prev_v < thr) & (v_new >= thr)
        out = (spk, v_new) if record_v else spk
        return (v_new, m, h, n, ou, s, v_new, rkey, n_held_s, n_held_ou), out

    if ou_key is None:
        rkey0 = jnp.zeros((), jnp.int32)
    else:
        rkey0 = ou_key
    g_init = _check_gate_init(gate_init)
    if g_init is None:
        m0, h0, n0 = hh_gate_inf(V0)
        m0_j, h0_j, n0_j = (
            jnp.asarray(m0, dtype), jnp.asarray(h0, dtype), jnp.asarray(n0, dtype))
    else:
        m0_j = h0_j = n0_j = jnp.full((N,), g_init, dtype)
    carry0 = (
        V0,
        m0_j,
        h0_j,
        n0_j,
        jnp.zeros((N,), dtype),
        jnp.full((n_edges,), s0, dtype),
        V0,
        rkey0,
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
    )
    (v, m, h, n, ou, s, _, _, n_held_s, n_held_ou), ys = jax.lax.scan(body, carry0, I_all, length=n_steps)
    result = {
        "spikes": ys[0] if record_v else ys,
        "v": v,
        "m": m,
        "h": h,
        "n": n,
        "ou": ou,
        "s": s,
        "held_steps": {"s": n_held_s, "ou": n_held_ou},
    }
    if record_v:
        result["V_mV"] = ys[1]
    return result


def _simulate_hh_multi(
    params: HHParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight=None,
    edge_receptor=None,
    receptor_e_rev=(),
    receptor_tau_r=(),
    receptor_tau_d=(),
    receptor_slope=(),
    receptor_vth=(),
    receptor_s0: float = HH_GRADED_S0_DEFAULT,
    ou_tau=None,
    ou_sigma=None,
    ou_key=None,
    noise_gain: float = 1.0,
    I_ext=None,
    I_ext_nA=None,
    V0_mV=None,
    gate_init=None,
    record_v: bool = True,
    morphology=None,
    edge_pre_comp=None,
    edge_post_comp=None,
    edge_weight_uS=None,
    stim_comp=None,
    record_comp=None,
    _prevalidated: bool = False,
) -> dict[str, Any]:
    """Multi-compartment ``simulate_hh`` (see its docstring for the contract).

    Same split as the single-compartment path, with per-compartment HH
    gates and the reference implicit voltage solve over the axial tree.
    """
    if morphology is None:
        raise ValueError(
            "morphology is required when any compartment selection or "
            "point-process input is given"
        )
    mx = _check_morphology(morphology)
    C = mx["n_comp"]
    N = int(jnp.shape(params.gNa_mS_cm2)[0])
    dtype = jnp.asarray(params.gNa_mS_cm2).dtype

    e_rev = np.asarray(receptor_e_rev, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(receptor_tau_r, dtype=np.float64).reshape(-1)
    tau_d = np.asarray(receptor_tau_d, dtype=np.float64).reshape(-1)
    slope = np.asarray(receptor_slope, dtype=np.float64).reshape(-1)
    vth = np.asarray(receptor_vth, dtype=np.float64).reshape(-1)
    if not (e_rev.shape == tau_r.shape == tau_d.shape == slope.shape == vth.shape):
        raise ValueError(
            "receptor_e_rev, receptor_tau_r, receptor_tau_d, receptor_slope and "
            f"receptor_vth must have the same length; got {[a.size for a in (e_rev, tau_r, tau_d, slope, vth)]}"
        )
    if np.any(tau_r <= 0):
        raise ValueError(f"receptor_tau_r must be > 0 for every receptor; got {tau_r.tolist()}")
    if np.any(tau_d <= 0):
        raise ValueError(f"receptor_tau_d must be > 0 for every receptor; got {tau_d.tolist()}")
    if np.any(slope <= 0):
        raise ValueError(f"receptor_slope must be > 0 for every receptor; got {slope.tolist()}")
    try:
        s0 = float(receptor_s0)
    except (TypeError, ValueError):
        raise ValueError(f"receptor_s0 must be a number; got {receptor_s0!r}")
    if not math.isfinite(s0):
        raise ValueError(f"receptor_s0 must be finite; got {receptor_s0!r}")
    R = int(e_rev.size)

    if edge_weight_uS is not None and edge_weight is not None:
        raise ValueError(
            "edge_weight_uS and edge_weight are alternative input forms; got both"
        )
    use_uS = edge_weight_uS is not None
    if use_uS:
        w_us_host = np.asarray(edge_weight_uS, dtype=np.float64) \
            if not isinstance(edge_weight_uS, jax.Array) else None
        if w_us_host is not None and int(w_us_host.shape[0]) == 0:
            # An empty uS list behaves like an empty density list: no edges,
            # even when the cell/post/receptor lists are absent.
            return _simulate_hh_multi(
                params, n_steps=n_steps, dt_ms=dt_ms,
                edge_pre=None, edge_post=None, edge_weight=None,
                edge_receptor=None, receptor_e_rev=receptor_e_rev,
                receptor_tau_r=receptor_tau_r, receptor_tau_d=receptor_tau_d,
                receptor_slope=receptor_slope, receptor_vth=receptor_vth,
                receptor_s0=receptor_s0, ou_tau=ou_tau, ou_sigma=ou_sigma,
                ou_key=ou_key, noise_gain=noise_gain, I_ext=I_ext,
                I_ext_nA=I_ext_nA, V0_mV=V0_mV, gate_init=gate_init,
                record_v=record_v, morphology=morphology,
                edge_pre_comp=None, edge_post_comp=None,
                edge_weight_uS=None, stim_comp=stim_comp,
                record_comp=record_comp, _prevalidated=_prevalidated,
            )
        for k in ("edge_pre", "edge_post", "edge_receptor"):
            given = {"edge_pre": edge_pre, "edge_post": edge_post,
                     "edge_receptor": edge_receptor}[k]
            if given is None:
                raise ValueError(
                    f"edge arrays must be all given or all None; missing {[k]}"
                )
        w_us = np.asarray(edge_weight_uS, dtype=np.float64)
        if w_us.ndim != 1:
            raise ValueError(f"edge_weight_uS must be one-dimensional, got shape {w_us.shape}")
        post_c = _comp_index_array(edge_post_comp, int(w_us.shape[0]), "edge_post_comp")
        _check_comp_range(post_c, C, "edge_post_comp")
        pre_c = _comp_index_array(edge_pre_comp, int(w_us.shape[0]), "edge_pre_comp")
        _check_comp_range(pre_c, C, "edge_pre_comp")
        if not np.all(np.isfinite(w_us)):
            raise ValueError(f"edge_weight_uS must be finite; got {w_us}")
        if np.any(w_us < 0.0):
            raise ValueError(
                f"edge_weight_uS must be >= 0 (the reversal sets the sign); got min {w_us.min()}"
            )
        dens = point_process_to_distributed(w_us, mx["area"][post_c])
        edges = _check_edges(
            N, R,
            {"edge_pre": edge_pre, "edge_post": edge_post,
             "edge_weight": dens, "edge_receptor": edge_receptor},
        )
    else:
        edges = _check_edges(
            N, R,
            {"edge_pre": edge_pre, "edge_post": edge_post,
             "edge_weight": edge_weight, "edge_receptor": edge_receptor},
        )
        n_e = 0 if edges is None else int(edges["pre"].shape[0])
        pre_c = _comp_index_array(edge_pre_comp, n_e, "edge_pre_comp")
        _check_comp_range(pre_c, C, "edge_pre_comp")
        post_c = _comp_index_array(edge_post_comp, n_e, "edge_post_comp")
        _check_comp_range(post_c, C, "edge_post_comp")
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "receptor": np.zeros((0,), np.int64),
        }
        pre_c = np.zeros((0,), np.int64)
        post_c = np.zeros((0,), np.int64)
    n_edges = int(edges["pre"].shape[0])

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    post_j = jnp.asarray(edges["post"], jnp.int32)
    w_j = jnp.asarray(edges["weight"], dtype)
    rec_j = jnp.asarray(edges["receptor"], jnp.int32)
    erev_j = jnp.asarray(e_rev, dtype)
    taur_j = jnp.asarray(tau_r, dtype)
    taud_j = jnp.asarray(tau_d, dtype)
    slope_j = jnp.asarray(slope, dtype)
    vth_j = jnp.asarray(vth, dtype)
    prec_j = jnp.asarray(pre_c, jnp.int32)
    postc_j = jnp.asarray(post_c, jnp.int32)

    stim_c = _comp_index_array(stim_comp, N, "stim_comp")
    _check_comp_range(stim_c, C, "stim_comp")
    rec_c = _comp_index_array(record_comp, N, "record_comp")
    _check_comp_range(rec_c, C, "record_comp")
    rec_cj = jnp.asarray(rec_c, jnp.int32)

    if ou_sigma is None:
        sig_j = jnp.zeros((N,), dtype)
        eff_nonzero = False
    else:
        sig_host = np.asarray(ou_sigma, dtype=np.float64)
        if sig_host.shape != (N,):
            raise ValueError(f"ou_sigma must be None or shape ({N},); got {sig_host.shape}")
        if not bool(np.all(np.isfinite(sig_host))):
            raise ValueError(f"ou_sigma must be finite; got {ou_sigma!r}")
        sig_j = jnp.asarray(ou_sigma, dtype)
    if _prevalidated:
        # Batch path: this element's gain was validated on the host and the
        # slice may be a tracer, so take it as-is without value judgments.
        # ``eff_nonzero`` stays None (falsy below): the per-element liveness
        # raises already ran in :func:`simulate_hh_batch`.
        gain_arr = jnp.asarray(noise_gain)
        sig_j = sig_j * gain_arr
        eff_nonzero = None
    else:
        try:
            gain = float(noise_gain)
        except (TypeError, ValueError):
            raise ValueError(f"noise_gain must be a number; got {noise_gain!r}")
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError(f"noise_gain must be finite and >= 0; got {noise_gain!r}")
        sig_j = sig_j * jnp.asarray(gain, dtype)
        # Host-side nonzero check on the params-dtype values the scan consumes:
        # identical to the old ``bool(jnp.any(...))`` eagerly, and keeps Python
        # control flow off traced values (jittable).
        if ou_sigma is None:
            eff_nonzero = False
        else:
            eff_nonzero = bool(np.any(sig_host * float(np.asarray(gain, dtype=np.float64)) != 0.0))
    if ou_tau is None:
        if ou_key is not None and eff_nonzero:
            raise ValueError(
                "ou_tau is required when OU noise is live (ou_key given and "
                "ou_sigma * noise_gain nonzero); got None"
            )
        tau_j = jnp.full((N,), HH_OU_TAU_FALLBACK_MS, dtype=dtype)
    else:
        tau_host = np.asarray(ou_tau, dtype=np.float64)
        if tau_host.shape != (N,):
            raise ValueError(f"ou_tau must be None or shape ({N},); got {tau_host.shape}")
        if not bool(np.all(np.isfinite(tau_host))) or bool(np.any(tau_host <= 0.0)):
            raise ValueError(f"ou_tau must be finite and > 0; got {ou_tau!r}")
        tau_j = jnp.asarray(ou_tau, dtype)
    if eff_nonzero and ou_key is None:
        raise ValueError("ou_key is required when ou_sigma * noise_gain is nonzero")
    noise_on = ou_key is not None
    if ou_key is not None:
        try:
            jax.random.split(ou_key)
        except (TypeError, ValueError):
            raise ValueError(f"ou_key must be a JAX PRNG key; got {ou_key!r}")

    if I_ext is not None and I_ext_nA is not None:
        raise ValueError("I_ext and I_ext_nA are alternative input forms; got both")
    I_all = None
    I_const = None
    # Under the batch vmap, I_ext slices are tracers that np.asarray cannot
    # read; with _prevalidated their values were checked per element on the
    # host, so read them through JAX instead. The serial path keeps np.asarray.
    host_array = _tolerant_asarray if _prevalidated else np.asarray
    if I_ext_nA is not None:
        nA = np.asarray(I_ext_nA, dtype=np.float64)
        if nA.shape == (N,):
            I_all = None
            I_const = point_process_to_distributed(nA, mx["area"][stim_c])
        elif nA.shape == (n_steps, N):
            I_all = point_process_to_distributed(
                nA, np.broadcast_to(mx["area"][stim_c], (n_steps, N))
            )
        else:
            raise ValueError(
                f"I_ext_nA must be None, shape ({N},) or ({n_steps}, {N}); got {nA.shape}"
            )
    elif I_ext is not None:
        I_arr = host_array(I_ext, dtype=np.float64)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )
    onehot = np.zeros((N, C), dtype=np.float64)
    onehot[np.arange(N), stim_c] = 1.0
    onehot_j = jnp.asarray(onehot, dtype)
    if I_const is None:
        I_const_j = jnp.zeros((N, C), dtype)
    else:
        I_const_j = jnp.asarray(host_array(I_const, dtype=np.float64)[:, None] * onehot, dtype)
    if I_all is not None:
        I_all_j = jnp.asarray(host_array(I_all, dtype=np.float64)[:, :, None] * onehot_j[None], dtype)
    else:
        I_all_j = None

    if V0_mV is None:
        V0 = jnp.full((N, C), HH_V_REST_MV, dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape == (N,):
            V0 = jnp.broadcast_to(V0[:, None], (N, C))
        if V0.shape != (N, C):
            raise ValueError(f"V0_mV must be None, shape ({N},) or ({N}, {C}); got {V0.shape}")

    thr = jnp.asarray(HH_SPIKE_THR_MV, dtype)
    cm_j = jnp.asarray(params.Cm_uF_cm2, dtype)
    gna_j = jnp.asarray(params.gNa_mS_cm2, dtype)
    gk_j = jnp.asarray(params.gK_mS_cm2, dtype)
    gl_j = jnp.asarray(params.gL_mS_cm2, dtype)
    ena_j = jnp.asarray(params.ENa_mV, dtype)
    ek_j = jnp.asarray(params.EK_mV, dtype)
    el_j = jnp.asarray(params.EL_mV, dtype)

    # Axial conductances per cell (mS/cm2, already divided by Cm): the
    # reference ``compute_axial_conductances`` values divided by each
    # cell's capacitance.
    la = jnp.asarray(mx["link_child"], jnp.int32)
    lb = jnp.asarray(mx["link_parent"], jnp.int32)
    ax0 = jnp.asarray(mx["axial_base"][:, 0], dtype) / cm_j[:, None]
    ax1 = jnp.asarray(mx["axial_base"][:, 1], dtype) / cm_j[:, None]
    eye_c = jnp.eye(C, dtype=dtype)

    def body(carry, I_t):
        v, m, h, n, ou, s, prev_v, rkey, n_held_s, n_held_ou = carry
        if noise_on:
            rkey, uk = jax.random.split(rkey)
        I_t = I_const_j if I_t is None else I_t
        g_syn = jnp.zeros((N, C), dtype)
        c_syn = jnp.zeros((N, C), dtype)
        if n_edges:
            v_pre = v[pre_j, prec_j]
            act = 0.5 * (1.0 + jnp.tanh((v_pre - vth_j[rec_j]) / slope_j[rec_j]))
            new_s = s + ((-s / taud_j[rec_j]) + act * (1.0 - s) / taur_j[rec_j]) * dt_ms
            n_held_s = n_held_s + jnp.sum((~jnp.isfinite(new_s)).astype(jnp.int32)).astype(jnp.int32)
            s = jnp.where(jnp.isfinite(new_s), new_s, s)
            gs = w_j * s
            flat = (post_j * C + postc_j).astype(jnp.int32)
            g_syn = jax.ops.segment_sum(gs, flat, num_segments=N * C).reshape(N, C)
            c_syn = jax.ops.segment_sum(gs * erev_j[rec_j], flat, num_segments=N * C).reshape(N, C)
        if noise_on:
            xi = jax.random.normal(uk, shape=(N, C), dtype=dtype)
            tau_c = tau_j[:, None]
            sig_c = sig_j[:, None]
            new_ou = ou + (0.0 - ou) / tau_c * dt_ms + sig_c * jnp.sqrt(2.0 / tau_c) * xi * jnp.sqrt(dt_ms)
            n_held_ou = n_held_ou + jnp.sum((~jnp.isfinite(new_ou)).astype(jnp.int32)).astype(jnp.int32)
            ou = jnp.where(jnp.isfinite(new_ou), new_ou, ou)
        i_noise = -ou
        am, bm, ah, bh, an, bn = _hh_rate_coeffs(v)
        m = _exp_euler_gate(m, dt_ms, am, bm)
        h = _exp_euler_gate(h, dt_ms, ah, bh)
        n = _exp_euler_gate(n, dt_ms, an, bn)
        g_na = gna_j[:, None] * m ** 3 * h
        g_k = gk_j[:, None] * n ** 4
        g_l = gl_j[:, None] * jnp.ones((N, C), dtype)
        g_tot = g_na + g_k + g_l + g_syn
        cm_c = cm_j[:, None]
        c_tot = (g_na * ena_j[:, None] + g_k * ek_j[:, None]
                 + g_l * el_j[:, None] + c_syn - i_noise + I_t)
        # Implicit backward-Euler voltage solve over the axial tree (the
        # reference ``step_voltage_implicit_with_dhs_solve`` system).
        diag = 1.0 + dt_ms * g_tot / cm_c
        if int(la.shape[0]):
            diag = diag.at[:, la].add(dt_ms * ax0)
            diag = diag.at[:, lb].add(dt_ms * ax1)
        mat = eye_c[None, :, :] * diag[:, :, None]
        if int(la.shape[0]):
            mat = mat.at[:, la, lb].add(-dt_ms * ax0)
            mat = mat.at[:, lb, la].add(-dt_ms * ax1)
        rhs = v + dt_ms * c_tot / cm_c
        v_new = jnp.linalg.solve(mat, rhs[..., None]).squeeze(-1)
        rec_v = v_new[jnp.arange(N), rec_cj]
        rec_prev = prev_v[jnp.arange(N), rec_cj]
        spk = (rec_prev < thr) & (rec_v >= thr)
        out = (spk, v_new) if record_v else spk
        return (v_new, m, h, n, ou, s, v_new, rkey, n_held_s, n_held_ou), out

    if ou_key is None:
        rkey0 = jnp.zeros((), jnp.int32)
    else:
        rkey0 = ou_key
    # Gate init is the reference default state 0.2 (jaxley 0.14.0,
    # ``channels/hh.py`` default states; ``init_state`` is not called on the
    # reference integrate path), not the rate fixed point the
    # single-compartment path uses, unless ``gate_init`` is given explicitly.
    g_init = _check_gate_init(gate_init)
    gate0 = jnp.full((N, C), HH_GATE_DEFAULT if g_init is None else g_init, dtype)
    carry0 = (
        V0,
        gate0,
        gate0,
        gate0,
        jnp.zeros((N, C), dtype),
        jnp.full((n_edges,), s0, dtype),
        V0,
        rkey0,
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
    )
    (v, m, h, n, ou, s, _, _, n_held_s, n_held_ou), ys = jax.lax.scan(body, carry0, I_all_j, length=n_steps)
    result = {
        "spikes": ys[0] if record_v else ys,
        "v": v,
        "m": m,
        "h": h,
        "n": n,
        "ou": ou,
        "s": s,
        "held_steps": {"s": n_held_s, "ou": n_held_ou},
    }
    if record_v:
        result["V_mV"] = ys[1]
    return result


# Keys that may vary across the leading batch axis of simulate_hh_batch.
# Everything else (topology, receptors, morphology, compartment selections,
# OU scales) is shared: it is validated once on the host and closed over,
# hence static under jit.
_HH_BATCH_KEYS = (
    "edge_weight",
    "edge_weight_uS",
    "ou_key",
    "I_ext",
    "I_ext_nA",
    "V0_mV",
    "noise_gain",
)

_MULTI_TRIGGERS = (
    "morphology",
    "edge_pre_comp",
    "edge_post_comp",
    "edge_weight_uS",
    "stim_comp",
    "record_comp",
    "I_ext_nA",
)


def simulate_hh_batch(
    params: HHParams,
    *,
    n_steps: int,
    dt_ms: float,
    batch=None,
    **shared: Any,
) -> dict[str, Any]:
    """Simulate a batch of HH runs that share topology but vary per-trial inputs.

    Every element runs :func:`simulate_hh` with the same ``params``,
    ``n_steps``, ``dt_ms`` and shared arguments. ``batch`` maps a subset of
    input names to values stacked along a leading axis B: ``edge_weight``
    ``(B, E)``, ``edge_weight_uS`` ``(B, E)``, ``ou_key`` (one key per
    element), ``I_ext`` ``(B, N)`` or ``(B, n_steps, N)``, ``I_ext_nA`` (same
    shapes), ``V0_mV`` ``(B, N)`` (or ``(B, N, C)`` on the multi-compartment
    path) and ``noise_gain`` ``(B,)``. All batch values must share one B;
    unknown keys, keys given both batched and shared, and mixing
    ``edge_weight`` with ``edge_weight_uS`` (or ``I_ext`` with ``I_ext_nA``)
    are refused with ``ValueError``.

    Each batch element's values are validated on the host before tracing
    (element 0 through :func:`simulate_hh` itself, the rest against the same
    value rules), and point-process inputs (``edge_weight_uS``,
    ``I_ext_nA``) are converted to density forms on the host per element, so
    the traced kernel only ever sees the density path. The traced kernel runs
    with the internal ``_prevalidated`` flag, which skips only those
    already-done value checks; the public serial path is unchanged for every
    input, concrete and traced. The batch runs under
    ``jax.jit(jax.vmap(...))`` with ``n_steps``, ``dt_ms`` and ``morphology``
    closed over (static). Output leaves match :func:`simulate_hh` with one
    leading batch axis (``spikes`` ``[B, n_steps, N]``, ``V_mV``
    ``[B, n_steps, N]``, final states ``[B, N]``, ``held_steps`` counts
    ``[B]``).
    """
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    _require_finite_dt(dt_ms, "dt_ms")
    if not isinstance(batch, Mapping) or not batch:
        raise ValueError(
            "batch must be a non-empty mapping of batch key to stacked values; "
            f"got {batch!r}"
        )
    unknown = [k for k in batch if k not in _HH_BATCH_KEYS]
    if unknown:
        raise ValueError(
            f"unknown batch keys {unknown}; expected a subset of {list(_HH_BATCH_KEYS)}"
        )
    dupes = [k for k in batch if k in shared]
    if dupes:
        raise ValueError(
            f"batch keys {dupes} are also given as shared arguments; pass each input once"
        )
    stacked: dict[str, Any] = {}
    B = None
    for key, value in batch.items():
        arr = jnp.asarray(value)
        if arr.ndim < 1:
            raise ValueError(
                f"batch[{key!r}] must stack B elements along a leading axis; "
                f"got shape {arr.shape}"
            )
        if B is None:
            B = int(arr.shape[0])
            if B < 1:
                raise ValueError(f"batch must hold at least one element; got B={B}")
        elif int(arr.shape[0]) != B:
            raise ValueError(
                f"batch keys must share one leading size B; {key!r} has "
                f"{int(arr.shape[0])}, expected {B}"
            )
        stacked[key] = arr
    assert B is not None
    if (("edge_weight_uS" in stacked or shared.get("edge_weight_uS") is not None)
            and ("edge_weight" in stacked or shared.get("edge_weight") is not None)):
        raise ValueError(
            "edge_weight_uS and edge_weight are alternative input forms; got both"
        )
    if (("I_ext_nA" in stacked or shared.get("I_ext_nA") is not None)
            and ("I_ext" in stacked or shared.get("I_ext") is not None)):
        raise ValueError("I_ext and I_ext_nA are alternative input forms; got both")
    N = int(jnp.shape(params.gNa_mS_cm2)[0])

    if ("edge_weight_uS" in stacked or shared.get("edge_weight_uS") is not None
            or "I_ext_nA" in stacked or shared.get("I_ext_nA") is not None):
        if shared.get("morphology") is None:
            raise ValueError(
                "morphology is required when any compartment selection or "
                "point-process input is given"
            )
        mx = _check_morphology(shared["morphology"])
        area = mx["area"]
        n_comp = mx["n_comp"]
        if "edge_weight_uS" in stacked:
            uw = stacked.pop("edge_weight_uS")
            if uw.ndim != 2:
                raise ValueError(
                    f"batch['edge_weight_uS'] must have shape (B, E); got {uw.shape}"
                )
            n_e = int(uw.shape[1])
            missing = [k for k in ("edge_pre", "edge_post", "edge_receptor")
                       if shared.get(k) is None]
            if missing:
                raise ValueError(
                    f"edge arrays must be all given or all None; missing {missing}"
                )
            post_c = _comp_index_array(shared.get("edge_post_comp"), n_e, "edge_post_comp")
            _check_comp_range(post_c, n_comp, "edge_post_comp")
            pre_c = _comp_index_array(shared.get("edge_pre_comp"), n_e, "edge_pre_comp")
            _check_comp_range(pre_c, n_comp, "edge_pre_comp")
            dens = np.empty((B, n_e), dtype=np.float64)
            for b in range(B):
                wb = np.asarray(uw[b], dtype=np.float64)
                if not np.all(np.isfinite(wb)):
                    raise ValueError(f"edge_weight_uS must be finite; got {wb}")
                if np.any(wb < 0.0):
                    raise ValueError(
                        "edge_weight_uS must be >= 0 (the reversal sets the sign); "
                        f"got min {wb.min()}"
                    )
                dens[b] = point_process_to_distributed(wb, area[post_c])
            stacked["edge_weight"] = jnp.asarray(dens)
        if shared.get("edge_weight_uS") is not None:
            w_us = np.asarray(shared.pop("edge_weight_uS"), dtype=np.float64)
            if w_us.ndim != 1:
                raise ValueError(
                    f"edge_weight_uS must be one-dimensional, got shape {w_us.shape}"
                )
            post_c = _comp_index_array(shared.get("edge_post_comp"), int(w_us.shape[0]),
                                       "edge_post_comp")
            _check_comp_range(post_c, n_comp, "edge_post_comp")
            shared["edge_weight"] = point_process_to_distributed(w_us, area[post_c])
        stim_c = _comp_index_array(shared.get("stim_comp"), N, "stim_comp")
        _check_comp_range(stim_c, n_comp, "stim_comp")
        if "I_ext_nA" in stacked:
            nA_b = stacked.pop("I_ext_nA")
            conv = []
            for b in range(B):
                nb = np.asarray(nA_b[b], dtype=np.float64)
                if nb.shape == (N,):
                    conv.append(point_process_to_distributed(nb, area[stim_c]))
                elif nb.shape == (n_steps, N):
                    conv.append(point_process_to_distributed(
                        nb, np.broadcast_to(area[stim_c], (n_steps, N))))
                else:
                    raise ValueError(
                        f"I_ext_nA must be None, shape ({N},) or ({n_steps}, {N}); "
                        f"got {nb.shape}"
                    )
            stacked["I_ext"] = jnp.asarray(np.asarray(conv, dtype=np.float64))
        if shared.get("I_ext_nA") is not None:
            nA = np.asarray(shared.pop("I_ext_nA"), dtype=np.float64)
            if nA.shape == (N,):
                shared["I_ext"] = point_process_to_distributed(nA, area[stim_c])
            elif nA.shape == (n_steps, N):
                shared["I_ext"] = point_process_to_distributed(
                    nA, np.broadcast_to(area[stim_c], (n_steps, N)))
            else:
                raise ValueError(
                    f"I_ext_nA must be None, shape ({N},) or ({n_steps}, {N}); "
                    f"got {nA.shape}"
                )

    # Host validation: element 0 runs through simulate_hh itself (exact
    # shared and per-element errors), the rest against the same value rules
    # (shapes are static and raise inside the traced call either way).
    elem0 = dict(shared)
    for key, arr in stacked.items():
        elem0[key] = arr[0]
    simulate_hh(params, n_steps=n_steps, dt_ms=dt_ms, **elem0)

    sig_shared = shared.get("ou_sigma", None)
    signp = None if sig_shared is None else np.asarray(sig_shared, dtype=np.float64)
    if shared.get("noise_gain") is not None and "noise_gain" not in stacked:
        try:
            shared_gain = float(shared["noise_gain"])
        except (TypeError, ValueError):
            raise ValueError(
                f"noise_gain must be a number; got {shared['noise_gain']!r}"
            )
    else:
        shared_gain = 1.0
    key_present_shared = shared.get("ou_key") is not None
    tau_present = shared.get("ou_tau") is not None
    for b in range(1, B):
        if "edge_weight" in stacked:
            wb = np.asarray(stacked["edge_weight"][b], dtype=np.float64)
            if not np.all(np.isfinite(wb)):
                raise ValueError(f"edge_weight must be finite; got {wb}")
            if np.any(wb < 0.0):
                raise ValueError(
                    "edge_weight must be >= 0 (the reversal sets the sign); "
                    f"got min {wb.min()}"
                )
        if "ou_key" in stacked:
            try:
                jax.random.split(stacked["ou_key"][b])
            except (TypeError, ValueError):
                raise ValueError(
                    f"ou_key must be a JAX PRNG key; got {stacked['ou_key'][b]!r}"
                )
        if "noise_gain" in stacked:
            gb = stacked["noise_gain"][b]
            try:
                g = float(gb)
            except (TypeError, ValueError):
                raise ValueError(f"noise_gain must be a number; got {gb!r}")
            if not math.isfinite(g) or g < 0.0:
                raise ValueError(f"noise_gain must be finite and >= 0; got {gb!r}")
        else:
            g = shared_gain
        eff = False if signp is None else bool(np.any(signp * g != 0.0))
        if not tau_present and (key_present_shared or "ou_key" in stacked) and eff:
            raise ValueError(
                "ou_tau is required when OU noise is live (ou_key given and "
                "ou_sigma * noise_gain nonzero); got None "
                "(PORT_SPEC.md section 3 gives a per-cell U(10, 50) range, not a default)"
            )
        if eff and not key_present_shared and "ou_key" not in stacked:
            raise ValueError("ou_key is required when ou_sigma * noise_gain is nonzero")

    def _single(slices):
        kw = dict(shared)
        kw.update(slices)
        return simulate_hh(params, n_steps=n_steps, dt_ms=dt_ms,
                           _prevalidated=True, **kw)

    return jax.jit(jax.vmap(_single))(stacked)
