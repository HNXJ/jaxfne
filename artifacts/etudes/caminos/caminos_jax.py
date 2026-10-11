"""CAMINOS stage-A etude: standalone JAX AdEx network (E/PV/SOM).

Reference model: Tahvili, Vinck, di Volo, PLoS Comput Biol 2026
(PMC13286281, CC-BY), Methods Eqs. 15-18, Tables 1-2. Every biophysical
value below comes from ``artifacts/etudes/caminos/PORT_SPEC.md``; nothing
is invented. Whether these equal the Cell Reports 2025 network is an open
assumption until its STAR Methods are read (see PORT_SPEC.md).

Conventions
-----------
* SI units throughout (V, s, F, S, A). ``PARAMS`` carries one commented
  entry per Table 1/2 value.
* Explicit Euler at dt = 0.1 ms, as the paper's Brian2 reference.
* Conductance synapses ``I_syn = g (E_rev - V)`` with exponential decay
  (tau_E = 2 ms, tau_I = 8 ms) and a uniform 2 ms delay realised as a
  ring buffer (20 steps at dt = 0.1 ms).
* External drive: 560 independent Poisson sources at 4 Hz per E and PV
  cell, no drive to SOM. The summed input per cell per step is exactly
  Poisson(560 * 4 Hz * dt); multiplicities are preserved through the
  delay ring (integer counts, not booleans).
* Refractory: on spike V is reset to Vreset and clamped there for Tref
  (integer step counter); w keeps evolving during the clamp, as in
  Brian2, with V = Vreset in its driving term. w += b at the spike.
* Step order differs from Brian2 by one step in two places (second review,
  2026-10-11): an arrival read from the ring enters the V update of the same
  step, so the effective delay is one step shorter than Brian2's; and the
  clamp covers the 50 steps after the spike step, so the refractory period is
  one step longer. PREREG_P3.md S5 tests both.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
while _repo_root.parent != _repo_root and not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jax
import jax.numpy as jnp

PARAMS = {
    # --- PLoS CB Table 1: AdEx cell parameters (SI units) ---
    "E_VT_V": -50e-3,  # Table 1, VT (E)
    "E_Delta_V": 2e-3,  # Table 1, Delta (E)
    "E_Tref_s": 5e-3,  # Table 1, Tref (E)
    "E_tauw_s": 500e-3,  # Table 1, tau_w (E)
    "E_a_S": 4e-9,  # Table 1, a (E)
    "E_b_A": 50e-12,  # Table 1, b (E)
    "E_C_F": 200e-12,  # Table 1, C (E)
    "E_gL_S": 10e-9,  # Table 1, gL (E)
    "E_EL_V": -65e-3,  # Table 1, EL (E)
    "E_Vreset_V": -65e-3,  # Table 1, Vreset (E)
    "PV_VT_V": -50e-3,  # Table 1, VT (PV)
    "PV_Delta_V": 0.5e-3,  # Table 1, Delta (PV)
    "PV_Tref_s": 5e-3,  # Table 1, Tref (PV)
    "PV_tauw_s": 500e-3,  # Table 1, tau_w (PV)
    "PV_a_S": 0.0,  # Table 1, a (PV)
    "PV_b_A": 0.0,  # Table 1, b (PV)
    "PV_C_F": 200e-12,  # Table 1, C (PV)
    "PV_gL_S": 10e-9,  # Table 1, gL (PV)
    "PV_EL_V": -65e-3,  # Table 1, EL (PV)
    "PV_Vreset_V": -65e-3,  # Table 1, Vreset (PV)
    "SOM_VT_V": -50e-3,  # Table 1, VT (SOM)
    "SOM_Delta_V": 2e-3,  # Table 1, Delta (SOM)
    "SOM_Tref_s": 5e-3,  # Table 1, Tref (SOM)
    "SOM_tauw_s": 500e-3,  # Table 1, tau_w (SOM)
    "SOM_a_S": 4e-9,  # Table 1, a (SOM)
    "SOM_b_A": 80e-12,  # Table 1, b (SOM)
    "SOM_C_F": 200e-12,  # Table 1, C (SOM)
    "SOM_gL_S": 10e-9,  # Table 1, gL (SOM)
    "SOM_EL_V": -55e-3,  # Table 1, EL (SOM)
    "SOM_Vreset_V": -65e-3,  # Table 1, Vreset (SOM)
    # --- PLoS CB Table 2: synapse parameters (SI units) ---
    "tau_d_s": 2e-3,  # Table 2, delay, all synapses
    "tauE_s": 2e-3,  # Table 2, tau_E
    "tauI_s": 8e-3,  # Table 2, tau_I
    "ErevE_V": 0.0,  # Table 2, E_rev (E)
    "ErevI_V": -80e-3,  # Table 2, E_rev (I)
    "Q_PV_E_S": 5e-9,  # Table 2, Q PV -> E
    "Q_SOM_E_S": 5e-9,  # Table 2, Q SOM -> E
    "Q_PV_PV_S": 5e-9,  # Table 2, Q PV -> PV
    "Q_SOM_PV_S": 5e-9,  # Table 2, Q SOM -> PV
    "Q_E_E_S": 2e-9,  # Table 2, Q E -> E
    "Q_E_PV_S": 3e-9,  # Table 2, Q E -> PV
    "Q_E_SOM_S": 3e-9,  # Table 2, Q E -> SOM
    "Q_PV_SOM_S": 0.0,  # Table 2, Q PV -> SOM (absent)
    "Q_SOM_SOM_S": 0.0,  # Table 2, Q SOM -> SOM (absent)
    "Q_ext_S": 2.5e-9,  # Table 2, Q Ext -> E, PV
    # --- PLoS CB Methods text: network and drive ---
    "NE": 8000,  # Methods, E population size
    "NPV": 750,  # Methods, PV population size
    "NSOM": 750,  # Methods, SOM population size
    "p": 0.07,  # Methods, connection probability
    "ext_sources": 560,  # Methods, p * NE Poisson sources per target cell
    "ext_rate_Hz": 4.0,  # Methods, rate per external source
    "dt_s": 1e-4,  # Methods, Euler dt (0.1 ms)
}

CELL_TYPES = ("E", "PV", "SOM")

# Clip for the AdEx exponential argument (V - VT) / Delta. The spike
# threshold binds at an argument of 5; exp(20) ~= 4.9e8 keeps float32
# (max ~3.4e38) far from overflow on any reachable trajectory.
_EXP_ARG_MAX = 20.0


def cell_params(kind):
    """Per-type AdEx parameter dict in SI units, from ``PARAMS``."""
    return {
        "VT_V": PARAMS[kind + "_VT_V"],
        "Delta_V": PARAMS[kind + "_Delta_V"],
        "Tref_s": PARAMS[kind + "_Tref_s"],
        "tauw_s": PARAMS[kind + "_tauw_s"],
        "a_S": PARAMS[kind + "_a_S"],
        "b_A": PARAMS[kind + "_b_A"],
        "C_F": PARAMS[kind + "_C_F"],
        "gL_S": PARAMS[kind + "_gL_S"],
        "EL_V": PARAMS[kind + "_EL_V"],
        "Vreset_V": PARAMS[kind + "_Vreset_V"],
    }


def adex_step(state, I, p, dt):
    """One explicit-Euler step of Eqs. 15-16 for a vector of cells.

    ``state`` is ``(V, w, ref)``: membrane voltage (V), adaptation (A),
    refractory steps remaining (int). ``I`` is injected current (A),
    ``p`` a per-type (scalar entries) or per-neuron (array entries)
    parameter dict as returned by :func:`cell_params`, ``dt`` seconds.

    A cell with ``ref > 0`` is clamped at Vreset (V held, w still
    evolving with V = Vreset). An active cell spikes when its updated V
    reaches ``VT + 5 * Delta``; on spike V is reset, w += b, and ref is
    set to ``round(Tref / dt)``. Pure jax.numpy; dtype follows inputs.
    Returns ``(new_state, spk)`` with boolean spike vector ``spk``.
    """
    V, w, ref = state
    v_thr = p["VT_V"] + 5.0 * p["Delta_V"]
    arg = jnp.minimum((V - p["VT_V"]) / p["Delta_V"], _EXP_ARG_MAX)
    dV = (
        p["gL_S"] * (p["EL_V"] - V)
        + p["gL_S"] * p["Delta_V"] * jnp.exp(arg)
        - w
        + I
    ) / p["C_F"] * dt
    dw_act = (p["a_S"] * (V - p["EL_V"]) - w) / p["tauw_s"] * dt
    dw_ref = (p["a_S"] * (p["Vreset_V"] - p["EL_V"]) - w) / p["tauw_s"] * dt
    n_ref = jnp.round(p["Tref_s"] / dt).astype(jnp.int32)
    active = ref <= 0
    v_cand = V + dV
    spk = active & (v_cand >= v_thr)
    v_new = jnp.where(spk, p["Vreset_V"], jnp.where(active, v_cand, p["Vreset_V"]))
    w_new = jnp.where(active, w + dw_act + jnp.where(spk, p["b_A"], 0.0), w + dw_ref)
    ref_new = jnp.where(spk, n_ref, jnp.maximum(ref - 1, 0))
    return (v_new, w_new, ref_new), spk


def build_network(key, p=0.07, NE=8000, NPV=750, NSOM=750):
    """Bernoulli connectivity between the E/PV/SOM blocks.

    Every ordered pair connects independently with probability ``p``;
    diagonal (autapse) entries of the E->E and PV->PV blocks are forced
    to zero. PV->SOM and SOM->SOM are absent in the reference model, so
    no blocks are built for them.

    Returns a dict with sizes, ``p``, and ``blocks`` mapping
    ``(pre, post)`` pairs to dense boolean arrays of shape
    ``[N_pre, N_post]``. Dense booleans at full size hold
    64.0M (E->E) + 6.0M (E->PV) + 6.0M (E->SOM) + 6.0M (PV->E)
    + 0.5625M (PV->PV) + 6.0M (SOM->E) + 0.5625M (SOM->PV)
    = 89.0625M entries ~= 85.0 MiB, acceptable on GPU.
    """
    sizes = {"E": NE, "PV": NPV, "SOM": NSOM}
    pairs = [("E", "E"), ("E", "PV"), ("E", "SOM"), ("PV", "E"),
             ("PV", "PV"), ("SOM", "E"), ("SOM", "PV")]
    keys = jax.random.split(key, len(pairs))
    blocks = {}
    for k, (pre, post) in zip(keys, pairs):
        blk = jax.random.bernoulli(k, p, (sizes[pre], sizes[post]))
        if pre == post:  # no autapses
            n = sizes[pre]
            blk = blk.at[jnp.arange(n), jnp.arange(n)].set(False)
        blocks[(pre, post)] = blk
    return {"NE": NE, "NPV": NPV, "NSOM": NSOM, "p": p, "blocks": blocks}


def _concat_params(sizes, dtype):
    """Per-neuron parameter arrays in [E | PV | SOM] order."""
    out = {}
    for name in ("VT_V", "Delta_V", "Tref_s", "tauw_s", "a_S", "b_A",
                 "C_F", "gL_S", "EL_V", "Vreset_V"):
        out[name] = jnp.concatenate(
            [jnp.full((sizes[k],), cell_params(k)[name], dtype=dtype) for k in CELL_TYPES]
        )
    return out


def route_inputs(blocks, q, arr, earr, sizes, dtype):
    """Conductance increments (gE, gI) from delayed arrivals, in [E | PV | SOM] order.

    ``arr`` holds recurrent spikes of all cells, ``earr`` external counts for E and PV.
    E and external spikes raise gE; PV and SOM spikes raise gI (Table 2, E_rev I).
    """
    NE, NPV, NSOM = sizes["E"], sizes["PV"], sizes["SOM"]
    ae, ap, as_ = arr[:NE], arr[NE:NE + NPV], arr[NE + NPV:]

    def mv(pair, x):
        return jnp.matmul(blocks[pair].T.astype(dtype), x.astype(dtype))

    ge = jnp.concatenate([
        q["q_ee"] * mv(("E", "E"), ae) + q["q_ext"] * earr[:NE].astype(dtype),
        q["q_ep"] * mv(("E", "PV"), ae) + q["q_ext"] * earr[NE:].astype(dtype),
        q["q_es"] * mv(("E", "SOM"), ae),
    ])
    gi = jnp.concatenate([
        q["q_pe"] * mv(("PV", "E"), ap) + q["q_se"] * mv(("SOM", "E"), as_),
        q["q_pp"] * mv(("PV", "PV"), ap) + q["q_sp"] * mv(("SOM", "PV"), as_),
        jnp.zeros((NSOM,), dtype),  # PV->SOM and SOM->SOM absent
    ])
    return ge, gi


def simulate(key, T, dt=1e-4, p=0.07, dtype=jnp.float32, record="counts",
             NE=None, NPV=None, NSOM=None):
    """Run the full E/PV/SOM network for ``T`` seconds (Euler, step ``dt``).

    ``NE/NPV/NSOM`` default to the reference sizes in ``PARAMS`` and are
    exposed only so small smoke nets can use this same path. ``record``
    must be ``"counts"`` (any other value is refused). Seeding comes
    only through ``key``: one split builds connectivity, the other
    drives the external Poisson sources inside the scan.

    Per step, in order: (i) pop spikes (recurrent booleans, external
    integer counts) delayed by ``round(tau_d / dt)`` steps from the ring
    buffers; (ii) decay conductances by ``(1 - dt / tau)`` and add
    ``Q * arrivals``; (iii) ``I_syn = gE (ErevE - V) + gI (ErevI - V)``;
    (iv) :func:`adex_step`; (v) push fresh spikes/counts into the rings.

    Returns ``(counts, final_state)``: per-step population spike counts
    ``[E, PV, SOM]`` as int32 of shape ``[steps, 3]``, and a dict with
    final ``V, w, ref, gE, gI`` plus sizes.
    """
    if record != "counts":
        raise ValueError("only record='counts' is supported, got %r" % (record,))
    NE = PARAMS["NE"] if NE is None else NE
    NPV = PARAMS["NPV"] if NPV is None else NPV
    NSOM = PARAMS["NSOM"] if NSOM is None else NSOM
    sizes = {"E": NE, "PV": NPV, "SOM": NSOM}
    n_tot = NE + NPV + NSOM
    steps = int(round(T / dt))
    n_delay = max(int(round(PARAMS["tau_d_s"] / dt)), 1)

    k_net, k_ext = jax.random.split(key)
    net = build_network(k_net, p=p, NE=NE, NPV=NPV, NSOM=NSOM)
    blocks = net["blocks"]
    par = _concat_params(sizes, dtype)

    def _f(x):
        return jnp.asarray(x, dtype=dtype)

    el = jnp.concatenate(
        [jnp.full((sizes[k],), cell_params(k)["EL_V"], dtype=dtype) for k in CELL_TYPES]
    )
    v0 = el
    w0 = jnp.zeros((n_tot,), dtype)
    r0 = jnp.zeros((n_tot,), jnp.int32)
    ge0 = jnp.zeros((n_tot,), dtype)
    gi0 = jnp.zeros((n_tot,), dtype)
    ring0 = jnp.zeros((n_delay, n_tot), bool)
    ering0 = jnp.zeros((n_delay, NE + NPV), jnp.int32)
    head0 = jnp.int32(0)

    decay_e = _f(1.0 - dt / PARAMS["tauE_s"])
    decay_i = _f(1.0 - dt / PARAMS["tauI_s"])
    erev_e = _f(PARAMS["ErevE_V"])
    erev_i = _f(PARAMS["ErevI_V"])
    q_ee = _f(PARAMS["Q_E_E_S"])
    q_ep = _f(PARAMS["Q_E_PV_S"])
    q_es = _f(PARAMS["Q_E_SOM_S"])
    q_pe = _f(PARAMS["Q_PV_E_S"])
    q_pp = _f(PARAMS["Q_PV_PV_S"])
    q_se = _f(PARAMS["Q_SOM_E_S"])
    q_sp = _f(PARAMS["Q_SOM_PV_S"])
    q_ext = _f(PARAMS["Q_ext_S"])
    lam = PARAMS["ext_sources"] * PARAMS["ext_rate_Hz"] * dt
    sl = {"E": slice(0, NE), "PV": slice(NE, NE + NPV),
          "SOM": slice(NE + NPV, n_tot)}
    q ={"q_ee": q_ee, "q_ep": q_ep, "q_es": q_es, "q_pe": q_pe, "q_pp": q_pp,
         "q_se": q_se, "q_sp": q_sp, "q_ext": q_ext}

    def step_fn(carry, _):
        V, w, ref, gE, gI, ring, ering, head, rkey = carry
        rkey, ks = jax.random.split(rkey)
        inc_gE, inc_gI = route_inputs(blocks, q, ring[head], ering[head], sizes, dtype)
        gE = gE * decay_e + inc_gE
        gI = gI * decay_i + inc_gI
        I = gE * (erev_e - V) + gI * (erev_i - V)
        (V, w, ref), spk = adex_step((V, w, ref), I, par, dt)
        ext = jax.random.poisson(ks, lam, shape=(NE + NPV,)).astype(jnp.int32)
        ring = ring.at[head].set(spk)
        ering = ering.at[head].set(ext)
        head = (head + 1) % n_delay
        ce = jnp.sum(spk[sl["E"]]).astype(jnp.int32)
        cp = jnp.sum(spk[sl["PV"]]).astype(jnp.int32)
        cs = jnp.sum(spk[sl["SOM"]]).astype(jnp.int32)
        return (V, w, ref, gE, gI, ring, ering, head, rkey), jnp.array([ce, cp, cs])

    init = (v0, w0, r0, ge0, gi0, ring0, ering0, head0, k_ext)
    (V, w, ref, gE, gI, _, _, _, _), counts = jax.jit(
        lambda c: jax.lax.scan(step_fn, c, None, length=steps))(init)
    jax.block_until_ready(counts)
    final = {"V": V, "w": w, "ref": ref, "gE": gE, "gI": gI,
             "NE": NE, "NPV": NPV, "NSOM": NSOM}
    return counts, final
