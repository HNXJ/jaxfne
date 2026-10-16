"""Time-binned JAX port of the PCL two-layer network (port stage B).

Same network and config loading as the event-exact oracle pcl_event_ref.py (stage A). Stage A is
imported and its constructor reads the configs, weights, geometry and cover tables. Time is cut
into bins of dt_us. Events of a bin are summed per position and applied at the bin start
T_b = b * dt_us. One lax.scan step per bin. float32 throughout, x64 off, no device-specific code.

Within a bin the simple layer runs a depth-sequential cascade per position, in the C++ visit order
(SpikingNetwork.cpp:41-67): depths are visited in ascending index; a depth above threshold at its
visit spikes, and its local inhibition lowers every other depth at that position (the visited ones
and the not-yet-visited ones). The next winner is the lowest-index depth above threshold after the
earlier winners' inhibition. The cascade is resolved in R_ROUNDS rounds; a crossing that the rounds
do not reach is counted as missed (it must be 0 on the gate data). The complex layer runs the same
scheme per complex position, on the summed input of the bin's simple spikes.

Deviations from stage A (also in the port report):
- Exact only when a bin holds at most one event per simple position. Events of a bin that share a
  position are summed and checked once; the C++ applies them one after another (the report gives
  the share of such bins for each dt).
- A cell spikes at most once per bin. Missed crossings: a position whose next winner the R_ROUNDS
  rounds do not reach is counted in missed_s / missed_c (must be 0 on the gate data).
- Complex input is summed over the bin's simple winners, and the complex cascade runs after the whole
  simple cascade. C++ forwards each simple spike to the complex layer before the next depth.
- Lateral and top-down inhibition are applied after their layer's cascade, not interleaved with it.
- Refractory and adaptation: C++ subtracts R and A per event at the event time; here k events at a
  position subtract k * (R(T_b) + A(T_b)) per visit, with R and A read at the bin time.
- Never-spiked is a flag, not the C++ t = 0 sentinel (differs only for a spike at exactly t = 0).
- Learning uses traces, not a per-event loop: LTP and LTD sums over the events since the last spike,
  weight read once per spike interval; updates of depth-shared weights summed over the neurons that
  spike in the bin, clipped and L1-normalised once per bin. Local inhibition weights used by the
  cascade are the bin-start values. Stage A's 1000-entry spike buffer (CAP, Neuron.cpp:22) is not modelled.
- Not ported: threshold adaptation (ETA_TA != 0), thresholds below MIN_THRESH (stage A's
  measurement clamp), and lateral or top-down plasticity (learning with those maps raises).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root(start: Path) -> Path:
    """Nearest ancestor that holds pyproject.toml."""
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_HERE = Path(__file__).resolve().parent
_ROOT = str(_repo_root(_HERE))
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(_HERE) not in sys.path:
    sys.path.insert(1, str(_HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

import pcl_event_ref as ref  # noqa: E402

if bool(getattr(jax.config, "jax_enable_x64", False)):
    raise RuntimeError("pcl_binned_jax is float32 throughout: run with jax_enable_x64 off")

F32 = jnp.float32
NEG_LIMIT = ref.NEG_LIMIT  # -20, Neuron.cpp:26
NEVER = -1.0e30  # time of "no spike yet"; the flag is tsp > HAS_SPIKED
HAS_SPIKED = -1.0e29
R_ROUNDS = 4  # declared: winners per position and per bin that the cascade rounds can resolve
# Soft bounds hard-coded in the C++ (PORT_SPEC, Plasticity): (w_max, factor) per rule.
SOFT_S_EXC = (3.0, 0.33)  # SimpleNeuron.cpp:196-216
SOFT_S_LOC = (50.0, 0.02)  # SimpleNeuron.cpp:222-237
SOFT_C_EXC = (4.0, 0.25)  # ComplexNeuron.cpp:101-138
SOFT_C_LOC = (25.0, 0.04)  # ComplexNeuron.cpp:146-165


@dataclass(frozen=True)
class Flags:
    """Static switches; a change recompiles the scan."""

    local_s: bool  # SimpleNeuron local inhibition (layerInhibitions[0] has "local")
    local_c: bool  # ComplexNeuron local inhibition (layerInhibitions[1] has "local")
    lat: bool  # lateral inhibition on and activation on (SpikingNetwork.cpp:60)
    td: bool  # top-down inhibition on and activation on (SpikingNetwork.cpp:98)
    learn_s: bool  # simple excitatory and local STDP (STDP_LEARNING excitatory or all)
    learn_c: bool  # complex excitatory and local STDP (STDP_LEARNING excitatory or all)


def _l1(w, axes, target):
    """WeightMatrix::normalizeL1Norm (WeightMatrix.cpp:151): scale so that sum |w| over axes = target."""
    norm = jnp.sum(jnp.abs(w), axis=axes, keepdims=True)
    ok = norm != 0.0
    scale = jnp.where(ok, target / jnp.where(ok, norm, 1.0), 1.0)
    return w * scale


def _inh_sum(F, M):
    """sum over w of F[..., w] * M[d, w], returned with the depth axis last."""
    return jnp.einsum("...w,dw->...d", F, M)


def _cascade(V, I, Kn, RA, RA_post, thr, inh, vreset, rounds):
    """Depth-sequential cascade of one bin at every position (C++ visit order, see module docstring).

    V: potential after decay, before this bin's input (..., D). I: summed input. Kn: events (simple
    spikes for the complex layer) at the position in this bin; every depth is visited once per event.
    RA: refractory + adaptation at the bin time before a spike (C++ refractoryPotential and A).
    RA_post: the same for a depth that spikes at the bin time. inh[d, w]: inhibition from w to d
    with a zero diagonal. vreset: reset potential. rounds: declared number of winner rounds.

    Returns (winners (..., D) bool, potential after the bin (..., D), missed: number of positions
    whose next crossing the rounds did not reach; must be 0).
    """
    D = V.shape[-1]
    dix = jnp.arange(D, dtype=jnp.int32)
    act = Kn > 0
    lead = V.shape[:-1]

    def visit(found):
        # Visit value of every depth given the winners found so far. For a candidate (a depth
        # above the last winner) all found winners lie below it, so the inhibition is C++ order.
        Fl = found.astype(F32)
        sI = _inh_sum(Fl, inh)
        nF = jnp.sum(Fl, axis=-1, keepdims=True)
        vin = jnp.maximum(V - sI - nF * RA, NEG_LIMIT)  # inhibition, then clamp at -20
        return jnp.maximum(vin + I - Kn * RA, NEG_LIMIT)  # excitation, then clamp at -20

    found = jnp.zeros(V.shape, dtype=bool)
    lo = jnp.full(lead + (1,), -1, dtype=jnp.int32)  # last winner index per position
    for _ in range(rounds):
        cross = act & (dix > lo) & (visit(found) > thr)
        has = jnp.any(cross, axis=-1, keepdims=True)
        ws = jnp.argmax(cross.astype(jnp.int32), axis=-1)[..., None].astype(jnp.int32)
        found = found | ((dix == ws) & has)
        lo = jnp.where(has, ws, lo)
    cross = act & (dix > lo) & (visit(found) > thr)
    missed = jnp.sum(jnp.any(cross, axis=-1)).astype(jnp.int32)

    # Potential after the bin. A depth's visit sees only the winners below it; the winners above it
    # inhibit it after its visit. A winner is reset to vreset at its visit, and its refractory and
    # adaptation after the spike are RA_post.
    Fl = found.astype(F32)
    lowm = (dix[None, :] < dix[:, None]).astype(F32)  # [d, w] = 1 if w < d
    upm = (dix[None, :] > dix[:, None]).astype(F32)  # [d, w] = 1 if w > d
    sI_lo = _inh_sum(Fl, inh * lowm)
    n_lo = _inh_sum(Fl, lowm)
    vin_lo = jnp.maximum(V - sI_lo - n_lo * RA, NEG_LIMIT)
    vis_lo = jnp.maximum(vin_lo + I - Kn * RA, NEG_LIMIT)
    base = jnp.where(found, vreset, vis_lo)
    RA_t = jnp.where(found, RA_post, RA)
    sI_up = _inh_sum(Fl, inh * upm)
    n_up = _inh_sum(Fl, upm)
    V_fin = jnp.maximum(base - sI_up - n_up * RA_t, NEG_LIMIT)
    return found, V_fin, missed


def _step(st, x, K, lane, ex_pad, ey_pad, ep_pad, fl):
    """One bin at T = b * dt: input, simple cascade, learning, complex cascade, post inhibition."""
    T, start, cnt = x
    NX, NY, ND = st["Vs"].shape
    SX, SY = st["Ws"].shape[2:]
    CX, CY, CD = st["Vc"].shape
    CSX, CSY = st["Wc"].shape[1:3]
    NS = NX * NY * ND
    f32 = F32

    # Decay to T. Per-event C++ decays compose to these per-bin factors exactly.
    Vs = st["Vs"] * K["dm_s"]
    As = st["As"] * K["dsra"]
    Vc = st["Vc"] * K["dm_c"]
    tsp, tpv, tc, tcp = st["tsp"], st["tpv"], st["tc"], st["tcp"]
    Ws, Wi, Wc, Wic = st["Ws"], st["Wi"], st["Wc"], st["Wic"]
    if fl.learn_s:
        Gs = st["Gs"] * K["dl_s"]
        Hs = st["Hs"]
        GLs = st["GLs"] * K["dl_s"]
        HLs = st["HLs"]
    if fl.learn_c:
        Gc = st["Gc"] * K["dl_c"]
        Hc = st["Hc"]
        GLc = st["GLc"] * K["dl_c"]
        HLc = st["HLc"]

    # Input events of this bin (padded to the lane width M). Each event reaches up to KX*KY
    # simple positions and all ND depths (SpikingNetwork.cpp:41-67).
    idx = start + lane
    vm = lane < cnt
    ex_b = ex_pad[idx]
    ey_b = ey_pad[idx]
    ep_b = ep_pad[idx]
    ci_ = K["cvx"][ex_b]  # (M, KX): simple i covering x, -1 if none
    cj_ = K["cvy"][ey_b]  # (M, KY)
    pv = ((ci_ >= 0) & vm[:, None])[:, :, None] & (cj_ >= 0)[:, None, :]  # (M, KX, KY)
    i_c = jnp.maximum(ci_, 0)
    j_c = jnp.maximum(cj_, 0)
    xr = jnp.where(pv, (ex_b[:, None] - K["s_ox"][i_c])[:, :, None], 0)
    yr = jnp.where(pv, (ey_b[:, None] - K["s_oy"][j_c])[:, None, :], 0)
    pp = jnp.where(pv, ep_b[:, None, None], 0)
    nb = jnp.where(pv, (i_c[:, :, None] * NY + j_c[:, None, :]) * ND, NS)
    tgt = nb[..., None] + jnp.arange(ND, dtype=jnp.int32)  # (M, KX, KY, ND); NS = dump slot
    wv = jnp.moveaxis(Ws[:, pp, xr, yr], 0, -1) * pv[..., None]  # (M, KX, KY, ND)
    pvf = jnp.broadcast_to(pv[..., None], tgt.shape).astype(f32).reshape(-1)
    tgt_f = tgt.reshape(-1)
    In_s = jnp.zeros(NS + ND, f32).at[tgt_f].add(wv.reshape(-1))[:NS].reshape(NX, NY, ND)
    Kin = jnp.zeros(NS + ND, f32).at[tgt_f].add(pvf)[:NS].reshape(NX, NY, ND)

    has_s = tsp > HAS_SPIKED  # before this bin's spikes
    if fl.learn_s:
        syn = (pp * (SX * SY) + xr * SY + yr)[..., None]  # synapse within a depth: (polarity, x, y)
        idx2 = (tgt * (2 * SX * SY) + syn).reshape(-1)
        ev = jnp.zeros((NS + ND) * 2 * SX * SY, f32).at[idx2].add(pvf)
        ev_s = ev[: NS * 2 * SX * SY].reshape(NX, NY, ND, 2, SX, SY)
        # LTD reference is the last spike at accumulation time: t_last_spk at the next spike.
        fac_s = jnp.where(has_s, jnp.exp(-(T - tsp) / K["tltd_s"]), 0.0)
        Gs = Gs + ev_s
        Hs = Hs + ev_s * fac_s[..., None, None, None]

    # Simple layer: depth-sequential cascade per position, C++ visit order. Adaptation of a
    # visited depth is read before the spike; the winner's post-spike terms are RA_s_post.
    R_s = jnp.where(has_s, K["rp_s"] * jnp.exp(-(T - tsp) / K["trp_s"]), 0.0)
    RA_s = R_s + As
    RA_s_post = K["rp_s"] + As + K["dsra_inc"]
    inh_s = Wi * K["offd_s"] if fl.local_s else jnp.zeros_like(Wi)
    S_s, Vs, miss_s = _cascade(Vs, In_s, Kin, RA_s, RA_s_post, K["thr_s"], inh_s, K["vr_s"], R_ROUNDS)
    Sf = S_s.astype(f32)

    if fl.learn_s:
        # Excitatory STDP at each spike (SimpleNeuron.cpp:186-220), traces read at T.
        has_pv = tpv > HAS_SPIKED
        Wb = Ws[None, None]
        dec = K["dec_s"][..., None, None, None]
        t_exc = (SOFT_S_EXC[0] - Wb) * SOFT_S_EXC[1] * dec * K["eLTP_s"] * Gs
        t_exc = t_exc + jnp.where(has_pv[..., None, None, None],
                                  Wb * SOFT_S_EXC[1] * K["eLTD_s"] * Hs, 0.0)
        dW = jnp.sum(t_exc * S_s[..., None, None, None], axis=(0, 1))
        upd = jnp.any(S_s, axis=(0, 1))
        Ws = jnp.where(upd[:, None, None, None],
                       _l1(jnp.maximum(Ws + dW, 0.0), (1, 2, 3), K["NORM_s"]), Ws)
        # Local-inhibition STDP at each spike (SimpleNeuron.cpp:222-237), row of the target depth.
        Wib = Wi[None, None]
        t_inh = (SOFT_S_LOC[0] - Wib) * SOFT_S_LOC[1] * K["iLTP_s"] * GLs
        t_inh = t_inh + jnp.where(has_pv[..., None], Wib * SOFT_S_LOC[1] * K["iLTD_s"] * HLs, 0.0)
        dWi = jnp.sum(t_inh * S_s[..., None], axis=(0, 1))
        upd_i = jnp.any(S_s, axis=(0, 1))
        Wi = jnp.where(upd_i[:, None], _l1(jnp.maximum(Wi + dWi, 0.0), (1,), K["ETA_INH_s"]), Wi)

    # Spike bookkeeping: previous and last spike times, adaptation increment.
    tpv = jnp.where(S_s, tsp, tpv)
    tsp = jnp.where(S_s, T, tsp)
    As = As + jnp.where(S_s, K["dsra_inc"], 0.0)
    if fl.learn_s:
        Gs = jnp.where(S_s[..., None, None, None], 0.0, Gs)
        Hs = jnp.where(S_s[..., None, None, None], 0.0, Hs)
        GLs = jnp.where(S_s[..., None], 0.0, GLs)
        HLs = jnp.where(S_s[..., None], 0.0, HLs)

    has_post = tsp > HAS_SPIKED
    R_post = jnp.where(has_post, K["rp_s"] * jnp.exp(-(T - tsp) / K["trp_s"]), 0.0)
    if fl.local_s and fl.learn_s:
        # Inhibition events of this bin's winners, after the reset (they open the next interval).
        Cl = Sf[:, :, None, :] * K["offd_s"]  # (NX, NY, target d, source k)
        GLs = GLs + Cl
        HLs = HLs + Cl * jnp.where(has_post, jnp.exp(-(T - tsp) / K["tltd_s"]), 0.0)[..., None]

    # Lateral inhibition (SpikingNetwork.cpp:157-165): LAT[m, n] per spike of n, targets m.
    # Applied after the cascade (deviation, see module docstring).
    if fl.lat:
        sflat = Sf.reshape(-1)
        Inh_l = (K["lat"] @ sflat).reshape(NX, NY, ND)
        n_l = (K["lmaskT"] @ sflat).reshape(NX, NY, ND)
        Vs = Vs - (Inh_l + n_l * (R_post + As))
    Vs = jnp.maximum(Vs, NEG_LIMIT)

    # Complex layer: input from the bin's simple winners (SpikingNetwork.cpp:87-107), then the same
    # cascade per complex position over the complex depths.
    S_pad = jnp.pad(Sf, ((0, 1), (0, 1), (0, 0)))
    Sg = S_pad[K["ci"][:, :, None, None], K["cj"][None, None, :, :], :]  # (CX, CSX, CY, CSY, ND)
    Ic = jnp.einsum("pxqyk,zxyk->pqz", Sg, Wc)
    Kc = jnp.broadcast_to(jnp.sum(Sg, axis=(1, 3, 4))[:, :, None], (CX, CY, CD))
    has_c = tc > HAS_SPIKED  # before this bin's complex spikes
    if fl.learn_c:
        ev_c = jnp.broadcast_to(jnp.transpose(Sg, (0, 2, 1, 3, 4))[:, :, None],
                                (CX, CY, CD, CSX, CSY, ND))
        fac_c = jnp.where(has_c, jnp.exp(-(T - tc) / K["tltd_c"]), 0.0)
        Gc = Gc + ev_c
        Hc = Hc + ev_c * fac_c[..., None, None, None]

    R_c = jnp.where(has_c, K["rp_c"] * jnp.exp(-(T - tc) / K["trp_c"]), 0.0)  # no adaptation term
    inh_c = Wic * K["offd_c"] if fl.local_c else jnp.zeros_like(Wic)
    S_c, Vc, miss_c = _cascade(Vc, Ic, Kc, R_c, K["rp_c"], K["thr_c"], inh_c, K["vr_c"], R_ROUNDS)
    Scf = S_c.astype(f32)

    if fl.learn_c:
        # Complex excitatory STDP at each spike (ComplexNeuron.cpp:95-141); the counter gate in
        # the C++ is always true here, because the counter is reset inside the block.
        has_pv_c = tcp > HAS_SPIKED
        Wcb = Wc[None, None]
        t_c = (SOFT_C_EXC[0] - Wcb) * SOFT_C_EXC[1] * K["eLTP_c"] * Gc
        t_c = t_c + jnp.where(has_pv_c[..., None, None, None],
                              Wcb * SOFT_C_EXC[1] * K["eLTD_c"] * Hc, 0.0)
        dWc = jnp.sum(t_c * S_c[..., None, None, None], axis=(0, 1))
        upd_c = jnp.any(S_c, axis=(0, 1))
        Wc = jnp.where(upd_c[:, None, None, None],
                       _l1(jnp.maximum(Wc + dWc, 0.0), (1, 2, 3), K["NORM_c"]), Wc)
        # Complex local STDP (ComplexNeuron.cpp:144-166): LTD uses the weight after LTP.
        Wibc = Wic[None, None]
        dLTP = (SOFT_C_LOC[0] - Wibc) * SOFT_C_LOC[1] * K["iLTP_c"] * GLc
        dLTD = jnp.where(has_pv_c[..., None],
                         (Wibc + dLTP) * SOFT_C_LOC[1] * K["iLTD_c"] * HLc, 0.0)
        dWic = jnp.sum((dLTP + dLTD) * S_c[..., None], axis=(0, 1))
        upd_ic = jnp.any(S_c, axis=(0, 1))
        Wic = jnp.where(upd_ic[:, None], _l1(jnp.maximum(Wic + dWic, 0.0), (1,), K["ETA_INH_c"]), Wic)

    tcp = jnp.where(S_c, tc, tcp)
    tc = jnp.where(S_c, T, tc)
    if fl.learn_c:
        Gc = jnp.where(S_c[..., None, None, None], 0.0, Gc)
        Hc = jnp.where(S_c[..., None, None, None], 0.0, Hc)
        GLc = jnp.where(S_c[..., None], 0.0, GLc)
        HLc = jnp.where(S_c[..., None], 0.0, HLc)
    has_cp = tc > HAS_SPIKED
    if fl.local_c and fl.learn_c:
        Clc = Scf[:, :, None, :] * K["offd_c"]
        GLc = GLc + Clc
        HLc = HLc + Clc * jnp.where(has_cp, jnp.exp(-(T - tc) / K["tltd_c"]), 0.0)[..., None]

    # Top-down inhibition of simple cells (SpikingNetwork.cpp:142-151): TD[s, c] per complex spike.
    # Applied after the complex cascade (deviation, see module docstring).
    if fl.td:
        cflat = Scf.reshape(-1)
        Inh_t = (K["td"] @ cflat).reshape(NX, NY, ND)
        n_t = (K["tdm"] @ cflat).reshape(NX, NY, ND)
        Vs = Vs - (Inh_t + n_t * (R_post + As))
    Vs = jnp.maximum(Vs, NEG_LIMIT)

    new = dict(Vs=Vs, As=As, tsp=tsp, tpv=tpv, Vc=Vc, tc=tc, tcp=tcp,
               Ws=Ws, Wi=Wi, Wc=Wc, Wic=Wic)
    if fl.learn_s:
        new.update(Gs=Gs, Hs=Hs, GLs=GLs, HLs=HLs)
    if fl.learn_c:
        new.update(Gc=Gc, Hc=Hc, GLc=GLc, HLc=HLc)
    return new, (S_s, S_c, miss_s, miss_c)


def _scan_impl(st, K, xs, lane, ex_pad, ey_pad, ep_pad, fl):
    def body(carry, x):
        return _step(carry, x, K, lane, ex_pad, ey_pad, ep_pad, fl)

    return lax.scan(body, st, xs)


_scan = jax.jit(_scan_impl, static_argnames=("fl",))


def _cwindow(off, n_src, n_rf):
    """Indices off[o] + e for e < n_rf; out-of-range entries point at the zero pad row n_src."""
    idx = np.array([[o + e for e in range(n_rf)] for o in off], dtype=np.int64)
    ok = (idx >= 0) & (idx < n_src)
    return np.where(ok, idx, n_src).astype(np.int32)


def _check_complex_cover(a, ci, cj):
    """The gather must cover each simple position with the same complex cells as stage A's c_cover."""
    for i in range(a.NX):
        for j in range(a.NY):
            want = sorted(a.c_cover[(i, j)])
            got = sorted((cx, cy) for cx in range(a.CX) for cy in range(a.CY)
                         if i in ci[cx] and j in cj[cy])
            if want != got:
                raise RuntimeError(f"complex gather differs from c_cover at simple ({i}, {j})")


def _cover_tables(a):
    """Padded cover lists: simple positions covering each pixel x (cvx) and y (cvy), -1 if none."""
    KX = max(len(v) for v in a.cover_x)
    KY = max(len(v) for v in a.cover_y)
    cvx = np.full((a.VW, KX), -1, dtype=np.int32)
    for x, lst in enumerate(a.cover_x):
        cvx[x, : len(lst)] = lst
    cvy = np.full((a.VH, KY), -1, dtype=np.int32)
    for y, lst in enumerate(a.cover_y):
        cvy[y, : len(lst)] = lst
    return cvx, cvy


def _consts(a, dt: int, fl: Flags) -> dict:
    f = lambda v: jnp.asarray(v, dtype=F32)  # noqa: E731
    sc, cc = a.scfg, a.ccfg
    cvx, cvy = _cover_tables(a)
    ci = _cwindow(a.c_ox, a.NX, a.CSX)
    cj = _cwindow(a.c_oy, a.NY, a.CSY)
    _check_complex_cover(a, ci, cj)
    shape_s = (a.NX, a.NY, a.ND)
    shape_c = (a.CX, a.CY, a.CD)
    K = {
        "cvx": jnp.asarray(cvx), "cvy": jnp.asarray(cvy),
        "s_ox": jnp.asarray(np.asarray(a.s_ox, dtype=np.int32)),
        "s_oy": jnp.asarray(np.asarray(a.s_oy, dtype=np.int32)),
        "ci": jnp.asarray(ci), "cj": jnp.asarray(cj),
        "thr_s": f(np.array([c.thr for c in a.s_cells]).reshape(shape_s)),
        "thr_c": f(np.array([c.thr for c in a.c_cells]).reshape(shape_c)),
        "dec_s": f(np.array([c.decay for c in a.s_cells]).reshape(shape_s)),
        "dm_s": f(math.exp(-dt / sc.TAU_M)),
        "dm_c": f(math.exp(-dt / cc.TAU_M)),
        "dsra": f(math.exp(-dt / sc.TAU_SRA)),
        "dsra_inc": f(sc.DELTA_SRA),
        "vr_s": f(sc.VRESET), "vr_c": f(cc.VRESET),
        "rp_s": f(sc.DELTA_RP), "trp_s": f(sc.TAU_RP),
        "rp_c": f(cc.DELTA_RP), "trp_c": f(cc.TAU_RP),
        "dl_s": f(math.exp(-dt / sc.TAU_LTP)), "dl_c": f(math.exp(-dt / cc.TAU_LTP)),
        "tltd_s": f(sc.TAU_LTD), "tltd_c": f(cc.TAU_LTD),
        "eLTP_s": f(sc.eLTP), "eLTD_s": f(sc.eLTD), "iLTP_s": f(sc.iLTP), "iLTD_s": f(sc.iLTD),
        "NORM_s": f(sc.NORM_FACTOR), "ETA_INH_s": f(sc.ETA_INH),
        "eLTP_c": f(cc.eLTP), "eLTD_c": f(cc.eLTD), "iLTP_c": f(cc.iLTP), "iLTD_c": f(cc.iLTD),
        "NORM_c": f(cc.NORM_FACTOR), "ETA_INH_c": f(cc.ETA_INH),
        "offd_s": f(1.0 - np.eye(a.ND)), "offd_c": f(1.0 - np.eye(a.CD)),
    }
    if fl.lat:
        K["lat"] = f(a.LAT)  # (NS, NS), LAT[m, n]
        K["lmaskT"] = f(np.ascontiguousarray(a.lat_mask.T))
    else:
        K["lat"] = f(np.zeros((1, 1)))
        K["lmaskT"] = f(np.zeros((1, 1)))
    if fl.td:
        K["td"] = f(a.TD)  # (NS, NC), TD[s, c]
        K["tdm"] = f(a.td_mask)
    else:
        K["td"] = f(np.zeros((1, 1)))
        K["tdm"] = f(np.zeros((1, 1)))
    return K


def _initial_state(a, fl: Flags) -> dict:
    f = lambda v: jnp.asarray(v, dtype=F32)  # noqa: E731
    NX, NY, ND, SX, SY = a.NX, a.NY, a.ND, a.SX, a.SY
    CX, CY, CD, CSX, CSY = a.CX, a.CY, a.CD, a.CSX, a.CSY
    z = lambda shape: jnp.zeros(shape, dtype=F32)  # noqa: E731
    never_s = jnp.full((NX, NY, ND), NEVER, dtype=F32)
    never_c = jnp.full((CX, CY, CD), NEVER, dtype=F32)
    st = {
        "Vs": z((NX, NY, ND)), "As": z((NX, NY, ND)), "tsp": never_s, "tpv": never_s,
        "Vc": z((CX, CY, CD)), "tc": never_c, "tcp": never_c,
        "Ws": f(a.w_s), "Wi": f(a.wi_s), "Wc": f(a.w_c), "Wic": f(a.wi_c),
    }
    if fl.learn_s:
        st.update(Gs=z((NX, NY, ND, 2, SX, SY)), Hs=z((NX, NY, ND, 2, SX, SY)),
                  GLs=z((NX, NY, ND, ND)), HLs=z((NX, NY, ND, ND)))
    if fl.learn_c:
        st.update(Gc=z((CX, CY, CD, CSX, CSY, ND)), Hc=z((CX, CY, CD, CSX, CSY, ND)),
                  GLc=z((CX, CY, CD, CD)), HLc=z((CX, CY, CD, CD)))
    return st


def _spike_times(mask: np.ndarray, first: int, dt: int) -> list:
    """Per-cell spike times in microseconds from a (bins, cells) boolean raster."""
    out = [[] for _ in range(mask.shape[1])]
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        out[c].append((first + r) * dt)
    return out


class PCLBinned:
    """Time-binned network built from a stage-A network: same configs, weights and geometry.

    directory: a network folder as stage A reads it, or None for DefaultConfig.cpp values.
    dt_us: bin width in microseconds (declared; default 100).
    Call run() with time-ordered events; the bin clock advances, so a later call continues the
    same network.
    """

    def __init__(self, directory=None, *, seed: int = 0, learning: bool = False,
                 activation: bool = True, dt_us: int = 100) -> None:
        dt = int(dt_us)
        if dt != dt_us or dt < 1:
            raise ValueError(f"dt_us must be a positive integer, got {dt_us}")
        a = ref.PCLEventRef(directory, seed=seed, learning=learning, activation=activation)
        if a.scfg.ETA_SR != 0.0 or a.ccfg.ETA_SR != 0.0:
            raise NotImplementedError("threshold adaptation (ETA_TA) is not ported")
        if a.scfg.VTHRESH < a.scfg.MIN_THRESH or a.ccfg.VTHRESH < a.ccfg.MIN_THRESH:
            raise NotImplementedError("a threshold below MIN_THRESH needs stage A's measurement clamp")
        if learning and (a.lat_on or a.td_on):
            raise NotImplementedError("lateral and top-down plasticity are not ported; use learning=False")
        fl = Flags(
            local_s=bool(a.local_s), local_c=bool(a.local_c),
            lat=bool(a.lat_on and activation), td=bool(a.td_on and activation),
            learn_s=bool(learning and a.scfg.STDP_LEARNING in ("excitatory", "all")),
            learn_c=bool(learning and a.ccfg.STDP_LEARNING in ("excitatory", "all")),
        )
        self.dt = dt
        self.flags = fl
        self.NX, self.NY, self.ND = a.NX, a.NY, a.ND
        self.CX, self.CY, self.CD = a.CX, a.CY, a.CD
        self.VW, self.VH = a.VW, a.VH
        self._cvx, self._cvy = _cover_tables(a)  # numpy, for multi_event_fraction
        self.K = _consts(a, dt, fl)
        self.state = _initial_state(a, fl)
        self.next_bin = 0

    def multi_event_fraction(self, t, x, y) -> dict:
        """Share of non-empty bins that hold more than one event at some simple position (this dt)."""
        t = np.asarray(t, dtype=np.int64).reshape(-1)
        x = np.asarray(x, dtype=np.int64).reshape(-1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        bins = t // self.dt
        ci = self._cvx[x]
        cj = self._cvy[y]
        ok = (ci[:, :, None] >= 0) & (cj[:, None, :] >= 0)
        b = np.broadcast_to(bins[:, None, None], ok.shape)[ok]
        i = np.broadcast_to(ci[:, :, None], ok.shape)[ok].astype(np.int64)
        j = np.broadcast_to(cj[:, None, :], ok.shape)[ok].astype(np.int64)
        key = (b * self.NX + i) * self.NY + j
        uniq, cnt = np.unique(key, return_counts=True)
        multi = np.unique(uniq[cnt > 1] // (self.NX * self.NY))
        nonempty = np.unique(bins)
        frac = len(multi) / len(nonempty) if len(nonempty) else 0.0
        return {"fraction": frac, "multi_bins": int(len(multi)), "nonempty_bins": int(len(nonempty))}

    def run(self, t, x, y, p, *, record: bool = False) -> dict:
        """Apply events (time-ordered microseconds, pixels, polarity). Returns per-cell counts.

        counts_s: simple spike counts, flat index (i * NY + j) * ND + k, as stage A orders cells.
        counts_c: complex spike counts, flat index (cx * CY + cy) * CD + cz.
        missed_s, missed_c: positions whose crossing the cascade rounds did not reach (must be 0).
        record=True also returns spike_times_s / spike_times_c: per-cell lists of spike times (us).
        """
        t = np.asarray(t, dtype=np.int64).reshape(-1)
        x = np.asarray(x, dtype=np.int64).reshape(-1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        p = (np.asarray(p, dtype=np.int64).reshape(-1) != 0).astype(np.int64)
        n = t.shape[0]
        if not (x.shape[0] == y.shape[0] == p.shape[0] == n):
            raise ValueError("t, x, y, p must have the same length")
        n_s, n_c = self.NX * self.NY * self.ND, self.CX * self.CY * self.CD
        if n == 0:
            out = {"counts_s": np.zeros(n_s, np.int64), "counts_c": np.zeros(n_c, np.int64),
                   "missed_s": 0, "missed_c": 0}
            if record:
                out["spike_times_s"], out["spike_times_c"] = [[] for _ in range(n_s)], [[] for _ in range(n_c)]
            return out
        if np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("events must be non-negative and time-ordered")
        if np.any((x < 0) | (x >= self.VW) | (y < 0) | (y >= self.VH)):
            raise ValueError(f"event pixel outside {self.VW}x{self.VH}")
        bins = t // self.dt
        first = self.next_bin
        if int(bins[0]) < first:
            raise ValueError(f"events start in bin {int(bins[0])}, before the bin clock {first}")
        last = int(bins[-1])
        nb = last - first + 1
        cnt = np.bincount(bins - first, minlength=nb).astype(np.int32)
        starts = (np.cumsum(cnt) - cnt).astype(np.int32)
        M = max(int(cnt.max()), 1)
        pad = np.zeros(M, dtype=np.int64)
        ex_pad = np.concatenate([x, pad]).astype(np.int32)
        ey_pad = np.concatenate([y, pad]).astype(np.int32)
        ep_pad = np.concatenate([p, pad]).astype(np.int32)
        T_arr = (np.arange(first, last + 1, dtype=np.int64) * self.dt).astype(np.float32)
        lane = jnp.arange(M, dtype=jnp.int32)
        st, (s_s, s_c, m_s, m_c) = _scan(
            self.state, self.K,
            (jnp.asarray(T_arr), jnp.asarray(starts), jnp.asarray(cnt)),
            lane, jnp.asarray(ex_pad), jnp.asarray(ey_pad), jnp.asarray(ep_pad),
            fl=self.flags,
        )
        self.state = st
        self.next_bin = last + 1
        raster_s = np.asarray(s_s).reshape(nb, n_s)
        raster_c = np.asarray(s_c).reshape(nb, n_c)
        out = {"counts_s": raster_s.sum(axis=0).astype(np.int64),
               "counts_c": raster_c.sum(axis=0).astype(np.int64),
               "missed_s": int(np.asarray(m_s).sum()), "missed_c": int(np.asarray(m_c).sum())}
        if record:
            out["spike_times_s"] = _spike_times(raster_s, first, self.dt)
            out["spike_times_c"] = _spike_times(raster_c, first, self.dt)
        return out
