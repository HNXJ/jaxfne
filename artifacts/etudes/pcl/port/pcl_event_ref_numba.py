"""Numba run path for the event-exact reference (pcl_event_ref.py), inference with learning off.

The kernel repeats the update order of PCLEventRef.process_event line by line: addEvent over the
simple cells, then the simple-spike chain (local, lateral, forward to the complex layer), then the
complex-spike chain (local, top-down), then the periodic rate and threshold update. Arithmetic is
written in the same order as the Python, so spike times match the pure-Python path bit for bit
(checked in test_pcl_trained_load.py). No fastmath.

Not ported: learning on (weight updates and the event buffers they consume). run_numba raises
when the network was built with learning=True.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

import pcl_event_ref as ref

ALPHA = ref.ALPHA
E6 = ref.E6
NEG = ref.NEG_LIMIT

# Layout of the int64 parameter vector G.
G_NX, G_NY, G_ND, G_CX, G_CY, G_CD, G_CSX, G_CSY, G_SX, G_SY = range(10)
G_VW, G_VH, G_NS, G_NC, G_LOCS, G_LATON, G_LOCC, G_TDON, G_ACT = range(10, 19)
# Layout of the float64 parameter vector F.
F_S_VRESET, F_S_TAU_M, F_S_TAU_SRA, F_S_TAU_RP, F_S_DELTA_RP = 0, 1, 2, 3, 4
F_S_DELTA_SRA, F_S_ETA_SR, F_S_TARGET, F_S_MIN = 5, 6, 7, 8
F_C_VRESET, F_C_TAU_M, F_C_TAU_RP, F_C_DELTA_RP, F_C_ETA_SR = 9, 10, 11, 12, 13
F_C_TARGET, F_C_MIN, F_MEAS = 14, 15, 16


@njit(cache=False)
def _refr(tsp, t, delta_rp, tau_rp):
    """Neuron::refractoryPotential (Neuron.cpp:51-58); tsp == 0 is the never-spiked sentinel."""
    if tsp == 0:
        return 0.0
    return delta_rp * math.exp(-(t - tsp) / tau_rp)


@njit(cache=False)
def _exc_s(n, t, w, F, V, A, tl, tsp, tlsp, thr, cnt):
    """SimpleNeuron::membraneUpdate (SimpleNeuron.cpp:141-159). True when the cell spikes."""
    dt = t - tl[n]
    V[n] *= math.exp(-dt / F[F_S_TAU_M])
    A[n] *= math.exp(-dt / F[F_S_TAU_SRA])
    V[n] += w - _refr(tsp[n], t, F[F_S_DELTA_RP], F[F_S_TAU_RP]) - A[n]
    tl[n] = t
    if V[n] < NEG:
        V[n] = NEG
    if V[n] > thr[n]:
        tlsp[n] = tsp[n]
        tsp[n] = t
        cnt[n] += 1
        V[n] = F[F_S_VRESET]
        A[n] += F[F_S_DELTA_SRA]
        return True
    return False


@njit(cache=False)
def _inh_s(n, t, w, F, V, A, tl, tsp):
    """Local, top-down or lateral inhibition of a simple cell: V -= w + R + A (inhibit_simple)."""
    dt = t - tl[n]
    V[n] *= math.exp(-dt / F[F_S_TAU_M])
    A[n] *= math.exp(-dt / F[F_S_TAU_SRA])
    V[n] -= w + _refr(tsp[n], t, F[F_S_DELTA_RP], F[F_S_TAU_RP]) + A[n]
    tl[n] = t
    if V[n] < NEG:
        V[n] = NEG


@njit(cache=False)
def _exc_c(c, t, w, F, V, tl, tsp, tlsp, thr, cnt):
    """ComplexNeuron::membraneUpdate (ComplexNeuron.cpp:53-71). True when the cell spikes."""
    dt = t - tl[c]
    V[c] *= math.exp(-dt / F[F_C_TAU_M])
    V[c] += w - _refr(tsp[c], t, F[F_C_DELTA_RP], F[F_C_TAU_RP])
    tl[c] = t
    if V[c] < NEG:
        V[c] = NEG
    if V[c] > thr[c]:
        tlsp[c] = tsp[c]
        tsp[c] = t
        cnt[c] += 1
        V[c] = F[F_C_VRESET]
        return True
    return False


@njit(cache=False)
def _inh_c(c, t, w, F, V, tl, tsp):
    """Local inhibition of a complex cell: V -= w + R (inhibit_complex)."""
    dt = t - tl[c]
    V[c] *= math.exp(-dt / F[F_C_TAU_M])
    V[c] -= w + _refr(tsp[c], t, F[F_C_DELTA_RP], F[F_C_TAU_RP])
    if V[c] < NEG:
        V[c] = NEG
    tl[c] = t


@njit(cache=False)
def _grow(a, b):
    na = np.empty(2 * a.shape[0], np.int64)
    na[:a.shape[0]] = a
    nb = np.empty(2 * b.shape[0], np.int64)
    nb[:b.shape[0]] = b
    return na, nb


@njit(cache=False)
def _kernel(t_ev, x_ev, y_ev, p_ev, G, F,
            Vs, As, tls, tsp, tlsp, ths, cns, rts,
            Vc, tlc, tsc, tlsc, thc, cnc, rtc,
            w_s, wi_s, w_c, wi_c, LAT, LATM, TDT, TDM,
            cvx, cvxn, cvy, cvyn, s_ox, s_oy, c_ox, c_oy, ccx, ccy, ccn,
            last, out_id, out_t):
    NY, ND, CY, CD = G[G_NY], G[G_ND], G[G_CY], G[G_CD]
    NS, NC = G[G_NS], G[G_NC]
    LOCS, LATON, LOCC, TDON, ACT = G[G_LOCS], G[G_LATON], G[G_LOCC], G[G_TDON], G[G_ACT]
    nout = 0
    for e in range(t_ev.shape[0]):
        t = t_ev[e]
        x = x_ev[e]
        y = y_ev[e]
        p = p_ev[e]
        # SpikingNetwork::addEvent (SpikingNetwork.cpp:41-67): simple positions in ascending order.
        for ii in range(cvxn[x]):
            i = cvx[x, ii]
            xr = x - s_ox[i]
            for jj in range(cvyn[y]):
                j = cvy[y, jj]
                yr = y - s_oy[j]
                base = (i * NY + j) * ND
                for k in range(ND):
                    n = base + k
                    if not _exc_s(n, t, w_s[k, p, xr, yr], F, Vs, As, tls, tsp, tlsp, ths, cns):
                        continue
                    if nout == out_id.shape[0]:
                        out_id, out_t = _grow(out_id, out_t)
                    out_id[nout] = n
                    out_t[nout] = t
                    nout += 1
                    # lateralLocalInhibition (SpikingNetwork.cpp:172-179), other depths at (i, j).
                    if LOCS == 1:
                        for d in range(ND):
                            if d != k:
                                _inh_s(base + d, t, wi_s[d, k], F, Vs, As, tls, tsp)
                    # lateralDynamicInhibition (SpikingNetwork.cpp:157-165), ascending targets.
                    if ACT == 1 and LATON == 1:
                        for m in range(NS):
                            if LATM[n, m]:
                                _inh_s(m, t, LAT[m, n], F, Vs, As, tls, tsp)
                    # addNeuronEvent (SpikingNetwork.cpp:87-107): complex cells in ascending index.
                    for q in range(ccn[i, j]):
                        cx = ccx[i, j, q]
                        cy = ccy[i, j, q]
                        ex = i - c_ox[cx]
                        ey = j - c_oy[cy]
                        cbase = (cx * CY + cy) * CD
                        for cz in range(CD):
                            c = cbase + cz
                            if not _exc_c(c, t, w_c[cz, ex, ey, k], F, Vc, tlc, tsc, tlsc, thc, cnc):
                                continue
                            if nout == out_id.shape[0]:
                                out_id, out_t = _grow(out_id, out_t)
                            out_id[nout] = NS + c
                            out_t[nout] = t
                            nout += 1
                            if LOCC == 1:
                                for cz2 in range(CD):
                                    if cz2 != cz:
                                        _inh_c(cbase + cz2, t, wi_c[cz2, cz], F, Vc, tlc, tsc)
                            # topDownDynamicInhibition (SpikingNetwork.cpp:142-151), ascending targets.
                            if ACT == 1 and TDON == 1:
                                for s in range(NS):
                                    if TDM[c, s]:
                                        _inh_s(s, t, TDT[c, s], F, Vs, As, tls, tsp)
        # NetworkHandle::updateNeurons (NetworkHandle.cpp:137) and Neuron::updateState and
        # thresholdAdaptation (Neuron.cpp:357-387), run after each event when the interval has passed.
        dt = t - last
        if float(dt) > F[F_MEAS]:
            for n in range(NS):
                sps = float(cns[n]) * (E6 / float(dt))
                rts[n] = ALPHA * sps + (1.0 - ALPHA) * rts[n]
                cns[n] = 0
                if rts[n] > F[F_S_TARGET]:
                    ths[n] += F[F_S_ETA_SR] * (1 - math.exp(F[F_S_TARGET] - rts[n]))
                else:
                    ths[n] -= F[F_S_ETA_SR] * (1 - math.exp(rts[n] - F[F_S_TARGET]))
                if ths[n] < F[F_S_MIN]:
                    ths[n] = F[F_S_MIN]
            for c in range(NC):
                sps = float(cnc[c]) * (E6 / float(dt))
                rtc[c] = ALPHA * sps + (1.0 - ALPHA) * rtc[c]
                cnc[c] = 0
                if rtc[c] > F[F_C_TARGET]:
                    thc[c] += F[F_C_ETA_SR] * (1 - math.exp(F[F_C_TARGET] - rtc[c]))
                else:
                    thc[c] -= F[F_C_ETA_SR] * (1 - math.exp(rtc[c] - F[F_C_TARGET]))
                if thc[c] < F[F_C_MIN]:
                    thc[c] = F[F_C_MIN]
            last = t
    return nout, last, out_id, out_t


def _pad(rows):
    """Ragged lists to a padded int64 matrix and a count vector."""
    width = max(1, max((len(r) for r in rows), default=0))
    out = np.zeros((len(rows), width), dtype=np.int64)
    cnt = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
        cnt[i] = len(r)
    return out, cnt


def run_numba(net: "ref.PCLEventRef", t, x, y, p) -> None:
    """Feed an event stream through the numba kernel; the network's state and spikes are updated
    exactly as net.run(t, x, y, p) would update them. Learning must be off."""
    if net.learning:
        raise NotImplementedError("learning-on numba path is not ported")
    t_ev = np.ascontiguousarray(np.asarray(t, dtype=np.int64))
    x_ev = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
    y_ev = np.ascontiguousarray(np.asarray(y, dtype=np.int64))
    p_ev = np.ascontiguousarray((np.asarray(p) != 0).astype(np.int64))
    if t_ev.size and (np.any(x_ev < 0) or np.any(x_ev >= net.VW) or np.any(y_ev < 0) or np.any(y_ev >= net.VH)):
        raise ValueError(f"event pixel outside {net.VW}x{net.VH}")

    G = np.array([net.NX, net.NY, net.ND, net.CX, net.CY, net.CD, net.CSX, net.CSY, net.SX, net.SY,
                  net.VW, net.VH, net.NS, net.NC, int(net.local_s), int(net.lat_on), int(net.local_c),
                  int(net.td_on), int(net.activation)], dtype=np.int64)
    sc, cc = net.scfg, net.ccfg
    F = np.array([sc.VRESET, sc.TAU_M, sc.TAU_SRA, sc.TAU_RP, sc.DELTA_RP, sc.DELTA_SRA, sc.ETA_SR,
                  sc.TARGET_SPIKE_RATE, sc.MIN_THRESH,
                  cc.VRESET, cc.TAU_M, cc.TAU_RP, cc.DELTA_RP, cc.ETA_SR, cc.TARGET_SPIKE_RATE,
                  cc.MIN_THRESH, net.measurement_interval], dtype=np.float64)

    Vs = np.array([c.V for c in net.s_cells], dtype=np.float64)
    As = np.array([c.A for c in net.s_cells], dtype=np.float64)
    tls = np.array([c.tlast for c in net.s_cells], dtype=np.int64)
    tsp = np.array([c.t_spk for c in net.s_cells], dtype=np.int64)
    tlsp = np.array([c.t_last_spk for c in net.s_cells], dtype=np.int64)
    ths = np.array([c.thr for c in net.s_cells], dtype=np.float64)
    cns = np.array([c.count for c in net.s_cells], dtype=np.int64)
    rts = np.array([c.rate for c in net.s_cells], dtype=np.float64)
    Vc = np.array([c.V for c in net.c_cells], dtype=np.float64)
    tlc = np.array([c.tlast for c in net.c_cells], dtype=np.int64)
    tsc = np.array([c.t_spk for c in net.c_cells], dtype=np.int64)
    tlsc = np.array([c.t_last_spk for c in net.c_cells], dtype=np.int64)
    thc = np.array([c.thr for c in net.c_cells], dtype=np.float64)
    cnc = np.array([c.count for c in net.c_cells], dtype=np.int64)
    rtc = np.array([c.rate for c in net.c_cells], dtype=np.float64)

    if net.lat_on:
        LAT = np.ascontiguousarray(net.LAT, dtype=np.float64)
        LATM = np.ascontiguousarray(net.lat_mask, dtype=np.bool_)
    else:
        LAT, LATM = np.zeros((1, 1)), np.zeros((1, 1), dtype=np.bool_)
    if net.td_on:
        TDT = np.ascontiguousarray(net.TD.T, dtype=np.float64)
        TDM = np.ascontiguousarray(net.td_mask.T, dtype=np.bool_)
    else:
        TDT, TDM = np.zeros((1, 1)), np.zeros((1, 1), dtype=np.bool_)

    cvx, cvxn = _pad(net.cover_x)
    cvy, cvyn = _pad(net.cover_y)
    cells = [net.c_cover[(i, j)] for i in range(net.NX) for j in range(net.NY)]
    ccx, ccn = _pad([[cx for cx, _ in lst] for lst in cells])
    ccy, _ = _pad([[cy for _, cy in lst] for lst in cells])
    ccx = ccx.reshape(net.NX, net.NY, -1)
    ccy = ccy.reshape(net.NX, net.NY, -1)
    ccn = ccn.reshape(net.NX, net.NY)
    s_ox = np.asarray(net.s_ox, dtype=np.int64)
    s_oy = np.asarray(net.s_oy, dtype=np.int64)
    c_ox = np.asarray(net.c_ox, dtype=np.int64)
    c_oy = np.asarray(net.c_oy, dtype=np.int64)

    cap = max(1024, int(t_ev.size) * 4)
    out_id = np.zeros(cap, dtype=np.int64)
    out_t = np.zeros(cap, dtype=np.int64)
    w_s = np.ascontiguousarray(net.w_s, dtype=np.float64)
    wi_s = np.ascontiguousarray(net.wi_s, dtype=np.float64)
    w_c = np.ascontiguousarray(net.w_c, dtype=np.float64)
    wi_c = np.ascontiguousarray(net.wi_c, dtype=np.float64)

    nout, last, out_id, out_t = _kernel(
        t_ev, x_ev, y_ev, p_ev, G, F,
        Vs, As, tls, tsp, tlsp, ths, cns, rts,
        Vc, tlc, tsc, tlsc, thc, cnc, rtc,
        w_s, wi_s, w_c, wi_c, LAT, LATM, TDT, TDM,
        cvx, cvxn, cvy, cvyn, s_ox, s_oy, c_ox, c_oy, ccx, ccy, ccn,
        int(net._last_update), out_id, out_t)

    for n, cell in enumerate(net.s_cells):
        cell.V, cell.A = float(Vs[n]), float(As[n])
        cell.tlast, cell.t_spk, cell.t_last_spk = int(tls[n]), int(tsp[n]), int(tlsp[n])
        cell.thr, cell.count, cell.rate = float(ths[n]), int(cns[n]), float(rts[n])
    for c, cell in enumerate(net.c_cells):
        cell.V = float(Vc[c])
        cell.tlast, cell.t_spk, cell.t_last_spk = int(tlc[c]), int(tsc[c]), int(tlsc[c])
        cell.thr, cell.count, cell.rate = float(thc[c]), int(cnc[c]), float(rtc[c])
    ids = out_id[:nout]
    ts = out_t[:nout]
    order = np.argsort(ids, kind="stable")  # stable: each neuron keeps time order
    ids, ts = ids[order], ts[order]
    cells_all = net.s_cells + net.c_cells
    bounds = np.searchsorted(ids, np.arange(len(cells_all) + 1))
    for g, cell in enumerate(cells_all):
        if bounds[g + 1] > bounds[g]:
            cell.spikes.extend(ts[bounds[g]:bounds[g + 1]].tolist())
    net._last_update = int(last)


def warm_up() -> None:
    """Compile the kernel on a zero-event network so a timed call measures the run only."""
    net = ref.PCLEventRef(seed=0, learning=False)
    run_numba(net, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
              np.zeros(0, np.int64))
