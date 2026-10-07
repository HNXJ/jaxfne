"""PCL gate C1: small two-layer column (simple + complex cells) on synthetic event streams.

Clock-driven (dt = 0.5 ms) version of the PCL model of N'dri et al. (Nat Commun 2025):
LIF with per-event relative-refractory subtraction, causal STDP on every connection
(LTP over presynaptic events since the last postsynaptic spike, LTD deferred to the
next postsynaptic spike), soft bounds, per-type L1 normalization of incoming
weights, convolutional weight sharing for excitatory and local inhibitory kernels.
Deviations from the paper: README.md.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DT = 0.5  # ms
G = 16  # input grid (G x G x 2 polarities)
NP_S, RF, STRIDE, FS = 4, 7, 3, 16  # simple grid 4x4, 7x7 receptive field, 16 features
NP_C, CRF, FC = 2, 3, 8  # complex grid 2x2, 3x3 simple positions, 8 features
NS, NC, NIN = NP_S * NP_S * FS, NP_C * NP_C * FC, G * G * 2
N_ORI = 8

# Paper Table 2. Mean incoming weight per synapse is kept at the paper's value
# (lambda_paper / n_paper); lambda here = mean * our synapse count (README).
SIMPLE = dict(v_reset=-10.0, v_th=10.0, v_min=-20.0, tau_m=18.0, tau_rp=5.0, eta_rp=10.0, tau=7.0)
COMPLEX = dict(v_reset=-10.0, v_th=3.0, v_min=-20.0, tau_m=50.0, tau_rp=5.0, eta_rp=5.0, tau=40.0)
CONN = {  # w_max, eta_LTP (= -eta_LTD), soft-bound factor, mean weight per synapse (lambda/n in paper)
    "s_exc": (3.0, 0.000204, 0.33, 50.0 / 200),
    "s_loc": (50.0, 0.816, 0.02, 1500.0 / 63),
    "s_dist": (50.0, 2.652, 0.02, 6500.0 / 960),
    "s_td": (50.0, 1.428, 0.02, 3500.0 / 512),
    "c_exc": (4.0, 0.08, 0.25, 1000.0 / 1024),
    "c_loc": (25.0, 0.048, 0.04, 600.0 / 31),
}
STIM = dict(kind="bar", width=2.0, period=6.0, speed=0.13, events=3.0, noise_hz=2.0, seq_ms=200.0)


def build_indices():
    pos_s = [(py, px) for py in range(NP_S) for px in range(NP_S)]
    in_idx = np.array([[((STRIDE * py + dy) * G + STRIDE * px + dx) * 2 + p
                        for dy in range(RF) for dx in range(RF) for p in range(2)] for py, px in pos_s])
    pos_c = [(cy, cx) for cy in range(NP_C) for cx in range(NP_C)]
    covered = [[(cy + i) * NP_S + (cx + j) for i in range(CRF) for j in range(CRF)] for cy, cx in pos_c]
    c_idx = np.array([[p * FS + f for p in cov for f in range(FS)] for cov in covered])
    pos_of = np.repeat(np.arange(NP_S * NP_S), FS)
    dist_m = (pos_of[:, None] != pos_of[None, :]).astype(np.float32)
    td_m = np.zeros((NS, NC), np.float32)
    for c, cov in enumerate(covered):
        for p in cov:
            td_m[p * FS:(p + 1) * FS, c * FC:(c + 1) * FC] = 1.0
    return in_idx, c_idx, dist_m, td_m


IN_IDX, C_IDX, DIST_M, TD_M = build_indices()
LOC_S_M = 1.0 - np.eye(FS, dtype=np.float32)
LOC_C_M = 1.0 - np.eye(FC, dtype=np.float32)
MASKS = {"s_exc": np.ones((FS, RF * RF * 2), np.float32), "s_loc": LOC_S_M, "s_dist": DIST_M,
         "s_td": TD_M, "c_exc": np.ones((FC, CRF * CRF * FS), np.float32), "c_loc": LOC_C_M}


def normalize(W, mask, name):
    lam = CONN[name][3] * mask.sum(-1, keepdims=True)
    W = W * mask
    return W * lam / jnp.maximum(W.sum(-1, keepdims=True), 1e-12)


def init_weights(key):
    ks = jax.random.split(key, len(MASKS))
    return {n: normalize(jax.random.uniform(k, m.shape, minval=0.5, maxval=1.5), m, n)
            for k, (n, m) in zip(ks, MASKS.items())}


def init_state():
    def z(*s):
        return jnp.zeros(s, jnp.float32)

    return dict(Vs=z(NS), ts_s=jnp.full(NS, -1e9), Vc=z(NC), ts_c=jnp.full(NC, -1e9), ss=z(NS), sc=z(NC),
                inside=jnp.zeros(G * G, bool),
                A_s_exc=z(NS, RF * RF * 2), B_s_exc=z(NS, RF * RF * 2), A_s_loc=z(NS, FS), B_s_loc=z(NS, FS),
                A_s_dist=z(NS, NS), B_s_dist=z(NS, NS), A_s_td=z(NS, NC), B_s_td=z(NS, NC),
                A_c_exc=z(NC, CRF * CRF * FS), B_c_exc=z(NC, CRF * CRF * FS), A_c_loc=z(NC, FC), B_c_loc=z(NC, FC))


def stimulus_step(key, inside_prev, t, ori, direction, c0):
    """Bright bar of orientation ori*pi/8 drifting along its normal; ON/OFF events at edges."""
    th = ori * jnp.pi / N_ORI + jnp.pi / 2  # normal to the bar
    yy, xx = jnp.meshgrid(jnp.arange(G), jnp.arange(G), indexing="ij")
    proj = ((xx - 7.5) * jnp.cos(th) + (yy - 7.5) * jnp.sin(th)).ravel()
    d = proj - (c0 + direction * STIM["speed"] * t)
    if STIM["kind"] == "grating":  # square-wave grating: bright half of each period
        inside = jnp.mod(d, STIM["period"]) < STIM["period"] / 2
    else:
        inside = jnp.abs(d) < STIM["width"] / 2
    k1, k2 = jax.random.split(key)
    on = (inside & ~inside_prev).astype(jnp.float32)
    off = (~inside & inside_prev).astype(jnp.float32)
    lam = STIM["events"] * jnp.stack([on, off], -1).ravel() + STIM["noise_hz"] * DT / 1000.0
    return jax.random.poisson(k1, lam).astype(jnp.float32), inside


def stdp(W, A, B, e, t_prev, spk, t, name, tau):
    """Causal PCL STDP for one connection type. W/A/B: (post, pre); e: presynaptic events (post, pre)."""
    w_max, eta, f, _ = CONN[name]
    A = A * jnp.exp(-DT / tau) + e
    B = B + e * jnp.exp(-(t - t_prev) / tau)[:, None]
    dW = spk[:, None] * ((w_max - W) * f * eta * A - W * f * eta * B)
    keep = 1.0 - spk[:, None]
    return dW, A * keep, B * keep


def lif(V, ts, drive, n_ev, t, p):
    V = V * np.exp(-DT / p["tau_m"]) + drive - n_ev * p["eta_rp"] * jnp.exp(-(t - ts) / p["tau_rp"])
    V = jnp.maximum(V, p["v_min"])
    spk = V >= p["v_th"]
    return jnp.where(spk, p["v_reset"], V), jnp.where(spk, t, ts), spk.astype(jnp.float32)


def step(carry, xs, flags, W_frozen):
    W, S = carry
    t, key, ori, direction, c0, keep_q = xs
    learn_exc, learn_inh, dist_on = flags
    k_stim, k_rm = jax.random.split(key)
    x, inside = stimulus_step(k_stim, S["inside"], t, ori, direction, c0)
    Wd = W if W_frozen is None else W_frozen

    # simple layer
    Xs = x[IN_IDX]  # (pos, 98)
    e_exc = jnp.repeat(Xs, FS, axis=0)  # (NS, 98): each neuron sees its position's input
    ss_prev = S["ss"].reshape(-1, FS)
    e_loc = jnp.repeat(ss_prev, FS, axis=0) * jnp.tile(LOC_S_M, (NP_S * NP_S, 1))  # (NS, FS)
    e_dist = S["ss"][None, :] * DIST_M * dist_on
    e_td = S["sc"][None, :] * TD_M * dist_on
    Ws_exc = jnp.tile(Wd["s_exc"], (NP_S * NP_S, 1))
    Ws_loc = jnp.tile(Wd["s_loc"], (NP_S * NP_S, 1))
    drive = ((Ws_exc * e_exc).sum(1) - (Ws_loc * e_loc).sum(1)
             - (Wd["s_dist"] * e_dist).sum(1) - (Wd["s_td"] * e_td).sum(1))
    n_ev = e_exc.sum(1) + e_loc.sum(1) + e_dist.sum(1) + e_td.sum(1)
    ts_s_prev = S["ts_s"]
    Vs, ts_s, ss = lif(S["Vs"], S["ts_s"], drive, n_ev, t, SIMPLE)
    ss = ss * (jax.random.uniform(k_rm, (NS,)) < keep_q)  # random spike removal control (keep_q = 1: none)

    # complex layer (feedforward from this step's simple spikes)
    Xc = ss[C_IDX]  # (cpos, 144)
    e_cexc = jnp.repeat(Xc, FC, axis=0)
    sc_prev = S["sc"].reshape(-1, FC)
    e_cloc = jnp.repeat(sc_prev, FC, axis=0) * jnp.tile(LOC_C_M, (NP_C * NP_C, 1))
    Wc_exc = jnp.tile(Wd["c_exc"], (NP_C * NP_C, 1))
    Wc_loc = jnp.tile(Wd["c_loc"], (NP_C * NP_C, 1))
    drive_c = (Wc_exc * e_cexc).sum(1) - (Wc_loc * e_cloc).sum(1)
    ts_c_prev = S["ts_c"]
    Vc, ts_c, sc = lif(S["Vc"], S["ts_c"], drive_c, e_cexc.sum(1) + e_cloc.sum(1), t, COMPLEX)

    # plasticity
    S2 = dict(S, Vs=Vs, ts_s=ts_s, Vc=Vc, ts_c=ts_c, ss=ss, sc=sc, inside=inside)
    W2 = dict(W)
    shared = {"s_exc": (e_exc, ss, ts_s_prev, SIMPLE["tau"], NP_S * NP_S, Ws_exc),
              "s_loc": (e_loc, ss, ts_s_prev, SIMPLE["tau"], NP_S * NP_S, Ws_loc),
              "c_exc": (e_cexc, sc, ts_c_prev, COMPLEX["tau"], NP_C * NP_C, Wc_exc),
              "c_loc": (e_cloc, sc, ts_c_prev, COMPLEX["tau"], NP_C * NP_C, Wc_loc)}
    for name, (e, spk, tp, tau, npos, Wfull) in shared.items():
        dW, S2["A_" + name], S2["B_" + name] = stdp(Wfull, S["A_" + name], S["B_" + name], e, tp, spk, t, name, tau)
        dK = dW.reshape(npos, -1, dW.shape[-1]).sum(0)  # weight sharing: every position updates the kernel
        newK = normalize(jnp.maximum(W[name] + dK, 0.0), MASKS[name], name)
        W2[name] = jnp.where(learn_exc, newK, W[name])
    for name, e in (("s_dist", e_dist), ("s_td", e_td)):
        dW, S2["A_" + name], S2["B_" + name] = stdp(W[name], S["A_" + name], S["B_" + name], e, ts_s_prev, ss, t, name, SIMPLE["tau"])
        newW = normalize(jnp.maximum(W[name] + dW, 0.0), MASKS[name], name)
        W2[name] = jnp.where(learn_inh, newW, W[name])
    return (W2, S2), (ss, sc)


@partial(jax.jit, static_argnames=("n_steps",))
def run_sequence(W, key, ori, direction, keep_q, flags, n_steps):
    """One stimulus sequence; returns updated weights and spike counts (simple, complex)."""
    k0, k1 = jax.random.split(key)
    c0 = -direction * 13.0 + jax.random.uniform(k0, minval=-1.0, maxval=1.0)
    ts = jnp.arange(n_steps) * DT
    keys = jax.random.split(k1, n_steps)
    xs = (ts, keys, jnp.full(n_steps, ori), jnp.full(n_steps, direction), jnp.full(n_steps, c0), jnp.full(n_steps, keep_q))
    (W, _), (ss, sc) = jax.lax.scan(lambda c, x: step(c, x, flags, None), (W, init_state()), xs)
    return W, ss.sum(0), sc.sum(0)


def run_block(W, key, n_seq, flags, keep_q=1.0, labels=None):
    n_steps = int(STIM["seq_ms"] / DT)
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    oris = labels if labels is not None else rng.integers(0, N_ORI, n_seq)
    dirs = rng.choice([-1.0, 1.0], len(oris))
    keys = jax.random.split(key, len(oris))
    cs, cc = [], []
    for k, o, d in zip(keys, oris, dirs):
        W, s, c = run_sequence(W, k, int(o), float(d), float(keep_q), flags, n_steps)
        cs.append(np.asarray(s)), cc.append(np.asarray(c))
    return W, np.array(cs), np.array(cc), np.asarray(oris)


def osi(counts, oris):
    """Orientation selectivity |sum r e^{2i theta}| / sum r per cell, from mean counts per orientation."""
    r = np.stack([counts[oris == k].mean(0) for k in range(N_ORI)])  # (ori, cells)
    th = np.arange(N_ORI) * np.pi / N_ORI
    return np.abs((r * np.exp(2j * th)[:, None]).sum(0)) / np.maximum(r.sum(0), 1e-12), r.sum(0) / N_ORI


def decode(features, oris, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, np.log1p(features), oris, cv=5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-exc", type=int, default=600, help="phase-1 sequences (excitatory + local inhibition)")
    ap.add_argument("--n-inh", type=int, default=300, help="phase-2 sequences (distant lateral + top-down)")
    ap.add_argument("--n-test-per-ori", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stim", choices=("bar", "grating"), default="bar")
    a = ap.parse_args()
    STIM["kind"] = a.stim
    t0 = time.time()
    key = jax.random.PRNGKey(a.seed)
    k_init, k_untr, k_p1, k_p2, k_test = jax.random.split(key, 5)
    W0 = init_weights(k_init)
    labels = np.repeat(np.arange(N_ORI), a.n_test_per_ori)
    off = (False, False, 0.0)
    _, s_untr, _, _ = run_block(W0, k_test, 0, off, labels=labels)
    W1, *_ = run_block(W0, k_p1, a.n_exc, (True, False, 0.0))
    W2, *_ = run_block(W1, k_p2, a.n_inh, (False, True, 1.0))
    _, s_a, c_a, _ = run_block(W2, k_test, 0, (False, False, 1.0), labels=labels)  # PCL
    _, s_b, c_b, _ = run_block(W2, k_test, 0, off, labels=labels)  # no distant / top-down inhibition
    q = s_a.sum() / max(s_b.sum(), 1)
    _, s_c, c_c, _ = run_block(W2, k_test, 0, off, keep_q=q, labels=labels)  # random removal, matched
    osi_t, rate_t = osi(s_b, labels)
    osi_u, rate_u = osi(s_untr, labels)
    resp_t, resp_u = rate_t >= 0.5, rate_u >= 0.5
    res = dict(
        params=dict(seed=a.seed, n_exc=a.n_exc, n_inh=a.n_inh, n_test_per_ori=a.n_test_per_ori, stim=STIM),
        wall_s=round(time.time() - t0, 1),
        osi_trained_median=float(np.median(osi_t[resp_t])) if resp_t.any() else None, n_resp_trained=int(resp_t.sum()),
        osi_untrained_median=float(np.median(osi_u[resp_u])) if resp_u.any() else None, n_resp_untrained=int(resp_u.sum()),
        spikes=dict(pcl=[int(s_a.sum()), int(c_a.sum())], no_inh=[int(s_b.sum()), int(c_b.sum())],
                    random=[int(s_c.sum()), int(c_c.sum())]),
        keep_q=float(q),
        acc={k: {"simple": decode(s, labels, a.seed), "complex": decode(c, labels, a.seed),
                 "both": decode(np.hstack([s, c]), labels, a.seed)}
             for k, (s, c) in {"pcl": (s_a, c_a), "no_inh": (s_b, c_b), "random": (s_c, c_c)}.items()},
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".npz"), **{k: np.asarray(v) for k, v in W2.items()})
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
