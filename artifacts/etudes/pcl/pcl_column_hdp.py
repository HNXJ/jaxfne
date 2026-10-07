"""PCL gate K1: the C1 column on the jaxfne registrable HDP kernel (Izhikevich, rule ``pcl_stdp``).

800 neurons: 512 input relays (one per pixel x polarity, driven by 1 ms current
pulses at their events), 256 simple and 32 complex cells. Connection types and
paper parameters as ``pcl_column.py``; no weight sharing (one weight per edge).
Weights in native units: 1 mV <-> s (calibrated as in ``pcl_fig2_hdp.py`` at
this dt); inputs onto complex cells carry an extra 10/3 (paper threshold 3 mV
vs 10 mV for simple cells). Deviations: README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_column as C  # noqa: E402
import pcl_fig2_hdp as K  # noqa: E402
import pcl_hdp_rule as R  # noqa: E402
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402
from jaxfne.emitters import EdgeList  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

DT = C.DT  # 0.5 ms, shared with the stimulus generator
N_IN, S0, C0 = C.NIN, C.NIN, C.NIN + C.NS
N = C.NIN + C.NS + C.NC
TYPES = ("s_exc", "s_loc", "s_dist", "s_td", "c_exc", "c_loc")
EXC = (0, 4)
N_STEPS = int(C.STIM["seq_ms"] / DT)
V_FLOOR = -85.0  # mV, 20 below rest (PCL's Vmin is 20 mV below its reset scale); blocks the Izhikevich blow-up under strong inhibition
R.register()


def build_edges():
    pre, post, typ = [], [], []

    def add(p, q, t):
        pre.append(p), post.append(q), typ.append(t)

    for n in range(C.NS):
        pos, f = divmod(n, C.FS)
        for i in C.IN_IDX[pos]:
            add(int(i), S0 + n, 0)
        for g in range(C.FS):
            if g != f:
                add(S0 + pos * C.FS + g, S0 + n, 1)
        for m in np.nonzero(C.DIST_M[n])[0]:
            add(S0 + int(m), S0 + n, 2)
        for c in np.nonzero(C.TD_M[n])[0]:
            add(C0 + int(c), S0 + n, 3)
    for c in range(C.NC):
        cp, h = divmod(c, C.FC)
        for i in C.C_IDX[cp]:
            add(S0 + int(i), C0 + c, 4)
        for h2 in range(C.FC):
            if h2 != h:
                add(C0 + cp * C.FC + h2, C0 + c, 5)
    return np.array(pre), np.array(post), np.array(typ)


PRE, POST, TYP = build_edges()
GROUP = POST * len(TYPES) + TYP


def type_scale(s):
    return np.array([s, s, s, s, s * 10 / 3, s * 10 / 3])[TYP]


def init_weights(rng, s, types):
    """Random magnitudes normalized per (post, type) group to lambda = mean * count (native units)."""
    w = rng.uniform(0.5, 1.5, len(PRE))
    mean = np.array([C.CONN[t][3] for t in TYPES])[TYP] * type_scale(s)
    counts = np.bincount(GROUP, minlength=N * len(TYPES))[GROUP]
    sums = np.bincount(GROUP, weights=w, minlength=N * len(TYPES))[GROUP]
    w = w * mean * counts / sums
    return np.where(np.isin(TYP, types), w, 0.0)


def rule_params(s):
    conn = np.array([C.CONN[t] for t in TYPES])  # w_max, eta, f, mean
    counts = np.bincount(GROUP, minlength=N * len(TYPES))[GROUP]
    tau = np.where(TYP >= 4, C.COMPLEX["tau"], C.SIMPLE["tau"])
    return R.rule_params(tau, conn[TYP, 0] * type_scale(s), conn[TYP, 1], conn[TYP, 2],
                         conn[TYP, 3] * type_scale(s) * counts, np.ones(len(PRE)), GROUP, N * len(TYPES))


def make_runner(s, amp, plastic_types, full=False):
    """Jitted sequence runner; ``full`` returns (v, spikes, sources) traces instead of (w, counts)."""
    params = K.izh(N)
    edges = EdgeList(pre=jnp.asarray(PRE, jnp.int32), post=jnp.asarray(POST, jnp.int32),
                     weight=jnp.where(jnp.isin(jnp.asarray(TYP), jnp.asarray(EXC)), 1.0, -1.0).astype(jnp.float32),
                     receptor_index=jnp.where(jnp.isin(jnp.asarray(TYP), jnp.asarray(EXC)), 0, 1).astype(jnp.int32),
                     tau_ms=jnp.full(len(PRE), K.TAU_INH, jnp.float32), source_calibration_status="x")
    rp = rule_params(s)
    mask = np.isin(TYP, plastic_types).astype(np.float32)
    sign = np.where(np.isin(TYP, EXC), 1.0, -1.0).astype(np.float32)

    @jax.jit
    def run(w_mag, key, ori, direction):
        k0, k1, k2 = jax.random.split(key, 3)
        c0 = -direction * 13.0 + jax.random.uniform(k0, minval=-1.0, maxval=1.0)

        def stim(inside, xs):
            t, k = xs
            x, inside = C.stimulus_step(k, inside, t, ori, direction, c0)
            return inside, x > 0

        _, ev = jax.lax.scan(stim, jnp.zeros(C.G * C.G, bool),
                             (jnp.arange(N_STEPS) * DT, jax.random.split(k1, N_STEPS)))
        pulse = ev | jnp.concatenate([jnp.zeros((1, N_IN), bool), ev[:-1]])  # 1 ms = 2 steps
        sched = jnp.zeros((N_STEPS, N), jnp.float32).at[:, :N_IN].set(amp * pulse)
        st = {"v": jnp.full(N, -65.0), "u": jnp.full(N, -13.0), "prev_spikes": jnp.zeros(N),
              "syn_state": jnp.zeros(len(PRE)), "w_final": jnp.asarray(sign) * w_mag,
              "aux_final": jnp.zeros((len(PRE), 3), jnp.float32)}
        v, spikes, src, diag = sim(params, edges, N_STEPS, DT, k2, drive_schedule=sched, noise_scale=0.0,
                                   init_state=st, hdp_rule=R.NAME, hdp_rule_params=rp,
                                   record_weight_trace=False, plasticity_mask=mask, v_floor=V_FLOOR)
        if full:
            return v, spikes, src
        return jnp.abs(diag["w_final"]), spikes.sum(0)

    return run


def block(runner, w, key, n_seq, labels=None):
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    oris = labels if labels is not None else rng.integers(0, C.N_ORI, n_seq)
    dirs = rng.choice([-1.0, 1.0], len(oris))
    out = []
    for k, o, d in zip(jax.random.split(key, len(oris)), oris, dirs):
        w, cnt = runner(w, k, int(o), float(d))
        out.append(np.asarray(cnt))
    out = np.array(out)
    return w, out[:, S0:C0], out[:, C0:], np.asarray(oris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-exc", type=int, default=600)
    ap.add_argument("--n-inh", type=int, default=300)
    ap.add_argument("--n-test-per-ori", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    K.DT = DT
    amp, w_cancel = K.calibrate()
    s = w_cancel / 5.3
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    k_p1, k_p2, k_test, k_rm = jax.random.split(key, 4)
    labels = np.repeat(np.arange(C.N_ORI), a.n_test_per_ori)
    frozen = make_runner(s, amp, ())
    w0 = init_weights(rng, s, (0, 1, 4, 5))
    _, s_untr, _, _ = block(frozen, jnp.asarray(w0, jnp.float32), k_test, 0, labels)
    w1, *_ = block(make_runner(s, amp, (0, 1, 4, 5)), jnp.asarray(w0, jnp.float32), k_p1, a.n_exc)
    w1 = np.asarray(w1)
    w_inh = init_weights(rng, s, (2, 3))
    w1b = np.where(np.isin(TYP, (2, 3)), w_inh, w1)
    w2, *_ = block(make_runner(s, amp, (2, 3)), jnp.asarray(w1b, jnp.float32), k_p2, a.n_inh)
    w2 = np.asarray(w2)
    w2_off = np.where(np.isin(TYP, (2, 3)), 0.0, w2)
    _, s_a, c_a, _ = block(frozen, jnp.asarray(w2, jnp.float32), k_test, 0, labels)
    _, s_b, c_b, _ = block(frozen, jnp.asarray(w2_off, jnp.float32), k_test, 0, labels)
    q = s_a.sum() / max(s_b.sum(), 1)
    s_c = np.random.default_rng(a.seed + 1).binomial(s_b.astype(int), q)  # post-hoc thinning (README)
    osi_t, rate_t = C.osi(s_b, labels)
    osi_u, rate_u = C.osi(s_untr, labels)
    resp_t, resp_u = rate_t >= 0.5, rate_u >= 0.5
    res = dict(
        params=dict(seed=a.seed, n_exc=a.n_exc, n_inh=a.n_inh, n_test_per_ori=a.n_test_per_ori, dt_ms=DT,
                    pulse_amp=amp, w_cancel=w_cancel, scale=s, n_edges=int(len(PRE)), stim=C.STIM),
        wall_s=round(time.time() - t0, 1),
        osi_trained_median=float(np.median(osi_t[resp_t])) if resp_t.any() else None, n_resp_trained=int(resp_t.sum()),
        osi_untrained_median=float(np.median(osi_u[resp_u])) if resp_u.any() else None, n_resp_untrained=int(resp_u.sum()),
        spikes=dict(pcl=[int(s_a.sum()), int(c_a.sum())], no_inh=[int(s_b.sum()), int(c_b.sum())],
                    random_simple=int(s_c.sum())),
        keep_q=float(q),
        acc=dict(pcl=dict(simple=C.decode(s_a, labels, 0), complex=C.decode(c_a, labels, 0),
                          both=C.decode(np.hstack([s_a, c_a]), labels, 0)),
                 no_inh=dict(simple=C.decode(s_b, labels, 0), complex=C.decode(c_b, labels, 0),
                             both=C.decode(np.hstack([s_b, c_b]), labels, 0)),
                 random=dict(simple=C.decode(s_c, labels, 0))),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".npz"), w=w2, typ=TYP, pre=PRE, post=POST)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
