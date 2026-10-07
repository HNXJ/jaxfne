"""Stage-1 balance etude: modulated three-factor learning under an inner homeostatic loop.

Standalone JAX probe; it does not run the jaxfne kernels. Protocol and pass
criteria: ``README.md`` next to this file.

Loops:
  inner (RBD): per-neuron threshold H_i tracks its rate to a target r*_i, and
      each neuron's plastic excitatory input sum is held at its initial value.
  outer: modulator M(t) = -d|theta|/dt (failure = -1 pulse), centred by a
      slow running mean; dW = eta * M * Elig * dt on plastic synapses.
Task: unstable 1-D object, d theta = (a theta + b u) dt + sigma dW_t; the
  network must learn which motor pool to fire for each sign of theta.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DT = 1e-3  # s
N_S, N_E, N_I, N_M = 20, 48, 12, 20
N = N_S + N_E + N_I + N_M
S = np.arange(0, 20)
E = np.arange(20, 68)
I = np.arange(68, 80)
MP = np.arange(80, 90)  # pushes theta up
MM = np.arange(90, 100)  # pushes theta down
PREF = np.linspace(-1.0, 1.0, N_S)  # sensory preferred theta

P = dict(
    tau_m=0.02, tau_s=0.005, sigma_v=0.3, tau_pre=0.02, tau_e=0.3, tau_r=0.05,
    tau_rho=2.0, tau_H=5.0, a=1.0, b=1.0, u_max=2.5, u_scale=10.0,
    sigma_th=0.3, th_fail=1.0, th_band=0.25, th_reset=0.3,
    tau_err=0.02, tau_mbar=20.0, sens_peak=60.0, sens_base=2.0, sens_width=0.2,
)
RSTAR = np.zeros(N, np.float32)
RSTAR[E], RSTAR[I], RSTAR[MP], RSTAR[MM] = 5.0, 10.0, 10.0, 10.0
NONSENS = RSTAR > 0
PLASTIC_SCOPE = "all"  # "all": excitatory onto E and motor; "sensory_motor": sensory->motor only


def init_network(seed: int, wired: bool = False):
    """Random sparse E/I network; ``wired`` adds a correct sensory->motor map (positive control)."""
    rng = np.random.default_rng(seed)
    C = np.zeros((N, N), bool)
    W = np.zeros((N, N))
    post_all = np.concatenate([E, I, MP, MM])
    post_em = np.concatenate([E, MP, MM])
    for pre, post, p, w0 in ((S, post_all, 0.5, 1.5), (E, post_all, 0.2, 1.0), (I, post_em, 0.5, -2.0)):
        c = rng.random((len(post), len(pre))) < p
        C[np.ix_(post, pre)] = c
        W[np.ix_(post, pre)] = c * w0 * rng.uniform(0.5, 1.5, c.shape)
    np.fill_diagonal(C, False)
    np.fill_diagonal(W, 0.0)
    if wired:
        up, down = S[PREF < 0], S[PREF > 0]
        for post, pre_on, pre_off in ((MP, up, down), (MM, down, up)):
            C[np.ix_(post, pre_on)] = True
            W[np.ix_(post, pre_on)] = 3.0
            C[np.ix_(post, pre_off)] = False
            W[np.ix_(post, pre_off)] = 0.0
    plastic = C.copy()
    plastic[:, I] = False
    plastic[I, :] = False
    plastic[S, :] = False
    if PLASTIC_SCOPE == "sensory_motor":
        keep = np.zeros_like(plastic)
        keep[np.ix_(np.concatenate([MP, MM]), S)] = True
        plastic &= keep
    return W.astype(np.float32), plastic


def _step(carry, _, eta, perm, plastic, S0):
    (key, v, s, xpre, r, rho, H, W, Elig, theta, err_f, mbar) = carry
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    B = v.shape[0]
    rates = P["sens_base"] + P["sens_peak"] * jnp.exp(
        -((theta[:, None] - PREF[None, :]) ** 2) / (2 * P["sens_width"] ** 2))
    spk_s = jax.random.uniform(k1, (B, N_S)) < rates * DT
    I_syn = jnp.einsum("bij,bj->bi", W, s)
    v = v + DT / P["tau_m"] * (-v + I_syn) + P["sigma_v"] * np.sqrt(DT / P["tau_m"]) * jax.random.normal(k2, v.shape)
    spk = (v > H) & NONSENS
    spk = spk.at[:, S].set(spk_s)
    v = jnp.where(spk, 0.0, v)
    f = spk.astype(v.dtype)
    xpre_old = xpre
    s = s * (1 - DT / P["tau_s"]) + f
    xpre = xpre * (1 - DT / P["tau_pre"]) + f
    r = r * (1 - DT / P["tau_r"]) + f / P["tau_r"]
    rho = rho * (1 - DT / P["tau_rho"]) + f / P["tau_rho"]
    H = jnp.where(NONSENS, jnp.clip(H + DT / P["tau_H"] * (rho / np.where(NONSENS, RSTAR, 1.0) - 1.0), 0.1, 10.0), H)

    u = P["u_max"] * jnp.tanh((r[:, MP].mean(1) - r[:, MM].mean(1)) / P["u_scale"])
    theta_new = theta + DT * (P["a"] * theta + P["b"] * u) + P["sigma_th"] * np.sqrt(DT) * jax.random.normal(k3, theta.shape)
    fail = jnp.abs(theta_new) > P["th_fail"]
    theta_new = jnp.where(fail, jax.random.uniform(k4, theta.shape, minval=-P["th_reset"], maxval=P["th_reset"]), theta_new)
    e = jnp.abs(theta_new)
    err_new = err_f + DT / P["tau_err"] * (e - err_f)
    M_raw = jnp.where(fail, -1.0 / DT, -(err_new - err_f) / DT)
    err_new = jnp.where(fail, e, err_new)
    mbar = mbar + DT / P["tau_mbar"] * (M_raw - mbar)
    M = (M_raw - mbar)[perm]

    Elig = Elig * (1 - DT / P["tau_e"]) + f[:, :, None] * xpre_old[:, None, :] * plastic
    W_new = jnp.where(plastic, jnp.maximum(W + eta[:, None, None] * M[:, None, None] * Elig * DT, 0.0), W)
    sums = (W_new * plastic).sum(-1)
    W_new = jnp.where(plastic, W_new * (S0 / jnp.maximum(sums, 1e-9))[:, :, None], W_new)
    W = jnp.where((eta > 0)[:, None, None], W_new, W)

    out = (jnp.abs(theta_new) < P["th_band"], fail, f[:, E].mean(1))
    return (key, v, s, xpre, r, rho, H, W, Elig, theta_new, err_new, mbar), out


def _run_phase(carry, eta, perm, plastic, S0, n_bins: int, bin_steps: int = 1000):
    """Run n_bins x bin_steps; returns carry and per-bin (in-band fraction, failures, E rate Hz)."""
    def step(c, x):
        return _step(c, x, eta, perm, plastic, S0)

    def bin_fn(c, _):
        c, (band, fail, rate) = jax.lax.scan(step, c, None, length=bin_steps)
        return c, (band.mean(0), fail.sum(0), rate.mean(0) / DT)

    return jax.lax.scan(bin_fn, carry, None, length=n_bins)


run_phase = jax.jit(_run_phase, static_argnames=("n_bins", "bin_steps"))


def build_batch(seeds, conditions, eta_learn):
    """Stack agents; condition in {'full','frozen','shuffled','wired'}. Shuffled agents get the
    modulator of the next shuffled agent (cyclic), so M is independent of their own actions."""
    Ws, plas, etas = [], [], []
    for cond, seed in zip(conditions, seeds):
        W, pl = init_network(seed, wired=cond == "wired")
        Ws.append(W), plas.append(pl), etas.append(eta_learn if cond in ("full", "shuffled") else 0.0)
    perm = np.arange(len(seeds))
    sh = [i for i, c in enumerate(conditions) if c == "shuffled"]
    perm[sh] = np.roll(sh, -1)
    W = jnp.asarray(np.stack(Ws))
    plastic = jnp.asarray(np.stack(plas))
    S0 = (W * plastic).sum(-1)
    return W, plastic, S0, jnp.asarray(etas, jnp.float32), jnp.asarray(perm)


def init_carry(W, key):
    B = W.shape[0]
    k1, k2 = jax.random.split(key)
    z = jnp.zeros((B, N), jnp.float32)
    theta = jax.random.uniform(k1, (B,), minval=-P["th_reset"], maxval=P["th_reset"])
    return (k2, z, z, z, z, jnp.broadcast_to(jnp.asarray(RSTAR), (B, N)),
            jnp.ones((B, N), jnp.float32), W, jnp.zeros_like(W), theta, jnp.abs(theta), jnp.zeros(B))


def motor_map_index(W):
    """>0 when theta>0 sensors drive the down pool and theta<0 sensors drive the up pool."""
    up, dn = S[PREF < 0], S[PREF > 0]
    return float(W[np.ix_(MM, dn)].mean() - W[np.ix_(MP, dn)].mean()
                 + W[np.ix_(MP, up)].mean() - W[np.ix_(MM, up)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--n-wired", type=int, default=4)
    ap.add_argument("--train-s", type=int, default=300)
    ap.add_argument("--test-s", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override entries of P (recorded in params)")
    ap.add_argument("--plastic-scope", choices=("all", "sensory_motor"), default="all")
    a = ap.parse_args()
    global PLASTIC_SCOPE
    PLASTIC_SCOPE = a.plastic_scope
    for kv in a.set:
        k, val = kv.split("=")
        if k not in P:
            raise SystemExit(f"unknown parameter {k}")
        P[k] = float(val)

    seeds = list(range(a.seed0, a.seed0 + a.n_seeds))
    conds = ["full"] * a.n_seeds + ["frozen"] * a.n_seeds + ["shuffled"] * a.n_seeds + ["wired"] * a.n_wired
    all_seeds = seeds * 3 + seeds[: a.n_wired]
    W, plastic, S0, eta, perm = build_batch(all_seeds, conds, a.eta)
    t0 = time.time()
    carry = init_carry(W, jax.random.PRNGKey(a.seed0))
    carry, (tb, tf, tr) = run_phase(carry, eta, perm, plastic, S0, n_bins=a.train_s)
    carry, (sb, sf, sr) = run_phase(carry, jnp.zeros_like(eta), perm, plastic, S0, n_bins=a.test_s)
    jax.block_until_ready(sb)
    W_end = np.asarray(carry[7])
    res = dict(
        params=dict(P, eta=a.eta, plastic_scope=PLASTIC_SCOPE, seed0=a.seed0, n_seeds=a.n_seeds, train_s=a.train_s, test_s=a.test_s),
        wall_s=round(time.time() - t0, 1),
        conditions=conds, seeds=all_seeds,
        train_band=np.asarray(tb).T.tolist(), train_fail=np.asarray(tf).T.tolist(),
        train_rate_E=np.asarray(tr).T.tolist(),
        test_band=np.asarray(sb).mean(0).tolist(),
        test_fail_per_min=(np.asarray(sf).sum(0) * 60.0 / a.test_s).tolist(),
        test_rate_E=np.asarray(sr).mean(0).tolist(),
        motor_map_index=[motor_map_index(w) for w in W_end],
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(summarize(res))


def summarize(res) -> str:
    c = np.array(res["conditions"])
    band, fail, rate, mmi = (np.array(res[k]) for k in ("test_band", "test_fail_per_min", "test_rate_E", "motor_map_index"))
    lines = [f"wall {res['wall_s']} s  eta {res['params']['eta']}"]
    for k in ("full", "frozen", "shuffled", "wired"):
        m = c == k
        if m.any():
            lines.append(f"{k:9s} band {band[m].mean():.3f}+-{band[m].std():.3f}  fail/min {fail[m].mean():5.1f}"
                         f"  E Hz {rate[m].mean():.2f}  map {mmi[m].mean():+.3f}")
    n = int(res["params"]["n_seeds"])
    for k, off in (("frozen", n), ("shuffled", 2 * n)):
        d = band[:n] - band[off:off + n]
        lines.append(f"full-{k}: wins {int((d > 0).sum())}/{n}  mean diff {d.mean():+.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
