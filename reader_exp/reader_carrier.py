#!/usr/bin/env python3
"""Jitter reader on a carrier (reader_exp, scratch).

Circuit: A (100 clamped afferents via SpikeTrainInput) -> I (20 FS) -> R
(n_readers RS) with feedforward inhibition; A -> R excitation. Everything
through the public jaxfne API: configuration -> construct -> Model.simulate
-> Signals. Plasticity (when --hdp on) is trace_stdp realized as the additive
superposition of trace_ltp + trace_ltd via jaxfne.hdp_rule.superpose_hdp_rules.

Input: carrier inhomogeneous Poisson (r0=10 Hz, f=40 Hz, modulation m) plus
one pattern spike per carrier cycle on K=20 afferents at offsets d_k uniform
in [0, 5] ms after the rate peak plus N(0, sigma_j) jitter. Class Q shuffles
the offsets across the K afferents. Optional OU frequency drift (tau 200 ms)
with the pattern locked to the instantaneous carrier phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import resolve_edge_weight
from jaxfne.hdp_rule import list_registered_hdp_rules, superpose_hdp_rules

DT_MS = 0.05
N_A = 100
N_I = 20
K_PAT = 20
R0_HZ = 10.0
F0_HZ = 40.0
TAU_OU_MS = 200.0
BURN_IN_MS = 50.0
READER_RULE = "reader_exp_stdp"

IDX_A = np.arange(0, N_A)
IDX_I = np.arange(N_A, N_A + N_I)


# --------------------------------------------------------------------------
# input generation (numpy only)
# --------------------------------------------------------------------------

def _ou_noise(rng, n_steps, dt_ms, tau_ms=TAU_OU_MS):
    """Unit-stationary-variance OU process sampled at dt_ms."""
    dt_s = dt_ms / 1000.0
    tau_s = tau_ms / 1000.0
    decay = float(np.exp(-dt_s / tau_s))
    diffusion = float(np.sqrt(max(0.0, 1.0 - decay * decay)))
    x = np.empty(n_steps)
    x[0] = float(rng.standard_normal())
    for t in range(1, n_steps):
        x[t] = decay * x[t - 1] + diffusion * float(rng.standard_normal())
    return x


def draw_offsets(rng):
    """Pattern offsets d_k uniform in [0, 5] ms after the rate peak."""
    return np.sort(rng.uniform(0.0, 5.0, size=K_PAT))


def draw_permutation(rng):
    return rng.permutation(K_PAT)


def make_trial_input(trial_rng, duration_ms, m, df_hz, sigma_ms, offsets, cls):
    """Build one trial's clamped spikes.

    Returns (spike_times dict for SpikeTrainInput, info dict with peak times,
    phase, pattern indices, dropped counts).
    """
    n_steps = int(round(duration_ms / DT_MS))
    ou = _ou_noise(trial_rng, n_steps, DT_MS)
    freq = F0_HZ + float(df_hz) * ou
    phase = 2.0 * np.pi * np.cumsum(freq) * (DT_MS / 1000.0)
    rate = R0_HZ * (1.0 + float(m) * np.cos(phase))
    rate = np.clip(rate, 0.0, None)
    prob = rate * DT_MS / 1000.0

    pois = trial_rng.random((n_steps, N_A)) < prob[:, None]

    # rate-peak (phase == 0 mod 2pi) crossings: cycle k spans [cross_k, ...).
    cyc = phase / (2.0 * np.pi)
    k_all = np.arange(int(np.floor(cyc[0])) + 1, int(np.floor(cyc[-1])) + 1)
    # fractional step of each crossing by linear interpolation
    cross_step = []
    for k in k_all:
        j = int(np.searchsorted(cyc, float(k), side="left"))
        j = min(max(j, 1), n_steps - 1)
        c0, c1 = cyc[j - 1], cyc[j]
        frac = (float(k) - c0) / (c1 - c0) if c1 != c0 else 0.0
        cross_step.append((j - 1) + frac)
    cross_step = np.asarray(cross_step)
    peak_ms = cross_step * DT_MS
    # full cycles only: need the next peak inside the trial
    full = peak_ms[:-1][peak_ms[1:] <= duration_ms - 1e-9]
    n_cyc = int(full.shape[0])

    pattern_idx = np.arange(K_PAT)  # afferents 0..K-1 carry the payload
    pat_bins = set()
    dropped = 0
    spike_bins = {i: set(np.flatnonzero(pois[:, i]).tolist()) for i in range(N_A)}
    for pos, aff in enumerate(pattern_idx):
        d = float(offsets[pos])
        for ck in range(n_cyc):
            t = float(full[ck]) + d + float(trial_rng.normal(0.0, sigma_ms))
            b = int(np.floor(t / DT_MS + 1e-9))
            if b < 0 or b >= n_steps:
                dropped += 1
                continue
            pat_bins.add((int(aff), b))
            spike_bins[int(aff)].add(b)  # pattern spike present ...
    # ... with Poisson background removed at pattern bins (pattern precedence)
    for (aff, b) in pat_bins:
        pass
    # rebuild: poisson bins minus pattern bins, plus pattern bins
    pat_by_aff = {}
    for (aff, b) in pat_bins:
        pat_by_aff.setdefault(aff, set()).add(b)
    spike_times = {}
    for i in range(N_A):
        pb = pat_by_aff.get(i, set())
        base = set(np.flatnonzero(pois[:, i]).tolist()) - pb
        bins = sorted(base | pb)
        if bins:
            spike_times[i] = [float((bb + 0.5) * DT_MS) for bb in bins]
    info = {
        "peak_ms": [float(v) for v in full],
        "n_cycles": n_cyc,
        "pattern_idx": [int(v) for v in pattern_idx],
        "dropped": int(dropped),
        "freq_mean": float(freq.mean()),
    }
    return spike_times, info


# --------------------------------------------------------------------------
# circuit
# --------------------------------------------------------------------------

def build_model(n_readers, ffi_on, w_ai, w_ar, w_ir, d_I, d_R, seed,
                duration_ms):
    """Configuration -> construct -> per-neuron emitter overrides."""
    w_ir_eff = float(w_ir) if ffi_on else 0.0
    cfg = (
        jtfne.configuration()
        .runtime(seed=int(seed), dtype="float32",
                 duration_ms=float(duration_ms), dt_ms=float(DT_MS))
        .column("A", layers=["Laff"], n=N_A)
        .column("I", layers=["Linh"], n=N_I)
        .column("R", layers=["Lrd"], n=int(n_readers))
        .cell_types({"E": 1.0})
        .connectivity(p_connect=0.0)
        .connections(name="A_to_I", source={"layer": "Laff"},
                     target={"layer": "Linh"}, probability=1.0,
                     weight=float(w_ai), sign="excitatory")
        .connections(name="A_to_R", source={"layer": "Laff"},
                     target={"layer": "Lrd"}, probability=1.0,
                     weight=float(w_ar), sign="excitatory")
        .connections(name="I_to_R", source={"layer": "Linh"},
                     target={"layer": "Lrd"}, probability=1.0,
                     weight=float(w_ir_eff), sign="inhibitory",
                     delay_ms=2.0)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=4)
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann")
        .drive(baseline_drive_by_cell_type={"E": 0.0})
    )
    model = jtfne.construct(cfg)
    n = N_A + N_I + int(n_readers)
    assert int(model.params["emitter"].n_neurons) == n
    # verify layer layout: A=0..99, I=100..119, R=rest
    layers = [r["layer"] for r in model.neuron_table()]
    assert layers[:N_A] == ["Laff"] * N_A, layers[:5]
    assert layers[N_A:N_A + N_I] == ["Linh"] * N_I
    assert layers[N_A + N_I:] == ["Lrd"] * int(n_readers)
    # verify realized edge counts and signs (weights via the
    # test-pinned accessor; el.weight may be empty when compacted)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    w = resolved_weights(model)
    n_ai = int(((pre < N_A) & (post >= N_A) & (post < N_A + N_I)).sum())
    n_ar = int(((pre < N_A) & (post >= N_A + N_I)).sum())
    n_ir = int(((pre >= N_A) & (pre < N_A + N_I) & (post >= N_A + N_I)).sum())
    assert (n_ai, n_ar, n_ir) == (N_A * N_I, N_A * int(n_readers),
                                  N_I * int(n_readers)), (n_ai, n_ar, n_ir)
    assert int(el.n_edges) == n_ai + n_ar + n_ir
    assert float(w[(pre < N_A)].min()) > 0, "A efferents must be excitatory"
    ir_w = w[(pre >= N_A) & (pre < N_A + N_I)]
    if ffi_on:
        assert float(ir_w.max()) < 0, "I->R must be inhibitory when FFI on"
    else:
        assert float(np.abs(ir_w).max()) == 0.0, "FFI off needs zero I->R"
    a = np.full(n, 0.02)
    a[N_A:N_A + N_I] = 0.10  # FS
    d = np.full(n, 8.0)
    d[N_A:N_A + N_I] = 2.0  # FS
    drive = np.zeros(n)
    drive[N_A:N_A + N_I] = float(d_I)
    drive[N_A + N_I:] = float(d_R)
    model = model.with_emitter_parameters(
        a_per_neuron=a,
        b_per_neuron=np.full(n, 0.20),
        c_per_neuron=np.full(n, -65.0),
        d_per_neuron=d,
        drive_per_neuron=drive,
    )
    idx_R = np.arange(N_A + N_I, n)
    return model, idx_R


def ensure_reader_rule(w_max):
    if READER_RULE not in list_registered_hdp_rules():
        superpose_hdp_rules(READER_RULE, ["trace_ltp", "trace_ltd"],
                            w_bounds=(0.0, float(w_max)))
    return READER_RULE


def plasticity_mask_ar(model):
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    m = ((pre < N_A) & (post >= N_A + N_I)).astype(np.float64)
    assert int(m.sum()) == N_A * int((post.max() + 1) - (N_A + N_I))
    return m


def resolved_weights(model):
    """Per-edge signed weights via the test-pinned accessor.

    The edge_list backend compacts uniform magnitudes, so ``el.weight``
    may be empty; ``resolve_edge_weight`` materializes every storage mode.
    """
    el = model.params["edge_list"]
    sign = jnp.asarray(np.asarray(model.params["emitter"].sign))
    return np.asarray(resolve_edge_weight(el, el.weight.dtype,
                                          presynaptic_sign=sign), float)


def hdp_runtime(rule, mask):
    return jtfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"hdp_rule": rule, "plasticity_mask": np.asarray(mask),
                    "record_weight_trace": False})


def run_trial(model, sim_cfg, spike_times, n_total):
    inp = jtfne.SpikeTrainInput(spike_times, n_neurons=int(n_total))
    t0 = time.perf_counter()
    sig = jtfne.simulate(model, sim_cfg, inp)
    dt = time.perf_counter() - t0
    return sig, dt


# --------------------------------------------------------------------------
# measures
# --------------------------------------------------------------------------

def pop_rate(spikes, idx, duration_ms):
    s = np.asarray(spikes)[:, np.asarray(idx)]
    return float(s.sum() / (len(idx) * (duration_ms / 1000.0)))


def cycle_features(spikes_R, peak_ms, duration_ms):
    """Per-cycle (count, first-spike latency) of the R population."""
    s = np.asarray(spikes_R)
    r_any = s.sum(axis=1) > 0.5
    feats = []
    usable = [p for p in peak_ms if p >= BURN_IN_MS]
    for k, p0 in enumerate(usable):
        p1 = usable[k + 1] if k + 1 < len(usable) else float(duration_ms)
        if p1 - p0 <= 0:
            continue
        lo = int(np.floor(p0 / DT_MS + 1e-9))
        hi = int(np.floor(p1 / DT_MS + 1e-9))
        seg = np.flatnonzero(r_any[lo:hi])
        cnt = int(seg.shape[0])
        lat = float((seg[0] + lo) * DT_MS - p0) if cnt else float(p1 - p0)
        feats.append((cnt, lat, cnt > 0))
    return feats


def dprime(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or y.size < 2:
        return None
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    denom = (vx + vy) / 2.0
    if not np.isfinite(denom) or denom <= 0:
        return None
    return float((x.mean() - y.mean()) / np.sqrt(denom))


def lda_cv_accuracy(X, y, seed=0, n_folds=5):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    rng = np.random.default_rng(int(seed))
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    f0 = np.array_split(idx0, n_folds)
    f1 = np.array_split(idx1, n_folds)
    correct = total = 0
    for f in range(n_folds):
        te = np.concatenate([f0[f], f1[f]])
        tr = np.concatenate([f0[j] for j in range(n_folds) if j != f] +
                             [f1[j] for j in range(n_folds) if j != f])
        if tr.size == 0 or te.size == 0:
            continue
        m0 = X[tr[y[tr] == 0]].mean(axis=0)
        m1 = X[tr[y[tr] == 1]].mean(axis=0)
        v = (X[tr[y[tr] == 0]].var(axis=0, ddof=1)
             + X[tr[y[tr] == 1]].var(axis=0, ddof=1)) / 2.0
        v = np.where(np.isfinite(v) & (v > 1e-12), v, 1e-12)
        pr0 = float((y[tr] == 0).mean())
        pr1 = 1.0 - pr0
        ll = []
        for m, pr in ((m0, pr0), (m1, pr1)):
            ll.append(-0.5 * (((X[te] - m) ** 2) / v).sum(axis=1)
                      + np.log(max(pr, 1e-12)))
        pred = (np.asarray(ll[1]) > np.asarray(ll[0])).astype(int)
        correct += int((pred == y[te]).sum())
        total += int(te.size)
    return float(correct / total) if total else None


def periodogram_power(spikes, dt_ms, f0_hz):
    x = np.asarray(spikes, dtype=float).sum(axis=1)
    x = x - x.mean()
    n = x.shape[0]
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    k = int(np.argmin(np.abs(freqs - f0_hz)))
    return float((np.abs(spec[k]) ** 2) / n), float(freqs[k])


def trial_seed(base_seed, counter):
    return int(base_seed) * 1000003 + int(counter) * 7919 + 17


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="jitter reader on a carrier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-ms", type=float, default=1000.0)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--m", type=float, default=0.5)
    p.add_argument("--sigma-ms", type=float, default=1.0)
    p.add_argument("--ffi", choices=("on", "off"), default="on")
    p.add_argument("--df-hz", type=float, default=0.0)
    p.add_argument("--hdp", choices=("on", "off"), default="off")
    p.add_argument("--n-readers", type=int, default=5)
    p.add_argument("--out", type=str, default="run")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def run_condition_set(n_readers, ffi_on, w_ai, w_ar, w_ir, d_I, d_R, seed,
                      duration_ms, n_trials, m, sigma_ms, df_hz, rule,
                      hdp_on, w_init, train_rng_counters, collect_raster=False):
    """Run P and Q trials; returns dict of arrays and per-trial wall times."""
    model, idx_R = build_model(n_readers, ffi_on, w_ai, w_ar, w_ir, d_I, d_R,
                               seed, duration_ms)
    n_total = N_A + N_I + n_readers
    mask_ar = plasticity_mask_ar(model)
    zero_mask = np.zeros_like(mask_ar)
    rt_fixed = hdp_runtime(rule, zero_mask)
    rt_plast = hdp_runtime(rule, mask_ar)
    base = np.random.default_rng(int(seed))
    offsets_P = draw_offsets(base)
    perm = draw_permutation(base)
    offsets_Q = offsets_P[np.asarray(perm)]
    out = {"P": [], "Q": []}
    wall = []
    w_cur = None if w_init is None else np.asarray(w_init, float).copy()
    raster = None
    for cls_i, cls in enumerate(("P", "Q")):
        offs = offsets_P if cls == "P" else offsets_Q
        for t in range(n_trials):
            ctr = train_rng_counters + cls_i * 100000 + t
            trng = np.random.default_rng(trial_seed(seed, ctr))
            spike_times, info = make_trial_input(
                trng, duration_ms, m, df_hz, sigma_ms, offs, cls)
            rt = rt_plast if hdp_on else rt_fixed
            m_use = model
            if w_cur is not None:
                m_use = model.with_hdp_initial_state(w0=np.asarray(w_cur))
            sim_cfg = jtfne.simulation(duration_ms=float(duration_ms),
                                       dt_ms=float(DT_MS),
                                       seed=trial_seed(seed, ctr + 555),
                                       runtime=rt)
            sig, dt = run_trial(m_use, sim_cfg, spike_times, n_total)
            wall.append(dt)
            if hdp_on:
                w_cur = np.asarray(
                    m_use.last_hdp_diagnostics()["w_final"], float)
            spk = np.asarray(sig.spikes)
            out[cls].append({
                "spikes": spk,
                "peaks": info["peak_ms"],
                "n_cycles": info["n_cycles"],
            })
            if collect_raster and cls == "P" and t == 0 and raster is None:
                raster = {"spikes": spk, "peaks": info["peak_ms"],
                          "freq_mean": info["freq_mean"]}
    res = {
        "model": model, "idx_R": idx_R, "wall": wall,
        "offsets_P": offsets_P, "offsets_Q": offsets_Q,
        "mask_ar": mask_ar, "w_final": w_cur, "raster": raster,
    }
    for cls in ("P", "Q"):
        res[cls] = out[cls]
    return res


def summarize_condition(res, n_readers, duration_ms):
    """Rates, d', LDA accuracy, leakage from one condition set."""
    idx_R = np.asarray(res["idx_R"])
    n_cyc = min([r["n_cycles"] for r in res["P"]] +
                [r["n_cycles"] for r in res["Q"]])
    feats = {}
    for cls in ("P", "Q"):
        rows = []
        for r in res[cls]:
            spk = r["spikes"]
            feats_c = cycle_features(spk[:, idx_R], r["peaks"][:n_cyc + 1],
                                     duration_ms)
            rows.append([c[0] for c in feats_c] + [c[1] for c in feats_c])
        feats[cls] = np.asarray(rows)
    cnt_P = np.concatenate([np.asarray(
        [c for c in cycle_features(r["spikes"][:, idx_R],
                                  r["peaks"][:n_cyc + 1], duration_ms)],
        dtype=float)[:, 0] for r in res["P"]])
    cnt_Q = np.concatenate([np.asarray(
        [c for c in cycle_features(r["spikes"][:, idx_R],
                                  r["peaks"][:n_cyc + 1], duration_ms)],
        dtype=float)[:, 0] for r in res["Q"]])
    lat_P = np.concatenate([np.asarray(
        [c[1] for c in cycle_features(r["spikes"][:, idx_R],
                                     r["peaks"][:n_cyc + 1], duration_ms)
         if c[2]], dtype=float) for r in res["P"]])
    lat_Q = np.concatenate([np.asarray(
        [c[1] for c in cycle_features(r["spikes"][:, idx_R],
                                     r["peaks"][:n_cyc + 1], duration_ms)
         if c[2]], dtype=float) for r in res["Q"]])
    X = np.concatenate([feats["P"], feats["Q"]])
    y = np.concatenate([np.zeros(len(feats["P"]), dtype=int),
                        np.ones(len(feats["Q"]), dtype=int)])
    # R and A power at f0 over burn-in-cut concatenated trials
    lo = int(BURN_IN_MS / DT_MS)
    Rcat = np.concatenate([r["spikes"][lo:, idx_R] for r in res["P"]]
                          + [r["spikes"][lo:, idx_R] for r in res["Q"]],
                          axis=0)
    Acat = np.concatenate([r["spikes"][lo:, :N_A] for r in res["P"]]
                          + [r["spikes"][lo:, :N_A] for r in res["Q"]],
                          axis=0)
    pR, fR = periodogram_power(Rcat, DT_MS, F0_HZ)
    pA, fA = periodogram_power(Acat, DT_MS, F0_HZ)
    rates = {}
    for name, idx in (("A", IDX_A), ("I", IDX_I), ("R", idx_R)):
        vals = [pop_rate(r["spikes"], idx, duration_ms)
                for r in res["P"]] + [pop_rate(r["spikes"], idx,
                                              duration_ms) for r in res["Q"]]
        rates[name] = {"mean": float(np.mean(vals)),
                       "per_trial": [float(v) for v in vals]}
    h = hashlib.sha256()
    for cls in ("P", "Q"):
        for r in res[cls]:
            h.update(np.ascontiguousarray(r["spikes"][:, idx_R]).tobytes())
    return {
        "n_cycles_used": int(n_cyc),
        "rates_hz": rates,
        "dprime_count": dprime(cnt_P, cnt_Q),
        "dprime_latency": dprime(lat_P, lat_Q),
        "lda_accuracy": lda_cv_accuracy(X, y),
        "leakage_R_over_A": float(pR / pA) if pA > 0 else None,
        "power_R_f0": pR, "power_A_f0": pA, "power_bin_hz": fR,
        "r_spike_hash": h.hexdigest()[:16],
    }


def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    root = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root, "out")
    os.makedirs(outdir, exist_ok=True)

    smoke = bool(args.smoke)
    duration_ms = 500.0 if smoke else float(args.duration_ms)
    n_trials = 6 if smoke else int(args.trials)
    m = 0.5 if smoke else float(args.m)
    sigma_ms = 1.0 if smoke else float(args.sigma_ms)
    ffi_list = ["on", "off"] if smoke else [args.ffi]
    seeds = [0, 1] if smoke else [int(args.seed)]
    hdp_on = (args.hdp == "on")
    n_readers = int(args.n_readers)

    # --- weight sweep (tune on R rate only, FFI on, P class) ---
    w_ai, d_I = 1.0, 2.0
    sweep = [{"w_ar": 0.5, "w_ir": 0.5, "d_R": 3.0},
             {"w_ar": 1.0, "w_ir": 1.0, "d_R": 3.0},
             {"w_ar": 1.0, "w_ir": 2.0, "d_R": 3.0},
             {"w_ar": 0.5, "w_ir": 1.0, "d_R": 3.0},
             {"w_ar": 1.0, "w_ir": 1.0, "d_R": 1.0},
             {"w_ar": 1.0, "w_ir": 1.0, "d_R": 6.0}]
    sweep_rows = []
    chosen = None
    rule = "trace_stdp"  # core-registered; the sweep runs HDP-off (zero mask)
    for cand in sweep:
        res = run_condition_set(
            n_readers, True, w_ai, cand["w_ar"], cand["w_ir"], d_I,
            cand["d_R"], seeds[0], duration_ms, 1, m, sigma_ms,
            float(args.df_hz), rule, False, None, 900000 + len(sweep_rows))
        summ = summarize_condition(res, n_readers, duration_ms)
        rR = summ["rates_hz"]["R"]["mean"]
        rI = summ["rates_hz"]["I"]["mean"]
        row = {**cand, "R_hz": rR, "I_hz": rI,
               "wall_s": float(np.mean(res["wall"]))}
        sweep_rows.append(row)
        print(f"sweep w_ar={cand['w_ar']} w_ir={cand['w_ir']} "
              f"d_R={cand['d_R']}: R={rR:.2f} Hz I={rI:.2f} Hz "
              f"{row['wall_s']:.1f}s/trial", flush=True)
        if chosen is None and 5.0 <= rR <= 20.0:
            chosen = dict(cand)
    if chosen is None:
        print("STOP: R silent or saturated after the weight sweep", flush=True)
        payload = {"status": "STOP_sweep_failed", "sweep": sweep_rows,
                   "flags": vars(args)}
        with open(os.path.join(outdir, f"{args.out}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return 2
    w_max = 4.0 * float(chosen["w_ar"])
    rule = ensure_reader_rule(w_max)
    print(f"chosen {chosen} w_max={w_max}", flush=True)

    t_per_trial = float(np.mean([r["wall_s"] for r in sweep_rows[-1:]]))
    result = {
        "flags": vars(args),
        "dt_ms": DT_MS, "n_a": N_A, "n_i": N_I, "k_pattern": K_PAT,
        "r0_hz": R0_HZ, "f0_hz": F0_HZ, "tau_ou_ms": TAU_OU_MS,
        "burn_in_ms": BURN_IN_MS,
        "sweep": sweep_rows, "chosen": chosen, "w_max": w_max,
        "hdp_rule": rule,
        "hdp_off_method": ("plasticity_mask=zeros with registered "
                           "trace_stdp (SpikeTrainInput requires the "
                           "registered-rule kernel)"),
        "conditions": {},
    }

    # --- main condition sets ---
    for ffi in ffi_list:
        for seed in seeds:
            t_set = time.perf_counter()
            res = run_condition_set(
                n_readers, ffi == "on", w_ai, chosen["w_ar"],
                chosen["w_ir"], d_I, chosen["d_R"], seed, duration_ms,
                n_trials, m, sigma_ms, float(args.df_hz), rule, False,
                None, 1000 + 100 * (ffi == "off") + seed, collect_raster=True)
            summ = summarize_condition(res, n_readers, duration_ms)
            key = f"ffi_{ffi}_seed_{seed}"
            result["conditions"][key] = {
                **summ,
                "wall_s_per_trial": float(np.mean(res["wall"])),
                "wall_s_set": float(time.perf_counter() - t_set),
            }
            c = result["conditions"][key]
            print(f"{key}: R={c['rates_hz']['R']['mean']:.2f}Hz "
                  f"I={c['rates_hz']['I']['mean']:.2f}Hz "
                  f"A={c['rates_hz']['A']['mean']:.2f}Hz "
                  f"d'_c={c['dprime_count']} d'_l={c['dprime_latency']} "
                  f"acc={c['lda_accuracy']} leak={c['leakage_R_over_A']} "
                  f"hash={c['r_spike_hash']} "
                  f"{c['wall_s_per_trial']:.1f}s/trial", flush=True)
            if "raster_first" not in result and res["raster"] is not None:
                rr = res["raster"]
                result["raster_first"] = {
                    "condition": key,
                    "peaks_ms": rr["peaks"][:12],
                    "freq_mean_hz": rr["freq_mean"],
                    "spikes": np.asarray(rr["spikes"]).tolist(),
                }
            elapsed = time.perf_counter() - t_start
            print(f"  elapsed {elapsed:.0f}s", flush=True)

    # --- HDP smoke: 20 P training trials, then frozen P/Q tests ---
    hdp_report = None
    hdp_error = None
    if smoke or hdp_on:
        seed_h = seeds[0]
        try:
            hdp_report = run_hdp_leg(
                n_readers, w_ai, d_I, chosen, seed_h, duration_ms,
                n_trials, m, sigma_ms, float(args.df_hz), rule,
                n_train=20 if smoke else int(args.trials))
        except Exception as e:  # noqa: BLE001 - record the STOP, keep HDP-off outputs
            import traceback
            hdp_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"STOP: HDP leg failed: {type(e).__name__}: {e}", flush=True)
    result["hdp"] = hdp_report
    result["hdp_error"] = hdp_error

    result["wall_s_total"] = float(time.perf_counter() - t_start)
    result["status"] = "STOP_hdp_leg_failed" if hdp_error else "ok"

    # --- outputs ---
    json_path = os.path.join(outdir, f"{args.out}.json")
    png_path = os.path.join(outdir, f"{args.out}.png")
    serializable = json.loads(json.dumps(result, default=float))
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    make_png(png_path, result, n_readers, duration_ms, hdp_report)
    assert os.path.exists(json_path) and os.path.exists(png_path)
    print(f"wrote {json_path} {png_path}", flush=True)
    return 2 if hdp_error else 0


def run_hdp_leg(n_readers, w_ai, d_I, chosen, seed_h, duration_ms,
                n_trials, m, sigma_ms, df_hz, rule, n_train):
    """HDP training leg: 20 P trials carrying w, then frozen P/Q tests."""
    model0, _ = build_model(
        n_readers, True, w_ai, chosen["w_ar"], chosen["w_ir"], d_I,
        chosen["d_R"], seed_h, duration_ms)
    mask_ar = plasticity_mask_ar(model0)
    n_total = N_A + N_I + n_readers
    base = np.random.default_rng(int(seed_h))
    offsets_P = draw_offsets(base)
    perm = draw_permutation(base)
    offsets_Q = offsets_P[np.asarray(perm)]
    rt_plast = hdp_runtime(rule, mask_ar)
    el = model0.params["edge_list"]
    pre = np.asarray(el.pre)
    is_pat = np.isin(pre, np.arange(K_PAT))
    on_ar = (pre < N_A) & (np.asarray(el.post) >= N_A + N_I)
    w_cur = resolved_weights(model0)
    traj_pat, traj_car = [], []
    for t in range(n_train):
        trng = np.random.default_rng(trial_seed(seed_h, 500000 + t))
        spike_times, _ = make_trial_input(
            trng, duration_ms, m, df_hz, sigma_ms, offsets_P, "P")
        m_use = model0.with_hdp_initial_state(w0=np.asarray(w_cur))
        sim_cfg = jtfne.simulation(
            duration_ms=float(duration_ms), dt_ms=float(DT_MS),
            seed=trial_seed(seed_h, 500555 + t), runtime=rt_plast)
        t0 = time.perf_counter()
        sig = jtfne.simulate(
            m_use, sim_cfg,
            jtfne.SpikeTrainInput(spike_times, n_neurons=int(n_total)))
        dt = time.perf_counter() - t0
        w_cur = np.asarray(m_use.last_hdp_diagnostics()["w_final"], float)
        traj_pat.append(float(w_cur[on_ar & is_pat].mean()))
        traj_car.append(float(w_cur[on_ar & ~is_pat].mean()))
        print(f"hdp train {t + 1}/{n_train}: w_pat={traj_pat[-1]:.4f} "
              f"w_car={traj_car[-1]:.4f} {dt:.1f}s", flush=True)

    def frozen_accuracy(w, tag):
        res = run_condition_set(
            n_readers, True, w_ai, chosen["w_ar"], chosen["w_ir"],
            d_I, chosen["d_R"], seed_h, duration_ms, n_trials, m,
            sigma_ms, df_hz, rule, False, w,
            700000 + (1000 if tag == "after" else 0))
        return summarize_condition(res, n_readers, duration_ms)

    acc_before = frozen_accuracy(resolved_weights(model0), "before")
    acc_after = frozen_accuracy(w_cur, "after")
    hdp_report = {
        "n_train": n_train,
        "w_pattern_mean_start": traj_pat[0],
        "w_pattern_mean_end": traj_pat[-1],
        "w_carrier_mean_start": traj_car[0],
        "w_carrier_mean_end": traj_car[-1],
        "w_pattern_traj": traj_pat,
        "w_carrier_traj": traj_car,
        "test_acc_before": acc_before["lda_accuracy"],
        "test_acc_after": acc_after["lda_accuracy"],
        "dprime_before": acc_before["dprime_count"],
        "dprime_after": acc_after["dprime_count"],
    }
    print(f"HDP: w_pat {traj_pat[0]:.4f}->{traj_pat[-1]:.4f} "
          f"w_car {traj_car[0]:.4f}->{traj_car[-1]:.4f} "
          f"acc {acc_before['lda_accuracy']}->{acc_after['lda_accuracy']}",
          flush=True)
    return hdp_report


def make_png(png_path, result, n_readers, duration_ms, hdp_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rast = result.get("raster_first")
    n_panels = 2 if hdp_report else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(10, 3.5 * n_panels), squeeze=False)
    ax = axes[0][0]
    if rast is not None:
        spk = np.asarray(rast["spikes"])
        peaks = np.asarray(rast["peaks_ms"])
        show_A = list(range(min(K_PAT, N_A))) + list(
            range(K_PAT, min(K_PAT + 10, N_A)))
        for i in show_A:
            tt = np.flatnonzero(spk[:, i]) * DT_MS
            ax.scatter(tt, np.full_like(tt, i, dtype=float), s=1, c="k")
        for i in range(N_I):
            tt = np.flatnonzero(spk[:, N_A + i]) * DT_MS
            ax.scatter(tt, np.full_like(tt, N_A + i, dtype=float), s=4,
                       c="b")
        base = N_A + N_I
        for j in range(n_readers):
            tt = np.flatnonzero(spk[:, base + j]) * DT_MS
            ax.scatter(tt, np.full_like(tt, base + j, dtype=float), s=9,
                       c="r")
        for p in peaks:
            if p <= duration_ms:
                ax.axvline(p, color="g", lw=0.5, alpha=0.6)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("neuron idx (A black / I blue / R red; green=peaks)")
        ax.set_title(f"raster {rast['condition']} "
                     f"(carrier freq mean {rast['freq_mean_hz']:.2f} Hz)")
    else:
        ax.text(0.5, 0.5, "no raster", ha="center")
    if hdp_report:
        ax2 = axes[1][0]
        ax2.plot(hdp_report["w_pattern_traj"], label="pattern-mean w")
        ax2.plot(hdp_report["w_carrier_traj"], label="carrier-mean w")
        ax2.set_xlabel("training trial (class P)")
        ax2.set_ylabel("mean A->R weight")
        ax2.legend()
        ax2.set_title("HDP weight trajectories")
    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
