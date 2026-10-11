"""K1h diagnostic: where pcl_stdp and pcl_stdp_h2 first differ on the column (diagnostic only).

Replays pcl_column_hdp.main() up to w2 for both rules in lockstep, from the same initial
weights and the same per-sequence (key, orientation, direction) as block() draws them.
The test blocks are skipped: they run the frozen rule and do not feed w1 or w2.
One JSON line per sequence. On the first sequence where the weights differ, that sequence
is rerun with weight traces and the first differing step is localised.
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
import pcl_column_hdp as D  # noqa: E402  (also registers both rules)
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402
from jaxfne.emitters import EdgeList  # noqa: E402
from jaxfne.hdp_rule import expected_aux_shape, get_hdp_rule  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

NA, NB = D.R.NAME, D.RH.NAME2
N, N_IN, N_STEPS, EXC = D.N, D.N_IN, D.N_STEPS, D.EXC
PRE, POST, TYP = D.PRE, D.POST, D.TYP
E = len(PRE)


def make_run(rule, s, amp, plastic_types, record):
    """make_runner of the driver with the rule passed explicitly and the weight trace optional."""
    params = K.izh(N)
    exc = jnp.isin(jnp.asarray(TYP), jnp.asarray(EXC))
    edges = EdgeList(pre=jnp.asarray(PRE, jnp.int32), post=jnp.asarray(POST, jnp.int32),
                     weight=jnp.where(exc, 1.0, -1.0).astype(jnp.float32),
                     receptor_index=jnp.where(exc, 0, 1).astype(jnp.int32),
                     tau_ms=jnp.full(E, K.TAU_INH, jnp.float32), source_calibration_status="x")
    rp = D.rule_params(s)
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
                             (jnp.arange(N_STEPS) * D.DT, jax.random.split(k1, N_STEPS)))
        pulse = ev | jnp.concatenate([jnp.zeros((1, N_IN), bool), ev[:-1]])
        sched = jnp.zeros((N_STEPS, N), jnp.float32).at[:, :N_IN].set(amp * pulse)
        st = {"v": jnp.full(N, -65.0), "u": jnp.full(N, -13.0), "prev_spikes": jnp.zeros(N),
              "syn_state": jnp.zeros(E), "w_final": jnp.asarray(sign) * w_mag,
              "aux_final": jnp.zeros((E, 3), jnp.float32)}
        if rule == NB:
            st["aux_final"] = jnp.zeros((E, 2), jnp.float32)
            st["H_final"] = jnp.zeros((N, 3), jnp.float32)
        v, spikes, src, diag = sim(params, edges, N_STEPS, D.DT, k2, drive_schedule=sched, noise_scale=0.0,
                                   init_state=st, hdp_rule=rule, hdp_rule_params=rp,
                                   record_weight_trace=record, plasticity_mask=mask, v_floor=D.V_FLOOR)
        return jnp.abs(diag["w_final"]), spikes.sum(0), diag["w_trace"], diag["aux_final"]

    return run


def draws(key, n_seq):
    """Same draws as driver block(): orientations and directions from a generator seeded by key."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    oris = rng.integers(0, C.N_ORI, n_seq)
    dirs = rng.choice([-1.0, 1.0], len(oris))
    return list(zip(jax.random.split(key, n_seq), oris, dirs))


def driver_check(rule, s, amp, plastic, w_in, k, o, d, mine):
    """The replica must match the driver's make_runner for the same rule (driver reads its RULE at trace)."""
    D.RULE = rule
    drv = D.make_runner(s, amp, plastic)
    assert D.RULE == rule
    w_d, _ = drv(w_in, k, o, d)
    same = np.array_equal(np.asarray(w_d), mine)
    print(f"[check] replica vs driver make_runner, rule {rule}, phase 1 seq 0: identical={same}", flush=True)
    assert same, f"replica differs from driver make_runner for {rule}"


def repr_f32(x):
    return np.format_float_scientific(np.float32(x), unique=True)


def localise(plastic, w_in, k, o, d, s, amp, sweep_a, sweep_b):
    ra = make_run(NA, s, amp, plastic, True)
    rb = make_run(NB, s, amp, plastic, True)
    wa, _, tra, _ = ra(w_in, k, o, d)
    wb, _, trb, _ = rb(w_in, k, o, d)
    tra, trb = np.asarray(tra), np.asarray(trb)
    assert tra.shape == (N_STEPS, E) and trb.shape == (N_STEPS, E), (tra.shape, trb.shape)
    rerun_ok = bool(np.array_equal(np.asarray(wa), sweep_a) and np.array_equal(np.asarray(wb), sweep_b))
    trace_ok = bool(np.array_equal(np.abs(tra[-1]), np.asarray(wa)) and np.array_equal(np.abs(trb[-1]), np.asarray(wb)))
    neq = tra != trb
    any_t = neq.any(axis=1)
    if not any_t.any():
        return dict(found=False, rerun_matches_sweep=rerun_ok, trace_last_matches=trace_ok)
    t0 = int(np.argmax(any_t))
    e = int(np.argmax(neq[t0]))
    sgn = 1.0 if int(TYP[e]) in EXC else -1.0
    if t0 > 0:
        prev_a, prev_b = float(tra[t0 - 1, e]), float(trb[t0 - 1, e])
    else:
        prev_a = prev_b = sgn * float(np.abs(np.asarray(w_in))[e])
    return dict(
        found=True, rerun_matches_sweep=rerun_ok, trace_last_matches=trace_ok,
        step=t0, time_ms=float(t0 * D.DT), n_edges_differ_at_step=int(neq[t0].sum()),
        edge=e, pre=int(PRE[e]), post=int(POST[e]), type=D.TYPES[int(TYP[e])],
        value_before_a=repr_f32(prev_a), value_before_b=repr_f32(prev_b),
        value_a=repr_f32(tra[t0, e]), value_b=repr_f32(trb[t0, e]),
    )


def run_phase(ph, plastic, wa, wb, key, n_seq, s, amp, jl, found):
    ra = make_run(NA, s, amp, plastic, False)
    rb = make_run(NB, s, amp, plastic, False)
    last = None
    for i, (k, o, d) in enumerate(draws(key, n_seq)):
        o, d = int(o), float(d)
        wa_in, wb_in = wa, wb
        wa, ca, _, aux_a = ra(wa_in, k, o, d)
        wb, cb, _, aux_b = rb(wb_in, k, o, d)
        if i == 0:
            shapes = (tuple(aux_a.shape), tuple(aux_b.shape))
            assert shapes == ((E, 3), (E, 2)), shapes
            assert shapes[0] == tuple(expected_aux_shape(get_hdp_rule(NA)[0], n_neurons=N, n_edges=E))
            assert shapes[1] == tuple(expected_aux_shape(get_hdp_rule(NB)[0], n_neurons=N, n_edges=E))
            print(f"[check] phase {ph} seq 0 aux shapes {NA}={shapes[0]} {NB}={shapes[1]}", flush=True)
            if ph == 1:
                driver_check(NA, s, amp, plastic, wa_in, k, o, d, np.asarray(wa))
                driver_check(NB, s, amp, plastic, wb_in, k, o, d, np.asarray(wb))
        wa_np, wb_np = np.asarray(wa), np.asarray(wb)
        mx = float(np.abs(wa_np - wb_np).max())
        rec = dict(phase=ph, seq=i, ori=o, dir=d, max_dw=mx,
                   n_diff=int(np.count_nonzero(wa_np != wb_np)),
                   spikes_a=int(np.asarray(ca).sum()), spikes_b=int(np.asarray(cb).sum()))
        jl.write(json.dumps(rec) + "\n")
        jl.flush()
        if i % 50 == 0:
            print(f"[progress] phase {ph} seq {i}/{n_seq} max_dw={mx:.6g}", flush=True)
        if mx > 0 and found["first"] is None:
            assert np.array_equal(np.asarray(wa_in), np.asarray(wb_in)), "inputs already differ"
            found["first"] = dict(phase=ph, seq=i, ori=o, dir=d, max_dw=mx)
            found["loc"] = localise(plastic, wa_in, k, o, d, s, amp, wa_np, wb_np)
            print("[first divergence] " + json.dumps(dict(found["first"], loc=found["loc"])), flush=True)
        last = mx
    return wa, wb, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-exc", type=int, default=600)
    ap.add_argument("--n-inh", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    K.DT = D.DT
    amp, w_cancel = K.calibrate()
    s = w_cancel / 5.3
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    k_p1, k_p2, _, _ = jax.random.split(key, 4)
    w0 = D.init_weights(rng, s, (0, 1, 4, 5))
    w_inh = D.init_weights(rng, s, (2, 3))
    found = {"first": None, "loc": None}
    jl_path = a.out_dir / f"k1h_seed{a.seed}.jsonl"
    jl = open(jl_path, "w", encoding="utf-8")
    w0j = jnp.asarray(w0, jnp.float32)
    wa, wb, last1 = run_phase(1, (0, 1, 4, 5), w0j, w0j, k_p1, a.n_exc, s, amp, jl, found)
    w1 = np.asarray(wa)
    w1_b = np.asarray(wb)
    wa2 = jnp.asarray(np.where(np.isin(TYP, (2, 3)), w_inh, w1), jnp.float32)
    wb2 = jnp.asarray(np.where(np.isin(TYP, (2, 3)), w_inh, w1_b), jnp.float32)
    wa, wb, last2 = run_phase(2, (2, 3), wa2, wb2, k_p2, a.n_inh, s, amp, jl, found)
    jl.close()
    with open(jl_path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    summary = dict(seed=a.seed, n_exc=a.n_exc, n_inh=a.n_inh, n_jsonl_lines=n_lines, rules=[NA, NB],
                   n_edges=E, scale=float(s), pulse_amp=float(amp), w_cancel=float(w_cancel),
                   phase1_last_max_dw=last1, phase2_last_max_dw=last2,
                   first_divergence=found["first"], localisation=found["loc"],
                   wall_s=round(time.time() - t0, 1))
    (a.out_dir / f"k1h_seed{a.seed}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("[summary] " + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
