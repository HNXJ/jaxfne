"""E2b V2 SSA confirmatory executor — resumable per-replicate, frozen-only symbols.
Reads e2_ssa_spec.v5.json (chain v1..v5) + theta* inheritance. Per-replicate receipts
enable batch resumption. Memory-safe: per-event window sums only, no trace materialization.

Usage: python e2b_v2_executor.py <replicate_idx>
"""
import sys; sys.path.insert(0, '../../../..')  # repo root
sys.path.insert(0, '.')
import json, hashlib, pathlib, datetime
import numpy as np
import jax, jax.numpy as jnp
from jaxfne.emitters import (
    simulate_edge_recurrent_izhikevich, _simulate_edge_recurrent_izhikevich_delayed,
    EdgeList, IZHIKEVICH_CELL_TYPE_DEFAULTS, IzhikevichParams)

REPO = pathlib.Path('.').resolve()
PR = REPO / 'artifacts/e2/preregistration'
OUTD = PR / 'E2b_confirmatory/v2_runs'
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'artifacts/e2/e2_exec'))
import e2_exec_lib as lib

spec = json.loads((PR / 'e2_ssa_spec.v6.json').read_bytes())
ping = json.loads((PR / 'e2_ping_prereg.json').read_bytes())
e2a = json.loads((PR / 'E2a_search/e2a_search_receipt.json').read_bytes())
assert lib.canon_spec_hash(spec) == spec['spec_hash']
TH = spec['execution_grammar']['operating_point']
EG = spec['execution_grammar']
TOP, DL = spec['topology'], spec['delays']
N_E, N_I, N = TOP['N_E'], TOP['N_I'], TOP['N_total']
DT = DL['dt_ms']
WD = TOP['weight_distributions']; SIG = {k: v['sigma'] for k, v in WD.items()}
MU = {k: abs(v['mu']) for k, v in WD.items()} ; SIGN = {k: v['sign'] for k, v in WD.items()}
GRID = np.array(DL['discrete_steps'], dtype=np.int64)
PV = ping['inhibitory_model']; EDEF = IZHIKEVICH_CELL_TYPE_DEFAULTS['E']
ISI_STEPS = {'oddball': int(round(spec['SSA']['ISI_ms'] / DT)),
             'rec500': int(round(500 / DT)), 'rec1000': int(round(1000 / DT))}
STIM_STEPS = int(round(spec['SSA']['stim_duration_ms'] / DT))
W0, W1 = spec['SSA']['W_primary_ms']
OFFS = spec['seeds']['offsets']; ORDER = spec['seeds']['canonical_order']
N_REPS = EG['replicates']['n_outer_per_battery']

def build_circuit(rep):
    keys = lib.domain_keys(101, rep, OFFS, ORDER)
    s_int = lib.child_seed(keys['structure'], 'edges')
    rng = np.random.default_rng(s_int)
    r = TH['weight_mu'] / MU['E_E']
    pre_l, post_l, w_l = [], [], []
    for name, (npre, npost), op, oq in [('E_E', (N_E, N_E), 0, 0), ('E_I', (N_E, N_I), 0, N_E),
                                        ('I_E', (N_I, N_E), N_E, 0), ('I_I', (N_I, N_I), N_E, N_E)]:
        p = {'E_E': TOP['p_EE'], 'E_I': TOP['p_EI'], 'I_E': TOP['p_IE'], 'I_I': TOP['p_II']}[name]
        m = rng.random((npre, npost)) < p
        ii, jj = np.nonzero(m)
        keep = ii + op != jj + oq
        ii, jj = ii[keep], jj[keep]
        w_l.append(rng.normal(MU[name] * r * SIGN[name], SIG[name], ii.size))
        pre_l.append(ii + op); post_l.append(jj + oq)
    pre = np.concatenate(pre_l).astype(np.int32); post = np.concatenate(post_l).astype(np.int32)
    weight = np.concatenate(w_l).astype(np.float32)
    delays = rng.choice(GRID, size=pre.size).astype(np.int32)
    a = np.full(N, EDEF['a'], np.float32); b = np.full(N, EDEF['b'], np.float32)
    c = np.full(N, EDEF['c'], np.float32); d = np.full(N, EDEF['d'], np.float32)
    a[N_E:] = PV['a']; b[N_E:] = PV['b']; c[N_E:] = PV['c']; d[N_E:] = PV['d']
    drive = np.full(N, TH['drive_E'], np.float32); drive[N_E:] = TH['drive_I']
    sign = np.ones(N, np.float32); sign[N_E:] = -1.0
    params = IzhikevichParams(
        a=jnp.asarray(a), b=jnp.asarray(b), c=jnp.asarray(c), d=jnp.asarray(d),
        drive=jnp.asarray(drive), sign=jnp.asarray(sign), W=jnp.zeros((N, N), jnp.float32),
        v0=jnp.full((N,), -65.0, jnp.float32), u0=jnp.asarray(b * (-65.0), jnp.float32),
        source_scale=jnp.ones(N, jnp.float32), labels=tuple(['E'] * N_E + ['I'] * N_I))
    rec = (jnp.asarray(weight) < 0).astype(jnp.int32)
    edges = EdgeList(pre=jnp.asarray(pre), post=jnp.asarray(post), weight=jnp.asarray(weight),
                     receptor_index=rec, tau_ms=jnp.where(rec == 0, jnp.float32(2.0), jnp.float32(5.0)),
                     delay_steps=jnp.asarray(delays))
    return params, edges, keys

def deviant_positions(n_events, n_dev_expected_p, key_stim, tag):
    """Frozen rule: no consecutive deviants, >=2 standards between; positions from K_stimulus child."""
    s = lib.child_seed(key_stim, tag)
    rng = np.random.default_rng(s)
    n_dev = int(round(n_events * EG['paradigm']['p_deviant']))
    for _ in range(10000):
        pos = np.sort(rng.choice(np.arange(2, n_events), size=n_dev, replace=False))
        if np.all(np.diff(pos) >= 3):
            return pos, s
    raise RuntimeError('placement failed')

def block_schedule(n_events, isi_steps, std_pop, dev_pop, pos):
    steps = n_events * isi_steps
    sched = np.zeros((steps, N), np.float32)
    onsets = []
    for k in range(n_events):
        onset = k * isi_steps
        onsets.append(onset)
        pop = dev_pop if k in set(pos.tolist()) else std_pop
        sched[onset:onset + STIM_STEPS, pop[0]:pop[1]] = 1.0
    return sched, np.array(onsets), np.isin(np.arange(n_events), pos)

def _make_block_fn(params, edges, n_events, isi_steps):
    """JIT'd block runner with params/edges CLOSED OVER as constants (kernel's host-side
    delay guards then operate on real arrays). Outputs limited to per-event R/Q + flags:
    final_state/presyn_trace never returned -> DCE'd under jit (frozen memory_rule)."""
    W0s, W1s = int(round(W0 / DT)), int(round(W1 / DT))
    nev, isi = n_events, isi_steps
    n_steps = n_events * isi_steps

    @jax.jit
    def fn(rk, sched):
        out = _simulate_edge_recurrent_izhikevich_delayed(
            params, edges, n_steps, DT, rk, dtype='float32',
            drive_schedule=sched, noise_scale=TH['noise_scale'])
        spk, src = out[1], out[2]
        fin = jnp.isfinite(spk).all() & jnp.isfinite(src).all()
        sp_e = spk[:, :N_E].reshape(nev, isi, N_E)
        win = sp_e[:, W0s:W1s, :]
        R = win.sum(axis=(1, 2)) / (N_E * 0.08)
        Q = jnp.abs(src).reshape(nev, isi, N).sum(axis=1).mean(axis=1)
        mean_rate = spk.mean() * (1000.0 / DT)
        return R, Q, fin, mean_rate
    return fn

_BLOCK_FNS = {}

def get_block_fn(n_events, isi_steps):
    k = (n_events, isi_steps)
    if k not in _BLOCK_FNS:
        _BLOCK_FNS[k] = _make_block_fn(n_events, isi_steps)
    return _BLOCK_FNS[k]

A_POP, B_POP = (0, 400), (400, 800)

def run_block(params, edges, name, n_events, isi_steps, mode, rep):
    """mode: ('oddball', std_pop, dev_pop) | ('many_alt',) | ('std_only', pop)"""
    keys = _KEYS
    if mode[0] == 'oddball':
        _, std_pop, dev_pop = mode
        pos, seed_used = deviant_positions(n_events, None, keys['stimulus'],
                                           f"devpos_{name}_rep{rep}")
        sched, onsets, is_dev = block_schedule(n_events, isi_steps, std_pop, dev_pop, pos)
    elif mode[0] == 'many_alt':
        steps = n_events * isi_steps
        sched = np.zeros((steps, N), np.float32)
        onsets = np.arange(n_events) * isi_steps
        for k in range(n_events):
            pop = A_POP if k % 2 == 0 else B_POP
            o = int(onsets[k]); sched[o:o + STIM_STEPS, pop[0]:pop[1]] = 1.0
        is_dev = np.zeros(n_events, bool); pos = np.array([], int); seed_used = 0
    else:
        pop = mode[1]
        steps = n_events * isi_steps
        sched = np.zeros((steps, N), np.float32)
        onsets = np.arange(n_events) * isi_steps
        for k in range(n_events):
            o = int(onsets[k]); sched[o:o + STIM_STEPS, pop[0]:pop[1]] = 1.0
        is_dev = np.zeros(n_events, bool); pos = np.array([], int); seed_used = 0
    rk = keys['runtime']
    fn = _make_block_fn(params, edges, n_events, isi_steps)
    R_arr, Q_arr, fin, mean_rate_j = fn(rk, jnp.asarray(sched))
    R_h = np.asarray(R_arr); Q_h = np.asarray(Q_arr)
    inv = {}
    if not bool(fin):
        inv['INVALID_NUMERIC_NONFINITE'] = True
    ev_R = [float(x) for x in R_h]
    ev_Q = [float(x) for x in Q_h]
    ev_ids = [bool(b) for b in is_dev]
    seq_hash = hashlib.sha256(json.dumps({'pos': pos.tolist(), 'onsets': [int(o) for o in onsets]}).encode()).hexdigest()
    return dict(name=name, R=ev_R, Q=ev_Q, is_dev=ev_ids, seq_hash=seq_hash,
                stim_seed=(int(seed_used) if mode[0] == 'oddball' else None), INVALID=inv,
                mean_rate=float(mean_rate_j))

_KEYSHOLDER = {}
_KEYS = None

def main(rep: int):
    global _KEYS
    out_path = OUTD / f'rep_{rep:02d}.json'
    if out_path.exists():
        try:
            j = json.loads(out_path.read_bytes())
            assert j.get('schema') == 'e2b_v2_replicate.v1' and j.get('spec_hash') == spec['spec_hash']
            print(f'rep {rep} already done (validated)'); return
        except Exception:
            out_path.unlink()  # stale/corrupt -> re-run
    params, edges, keys = build_circuit(rep)
    _KEYS = keys
    blocks = []
    B = EG['paradigm']['blocks_per_replicate']
    for bd in B:
        nm = bd['name']
        if nm == 'oddball_A_std':
            bl = run_block(params, edges, nm, bd['events'], ISI_STEPS['oddball'], ('oddball', A_POP, B_POP), rep)
        elif nm == 'oddball_B_std_flip':
            bl = run_block(params, edges, nm, bd['events'], ISI_STEPS['oddball'], ('oddball', B_POP, A_POP), rep)
        elif nm == 'many_standards_control':
            bl = run_block(params, edges, nm, bd['events'], ISI_STEPS['oddball'], ('many_alt',), rep)
        elif nm == 'recovery_isi500':
            bl = run_block(params, edges, nm, bd['events'], ISI_STEPS['rec500'], ('std_only', A_POP), rep)
        elif nm == 'recovery_isi1000':
            bl = run_block(params, edges, nm, bd['events'], ISI_STEPS['rec1000'], ('std_only', A_POP), rep)
        else:
            raise ValueError(nm)
        blocks.append(bl)
        print(f"rep{rep} {nm}: INVALID={bl['INVALID']} mean_rate={bl['mean_rate']:.2f} "
              f"dev_events={sum(bl['is_dev'])}")
    any_inv = any(b['INVALID'] for b in blocks)
    import subprocess
    code_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip()
    executor_hash = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    receipt = dict(schema='e2b_v2_replicate.v1', replicate=rep,
                   generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
                   spec_hash=spec['spec_hash'],
                   code_head=code_head, executor_sha256=executor_hash,
                   theta=dict(drive_E=TH['drive_E'], drive_I=TH['drive_I'],
                              weight_mu=TH['weight_mu'], noise_scale=TH['noise_scale']),
                   seed_domains={k: int(lib.child_seed(v, 'identity')) for k, v in keys.items()},
                   blocks=blocks, INVALID=any_inv, write_once=True)
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt))
    import os
    os.replace(tmp, out_path)
    print(f'rep {rep} written -> {out_path.name}')

if __name__ == '__main__':
    main(int(sys.argv[-1]))
