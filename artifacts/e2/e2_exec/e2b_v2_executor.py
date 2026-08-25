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
from jaxfne.emitters import simulate_edge_recurrent_izhikevich, EdgeList, IZHIKEVICH_CELL_TYPE_DEFAULTS
from jaxfne.emitters import IzhikevichParams

REPO = pathlib.Path('.').resolve()
PR = REPO / 'artifacts/e2/preregistration'
OUTD = PR / 'E2b_confirmatory/v2_runs'
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'artifacts/e2/e2_exec'))
import e2_exec_lib as lib

spec = json.loads((PR / 'e2_ssa_spec.v5.json').read_bytes())
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
    v, spk, src, fst = simulate_edge_recurrent_izhikevich(
        params, edges, sched.shape[0], DT, rk, dtype='float32',
        drive_schedule=jnp.asarray(sched), noise_scale=TH['noise_scale'])
    sp = np.asarray(spk); sr = np.asarray(src)
    inv = {}
    if not (np.isfinite(sp).all() and np.isfinite(sr).all()):
        inv['INVALID_NUMERIC_NONFINITE'] = True
    W0s, W1s = int(round(W0 / DT)), int(round(W1 / DT))
    ev_R, ev_Q, ev_ids = [], [], []
    for k in range(n_events):
        o = int(onsets[k])
        sl = slice(o + W0s, o + W1s)
        ev_R.append(float(sp[sl, :N_E].mean() / 0.08))
        ev_Q.append(float(np.abs(sr[sl, :]).mean()))
        ev_ids.append(bool(is_dev[k]))
    seq_hash = hashlib.sha256(json.dumps({'pos': pos.tolist(), 'onsets': [int(o) for o in onsets]}).encode()).hexdigest()
    ds = fst.get('delay_state')
    if ds is not None and not np.isfinite(np.asarray(ds)).all():
        inv['INVALID_DELAY_STATE'] = True
    return dict(name=name, R=ev_R, Q=ev_Q, is_dev=ev_ids, seq_hash=seq_hash,
                stim_seed=int(seed_used), INVALID=inv,
                mean_rate=float(sp.mean() * (1000.0 / DT)))

_KEYSHOLDER = {}
_KEYS = None

def main(rep: int):
    global _KEYS
    out_path = OUTD / f'rep_{rep:02d}.json'
    if out_path.exists():
        print(f'rep {rep} already done'); return
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
    receipt = dict(schema='e2b_v2_replicate.v1', replicate=rep, generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
                   spec_hash=spec['spec_hash'], theta=dict(drive_E=TH['drive_E'], drive_I=TH['drive_I'],
                   weight_mu=TH['weight_mu'], noise_scale=TH['noise_scale']),
                   seed_domains={k: int(lib.child_seed(v, 'identity')) for k, v in keys.items()},
                   blocks=blocks, INVALID=any_inv, write_once=True)
    out_path.write_text(json.dumps(receipt))
    print(f'rep {rep} written -> {out_path.name}')

if __name__ == '__main__':
    main(int(sys.argv[-1]))
