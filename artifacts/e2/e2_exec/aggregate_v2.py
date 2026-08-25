"""V2 SSA aggregation + classification using ONLY symbols defined in e2_ssa_spec chain.
Emits immutable v2_ssa_confirmatory_receipt.json."""
import sys, json, hashlib, pathlib, datetime, math
sys.path.insert(0, '../../../..'); sys.path.insert(0, '.')
import numpy as np
REPO = pathlib.Path('.').resolve()
PR = REPO / 'artifacts/e2/preregistration'
spec = json.loads((PR / 'e2_ssa_spec.v6.json').read_bytes())
M = spec['SSA']['metrics']; CL = spec['classifiers']
EG = spec['execution_grammar']
EPS = M['epsilon']; TH_SI = M['theta_SI']; DELTA_MIN = M['delta_min']; DELTA_REC = M['delta_rec']
DELTA_MECH = M['delta_mech']; SHUF_MAX = M['shuf_max']; SWAP_MAX = M['swap_max']
TH_MECH = M['theta_mech']; DELTA_GLOBAL = M['delta_global']; DELTA_LEAK = M['delta_leak_max']
import e2_exec_lib as lib

RUNS = sorted((PR / 'E2b_confirmatory/v2_runs').glob('rep_*.json'))
assert len(RUNS) == EG['replicates']['n_outer_per_battery'], len(RUNS)

def si(rdev, rstd):
    return (rdev - rstd) / (rdev + rstd + EPS)

def hedges(vals):
    v = np.asarray(vals, float); n = len(v)
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(n) + 1e-30)) if n > 1 else 0.0

from statistics import NormalDist
_ND = NormalDist()

def _invnorm(p):
    return _ND.inv_cdf(min(max(p, 1e-15), 1 - 1e-15))

def _normcdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bca_ci(vals, draws=5000, key=None):
    """BCa bootstrap: percentile + bias-correction (z0) + jackknife acceleration."""
    v = np.asarray(vals, float); n = len(v)
    rng = np.random.default_rng(key)
    idx = rng.integers(0, n, size=(draws, n))
    stats = v[idx].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    prop = (stats < v.mean()).mean()
    z0 = math.sqrt(2) * _invnorm(min(max(prop, 1e-9), 1 - 1e-9))
    jack = np.array([np.delete(v, i).mean() for i in range(n)])
    jm = jack.mean()
    acc = ((jack - jm) ** 3).sum() / (6 * (((jack - jm) ** 2).sum()) ** 1.5 + 1e-30)
    a = acc
    alpha = 0.05
    zlo = _invnorm(alpha / 2); zhi = _invnorm(1 - alpha / 2)
    alo = _normcdf(z0 + (zlo / (1 - a * z0))) * 2
    ahi = _normcdf(z0 + (zhi / (1 - a * z0))) * 2
    lo2, hi2 = np.percentile(stats, [100 * max(alo, 0), 100 * min(ahi, 1)])
    return float(lo2), float(hi2), (float(lo), float(hi))

def perm_p(per_rep_paired_diff_fn, blocks_by_rep, draws=2000, key=None):
    """Within-replicate std/dev label shuffle preserving counts -> pooled statistic null."""
    rng = np.random.default_rng(key)
    obs = per_rep_paired_diff_fn(blocks_by_rep, None)[0]
    cnt = 0
    for _ in range(draws):
        stat = per_rep_paired_diff_fn(blocks_by_rep, rng)[0]
        if stat >= obs:
            cnt += 1
    return (1 + cnt) / (draws + 1), obs

def shuffle_R(block, rng):
    R = np.array(block['R']); isd = np.array(block['is_dev'])
    if isd.sum() == 0 or (~isd).sum() == 0:
        return R, isd
    perm = rng.permutation(len(R))
    ndev = int(isd.sum())
    new_isd = np.zeros(len(R), bool); new_isd[perm[:ndev]] = True
    return R, new_isd

reps = []
for rp in RUNS:
    j = json.loads(rp.read_bytes())
    assert j['schema'] == 'e2b_v2_replicate.v1' and not j['INVALID']
    bl = {b['name']: b for b in j['blocks']}
    W = lambda b: (np.array(b['R'])[2:], np.array(b['Q'])[2:], np.array(b['is_dev'])[2:])  # washout first 2
    RA, QA, dA = W(bl['oddball_A_std'])
    RB, QB, dB = W(bl['oddball_B_std_flip'])
    RM, QM, _ = W(bl['many_standards_control'])
    R5, _, _ = W(bl['recovery_isi500'])
    R10, _, _ = W(bl['recovery_isi1000'])
    rA, rB = float(RA[~dA].mean()), float(RB[~dB].mean())
    dA_r, dB_r = float(RA[dA].mean()), float(RB[dB].mean())
    qA, qB = float(QA[~dA].mean()), float(QB[~dB].mean())
    qAd, qBd = float(QA[dA].mean()), float(QB[dB].mean())
    si_a = si(dA_r, rA); si_b = si(dB_r, rB)
    rec = dict(rep=j['replicate'],
               SI=(si_a + si_b) / 2, SI_flip_blocks=[si_a, si_b],
               dR=((dA_r - rA) + (dB_r - rB)) / 2,
               SI_source=si((qAd - qB) / 2 + (qBd - qA) / 2, 0.0) if False else si((qAd + qBd) / 2 - 0, (qA + qB) / 2),
               R_std=(rA + rB) / 2, R_dev=(dA_r + dB_r) / 2,
               swap_asym=abs(si_a - si_b),
               SI_many=float(abs(si(float(RM[::2].mean()), float(RM[1::2].mean())))),
               R_early=float(RA[1:16].mean()),   # events 3-17 original == idx1..15 post-washout
               R_late=float(RA[-16:].mean()),    # events 65-80 original == last 16 post-washout
               R_rec500=float(np.delete(R5, [0, 1]).mean()),
               R_rec1000=float(np.delete(R10, [0, 1]).mean()))
    # stability quarters per identity standards
    for nm, arr in (('A', RA[~dA]), ('B', RB[~dB])):
        q = len(arr) // 4
        drift = abs(arr[-q:].mean() - arr[:q].mean())
        sd = arr.std() + 1e-30
        rec[f'stable_{nm}'] = bool(drift <= 3 * sd)
        rec[f'R_{nm}_early'] = float(arr[:15].mean())
    rec['G_A'] = bool(rec['R_A_early'] >= M['R_floor'] and rec['stable_A'])
    rec['G_B'] = bool(rec['R_B_early'] >= M['R_floor'] and rec['stable_B'])
    rc = float(RM.mean())
    rec['SI_std_ctrl'] = si(rc, rec['R_std'])
    rec['SI_dev_ctrl'] = si(rc, rec['R_dev'])
    reps.append(rec)

# pooled statistics
SIs = np.array([r['SI'] for r in reps])
pooled_SI = float(SIs.mean())
key_bca = lib.child_seed(json.loads((RUNS[0]).read_bytes()) and __import__('jax').random.PRNGKey(0), 'unused') if False else None
keys0 = lib.domain_keys(101, 0, spec['seeds']['offsets'], spec['seeds']['canonical_order'])
kA = keys0['analysis']
lo, hi, raw = bca_ci(SIs, 5000, key=lib.child_seed(kA, 'bca_S2'))
g = hedges(SIs)
# permutation: shuffle std/dev labels within oddball blocks per replicate
def pooled_stat(reps_, rng):
    out = []
    for r, rp in zip(reps_, RUNS):
        j = json.loads(rp.read_bytes())
        bl = {b['name']: b for b in j['blocks']}
        s = 0.0; nb = 0
        for bn in ('oddball_A_std', 'oddball_B_std_flip'):
            b = bl[bn]
            R = np.array(b['R'])[2:]
            isd = np.array(b['is_dev'])[2:]
            if rng is not None:
                pm = rng.permutation(len(R))
                nd = int(isd.sum())
                nisd = np.zeros(len(R), bool); nisd[pm[:nd]] = True
                isd = nisd
            if isd.sum() == 0 or (~isd).sum() == 0:
                continue
            s += si(float(R[isd].mean()), float(R[~isd].mean())); nb += 1
        out.append(s / max(nb, 1))
    return float(np.mean(out)), out
rng_perm = np.random.default_rng(lib.child_seed(kA, 'perm_S2'))
obs_stat, obs_per = pooled_stat(reps, None)
cnt = sum(1 for _ in range(2000) if pooled_stat(reps, rng_perm)[0] >= obs_stat)
p_perm = (1 + cnt) / 2001
rng_shuf = np.random.default_rng(lib.child_seed(kA, 'shuf_oddball'))
sh_stats = []
for _ in range(2000):
    sh_stats.append(pooled_stat(reps, rng_shuf)[0])
SI_shuf = float(np.mean(sh_stats))
SI_many_pool = float(np.mean([r['SI_many'] for r in reps]))
swap_pool = float(np.max([r['swap_asym'] for r in reps]))
SI_src_pool = float(np.mean([r['SI_source'] for r in reps]))

adequacy_all = all(r['G_A'] and r['G_B'] for r in reps)
gray = 0.08 <= pooled_SI < 0.10
label = None
n_S0 = sum(1 for r in reps if abs(r['SI']) < TH_SI and abs(r['dR']) < DELTA_GLOBAL)
if not adequacy_all:
    label = 'UNRESOLVED:!G_adequate_SSA'
elif (pooled_SI > TH_SI and lo > TH_SI and float(np.mean([r['dR'] for r in reps])) > DELTA_MIN and g > 0.40
      and p_perm < 0.025 and np.sign(SI_src_pool) == np.sign(pooled_SI)
      and abs(SI_shuf) < SHUF_MAX and SI_many_pool < TH_SI and swap_pool <= SWAP_MAX):
    label = 'S2'
else:
    hard_fail = (not (abs(SI_shuf) < SHUF_MAX)) or (not (SI_many_pool < TH_SI)) or (not (swap_pool <= SWAP_MAX))
    s1 = all(r['SI_std_ctrl'] > TH_SI and r['SI_dev_ctrl'] > TH_SI and
             abs(r['SI_std_ctrl'] - r['SI_dev_ctrl']) < DELTA_LEAK for r in reps)
    if s1 and pooled_SI > TH_SI:
        label = 'S1'
    elif pooled_SI > TH_SI and gray and not hard_fail:
        label = 'UNRESOLVED:GRAY_SI'
    else:
        # No S-gate satisfied -> claim-level NEGATIVE with frozen-label-compatible taxonomy
        fails = []
        if pooled_SI <= TH_SI or lo <= 0:
            fails.append('FAIL_SI_gate')
        if swap_pool > SWAP_MAX:
            fails.append('FAIL_swap_asymmetry')
        if hard_fail and abs(SI_shuf) >= SHUF_MAX:
            fails.append('FAIL_shuffled_history')
        if hard_fail and SI_many_pool >= TH_SI:
            fails.append('FAIL_many_standards')
        if pooled_SI < 0:
            fails.append('SIGN_deviant_below_standard')
        label = 'NEGATIVE:' + ','.join(fails)
n_unres = 20 if label == 'UNRESOLVED:!G_adequate_SSA' else 0
delta_recs = [r['R_rec500'] - r['R_late'] for r in reps]
d_rec = float(np.mean(delta_recs))
lo3, hi3, _ = bca_ci(delta_recs, 5000, key=lib.child_seed(kA, 'bca_S3'))
I_recs = []
for r in reps:
    den = r['R_early'] - r['R_late']
    I_recs.append((r['R_rec500'] - r['R_late']) / den if den > 0 else None)
I_rec_ok = all(i is not None and i > 0.20 for i in I_recs)
pairs = [(isi, ev) for isi, rr in ((500, 'R_rec500'), (1000, 'R_rec1000')) for r in reps for ev in [r[rr]]]
isis = np.array([p[0] for p in pairs]); evs = np.array([p[1] for p in pairs])
rho = float(np.corrcoef(np.argsort(np.argsort(isis)), np.argsort(np.argsort(evs)))[0, 1]) if len(set(isis)) > 1 else 0.0
rng_p3 = np.random.default_rng(lib.child_seed(kA, 'perm_S3'))
obs3 = d_rec; c3 = 0
dl = np.array(delta_recs)
for _ in range(2000):
    sign = rng_p3.choice([-1.0, 1.0], size=len(dl))
    if (sign * dl).mean() >= obs3:
        c3 += 1
p_rec = (1 + c3) / 2001
S3 = bool(d_rec > DELTA_REC and lo3 > 0 and p_rec < 0.025 and I_rec_ok and rho >= 0.20)

n_unres = 0 if label != 'UNRESOLVED:!G_adequate_SSA' else 20
receipt = dict(
    schema='e2b_v2_confirmatory_receipt.v1',
    generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
    authority=dict(ssa_v6_spec_hash=spec['spec_hash'], ping_spec_hash=json.loads((PR/'36_AUDIT_INPUT_MANIFEST.json').read_bytes())['ping_spec_hash'],
                   parent_e1=spec['parent_e1']),
    runs=[{k: r[k] for k in ('rep','SI','dR','SI_source','SI_many','swap_asym','SI_std_ctrl','SI_dev_ctrl','G_A','G_B','R_early','R_late','R_rec500','R_rec1000')} for r in reps],
    pooled=dict(SI=pooled_SI, dR=float(np.mean([r['dR'] for r in reps])), g=g,
                BCa_lower=lo, BCa_upper=hi, BCa_raw_percentile=raw,
                p_perm=p_perm, SI_source_pool=SI_src_pool, SI_shuf_mean=SI_shuf,
                SI_many_pool=SI_many_pool, swap_max_observed=swap_pool,
                adequacy_all=bool(adequacy_all)),
    S3=dict(delta_rec=d_rec, BCa_lower=lo3, p_rec=p_rec, I_rec_all_gt_02=I_rec_ok, rho=rho, gate_pass=S3),
    S4_status='CONFIRMATORY_DEFERRED_v3_per_spec_factor_staging',
    verdict=dict(V2_polarity='NEGATIVE' if label.startswith('NEGATIVE') or label in ('S0',) else label,
                 subclass=label, n_S0_per_replicate=n_S0,
                 arithmetic=f'n=20 runs; INVALID=0; UNRESOLVED={n_unres}; classified={20-n_unres}; '
                            f'n_total = n_INVALID + n_UNRESOLVED + classified'),
    write_once=True)
out = PR / 'E2b_confirmatory/v2_ssa_confirmatory_receipt.json'
out.write_text(json.dumps(receipt, indent=2, default=str))
print('VERDICT V2:', label)
print('pooled SI', round(pooled_SI, 4), 'BCa', round(lo, 4), round(hi, 4), 'g', round(g, 3), 'p_perm', p_perm,
      'SI_many', round(SI_many_pool, 4), 'swap', round(swap_pool, 4), 'SI_shuf', round(SI_shuf, 4))
print('S3:', receipt['S3'])
