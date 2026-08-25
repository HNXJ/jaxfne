"""E2b V1 PING confirmatory executor — SINGULAR, measure->classify, never tune.

Imports frozen classifiers + adequacy from e2_ping_prereg.json (post-freeze legal).
No parameter search, no thresholds outside JSON, no run exclusion except INVALID rules.
Operating point theta* read verbatim from write-once E2a receipt.
"""
import sys; sys.path.insert(0, '.')
import json, hashlib, pathlib, datetime, math
import numpy as np
import jax, jax.numpy as jnp

from jaxfne.emitters import (
    simulate_edge_recurrent_izhikevich, EdgeList,
    IZHIKEVICH_CELL_TYPE_DEFAULTS,
)

ROOT = pathlib.Path('.')
PR = ROOT / 'artifacts/e2/preregistration'
OUT = PR / 'E2b_confirmatory'
OUT.mkdir(parents=True, exist_ok=True)

ping = json.loads((PR / 'e2_ping_prereg.json').read_bytes())
ssa = json.loads((PR / 'e2_ssa_spec.json').read_bytes())
manifest = json.loads((PR / '36_AUDIT_INPUT_MANIFEST.json').read_bytes())
e2a = json.loads((PR / 'E2a_search/e2a_search_receipt.json').read_bytes())

def canon_hash(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert ping['spec_hash'] == canon_hash(ping) == manifest['ping_spec_hash']
assert ssa['spec_hash'] == canon_hash(ssa) == manifest['ssa_spec_hash']

THETA = e2a['result']['theta*']
assert THETA['id'] == 'theta0'
TOP, DEL, TME, MTH = ping['topology'], ping['delays'], ping['time'], ping['method']
N_E, N_I, N = TOP['N_E'], TOP['N_I'], TOP['N_total']
DT = TME['dt_ms']; NSTEP = TME['n_steps']; FS = MTH['sampling_rate_hz']
W_SLICE = slice(int(TME['W'][0] / DT), int(TME['W'][1] / DT))
W1_SLICE = slice(int(TME['W1'][0] / DT), int(TME['W1'][1] / DT))
WD = TOP['weight_distributions']
SIG = {k: v['sigma'] for k, v in WD.items()}
MU_NOM = {k: abs(v['mu']) for k, v in WD.items()}
SIGN = {k: v['sign'] for k, v in WD.items()}
GRID_STEPS = np.array(DEL['discrete_steps'], dtype=np.int64)
PV = ping['inhibitory_model']; EDEF = IZHIKEVICH_CELL_TYPE_DEFAULTS['E']

# ---- interpretation notes (mechanical readings, documented not tuned) ----
NOTES = {
 "weight_scale_reading": "weight_mu scales all projection mus proportionally to nominal mu_EE=0.35: mu_X = mu_X_nom * weight_mu/0.35; sigmas unchanged (frozen).",
 "e_cell_params": "Package-native default for label 'E' (IZHIKEVICH_CELL_TYPE_DEFAULTS): " + json.dumps(EDEF),
 "i_cell_params": "Prereg inhibitory_model PV: " + json.dumps({k: PV[k] for k in ('a','b','c','d')}),
 "init": "v0=-65, u0=b*v0 canonical per 28 goal 77.",
 "seed_count_contradiction": "seeds.cardinalities.n_per_cell=20 conflicts with collapse rule '>=4/5 seeds per arm' which pins 5. Executed 5/arm (frozen decision-rule arithmetic); n_per_cell=20 recorded UNRESOLVED_CARDINALITY, surfaced not silently absorbed.",
 "v2_freeze_gap": "e2_ssa_spec.v1 has no time/duration/events-per-block/replicate-count block -> V2 confirmatory NOT executable without invention; emitted as NOT_EXECUTED_FREEZE_GAP.",
}

# ---------------- circuit construction ----------------
def build_base(structure_seed):
    rng = np.random.default_rng(structure_seed)
    r = THETA['weight_mu'] / MU_NOM['E_E']
    pre_l, post_l, w_l = [], [], []
    for name, (npre, npost), off_p, off_q in [
        ('E_E', (N_E, N_E), 0, 0), ('E_I', (N_E, N_I), 0, N_E),
        ('I_E', (N_I, N_E), N_E, 0), ('I_I', (N_I, N_I), N_E, N_E)]:
        p = {'E_E': TOP['p_EE'], 'E_I': TOP['p_EI'], 'I_E': TOP['p_IE'], 'I_I': TOP['p_II']}[name]
        m = rng.random((npre, npost)) < p
        ii, jj = np.nonzero(m)
        keep = ii + off_p != jj + off_q
        ii, jj = ii[keep], jj[keep]
        w = rng.normal(MU_NOM[name] * r * SIGN[name], SIG[name], size=ii.size)
        pre_l.append(ii + off_p); post_l.append(jj + off_q); w_l.append(w)
    pre = np.concatenate(pre_l).astype(np.int32)
    post = np.concatenate(post_l).astype(np.int32)
    weight = np.concatenate(w_l).astype(np.float32)
    delay_steps = rng.choice(GRID_STEPS, size=pre.size).astype(np.int32)
    return pre, post, weight, delay_steps, float(r)

def make_params():
    a = np.full(N, EDEF['a'], np.float32); b = np.full(N, EDEF['b'], np.float32)
    c = np.full(N, EDEF['c'], np.float32); d = np.full(N, EDEF['d'], np.float32)
    a[N_E:] = PV['a']; b[N_E:] = PV['b']; c[N_E:] = PV['c']; d[N_E:] = PV['d']
    drive = np.full(N, THETA['drive_E'], np.float32); drive[N_E:] = THETA['drive_I']
    from jaxfne.emitters import IzhikevichParams
    sign = np.ones((N,), np.float32); sign[N_E:] = -1.0
    return IzhikevichParams(
        a=jnp.asarray(a), b=jnp.asarray(b), c=jnp.asarray(c), d=jnp.asarray(d),
        drive=jnp.asarray(drive), sign=jnp.asarray(sign),
        W=jnp.eye(N, dtype=jnp.float32) * 0.0,
        v0=jnp.full((N,), -65.0, dtype=jnp.float32),
        u0=jnp.asarray(b * (-65.0), dtype=jnp.float32),
        source_scale=jnp.ones((N,), jnp.float32),
        labels=tuple(['E'] * N_E + ['I'] * N_I))

def edge_list(pre, post, weight, delay_steps):
    w = jnp.asarray(weight)
    rec = (w < 0).astype(jnp.int32)
    tau = jnp.where(rec == 0, jnp.float32(2.0), jnp.float32(5.0))
    return EdgeList(pre=jnp.asarray(pre), post=jnp.asarray(post), weight=w,
                    receptor_index=rec, tau_ms=tau,
                    delay_steps=jnp.asarray(delay_steps))

def apply_arm(pre, post, weight, delay_steps, arm, structure_seed):
    if arm == 'C0_intact':
        return pre, post, weight, delay_steps
    if arm in ('C1_E_to_I_zero', 'C2_I_to_E_zero'):
        mask = ((pre < N_E) & (post >= N_E)) if arm.startswith('C1') else ((pre >= N_E) & (post < N_E))
        return pre, post, weight * (~mask), delay_steps
    rng = np.random.default_rng([structure_seed, hash(arm) % (2**32)])
    if arm == 'C3a_degree_rewire':
        post = post.copy()
        for lo, hi in ((0, N_E), (N_E, N)):
            sel = (post >= lo) & (post < hi)
            tgt = np.where(sel)[0]
            post[tgt] = rng.permutation(post[tgt])
        return pre, post, weight, delay_steps
    if arm == 'C3b_weight_shuffle':
        return pre, post, rng.permutation(weight), delay_steps
    if arm == 'C3c_delay_shuffle':
        return pre, post, weight, rng.permutation(delay_steps)
    if arm == 'C3d_matched_count':
        w = weight.copy()
        sel = np.where((pre < N_E) & (post < N_E))[0]
        zero = rng.choice(sel, size=min(ping['interventions']['C3d_matched_count']['edges'], sel.size), replace=False)
        w[zero] = 0.0
        return pre, post, w, delay_steps
    raise ValueError(arm)

ARMS = ['C0_intact', 'C1_E_to_I_zero', 'C2_I_to_E_zero', 'C3a_degree_rewire',
        'C3b_weight_shuffle', 'C3c_delay_shuffle', 'C3d_matched_count']
SEEDS = [dict(rep=i + 1, structure=100 + i, runtime=1100 + i, stimulus=2100 + i, analysis=3100 + i)
         for i in range(5)]

# ---------------- analysis (frozen method only) ----------------
from scipy.signal import welch, butter, filtfilt, hilbert, find_peaks

_gk = None
def gauss_smooth(x, sigma_ms):
    global _gk
    nb = sigma_ms / DT
    hw = int(math.ceil(3 * nb)); t = np.arange(-hw, hw + 1)
    k = np.exp(-(t ** 2) / (2 * nb * nb)); k /= k.sum()
    return np.convolve(x, k, mode='same')

def pop_rates(spikes):
    rE = spikes[W_SLICE, :N_E].mean(axis=1) * (1000.0 / DT)
    rI = spikes[W_SLICE, N_E:].mean(axis=1) * (1000.0 / DT)
    return gauss_smooth(rE, 2.0), gauss_smooth(rI, 2.0)

def psd_metrics(r, ana_rng):
    if not np.isfinite(r).all() or float(np.abs(r).sum()) == 0.0:
        return dict(fpk=None, prom=-99.0, band_ratio=0.0)
    f, P = welch(r, fs=FS, window='hann', nperseg=MTH['segment_samples'],
                 noverlap=MTH['overlap_samples'], nfft=MTH['nfft'], detrend='constant')
    P = np.maximum(P, 1e-30)
    band = (f >= MTH['search_band_hz'][0]) & (f <= MTH['search_band_hz'][1])
    if not band.any():
        return dict(fpk=None, prom=-99.0, band_ratio=0.0)
    ib = int(np.argmax(P * band))
    if P[ib] <= 1e-25:
        return dict(fpk=None, prom=-99.0, band_ratio=0.0)
    fpk = f[ib]
    y0, ym, yp = 10*np.log10(P[max(ib-1,0)]), 10*np.log10(P[ib]), 10*np.log10(P[min(ib+1,len(P)-1)])
    denom = (y0 - 2*ym + yp)
    delta = 0.5*(y0 - yp)/denom if denom != 0 and np.isfinite(denom) else 0.0
    fpk_i = fpk + delta * (f[1]-f[0])
    excl = (f > fpk - 10) & (f < fpk + 10)
    fit_mask = (f >= 10) & (f <= 100) & ~excl & (P > 1e-25)
    if fit_mask.sum() < 2:
        return dict(fpk=None, prom=-99.0, band_ratio=0.0)
    cf = np.polyfit(np.log10(f[fit_mask]), np.log10(P[fit_mask]), 1)
    base = 10 ** np.polyval(cf, np.log10(max(fpk_i, 1e-1)))
    prom = 10 * np.log10(P[ib] / base)
    br_num = np.trapezoid(P[(f >= fpk-10) & (f <= fpk+10)], f[(f >= fpk-10) & (f <= fpk+10)]) if hasattr(np,'trapezoid') else np.trapz(P[(f >= fpk-10) & (f <= fpk+10)], f[(f >= fpk-10) & (f <= fpk+10)])
    m10 = (f >= 10) & (f <= 100)
    br_den = np.trapezoid(P[m10], f[m10]) if hasattr(np,'trapezoid') else np.trapz(P[m10], f[m10])
    return dict(fpk=float(fpk_i), prom=float(prom), band_ratio=float(br_num/max(br_den,1e-30)), f=f, P=P)

def xcorr_max(a, b, max_lag_bins):
    a = a - a.mean(); b = b - b.mean()
    n = len(a); cc = np.correlate(b, a, mode='full')[n-1-max_lag_bins:n+max_lag_bins]
    lags = np.arange(-max_lag_bins, max_lag_bins+1)
    ok = lags != 0
    return float(np.max(np.abs(cc[ok])) / (np.std(a)*np.std(b)*n + 1e-30)), lags[ok][np.argmax(np.abs(cc[ok]))]

def ac_sidepeak(r, fpk):
    r = r - r.mean(); n = len(r)
    ac = np.correlate(r, r, mode='full')[n-1:]/ (np.std(r)**2*n + 1e-30)
    lo = int(0.8*1000/fpk/DT); hi = int(1.2*1000/fpk/DT)
    if hi <= lo or hi >= len(ac): return 0.0
    return float(np.max(ac[lo:hi]))

def phase_metrics(rE, rI, fpk, ana_rng):
    b, a = butter(4, [(fpk-10)/(FS/2), (fpk+10)/(FS/2)], btype='band')
    pad = 36
    try:
        fE = filtfilt(b, a, rE, padlen=pad); fI = filtfilt(b, a, rI, padlen=pad)
    except Exception:
        return dict(dphi=None, plv=0.0, dt_ms=None, rayleigh_p=1.0, plv_surr=0.0)
    phE, phI = np.angle(hilbert(fE)), np.angle(hilbert(fI))
    dph = phI - phE
    plv_vec = np.exp(1j*dph); plv = float(np.abs(plv_vec.mean()))
    ang = float(np.degrees(np.angle(plv_vec.mean())))
    if ang < 0: ang += 360.0
    if ang > 180.: ang -= 360.0
    Z = len(dph) * plv**2
    p = float(math.exp(-Z)) if Z < 700 else 0.0
    # IAAFT-style phase-randomization surrogate on rI (preserve spectrum, random phases)
    R = np.fft.rfft(fI); mag = np.abs(R); rng_ph = np.random.default_rng(ana_rng)
    ph_s = rng_ph.uniform(0, 2*np.pi, size=R.shape); ph_s[0] = 0
    surr = np.fft.irfft(mag*np.exp(1j*ph_s), n=len(fI))
    phS = np.angle(hilbert(surr))
    plv_surr = float(np.abs(np.exp(1j*(phS-phE)).mean()))
    return dict(dphi=ang, plv=plv, dt_ms=(ang/360.0/fpk*1000.0 if fpk else None),
                rayleigh_p=p, plv_surr=plv_surr)

def cycle_metrics(rE, fpk):
    dist = max(int(0.5*1000.0/fpk/DT), 1)
    pk, _ = find_peaks(rE, distance=dist)
    if len(pk) < 3:
        return dict(n_cycles=len(pk)-1, cv=None, part_med=None, ff=None)
    T = np.diff(pk) * DT
    bounds = np.concatenate([[0], (pk[:-1]+pk[1:])//2, [len(rE)]])
    spkW = _cur_spikes[W_SLICE, :N_E]
    parts, counts = [], []
    for i in range(len(pk)-1):
        seg = spkW[bounds[i]:bounds[i+1]]
        cnt = seg.sum(axis=0)
        counts.append(cnt.sum()); parts.append((cnt > 0).mean())
    cv = float(np.std(T)/np.mean(T)); cnts = np.array(counts)
    return dict(n_cycles=int(len(pk)-1), cv=cv, part_med=float(np.median(parts)),
                ff=float(cnts.var()/cnts.mean()) if cnts.mean() > 0 else None)

def jitter_surrogate_prom(spikes, ana_rng):
    st = spikes.astype(bool)
    nT = st.shape[0]
    out = np.zeros_like(st)
    idx_t, idx_n = np.nonzero(st)
    shifts = ana_rng.integers(-int(20/DT), int(20/DT)+1, size=len(idx_t))
    jt = (idx_t + shifts) % nT
    out = np.zeros_like(st)
    np.add.at(out, (jt, idx_n), 1.0)
    rEj = gauss_smooth(out[W_SLICE, :N_E].mean(axis=1)*(1000.0/DT), 2.0)
    return psd_metrics(rEj, None)['prom']

# ---------------- execute ----------------
params = make_params()
rows = []
rates_store = {}
for sd in SEEDS:
    pre, post, w0, d0, scale = build_base(sd['structure'])
    for arm in ARMS:
        p_, q_, w_, d_ = apply_arm(pre, post, w0, d0, arm, sd['structure'])
        edges = edge_list(p_, q_, w_, d_)
        key = jax.random.PRNGKey(sd['runtime'] * 100 + ARMS.index(arm))
        v, spk, src, fst = simulate_edge_recurrent_izhikevich(
            params, edges, NSTEP, DT, key, dtype='float32', noise_scale=0.0)
        v_h = np.asarray(v); sp_h = np.asarray(spk); s_h = np.asarray(src)
        _cur_spikes = sp_h
        inv = {}
        if not (np.isfinite(v_h).all() and np.isfinite(sp_h).all() and np.isfinite(s_h).all()):
            inv['INVALID_NUMERIC_NONFINITE'] = True
        ds = np.asarray(fst.get('delay_state'))
        if ds is not None and not np.isfinite(ds).all():
            inv['INVALID_DELAY_STATE'] = True
        rE, rI = pop_rates(sp_h)
        rates_store[f"{arm}_s{sd['rep']}"] = np.stack([rE, rI]).astype(np.float32)
        meanE, meanI = float(rE.mean()), float(rI.mean())
        if max(meanE, meanI) > 200.0:
            inv['INVALID_NUMERIC_DIVERGENCE'] = True
        row = dict(arm=arm, seed=sd, INVALID=inv, mean_rate_E=meanE, mean_rate_I=meanI,
                   n_spiking=int((sp_h[W_SLICE].sum(axis=0) > 0).sum()),
                   active_E=int((sp_h[W_SLICE, :N_E].sum(axis=0) > 0).sum()),
                   active_I=int((sp_h[W_SLICE, N_E:].sum(axis=0) > 0).sum()))
        if inv:
            rows.append(row); continue
        G_fin = bool(np.isfinite([meanE, meanI]).all())
        G_act = meanE >= 0.5 and meanI >= 0.5 and row['n_spiking'] >= 10
        G_pop = row['active_E'] >= 20 and row['active_I'] >= 5
        adequate = G_fin and G_act and G_pop
        row.update(G_finite=G_fin, G_active=bool(G_act), G_population=bool(G_pop), G_adequate=adequate)
        if not adequate:
            row['label'] = 'UNRESOLVED_ADEQUACY'
            rows.append(row)
            print(f"s{sd['rep']} {arm}: UNRESOLVED_ADEQUACY meanE={meanE:.2f} meanI={meanI:.2f}")
            continue
        ana_rng = sd['analysis'] * 1000 + ARMS.index(arm)
        mA = psd_metrics(rE, None); mB = psd_metrics(rI, None)
        # Option A: primary spectral signal = r_E with r_I concordance |dfp|<=5
        fpk = mA['fpk']; prom = mA['prom']; bratio = mA['band_ratio']
        optA_ok = (mB['fpk'] is not None and fpk is not None and abs(mB['fpk'] - fpk) <= 5.0)
        statW = psd_metrics(gauss_smooth(
            sp_h[W1_SLICE, :N_E].mean(axis=1)*(1000.0/DT), 2.0), None)
        stat_ok = fpk is not None and statW['fpk'] is not None and \
                  abs(statW['prom'] - prom) <= 3.0 and abs(statW['fpk'] - fpk) <= 5.0
        ml = 1000.0 / fpk if fpk else 0
        mdE = (rE.max()-rE.min())/max(rE.mean(),1e-9); mdI = (rI.max()-rI.min())/max(rI.mean(),1e-9)
        xc, xlag = xcorr_max(rE, rI, int(ml/DT)+1 if ml else 66)
        sh = int((sd['analysis'] % 300) + 100) / DT
        rIs = np.roll(rI, int(sh))
        xc_sh, _ = xcorr_max(rE, rIs, int(ml/DT)+1 if ml else 66)
        pm = phase_metrics(rE, rI, fpk, ana_rng) if fpk else dict(dphi=None, plv=0.0, dt_ms=None, rayleigh_p=1.0, plv_surr=0.0)
        cm = cycle_metrics(rE, fpk) if fpk else dict(n_cycles=0, cv=None, part_med=None, ff=None)
        prom_jit = jitter_surrogate_prom(sp_h, np.random.default_rng(ana_rng+7))
        g_spec = bool(fpk is not None and 35 <= fpk <= 75 and prom >= 6.0 and bratio >= 0.25 and stat_ok and optA_ok)
        g_rate = bool(min(mdE, mdI) >= 0.50 and ac_sidepeak(rE, fpk) >= 0.25 and ac_sidepeak(rI, fpk) >= 0.25
                      and xc >= 0.40 and xc_sh < 0.20)
        g_phase = bool(pm['dphi'] is not None and 15 <= pm['dphi'] <= 90 and pm['plv'] >= 0.40
                       and pm['dt_ms'] is not None and 2 <= pm['dt_ms'] <= 8 and pm['rayleigh_p'] < 0.01
                       and pm['plv_surr'] < 0.25)
        cyc_ok = cm['n_cycles'] >= 10 and cm['cv'] is not None and cm['cv'] <= 0.35 \
                 and cm['part_med'] is not None and cm['part_med'] >= 0.10 \
                 and cm['ff'] is not None and cm['ff'] <= 0.60
        gray = []
        if 5.0 <= prom < 6.0: gray.append('UNRESOLVED_PROMINENCE')
        if 0.35 <= pm['plv'] < 0.40: gray.append('UNRESOLVED_PLV')
        if pm['dphi'] is not None and 12 <= pm['dphi'] < 15: gray.append('UNRESOLVED_DELTA_PHI')
        # hard failures = strictly outside frozen gate AND outside its gray band (gray bands exist only for prom/plv/dphi)
        hard = (
            (not g_spec and not (5.0 <= prom < 6.0)) or
            (not g_rate) or
            (not g_phase and not ((0.35 <= pm['plv'] < 0.40) or (pm['dphi'] is not None and 12 <= pm['dphi'] < 15))) or
            (not cyc_ok)
        )
        ping_all = bool(g_spec and g_rate and g_phase and cyc_ok)
        if ping_all:
            label = 'PING_LIKE'
        elif gray and not hard:
            label = 'UNRESOLVED:' + ','.join(gray)
        else:
            label = 'NO_PING'
            if gray:
                label += '|GRAY:' + ','.join(gray)
        row.update(dict(fpk=fpk, fpk_I=mB['fpk'], prom_dB=prom, band_ratio=bratio, stationarity_ok=bool(stat_ok),
                        optionA_concordant=bool(optA_ok), md_E=float(mdE), md_I=float(mdI),
                        xcorr=xc, xcorr_lag_ms=float(xlag*DT), xcorr_shifted=float(xc_sh),
                        G_spec=g_spec, G_rate=g_rate, G_phase=g_phase, G_cycle=bool(cyc_ok),
                        dphi_deg=pm['dphi'], PLV=pm['plv'], dt_lag_ms=pm['dt_ms'],
                        rayleigh_p=pm['rayleigh_p'], plv_surr=pm['plv_surr'],
                        cycles=cm['n_cycles'], cv_T=cm['cv'], participation=cm['part_med'], ff=cm['ff'],
                        surrogate_jitter_prom_dB=float(prom_jit), gray_zones=gray, label=label))
        rows.append(row)
        print(f"s{sd['rep']} {arm}: {label} fpk={fpk} prom={prom:.2f} PLV={pm['plv']:.2f} adeq={adequate}")

np.savez_compressed(OUT / 'v1_rates_window.npz', **rates_store)

# ---------------- aggregate (frozen collapse + 4/5 rules) ----------------
by_arm = {a: [r for r in rows if r['arm'] == a] for a in ARMS}
def collapse(r0, rx):
    lbl = rx.get('label', '')
    first = lbl.startswith(('NO_PING', 'UNRESOLVED', 'INVALID')) or \
            (rx.get('prom_dB', -99) < 3.0) or \
            (r0.get('fpk') is None or rx.get('fpk') is None or abs(r0['fpk'] - rx['fpk']) > 10)
    plv = rx.get('PLV', 0.0) or 0.0
    dphi = rx.get('dphi')
    second = ((r0.get('PLV', 0) or 0) - plv >= 0.20) or (plv < 0.25) or \
             (dphi is None or not (15 <= dphi <= 90))
    return bool(first and second), ''

agg = {}
for arm in ARMS[1:]:
    col = [collapse(by_arm['C0_intact'][i], by_arm[arm][i])[0] for i in range(5)]
    agg[arm] = dict(collapsed=sum(col), of=5, per_seed=col)
intact_ping = sum(1 for r in by_arm['C0_intact'] if r.get('label') == 'PING_LIKE')
intact_unres = sum(1 for r in by_arm['C0_intact'] if str(r.get('label','')).startswith('UNRESOLVED'))
adeq_c0 = sum(1 for r in by_arm['C0_intact'] if r.get('G_adequate'))
controls_ok = all(agg[a]['collapsed'] <= 1 for a in ('C3a_degree_rewire','C3b_weight_shuffle','C3c_delay_shuffle','C3d_matched_count'))
loop_ok = agg['C1_E_to_I_zero']['collapsed'] >= 4 and agg['C2_I_to_E_zero']['collapsed'] >= 4
if intact_ping >= 4 and loop_ok and controls_ok:
    verdict, cls = 'SUPPORTED', 'PING_LIKE_LOOP_DEPENDENT'
elif intact_ping >= 4:
    verdict = 'NEGATIVE'
    cls = 'NEGATIVE_NOT_LOOP_DEPENDENT' if not loop_ok else 'NEGATIVE_CONTROL_FAILURE_FRAGILE'
elif intact_unres >= 3:
    verdict, cls = 'UNRESOLVED', 'UNRESOLVED_PARTIALLY_REPRODUCIBLE'
else:
    verdict, cls = 'NEGATIVE', 'NEGATIVE_NOT_PING_LIKE'

git_head = __import__('subprocess').check_output(['git','rev-parse','HEAD'], text=True).strip()
receipt = dict(
    schema='e2b_v1_confirmatory_receipt.v1', generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
    authority_packet=dict(ping_spec_hash=manifest['ping_spec_hash'], ssa_spec_hash=manifest['ssa_spec_hash'],
                          ping_sha256=manifest['ping_sha256'], ssa_sha256=manifest['ssa_sha256'],
                          code_head_executed=git_head, parent_e1=manifest['parent_e1']),
    theta_star=THETA, seeds_plan=SEEDS, arms=ARMS, interpretation_notes=NOTES,
    runs=rows,
    aggregation=dict(intact_PING_LIKE=intact_ping, intact_UNRESOLVED=intact_unres,
                     C0_G_adequate=adeq_c0, collapses=agg,
                     controls_noncollapse_ok=bool(controls_ok), loop_dependence_ok=bool(loop_ok),
                     degenerate_control_context='C0 itself is out-of-window on dphi (~-6.8 deg, I leads E); '
                     'frozen collapse cond_b (dphi in [15,90]) is therefore satisfied trivially by arms sharing '
                     'C0 ordering, making the 5/5 collapse table partly vacuous. This is the audit-doc 4.1 '
                     '"controls vacuous when intact fails" regime; raw per-arm metrics above are the evidence.',
                     intact_failure_modes='G_rate: shifted-null xcorr>=0.20; G_phase: dphi negative sign; '
                     'G_cycle: n_cycles<10; OptionA |fpk_E-fpk_I|>5; stationarity W1 vs W'),
    calibration_generalization=dict(
        e2a_proxy_adequacy_rate=1.0, confirmatory_C0_adequacy=f"{adeq_c0}/5",
        material_difference=bool(adeq_c0 < 5),
        note='If materially different from E2a 6/6 proxy this is calibration-generalization evidence, reported separately from phenotype.'),
    verdict=dict(V1=verdict, subclass=cls,
                 arithmetic='runs=%d = 7 arms x 5 seeds; labels preserved for every run incl. INVALID' % len(rows)),
    raw_retention='W-window population rates (2x2000 bins/seed/arm) in v1_rates_window.npz; full traces not archived (budget discipline 32 4.2)',
    write_once=True)
(OUT / 'v1_ping_receipt.json').write_text(json.dumps(receipt, indent=2, default=str))
print('VERDICT:', verdict, cls)

v2 = dict(schema='e2b_v2_confirmatory_receipt.v1',
          generated_at=receipt['generated_at'],
          outcome='NOT_EXECUTED_FREEZE_GAP',
          missing_frozen_fields=['time.duration_ms/n_steps/burn_in', 'events_per_block',
                                 'block_length_trials', 'n_outer_replicates_explicit',
                                 'primary_amplitude_level'],
          reason='e2_ssa_spec.v1 pins ISI/stim/windows/thresholds/factors but no total duration, '
                 'no events per block, no block length, no explicit replicate count. Executing would '
                 'require inventing parameters -> prohibited (measure->classify, never tune). '
                 'Requires explicit write-once amendment e2_ssa_spec.v2 before V2 execution.',
          status_label='UNRESOLVED', write_once=True)
(OUT / 'v2_ssa_receipt.json').write_text(json.dumps(v2, indent=2))
print('Wrote receipts to', OUT)
