"""Independent frozen-only V1 rescorer — deliberately DIFFERENT implementation from
e2b_v1_executor.py (rule: receipt code and independent rescoring never share code).

Reads ONLY v1_rates_window.npz + frozen e2_ping_prereg.json. Recomputes the four
classifier gates with zero executor-added conjuncts, per-seed, and emits a table.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
import numpy as np
from scipy.signal import welch

ROOT = pathlib.Path('.').resolve()
PR = ROOT / 'artifacts/e2/preregistration'
ping = json.loads((PR / 'e2_ping_prereg.json').read_bytes())
import e2_exec_lib as lib  # noqa: E402

gates = lib.PingGates(ping)
z = np.load(PR / 'E2b_confirmatory/v1_rates_window.npz')
FS = ping['method']['sampling_rate_hz']
NPER = ping['method']['segment_samples']; NOV = ping['method']['overlap_samples']
NF = ping['method']['nfft']
SB = ping['method']['search_band_hz']; TB = ping['method']['ping_classification_band_hz']
DT = ping['time']['dt_ms']

def band_metrics(r):
    f, P = welch(r, fs=FS, nperseg=NPER, noverlap=NOV, nfft=NF, window='hann', detrend='constant')
    P = np.maximum(P, 1e-30)
    sel = (f >= SB[0]) & (f <= SB[1])
    ib = int(np.argmax(P * sel))
    fpk = float(f[ib])
    # prominence vs log-log fit, excluding +-10 Hz around peak, 10-100 Hz
    excl = (f > fpk - 10) & (f < fpk + 10)
    m = (f >= 10) & (f <= 100) & ~excl & (P > 1e-25)
    if m.sum() < 2:
        return None
    cf = np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)
    base = 10 ** np.polyval(cf, np.log10(fpk))
    prom = float(10 * np.log10(P[ib] / base))
    num = np.trapezoid(P[(f >= fpk - 10) & (f <= fpk + 10)], f[(f >= fpk - 10) & (f <= fpk + 10)])
    den = np.trapezoid(P[(f >= 10) & (f <= 100)], f[(f >= 10) & (f <= 100)])
    return dict(fpk=fpk, prom_dB=prom, band_ratio=float(num / max(den, 1e-30)))

def ac_sidepeak(r, fpk):
    r = r - r.mean(); n = len(r)
    ac = np.correlate(r, r, 'full')[n - 1:] / (r.std() ** 2 * n + 1e-30)
    lo = int(0.8 * 1000.0 / fpk); hi = int(1.2 * 1000.0 / fpk)
    hi = min(hi, len(ac) - 1)
    return float(np.max(ac[lo:hi])) if hi > lo else 0.0

def xcorr_abs(a, b, fpk):
    a = a - a.mean(); b = b - b.mean(); n = len(a)
    maxlag = max(int(1000.0 / fpk), 1)
    cc = np.correlate(b, a, 'full')[n - 1 - maxlag:n + maxlag]
    return float(np.max(np.abs(cc)) / (a.std() * b.std() * n + 1e-30))

def phase(rE, rI, fpk):
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(4, [(fpk - 10) / (FS / 2), (fpk + 10) / (FS / 2)], btype='band')
    pE = np.angle(hilbert(filtfilt(b, a, rE, padlen=36)))
    pI = np.angle(hilbert(filtfilt(b, a, rI, padlen=36)))
    dph = pI - pE
    plv = float(np.abs(np.exp(1j * dph).mean()))
    ang = float(np.degrees(np.angle(np.exp(1j * dph).mean())))
    if ang < 0: ang += 360.0
    if ang > 180: ang -= 360.0
    import math
    Z = len(dph) * plv ** 2
    return dict(dphi_deg=ang, plv=plv, dt_lag_ms=(ang / 360.0 / fpk * 1000.0),
                rayleigh_p=(float(math.exp(-Z)) if Z < 700 else 0.0))

def cycles(rE, fpk):
    from scipy.signal import find_peaks
    dist = max(int(0.5 * 1000.0 / fpk / DT), 1)
    pk, _ = find_peaks(rE, distance=dist)
    T = np.diff(pk) * DT
    spk = None  # participation needs spikes; npz has rates only -> gate fails closed on missing metrics
    return dict(n_cycles=int(max(len(pk) - 1, 0)), cv_T=float(np.std(T) / np.mean(T)) if len(T) > 1 else None,
                part_med=None, ff=None)

rows = []
for key in sorted(z.files):
    arm, _, seed = key.rpartition('_s')
    rE, rI = z[key]
    bmE, bmI = band_metrics(rE), band_metrics(rI)
    if bmE is None or bmI is None:
        rows.append(dict(key=key, arm=arm, seed=int(seed), label='INSUFFICIENT_SIGNAL')); continue
    fpk = bmE['fpk']
    vals = dict(**bmE, md_min=min((rE.max()-rE.min())/max(rE.mean(),1e-9), (rI.max()-rI.min())/max(rI.mean(),1e-9)),
                ac_min=min(ac_sidepeak(rE, fpk), ac_sidepeak(rI, fpk)),
                xcorr_abs=xcorr_abs(rE, rI, fpk))
    vals.update(phase(rE, rI, fpk)); vals.update(cycles(rE, fpk))
    g = dict(G_spec=gates.spec(vals), G_rate=gates.rate(vals), G_phase=gates.phase(vals), G_cycle=gates.cycle(vals))
    gray = []
    if gates.gray_prom[0] <= vals['prom_dB'] < gates.gray_prom[1]: gray.append('PROM')
    if gates.gray_plv[0] <= vals['plv'] < gates.gray_plv[1]: gray.append('PLV')
    hard_fail = any((not v) for v in g.values())
    label = 'PING_LIKE' if all(g.values()) else ('UNRESOLVED:' + ','.join(gray) if (gray and not hard_fail) else 'NO_PING')
    rows.append(dict(key=key, arm=arm, seed=int(seed), label=label,
                     gates=g, fpk_E=vals['fpk'], prom_dB=round(vals['prom_dB'], 3),
                     dphi_deg=round(vals['dphi_deg'], 3), plv=round(vals['plv'], 4),
                     ac_min=round(vals['ac_min'], 4), n_cycles=vals['n_cycles']))

intact = [r for r in rows if r['arm'] == 'C0_intact']
summary = dict(
    schema='v1_rescored_frozen_only.v1',
    implementation='independent rescorer (artifacts/e2/e2_exec/), shares no analysis code with e2b_v1_executor.py',
    conjuncts='exactly the four frozen classifier strings; no added criteria',
    intact_labels=[r['label'] for r in intact],
    intact_ping_like=sum(1 for r in intact if r.get('label') == 'PING_LIKE'),
    all_rows=rows)
out = PR / 'E2b_confirmatory/v1_rescored_frozen_only.json'
out.write_text(json.dumps(summary, indent=2, default=str))
print('intact PING_LIKE (frozen-only, independent):', summary['intact_ping_like'], '/5')
for r in intact:
    print(r['key'], r['label'], {k: r.get(k) for k in ('prom_dB', 'dphi_deg', 'ac_min', 'n_cycles')})
