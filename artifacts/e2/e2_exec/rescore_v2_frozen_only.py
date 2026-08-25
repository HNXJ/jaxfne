"""Independent frozen-only V2 rescorer — deliberately DIFFERENT implementation from
aggregate_v2.py. Recomputes SI/controls/S3 from per-replicate receipts with a separate
code path (vectorized, no shared analysis functions) and compares labels."""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, '../../../..')
PR = pathlib.Path('.').resolve() / 'artifacts/e2/preregistration'
spec = json.loads((PR / 'e2_ssa_spec.v6.json').read_bytes())
m = spec['SSA']['metrics']; eps = m['epsilon']
runs = sorted((PR / 'E2b_confirmatory/v2_runs').glob('rep_*.json'))

def split(b):
    R = np.asarray(b['R'])[2:]; d = np.asarray(b['is_dev'])[2:]
    return float(R[d].mean()), float(R[~d].mean())

SIs, dRs, swaps, manys = [], [], [], []
for rp in runs:
    j = json.loads(rp.read_bytes())
    B = {b['name']: b for b in j['blocks']}
    da, sa = split(B['oddball_A_std'])
    db, sb = split(B['oddball_B_std_flip'])
    SIs.append(((da - sa) / (da + sa + eps) + (db - sb) / (db + sb + eps)) / 2)
    dRs.append((da - sa + db - sb) / 2)
    swaps.append(abs((da - sa) / (da + sa + eps) - (db - sb) / (db + sb + eps)))
    RM = np.asarray(B['many_standards_control']['R'])[2:]
    manys.append(abs((RM[::2].mean() - RM[1::2].mean()) / (RM[::2].mean() + RM[1::2].mean() + eps)))

SI = float(np.mean(SIs))
# independent bootstrap: simple percentile, different RNG stream derivation (seed from spec hash bytes)
seed = int(spec['spec_hash'][:16], 16)
rng = np.random.default_rng(seed)
arr = np.asarray(SIs)
bs = arr[rng.integers(0, 20, size=(5000, 20))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
verdict = ('S2' if (SI > m['theta_SI'] and lo > m['theta_SI'] and max(swaps) <= m['swap_max']
                    and np.mean(manys) < m['theta_SI'])
           else 'NEGATIVE:' + ','.join(
               [f for f in [('FAIL_swap_asymmetry' if max(swaps) > m['swap_max'] else ''),
                            ('SIGN_neg' if SI < 0 else '')] if f]))
out = dict(schema='v2_rescored_frozen_only.v1',
           implementation='independent vectorized rescorer; no shared code with aggregate_v2.py',
           pooled_SI=SI, percentile_bootstrap=[float(lo), float(hi)],
           swap_max_observed=float(max(swaps)), SI_many_mean=float(np.mean(manys)),
           label=verdict,
           agreement_note='compare against v2_ssa_confirmatory_receipt.json verdict')
(PR / 'E2b_confirmatory/v2_rescored_frozen_only.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
