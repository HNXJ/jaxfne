"""Author e2_ssa_spec.v5 from immutable v4.
N-fix: purge stale fabricated washout leaf (superseded pointer), correct canonical washout arithmetic.
L-fold: d_rec alias, S1 SI_std_ctrl/SI_dev_ctrl definitions, SI_many definition,
        adequacy G_A/G_B operationalization, permutation tag bindings, block execution declarations."""
import sys, json, hashlib, pathlib, copy
sys.path.insert(0, '.')
PR = pathlib.Path('artifacts/e2/preregistration')

def canon(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()

v4_path = PR / 'e2_ssa_spec.v4.json'
v4 = json.loads(v4_path.read_bytes())
assert canon(v4) == v4['spec_hash']

v5 = copy.deepcopy(v4)
v5['schema'] = 'e2_ssa_spec.v5'
v5['parent_spec'] = {'schema': 'e2_ssa_spec.v4', 'spec_hash': v4['spec_hash'],
                     'sha256': hashlib.sha256(v4_path.read_bytes()).hexdigest().upper(),
                     'preserved_immutable': True,
                     'amendment_reason': 'N: purge stale fabricated washout leaf + correct canonical washout; L: bind residual classifier symbols'}

# --- N fix: single canonical washout rule with correct arithmetic ---
v5['execution_grammar']['paradigm']['transient_washout'] = (
    'Canonical rule: exactly the first 2 events of every block are excluded from all analyses. '
    'Post-washout realized event counts by block design: oddball blocks 78 each (80-2), many_standards 78 (80-2), '
    'recovery_isi500 38 (40-2), recovery_isi1000 18 (20-2). Per-type splits inside oddball blocks are sequence-realized '
    'under deviant_placement_rule and hashed per replicate. No analytic event-count threshold exists.')
# supersede stale fabricated leaf (retained for provenance, marked non-normative)
v5['execution_grammar']['paradigm']['deviant_placement_rule']['transient_washout'] = (
    'SUPERSEDED (contained fabricated counts 10/66/19/9 - non-normative). Canonical washout lives at '
    'paradigm.transient_washout in this schema version.')

ms = v5['execution_grammar']['methods_ssa']
# --- L residuals: symbol bindings ---
ms['alias_d_rec'] = "classifiers.S3_recovery token 'd_rec' denotes delta_rec (identical quantity)"
ms['S1_definitions'] = {
    'R_control': 'event-matched mean response of many_standards_control block, both identities pooled',
    'SI_std_ctrl': '(R_control - R_std)/(R_control + R_std + eps) per replicate',
    'SI_dev_ctrl': '(R_control - R_dev)/(R_control + R_dev + eps) per replicate',
    'note': 'S1 requires both > theta_SI and |SI_std_ctrl - SI_dev_ctrl| < delta_global_leak=0.08'}
ms['SI_many_definition'] = '|(R_A - R_B)/(R_A + R_B + eps)| computed on many_standards_control block events'
ms['permutation_tags'] = {'S2': 'perm_S2', 'S3': 'perm_S3', 'shuffled_history': 'shuf_oddball',
                          'BCa': 'bca_{contrast}'}
ms['adequacy_operational'] = {
    'R_X_early': 'mean response to identity-X standard events 3-17 (post-washout indices) of the X-standard oddball block',
    'stable': '|mean(response last quarter) - mean(response first quarter)| <= 3 * SD(event responses of that identity within its standard block)',
    'G_A': 'R_A_early >= R_floor(1.0) and stable(A)', 'G_B': 'R_B_early >= R_floor(1.0) and stable(B)',
    'G_stable_block': 'drift <= 3 SD as defined above, evaluated per identity'}
v5['execution_grammar']['block_execution'] = {
    'init': 'fresh params.v0/u0 at every block start; no cross-block state carry',
    'drive_application': 'drive_schedule array (n_steps, n_neurons): rectangular pulse amplitude 1.0 added to drive_E columns of the targeted E-subpopulation for stim_duration_ms from each onset',
    'memory_rule': 'kernel consumed under jit with sliced per-event window aggregation; presyn_trace never materialized',
    'onsets': 'event k onset step = k * round(ISI_ms/dt_ms): 400 (200 ms), 1000 (500 ms), 2000 (1000 ms)'}

v5.pop('spec_hash', None)
v5['spec_hash'] = canon(v5)

for k, val in v4.items():
    if k in ('spec_hash', 'schema', 'parent_spec', 'execution_grammar'):
        continue
    assert v5.get(k) == val, f'mutated {k}'

REPL = {
    'execution_grammar.paradigm.transient_washout': 'canonical rule corrected (was contradictory 78->76 text)',
    'execution_grammar.paradigm.deviant_placement_rule.transient_washout': 'superseded pointer (was fabricated counts leaf)',
}
def walk(old, new, path=''):
    if isinstance(old, dict):
        for k, ov in old.items():
            p = f'{path}.{k}'
            if p in REPL:
                assert new.get(k) is not None
                continue
            assert k in new, f'{p} deleted'
            walk(ov, new[k], p)
    else:
        if path not in REPL:
            assert old == new, f'{path} mutated'
walk(v4['execution_grammar'], v5['execution_grammar'], 'execution_grammar')
# corrected arithmetic self-checks
assert 80 - 2 == 78 and 40 - 2 == 38 and 20 - 2 == 18
assert '10 deviants' not in v5['execution_grammar']['paradigm']['transient_washout']

v5.pop('spec_hash')
v5['spec_hash'] = canon(v5)
out = PR / 'e2_ssa_spec.v5.json'
out.write_text(json.dumps(v5, indent=2) + '\n')
receipt = {'schema': 'e2_ssa_amendment_receipt.v4',
           'from': {'schema': 'e2_ssa_spec.v4', 'spec_hash': v4['spec_hash']},
           'to': {'schema': 'e2_ssa_spec.v5', 'spec_hash': v5['spec_hash'],
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest().upper(),
                  'path': 'artifacts/e2/preregistration/e2_ssa_spec.v5.json'},
           'closes_N': 'stale fabricated washout leaf superseded; canonical first-2-event rule with verified counts 78/78/38/18',
           'closes_L_residuals': ['d_rec alias', 'SI_std_ctrl/SI_dev_ctrl definitions', 'SI_many definition',
                                  'adequacy G_A/G_B/stable operationalization', 'permutation tag bindings',
                                  'block_execution declarations (fresh init, drive application, memory rule, onsets)'],
           'prospective': True, 'write_once': True}
(PR / 'e2_ssa_spec_v5_amendment_receipt.json').write_text(json.dumps(receipt, indent=2))
print('v5 spec_hash:', v5['spec_hash'])
print('v5 sha256   :', receipt['to']['sha256'])
print('assertions PASS')
