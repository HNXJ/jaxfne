"""Author e2_ssa_spec.v3 from immutable v2: closes LFNI-audit blockers B1-B3 + S3 formulas.
Still prospective: zero SSA trajectories exist. All v2 fields byte-equal except schema/spec_hash."""
import sys, json, hashlib, pathlib, copy
sys.path.insert(0, '.')
PR = pathlib.Path('artifacts/e2/preregistration')
import e2_exec_lib  # noqa

def canon(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()

v2_path = PR / 'e2_ssa_spec.v2.json'
v2 = json.loads(v2_path.read_bytes())
assert canon(v2) == v2['spec_hash']
e2a = json.loads((PR / 'E2a_search/e2a_search_receipt.json').read_bytes())
THETA = e2a['result']['theta*']

v3 = copy.deepcopy(v2)
v3['schema'] = 'e2_ssa_spec.v3'
v3['parent_spec'] = {'schema': 'e2_ssa_spec.v2', 'spec_hash': v2['spec_hash'],
                     'sha256': hashlib.sha256(v2_path.read_bytes()).hexdigest().upper(),
                     'preserved_immutable': True,
                     'amendment_reason': 'closes LFNI blockers B1 theta-pin, B2 deviant placement, B3/S3 estimator definitions'}

eg = v3['execution_grammar']
# B1: pin operating point by inheritance from write-once E2a receipt (not tuning: theta* already frozen there)
eg['operating_point'] = {
    'inherited_from': 'artifacts/e2/preregistration/E2a_search/e2a_search_receipt.json result.theta* (write-once)',
    'drive_E': THETA['drive_E'], 'drive_I': THETA['drive_I'],
    'weight_mu': THETA['weight_mu'], 'noise_scale': THETA['noise_scale'],
    'W_ms_note': 'W_ms=60 is an E2a candidate-window knob, unused by SSA (H7 compatibility-only here)',
    'interpretation_note': 'six-way adequacy tie in E2a; lexicographic tie-break selection, NOT unique optimum'}
# B2: deviant placement rule
eg['paradigm']['deviant_placement_rule'] = {
    'constraint': 'no two consecutive deviants; at least 2 standards between consecutive deviants',
    'realization': 'positions sampled without replacement from legal slots using fold_in-chained K_stimulus key child_seed(tag="devpos_block{name}_rep{idx}"); full sequences hashed into receipt',
    'transient_washout': 'first 2 events of each block excluded from all analyses (post-exclusion counts: 10 deviants / 66 standards oddball; 38 many-std; recovery 19 & 9)'}
# B3 + S3: methods/estimator block
eg['methods_ssa'] = {
    'population_R': 'mean spikes per neuron over [onset+30, onset+110) ms across all 800 E neurons, divided by 0.08 s (Hz); co-primary SI_source uses mean |sources| over same window',
    'SI_form': '(R_dev - R_std)/(R_dev + R_std + 1e-9) computed per replicate (event-averaged), then aggregated',
    'uncertainty': {'method': 'BCa bootstrap over 20 outer replicates',
                    'draws': 5000, 'alpha': 0.05,
                    'key': 'child_seed(fold_in-chained K_analysis, tag="bca_{contrast}")'},
    'permutation': {'scheme': 'within-replicate std/dev label shuffle preserving counts; pooled statistic recomputed',
                    'draws': 2000, 'one_sided_p': '(1 + #{perm >= observed})/(2001)',
                    'key': 'child_seed(fold_in-chained K_analysis, tag="perm_{contrast}")'},
    'shuffled_history_control': {'draws': 2000, 'key_tag': 'shuf_oddball'},
    'S1_symbols': {'R_control': 'event-matched mean response of many_standards_control block (both identities pooled)'},
    'S3_recovery_definition': {
        'R_late': 'mean response to A-standards in events 65-80 of oddball_A_std block (short-ISI late)',
        'R_early': 'mean response to A-standards in events 3-17 of oddball_A_std block (post-washout early)',
        'R_rec(ISI)': 'mean response to A-standards in recovery_isi{ISI} blocks',
        'delta_rec': 'R_rec(500) - R_late',
        'I_rec': '(R_rec(500) - R_late)/(R_early - R_late), undefined -> UNRESOLVED if R_early <= R_late',
        'rho': 'Spearman over pooled per-event recovery responses (both ISIs, all replicates) vs their ISI in {500,1000}'},
    'new_block_recovery_isi1000': {'standard_only': 'A', 'isi_ms': 1000, 'events': 20},
    'implementation_cautions_adopted': [
        'kernel outputs consumed under jit/sliced aggregation; never materialize presyn_trace (10-13 GB risk)',
        'n_events<8 exclusion NOT implemented (external-packet construct; design counts are fixed)']}
# steps arithmetic note
eg['time']['steps_per_replicate_derived'] = '3x80x400 + 40x1000 + 20x2000 = 156000 steps (78 s biological)'
v3.pop('spec_hash', None)
v3['spec_hash'] = canon(v3)

for k, val in v2.items():
    if k in ('spec_hash', 'schema', 'parent_spec', 'execution_grammar'):
        continue
    assert v3.get(k) == val, f'frozen field mutated: {k}'
# execution_grammar: additive-only assertion (recursive: every v2 leaf must survive unchanged)
def assert_additive(old, new, path=''):
    if isinstance(old, dict):
        assert isinstance(new, dict), f'{path}: type changed'
        for k, ov in old.items():
            assert k in new, f'{path}.{k}: deleted'
            assert_additive(ov, new[k], f'{path}.{k}')
    else:
        assert old == new, f'{path}: mutated ({old!r} -> {new!r})'
assert_additive(v2['execution_grammar'], v3['execution_grammar'], 'execution_grammar')

out = PR / 'e2_ssa_spec.v3.json'
out.write_text(json.dumps(v3, indent=2) + '\n')
receipt = {'schema': 'e2_ssa_amendment_receipt.v2',
           'from': {'schema': 'e2_ssa_spec.v2', 'spec_hash': v2['spec_hash']},
           'to': {'schema': 'e2_ssa_spec.v3', 'spec_hash': v3['spec_hash'],
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest().upper(),
                  'path': 'artifacts/e2/preregistration/e2_ssa_spec.v3.json'},
           'closes_LFNI_blockers': ['B1 operating point pinned by inheritance from write-once E2a theta*',
                                    'B2 deviant placement rule (no-consecutive, gap>=2, K_stimulus-derived, hashed)',
                                    'B3 full estimator/inference definitions incl. BCa 5000 + perm 2000 + keys',
                                    'F S3 recovery contrast fully defined (R_early/R_late/R_rec, delta/I_rec, rho via 500/1000 ladder)'],
           'prospective': 'zero SSA trajectories exist; v2 and v1 preserved immutable',
           'write_once': True}
(PR / 'e2_ssa_spec_v3_amendment_receipt.json').write_text(json.dumps(receipt, indent=2))
print('v3 spec_hash:', v3['spec_hash'])
print('v3 sha256   :', receipt['to']['sha256'])
print('frozen-fields assertion: ALL PASS')
