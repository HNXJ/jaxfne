"""Author e2_ssa_spec.v4 from immutable v3: H3 reconciliation fixes only.
- steps arithmetic corrected 176000 (88 s)
- washout restated as single mechanical rule; sequence-realized counts hashed, no fabricated numbers
- dR formula pinned; g bound to Hedges' g over replicates"""
import sys, json, hashlib, pathlib, copy
sys.path.insert(0, '.')
PR = pathlib.Path('artifacts/e2/preregistration')

def canon(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()

v3_path = PR / 'e2_ssa_spec.v3.json'
v3 = json.loads(v3_path.read_bytes())
assert canon(v3) == v3['spec_hash']

v4 = copy.deepcopy(v3)
v4['schema'] = 'e2_ssa_spec.v4'
v4['parent_spec'] = {'schema': 'e2_ssa_spec.v3', 'spec_hash': v3['spec_hash'],
                     'sha256': hashlib.sha256(v3_path.read_bytes()).hexdigest().upper(),
                     'preserved_immutable': True,
                     'amendment_reason': 'H3 reconciliation: steps arithmetic, washout rule restatement, dR/g symbol bindings'}
eg = v4['execution_grammar']
eg['time']['steps_per_replicate_derived'] = ('3 blocks x 80 events x 400 steps + recovery_isi500 40 x 1000 '
                                             '+ recovery_isi1000 20 x 2000 = 96000 + 40000 + 40000 '
                                             '= 176000 steps (88 s biological at dt_ms 0.5)')
eg['paradigm']['transient_washout'] = ('exactly the first 2 events of every block are excluded from all analyses; '
                                       'all post-washout counts are sequence-realized under deviant_placement_rule '
                                       '(expected deviants/block ~11.6 pre-washout; realized counts and full sequences '
                                       'hashed into each replicate receipt; many_standards 78->76, recovery 40->38 and '
                                       '20->18 post-washout); no analytic event-count threshold exists')
ms = eg['methods_ssa']
ms['dR'] = 'dR = R_dev - R_std in Hz, identical population/window/R definitions as R_event'
ms['g'] = "g = Hedges' g of per-replicate SI values across n_outer=20 replicates (mean diff / pooled SD)"
v4.pop('spec_hash', None)
v4['spec_hash'] = canon(v4)

for k, val in v3.items():
    if k in ('spec_hash', 'schema', 'parent_spec', 'execution_grammar'):
        continue
    assert v4.get(k) == val, f'mutated {k}'
# Corrections of false v3 content (whitelisted replacements, each with reason).
# Everything else must be strictly additive.
REPLACEMENTS = {
    'execution_grammar.time.steps_per_replicate_derived':
        'v3 arithmetic was false (156000); corrected to 176000 = 96000+40000+40000',
    'execution_grammar.paradigm.transient_washout':
        'v3 counts were fabricated/inconsistent; restated as single mechanical rule with hashed realized counts',
}
def assert_additive(old, new, path=''):
    if isinstance(old, dict):
        for k, ov in old.items():
            p = f'{path}.{k}'
            if p in REPLACEMENTS:
                assert new.get(k) != ov or True  # replacement allowed; content checked below
                continue
            assert k in new, f'{path}.{k} deleted'
            assert_additive(ov, new[k], p)
    else:
        if path not in REPLACEMENTS:
            assert old == new, f'{path} mutated'
assert_additive(v3['execution_grammar'], v4['execution_grammar'], 'execution_grammar')
for p in REPLACEMENTS:
    node = v4['execution_grammar']
    for part in p.split('.')[1:]:
        node = node[part]
    assert isinstance(node, str) and len(node) > 40

# H3 self-check
a, b, c = 3 * 80 * 400, 40 * 1000, 20 * 2000
assert a + b + c == 176000, (a, b, c)
out = PR / 'e2_ssa_spec.v4.json'
out.write_text(json.dumps(v4, indent=2) + '\n')
receipt = {'schema': 'e2_ssa_amendment_receipt.v3',
           'from': {'schema': 'e2_ssa_spec.v3', 'spec_hash': v3['spec_hash']},
           'to': {'schema': 'e2_ssa_spec.v4', 'spec_hash': v4['spec_hash'],
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest().upper(),
                  'path': 'artifacts/e2/preregistration/e2_ssa_spec.v4.json'},
           'closes': ['N/H3 steps arithmetic 176000=88s (was 156000)',
                      'N/H3 washout single mechanical rule, sequence-realized counts hashed (fabricated 10/66/19/9 removed)',
                      'L dR formula pinned', 'L g bound to Hedges g over replicates'],
           'prospective': True, 'write_once': True}
(PR / 'e2_ssa_spec_v4_amendment_receipt.json').write_text(json.dumps(receipt, indent=2))
print('v4 spec_hash:', v4['spec_hash'])
print('v4 sha256   :', receipt['to']['sha256'])
print('assertions + H3 self-check: PASS')
