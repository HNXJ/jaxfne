"""Author e2_ssa_spec.v6 from immutable v5: append recovery_isi1000 to blocks_per_replicate
(spec-authoring gap from v3: block was defined under methods_ssa but never added to the battery)."""
import sys, json, hashlib, pathlib, copy
sys.path.insert(0, '.')
PR = pathlib.Path('artifacts/e2/preregistration')

def canon(j):
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != 'spec_hash'},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()

v5_path = PR / 'e2_ssa_spec.v5.json'
v5 = json.loads(v5_path.read_bytes())
assert canon(v5) == v5['spec_hash']
v6 = copy.deepcopy(v5)
v6['schema'] = 'e2_ssa_spec.v6'
v6['parent_spec'] = {'schema': 'e2_ssa_spec.v5', 'spec_hash': v5['spec_hash'],
                     'sha256': hashlib.sha256(v5_path.read_bytes()).hexdigest().upper(),
                     'preserved_immutable': True,
                     'amendment_reason': 'append recovery_isi1000 to blocks_per_replicate (v3 defined the block only under methods_ssa; S3 rho ladder requires it in the executed battery)'}
blocks = v6['execution_grammar']['paradigm']['blocks_per_replicate']
assert not any(b['name'] == 'recovery_isi1000' for b in blocks)
blocks.append({'name': 'recovery_isi1000', 'standard_only': 'A', 'isi_ms': 1000, 'events': 20})
assert len(blocks) == 5
v6.pop('spec_hash', None)
v6['spec_hash'] = canon(v6)

for k, val in v5.items():
    if k in ('spec_hash', 'schema', 'parent_spec', 'execution_grammar'):
        continue
    assert v6.get(k) == val
def walk(old, new, path=''):
    if isinstance(old, dict):
        for k, ov in old.items():
            p = f'{path}.{k}'
            assert k in new, f'{p} deleted'
            walk(ov, new[k], p)
    else:
        assert old == new or path.endswith('blocks_per_replicate'), f'{path} mutated'
walk(v5['execution_grammar'], v6['execution_grammar'], 'execution_grammar')

out = PR / 'e2_ssa_spec.v6.json'
out.write_text(json.dumps(v6, indent=2) + '\n')
receipt = {'schema': 'e2_ssa_amendment_receipt.v5',
           'from': {'schema': 'e2_ssa_spec.v5', 'spec_hash': v5['spec_hash']},
           'to': {'schema': 'e2_ssa_spec.v6', 'spec_hash': v6['spec_hash'],
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest().upper(),
                  'path': 'artifacts/e2/preregistration/e2_ssa_spec.v6.json'},
           'change': 'blocks_per_replicate += recovery_isi1000 {standard_only A, isi 1000 ms, 20 events}; steps/rep now 216000 (108 s)',
           'prospective': True, 'write_once': True}
(PR / 'e2_ssa_spec_v6_amendment_receipt.json').write_text(json.dumps(receipt, indent=2))
print('v6 spec_hash:', v6['spec_hash'])
print('v6 sha256   :', receipt['to']['sha256'])
