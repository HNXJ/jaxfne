import json, pathlib
p = pathlib.Path('artifacts/tutorials/etudes/jaxfne_mechanism_03_hdp_H_W.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for cell in nb['cells']:
    src = ''.join(cell.get('source',[]))
    if 'with_hdp_initial_state(H=' in src:
        new_src = src.replace('with_hdp_initial_state(H=jnp.ones(N', 'with_hdp_initial_state(H0=jnp.ones(N')
        cell['source'] = new_src.splitlines(True)
        print('patched H->H0')
        break
p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('written')
