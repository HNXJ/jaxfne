import json, pathlib
for name in ['jaxfne_mechanism_01_relative_state_X_H_X','jaxfne_mechanism_02_rbd_memory_Xt_Ht1','jaxfne_mechanism_03_hdp_H_W']:
    p = pathlib.Path(f'artifacts/tutorials/etudes/{name}.ipynb')
    nb = json.loads(p.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        src = ''.join(cell.get('source',[]))
        if 'OUTPUT_DIR = Path("artifacts/tutorials/etudes/outputs/mechanism' in src:
            new = src.replace('OUTPUT_DIR = Path("artifacts/tutorials/etudes/outputs/mechanism', 'REPO_ROOT = next((q for q in [Path.cwd(), *Path.cwd().parents] if (q/"jaxfne").is_dir() and (q/"pyproject.toml").exists()), Path.cwd())\nOUTPUT_DIR = REPO_ROOT / "artifacts/tutorials/etudes/outputs/mechanism')
            cell['source'] = new.splitlines(True)
            print(f'patched {name}')
            break
    p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('done')
