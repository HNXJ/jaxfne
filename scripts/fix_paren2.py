import pathlib
for name in ['jaxfne_mechanism_01_relative_state_X_H_X','jaxfne_mechanism_02_rbd_memory_Xt_Ht1','jaxfne_mechanism_03_hdp_H_W']:
    p = pathlib.Path(f'artifacts/tutorials/etudes/{name}.ipynb')
    txt = p.read_text(encoding='utf-8')
    txt = txt.replace('mechanism_01\\")', 'mechanism_01\\"')
    txt = txt.replace('mechanism_02\\")', 'mechanism_02\\"')
    txt = txt.replace('mechanism_03\\")', 'mechanism_03\\"')
    p.write_text(txt, encoding='utf-8')
    print(name, 'fixed')
    txt2 = p.read_text(encoding='utf-8')
    idx = txt2.find('OUTPUT_DIR = REPO')
    print(txt2[idx:idx+250])
