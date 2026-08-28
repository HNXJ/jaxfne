import json, pathlib
for name in ['jaxfne_mechanism_01_relative_state_X_H_X','jaxfne_mechanism_02_rbd_memory_Xt_Ht1','jaxfne_mechanism_03_hdp_H_W']:
    p = pathlib.Path(f'artifacts/tutorials/etudes/{name}.ipynb')
    txt = p.read_text(encoding='utf-8')
    # fix double paren
    txt = txt.replace('outputs/mechanism_02")', 'outputs/mechanism_02"')
    txt = txt.replace('outputs/mechanism_03")', 'outputs/mechanism_03"')
    txt = txt.replace('outputs/mechanism_01")', 'outputs/mechanism_01"')
    p.write_text(txt, encoding='utf-8')
    print(name, 'fixed')
