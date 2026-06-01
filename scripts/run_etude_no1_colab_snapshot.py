"""Execute every code cell in Etude No. 1 against the local repo snapshot.

This is a Colab-style smoke executor: it puts the repo root on sys.path,
defaults to lightweight rendering, executes cells in order, writes a JSON receipt,
and exits directly so JAX worker threads do not keep CI/sandboxes open.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "tutorials" / "etudes" / "jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
OUT = REPO / "outputs" / "etude_no_1" / "cell_execution_receipt.json"

os.environ.setdefault("TFNE_RENDER_PLOTLY", "0")
os.environ.setdefault("TFNE_RENDER_MPL", "0")
os.environ.setdefault("TFNE_SMOKE", "1")
sys.path.insert(0, str(REPO))

nb = json.loads(NB.read_text(encoding="utf-8"))
ns: dict[str, object] = {"__name__": "__main__"}
receipt: list[dict[str, object]] = []

try:
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", ""))
        print(f"CELL {idx} start", flush=True)
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception as exc:
            elapsed = time.time() - t0
            receipt.append({"cell": idx, "status": "error", "seconds": elapsed, "error": repr(exc)})
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            traceback.print_exc()
            sys.exit(1)
        elapsed = time.time() - t0
        receipt.append({"cell": idx, "status": "ok", "seconds": elapsed})
        print(f"CELL {idx} done {elapsed:.2f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"ALL CELLS DONE; receipt={OUT}", flush=True)
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
