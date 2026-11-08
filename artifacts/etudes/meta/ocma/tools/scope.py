"""OCMA stage B: select the in-scope rows of inventory.json and give every excluded row its reason.

Usage: python artifacts/etudes/meta/ocma/tools/scope.py
Writes data/scope.json: {"in": [line, ...], "out": [{"line", "reference", "reason"}], "batches": [[line x5]]}.
Exit 0 written, 2 when data/inventory.json is missing or a section is unmapped.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data"
OUT_REASON = {
    "Background: simulators and model languages": "simulator or model language: no mechanism to reproduce",
    "Background: differentiable computation": "differentiable-computation tool: no mechanism to reproduce",
    "Background: reproducibility and data standards": "reproducibility or data standard: no mechanism to reproduce",
}
IN_SECTIONS = {
    "Neural dynamics", "Sources and extracellular observation", "Background: reduced and population models",
    "Background: circuits, rhythms and plasticity", "Research studies (study list)",
}


def main():
    inv_path = HERE / "inventory.json"
    if not inv_path.is_file():
        print("inventory.json missing: run inventory.py first")
        return 2
    rows = json.loads(inv_path.read_text(encoding="utf-8"))
    keep, out = [], []
    for r in rows:
        sec = r["section"]
        if sec in OUT_REASON:
            out.append({"line": r["line"], "reference": r["reference"][:90], "reason": OUT_REASON[sec]})
        elif sec in IN_SECTIONS:
            keep.append(r["line"])
        else:
            print(f"unmapped section {sec!r} (line {r['line']})")
            return 2
    keep.sort()
    batches = [keep[i:i + 5] for i in range(0, len(keep), 5)]
    (HERE / "scope.json").write_text(
        json.dumps({"in": keep, "out": out, "batches": batches}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"in {len(keep)}, out {len(out)}, batches {len(batches)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
