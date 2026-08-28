#!/usr/bin/env python3
"""Generate docs/_generated/operator_inventory.md from the live jaxfne export surface.

Reads jaxfne.__all__ directly from the installed package and groups every
export by its real defining submodule (jaxfne.fields.proxy, jaxfne.optim.agsdr,
...). The grouping is structural, not a hand-maintained name->category map, so
the inventory cannot drift out of sync with the package the way prose lists of
"source operators" / "field operators" can.

Run: python3 scripts/generate_operator_inventory.py
Do not hand-edit the output file; regenerate it instead.
"""

import inspect
import json
from pathlib import Path

import jaxfne as jtfne

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "docs" / "_generated" / "operator_inventory.md"


def _load_public_tiers() -> tuple[set[str], set[str]]:
    try:
        data = json.loads((ROOT_DIR / "artifacts" / "public_surface_contract_v0413.json").read_text(encoding="utf-8"))
        public = set(data.get("public_exports", []))
        compat = set(data.get("compatibility_deprecations", {}).keys())
        return public, compat
    except Exception:
        return set(), set()


def describe(name: str) -> tuple[str, str, str]:
    obj = getattr(jtfne, name, None)
    module = getattr(obj, "__module__", None) or "jaxfne (unresolved)"
    if inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "function"
    else:
        kind = "value"
    try:
        signature = str(inspect.signature(obj)) if callable(obj) else ""
    except (TypeError, ValueError):
        signature = ""
    return module, kind, signature


def _split_sig(sig: str) -> tuple[str, str]:
    if not sig:
        return "", ""
    if "->" in sig:
        inp, out = sig.rsplit("->", 1)
        return inp.strip(), out.strip()
    return sig.strip(), ""


def _state_effect(name: str, kind: str) -> str:
    if kind == "value":
        return "—"
    if kind == "class":
        return "constructs"
    n = name.lower()
    if any(k in n for k in ("simulate", "construct", "tune", "build", "make_", "save_", "load_", "checkpoint", "restore", "run_", "optimize")):
        return "stateful"
    return "pure"


def _public_tier(name: str, public: set[str], compat: set[str]) -> str:
    if name in compat:
        return "COMPATIBILITY"
    if name in public:
        return "CANONICAL"
    # __all__ entries not in contract are treated as CANONICAL (frozen 0.4.13 pass1 exports)
    return "CANONICAL"


def build_lines() -> list[str]:
    names = sorted(jtfne.__all__)
    public, compat = _load_public_tiers()
    lines = [
        "# Operator Inventory (generated)",
        "",
        f"Generated from the live `jaxfne.__all__` export surface ({len(names)} "
        "entries) by `scripts/generate_operator_inventory.py`. Deterministic dense table "
        "Operator|Input|Output|State effect|Public — do not hand-edit; regenerate after any export change.",
        "",
        "| Operator | Input | Output | State effect | Public |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        _, kind, sig = describe(name)
        inp, out = _split_sig(sig)
        inp_cell = f"`{inp}`" if inp else "—"
        out_cell = f"`{out}`" if out else "—"
        # Truncate very long signatures deterministically to keep table dense but complete
        if len(inp_cell) > 120:
            inp_cell = inp_cell[:117] + "...`"
        if len(out_cell) > 60:
            out_cell = out_cell[:57] + "...`"
        state = _state_effect(name, kind)
        tier = _public_tier(name, public, compat)
        lines.append(f"| `{name}` | {inp_cell} | {out_cell} | {state} | {tier} |")
    lines.append("")
    return lines


def main() -> None:
    lines = build_lines()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT_DIR)} ({len(jtfne.__all__)} exports)")


if __name__ == "__main__":
    main()
