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
from pathlib import Path

import jaxfne as jtfne

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "docs" / "_generated" / "operator_inventory.md"


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


def build_lines() -> list[str]:
    names = sorted(jtfne.__all__)
    by_module: dict[str, list[tuple[str, str, str]]] = {}
    for name in names:
        module, kind, signature = describe(name)
        by_module.setdefault(module, []).append((name, kind, signature))

    lines = [
        "# Operator Inventory (generated)",
        "",
        f"Generated from the live `jaxfne.__all__` export surface ({len(names)} "
        "entries) by `scripts/generate_operator_inventory.py`. Grouped by each "
        "export's real defining submodule, not a hand-maintained category list "
        "— do not hand-edit; regenerate after any export change.",
        "",
    ]
    for module in sorted(by_module):
        entries = by_module[module]
        lines.append(f"## `{module}` ({len(entries)})")
        lines.append("")
        lines.append("| Name | Kind | Signature |")
        lines.append("|---|---|---|")
        for name, kind, signature in entries:
            signature_cell = f"`{signature}`" if signature else ""
            lines.append(f"| `{name}` | {kind} | {signature_cell} |")
        lines.append("")
    return lines


def main() -> None:
    lines = build_lines()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT_DIR)} ({len(jtfne.__all__)} exports)")


if __name__ == "__main__":
    main()
