#!/usr/bin/env python3
"""Generate the raise-site index in docs/reference/errors.md from the jaxfne source.

The block between ``<!-- ERRORS-INDEX:START -->`` and ``<!-- ERRORS-INDEX:END -->`` holds one
table per module. A row is each ``raise ValueError(...)``, ``raise TFNEError(...)`` or
``raise EpistemicRefusal(...)`` under ``jaxfne/`` whose first argument is a string literal or an
f-string (f-string fields show as ``{...}``). Modules sort by path, rows by source order; line
numbers are left out so the block changes only when a raise or its message does. Messages are
flattened to one line and cut to 100 characters. No timestamps: two runs give identical bytes.
A source file that does not parse stops the run with a traceback (exit 1).

Usage:
    python scripts/generate_errors_index.py --check   # verify the block is current
    python scripts/generate_errors_index.py --write   # rewrite the block

Exit codes:
    0  the block is current (--check) or was rewritten (--write)
    1  --check: the block is stale or its markers are missing; --write: the markers are missing
    2  bad arguments
"""

from __future__ import annotations

import argparse
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "jaxfne"
PAGE = ROOT / "docs" / "reference" / "errors.md"
START = "<!-- ERRORS-INDEX:START -->"
END = "<!-- ERRORS-INDEX:END -->"
RAISED = ("ValueError", "TFNEError", "EpistemicRefusal")
HEAD_CHARS = 100

Row = tuple[int, str, str]


def _raised_name(call: ast.Call) -> str | None:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    return name if name in RAISED else None


def _message(arg: ast.expr) -> str | None:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{...}" for part in arg.values
        )
    return None


def _cell(text: str) -> str:
    head = " ".join(text.split())[:HEAD_CHARS]
    return "`" + head.replace("`", "'").replace("|", "\\|") + "`"


def collect(source: pathlib.Path = SOURCE) -> dict[str, list[Row]]:
    tables: dict[str, list[Row]] = {}
    for path in sorted(source.rglob("*.py"), key=lambda p: p.as_posix()):
        module = path.relative_to(source.parent).as_posix()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        rows: list[Row] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            name = _raised_name(node.exc)
            if name is None or not node.exc.args:
                continue
            text = _message(node.exc.args[0])
            if text is not None:
                rows.append((node.lineno, name, _cell(text)))
        if rows:
            tables[module] = sorted(rows)
    return tables


def render(tables: dict[str, list[Row]]) -> str:
    total = sum(len(rows) for rows in tables.values())
    lines = [
        START,
        "",
        f"{total} raise sites in {len(tables)} modules. Generated from `jaxfne/**/*.py` by "
        "`python scripts/generate_errors_index.py --write`; do not edit by hand.",
        "",
    ]
    for module, rows in tables.items():
        lines += [f"**`{module}`**", "", "| Class | Message |", "|---|---|"]
        lines += [f"| `{name}` | {cell} |" for _lineno, name, cell in rows]
        lines.append("")
    lines.append(END)
    return "\n".join(lines) + "\n"


def _split(text: str) -> tuple[str, str, str] | None:
    if START not in text or END not in text:
        return None
    before, rest = text.split(START, 1)
    inner, after = rest.split(END, 1)
    return before, inner, after


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    tables = collect()
    block = render(tables)
    count = f"{sum(len(rows) for rows in tables.values())} raise sites in {len(tables)} modules"
    raw = PAGE.read_bytes()
    parts = _split(raw.decode("utf-8").replace("\r\n", "\n"))
    if parts is None:
        print(f"{PAGE}: ERRORS-INDEX markers missing")
        return 1
    before, inner, after = parts
    if args.write:
        if after.startswith("\n"):  # the block ends with END's line break; keep one so writes are idempotent
            after = after[1:]
        newline = "\r\n" if b"\r\n" in raw else "\n"
        PAGE.write_bytes((before + block + after).replace("\n", newline).encode("utf-8"))
        print(f"errors index rewritten: {count}")
        return 0
    if _split(block)[1] != inner:
        print("errors.md index is stale against jaxfne/; run "
              "`python scripts/generate_errors_index.py --write`")
        return 1
    print(f"errors index current: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
