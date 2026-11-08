"""Summarize council files: per paper, the H1/H2/H3 means per rater and the spread of each factor.

Usage: python artifacts/etudes/meta/ocma/tools/summarize.py <council.json> [...]
H1, H2, H3 and the action follow glossary.md (Derived scores, J24.1); `exists` reads data/inventory.json.
Prints one row per (paper, file) with H1 H2 H3 and the dashboard action, then per paper the factors whose
range across files exceeds 0.4. Exit 0, or 2 on unreadable input.
"""
import json
import sys
from pathlib import Path

from validate import KEYS


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def groups(s):
    gap = s.get(KEYS[7])
    h1 = mean([s.get(k) for k in KEYS[0:7]] + ([gap] if gap is not None and gap < 0 else []))
    return {"H1": h1, "H2": mean([s.get(k) for k in KEYS[8:16]]), "H3": mean([s.get(k) for k in KEYS[16:24]])}


def ge(x, t):
    return x is not None and x >= t


def action(h, on_dashboard):
    if on_dashboard:
        return "exists"
    if ge(h["H2"], 0.6) and ge(h["H3"], 0.5):
        return "core-check"
    if ge(h["H1"], 0.6) and ge(h["H3"], 0.5):
        return "core-check (build)"
    if ge(h["H1"], 0.6):
        return "roadmap"
    return "study-list"


def fmt(x):
    return "  n/a" if x is None else f"{x:+.2f}"


def main(paths):
    inv = Path(__file__).resolve().parents[1] / "data" / "inventory.json"
    dash = {r["line"]: bool(r["dashboard"]) for r in json.loads(inv.read_text(encoding="utf-8"))}
    by_paper = {}
    for f in paths:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"unreadable {f}: {e}")
            return 2
        for p in data.get("papers", []):
            by_paper.setdefault(p["line"], []).append((Path(f).name, p["scores"]))
    for line in sorted(by_paper):
        print(f"L{line}")
        for name, s in by_paper[line]:
            h = groups(s)
            print(f"  {name:12s} H1 {fmt(h['H1'])}  H2 {fmt(h['H2'])}  H3 {fmt(h['H3'])}  "
                  f"{action(h, dash.get(line, False))}")
        wide = []
        for k in KEYS:
            vals = [s.get(k) for _, s in by_paper[line] if s.get(k) is not None]
            if vals and max(vals) - min(vals) > 0.4:
                wide.append(f"{k} [{min(vals):+.1f},{max(vals):+.1f}]")
        print("  spread>0.4: " + ("; ".join(wide) if wide else "none"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
