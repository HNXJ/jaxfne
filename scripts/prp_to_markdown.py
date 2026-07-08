#!/usr/bin/env python3
"""Render artifacts/developer/{plans,progress,review}.json as markdown tables
in the same folder. Run once, or with --watch to regenerate on every save.
"""
import json
import sys
import time
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "developer"

TABLES = {
    "progress.json": ("progress.md", ["path", "score", "status", "tbi", "tbd", "last_verified"]),
    "review.json": ("review.md", ["path", "score", "review_status", "moved_from_progress_on", "review_command"]),
    "plans.json": ("plans.md", None),  # handled specially: items + brainstorm
}


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)[:120]
    return str(v)[:120].replace("\n", " ").replace("|", "\\|")


def render_table(entries, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for e in sorted(entries, key=lambda x: (x.get("score") if x.get("score") is not None else -1)):
        lines.append("| " + " | ".join(cell(e.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def render_plans(data):
    out = ["## items\n"]
    items = data.get("items", [])
    cols = ["id", "title", "status", "target_files"]
    out.append(render_table(items, cols))
    out.append("\n## brainstorm\n")
    for b in data.get("brainstorm", []):
        out.append(f"- **{b.get('id','')}**: {cell(b.get('description', b))}")
    for mp in data.get("midterm_plans", []):
        out.append(f"\n## midterm plan: {mp.get('id','')} ({mp.get('created','')})\n")
        if mp.get("steps"):
            out.append("| step | target | target_score | achieved_score | status | definition_of_done |")
            out.append("|---|---|---|---|---|---|")
            for s in mp.get("steps", []):
                out.append(
                    "| " + " | ".join(cell(v) for v in (
                        s.get("step"), s.get("target"), s.get("target_score"),
                        s.get("achieved_score"), s.get("status"), s.get("definition_of_done"),
                    )) + " |"
                )
        if mp.get("chapters"):
            out.append("| chapter | title | target_categories | status | definition_of_done |")
            out.append("|---|---|---|---|---|")
            for c in mp.get("chapters", []):
                out.append(
                    "| " + " | ".join(cell(v) for v in (
                        c.get("id"), c.get("title"), ", ".join(c.get("target_categories", [])),
                        c.get("status"), c.get("definition_of_done"),
                    )) + " |"
                )
    return "\n".join(out)


def regenerate():
    if not DEV_DIR.exists():
        print(f"no such dir: {DEV_DIR}", file=sys.stderr)
        return
    for src_name, (out_name, cols) in TABLES.items():
        src = DEV_DIR / src_name
        if not src.exists():
            continue
        data = json.loads(src.read_text())
        if src_name == "plans.json":
            body = render_plans(data)
        else:
            entries = data.get("entries", [])
            body = render_table(entries, cols)
            worklists = {
                k: v for k, v in data.items()
                if k.endswith("_worklist_2026_07_07") and isinstance(v, dict)
            }
            if worklists:
                body += "\n\n## active worklists (handoff)\n"
                for name, wl in sorted(worklists.items()):
                    body += f"\n### {name}\n\n"
                    for key in (
                        "status", "achieved_score", "target_score", "for_agent",
                        "remaining_for_exit", "handoff_for_next_agent",
                    ):
                        if key in wl:
                            body += f"- **{key}**: {cell(wl[key])}\n"
                    if wl.get("notebook_ci_receipt"):
                        body += f"- **notebook_ci_receipt**: {cell(wl['notebook_ci_receipt'])}\n"
        header = f"<!-- auto-generated from {src_name} by scripts/prp_to_markdown.py — do not hand-edit -->\n\n"
        (DEV_DIR / out_name).write_text(header + body + "\n")
    print(f"regenerated markdown in {DEV_DIR}")


def watch():
    import subprocess
    print(f"watching {DEV_DIR} for changes (Ctrl-C to stop)...")
    regenerate()
    proc = subprocess.Popen(
        ["fswatch", "-o", str(DEV_DIR)] + [str(DEV_DIR / n) for n in TABLES],
        stdout=subprocess.PIPE,
    )
    try:
        for _ in proc.stdout:
            regenerate()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        regenerate()
