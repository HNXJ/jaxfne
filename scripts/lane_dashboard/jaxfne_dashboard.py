"""jaxfne lane dashboard data: one card per replication lane, one row per saved analysis output.

Modelled on the omission Draft-S dashboard (pattern copied, nothing imported from it). Read-only on the
repository; writes into --out (outside the repository):

  lanes.json            one row per folder under artifacts/etudes/papers (the unit of work)
  analyses.json         one row per saved data file under artifacts/ (control, legacy, archive skipped)
  doc_index.json        {"text": page} for files/index.html
  doc_app.json          {"text": script} for files/app.js
  doc_ds.json           the datasets/lanes document; --asset-url sets its source url
  doc_ds_analyses.json  the datasets/analyses document; --analyses-asset-url likewise

Every field is read, never typed: title and first sentence from the lane README, registry ids by
scripts/check_replications.paper_folder over dashboard/sources.json, stage, class and release from
dashboard/data/replications.json, render and receipt dates from git (file time only for untracked
files, and the row says so), open rows from artifacts/todo_stack.md and artifacts/open_stack.md.
jaxfne has no per-receipt staleness judge; receipt counts and dates are not freshness. With
--asset-url / --analyses-asset-url the run only rewrites the named documents from the stamps already in
lanes.json / analyses.json.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import re
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "artifacts" / "etudes" / "papers"
ARTIFACTS = ROOT / "artifacts"
TEMPLATE = Path(__file__).resolve().parent / "template"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import check_replications as CR  # noqa: E402  the registry-to-folder rule, not a copy

DATA_EXT = {".csv": "table", ".tsv": "table", ".gz": "table", ".npz": "array", ".npy": "array",
            ".h5": "array", ".json": "json", ".jsonl": "table"}
SKIP_PARTS = {"control", "legacy", "archive", ".lab", "changelog.d", "__pycache__"}
MAX_COUNT_BYTES = 300 * 1024 * 1024
DOC_MAX = 256 * 1024
ASSET_MAX = 8 * 1024 * 1024


def _utc(ts: float | str) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc) if isinstance(ts, float) else datetime.fromisoformat(ts)
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "-c", "core.quotepath=false", *args], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def git_dates(prefix: str) -> dict[str, str]:
    """{repo-relative path: UTC time of the newest commit touching it} under prefix."""
    dates, when = {}, ""
    for line in _git("log", "--format=@%cI", "--name-only", "--", prefix).splitlines():
        if line.startswith("@"):
            when = _utc(line[1:])
        elif line and line not in dates:
            dates[line] = when
    return dates


def _date(p: Path, dates: dict) -> tuple[str, str]:
    rel = p.relative_to(ROOT).as_posix()
    if rel in dates:
        return dates[rel], "git"
    return _utc(p.stat().st_mtime), "file time (untracked)"


README_FILES = ("README.md", "SOURCE_P1.md", "NOTES.md")


def readme_fields(lane: Path) -> tuple[str, str, str]:
    """(title, first sentence, file read) from the lane's first README_FILES entry; ("", "", "") if none.
    The sentence is the first of the first prose paragraph after the title; in SOURCE_P1.md the paragraph
    that names the target ("Target...") is preferred, and the title drops its ": P1 source record" tail."""
    p = next((lane / n for n in README_FILES if (lane / n).is_file()), None)
    if p is None:
        return "", "", ""
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^# (.+)$", text, re.M)
    title = re.sub(r":\s*P1 source record$", "", re.sub(r"[*`]", "", m.group(1).strip())) if m else ""
    body = text[m.end():] if m else text
    raw = [q.strip() for q in re.split(r"\n\s*\n", body)]
    pars = [re.sub(r"[*`]", "", re.sub(r"\s+", " ", q)) for q in raw
            if q and not q.startswith(("#", "|", "!", "<", "```", "-", ">", "* ", "+ "))]
    if p.name == "SOURCE_P1.md":
        pars = [q for q in pars if q.startswith("Target")] or [q for q in pars if not q.startswith("Status")] or pars
    # a sentence ends at . ! ? and a space, except after an initial ("E. M.") or an abbreviation
    end = r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!Fig\.)(?<!Figs\.)(?<!al\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Eq\.)\s"
    first = re.split(end, pars[0], maxsplit=1)[0] if pars else ""
    return title, first, p.name


def stack_items() -> list[tuple[str, str]]:
    """(label, text) per open row: todo table rows ("todo <#>"), T-n bullets with their continuation
    lines, and O-n lines of the open stack, in file order."""
    items = []
    for name, pat in (("todo_stack.md", r"^- (T-\d+[a-z]?)\b"), ("open_stack.md", r"^- (O-\d+)\b")):
        p = ARTIFACTS / name
        if not p.is_file():
            continue
        cur, prev_row = None, None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(pat, line)
            t = re.match(r"^\|\s*([^|]+?)\s*\|", line)
            if m:
                cur = [m.group(1), line]
                items.append(cur)
            elif cur is not None and line.startswith((" ", "\t")) and line.strip():
                cur[1] += " " + line.strip()
            else:
                cur = None
                if name == "todo_stack.md" and t:
                    if set(t.group(1)) <= set("-: "):
                        # a separator row: the row above it was the table header, not an item
                        if prev_row is not None and items and items[-1] is prev_row:
                            items.pop()
                        prev_row = None
                    else:
                        prev_row = ["todo " + t.group(1), line]
                        items.append(prev_row)
                        continue
            prev_row = None if not t else prev_row
    return [(a, b) for a, b in items]


def aliases(lane: str, ids: list[str]) -> list[str]:
    out = [lane, *ids]
    m = re.fullmatch(r"([a-z]+)(\d{4})", lane)
    if m:
        out.append(f"{m.group(1)} {m.group(2)}")
    return out


def named_by(names: list[str], items) -> list[str]:
    pats = [re.compile(rf"(?<![a-z0-9_-]){re.escape(n)}(?![a-z0-9_-])", re.I) for n in names]
    return [label for label, text in items if any(p.search(text) for p in pats)]


def thumb(png: Path, width: int = 420) -> str:
    from PIL import Image

    im = Image.open(png)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _is_data(p: Path) -> bool:
    if p.name.startswith("_temp_") or p.suffix not in DATA_EXT:
        return False
    return p.suffix != ".gz" or p.name.endswith(".csv.gz")


def _is_receipt(p: Path) -> bool:
    """A run receipt: a json named *receipt*, or a lane score file (score*.json, scores/*.json), which is
    the scored-run record of a jaxfne replication lane."""
    return p.suffix == ".json" and ("receipt" in p.name or p.name.startswith("score") or p.parent.name == "scores")


def registry() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """({lane folder name: registry ids}, {id: replications.json paper})."""
    by_lane: dict[str, list[str]] = {}
    for paper in CR.load_registry(ROOT):
        folder = CR.paper_folder(paper, ROOT)
        if folder:
            by_lane.setdefault(folder.rsplit("/", 1)[-1], []).append(paper["id"])
    rep = ROOT / "dashboard" / "data" / "replications.json"
    papers = json.loads(rep.read_text(encoding="utf-8"))["papers"] if rep.is_file() else []
    return by_lane, {p["id"]: p for p in papers}


def lane_rows() -> list[dict]:
    by_lane, rep = registry()
    items = stack_items()
    dates = git_dates("artifacts/etudes/papers")
    dates.update(git_dates("dashboard/figures"))
    rows = []
    for lane in sorted(p for p in PAPERS.iterdir() if p.is_dir() and p.name != "__pycache__"):
        name = lane.name
        ids = by_lane.get(name, [])
        title, cap, readme = readme_fields(lane)
        files = [p for p in lane.rglob("*") if p.is_file() and not p.name.startswith("_temp_")
                 and not SKIP_PARTS & set(p.relative_to(lane).parts)]
        pngs = [p for p in files if p.suffix == ".png"]
        if not pngs:
            pngs = [p for i in ids for p in sorted((ROOT / "dashboard" / "figures").glob(f"{i}-*.png"))]
        png = max(pngs, key=lambda p: _date(p, dates)[0]) if pngs else None
        rendered, render_src = _date(png, dates) if png else ("", "")
        receipts = sorted((p for p in files if _is_receipt(p)), key=lambda p: _date(p, dates)[0])
        data = [p for p in files if _is_data(p) and not _is_receipt(p)]
        stand = [rep[i] for i in ids if i in rep]
        named = named_by(aliases(name, ids), items)
        rows.append({
            "id": name,
            "title": title or name,
            "caption": cap,
            "caption_source": readme,
            "registry_ids": ", ".join(ids),
            "stage": "; ".join(str(p.get("stage", "")) for p in stand),
            "class": "; ".join(str(p.get("class", "")) for p in stand),
            "release": "; ".join(str(p.get("release", "")) for p in stand),
            "standing": "; ".join(f"{rep[i].get('stage', '')}, {rep[i].get('class', '')}" if i in rep
                                  else f"{i} registered, no row in replications.json" for i in ids)
                        or "not in the registry",
            "thumb": thumb(png) if png else "",
            "thumb_path": png.relative_to(ROOT).as_posix() if png else "",
            "rendered_utc": rendered,
            "render_source": render_src,
            "n_receipts": len(receipts),
            "newest_receipt": receipts[-1].relative_to(lane).as_posix() if receipts else "",
            "newest_receipt_utc": _date(receipts[-1], dates)[0] if receipts else "",
            "n_data_files": len(data),
            "data_files": ", ".join(sorted(p.relative_to(lane).as_posix() for p in data)[:8]),
            "data_kb": round(sum(p.stat().st_size for p in data) / 1024),
            "n_files": len(files),
            "open_items": ", ".join(named),
            "n_open_items": len(named),
        })
    return rows


def _count_rows(p: Path) -> int | str:
    if p.stat().st_size > MAX_COUNT_BYTES:
        return ""
    opener = gzip.open if p.suffix == ".gz" else open
    n = 0
    with opener(p, "rb") as fh:
        for n, _ in enumerate(fh, 1):
            pass
    return n if p.suffix == ".jsonl" else max(0, n - 1)


def analysis_rows() -> list[dict]:
    """One row per saved data file under artifacts/ (tables, arrays, receipts, json), newest first."""
    tracked = set(_git("ls-files", "artifacts").splitlines())
    dates = git_dates("artifacts")
    rows = []
    for p in sorted(ARTIFACTS.rglob("*")):
        rel_a = p.relative_to(ARTIFACTS)
        if not p.is_file() or SKIP_PARTS & set(rel_a.parts) or not _is_data(p):
            continue
        parts = rel_a.parts
        lane = parts[:2] == ("etudes", "papers") and len(parts) > 3
        start, stop = (1, 3) if lane else (0, min(2, len(parts) - 1))
        area = "/".join(parts[start:stop]) or "."
        kind = "receipt" if _is_receipt(p) else DATA_EXT[p.suffix]
        n_rows = ""
        if p.suffix in (".csv", ".tsv", ".gz", ".jsonl"):
            try:
                n_rows = _count_rows(p)
            except (OSError, EOFError, zlib.error):
                n_rows = ""
        written, src = _date(p, dates)
        rows.append({
            "path": p.relative_to(ROOT).as_posix(), "area": area, "file": "/".join(parts[stop:]),
            "kind": kind, "n_rows": n_rows, "size_kb": round(p.stat().st_size / 1024),
            "written_utc": written, "written_source": src, "tracked": p.relative_to(ROOT).as_posix() in tracked,
        })
    rows.sort(key=lambda r: r["written_utc"], reverse=True)
    return rows


def stamp(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run the generator once without --asset-url or --analyses-asset-url")
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["commit"], d["read_utc"]


def write_doc(out: Path, name: str, title: str, desc: str, url: str, source: str, read_utc: str) -> None:
    (out / name).write_text(json.dumps({
        "title": title, "description": desc,
        "source": {"kind": "file", "url": url, "name": source, "format": "json"},
        "updated": {"at": read_utc, "by": "Claude Code (local read of the repository)"},
    }), encoding="utf-8")


def lanes_doc(out, head, read_utc, url):
    write_doc(out, "doc_ds.json", "Replication lanes",
              "One row per folder in artifacts/etudes/papers, read from its README, files, git, the registry "
              f"(dashboard/sources.json, dashboard/data/replications.json) and the todo and open stacks at commit {head}.",
              url, "lanes.json", read_utc)


def analyses_doc(out, head, read_utc, url):
    write_doc(out, "doc_ds_analyses.json", "Saved analysis outputs",
              "One row per saved data file under artifacts/ (control, legacy and archive skipped), rows counted "
              f"from the files, at commit {head}.", url, "analyses.json", read_utc)


def check_sizes(out: Path) -> list[str]:
    bad = []
    for p in out.glob("doc_*.json"):
        if p.stat().st_size > DOC_MAX:
            bad.append(f"{p.name} {p.stat().st_size} B > {DOC_MAX}")
    for n in ("lanes.json", "analyses.json"):
        p = out / n
        if p.is_file() and p.stat().st_size > ASSET_MAX:
            bad.append(f"{n} {p.stat().st_size} B > {ASSET_MAX}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="directory outside the repository")
    ap.add_argument("--asset-url", default="", help="url the lanes.json upload returned")
    ap.add_argument("--analyses-asset-url", default="", help="url the analyses.json upload returned")
    a = ap.parse_args()
    for u in (a.asset_url, a.analyses_asset_url):
        if u and not re.fullmatch(r"/_blob/[0-9a-f]{32}", u):
            raise SystemExit(f"asset url {u!r} is not /_blob/<32 hex>; under Git Bash prefix MSYS_NO_PATHCONV=1")
    out = a.out.resolve()
    if out == ROOT.resolve() or ROOT.resolve() in out.parents:
        raise SystemExit(f"--out {out} is inside the repository; choose a scratch directory")
    out.mkdir(parents=True, exist_ok=True)
    if a.asset_url or a.analyses_asset_url:
        if a.asset_url:
            lanes_doc(out, *stamp(out / "lanes.json"), a.asset_url)
        if a.analyses_asset_url:
            analyses_doc(out, *stamp(out / "analyses.json"), a.analyses_asset_url)
        print(f"refresh: rewrote the named dataset documents in {out}; data files not reread")
        return 0
    head = _git("rev-parse", "--short", "HEAD").strip()
    read_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = lane_rows()
    (out / "lanes.json").write_text(json.dumps({"rows": rows, "commit": head, "read_utc": read_utc}), encoding="utf-8")
    arows = analysis_rows()
    (out / "analyses.json").write_text(json.dumps({"rows": arows, "commit": head, "read_utc": read_utc}),
                                       encoding="utf-8")
    page = (TEMPLATE / "index.html").read_text(encoding="utf-8").rstrip() + '\n<script src="app.js"></script>\n'
    (out / "doc_index.json").write_text(json.dumps({"text": page}), encoding="utf-8")
    (out / "doc_app.json").write_text(json.dumps({"text": (TEMPLATE / "app.js").read_text(encoding="utf-8")}),
                                      encoding="utf-8")
    lanes_doc(out, head, read_utc, "")
    analyses_doc(out, head, read_utc, "")
    bad = check_sizes(out)
    missing = [r["id"] for r in rows if r["standing"] == "not in the registry"]
    print(f"{len(rows)} lane rows, {len(arows)} analysis rows, commit {head}, {read_utc}; wrote 6 files to {out}")
    print("lanes not in the registry: " + (", ".join(missing) or "none"))
    print("lanes without a render: " + (", ".join(r["id"] for r in rows if not r["thumb"]) or "none"))
    if bad:
        raise SystemExit("over the size limits: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
