#!/usr/bin/env python3
"""Build dashboard/data/replications.json from dashboard/sources.json and the lane score files.

Every lane file is read with `git show <lane.sha>:<path>` (cwd = the repo root). Values are copied
from the scorer JSONs, never re-scored. This script does not import jaxfne, so the repo-root guard
(P-025) is not needed; the repo root is taken as the parent of this script's directory.

  python scripts/build_dashboard_data.py             write the data file
  python scripts/build_dashboard_data.py --check     exit 1 if the committed file is stale
  python scripts/build_dashboard_data.py --out PATH  write or check PATH instead

--check regenerates in memory and compares with the committed file, ignoring "generated". On a
difference it names the first differing path. Any input error exits 1 and names the paper, file
and key.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "jaxfne-replications/2"
SOURCES_SCHEMA = "jaxfne-dashboard-sources/2"
# Status classes (Hamm 2026-10-12, dashboard/ARCHITECTURE_v2.md). A demonstration carries no verdict.
# An in-progress paper is on the replication pipeline with no scored preregistration yet: no verdict, a stage.
CLASSES = ("verified", "partial", "not reproduced", "demonstration", "in progress")
TOL = 1e-9
FLOOR_REASON = "E-rate floor not met"
NO_VALUE = "—"

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "dashboard" / "sources.json"
DEFAULT_OUT = ROOT / "dashboard" / "data" / "replications.json"


class BuildError(Exception):
    """An input that cannot be read, or a file or key that the sources name and the lane does not hold."""


def git(args):
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, stdin=subprocess.DEVNULL)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def dotted(obj, key):
    """Return (present, value) at a dotted path such as 'r1.pass' or 'rates_Hz.E'."""
    cur = obj
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def matches(item, where):
    """True when every field in `where` equals the item's field (numbers within TOL)."""
    for key, want in where.items():
        present, got = dotted(item, key)
        if not present:
            return False
        if is_num(want):
            if not is_num(got) or abs(got - want) > TOL:
                return False
        elif got != want:
            return False
    return True


def value_text(v):
    """Three significant figures without trailing zeros, integers in full, null as an em dash."""
    if v is None:
        return NO_VALUE
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        if abs(v) >= 1000:
            return f"{v:.0f}"
        return f"{v:.3g}"
    return str(v)


class Files:
    """Lane files at pinned commits: presence by `git cat-file -e`, content by `git show`."""

    def __init__(self):
        self._commits = set()
        self._present = {}
        self._json = {}

    def exists(self, sha, path):
        if sha not in self._commits:
            if git(["cat-file", "-e", f"{sha}^{{commit}}"]).returncode != 0:
                raise BuildError(f"lane sha {sha} is not a commit in this clone")
            self._commits.add(sha)
        if (sha, path) not in self._present:
            self._present[(sha, path)] = git(["cat-file", "-e", f"{sha}:{path}"]).returncode == 0
        return self._present[(sha, path)]

    def load(self, owner, sha, path):
        if (sha, path) not in self._json:
            shown = git(["show", f"{sha}:{path}"])
            if shown.returncode != 0:
                err = shown.stderr.decode("utf-8", "replace").strip()
                raise BuildError(f"{owner}: git show {sha}:{path} failed: {err}")
            try:
                self._json[(sha, path)] = json.loads(shown.stdout.decode("utf-8"))
            except ValueError as exc:
                raise BuildError(f"{owner}: {path} at {sha} is not valid JSON ({exc})")
        return self._json[(sha, path)]


class Paper:
    """One paper's file table at its lane SHA. Every read goes through here so errors name the paper."""

    def __init__(self, spec, files, repo_url):
        self.id = spec["id"]
        self.sha = spec["lane"]["sha"]
        self.table = spec["files"]
        self.files = files
        self.repo_url = repo_url

    def entry(self, name):
        if name not in self.table:
            raise BuildError(f"paper {self.id}: file {name} is not in its files table")
        spec = self.table[name]
        if isinstance(spec, str):
            return spec, False
        return spec["path"], bool(spec.get("allow_missing", False))

    def absent(self, name):
        """True for an allow_missing file that is not at the lane SHA. A required missing file raises."""
        path, optional = self.entry(name)
        if self.files.exists(self.sha, path):
            return False
        if optional:
            return True
        raise BuildError(f"paper {self.id}: file {name} ({path}) is missing at lane sha {self.sha}")

    def data(self, name):
        """Parsed file, or None when the file is absent (allow_missing)."""
        if self.absent(name):
            return None
        path, _ = self.entry(name)
        return self.files.load(f"paper {self.id}", self.sha, path)

    def value(self, name, key):
        """Value at a dotted key. None when the file is absent; a missing key raises."""
        obj = self.data(name)
        if obj is None:
            return None
        present, value = dotted(obj, key)
        if not present:
            raise BuildError(f"paper {self.id}: file {name} ({self.entry(name)[0]}) has no key {key}")
        return value

    def runs(self, name, where, what):
        """per_run items matching `where`; None when the file is absent. Matching nothing raises."""
        obj = self.data(name)
        if obj is None:
            return None
        present, runs = dotted(obj, "per_run")
        if not present or not isinstance(runs, list):
            raise BuildError(f"paper {self.id}: file {name} has no per_run list ({what})")
        hits = [r for r in runs if matches(r, where)]
        if not hits:
            raise BuildError(
                f"paper {self.id}: file {name} has no per_run item matching {json.dumps(where)} ({what})"
            )
        return hits

    def field(self, name, item, key):
        present, value = dotted(item, key)
        if not present:
            raise BuildError(f"paper {self.id}: file {name} per_run item has no key {key}")
        return value

    def url(self, path):
        return f"{self.repo_url}/blob/{self.sha}/{path}"

    def evidence(self, name):
        path, _ = self.entry(name)
        return {"path": path, "url": None if self.absent(name) else self.url(path)}

    def prereg_url(self, path):
        return self.url(path) if self.files.exists(self.sha, path) else None


def floor_met(p, floor):
    """True when every selected value is at least `min`; a null does not meet it. None when absent."""
    if floor.get("from") != "per_run":
        raise BuildError(f"paper {p.id}: floor from must be per_run")
    hits = p.runs(floor["file"], floor["where"], "floor")
    if hits is None:
        return None
    met = True
    for item in hits:
        v = p.field(floor["file"], item, floor["key"])
        if v is None:
            met = False
        elif not is_num(v):
            raise BuildError(f"paper {p.id}: file {floor['file']} floor key {floor['key']} is not a number")
        elif v < floor["min"]:
            met = False
    return met


def verdict_of(p, c):
    """(verdict, reason) for one criterion. A floor can turn a true boolean into a fail."""
    reason = c.get("reason", "")
    spec = c["verdict"]
    if p.absent(spec["file"]):
        return "pending", reason
    flag = p.value(spec["file"], spec["key"])
    if spec.get("form") == "pass_fail":
        # Scorers that write "pass" / "fail" / "incomplete"; incomplete scores as a fail.
        if flag not in ("pass", "fail", "incomplete"):
            raise BuildError(f"paper {p.id}: verdict file {spec['file']} key {spec['key']} "
                             f"is not pass, fail or incomplete")
        flag = flag == "pass"
    if not isinstance(flag, bool):
        raise BuildError(f"paper {p.id}: verdict file {spec['file']} key {spec['key']} is not a boolean")
    if "floor" in c:
        met = floor_met(p, c["floor"])
        if met is None:
            return "pending", reason
        if not met:
            return "fail", FLOOR_REASON
    return ("pass" if flag else "fail"), reason


def build_extra(p, e):
    value = p.value(e["file"], e["key"])
    return {"label": e["label"], "value": value, "value_text": value_text(value)}


def build_criterion(p, c):
    vspec = c["value"]
    value = p.value(vspec["file"], vspec["key"])
    verdict, reason = verdict_of(p, c)
    return {
        "id": c["id"],
        "group": c.get("group", ""),
        "claim": c["claim"],
        "estimand": c["estimand"],
        "criterion": c["criterion"],
        "value": value,
        "value_text": value_text(value),
        "value_label": c.get("value_label", ""),
        "paper_value": c.get("paper_value"),
        "unit": c.get("unit", ""),
        "extra": [build_extra(p, e) for e in c.get("extra", [])],
        "verdict": verdict,
        "reason": reason,
        "prereg": c["prereg"],
        "evidence": p.evidence(vspec["file"]),
    }


def sens_entry(entry, rows):
    return {
        "id": entry["id"],
        "label": entry["label"],
        "status": entry["status"],
        "note": entry.get("note", ""),
        "rows": rows,
    }


def caminos_sensitivity(p, spec, entries, metrics):
    """Rows from one sensitivity file keyed '<id>.<key>'; reference and band from per_run items."""
    src, ref, rng = spec["sensitivity_source"], spec["sensitivity_ref"], spec["sensitivity_range"]
    ref_hits = p.runs(ref["file"], ref["where"], "sensitivity_ref")
    range_hits = p.runs(rng["file"], rng["where"], "sensitivity_range")
    if ref_hits is None or range_hits is None or len(ref_hits) != 1:
        raise BuildError(f"paper {p.id}: sensitivity_ref must select exactly one per_run item from a present file")
    bands = {}
    for m in metrics:
        values = [p.field(rng["file"], it, m["key"]) for it in range_hits]
        if any(v is not None and not is_num(v) for v in values):
            raise BuildError(f"paper {p.id}: sensitivity_range key {m['key']} holds a non-number")
        # A null inside the range leaves the band unknown rather than narrower than it is.
        bands[m["key"]] = None if any(v is None for v in values) else [min(values), max(values)]
    out = []
    for entry in entries:
        prefix = src["key_prefix"].replace("<id>", entry["id"])
        rows = [
            {
                "metric": m["metric"],
                "unit": m.get("unit", ""),
                "value": p.value(src["file"], prefix + m["key"]),
                "ref": p.field(ref["file"], ref_hits[0], m["key"]),
                "ref_range": bands[m["key"]],
            }
            for m in metrics
        ]
        out.append(sens_entry(entry, rows))
    return out


def pcl_sensitivity(p, spec, entries, metrics):
    """One file per entry, with an optional key prefix; the reference is the main file's key."""
    ref_file = spec["sensitivity_ref"]["file"]
    out = []
    for entry in entries:
        prefix = entry.get("keys_prefix", "")
        rows = []
        for m in metrics:
            if prefix and m.get("skip_with_prefix"):
                continue
            rows.append({
                "metric": m["metric"],
                "unit": m.get("unit", ""),
                "value": p.value(entry["file"], prefix + m["key"]),
                "ref": p.value(ref_file, m["key"]),
                "ref_range": None,
            })
        out.append(sens_entry(entry, rows))
    return out


def build_sensitivity(p, spec):
    entries = spec.get("sensitivity", [])
    if not entries:
        return []
    metrics = spec["sensitivity_metrics"]
    if "sensitivity_source" in spec:
        return caminos_sensitivity(p, spec, entries, metrics)
    return pcl_sensitivity(p, spec, entries, metrics)


def trace(p, s, name, group, items, markers, by_x):
    pts = []
    for it in items:
        x = p.field(s["file"], it, s["x"])
        y = p.field(s["file"], it, s["y"])
        if not is_num(x) or (y is not None and not is_num(y)):
            raise BuildError(f"paper {p.id}: series {group} x key {s['x']} or y key {s['y']} is not a number")
        pts.append((x, y))
    if by_x:
        pts.sort(key=lambda t: t[0])
    return {
        "name": name,
        "group": group,
        "x": [t[0] for t in pts],
        "y": [t[1] for t in pts],
        "markers_only": markers,
    }


def line_traces(p, ch):
    traces = []
    for s in ch["series"]:
        group = s["group"]
        if s.get("from") != "per_run":
            raise BuildError(f"paper {p.id}: chart {ch['id']} series {group} needs from per_run")
        if p.absent(s["file"]):
            continue  # an absent allow_missing file drops its series
        hits = p.runs(s["file"], s.get("where", {}), f"chart {ch['id']}")
        markers = bool(s.get("markers_only", False))
        if "split" not in s:
            traces.append(trace(p, s, group, group, hits, markers, by_x=False))
            continue
        if s["split"] != "seed":
            raise BuildError(f"paper {p.id}: chart {ch['id']} series {group} split must be seed")
        for seed in s["seeds"]:
            chosen = [h for h in hits if matches(h, {"seed": seed})]
            if chosen:  # a seed absent from the data is omitted
                traces.append(trace(p, s, f"{group} seed {seed}", group, chosen, markers, by_x=True))
    return traces


def bar_traces(p, ch):
    cats = ch["categories"]
    traces = []
    for s in ch["series"]:
        group = s["group"]
        if "keys" in s:
            if p.absent(s["file"]):
                continue  # an absent allow_missing file drops its series
            n = len(s["keys"])
        elif "pairs" in s:
            n = len(s["pairs"])
        else:
            raise BuildError(f"paper {p.id}: chart {ch['id']} series {group} needs keys or pairs")
        if n != len(cats):
            raise BuildError(f"paper {p.id}: chart {ch['id']} series {group} has {n} values for {len(cats)} categories")
        if "keys" in s:
            ys = [p.value(s["file"], k) for k in s["keys"]]
        else:
            ys = [p.value(f, k) for f, k in s["pairs"]]
        traces.append({"name": group, "group": group, "y": ys})
    return traces


def build_chart(p, ch):
    series = ch["series"]
    if not series:
        raise BuildError(f"paper {p.id}: chart {ch['id']} has no series")
    first = series[0]
    first_file = first["file"] if "file" in first else first["pairs"][0][0]
    x, y = ch.get("x", {}), ch.get("y", {})
    out = {
        "id": ch["id"],
        "title": ch["title"],
        "kind": ch["kind"],
        "x": {"label": x.get("label", ""), "unit": x.get("unit", "")},
        "y": {"label": y.get("label", ""), "unit": y.get("unit", ""), "log": bool(y.get("log", False))},
    }
    if ch["kind"] == "lines":
        out["traces"] = line_traces(p, ch)
    elif ch["kind"] == "bars":
        out["categories"] = list(ch["categories"])
        out["traces"] = bar_traces(p, ch)
    else:
        raise BuildError(f"paper {p.id}: chart {ch['id']} has unknown kind {ch['kind']}")
    out["refs"] = list(ch.get("refs", []))
    out["caption"] = ch.get("caption", "")
    out["evidence"] = p.evidence(first_file)
    return out


def model_page(spec):
    """Dashboard-relative path of the model's markdown page, or None. A named page must exist."""
    rel = spec.get("page")
    if rel is None:
        return None
    if not re.fullmatch(r"models/[a-z0-9_]+\.md", rel):
        raise BuildError(f"paper {spec['id']}: page {rel!r} is not models/<name>.md")
    if not (ROOT / "dashboard" / rel).is_file():
        raise BuildError(f"paper {spec['id']}: page dashboard/{rel} does not exist")
    return rel


def check_class(spec):
    cls = spec.get("class")
    if cls not in CLASSES:
        raise BuildError(f"paper {spec['id']}: class {cls!r} is not one of {list(CLASSES)}")
    if cls == "demonstration" and spec.get("verdict") is not None:
        raise BuildError(f"paper {spec['id']}: a demonstration carries no verdict")
    if cls == "in progress":
        if spec.get("verdict") is not None:
            raise BuildError(f"paper {spec['id']}: an in-progress paper carries no verdict")
        if not spec.get("stage"):
            raise BuildError(f"paper {spec['id']}: an in-progress paper needs a non-empty stage")
    elif cls != "demonstration" and not spec.get("preregs"):
        raise BuildError(f"paper {spec['id']}: class {cls!r} needs a signed preregistration")
    return cls


STAGES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")


def build_stage_artifacts(p, spec):
    """Per-stage files at the lane SHA, in stage order. A stage past the paper's own stage, an
    unknown stage, or a path the lane does not hold is an error."""
    declared = spec.get("stage_artifacts", {})
    if declared and spec.get("stage") not in STAGES:
        raise BuildError(f"paper {p.id}: stage_artifacts need a stage in {list(STAGES)}")
    out = []
    for stage, items in declared.items():
        if stage not in STAGES:
            raise BuildError(f"paper {p.id}: stage_artifacts key {stage!r} is not one of {list(STAGES)}")
        if STAGES.index(stage) > STAGES.index(spec["stage"]):
            raise BuildError(f"paper {p.id}: stage_artifacts list {stage}, past the paper's stage {spec['stage']}")
        links = []
        for item in items:
            if not p.files.exists(p.sha, item["path"]):
                raise BuildError(f"paper {p.id}: {stage} artifact {item['path']} is missing at lane sha {p.sha}")
            links.append({"label": item["label"], "path": item["path"], "url": p.url(item["path"])})
        out.append({"stage": stage, "items": links})
    out.sort(key=lambda s: STAGES.index(s["stage"]))
    return out


def build_paper(spec, files, repo_url):
    p = Paper(spec, files, repo_url)
    return {
        "id": spec["id"],
        "title": spec["title"],
        "citation": spec["citation"],
        "class": check_class(spec),
        "question": spec.get("question", ""),
        "page": model_page(spec),
        "release": spec["release"],
        "tier": spec["tier"],
        "stage": spec["stage"],
        "lane": {"branch": spec["lane"]["branch"], "sha": spec["lane"]["sha"]},
        "stage_artifacts": build_stage_artifacts(p, spec),
        "verdict": spec.get("verdict"),
        "verdict_note": spec.get("verdict_note"),
        "summary": spec["summary"],
        "preregs": [
            {"id": pr["id"], "signed": pr["signed"], "path": pr["path"], "url": p.prereg_url(pr["path"])}
            for pr in spec.get("preregs", [])
        ],
        "criteria": [build_criterion(p, c) for c in spec.get("criteria", [])],
        "sensitivity": build_sensitivity(p, spec),
        "charts": [build_chart(p, ch) for ch in spec.get("charts", [])],
        "open": list(spec.get("open", [])),
    }


def dev_sha():
    proc = git(["rev-parse", "HEAD"])
    sha = proc.stdout.decode("ascii", "replace").strip()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise BuildError("git rev-parse HEAD did not return a 40-hex sha")
    return sha


def build_document():
    try:
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read {SOURCES}: {exc}")
    if sources.get("schema") != SOURCES_SCHEMA:
        raise BuildError(f"sources schema is {sources.get('schema')!r}, expected {SOURCES_SCHEMA!r}")
    files = Files()
    papers = [build_paper(s, files, sources["repo_url"]) for s in sources["papers"]]
    return {
        "schema": SCHEMA,
        "sample": False,
        "generated": {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dev_sha": dev_sha(),
        },
        "papers": papers,
        "roadmap": list(sources.get("roadmap", [])),
    }


def join(path, key):
    return f"{path}.{key}" if path else str(key)


def first_diff(a, b, path=""):
    """Path of the first difference between two parsed documents, or None when they match."""
    if type(a) is not type(b):
        return path or "$"
    if isinstance(a, dict):
        extra = sorted(set(a) ^ set(b))
        if extra:
            return join(path, extra[0])
        for k in a:
            found = first_diff(a[k], b[k], join(path, k))
            if found is not None:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path or "$"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_diff(x, y, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    return None if a == b else (path or "$")


def check(doc, out):
    if not out.is_file():
        raise BuildError(f"{out} does not exist; run without --check to write it")
    try:
        committed = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read {out}: {exc}")
    if isinstance(committed, dict):
        committed.pop("generated", None)
    doc.pop("generated", None)
    where = first_diff(doc, committed)
    if where is not None:
        print(
            f"stale: {out} differs from the regenerated data at {where}; "
            "run scripts/build_dashboard_data.py",
            file=sys.stderr,
        )
        return 1
    print(f"fresh: {out} matches the sources and lane files (generated ignored)")
    return 0


def write(doc, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the replication dashboard data file.")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the data file is stale (generated is ignored)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="data file to write or check")
    args = parser.parse_args(argv)
    try:
        doc = build_document()
        if args.check:
            return check(doc, args.out)
        write(doc, args.out)
    except BuildError as exc:
        print("error: " + str(exc).encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
        return 1
    print(f"wrote {args.out} ({len(doc['papers'])} papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
