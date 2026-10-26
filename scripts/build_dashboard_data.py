#!/usr/bin/env python3
"""Build dashboard/data/replications.json from dashboard/sources.json and the lane score files.

Every lane file is read with `git show <lane.sha>:<path>` (cwd = the repo root). Values are copied
from the scorer JSONs, never re-scored. This script does not import jaxfne, so the repo-root guard
(P-025) is not needed; the repo root is taken as the parent of this script's directory.

  python scripts/build_dashboard_data.py             write the data file
  python scripts/build_dashboard_data.py --check     exit 1 if the committed file is stale
  python scripts/build_dashboard_data.py --out PATH  write or check PATH instead

--check regenerates in memory and compares with the committed file, ignoring "generated". On a
difference it names the first differing path. --check also validates the built document against
dashboard/replications.schema.json (every git SHA is 40-hex). Each receipt a paper names carries the
sha256 of the bytes git holds at that paper's lane SHA, the bytes its receipt URL links to; a changed
lane blob makes the check fail. Any input error exits 1 and names the paper, file and key.

The study list ("studies") is the table under "## Research studies (study list)" in
docs/reference/references.md; parse_studies refuses a row it cannot read.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "jaxfne-replications/2"
SOURCES_SCHEMA = "jaxfne-dashboard-sources/2"
# Status classes (Hamm 2026-10-16). reproduced, reproduced (conditional) and failed need a signed
# preregistration; a conditional reproduction states its condition in verdict_note. mixed (Hamm
# 2026-10-19): a signed preregistration scored with some primary criteria reproduced and others
# missed; verdict_note names which. An in-progress
# paper has a stage and no verdict; a preregistration scored without a pass keeps it in progress.
# A not-started paper has no verdict and no criteria.
CLASSES = ("reproduced", "reproduced (conditional)", "mixed", "in progress", "not started", "failed",
           "core method")
SCORED_CLASSES = ("reproduced", "reproduced (conditional)", "mixed", "failed")
TOL = 1e-9
FLOOR_REASON = "E-rate floor not met"
NO_VALUE = "—"

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "dashboard" / "sources.json"
DEFAULT_OUT = ROOT / "dashboard" / "data" / "replications.json"
REFERENCES = ROOT / "docs" / "reference" / "references.md"
SCHEMA_FILE = ROOT / "dashboard" / "replications.schema.json"
STUDY_HEADING = "## Research studies (study list)"
STUDY_COLUMNS = ["Reference", "Result compared", "Where"]
STUDY_STATUSES = ("reproduced", "reproduced (conditional)", "mixed", "in progress", "queued", "not started")
MIN_STUDIES = 40


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
        self._digests = {}

    def digest(self, sha, path):
        """sha256 of the file's bytes at a pinned commit (git show, no text conversion), cached per (sha, path)."""
        if (sha, path) not in self._digests:
            shown = git(["show", f"{sha}:{path}"])
            if shown.returncode != 0:
                err = shown.stderr.decode("utf-8", "replace").strip()
                raise BuildError(f"git show {sha}:{path} failed: {err}")
            self._digests[(sha, path)] = hashlib.sha256(shown.stdout).hexdigest()
        return self._digests[(sha, path)]

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
    if flag is None:
        # A scorer row that stays null (a skipped check, such as a missing optional
        # dependency) shows as pending, never as a pass or a synthesized value.
        return "pending", reason
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
    if "reference" in c:
        # A core-method criterion names its independent reference in the scorer file
        # ({"file", "key"}); a literal rides through for other classes.
        refspec = c["reference"]
        reference = p.value(refspec["file"], refspec["key"]) if isinstance(refspec, dict) else refspec
    else:
        reference = None
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
        "reference": reference,
        "reference_text": value_text(reference),
        "prereg": c.get("prereg"),
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
    if cls == "in progress":
        if spec.get("verdict") is not None:
            raise BuildError(f"paper {spec['id']}: an in-progress paper carries no verdict")
        if not spec.get("stage"):
            raise BuildError(f"paper {spec['id']}: an in-progress paper needs a non-empty stage")
    if cls == "not started" and (spec.get("verdict") is not None or spec.get("criteria")):
        raise BuildError(f"paper {spec['id']}: a not-started paper carries no verdict and no criteria")
    if cls == "core method":
        # Analytic checks of a core method, not a paper replication: no stage, no verdict,
        # no preregistration. Every criterion takes its verdict from the scorer file and names
        # the independent reference it was scored against; a skippable criterion (a check the
        # scorer can refuse, such as a missing optional dependency) names its scorer entry.
        if not spec.get("criteria"):
            raise BuildError(f"paper {spec['id']}: a core-method entry needs non-empty criteria")
        for c in spec["criteria"]:
            if "verdict" not in c:
                raise BuildError(f"paper {spec['id']}: criterion {c.get('id')!r} takes its verdict "
                                 "from the scorer file")
            if not c.get("reference"):
                raise BuildError(f"paper {spec['id']}: criterion {c.get('id')!r} needs a non-empty reference")
            if c.get("may_skip") and not c.get("entry"):
                raise BuildError(f"paper {spec['id']}: a skippable criterion names its scorer entry "
                                 "in 'entry'")
        return cls
    if cls in SCORED_CLASSES and not spec.get("preregs"):
        raise BuildError(f"paper {spec['id']}: class {cls!r} needs a signed preregistration")
    if cls == "reproduced (conditional)" and not spec.get("verdict_note"):
        raise BuildError(f"paper {spec['id']}: a conditional reproduction states its condition in verdict_note")
    if cls == "mixed" and not spec.get("verdict_note"):
        raise BuildError(f"paper {spec['id']}: a mixed verdict names what reproduced and what missed in verdict_note")
    return cls


STAGES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")

# Grammar stages a core-method row supports, in the order of the scientific grammar
# (Emitter -> Source -> Field -> Probe -> Objective -> Optimizer), then the adaptive dynamics
# (RBD/HDP) and the numerics every stage runs on.
GRAMMAR = ("Emitter", "Source", "Field", "Probe", "Objective", "Optimizer", "RBD/HDP", "Numerics")


def check_grammar(spec):
    """The grammar stages of a core-method row: a non-empty list from GRAMMAR, in GRAMMAR order.
    Other classes carry none."""
    grammar = spec.get("grammar")
    if spec.get("class") != "core method":
        if grammar is not None:
            raise BuildError(f"paper {spec['id']}: only a core-method entry names grammar stages")
        return None
    if not grammar or any(g not in GRAMMAR for g in grammar):
        raise BuildError(f"paper {spec['id']}: a core-method entry names grammar stages from {list(GRAMMAR)}")
    if list(grammar) != sorted(set(grammar), key=GRAMMAR.index):
        raise BuildError(f"paper {spec['id']}: grammar stages are unique and in grammar order")
    return list(grammar)


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


def check_dev_lane(spec):
    """A dev lane SHA must be in HEAD's history: a rebased-away SHA survives in a local clone but not in CI's."""
    lane = spec["lane"]
    if lane["branch"] != "dev":
        return
    if git(["cat-file", "-e", f"{lane['sha']}^{{commit}}"]).returncode != 0:
        raise BuildError(f"lane sha {lane['sha']} is not a commit in this clone")
    if git(["merge-base", "--is-ancestor", lane["sha"], "HEAD"]).returncode != 0:
        raise BuildError(f"paper {spec['id']}: dev lane sha {lane['sha']} is not in HEAD's history")


def build_paper(spec, files, repo_url):
    check_dev_lane(spec)
    p = Paper(spec, files, repo_url)
    cls = check_class(spec)
    criteria = [build_criterion(p, c) for c in spec.get("criteria", [])]
    library = build_library_path(p, spec)
    return {
        "id": spec["id"],
        "title": spec["title"],
        "citation": spec["citation"],
        "class": cls,
        "question": spec.get("question", ""),
        "page": model_page(spec),
        "release": spec["release"],
        "tier": spec["tier"],
        "stage": spec.get("stage"),
        "grammar": check_grammar(spec),
        "lane": {"branch": spec["lane"]["branch"], "sha": spec["lane"]["sha"]},
        "stage_artifacts": build_stage_artifacts(p, spec),
        "verdict": spec.get("verdict"),
        "verdict_note": spec.get("verdict_note"),
        "summary": spec["summary"],
        "preregs": [
            {"id": pr["id"], "signed": pr["signed"], "path": pr["path"], "url": p.prereg_url(pr["path"])}
            for pr in spec.get("preregs", [])
        ],
        "criteria": criteria,
        "sensitivity": build_sensitivity(p, spec),
        "charts": [build_chart(p, ch) for ch in spec.get("charts", [])],
        "open": list(spec.get("open", [])),
        "verdict_summary": build_verdict_summary(spec, criteria, library),
    }


def build_library_path(p, spec):
    """The scored-apart library route (B1): one value from a lane file, mapped to pass/fail/pending.

    The block names its stage, file and dotted key; the value is derived from the scorer file,
    never hand-written. A missing key is refused; a null, or an absent allow_missing file, is
    pending, never a pass. A paper without the block carries no library verdict."""
    block = spec.get("library_path")
    if block is None:
        return None
    if spec.get("class") == "core method":
        raise BuildError(f"paper {p.id}: a core-method entry carries no library_path")
    if not isinstance(block, dict):
        raise BuildError(f"paper {p.id}: library_path is not a block")
    for field in ("stage", "file", "key"):
        if not block.get(field):
            raise BuildError(f"paper {p.id}: library_path needs a non-empty {field!r}")
    if block["stage"] not in STAGES:
        raise BuildError(f"paper {p.id}: library_path stage {block['stage']!r} is not one of {list(STAGES)}")
    name, key = block["file"], block["key"]
    if p.absent(name):
        return {"stage": block["stage"], "verdict": "pending", "evidence": p.evidence(name)}
    flag = p.value(name, key)
    if flag is None:
        # A scorer row that stays null (a skipped check) shows as pending, never as a pass.
        verdict = "pending"
    elif flag == "pass":
        verdict = "pass"
    elif flag in ("fail", "incomplete"):
        # An incomplete library run scores as a fail, as in verdict_of.
        verdict = "fail"
    else:
        raise BuildError(f"paper {p.id}: library file {name} key {key} is not pass, fail or incomplete")
    return {"stage": block["stage"], "verdict": verdict, "evidence": p.evidence(name)}


def build_verdict_summary(spec, criteria, library):
    """The three verdicts side by side, all derived: the paper verdict from the sources entry,
    the library verdict from the lane's library-path score, the methods verdict from the
    core-method criteria. Failed and pending criteria ride along so no fail averages away."""
    failed = [c for c in criteria if c["verdict"] == "fail"]
    pending = [c for c in criteria if c["verdict"] == "pending"]
    if spec.get("class") == "core method":
        if failed:
            methods = "fail"
        elif pending:
            # One live pass and one pending check is pending, not passing.
            methods = "pending"
        else:
            methods = "pass"
        return {"kind": "core_method", "paper": None, "library": None, "methods": methods,
                "failed_criteria": failed, "pending_criteria": pending}
    return {"kind": "paper", "paper": spec.get("verdict"), "library": library, "methods": None,
            "failed_criteria": failed, "pending_criteria": pending}


def build_overview(papers):
    """Paper denominators exclude core-method rows; core checks are counted apart."""
    rows = [p for p in papers if p["verdict_summary"]["kind"] == "paper"]
    core = [p for p in papers if p["verdict_summary"]["kind"] == "core_method"]

    def counts(ps):
        n = {"pass": 0, "fail": 0, "pending": 0}
        for paper in ps:
            for c in paper["criteria"]:
                if c["verdict"] in n:
                    n[c["verdict"]] += 1
        return n

    by_paper = {}
    for paper in rows:
        key = paper["verdict_summary"]["paper"]
        if key is None:
            key = paper["class"]
        by_paper[key] = by_paper.get(key, 0) + 1
    return {
        "papers": len(rows),
        "by_paper": by_paper,
        "criteria": counts(rows),
        "core_methods": {"rows": len(core), "criteria": counts(core)},
    }


# A DOI may hold parentheses ("s0896-6273(02)01092-9"), so the link ends at a ")" followed by a space or the cell end.
DOI_LINK = re.compile(r"\[doi:([^\]]+)\]\(https://doi\.org/(\S+?)\)(?=\s|$)")
# Longest status name first, so "reproduced (conditional)" is a status and not "reproduced" plus a note.
STUDY_OPEN = re.compile(
    "(" + "|".join(re.escape(s) for s in sorted(STUDY_STATUSES, key=len, reverse=True)) + r")"
    r"(?:\s*\(([^()]*)\))?\.\s*(.*)",
    re.IGNORECASE,
)


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def study_row(cells, number):
    ref_cell, result_cell, where_cell = cells
    doi = None
    reference = ref_cell
    link = DOI_LINK.search(ref_cell)
    if link:
        if link.group(1) != link.group(2):
            raise BuildError(f"study row {number}: DOI text {link.group(1)!r} differs from its link {link.group(2)!r}")
        doi = link.group(2)
        reference = " ".join(s for s in (ref_cell[:link.start()].strip(), ref_cell[link.end():].strip()) if s)
    opened = STUDY_OPEN.match(where_cell)
    if not opened:
        raise BuildError(
            f"study row {number} ({reference[:60]!r}): Where cell opens with {where_cell[:40]!r}, "
            f"not one of {list(STUDY_STATUSES)} followed by a period"
        )
    note = opened.group(2)
    return {
        "reference": reference,
        "doi": doi,
        "result": result_cell,
        "status": opened.group(1).lower(),
        "status_note": None if note is None else note.strip(),
        "where": opened.group(3).strip(),
    }


def parse_studies(markdown):
    """One dict per row of the study-list table under STUDY_HEADING. A missing table, a header other
    than STUDY_COLUMNS, fewer than MIN_STUDIES rows, or a Where cell that opens with no status is an error.
    A parenthetical after the status (for example 'In progress (pilot).') is kept as status_note."""
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines) if line.rstrip() == STUDY_HEADING]
    if not starts:
        raise BuildError(f"references page has no heading {STUDY_HEADING!r}")
    block = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            block.append(line)
    if len(block) < 2 or split_row(block[0]) != STUDY_COLUMNS or not re.fullmatch(r"[|\s:-]+", block[1]):
        raise BuildError(f"study list under {STUDY_HEADING!r} has no header {' | '.join(STUDY_COLUMNS)}")
    rows = block[2:]
    if len(rows) < MIN_STUDIES:
        raise BuildError(f"study list has {len(rows)} rows, expected at least {MIN_STUDIES}")
    out = []
    for number, line in enumerate(rows, 1):
        cells = split_row(line)
        if len(cells) != 3:
            raise BuildError(f"study row {number} has {len(cells)} cells, expected 3")
        out.append(study_row(cells, number))
    return out


def dev_sha():
    proc = git(["rev-parse", "HEAD"])
    sha = proc.stdout.decode("ascii", "replace").strip()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise BuildError("git rev-parse HEAD did not return a 40-hex sha")
    return sha


def content_revision(doc):
    """sha256 over the document without `generated`: a deploy-comparable revision that stays
    stable across regenerations (the timestamp lives in `generated`)."""
    content = {k: v for k, v in doc.items() if k != "generated"}
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


STALE_AFTER_DAYS = 14


def stale_warning(generated, now=None):
    """None when the data file is fresh; a warning naming its date when the generated stamp is
    older than STALE_AFTER_DAYS or unreadable. Never a silent green badge."""
    if now is None:
        now = datetime.now(timezone.utc)
    utc = (generated or {}).get("utc")
    try:
        seen = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return f"stale: generated timestamp {utc!r} is unreadable; treat this data file as stale"
    age_days = (now - seen).days
    if age_days >= STALE_AFTER_DAYS:
        return f"stale: data generated {utc} is {age_days} days old; regenerate before citing it"
    return None


def add_receipt_hashes(papers, files):
    """Give each receipt a paper names a sha256 sibling: the digest of the bytes git holds at that
    paper's lane SHA, which is what the receipt URL links to. An object whose url is null names no
    file (an allow_missing receipt absent at the lane SHA) and gets none. A present receipt that the
    lane SHA does not hold is an error."""
    for paper in papers:
        _hash_named(paper, paper["lane"]["sha"], paper["id"], files)


def _hash_named(node, sha, pid, files):
    if isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, str) and node.get("url", "") is not None:
            if not files.exists(sha, path):
                raise BuildError(f"paper {pid}: receipt {path} is missing at lane sha {sha}")
            node["sha256"] = files.digest(sha, path)
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return
    for child in children:
        _hash_named(child, sha, pid, files)


SCHEMA_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": is_num,
    "null": lambda v: v is None,
}
SCHEMA_KEYWORDS = {"$schema", "title", "description", "type", "required", "properties", "items", "pattern", "enum"}


def load_schema():
    try:
        return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read {SCHEMA_FILE}: {exc}")


def schema_errors(value, schema, where="$"):
    """Problems of `value` against `schema`, as 'where: problem' strings; [] when it conforms.
    A pattern must match the whole string. A keyword this checker does not read is an error, so a
    misspelt keyword cannot pass unread."""
    unknown = sorted(set(schema) - SCHEMA_KEYWORDS)
    if unknown:
        raise BuildError(f"schema at {where} uses keyword(s) {unknown} that this checker does not read")
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else types
        for name in names:
            if name not in SCHEMA_TYPES:
                raise BuildError(f"schema at {where} names unknown type {name!r}")
        if not any(SCHEMA_TYPES[name](value) for name in names):
            return [f"{where}: {json.dumps(value)[:60]} is not {types}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {json.dumps(value)[:60]} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and not re.fullmatch(schema["pattern"], value):
        errors.append(f"{where}: {value[:60]!r} does not match {schema['pattern']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{where}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(schema_errors(value[key], sub, f"{where}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema["items"], f"{where}[{i}]"))
    return errors


def build_document():
    try:
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read {SOURCES}: {exc}")
    if sources.get("schema") != SOURCES_SCHEMA:
        raise BuildError(f"sources schema is {sources.get('schema')!r}, expected {SOURCES_SCHEMA!r}")
    files = Files()
    papers = [build_paper(s, files, sources["repo_url"]) for s in sources["papers"]]
    try:
        studies = parse_studies(REFERENCES.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BuildError(f"cannot read {REFERENCES}: {exc}")
    doc = {
        "schema": SCHEMA,
        "sample": False,
        "generated": {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dev_sha": dev_sha(),
        },
        "papers": papers,
        "overview": build_overview(papers),
        "roadmap": list(sources.get("roadmap", [])),
        "studies": studies,
    }
    add_receipt_hashes(papers, files)
    doc["generated"]["data_revision"] = content_revision(doc)
    return doc


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
    problems = schema_errors(doc, load_schema())
    if problems:
        for line in problems[:10]:
            print("invalid: " + line.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
        print(f"invalid: {len(problems)} schema problem(s) in the built document", file=sys.stderr)
        return 1
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
