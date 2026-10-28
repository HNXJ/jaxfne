"""Guards for the ``paper-lanes`` job in ``ci.yml`` (CI plan step C, jaxfne T-42).

The job runs every paper lane through the gate's own runner (``run_test_gate.py
paper-lanes``), not a copy of the lane list, on pushes to dev and on pull requests
whose changes touch a lane folder or the package. ``ci.yml`` is read with regex,
as ``test_release_gate_hierarchy.py`` reads its workflows (no yaml import).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.run_test_gate import discover_pytest_lanes

ROOT = Path(__file__).resolve().parents[1]
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
JOB = "paper-lanes"
GATE_CALL = "scripts/run_test_gate.py paper-lanes"
PATHS_ENV = "LANE_PATHS_RE"
# The two globs the job must react to: the lane folders and the package.
LANE_GLOBS = ("artifacts/etudes/papers/**", "jaxfne/**")
UNRELATED = ("docs/reference/x.md", "jaxfnex/x.py", "artifacts/etudes/paperz/x.py")
# (event, ref, must the job run?)
TRIGGERS = (
    ("push", "refs/heads/dev", True),
    ("pull_request", "refs/pull/1/merge", True),
    ("push", "refs/heads/main", False),
)


def _job_block(text: str, job: str) -> str:
    """Lines of one job under ``jobs:``: from its id line up to the next job id."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", line):
            if inside:
                break
            inside = line.strip().rstrip(":") == job
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def _step_of(block: str, needle: str) -> str:
    """The step (from its ``- `` line) that contains ``needle``, or an empty string."""
    for step in re.split(r"(?m)^      - ", block):
        if needle in step:
            return step
    return ""


def _if_holds(expr: str, event: str, ref: str) -> bool:
    """Evaluate a job ``if:`` for one trigger; an unsupported token raises (fails closed)."""
    py = re.sub(r"github\.event_name", "EVENT", expr)
    py = re.sub(r"github\.ref\b", "REF", py)
    py = py.replace("||", " or ").replace("&&", " and ")
    return bool(eval(py, {"__builtins__": {}}, {"EVENT": event, "REF": ref}))


def _trigger_failures(block: str) -> list[str]:
    m = re.search(r"^    if: (.+)$", block, re.M)
    if m is None:
        return [] if all(want for _, _, want in TRIGGERS) else [f"`{JOB}` has no job-level if:"]
    expr = m.group(1).strip()
    out: list[str] = []
    for event, ref, want in TRIGGERS:
        try:
            got = _if_holds(expr, event, ref)
        except Exception as exc:  # unsupported expression: report, do not pass
            return [f"`{JOB}` if: {expr!r} cannot be evaluated ({exc})"]
        if got != want:
            out.append(f"`{JOB}` {'does not run on' if want else 'runs on'} {event} on {ref}")
    return out


def _path_failures(block: str) -> list[str]:
    m = re.search(rf"^\s*{PATHS_ENV}: '([^']+)'\s*$", block, re.M)
    if m is None:
        return [f"`{JOB}` has no {PATHS_ENV} path condition"]
    rx = re.compile(m.group(1))
    out: list[str] = []
    for glob in LANE_GLOBS:
        sample = glob.replace("**", "probe/test_probe.py")
        if not rx.search(sample):
            out.append(f"path condition misses {glob!r} (sample {sample!r})")
    for other in UNRELATED:
        if rx.search(other):
            out.append(f"path condition matches unrelated path {other!r}")
    if "fetch-depth: 0" not in block:
        out.append("checkout lacks fetch-depth 0: the base commit for the diff is not in the clone")
    return out


def _runner_failures(block: str) -> list[str]:
    out: list[str] = []
    calls = [ln for ln in block.splitlines() if GATE_CALL in ln]
    if len(calls) != 1:
        out.append(f"`{JOB}` must call `{GATE_CALL}` once, found {len(calls)}")
    if "--lane" in block:
        out.append("the job passes --lane flags: a hard-coded lane list")
    if "python -m pytest" in block:
        out.append("the job runs pytest itself instead of the gate's lane runner")
    lanes = discover_pytest_lanes()
    if not lanes:
        out.append("no paper lanes discovered: the examined set is empty")
    for lane in lanes:
        if re.search(rf"\b{re.escape(lane)}\b", block):
            out.append(f"the job names lane {lane!r} itself")
    if "if: steps.changes.outputs.run == 'true'" not in _step_of(block, GATE_CALL):
        out.append("the lane step does not run only when the change detection says so")
    return out


def _setup_failures(text: str, block: str) -> list[str]:
    fast = _job_block(text, "fast-test")
    out: list[str] = []
    mine = re.findall(r'pip install -e "([^"]+)"', block)
    theirs = set(re.findall(r'pip install -e "([^"]+)"', fast))
    if not mine or not set(mine) <= theirs:
        out.append(f"install lines {mine} are not fast-test's install lines {sorted(theirs)}")
    matrix = re.search(r"python-version: (\[[^\]]*\])", block)
    fast_matrix = re.search(r"python-version: (\[[^\]]*\])", fast)
    if matrix is None or fast_matrix is None or matrix.group(1) != fast_matrix.group(1):
        out.append("python matrix differs from fast-test's")
    if "actions/setup-python@v7" not in block:
        out.append("no setup-python step")
    return out


def _guard_holds(expr: str, results: dict[str, str], event: str, ref: str) -> bool:
    """Evaluate fast-build's guard ``if:`` (true means the build fails) for job results and one trigger."""
    py = re.sub(r"contains\(needs\.\*\.result, '(\w+)'\)", r"('\1' in R.values())", expr)
    py = re.sub(r"needs\.([A-Za-z0-9_-]+)\.result", r"R['\1']", py)
    py = re.sub(r"github\.event_name", "EVENT", py)
    py = re.sub(r"github\.ref\b", "REF", py)
    py = py.replace("||", " or ").replace("&&", " and ")
    return bool(eval(py, {"__builtins__": {}}, {"R": results, "EVENT": event, "REF": ref}))


def _build_failures(text: str, block: str) -> list[str]:
    """fast-build needs the job and fails on its skip exactly where the job's own if: holds."""
    build = _job_block(text, "fast-build")
    needs = re.search(r"^    needs: \[([^\]]*)\]", build, re.M)
    if needs is None:
        return ["fast-build has no needs list"]
    names = [n.strip() for n in needs.group(1).split(",")]
    if JOB not in names:
        return [f"fast-build does not need `{JOB}`"]
    step = _step_of(build, "Needed jobs passed")
    guard = re.search(r"if: >-\n((?:          .+\n)+)", step + "\n")
    if guard is None:
        return ["fast-build has no folded guard if:"]
    expr = " ".join(line.strip() for line in guard.group(1).splitlines())
    job_if = re.search(r"^    if: (.+)$", block, re.M)
    out: list[str] = []
    for event, ref, _ in TRIGGERS:
        job_runs = True if job_if is None else _if_holds(job_if.group(1).strip(), event, ref)
        ok = {n: "success" for n in names}
        try:
            if _guard_holds(expr, ok, event, ref):
                out.append(f"guard fails a green build on {event} {ref}")
            for n in names:
                skipped = {**ok, n: "skipped"}
                want = n != JOB or job_runs
                if _guard_holds(expr, skipped, event, ref) != want:
                    out.append(f"guard {'passes' if want else 'fails'} a skipped {n} on {event} {ref}")
                if not _guard_holds(expr, {**ok, n: "failure"}, event, ref):
                    out.append(f"guard passes a failed {n} on {event} {ref}")
        except Exception as exc:  # unsupported expression: report, do not pass
            return [f"fast-build guard {expr!r} cannot be evaluated ({exc})"]
    return out


def check(text: str) -> dict[str, list[str]]:
    """Failures by category; an empty dict means the job meets the plan."""
    block = _job_block(text, JOB)
    if not block.strip():
        return {"job": [f"ci.yml has no `{JOB}` job"]}
    found = {
        "trigger": _trigger_failures(block),
        "paths": _path_failures(block),
        "runner": _runner_failures(block),
        "setup": _setup_failures(text, block),
        "build": _build_failures(text, block),
    }
    return {name: items for name, items in found.items() if items}


@pytest.fixture(scope="module")
def ci_text() -> str:
    return CI_YML.read_text(encoding="utf-8")


def test_paper_lanes_job_exists(ci_text):
    assert _job_block(ci_text, JOB).strip(), f"ci.yml has no `{JOB}` job"


def _assert_clean(text: str, *categories: str) -> None:
    """The job exists and none of the named categories has a failure (a missing job fails too)."""
    found = check(text)
    assert "job" not in found, found["job"]
    for name in categories:
        assert name not in found, found[name]


def test_trigger_and_path_cover_dev_pushes_pulls_and_both_globs(ci_text):
    _assert_clean(ci_text, "trigger", "paths")


def test_job_calls_the_gate_lane_runner_not_a_lane_list(ci_text):
    _assert_clean(ci_text, "runner")


def test_job_sets_up_python_and_installs_like_fast_test(ci_text):
    _assert_clean(ci_text, "setup")


def test_fast_build_needs_the_job_and_fails_on_its_skip_where_it_runs(ci_text):
    _assert_clean(ci_text, "build")


# Each variant is a planted defect in the real file: the anchor must match exactly once,
# and the mutated file must be caught in the named category.
PLANTED = [
    pytest.param(
        f"{PATHS_ENV}: '^(artifacts/etudes/papers/|jaxfne/)'",
        f"{PATHS_ENV}: '^(jaxfne/)'",
        "paths",
        id="drop-lane-folder-glob",
    ),
    pytest.param(
        f"{PATHS_ENV}: '^(artifacts/etudes/papers/|jaxfne/)'",
        f"{PATHS_ENV}: '^(artifacts/etudes/papers/)'",
        "paths",
        id="drop-package-glob",
    ),
    pytest.param(
        f"{PATHS_ENV}: '^(artifacts/etudes/papers/|jaxfne/)'",
        f"{PATHS_ENV}: '^.'",
        "paths",
        id="path-condition-matches-everything",
    ),
    pytest.param(
        "          fetch-depth: 0",
        "          fetch-depth: 1",
        "paths",
        id="shallow-checkout-no-base",
    ),
    pytest.param(
        f"run: PYTHONPATH=. python {GATE_CALL}",
        "run: PYTHONPATH=. python -m pytest -q -p xdist -n auto artifacts/etudes/papers/pcl",
        "runner",
        id="hard-coded-pytest-line",
    ),
    pytest.param(
        f"run: PYTHONPATH=. python {GATE_CALL}",
        f"run: PYTHONPATH=. python {GATE_CALL} --lane pcl --lane caminos",
        "runner",
        id="lane-flags-hard-coded",
    ),
    pytest.param(
        "if: steps.changes.outputs.run == 'true'",
        "if: always()",
        "runner",
        id="lane-step-not-gated-by-changes",
    ),
    pytest.param(
        "if: github.event_name == 'pull_request' || github.ref == 'refs/heads/dev'",
        "if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'",
        "trigger",
        id="dev-push-dropped",
    ),
    pytest.param(
        "if: github.event_name == 'pull_request' || github.ref == 'refs/heads/dev'",
        "if: github.ref == 'refs/heads/dev'",
        "trigger",
        id="pull-requests-dropped",
    ),
    pytest.param(
        "needs: [static, docs, fast-test, canary, examples, paper-lanes]",
        "needs: [static, docs, fast-test, canary, examples]",
        "build",
        id="fast-build-drops-the-job",
    ),
    pytest.param(
        "(needs.paper-lanes.result == 'skipped' && (github.event_name == 'pull_request' || github.ref == 'refs/heads/dev'))",
        "needs.paper-lanes.result == 'skipped'",
        "build",
        id="skip-fails-main-pushes",
    ),
    pytest.param(
        "          needs.examples.result == 'skipped' ||\n",
        "",
        "build",
        id="skipped-examples-passes",
    ),
    pytest.param(
        "\n  paper-lanes:\n    name:",
        "\n  paper-lane:\n    name:",
        "job",
        id="job-renamed",
    ),
]


@pytest.mark.parametrize("anchor, replacement, category", PLANTED)
def test_planted_defect_is_caught(ci_text, anchor, replacement, category):
    assert ci_text.count(anchor) == 1, f"anchor matches {ci_text.count(anchor)} times: {anchor!r}"
    mutant = ci_text.replace(anchor, replacement)
    assert mutant != ci_text
    found = check(mutant)
    assert found.get(category), f"planted defect not caught in {category!r}: {found}"
