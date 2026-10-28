"""Replication-checklist gate tests: per-rule fixtures under tmp_path.

Each rule test builds the violating tree it is named after, asserts the
fixture precondition, then collects findings end to end through
``collect_findings`` (the same entry point ``main`` uses), so killing a
check body or its call site fails the test.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
from pathlib import Path

import pytest

from scripts import check_replications as cr
from scripts import run_test_gate as rtg
from scripts.run_test_gate import AUDIT_STEPS, CHECK_FAMILIES, GATE_CHECK_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "artifacts" / "etudes" / "papers"
# Signed or historical records: their citations stay as written. Any path naming a prereg is a
# pinned record (the prereg class of scripts/check_record_integrity.py).
SIGNED_RECORD_GLOBS = ("PREREG_P3*.md", "RESULT*.md", "LEDGER.md")
LINE_CITE = re.compile(r"jaxfne/[A-Za-z0-9_/]+\.py:[0-9]")
SYMBOL_CITE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)` in `(jaxfne/[A-Za-z0-9_/]+\.py)`"
)


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _registry(root: Path, papers: list[dict]) -> None:
    _write(root, "dashboard/sources.json", json.dumps({"papers": papers}))


def _known(root: Path, entries: list[dict]) -> None:
    _write(root, "scripts/replication_known_violations.json", json.dumps(entries))


def _paper(pid: str, stage: str, paths: list[str]) -> dict:
    return {
        "id": pid,
        "stage": stage,
        "stage_artifacts": {"P1": [{"path": p} for p in paths]},
        "preregs": [],
    }


def _entry(rule: str, path: str, message: str = "holds 8 files (at most 7)") -> dict:
    return {"rule": rule, "path": path, "message": message,
            "reason": "audit 2026-10-17", "todo": "T-30"}


def test_findings_print_rule_path_message(tmp_path, capsys):
    _registry(tmp_path, [])
    for i in range(8):
        _write(tmp_path, f"artifacts/etudes/papers/crowded/f{i}.md")
    findings = cr.collect_findings(tmp_path)
    assert any(rule == "R6" for rule, _, _ in findings), "fixture must fail R6"
    cr.main(["--root", str(tmp_path)])
    assert "R6 artifacts/etudes/papers/crowded: " in capsys.readouterr().out, (
        "findings print as `RULE path: message`"
    )


def test_r1_missing_record_fails(tmp_path):
    keep = "artifacts/etudes/papers/alpha/keep.md"
    _registry(tmp_path, [_paper("alpha", "P1", [keep])])
    _write(tmp_path, keep)
    assert not (tmp_path / "artifacts/etudes/papers/alpha/SOURCE_P1.md").exists(), (
        "fixture must lack the P1 record"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "R1" and path == "artifacts/etudes/papers/alpha"
        for rule, path, _ in findings
    ), "stage P1 paper without SOURCE_P1.md must fail R1"


def test_r1_disagreeing_segments_fails(tmp_path):
    _registry(
        tmp_path,
        [_paper("split", "P1", [
            "artifacts/etudes/papers/alpha/x.md",
            "artifacts/etudes/papers/beta/y.md",
        ])],
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "R1" and path == "dashboard/sources.json" and "disagree" in msg
        for rule, path, msg in findings
    ), "paths naming two folders must fail R1"


def test_r1_missing_folder_fails(tmp_path):
    _registry(tmp_path, [_paper("ghost", "P1", ["artifacts/etudes/ghost/x.md"])])
    assert not (tmp_path / "artifacts/etudes/papers/ghost").exists(), (
        "fixture folder must be absent"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "R1" and path == "artifacts/etudes/papers/ghost"
        for rule, path, _ in findings
    ), "a named but absent folder must fail R1"


def test_r6_too_many_files_fails(tmp_path):
    _registry(tmp_path, [])
    for i in range(8):
        _write(tmp_path, f"artifacts/etudes/papers/crowded/f{i}.md")
    files = [p for p in (tmp_path / "artifacts/etudes/papers/crowded").iterdir()]
    assert len(files) == 8, "fixture must hold 8 files"
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "R6" and path == "artifacts/etudes/papers/crowded"
        for rule, path, _ in findings
    )


def test_s1_subprocess_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/s1/run.py",
        "import subprocess\nsubprocess.run(['x'])\n",
    )
    assert "subprocess" in run.read_text(encoding="utf-8"), (
        "fixture must call subprocess"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1" and path == "artifacts/etudes/papers/s1/run.py"
        for rule, path, _ in findings
    )


def test_s1_nongit_list_call_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/s1n/run.py",
        'import subprocess\nsubprocess.run(["python", "run.py"])\n',
    )
    assert '["python", "run.py"]' in run.read_text(encoding="utf-8"), (
        "fixture must call a non-git program"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1" and path == "artifacts/etudes/papers/s1n/run.py"
        for rule, path, _ in findings
    )


def _s1_must_fail(tmp_path, folder: str, code: str) -> None:
    """One bypass form: the fixture holds it and S1 flags the call itself."""
    _registry(tmp_path, [])
    run = _write(tmp_path, f"artifacts/etudes/papers/{folder}/run.py", code)
    assert code in run.read_text(encoding="utf-8"), "fixture must hold the bypass"
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == f"artifacts/etudes/papers/{folder}/run.py"
        and "calls " in msg
        for rule, path, msg in findings
    ), f"{code!r} must fail S1 with a call finding (an import-only flag would mask a dead call detector)"


def test_s1_subprocess_alias_fails(tmp_path):
    _s1_must_fail(tmp_path, "alias", "import subprocess as sp\nsp.run(['ls'])\n")


def test_s1_from_subprocess_import_fails(tmp_path):
    _s1_must_fail(tmp_path, "frm", "from subprocess import run\nrun(['ls'])\n")


def test_s1_from_os_system_alias_fails(tmp_path):
    _s1_must_fail(tmp_path, "oss", "from os import system as s\ns('ls')\n")


def test_s1_os_alias_system_fails(tmp_path):
    _s1_must_fail(tmp_path, "osa", "import os as o\no.system('ls')\n")


def test_s1_os_popen_fails(tmp_path):
    _s1_must_fail(tmp_path, "pop", "import os\nos.popen('ls')\n")


def test_s1_os_exec_fails(tmp_path):
    _s1_must_fail(tmp_path, "exv", "import os\nos.execv('/bin/ls', ['ls'])\n")


def test_s1_os_spawn_fails(tmp_path):
    _s1_must_fail(tmp_path, "spn", "import os\nos.spawnlp(os.P_WAIT, 'ls')\n")


def test_s1_importlib_import_module_fails(tmp_path):
    _s1_must_fail(tmp_path, "ilm", "import importlib\nimportlib.import_module('os')\n")


def test_s1_dunder_import_fails(tmp_path):
    _s1_must_fail(tmp_path, "dim", "mod = __import__('os')\n")


def test_s1_runpy_call_fails(tmp_path):
    _s1_must_fail(tmp_path, "rpy", "import runpy\nrunpy.run_path('x.py')\n")


def test_s1_exec_builtin_fails(tmp_path):
    _s1_must_fail(tmp_path, "exc", "exec('x = 1')\n")


def test_s1_eval_builtin_fails(tmp_path):
    _s1_must_fail(tmp_path, "evl", "y = eval('1 + 1')\n")


def test_s1_git_dash_c_flag_fails(tmp_path):
    _s1_must_fail(
        tmp_path,
        "gcf",
        "import subprocess\nsubprocess.run(['git', '-c', 'x=y', 'rev-parse', 'HEAD'])\n",
    )


def test_s1_git_shell_keyword_fails(tmp_path):
    _s1_must_fail(
        tmp_path,
        "gsh",
        "import subprocess\nsubprocess.run(['git', 'rev-parse', 'HEAD'], shell=True)\n",
    )


def _s2_paper_entry(pid: str, label: str, path: str) -> dict:
    return {
        "id": pid,
        "stage": "P1",
        "stage_artifacts": {"P1": [{"label": label, "path": path}]},
        "preregs": [],
    }


def test_s2_labelled_oracle_import_fails(tmp_path):
    _registry(tmp_path, [_s2_paper_entry(
        "s2lab", "oracle transcription", "artifacts/etudes/s2lab/oracle_trans.py")])
    oracle = _write(tmp_path, "artifacts/etudes/papers/s2lab/oracle_trans.py", "VALUE = 1\n")
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/s2lab/run.py",
        "import oracle_trans\nprint(oracle_trans.VALUE)\n",
    )
    _write(tmp_path, "artifacts/etudes/papers/s2lab/SOURCE_P1.md")
    assert oracle.is_file(), "fixture must hold the labelled oracle"
    assert "import oracle_trans" in run.read_text(encoding="utf-8"), (
        "fixture must import the oracle"
    )
    findings = cr.collect_findings(tmp_path)
    assert (
        "S2",
        "artifacts/etudes/papers/s2lab/run.py",
        "imports oracle artifacts/etudes/papers/s2lab/oracle_trans.py",
    ) in findings


def test_s2_test_gate_and_tests_importers_pass(tmp_path):
    _registry(tmp_path, [_s2_paper_entry(
        "s2ok", "oracle transcription", "artifacts/etudes/s2ok/oracle_trans.py")])
    _write(tmp_path, "artifacts/etudes/papers/s2ok/oracle_trans.py", "VALUE = 1\n")
    test_run = _write(
        tmp_path, "artifacts/etudes/papers/s2ok/test_run.py", "import oracle_trans\n")
    gate_run = _write(
        tmp_path, "artifacts/etudes/papers/s2ok/gate_run.py", "import oracle_trans\n")
    nested = _write(
        tmp_path, "artifacts/etudes/papers/s2ok/tests/check.py", "import oracle_trans\n")
    _write(tmp_path, "artifacts/etudes/papers/s2ok/SOURCE_P1.md")
    for importer in (test_run, gate_run, nested):
        assert "import oracle_trans" in importer.read_text(encoding="utf-8"), (
            "fixture must import the oracle"
        )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S2"] == [], (
        "test_, gate_ and tests/ importers of the oracle stay clean"
    )


def test_s2_ref_name_oracle_fails(tmp_path):
    _registry(tmp_path, [])
    oracle = _write(tmp_path, "artifacts/etudes/papers/q/mod_ref.py", "VALUE = 1\n")
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/q/run.py",
        "import mod_ref\nprint(mod_ref.VALUE)\n",
    )
    assert oracle.is_file(), "fixture must hold the *_ref.py oracle"
    assert "import mod_ref" in run.read_text(encoding="utf-8"), (
        "fixture must import the oracle"
    )
    findings = cr.collect_findings(tmp_path)
    assert (
        "S2",
        "artifacts/etudes/papers/q/run.py",
        "imports oracle artifacts/etudes/papers/q/mod_ref.py",
    ) in findings


def test_s2_relative_and_function_scope_imports_fail(tmp_path):
    _registry(tmp_path, [])
    oracle = _write(tmp_path, "artifacts/etudes/papers/r/sib_ref.py", "VALUE = 1\n")
    rel = _write(
        tmp_path, "artifacts/etudes/papers/r/rel.py", "from .sib_ref import VALUE\n")
    func = _write(
        tmp_path,
        "artifacts/etudes/papers/r/func.py",
        "def f():\n    import sib_ref\n    return sib_ref.VALUE\n",
    )
    assert oracle.is_file(), "fixture must hold the *_ref.py oracle"
    assert "from .sib_ref import" in rel.read_text(encoding="utf-8"), (
        "fixture must hold the relative import"
    )
    assert "import sib_ref" in func.read_text(encoding="utf-8"), (
        "fixture must hold the function-scope import"
    )
    findings = cr.collect_findings(tmp_path)
    assert (
        "S2",
        "artifacts/etudes/papers/r/rel.py",
        "imports oracle artifacts/etudes/papers/r/sib_ref.py",
    ) in findings
    assert (
        "S2",
        "artifacts/etudes/papers/r/func.py",
        "imports oracle artifacts/etudes/papers/r/sib_ref.py",
    ) in findings


def test_s2_oracle_importing_sibling_oracle_passes(tmp_path):
    _registry(tmp_path, [])
    first = _write(
        tmp_path, "artifacts/etudes/papers/s/a_ref.py", "import b_ref\nprint(b_ref.VALUE)\n")
    _write(tmp_path, "artifacts/etudes/papers/s/b_ref.py", "VALUE = 1\n")
    assert "import b_ref" in first.read_text(encoding="utf-8"), (
        "fixture must import the sibling oracle"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S2"] == [], (
        "an oracle importing a sibling oracle of the same paper stays clean"
    )


def test_s2_missing_labelled_oracle_passes(tmp_path):
    _registry(tmp_path, [_s2_paper_entry(
        "ghost", "reference arm", "artifacts/etudes/ghost/missing_oracle.py")])
    run = _write(
        tmp_path, "artifacts/etudes/papers/ghost/run.py", "import missing_oracle\n")
    _write(tmp_path, "artifacts/etudes/papers/ghost/SOURCE_P1.md")
    assert not (tmp_path / "artifacts/etudes/papers/ghost/missing_oracle.py").exists(), (
        "fixture oracle must be missing"
    )
    assert "import missing_oracle" in run.read_text(encoding="utf-8"), (
        "fixture must hold the import"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S2"] == [], (
        "a label naming a missing .py yields no S2 finding"
    )


@pytest.mark.parametrize("source", [
    "import importlib\nm = importlib.import_module('net_ref')\n",
    "import importlib.util\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\n"
    "spec = importlib.util.spec_from_file_location('n', HERE / 'net_ref.py')\n",
], ids=["import_module", "spec_from_file_location"])
def test_s2_dynamic_oracle_load_fails(tmp_path, source):
    _registry(tmp_path, [])
    oracle = _write(tmp_path, "artifacts/etudes/papers/dyn/net_ref.py", "VALUE = 1\n")
    run = _write(tmp_path, "artifacts/etudes/papers/dyn/run.py", source)
    assert oracle.is_file() and "net_ref" in run.read_text(encoding="utf-8"), (
        "fixture must hold the oracle and a dynamic load of it"
    )
    assert (
        "S2",
        "artifacts/etudes/papers/dyn/run.py",
        "imports oracle artifacts/etudes/papers/dyn/net_ref.py",
    ) in cr.collect_findings(tmp_path)


def test_e3_shadow_fails(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "jaxfne/analysis.py", "def tally_signal(x):\n    return x\n")
    _write(
        tmp_path,
        "artifacts/etudes/papers/e1/a.py",
        "def tally_signal(x):\n    return 1\n",
    )
    assert "tally_signal" in cr.analysis_names(tmp_path), (
        "fixture analysis name must be public"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "E3" and path == "artifacts/etudes/papers/e1/a.py"
        for rule, path, _ in findings
    )


def test_e3_nested_module_name_counts(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "jaxfne/analysis/__init__.py", "x = 1\n")
    _write(
        tmp_path, "jaxfne/analysis/sub/deep.py", "def deep_metric(x):\n    return x\n"
    )
    assert "deep_metric" in cr.analysis_names(tmp_path), (
        "nested analysis modules count"
    )


def test_e3_all_reexport_counts(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "jaxfne/analysis.py", '__all__ = ["reexported_thing"]\n')
    assert "reexported_thing" in cr.analysis_names(tmp_path), (
        "__all__ re-exports count without a def"
    )


def test_e3_method_with_core_name_passes(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "jaxfne/analysis.py", "def burst_index(x):\n    return x\n")
    assert "burst_index" in cr.analysis_names(tmp_path), (
        "fixture analysis name must be public"
    )
    _write(
        tmp_path,
        "artifacts/etudes/papers/m1/a.py",
        "class A:\n    def burst_index(self):\n        return 1\n",
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "E3"] == [], (
        "a method named like a core function is not a copy"
    )


def test_r1_no_artifacts_maps_to_id_folder(tmp_path):
    _registry(tmp_path, [_paper("qq", "P1", [])])
    _write(tmp_path, "artifacts/etudes/papers/qq/SOURCE_P1.md")
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R1"] == [], (
        "a paper with no stage_artifacts maps to papers/<id>/"
    )


def test_r1_no_artifacts_tries_id_prefix(tmp_path):
    _registry(tmp_path, [_paper("zz-fig2", "P1", [])])
    _write(tmp_path, "artifacts/etudes/papers/zz/SOURCE_P1.md")
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R1"] == [], (
        "a paper with no stage_artifacts tries the id before the first '-'"
    )


def test_r1_no_artifacts_no_folder_reports(tmp_path):
    _registry(tmp_path, [_paper("ww-at01", "P3", [])])
    assert not (tmp_path / "artifacts/etudes/papers/ww-at01").exists(), (
        "neither the id nor its prefix folder may exist"
    )
    assert not (tmp_path / "artifacts/etudes/papers/ww").exists()
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "R1" and path == "dashboard/sources.json" and "ww-at01" in msg
        for rule, path, msg in findings
    ), "a paper naming no folder must fail R1"


def _e1_paper(pid: str, p4: list[dict] | None, stage: str = "P4",
              cls: str = "in progress") -> dict:
    entry: dict = {"id": pid, "stage": stage, "class": cls, "preregs": []}
    if p4 is None:
        entry["stage_artifacts"] = {
            "P1": [{"label": "source", "path": f"artifacts/etudes/{pid}/SOURCE_P1.md"}]}
    else:
        entry["stage_artifacts"] = {"P4": p4}
    return entry


def _e1_entry(pid: str, label: str, name: str) -> dict:
    return {"label": label, "path": f"artifacts/etudes/{pid}/{name}"}


def test_e1_construct_and_simulate_passes(tmp_path):
    _registry(tmp_path, [_e1_paper("e1ok", [_e1_entry("e1ok", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/e1ok/run.py",
                 "import J\nmodel = J.construct(spec)\nsignals = model.simulate(sim)\n")
    text = run.read_text(encoding="utf-8")
    assert "construct(" in text and ".simulate(" in text, (
        "fixture must call construct( and .simulate("
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "E1"] == [], (
        "a scored module calling both stays clean"
    )


def test_e1_neither_fails(tmp_path):
    _registry(tmp_path, [_e1_paper("e1no", [_e1_entry("e1no", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/e1no/run.py", "VALUE = 1\n")
    text = run.read_text(encoding="utf-8")
    assert "construct(" not in text and ".simulate(" not in text, (
        "fixture must call neither"
    )
    assert (
        "E1",
        "artifacts/etudes/papers/e1no",
        "no scored P4 module calls construct( or .simulate(",
    ) in cr.collect_findings(tmp_path)


def test_e1_construct_only_fails(tmp_path):
    _registry(tmp_path, [_e1_paper("e1c", [_e1_entry("e1c", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/e1c/run.py",
                 "import J\nmodel = J.construct(spec)\n")
    text = run.read_text(encoding="utf-8")
    assert "construct(" in text and ".simulate(" not in text, (
        "fixture must call construct( without .simulate("
    )
    assert (
        "E1",
        "artifacts/etudes/papers/e1c",
        "scored P4 modules call construct( but not .simulate(",
    ) in cr.collect_findings(tmp_path)


def test_e1_oracle_and_test_calls_do_not_count(tmp_path):
    _registry(tmp_path, [_e1_paper("e1o", [
        _e1_entry("e1o", "model", "run.py"),
        _e1_entry("e1o", "oracle transcription", "oracle_mod.py"),
        _e1_entry("e1o", "tests", "test_mod.py"),
    ])])
    _write(tmp_path, "artifacts/etudes/papers/e1o/run.py", "VALUE = 1\n")
    both = "import J\nmodel = J.construct(spec)\nsignals = model.simulate(sim)\n"
    oracle = _write(tmp_path, "artifacts/etudes/papers/e1o/oracle_mod.py", both)
    test_mod = _write(tmp_path, "artifacts/etudes/papers/e1o/test_mod.py", both)
    for path in (oracle, test_mod):
        text = path.read_text(encoding="utf-8")
        assert "construct(" in text and ".simulate(" in text, (
            "fixture must hold both calls outside the scored module"
        )
    assert (
        "E1",
        "artifacts/etudes/papers/e1o",
        "no scored P4 module calls construct( or .simulate(",
    ) in cr.collect_findings(tmp_path)


def test_e1_p3_paper_out_of_scope(tmp_path):
    papers = [_e1_paper("e1p3", [_e1_entry("e1p3", "model", "run.py")], stage="P3")]
    _registry(tmp_path, papers)
    _write(tmp_path, "artifacts/etudes/papers/e1p3/run.py", "VALUE = 1\n")
    assert papers[0]["stage"] == "P3", "fixture must stay below P4"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "E1"] == [], (
        "a P3 paper is out of scope"
    )


def test_e1_core_method_out_of_scope(tmp_path):
    papers = [_e1_paper("e1core", [_e1_entry("e1core", "model", "run.py")],
                        stage="P5", cls="core method")]
    _registry(tmp_path, papers)
    _write(tmp_path, "artifacts/etudes/papers/e1core/run.py", "VALUE = 1\n")
    assert papers[0]["class"] == "core method", "fixture must be a core method"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "E1"] == [], (
        "a core-method entry is out of scope"
    )


def test_e1_no_scored_module_listed(tmp_path):
    papers = [_e1_paper("e1empty", None)]
    _registry(tmp_path, papers)
    assert "P4" not in papers[0]["stage_artifacts"], (
        "fixture must name no P4 module"
    )
    assert (
        "E1",
        "artifacts/etudes/papers/e1empty",
        "no scored P4 module listed",
    ) in cr.collect_findings(tmp_path)


def test_p4_full_writer_passes(tmp_path):
    _registry(tmp_path, [_e1_paper("p4ok", [_e1_entry("p4ok", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/p4ok/run.py",
                 'import json\nmanifest = {"git_sha": "abc", "backend": "cpu", '
                 '"dtype": "float32", "seed": 1, "regime": "AI"}\n'
                 'json.dumps(manifest)\n')
    text = run.read_text(encoding="utf-8")
    for key in ("git_sha", "backend", "dtype", "seed", "regime"):
        assert '"%s"' % key in text, "fixture must hold every P4 key class"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "P4"] == [], (
        "a writer covering all five classes stays clean"
    )


def test_p4_missing_dtype_and_regime_fails(tmp_path):
    _registry(tmp_path, [_e1_paper("p4mr", [_e1_entry("p4mr", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/p4mr/run.py",
                 'manifest = {"git_sha": "abc", "backend": "cpu", "seed": 1}\n')
    text = run.read_text(encoding="utf-8")
    assert '"dtype"' not in text and '"regime"' not in text, (
        "fixture must lack exactly dtype and regime"
    )
    assert (
        "P4",
        "artifacts/etudes/papers/p4mr",
        "manifest writer lacks dtype, regime",
    ) in cr.collect_findings(tmp_path)


def test_p4_no_writer_fails(tmp_path):
    _registry(tmp_path, [_e1_paper("p4nw", [_e1_entry("p4nw", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/p4nw/run.py", "VALUE = 1\n")
    assert "manifest" not in run.read_text(encoding="utf-8"), (
        "fixture must hold no manifest writer"
    )
    assert (
        "P4",
        "artifacts/etudes/papers/p4nw",
        "no manifest writer in the scored P4 modules",
    ) in cr.collect_findings(tmp_path)


def test_p4_unnamed_writer_passed_to_dumps_passes(tmp_path):
    _registry(tmp_path, [_e1_paper("p4un", [_e1_entry("p4un", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/p4un/run.py",
                 'import json\ndata = {"git_sha": "abc", "backend": "cpu", '
                 '"dtype": "float32", "seed": 1, "regime": "AI"}\n'
                 'json.dumps(data)\n')
    text = run.read_text(encoding="utf-8")
    assert "manifest" not in text, "fixture writer must not be named manifest"
    assert "json.dumps(data)" in text, "fixture must pass the writer to json.dumps"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "P4"] == [], (
        "a writer passed to json.dumps counts without the manifest name"
    )


def test_p4_manifest_helper_merges_its_dict_arguments(tmp_path):
    _registry(tmp_path, [_e1_paper("p4hp", [_e1_entry("p4hp", "model", "run.py")])])
    run = _write(tmp_path, "artifacts/etudes/papers/p4hp/run.py",
                 'static = {"baseline_sha": "abc", "backend": "cpu"}\n'
                 'record = {"dtype": "float32", "seeds": [1, 2]}\n'
                 '_write_manifest("out", record, static)\n')
    text = run.read_text(encoding="utf-8")
    assert "_write_manifest(" in text and '"regime"' not in text, (
        "fixture must pass two dicts to a manifest helper and lack only the regime"
    )
    assert (
        "P4",
        "artifacts/etudes/papers/p4hp",
        "manifest writer lacks regime",
    ) in cr.collect_findings(tmp_path)


def test_p4_oracle_only_writer_does_not_count(tmp_path):
    _registry(tmp_path, [_e1_paper("p4or", [
        _e1_entry("p4or", "model", "run.py"),
        _e1_entry("p4or", "oracle transcription", "oracle_mod.py"),
    ])])
    _write(tmp_path, "artifacts/etudes/papers/p4or/run.py", "VALUE = 1\n")
    oracle = _write(tmp_path, "artifacts/etudes/papers/p4or/oracle_mod.py",
                    'manifest = {"git_sha": "abc", "backend": "cpu", '
                    '"dtype": "float32", "seed": 1, "regime": "AI"}\n')
    text = oracle.read_text(encoding="utf-8")
    for key in ("git_sha", "backend", "dtype", "seed", "regime"):
        assert '"%s"' % key in text, "fixture oracle must hold the full writer"
    assert (
        "P4",
        "artifacts/etudes/papers/p4or",
        "no manifest writer in the scored P4 modules",
    ) in cr.collect_findings(tmp_path)


def test_p4_p3_paper_out_of_scope(tmp_path):
    papers = [_e1_paper("p4p3", [_e1_entry("p4p3", "model", "run.py")], stage="P3")]
    _registry(tmp_path, papers)
    _write(tmp_path, "artifacts/etudes/papers/p4p3/run.py", "VALUE = 1\n")
    assert papers[0]["stage"] == "P3", "fixture must stay below P4"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "P4"] == [], (
        "a P3 paper is out of scope"
    )


def test_missing_allowlist_fails_check(tmp_path):
    _registry(tmp_path, [])
    assert not (tmp_path / "scripts" / "replication_known_violations.json").exists(), (
        "fixture must hold no allowlist file"
    )
    assert cr.main(["--root", str(tmp_path), "--check"]) == 1, (
        "a missing allowlist fails closed under --check"
    )
    assert cr.main(["--root", str(tmp_path)]) == 0, "bare run always exits 0"


def test_s1_git_call_passes(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/s1g/git.py",
        'import subprocess\nout = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)\n',
    )
    assert "rev-parse" in run.read_text(encoding="utf-8"), (
        "fixture must hold a git argv call"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S1"] == [], (
        "a git-literal subprocess call does not run authors' code"
    )


def test_clean_tree_passes(tmp_path):
    _registry(tmp_path, [])
    keep = _write(tmp_path, "artifacts/etudes/papers/ok/keep.md")
    assert keep.is_file(), "fixture must hold one benign file"
    assert cr.collect_findings(tmp_path) == []


def test_listed_violation_passes_check(tmp_path):
    _registry(tmp_path, [])
    for i in range(8):
        _write(tmp_path, f"artifacts/etudes/papers/crowded/f{i}.md")
    findings = cr.collect_findings(tmp_path)
    assert [(r, p) for r, p, _ in findings] == [
        ("R6", "artifacts/etudes/papers/crowded")
    ], "fixture must yield exactly one finding"
    _known(tmp_path, [_entry("R6", "artifacts/etudes/papers/crowded")])
    assert cr.main(["--root", str(tmp_path), "--check"]) == 0


def test_new_finding_on_a_listed_path_fails_check(tmp_path):
    """An entry covers one exact finding: the same folder growing past its listed count fails."""
    _registry(tmp_path, [])
    for i in range(9):
        _write(tmp_path, f"artifacts/etudes/papers/crowded/f{i}.md")
    assert [m for _, _, m in cr.collect_findings(tmp_path)] == ["holds 9 files (at most 7)"], (
        "fixture must yield the grown finding"
    )
    _known(tmp_path, [_entry("R6", "artifacts/etudes/papers/crowded")])
    assert cr.main(["--root", str(tmp_path), "--check"]) == 1


def test_stale_entry_fails_check(tmp_path):
    _registry(tmp_path, [])
    assert cr.collect_findings(tmp_path) == [], "fixture must be clean"
    _known(tmp_path, [_entry("R6", "artifacts/etudes/papers/gone")])
    assert cr.main(["--root", str(tmp_path), "--check"]) == 1


def test_live_check_passes():
    assert cr.main(["--check"]) == 0
    assert cr.main([]) == 0, "bare run always exits 0"


def test_replication_audit_is_a_required_gate_family():
    assert "replication_audit" in CHECK_FAMILIES
    assert "etude_no_resim" in CHECK_FAMILIES
    for gate in ("broad", "report", "release", "rc"):
        assert "replication_audit" in GATE_CHECK_FAMILIES[gate], (
            f"{gate} gate must run the replication audit"
        )
        assert "etude_no_resim" in GATE_CHECK_FAMILIES[gate], (
            f"{gate} gate must run the no-resimulation gate"
        )
    steps = dict(AUDIT_STEPS)
    assert steps["replication_audit"][-2:] == [
        "scripts/check_replications.py",
        "--check",
    ]
    assert steps["etude_no_resim"][-2:] == [
        "scripts/gate_etude_no_resim.py",
        "--check",
    ]
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_test_gate.py").read_text(
        encoding="utf-8"
    )
    assert "for family, cmd in AUDIT_STEPS" in source
    assert "_run(cmd, family=family)" in source, (
        "replication_audit is declared but never executed; a declared-only family "
        "cannot fail the gate"
    )


def test_dev_gate_runs_off_main_audit_families(monkeypatch):
    # CI on dev runs `scripts/run_test_gate.py dev` only (ci.yml), so these
    # families must be in the dev set and executed by gate_dev; otherwise
    # nothing runs them before the broad gate on main.
    families = ("record_integrity", "harness_integrity", "replication_audit", "etude_no_resim")
    for family in families:
        assert family in CHECK_FAMILIES
        assert family in GATE_CHECK_FAMILIES["dev"], (
            f"dev gate must run {family} (CI on dev runs the dev gate only)"
        )
        assert family in dict(AUDIT_STEPS)
    ran = []
    monkeypatch.setattr(rtg, "_run", lambda cmd, family=None, **kw: ran.append((family, cmd)))
    rtg.gate_dev()
    assert ran, "gate_dev must call _run"
    for family in families:
        assert [cmd for f, cmd in ran if f == family] == [dict(AUDIT_STEPS)[family]], (
            f"gate_dev must run {family} once, with its AUDIT_STEPS command"
        )
    assert set(GATE_CHECK_FAMILIES["dev"]) == {f for f, _ in ran}, (
        "the dev set and what gate_dev executes must agree"
    )


def test_s1_git_nonconstant_argv_fails(tmp_path):
    _s1_must_fail(
        tmp_path,
        "gnc",
        'import subprocess\nsubprocess.run(["git", "-C", str(root), cmd, "HEAD"])\n',
    )


def test_s1_drive_path_is_outside_on_every_os(tmp_path):
    # Windows and Linux gates must give the same finding for a drive path.
    _registry(tmp_path, [])
    _write(
        tmp_path,
        "artifacts/etudes/papers/drv/run.py",
        'import sys\nsys.path.insert(0, "C:/workspace/elsewhere")\n',
    )
    messages = [m for r, _, m in cr.collect_findings(tmp_path) if r == "S1"]
    assert messages == [
        "line 2: sys.path.insert adds a path outside the repository (C:/workspace/elsewhere)"
    ]


def test_s1_git_computed_dir_after_dash_C_passes(tmp_path):
    _registry(tmp_path, [])
    _write(
        tmp_path,
        "artifacts/etudes/papers/gdir/run.py",
        'import subprocess\nsubprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"])\n',
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S1"] == [], (
        "a computed directory after -C only picks where git runs"
    )


def _s1_spec_must_fail(tmp_path, folder: str, code: str) -> None:
    """One loader form: the fixture holds it and S1 flags the load itself."""
    _registry(tmp_path, [])
    run = _write(tmp_path, f"artifacts/etudes/papers/{folder}/run.py", code)
    assert "spec_from_file_location" in run.read_text(encoding="utf-8"), (
        "fixture must hold the loader"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == f"artifacts/etudes/papers/{folder}/run.py"
        and "spec_from_file_location" in msg
        for rule, path, msg in findings
    ), f"{code!r} must fail S1 with a loader finding"


def test_s1_spec_outside_repo_fails(tmp_path):
    _s1_spec_must_fail(
        tmp_path,
        "spo",
        'import importlib.util\nimportlib.util.spec_from_file_location("m", "/tmp/evil.py")\n',
    )


def test_s1_spec_unresolvable_fails(tmp_path):
    _s1_spec_must_fail(
        tmp_path,
        "spu",
        'import importlib.util\nimportlib.util.spec_from_file_location("m", module_path)\n',
    )


def test_s1_spec_sibling_module_passes(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/sps/run.py",
        'from pathlib import Path\nimport importlib.util\n'
        'HERE = Path(__file__).resolve().parent\n'
        'spec = importlib.util.spec_from_file_location("sib", HERE / "sib.py")\n',
    )
    assert "sib.py" in run.read_text(encoding="utf-8"), (
        "fixture must load a sibling module"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S1"] == [], (
        "a sibling-module load inside the repository stays clean"
    )


def test_s1_getattr_os_alias_fails(tmp_path):
    _s1_must_fail(tmp_path, "gao", 'import os as o\ngetattr(o, "system")("ls")\n')


def test_s1_getattr_subprocess_alias_fails(tmp_path):
    _s1_must_fail(tmp_path, "gas", 'import subprocess as sp\ngetattr(sp, "run")(["ls"])\n')


def test_s1_builtins_dunder_import_fails(tmp_path):
    _s1_must_fail(tmp_path, "bdi", 'import builtins\nbuiltins.__import__("os")\n')


def test_s1_getattr_plain_object_passes(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/gpo/run.py",
        'value = getattr(args, "rate", 1.0)\n',
    )
    assert "getattr(args" in run.read_text(encoding="utf-8"), (
        "fixture must read an attribute off a plain object"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S1"] == [], (
        "getattr on anything but an os/subprocess alias stays clean"
    )


def test_s1_sys_path_alias_insert_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/spa/run.py",
        'import sys as s\ns.path.insert(0, "/nowhere/xyz")\n',
    )
    assert "s.path.insert" in run.read_text(encoding="utf-8"), (
        "fixture must mutate sys.path through an alias"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == "artifacts/etudes/papers/spa/run.py"
        and "sys.path.insert" in msg
        for rule, path, msg in findings
    )


def test_s1_from_sys_import_path_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/sfp/run.py",
        'from sys import path\npath.append("/nowhere/xyz")\n',
    )
    assert "path.append" in run.read_text(encoding="utf-8"), (
        "fixture must mutate the imported sys.path name"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == "artifacts/etudes/papers/sfp/run.py"
        and "sys.path.append" in msg
        for rule, path, msg in findings
    )


def test_s1_sys_path_augassign_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/saa/run.py",
        'import sys\nsys.path += ["/nowhere/xyz"]\n',
    )
    assert "sys.path +=" in run.read_text(encoding="utf-8"), (
        "fixture must extend sys.path with +="
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == "artifacts/etudes/papers/saa/run.py"
        and "sys.path +=" in msg
        for rule, path, msg in findings
    )


def test_s1_sys_path_subscript_assign_fails(tmp_path):
    _registry(tmp_path, [])
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/ssa/run.py",
        'import sys\nsys.path[0] = "/nowhere/xyz"\n',
    )
    assert "sys.path[0]" in run.read_text(encoding="utf-8"), (
        "fixture must assign a sys.path slot"
    )
    findings = cr.collect_findings(tmp_path)
    assert any(
        rule == "S1"
        and path == "artifacts/etudes/papers/ssa/run.py"
        and "sys.path" in msg
        for rule, path, msg in findings
    )


def test_s1_os_startfile_fails(tmp_path):
    _s1_must_fail(tmp_path, "osf", "import os\nos.startfile('x.py')\n")


def test_s1_os_fork_fails(tmp_path):
    _s1_must_fail(tmp_path, "ofk", "import os\nos.fork()\n")


def test_s1_os_posix_spawn_fails(tmp_path):
    _s1_must_fail(
        tmp_path, "opx", "import os\nos.posix_spawnp('ls', ['ls'], None, None, None)\n"
    )


def test_s1_pty_spawn_fails(tmp_path):
    _s1_must_fail(tmp_path, "ptx", "import pty\npty.spawn(['bash'])\n")


def test_s1_git_double_star_keywords_fail(tmp_path):
    _s1_must_fail(
        tmp_path,
        "gkw",
        "import subprocess\nsubprocess.run(['git', 'rev-parse', 'HEAD'], **{'shell': True})\n",
    )


def test_s1_git_upload_pack_fails(tmp_path):
    # Constant argv, yet clone runs the upload-pack command.
    _s1_must_fail(
        tmp_path,
        "gup",
        "import subprocess\nsubprocess.run(['git', 'clone', '--upload-pack=touch x', 'u'])\n",
    )


def test_s1_git_attached_dash_c_fails(tmp_path):
    _s1_must_fail(
        tmp_path,
        "gac",
        "import subprocess\nsubprocess.run(['git', '-ccore.pager=x', 'log'])\n",
    )


def test_s1_git_two_computed_elements_fail(tmp_path):
    _s1_must_fail(
        tmp_path,
        "g2c",
        "import subprocess\nsubprocess.run(['git', '-C', d1, '-C', d2, 'log'])\n",
    )


def test_s1_posix_root_path_is_outside_on_every_os(tmp_path):
    # Windows and Linux gates must give the same finding for a POSIX root path.
    _registry(tmp_path, [])
    _write(
        tmp_path,
        "artifacts/etudes/papers/psx/run.py",
        'import sys\nsys.path.insert(0, "/tmp/elsewhere")\n',
    )
    messages = [m for r, _, m in cr.collect_findings(tmp_path) if r == "S1"]
    assert messages == ["line 2: sys.path.insert adds a path outside the repository (/tmp/elsewhere)"]


#: The PCL port shape: a module-level walk to the nearest pyproject.toml.
_REPO_WALK = '''def _repo_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")
'''


def test_s1_repo_root_walk_is_inside_on_every_os(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "pyproject.toml", "[project]\nname = 'fixture'\n")
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/walk/port/run.py",
        "import sys\nfrom pathlib import Path\n\n" + _REPO_WALK
        + "\n_ROOT = str(_repo_root(Path(__file__).resolve().parent))\n"
        + "sys.path.insert(0, _ROOT)\n",
    )
    assert "_repo_root(Path(__file__)" in run.read_text(encoding="utf-8"), (
        "fixture must build the repo root with the walk"
    )
    parsed = {rel: tree for rel, _, tree in cr.parse_paper_files(tmp_path)}
    assert parsed.get("artifacts/etudes/papers/walk/port/run.py") is not None, (
        "fixture must parse, so S1 examines it"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "S1"] == [], (
        "a sys.path entry built by the repo-root walk stays inside the repository"
    )


def test_s1_repo_root_name_without_walk_fails(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "pyproject.toml", "[project]\nname = 'fixture'\n")
    run = _write(
        tmp_path,
        "artifacts/etudes/papers/fake/run.py",
        "import sys\nfrom pathlib import Path\n\n"
        "def _repo_root(start: Path) -> Path:\n"
        '    return Path("C:/elsewhere")\n\n'
        "sys.path.insert(0, str(_repo_root(Path(__file__).resolve().parent)))\n",
    )
    assert 'Path("C:/elsewhere")' in run.read_text(encoding="utf-8"), (
        "fixture must return a fixed outside path from a function named like the walk"
    )
    assert any(
        rule == "S1"
        and path == "artifacts/etudes/papers/fake/run.py"
        and "sys.path.insert adds a path that cannot be resolved" in msg
        for rule, path, msg in cr.collect_findings(tmp_path)
    ), "a function named _repo_root without the walk shape must stay unresolvable"


def test_s1_repo_root_walk_joined_outside_fails(tmp_path):
    _registry(tmp_path, [])
    _write(tmp_path, "pyproject.toml", "[project]\nname = 'fixture'\n")
    walk_rel = "artifacts/etudes/papers/esc/port/walk_up.py"
    ctrl_rel = "artifacts/etudes/papers/esc/port/parents_up.py"
    _write(
        tmp_path,
        walk_rel,
        "import sys\nfrom pathlib import Path\n\n" + _REPO_WALK
        + "\n_ROOT = str(_repo_root(Path(__file__).resolve().parent) / \"..\")\n"
        + "sys.path.insert(0, _ROOT)\n",
    )
    _write(
        tmp_path,
        ctrl_rel,
        "import sys\nfrom pathlib import Path\n\n"
        '_ROOT = str(Path(__file__).resolve().parents[5] / "..")\n'
        "sys.path.insert(0, _ROOT)\n",
    )
    assert "parents[5]" in (tmp_path / ctrl_rel).read_text(encoding="utf-8"), (
        "control must hold a parents[k] chain joined with .."
    )
    flagged = {
        path: msg
        for rule, path, msg in cr.collect_findings(tmp_path)
        if rule == "S1"
    }
    assert set(flagged) == {walk_rel, ctrl_rel}, (
        "a .. join escaping the repository is flagged for the walk and for parents[k]"
    )
    assert all("outside the repository" in flagged[p] for p in (walk_rel, ctrl_rel)), (
        "the walk escape is named outside the repository, as the parents[k] escape is"
    )


def _r4_paper(pid: str, verdict, stage: str = "P6", p6: list[dict] | None = None) -> dict:
    entry: dict = {
        "id": pid,
        "stage": stage,
        "stage_artifacts": {
            "P1": [{"label": "source", "path": f"artifacts/etudes/papers/{pid}/SOURCE_P1.md"}]},
        "preregs": [],
        "verdict": verdict,
    }
    if p6 is not None:
        entry["stage_artifacts"]["P6"] = p6
    return entry


def _r4_p6_records(root: Path, pid: str) -> None:
    """Every R1 record a stage-P6 paper needs, including the P6 result file."""
    for name in ("SOURCE_P1.md", "PORT_SPEC.md", "PREREG_P3.md",
                 "LEDGER.md", "RESULT_P3.md", "RESULT_P6.md"):
        _write(root, f"artifacts/etudes/papers/{pid}/{name}")


def test_r4_reproduced_without_p6_entry_fails(tmp_path):
    papers = [_r4_paper("r4noentry", "reproduced")]
    _registry(tmp_path, papers)
    _r4_p6_records(tmp_path, "r4noentry")
    assert "P6" not in papers[0]["stage_artifacts"], (
        "fixture must list no P6 stage_artifacts entry"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R1"] == [], (
        "the folder holds every R1 record, so only R4 may fire"
    )
    assert (
        "R4",
        "artifacts/etudes/papers/r4noentry",
        "paper 'r4noentry' claims reproduced without a P6 result file",
    ) in cr.collect_findings(tmp_path)


def test_r4_reproduced_with_missing_p6_file_fails(tmp_path):
    papers = [_r4_paper("r4miss", "reproduced",
                        p6=[{"label": "result P6",
                             "path": "artifacts/etudes/papers/r4miss/RESULT_P6_v2.md"}])]
    _registry(tmp_path, papers)
    _r4_p6_records(tmp_path, "r4miss")
    assert not (tmp_path / "artifacts/etudes/papers/r4miss/RESULT_P6_v2.md").exists(), (
        "fixture P6 entry must name a file absent from disk"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R1"] == [], (
        "the folder holds every R1 record, so only R4 may fire"
    )
    assert (
        "R4",
        "artifacts/etudes/papers/r4miss",
        "paper 'r4miss' claims reproduced without a P6 result file",
    ) in cr.collect_findings(tmp_path)


def test_r4_reproduced_with_p6_file_passes(tmp_path):
    papers = [_r4_paper("r4ok", "reproduced",
                        p6=[{"label": "result P6",
                             "path": "artifacts/etudes/papers/r4ok/RESULT_P6.md"}])]
    _registry(tmp_path, papers)
    _r4_p6_records(tmp_path, "r4ok")
    assert (tmp_path / "artifacts/etudes/papers/r4ok/RESULT_P6.md").is_file(), (
        "fixture P6 entry must resolve on disk"
    )
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R4"] == [], (
        "a reproduced paper whose P6 entry resolves stays clean"
    )


def test_r4_reproduced_below_p6_out_of_scope(tmp_path):
    papers = [_r4_paper("r4p5", "reproduced", stage="P5")]
    _registry(tmp_path, papers)
    for name in ("SOURCE_P1.md", "PORT_SPEC.md", "PREREG_P3.md",
                 "LEDGER.md", "RESULT_P3.md"):
        _write(tmp_path, f"artifacts/etudes/papers/r4p5/{name}")
    assert papers[0]["stage"] == "P5", "fixture must stay below P6"
    assert "P6" not in papers[0]["stage_artifacts"], "fixture must list no P6 entry"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R4"] == [], (
        "a reproduced claim below P6 is not an R4 finding"
    )


@pytest.mark.parametrize("verdict", [
    "reproduced (conditional)", "mixed", "in progress", "not started", "failed",
])
def test_r4_non_replication_verdicts_pass(tmp_path, verdict):
    papers = [_r4_paper("r4nv", verdict)]
    _registry(tmp_path, papers)
    _r4_p6_records(tmp_path, "r4nv")
    assert papers[0]["verdict"] == verdict, "fixture must hold the verdict it is named after"
    assert "P6" not in papers[0]["stage_artifacts"], "fixture must list no P6 entry"
    assert [f for f in cr.collect_findings(tmp_path) if f[0] == "R4"] == [], (
        f"verdict {verdict!r} is a stage-level claim, not a full-replication claim"
    )


def _paper_records() -> list[Path]:
    """Every paper markdown record except the signed or historical ones."""
    return [
        md for md in sorted(PAPERS_DIR.rglob("*.md"))
        if not any(fnmatch.fnmatch(md.name, g) for g in SIGNED_RECORD_GLOBS)
        and "prereg" not in md.relative_to(PAPERS_DIR).as_posix().lower()
    ]


def _defined_names(source: Path) -> set[str]:
    """Module-level defs, classes and assignments, plus Class.method, read with ast."""
    names: set[str] = set()
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                names.update(
                    f"{node.name}.{sub.name}" for sub in node.body
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_paper_records_cite_no_core_line_numbers():
    records = _paper_records()
    assert records, "fixture must scan at least one paper record"
    hits = []
    for md in records:
        rel = md.relative_to(ROOT).as_posix()
        for n, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            hits += [f"{rel}:{n} {m.group(0)}" for m in LINE_CITE.finditer(line)]
    assert not hits, "core code cited by line number (cite a symbol): " + "; ".join(hits)


def test_backticked_core_symbols_resolve_at_head():
    examined = []
    unresolved = []
    for md in _paper_records():
        rel = md.relative_to(ROOT).as_posix()
        for name, path in SYMBOL_CITE.findall(md.read_text(encoding="utf-8")):
            examined.append(f"{rel}: `{name}` in {path}")
            source = ROOT / path
            if not source.is_file() or name not in _defined_names(source):
                unresolved.append(f"{rel}: `{name}` is not defined in {path}")
    assert examined, "no `name` in `jaxfne/<path>.py` citation was examined"
    assert not unresolved, "; ".join(unresolved)
