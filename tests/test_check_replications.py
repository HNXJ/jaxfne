"""Replication-checklist gate tests: per-rule fixtures under tmp_path.

Each rule test builds the violating tree it is named after, asserts the
fixture precondition, then collects findings end to end through
``collect_findings`` (the same entry point ``main`` uses), so killing a
check body or its call site fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import check_replications as cr
from scripts import run_test_gate as rtg
from scripts.run_test_gate import AUDIT_STEPS, CHECK_FAMILIES, GATE_CHECK_FAMILIES


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
