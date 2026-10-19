"""Tree degree under artifacts/: at most 7 direct folders and 7 direct files per folder (Hamm 2026-10-15).

Counts tracked files only (git ls-files). Dot entries are exempt (Hamm 2026-10-15). A folder may
exceed the limit only when it is listed below with the reason it cannot be split: a frozen record,
the package, the harness or an external tool addresses its paths, or it is one generated series.
An exception that no longer exceeds the limit, or names a folder that is gone, fails too.
"""

import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIMIT = 7

# artifacts/ root entries that do not count toward the root limit.
ROOT_EXEMPT = {
    # control stacks, addressed by path by the harness and the global rules
    # (tests/test_artifacts_root_layout.py::ROOT_FILES)
    "AGENTS.md": "control stack", "README.md": "control stack", "context.md": "control stack",
    "fact_stack.md": "control stack", "goal_stack.md": "control stack", "memory.md": "control stack",
    "open_stack.md": "control stack", "state.md": "control stack",
    "todo_stack.md": "control stack",
    "publication": "frozen_manifest.json lives here and keys 21 publication/ paths",
    "figures": "frozen_manifest.json keys figures/publication/fig01-fig07",
    "release": "current_release_authorities.json is frozen in place (todo stack, 'Frozen, do not move')",
    "archive": "record class root in scripts/check_record_integrity.py (artifacts/archive/)",
    "legacy": "record class root in scripts/check_record_integrity.py; kept by Hamm 2026-10-11",
    "protocol_c": "frozen publication/fig05_wave_spec.json and publication_evidence_index.json cite it",
    "protocol_h_rbd": "frozen publication/fig06_hwd_spec.json and publication_evidence_index.json cite it",
    "protocol_w": "jaxfne/ runtime receipt paths; frozen publication/fig06_hwd_spec.json cites it",
    "protocol_d_biological_rbs": "jaxfne/ runtime paths; frozen publication/fig06_hwd_spec.json cites it",
    "protocol_e_integration": "jaxfne/ runtime paths; frozen publication/fig01 and fig07 specs cite it",
    "audit": "frozen publication/publication_evidence_index.json cites audit/v0417_a_capability_audit.json",
    "etudes": "jaxfne/ defaults and frozen publication/fig02-fig04 receipts cite etudes/experiment_a/",
    "viewers": "default output_path of the jaxfne.vis viewers",
    "project_sources": "jaxfne/tfne.py cites project_sources/7_tfne_algebra.md",
    "atlas": "frozen 0.5.5 benchmark runs/benchmark/agent_tasks_055.json cites atlas/ scripts",
    "e2": "preregistered records cite their own e2/ paths and hashes",
}

# Folders allowed over the limit, with everything below them.
EXEMPT_TREES = {
    "publication": "frozen manifest and the write-once evidence its receipts cite by path",
    "e2": "preregistered records cite their own paths and hashes",
    "legacy": "record class (scripts/check_record_integrity.py); kept by Hamm 2026-10-11",
    "etudes/papers/pcl": "the pinned record prereg/P3_fig3_diagnostic.md, dashboard/data and "
                         "docs/_static/atlas_registry.json cite its subfolders by path; open lane",
}

# Single folders allowed over the limit (their subfolders are still checked).
EXEMPT_FOLDERS = {
    "": "root: see ROOT_EXEMPT; the remaining entries must fit the limit",
    "figures/publication/final": "publication/polish/ and publication/final/ receipts cite each render",
    "release": "current_release_authorities.json (frozen in place) cites receipts here; "
               "past receipts keep the paths they were written with",
    "atlas": "the four scripts the frozen 0.5.5 benchmark cites and the sibling modules they import",
    "atlas/results": "one 0.5.5 result series, cited 13 times by programme/coverage/atlas_coverage.json",
    "project_sources": "numbered specification set 1-8 plus README; jaxfne/tfne.py cites part 7",
    "protocol_c": "frozen fig05_wave_spec.json and publication_evidence_index.json cite these files",
    "protocol_d_biological_rbs": "jaxfne/ runtime paths and frozen fig06_hwd_spec.json cite these files",
    "protocol_e_integration": "jaxfne/ runtime paths and frozen fig01/fig07 specs cite these files",
    "changelog.d": "one fragment per change since the last release; "
                   "scripts/assemble_changelog.py deletes them at release",
    "control/skills": "one folder per skill: sync_skills.py and check_harness_integrity.py resolve "
                      "control/skills/<name>/SKILL.md",
    "etudes/experiment_a": "frozen publication/fig02-fig04 receipts and jaxfne/ defaults cite these files",
    "etudes/papers": "one folder per paper lane (T-32 target: about one a day, Hamm 2026-10-19); "
                     "grouping lanes by year or status would split one registry (reasoned exemption)",
    "etudes/papers/caminos": "open replication lane (CAMINOS lag); split when the lane closes",
    "etudes/papers/izhikevich2003": "open replication lane; split when the lane closes",
    "etudes/bundles/hdp_balance": "preregistered stage record: README declares each confirmatory "
                                  "output path; docs/_static/atlas_registry.json copies the figures",
    "etudes/bundles/multiscale_observation": "one generated bundle: "
                                             "scripts/run_multiscale_observation_etude.py writes all files",
    # O-18: release- or script-generated series kept whole so each set stays addressable
    "programme/receipts/v0_4_23": "0.4.23 receipt series",
    "programme/receipts/v0_4_24": "0.4.24 receipt series",
    "programme/receipts/tfne": "TFNE receipt series",
    "programme/receipts/engine_053_054": "0.5.3-0.5.4 engine receipt series",
    "runs/perf/matrices_055/rerun": "matrix 0.5.5 rerun cell series",
    "tutorials/etudes": "lane notebook series",
    "tutorials/versioned": "versioned notebook series",
}


def _degrees():
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "artifacts"], capture_output=True, check=True
    ).stdout.decode("utf-8").split("\0")
    dirs, files = defaultdict(set), defaultdict(set)
    for path in filter(None, tracked):
        parts = path.split("/")[1:]
        if any(p.startswith(".") for p in parts):
            continue
        for i in range(len(parts)):
            parent = "/".join(parts[:i])
            (files if i == len(parts) - 1 else dirs)[parent].add(parts[i])
    return dirs, files


def _over(dirs, files, folder):
    exempt = ROOT_EXEMPT if folder == "" else {}
    d = {n for n in dirs[folder] if n not in exempt}
    f = {n for n in files[folder] if n not in exempt}
    return len(d) > LIMIT or len(f) > LIMIT, len(d), len(f)


def _in_tree(folder):
    return any(folder == t or folder.startswith(t + "/") for t in EXEMPT_TREES)


def test_no_folder_exceeds_seven_direct_folders_or_files():
    dirs, files = _degrees()
    folders = set(dirs) | set(files)
    assert len(folders) > 100, "fixture must see the artifacts tree"
    over = []
    for folder in sorted(folders):
        if folder in EXEMPT_FOLDERS or _in_tree(folder):
            continue
        bad, nd, nf = _over(dirs, files, folder)
        if bad:
            over.append(f"artifacts/{folder}: {nd} folders, {nf} files")
    assert not over, "split by kind or add an exception with its reason:\n" + "\n".join(over)


def test_root_holds_at_most_seven_unexempted_folders():
    dirs, files = _degrees()
    bad, nd, nf = _over(dirs, files, "")
    assert not bad, f"artifacts/ root: {nd} unexempted folders, {nf} unexempted files"
    assert dirs[""] & set(ROOT_EXEMPT), "fixture must see the exempt root folders"


def test_every_exception_is_live_and_needed():
    dirs, files = _degrees()
    folders = set(dirs) | set(files)
    for name in ROOT_EXEMPT:
        assert name in dirs[""] | files[""], f"ROOT_EXEMPT names a missing entry: {name}"
    for tree in EXEMPT_TREES:
        assert tree in folders, f"EXEMPT_TREES names a missing folder: {tree}"
        inside = [f for f in folders if f == tree or f.startswith(tree + "/")]
        assert any(_over(dirs, files, f)[0] for f in inside), f"tree fits the limit now: {tree}"
    for folder in EXEMPT_FOLDERS:
        if folder == "":
            continue
        assert folder in folders, f"EXEMPT_FOLDERS names a missing folder: {folder}"
        assert _over(dirs, files, folder)[0], f"folder fits the limit now, drop it: {folder}"
    for table in (ROOT_EXEMPT, EXEMPT_TREES, EXEMPT_FOLDERS):
        assert all(reason.strip() for reason in table.values()), "every exception needs a reason"
