"""artifacts/ root holds only the control files; outputs live in subfolders (H4).

Library defaults and scripts that wrote viewers/findings into the root made it
grow without bound. A new root file needs a reason here.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT_FILES = {
    "AGENTS.md", "README.md", "context.md", "fact_stack.md", "memory.md", "todo_stack.md",
    # state roles of the shared workspace layout (meet 204dfdaa, Hamm 2026-10-12)
    "goal_stack.md", "state.md",
    # open-items role (keeper proposal, Hamm 2026-10-12)
    "open_stack.md",
}


def test_artifacts_root_holds_only_control_files():
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "artifacts"], capture_output=True, text=True, check=True
    ).stdout.split()
    loose = {p.split("/", 1)[1] for p in tracked if p.count("/") == 1}
    assert loose <= ROOT_FILES, f"move into a subfolder: {sorted(loose - ROOT_FILES)}"


def test_readme_lists_every_tracked_folder():
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "artifacts"], capture_output=True, text=True, check=True
    ).stdout.split()
    folders = {p.split("/")[1] for p in tracked if p.count("/") >= 2}
    readme = (REPO / "artifacts" / "README.md").read_text(encoding="utf-8")
    # Only the first cell of a table row lists a folder; mentions in other cells do not count.
    listed = {name for line in readme.splitlines() if line.startswith("| ")
              for name in re.findall(r"`([^`/]+)/`", line.split("|")[1])}
    assert len(folders) > 20, "fixture must see the artifacts folders"
    missing = sorted(folders - listed)
    assert not missing, f"add to artifacts/README.md: {missing}"


def test_viewer_defaults_write_into_a_subfolder():
    import inspect

    from jaxfne.vis.column_viewer import render_column_viewer
    from jaxfne.vis.pseudogenome_viewer import render_pseudogenome_development_viewer

    for fn in (render_column_viewer, render_pseudogenome_development_viewer):
        default = Path(inspect.signature(fn).parameters["output_path"].default)
        assert default.parent != Path("artifacts"), f"{fn.__name__} writes into artifacts/ root"


# P0/P1 reorganization (Hamm 2026-10-15): low-reference folders grouped by kind.
MOVED_FOLDERS = {
    "assurance/private_acceptance",
    "assurance/issue_log",
    "runs/benchmark",
    "runs/mcc3_10s_checkpoint",
    "protocols/hdp/hdp_k_w_ctrl_sweep",
    "protocols/hdp/hdp_v2_rho_sweep",
    "protocols/hdp/hdp_presets",
}

OLD_FOLDERS = {
    "private_acceptance",
    "issue_log",
    "benchmark",
    "mcc3_10s_checkpoint",
    "hdp_k_w_ctrl_sweep",
    "hdp_v2_rho_sweep",
    "hdp_presets",
}


def test_p0_p1_folders_live_at_new_layout():
    for rel in sorted(MOVED_FOLDERS):
        assert (REPO / "artifacts" / rel).is_dir(), f"missing after P0/P1 move: {rel}"
    for rel in sorted(OLD_FOLDERS):
        assert not (REPO / "artifacts" / rel).exists(), f"stale after P0/P1 move: {rel}"


# P2 reorganization (Hamm 2026-10-15): perf/ and programme/ split by kind.
# P4 moved perf/ under runs/ (Hamm 2026-10-15 packet); programme/ stays.
P2_FOLDERS = {
    "runs/perf/baselines",
    "runs/perf/matrices_051",
    "runs/perf/matrices_055",
    "runs/perf/matrices_055/quiet",
    "runs/perf/matrices_055/rerun",
    "runs/perf/profiles",
    "runs/perf/scaling",
    "runs/perf/workflow",
    "programme/coverage",
    "programme/gaps",
    "programme/design/plans",
    "programme/design/capabilities",
    "programme/receipts/v0_4_23",
    "programme/receipts/v0_4_24",
    "programme/receipts/tfne",
    "programme/receipts/engine_053_054",
    "programme/receipts/investigations",
}

# Dated files keep the paths they were written with.
P2_PROGRAMME_ROOT_FILES = {
    "handout_opencode_2026-09-29.md",
    "metrics_definitions_2026-10-13.md",
    "refreeze_p016_p020_2026-09-30.md",
}

P2_KEY_FILES = {
    "runs/perf/baselines/baseline_050.json",
    "runs/perf/matrices_051/matrix_051.json",
    "runs/perf/matrices_051/matrix_051_spec.json",
    "runs/perf/matrices_055/matrix_055.json",
    "runs/perf/matrices_055/matrix_055_spec.json",
    "runs/perf/scaling/w11_atlas_10k.jsonl",
    "programme/coverage/atlas_coverage.json",
    "programme/coverage/claim_ledger_055.md",
    "programme/coverage/methods_055.md",
}


def test_p2_folders_live_at_new_layout():
    for rel in sorted(P2_FOLDERS):
        assert (REPO / "artifacts" / rel).is_dir(), f"missing after P2 move: {rel}"
    for rel in sorted(P2_KEY_FILES):
        assert (REPO / "artifacts" / rel).is_file(), f"missing after P2 move: {rel}"
    perf_loose = {p.name for p in (REPO / "artifacts" / "runs" / "perf").iterdir() if p.is_file()}
    assert perf_loose == set(), f"stale after P2 move: {sorted(perf_loose)}"
    prog_loose = {p.name for p in (REPO / "artifacts" / "programme").iterdir() if p.is_file()}
    assert prog_loose == P2_PROGRAMME_ROOT_FILES, f"unexpected programme/ root: {sorted(prog_loose)}"


# P3 reorganization (Hamm 2026-10-15): etudes/ and tutorials/ grouped by kind.
P3_ETUDES_ROOT = {
    "bundles",
    "core_checks",
    "experiment_a",
    "jdna",
    "meta",
    "papers",
    "theory",
}

P3_TUTORIALS_ROOT = {
    "etudes",
    "intro",
    "status",
    "suite",
    "templates",
    "versioned",
}

P3_OLD_ETUDES = {
    "caminos",
    "gsdr2026",
    "izhikevich2003",
    "pcl",
    "pcl_h",
    "qlearning",
    "hdp_balance",
    "hdp_controllability_reachability",
    "heterogeneous_emitters",
    "multiscale_observation",
    "theory_f1",
    "theory_f2",
    "theory_f4",
    "jdna-compact-grammar",
}

P3_KEY_FILES = {
    "etudes/papers/caminos/CALIBRATION.md",
    "etudes/papers/pcl/README.md",
    "etudes/bundles/hdp_balance/README.md",
    "etudes/theory/theory_f1/README.md",
    "etudes/jdna/jdna-compact-grammar/parser.py",
    "etudes/meta/REPLICATIONS.md",
    "tutorials/suite/jaxfne_suite_no_1_computational_biophysics.ipynb",
    "tutorials/intro/jaxfne_neuronal_tensor_first.ipynb",
    "tutorials/versioned/jaxfne_v031_single_neuron.ipynb",
    "tutorials/status/NOTEBOOK_STATUS.md",
}


def test_p3_folders_live_at_new_layout():
    etudes_root = {p.name for p in (REPO / "artifacts" / "etudes").iterdir()}
    assert etudes_root == P3_ETUDES_ROOT, f"unexpected etudes/ root: {sorted(etudes_root)}"
    tut_root = {p.name for p in (REPO / "artifacts" / "tutorials").iterdir()}
    assert tut_root == P3_TUTORIALS_ROOT, f"unexpected tutorials/ root: {sorted(tut_root)}"
    for rel in sorted(P3_OLD_ETUDES):
        assert not (REPO / "artifacts" / "etudes" / rel).exists(), f"stale after P3 move: {rel}"
    for rel in sorted(P3_KEY_FILES):
        assert (REPO / "artifacts" / rel).is_file(), f"missing after P3 move: {rel}"
    # papers/ holds one folder per lane and is exempt (test_artifacts_tree_degree.py states why).
    for sub in ("bundles", "theory", "jdna", "meta"):
        entries = list((REPO / "artifacts" / "etudes" / sub).iterdir())
        assert len(entries) <= 7, f"etudes/{sub} holds {len(entries)} entries"
    # experiment_a/ stays as written (frozen cited path); tutorials/etudes/ keeps
    # the lane notebook series and tutorials/versioned/ the versioned series.
    for sub in ("intro", "status", "suite", "templates"):
        entries = list((REPO / "artifacts" / "tutorials" / sub).iterdir())
        assert len(entries) <= 7, f"tutorials/{sub} holds {len(entries)} entries"


# P4 reorganization (Hamm 2026-10-15; amended on review the same day): perf/ grouped
# under runs/; audit/ split by kind inside it. A folder a frozen record cites by path
# stays: atlas/ (the frozen 0.5.5 agent benchmark artifacts/runs/benchmark/agent_tasks_055.json
# cites artifacts/atlas/... and its spec_digests are frozen), e2/ (preregistered records
# cite their own paths and hashes), audit/v0417_a_capability_audit.json (cited by the
# frozen publication/publication_evidence_index.json). protocol_c/,
# protocol_d_biological_rbs/, protocol_e_integration/, protocol_h_rbd/ and
# protocol_w/ stay (jaxfne/ runtime defaults + FROZEN supplement rows), as do
# programme/, etudes/, tutorials/ and changelog.d/ (artifacts/README.md).
P4_RUNS_ROOT = {
    "benchmark",
    "mcc3_10s_checkpoint",
    "perf",
}

P4_OLD_ROOT = {
    "perf",
}

P4_AUDIT_ROOT = {
    "reviews",
    "studies",
    "probe_results",
    "capability",
    "baselines",
}

P4_KEY_FILES = {
    "atlas/at_manifest.py",
    "atlas/results/at10_n20_055.json",
    "e2/E2_SYNTHESIS.md",
    "e2/preregistration/e2_ping_prereg.json",
    "runs/perf/matrices_055/matrix_055.json",
    "audit/reviews/deep_review_89_2026-10-04.md",
    "audit/studies/jomission_downstream_lessons.md",
    "audit/probe_results/h_representation_probe_results.json",
    "audit/v0417_a_capability_audit.json",
    "audit/baselines/w8_baseline.json",
}


def test_p4_folders_live_at_new_layout():
    runs_root = {p.name for p in (REPO / "artifacts" / "runs").iterdir()}
    assert runs_root == P4_RUNS_ROOT, f"unexpected runs/ root: {sorted(runs_root)}"
    for rel in sorted(P4_OLD_ROOT):
        assert not (REPO / "artifacts" / rel).exists(), f"stale after P4 move: {rel}"
    audit_root = {p.name for p in (REPO / "artifacts" / "audit").iterdir() if p.is_dir()}
    assert audit_root == P4_AUDIT_ROOT, f"unexpected audit/ root: {sorted(audit_root)}"
    for rel in sorted(P4_KEY_FILES):
        assert (REPO / "artifacts" / rel).is_file(), f"missing after P4 move: {rel}"
    for sub in sorted(P4_AUDIT_ROOT):
        entries = list((REPO / "artifacts" / "audit" / sub).iterdir())
        assert len(entries) <= 7, f"audit/{sub} holds {len(entries)} entries"
