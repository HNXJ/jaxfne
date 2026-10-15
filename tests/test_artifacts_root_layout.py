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
    "goal_stack.md", "problem_stack.md", "state.md",
    # open-items role (keeper proposal, Hamm 2026-10-12)
    "open_stack.md",
    # pinned by tests/test_public_surface_contract_v0413.py, test_public_api_snapshot_v034.py
    "public_api_before.json", "public_surface_contract_v0413.json",
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
P2_FOLDERS = {
    "perf/baselines",
    "perf/matrices_051",
    "perf/matrices_055",
    "perf/matrices_055/quiet",
    "perf/matrices_055/rerun",
    "perf/profiles",
    "perf/scaling",
    "perf/workflow",
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
    "perf/baselines/baseline_050.json",
    "perf/matrices_051/matrix_051.json",
    "perf/matrices_051/matrix_051_spec.json",
    "perf/matrices_055/matrix_055.json",
    "perf/matrices_055/matrix_055_spec.json",
    "perf/scaling/w11_atlas_10k.jsonl",
    "programme/coverage/atlas_coverage.json",
    "programme/coverage/claim_ledger_055.md",
    "programme/coverage/methods_055.md",
}


def test_p2_folders_live_at_new_layout():
    for rel in sorted(P2_FOLDERS):
        assert (REPO / "artifacts" / rel).is_dir(), f"missing after P2 move: {rel}"
    for rel in sorted(P2_KEY_FILES):
        assert (REPO / "artifacts" / rel).is_file(), f"missing after P2 move: {rel}"
    perf_loose = {p.name for p in (REPO / "artifacts" / "perf").iterdir() if p.is_file()}
    assert perf_loose == set(), f"stale after P2 move: {sorted(perf_loose)}"
    prog_loose = {p.name for p in (REPO / "artifacts" / "programme").iterdir() if p.is_file()}
    assert prog_loose == P2_PROGRAMME_ROOT_FILES, f"unexpected programme/ root: {sorted(prog_loose)}"
