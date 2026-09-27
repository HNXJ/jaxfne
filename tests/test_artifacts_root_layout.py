"""artifacts/ root holds only the control files; outputs live in subfolders (H4).

Library defaults and scripts that wrote viewers/findings into the root made it
grow without bound. A new root file needs a reason here.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT_FILES = {
    "AGENTS.md", "README.md", "context.md", "fact_stack.md", "memory.md", "todo_stack.md",
    # pinned by tests/test_public_surface_contract_v0413.py, test_public_api_snapshot_v034.py
    "public_api_before.json", "public_surface_contract_v0413.json",
}


def test_artifacts_root_holds_only_control_files():
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "artifacts"], capture_output=True, text=True, check=True
    ).stdout.split()
    loose = {p.split("/", 1)[1] for p in tracked if p.count("/") == 1}
    assert loose <= ROOT_FILES, f"move into a subfolder: {sorted(loose - ROOT_FILES)}"


def test_viewer_defaults_write_into_a_subfolder():
    import inspect

    from jaxfne.vis.column_viewer import render_column_viewer
    from jaxfne.vis.pseudogenome_viewer import render_pseudogenome_development_viewer

    for fn in (render_column_viewer, render_pseudogenome_development_viewer):
        default = Path(inspect.signature(fn).parameters["output_path"].default)
        assert default.parent != Path("artifacts"), f"{fn.__name__} writes into artifacts/ root"
