"""v0.3.42 evidence inventory/context path compatibility checks."""

from pathlib import Path
import subprocess
import sys


def test_evidence_inventory_wrapper_command_exists():
    assert Path("scripts/evidence_inventory.py").is_file()
    assert Path("scripts/evidence_figures_inventory.py").is_file()


def test_agent_context_inventory_commands_resolve():
    command_refs = []
    for path in [
        Path("AGENTS.md"),
        Path(".github/CONTRIBUTING.md"),
        Path("docs/contributing.md"),
        Path("internal_docs/loop_context/AGENT_QUICKREF.md"),
    ]:
        # internal_docs/ is gitignored (internal-only), so it is absent in clean
        # checkouts / CI; only assert on context files that are actually present.
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "scripts/evidence_inventory.py" in text:
            command_refs.append("scripts/evidence_inventory.py")
        if "scripts/evidence_figures_inventory.py" in text:
            command_refs.append("scripts/evidence_figures_inventory.py")
    assert command_refs
    for ref in command_refs:
        assert Path(ref).is_file(), f"missing command referenced by context: {ref}"


def test_inventory_accepts_publication_figure_compatibility_paths():
    from scripts.evidence_inventory import build_inventory

    inventory = build_inventory()
    summary = inventory["summary"]
    assert "figures/evidence" in summary["figure_dir_candidates"]
    assert "figures/publication" in summary["figure_dir_candidates"]
    # In source zips that already contain publication PNGs but have not rerun the
    # evidence generators, inventory should not report a false 0/18 state.
    # The v0.3.42-era 8-main/10-extended scheme was superseded by the committed
    # publication figure set (7 main figures); extended-data figures are
    # assigned during Supplement assembly, so the extended assertion tracks
    # whatever the inventory enumerates rather than a stale fixed floor.
    if Path("figures/publication").is_dir():
        assert summary["main_figures_present"] >= 7
        assert summary["extended_data_present"] >= summary["extended_data_total"]


def test_legacy_inventory_wrapper_executes(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/evidence_figures_inventory.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jaxfne evidence inventory" in result.stdout
