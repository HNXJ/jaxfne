"""End-to-end scientific tasks route to the jaxfne-workflow skill."""

from pathlib import Path


def test_task_router_routes_end_to_end_science_to_workflow_skill():
    context = Path("artifacts/context.md").read_text(encoding="utf-8")

    assert "## Task router (canonical)" in context
    router_section = context.split("## Task router (canonical)", 1)[1]
    assert "artifacts/skills/jaxfne-workflow/SKILL.md" in router_section, (
        "Task router must send end-to-end scientific tasks to jaxfne-workflow"
    )
