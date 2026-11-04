"""The dev extra installs what the broad gate's scripts import.

CI (Fast) on a push to main installs ``.[dev,viz,jaxley,jnwb]`` and runs the broad gate, which runs
``scripts/check_docs_orphans.py``; that script imports ``yaml``, so the dev extra must carry PyYAML.
"""

import pathlib
import re
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _requirement_names(specs):
    return {re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].strip().lower() for spec in specs}


def test_dev_extra_carries_pyyaml_for_check_docs_orphans():
    script = (REPO / "scripts" / "check_docs_orphans.py").read_text(encoding="utf-8")
    assert re.search(r"^import yaml$", script, re.MULTILINE), "precondition: the script imports yaml"
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert "pyyaml" in _requirement_names(extras["dev"])


def test_viz_extra_carries_markdown_for_reportlab_manuscript():
    script = (REPO / "scripts" / "build_manuscript.py").read_text(encoding="utf-8")
    assert re.search(r"^\s+import markdown$", script, re.MULTILINE), "precondition: the build imports markdown"
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert {"reportlab", "markdown"} <= _requirement_names(extras["viz"])
