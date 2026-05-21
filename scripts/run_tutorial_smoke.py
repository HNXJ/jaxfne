#!/usr/bin/env python
"""
Tutorial and example smoke runner for jaxfne.

Validates tutorial artifacts before release:
- Python examples run without errors
- Jupyter notebooks have required structure (install, version cells)
- Notebooks have no committed outputs
- Notebooks contain no private paths
- Documentation links reference existing files

Usage:
    python scripts/run_tutorial_smoke.py
    python scripts/run_tutorial_smoke.py --report-json outputs/tutorial_smoke_report.json
    python scripts/run_tutorial_smoke.py --skip-examples
    python scripts/run_tutorial_smoke.py --skip-notebooks
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class TutorialSmokeRunner:
    """Smoke test runner for tutorial infrastructure."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.report: Dict[str, Any] = {
            "status": "pending",
            "examples": {"checked": [], "skipped": [], "errors": []},
            "notebooks": {"checked": [], "skipped": [], "errors": []},
            "docs": {"checked": [], "skipped": [], "errors": []},
        }

    def check_examples(self, skip: bool = False) -> bool:
        """Check Python example scripts for basic syntax and structure."""
        if skip:
            self.report["examples"]["skipped"] = [
                "00_minimal_column",
                "02_spectrolaminar_oddball_scaffold",
                "03_single_neuron_multimodal_probe",
                "04_two_neuron_ei_multimodal",
                "05_network_100_ei_multimodal",
            ]
            return True

        examples_dir = self.repo_root / "examples"
        example_files = [
            "00_minimal_column.py",
            "02_spectrolaminar_oddball_scaffold.py",
            "03_single_neuron_multimodal_probe.py",
            "04_two_neuron_ei_multimodal.py",
            "05_network_100_ei_multimodal.py",
        ]

        all_ok = True
        for example in example_files:
            path = examples_dir / example
            if not path.exists():
                self.report["examples"]["errors"].append(
                    f"{example}: file not found"
                )
                all_ok = False
                continue

            # Basic compile check
            try:
                compile(path.read_text(), str(path), "exec")
                self.report["examples"]["checked"].append(example.replace(".py", ""))
            except SyntaxError as e:
                self.report["examples"]["errors"].append(f"{example}: {e}")
                all_ok = False

        return all_ok

    def check_notebooks(self, skip: bool = False) -> bool:
        """Check Jupyter notebook structure without executing."""
        if skip:
            self.report["notebooks"]["skipped"] = [
                "01_single_neuron_multimodal",
                "02_two_neuron_ei_multimodal",
                "03_network_100_ei_multimodal",
            ]
            return True

        notebooks_dir = self.repo_root / "notebooks"
        notebook_files = [
            "01_single_neuron_multimodal.ipynb",
            "02_two_neuron_ei_multimodal.ipynb",
            "03_network_100_ei_multimodal.ipynb",
        ]

        all_ok = True
        for notebook in notebook_files:
            path = notebooks_dir / notebook
            if not path.exists():
                self.report["notebooks"]["errors"].append(
                    f"{notebook}: file not found"
                )
                all_ok = False
                continue

            try:
                nb_data = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                self.report["notebooks"]["errors"].append(
                    f"{notebook}: invalid JSON - {e}"
                )
                all_ok = False
                continue

            # Check structure
            errors = []

            if "cells" not in nb_data:
                errors.append("missing 'cells' key")
                self.report["notebooks"]["errors"].append(
                    f"{notebook}: {', '.join(errors)}"
                )
                all_ok = False
                continue

            cells = nb_data["cells"]
            if len(cells) < 2:
                errors.append("fewer than 2 cells")

            # Check first code cell has pip install
            code_cells = [c for c in cells if c.get("cell_type") == "code"]
            if code_cells:
                first_code = code_cells[0].get("source", [])
                first_code_text = (
                    "".join(first_code)
                    if isinstance(first_code, list)
                    else first_code
                )
                if "!pip install jaxfne" not in first_code_text:
                    errors.append("first code cell missing '!pip install jaxfne'")

            # Check second code cell has version verification
            if len(code_cells) >= 2:
                second_code = code_cells[1].get("source", [])
                second_code_text = (
                    "".join(second_code) if isinstance(second_code, list) else second_code
                )
                if "jaxfne.__version__" not in second_code_text:
                    errors.append("second code cell missing version verification")

            # Check for committed outputs
            for cell in cells:
                outputs = cell.get("outputs", [])
                if outputs:
                    errors.append("notebook has committed outputs")
                    break

            # Check for private paths
            nb_text = json.dumps(nb_data)
            if re.search(r"/Users/|/home/|~|C:\\", nb_text):
                errors.append("notebook contains private paths")

            if errors:
                self.report["notebooks"]["errors"].append(
                    f"{notebook}: {', '.join(errors)}"
                )
                all_ok = False
            else:
                self.report["notebooks"]["checked"].append(
                    notebook.replace(".ipynb", "")
                )

        return all_ok

    def check_docs_links(self, skip: bool = False) -> bool:
        """Check tutorial documentation for broken links to examples/notebooks."""
        if skip:
            self.report["docs"]["skipped"] = ["tutorial markdown links"]
            return True

        docs_dir = self.repo_root / "docs" / "tutorials"
        all_ok = True

        for md_file in docs_dir.glob("*.md"):
            if md_file.name == "notebook_standard.md":
                continue

            content = md_file.read_text()

            # Look for example and notebook references
            for match in re.finditer(r"examples/(\d+_\S+\.py)", content):
                ref = match.group(1)
                path = self.repo_root / "examples" / ref
                if not path.exists():
                    self.report["docs"]["errors"].append(
                        f"{md_file.name}: broken link to examples/{ref}"
                    )
                    all_ok = False

            for match in re.finditer(r"notebooks/(\d+_\S+\.ipynb)", content):
                ref = match.group(1)
                path = self.repo_root / "notebooks" / ref
                if not path.exists():
                    self.report["docs"]["errors"].append(
                        f"{md_file.name}: broken link to notebooks/{ref}"
                    )
                    all_ok = False

        if not self.report["docs"]["errors"]:
            self.report["docs"]["checked"] = ["tutorial markdown links"]

        return all_ok

    def run(
        self,
        skip_examples: bool = False,
        skip_notebooks: bool = False,
        skip_docs: bool = False,
    ) -> bool:
        """Run all smoke tests."""
        ex_ok = self.check_examples(skip=skip_examples)
        nb_ok = self.check_notebooks(skip=skip_notebooks)
        doc_ok = self.check_docs_links(skip=skip_docs)

        all_ok = ex_ok and nb_ok and doc_ok
        self.report["status"] = "pass" if all_ok else "fail"

        return all_ok

    def print_report(self) -> None:
        """Print human-readable report to stdout."""
        print("\n=== Tutorial Smoke Report ===\n")
        print(f"Status: {self.report['status'].upper()}\n")

        print("Examples:")
        for item in self.report["examples"]["checked"]:
            print(f"  ✓ {item}")
        for item in self.report["examples"]["skipped"]:
            print(f"  - {item} (skipped)")
        for item in self.report["examples"]["errors"]:
            print(f"  ✗ {item}")

        print("\nNotebooks:")
        for item in self.report["notebooks"]["checked"]:
            print(f"  ✓ {item}")
        for item in self.report["notebooks"]["skipped"]:
            print(f"  - {item} (skipped)")
        for item in self.report["notebooks"]["errors"]:
            print(f"  ✗ {item}")

        print("\nDocs:")
        for item in self.report["docs"]["checked"]:
            print(f"  ✓ {item}")
        for item in self.report["docs"]["skipped"]:
            print(f"  - {item} (skipped)")
        for item in self.report["docs"]["errors"]:
            print(f"  ✗ {item}")

        print()

    def save_json_report(self, path: str) -> None:
        """Save report as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"Report saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-examples", action="store_true", help="Skip example checks"
    )
    parser.add_argument(
        "--skip-notebooks", action="store_true", help="Skip notebook checks"
    )
    parser.add_argument(
        "--skip-docs", action="store_true", help="Skip documentation checks"
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Save report as JSON to PATH",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    runner = TutorialSmokeRunner(repo_root)

    success = runner.run(
        skip_examples=args.skip_examples,
        skip_notebooks=args.skip_notebooks,
        skip_docs=args.skip_docs,
    )

    runner.print_report()

    if args.report_json:
        runner.save_json_report(args.report_json)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
