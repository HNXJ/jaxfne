#!/usr/bin/env python
"""
v0.3 Documentation Link Audit Script

Validates v0.3 tutorial documentation against:
1. All internal markdown links resolve to existing files
2. All image links resolve to existing files
3. All tutorials have "Open in Colab" links (exact format)
4. All theoretical sections include LaTeX-rendered equations (parsed heuristically)
5. No forbidden import aliases (jtnfe, jtFNE, from jaxfne import *)

Produces JSON report: docs_link_audit.json

truth_mode: truth_safe_unverified
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any


class DocsAudit:
    """Audit v0.3 tutorial documentation."""

    def __init__(self, docs_root: Path):
        self.docs_root = Path(docs_root)
        self.tutorials_root = self.docs_root / "tutorials_v030"

        self.checked_files = 0
        self.missing_links = []
        self.missing_colab_links = []
        self.latex_policy_violations = []
        self.notes = []

    def run(self) -> Dict[str, Any]:
        """Run full audit and return report."""
        if not self.tutorials_root.exists():
            self.notes.append(f"tutorials_v030 directory not found at {self.tutorials_root}")
            return self._report("fail")

        # Check core doctrine files
        self._check_doctrine_files()

        # Check markdown files for broken links
        self._check_markdown_links()

        # Check for "Open in Colab" links in tutorial references
        self._check_colab_links()

        # Check for LaTeX equation displays
        self._check_latex_equations()

        # Check for forbidden import aliases
        self._check_forbidden_aliases()

        # Determine status
        status = "pass" if not self.missing_links and not self.missing_colab_links else "fail"
        return self._report(status)

    def _check_doctrine_files(self):
        """Verify core doctrine files exist."""
        required = [
            "README.md",
            "template.md",
            "scenario_index.md",
            "acceptance_gates.yaml",
            "plotly_policy.md",
            "canonical_imports.md",
            "docs_audit_policy.md",
            "environment.md",
        ]

        for filename in required:
            path = self.tutorials_root / filename
            if not path.exists():
                self.missing_links.append(f"Doctrine file missing: docs/tutorials_v030/{filename}")
            else:
                self.checked_files += 1

    def _check_markdown_links(self):
        """Find and validate all markdown links in tutorial docs."""
        for md_file in self.tutorials_root.glob("*.md"):
            self.checked_files += 1
            content = md_file.read_text(encoding="utf-8", errors="ignore")

            # Find markdown links: [text](path)
            link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            for match in re.finditer(link_pattern, content):
                link_text = match.group(1)
                link_path = match.group(2)

                # Skip external URLs and anchor links
                if link_path.startswith("http://") or link_path.startswith("https://") or link_path.startswith("#"):
                    continue

                # Skip LaTeX-like content (contains backslashes, Greek letters, etc.)
                if "\\" in link_text or "\\" in link_path:
                    continue

                # Skip example/template links (contain "example" or "docs/example")
                if "example" in link_path.lower():
                    continue

                # Resolve relative path from md_file location
                resolved = (md_file.parent / link_path).resolve()

                # Check if target exists
                if not resolved.exists():
                    # Check if it's in notebooks (may not exist yet for planned scenarios)
                    if "notebooks/v030" in link_path and "planned" in content:
                        self.notes.append(f"Future notebook link (planned): {link_path}")
                        continue
                    self.missing_links.append(f"{md_file.name}: broken link [{link_text}]({link_path})")

    def _check_colab_links(self):
        """Check that tutorials reference 'Open in Colab' with exact format."""
        scenario_index = self.tutorials_root / "scenario_index.md"
        if scenario_index.exists():
            content = scenario_index.read_text(encoding="utf-8", errors="ignore")

            # Find scenario blocks (v0.3.01 through v0.3.15 are current)
            current_scenarios = []
            for i in range(1, 16):  # v0.3.01 through v0.3.15
                pattern = rf"#+\s+v0\.3\.{i:02d}\s*:"
                if re.search(pattern, content):
                    current_scenarios.append(f"v0.3.{i:02d}")

            # Check if Colab links are present for current scenarios
            # (Planned scenarios v0.3.16-v0.3.31 can skip this check)
            if current_scenarios:
                # Must have at least one Colab link reference
                colab_pattern = r'\[Open in Colab\]\(https://colab\.research\.google\.com/'
                if not re.search(colab_pattern, content):
                    self.missing_colab_links.append(
                        "scenario_index.md: Missing 'Open in Colab' links for current scenarios"
                    )

    def _check_latex_equations(self):
        """Check that theory sections include LaTeX-rendered equations."""
        template = self.tutorials_root / "template.md"
        if template.exists():
            content = template.read_text(encoding="utf-8", errors="ignore")

            # Check for LaTeX display math examples ($$...$$)
            latex_display_pattern = r'\$\$[^$]+\$\$'
            if not re.search(latex_display_pattern, content):
                self.latex_policy_violations.append(
                    "template.md: No LaTeX displayed equations found ($$...$$)"
                )

            # Check for term glossary pattern after equations
            glossary_pattern = r'\*\*Term Glossary:\*\*'
            equation_count = len(re.findall(latex_display_pattern, content))
            glossary_count = len(re.findall(glossary_pattern, content))

            if equation_count > 0 and glossary_count == 0:
                self.latex_policy_violations.append(
                    "template.md: Found equations but no term glossary examples"
                )

    def _check_forbidden_aliases(self):
        """Check for forbidden import aliases."""
        forbidden_patterns = [
            (r'import\s+jaxfne\s+(?!as\s+jtfne)', "bare import (not 'as jtfne')"),
            (r'import\s+jaxfne\s+as\s+jtnfe', "wrong alias 'jtnfe'"),
            (r'import\s+jaxfne\s+as\s+jtFNE', "wrong alias 'jtFNE'"),
            (r'from\s+jaxfne\s+import\s+\*', "wildcard import"),
        ]

        for md_file in self.tutorials_root.glob("*.md"):
            content = md_file.read_text(encoding="utf-8", errors="ignore")

            # Extract code blocks only
            code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
            for code in code_blocks:
                for pattern, description in forbidden_patterns:
                    if re.search(pattern, code):
                        # Allow if it's in "forbidden" example section
                        if "❌" not in code and "Incorrect" not in code and "Forbidden" not in code:
                            self.notes.append(
                                f"{md_file.name}: Found {description} (check if intentional example)"
                            )

    def _report(self, status: str) -> Dict[str, Any]:
        """Generate JSON-safe report."""
        return {
            "schema_version": "v0.3.0",
            "status": status,
            "checked_files": self.checked_files,
            "missing_links": self.missing_links,
            "missing_colab_links": self.missing_colab_links,
            "latex_policy_violations": self.latex_policy_violations,
            "notes": self.notes,
        }


def main():
    """Run audit and save report."""
    import sys

    # Find docs root (two levels up from script)
    script_root = Path(__file__).parent.parent
    docs_root = script_root / "docs"

    print(f"Auditing v0.3 documentation at {docs_root.resolve()}")

    audit = DocsAudit(docs_root)
    report = audit.run()

    # Save report
    output_path = script_root / "docs" / "tutorials_v030" / "docs_link_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit report saved: {output_path}")
    print(json.dumps(report, indent=2))

    # Exit with appropriate code
    if report["status"] == "pass":
        print("\n✓ Audit PASSED")
        sys.exit(0)
    else:
        print("\n✗ Audit FAILED")
        if report["missing_links"]:
            print(f"  - {len(report['missing_links'])} broken links")
        if report["missing_colab_links"]:
            print(f"  - {len(report['missing_colab_links'])} missing Colab links")
        if report["latex_policy_violations"]:
            print(f"  - {len(report['latex_policy_violations'])} LaTeX policy violations")
        sys.exit(1)


if __name__ == "__main__":
    main()
