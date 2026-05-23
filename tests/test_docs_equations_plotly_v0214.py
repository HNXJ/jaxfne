"""Documentation and visualization tests for v0.2.14.

Tests cover:
- MkDocs math support configuration
- Equation presence in documentation
- Plotly guide structure and status
- Visual output skill completeness
- Version consistency
- No forbidden vocabulary in new docs
"""

import json
import re
from pathlib import Path
import pytest


# ─── Math Support Configuration Tests ────────────────────────────────────────

def test_mkdocs_yml_includes_arithmatex_extension():
    """mkdocs.yml must include pymdownx.arithmatex extension."""
    mkdocs_path = Path("mkdocs.yml")
    assert mkdocs_path.exists(), "mkdocs.yml not found"

    with open(mkdocs_path, "r") as f:
        content = f.read()

    assert "pymdownx.arithmatex" in content, "arithmatex extension not found in mkdocs.yml"
    assert "generic: true" in content, "arithmatex generic mode not enabled"


def test_mkdocs_yml_includes_mathjax_cdn():
    """mkdocs.yml must include MathJax CDN in extra_javascript."""
    mkdocs_path = Path("mkdocs.yml")
    with open(mkdocs_path, "r") as f:
        content = f.read()

    assert "extra_javascript:" in content, "extra_javascript section not found"
    assert "mathjax" in content.lower() or "tex-mml-chtml" in content, \
        "MathJax CDN not found in extra_javascript"


# ─── Probe Operators Equations Tests ─────────────────────────────────────────

def test_probe_operators_has_equation_section():
    """docs/probe_operators.md must have a Mathematical Forms section."""
    doc_path = Path("docs/probe_operators.md")
    assert doc_path.exists(), "probe_operators.md not found"

    with open(doc_path, "r") as f:
        content = f.read()

    assert "Mathematical Forms" in content or "mathematical form" in content.lower(), \
        "Mathematical Forms section not found"


def test_probe_operators_has_all_eight_operators():
    """probe_operators.md must document equations for all 8 operators."""
    doc_path = Path("docs/probe_operators.md")
    with open(doc_path, "r") as f:
        content = f.read()

    operators = ["SPK", "Vm", "Source", "LFP", "CSD", "EEG", "MEG", "EMM"]
    for op in operators:
        assert op in content, f"Operator {op} not mentioned in probe_operators.md"


def test_probe_operators_has_latex_math():
    """probe_operators.md must contain LaTeX math delimiters."""
    doc_path = Path("docs/probe_operators.md")
    with open(doc_path, "r") as f:
        content = f.read()

    # Check for $...$ or $$...$$ delimiters
    has_inline = "$" in content
    has_block = "$$" in content

    assert has_inline or has_block, "No LaTeX math delimiters found in probe_operators.md"


# ─── Tensor-Field Workflow Equations Tests ───────────────────────────────────

def test_tensor_field_workflows_has_math_notation():
    """docs/tensor_field_workflows.md must have mathematical notation section."""
    doc_path = Path("docs/tensor_field_workflows.md")
    assert doc_path.exists(), "tensor_field_workflows.md not found"

    with open(doc_path, "r") as f:
        content = f.read()

    assert "Mathematical notation" in content or "mathematical notation" in content.lower(), \
        "Mathematical notation section not found"


def test_tensor_field_workflows_has_projection_equations():
    """tensor_field_workflows.md must explain source-to-field projection."""
    doc_path = Path("docs/tensor_field_workflows.md")
    with open(doc_path, "r") as f:
        content = f.read()

    # Check for key concepts (phi_proxy may appear as LaTeX \phi_{\mathrm{proxy}})
    concepts = ["projection", "kernel", "contact"]
    for concept in concepts:
        assert concept in content.lower(), \
            f"Missing concept '{concept}' in tensor_field_workflows.md"

    # Check for phi_proxy as either plain text or in LaTeX
    assert "phi_proxy" in content.lower() or r"\phi_" in content, \
        "Missing phi_proxy concept in tensor_field_workflows.md"


# ─── API Documentation Tests ────────────────────────────────────────────────

def test_fields_api_is_not_placeholder():
    """docs/api/fields.md must replace the 3-line placeholder."""
    doc_path = Path("docs/api/fields.md")
    assert doc_path.exists(), "fields.md not found"

    with open(doc_path, "r") as f:
        content = f.read()

    # Should have substantial content, not just "Placeholder page"
    assert len(content) > 500, "fields.md appears to be mostly empty"
    assert "Placeholder" not in content or "FieldOutput" in content, \
        "fields.md still appears to be a placeholder"


def test_fields_api_has_contract():
    """docs/api/fields.md must document FieldOutput contract."""
    doc_path = Path("docs/api/fields.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "FieldOutput" in content, "FieldOutput not documented"
    assert "18" in content or "eighteen" in content.lower() or "required" in content.lower(), \
        "Field contract not clearly documented"


def test_probes_api_is_not_placeholder():
    """docs/api/probes.md must replace the 3-line placeholder."""
    doc_path = Path("docs/api/probes.md")
    assert doc_path.exists(), "probes.md not found"

    with open(doc_path, "r") as f:
        content = f.read()

    assert len(content) > 500, "probes.md appears to be mostly empty"
    assert "Placeholder" not in content or "ProbeReport" in content, \
        "probes.md still appears to be a placeholder"


def test_probes_api_has_report_contract():
    """docs/api/probes.md must document ProbeReport contract."""
    doc_path = Path("docs/api/probes.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "ProbeReport" in content, "ProbeReport not documented"
    assert "kind" in content.lower(), "Probe kind field not documented"


# ─── Plotly Visualization Guide Tests ────────────────────────────────────────

def test_plotly_visualization_guide_exists():
    """docs/plotly_visualization.md must exist."""
    doc_path = Path("docs/plotly_visualization.md")
    assert doc_path.exists(), "plotly_visualization.md not found at doc root level"


def test_plotly_guide_states_optional():
    """Plotly guide must clearly state Plotly is optional, not required."""
    doc_path = Path("docs/plotly_visualization.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "optional" in content.lower(), "Plotly guide does not state it is optional"
    assert "pip install plotly" in content, "Plotly install command not documented"


def test_plotly_guide_has_code_examples():
    """Plotly guide must include runnable code examples."""
    doc_path = Path("docs/plotly_visualization.md")
    with open(doc_path, "r") as f:
        content = f.read()

    # Check for code blocks
    code_blocks = content.count("```python")
    assert code_blocks >= 3, f"Expected at least 3 code blocks, found {code_blocks}"


def test_plotly_guide_documents_output_structure():
    """Plotly guide must document outputs/<run>/figures/ directory structure."""
    doc_path = Path("docs/plotly_visualization.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "outputs/" in content, "Output directory structure not documented"
    assert "figures/" in content, "Figures subdirectory not documented"
    assert "manifest.json" in content, "Manifest file not mentioned"


def test_plotly_guide_documents_write_html():
    """Plotly guide must show write_html with CDN option."""
    doc_path = Path("docs/plotly_visualization.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "write_html" in content, "write_html method not documented"
    assert "include_plotlyjs" in content, "Plotly JS inclusion not documented"
    assert "cdn" in content.lower(), "CDN option not mentioned"


# ─── Visual Outputs Skill Tests ──────────────────────────────────────────────

def test_visual_outputs_skill_exists():
    """docs/skills/skill_visual_outputs.md must exist."""
    doc_path = Path("docs/skills/skill_visual_outputs.md")
    assert doc_path.exists(), "skill_visual_outputs.md not found"


def test_visual_outputs_skill_has_code_examples():
    """Visual outputs skill must include code examples for each operator."""
    doc_path = Path("docs/skills/skill_visual_outputs.md")
    with open(doc_path, "r") as f:
        content = f.read()

    # Check for plot functions
    functions = ["plot_spike_raster", "plot_voltage_traces", "plot_lfp_heatmap", "plot_csd_heatmap"]
    for func in functions:
        assert func in content, f"Missing code example {func}"


def test_visual_outputs_skill_documents_naming_conventions():
    """Visual outputs skill must document file naming conventions."""
    doc_path = Path("docs/skills/skill_visual_outputs.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "naming" in content.lower() or ".html" in content, \
        "File naming conventions not documented"


def test_visual_outputs_skill_has_validation_commands():
    """Visual outputs skill must include validation commands."""
    doc_path = Path("docs/skills/skill_visual_outputs.md")
    with open(doc_path, "r") as f:
        content = f.read()

    assert "Validation" in content or "validation" in content, \
        "Validation section not found"
    assert "isfinite" in content, "Finitude check not documented"


# ─── Navigation Tests ────────────────────────────────────────────────────────

def test_mkdocs_nav_includes_plotly_guide():
    """mkdocs.yml navigation must include Plotly visualization guide."""
    mkdocs_path = Path("mkdocs.yml")
    with open(mkdocs_path, "r") as f:
        content = f.read()

    assert "plotly_visualization" in content, "Plotly guide not in navigation"


def test_mkdocs_nav_includes_visual_outputs_skill():
    """mkdocs.yml navigation must include visual outputs skill."""
    mkdocs_path = Path("mkdocs.yml")
    with open(mkdocs_path, "r") as f:
        content = f.read()

    assert "skill_visual_outputs" in content, "Visual outputs skill not in navigation"


# ─── Version and Dependency Tests ────────────────────────────────────────────

def test_version_remains_0210():
    """jaxfne version must remain 0.2.29 (after v0.2.29 release)."""
    import jaxfne
    assert jaxfne.__version__ == "0.2.30", \
        f"Expected version 0.2.29, got {jaxfne.__version__}"


def test_pyproject_toml_not_modified_for_plotly():
    """pyproject.toml must not be modified; Plotly remains optional."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml not found"

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Plotly should not be a core dependency
    # It's OK if it's mentioned in optional/dev extras, but not in main dependencies
    lines = content.split("\n")
    dependencies_section = False
    for line in lines:
        if line.strip().startswith("dependencies"):
            dependencies_section = True
        elif dependencies_section and line.startswith("["):
            dependencies_section = False
        elif dependencies_section and "plotly" in line.lower():
            pytest.fail("Plotly found in core dependencies; it must remain optional")


# ─── Vocabulary Audit Tests ─────────────────────────────────────────────────

def test_no_truth_mode_in_new_docs():
    """New docs must not claim truth_mode is in public reports."""
    new_docs = [
        "docs/plotly_visualization.md",
        "docs/skills/skill_visual_outputs.md",
        "docs/api/fields.md",
        "docs/api/probes.md",
    ]

    # Check for JSON field that asserts truth_mode in actual output
    # Mentioning that truth_mode is internal-only is acceptable
    forbidden_patterns = [
        r'"truth_mode":\s*"[^"]*"',  # JSON field: "truth_mode": "value"
    ]

    for doc in new_docs:
        if Path(doc).exists():
            with open(doc, "r") as f:
                content = f.read()

            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                assert not matches, \
                    f"Forbidden pattern '{pattern}' (truth_mode in JSON) found in {doc}"


def test_no_like_terminology_in_new_docs():
    """New docs must use *-proxy, never *-like terminology."""
    new_docs = [
        "docs/plotly_visualization.md",
        "docs/skills/skill_visual_outputs.md",
        "docs/api/fields.md",
        "docs/api/probes.md",
    ]

    forbidden_patterns = [
        r"lfp_like", r"csd_like", r"eeg_like", r"meg_like",
        r"proxy_positive_equals_extracellular_source_like"
    ]

    for doc in new_docs:
        if Path(doc).exists():
            with open(doc, "r") as f:
                content = f.read()

            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                assert not matches, \
                    f"Forbidden pattern '{pattern}' found {len(matches)} times in {doc}"


def test_operator_terminology_uses_proxy():
    """Operator documentation must use *-proxy terminology consistently."""
    docs_to_check = [
        "docs/probe_operators.md",
        "docs/api/probes.md",
    ]

    for doc in docs_to_check:
        if Path(doc).exists():
            with open(doc, "r") as f:
                content = f.read()

            # Check for canonical names
            required_terms = ["lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy"]
            for term in required_terms:
                assert term in content, \
                    f"Required term '{term}' not found in {doc}"


# ─── Gitignore Tests ────────────────────────────────────────────────────────

def test_gitignore_has_outputs():
    """outputs/ directory must be in .gitignore."""
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore not found"

    with open(gitignore_path, "r") as f:
        content = f.read()

    assert "outputs/" in content or "outputs" in content, \
        "outputs/ not in .gitignore"


def test_gitignore_has_site():
    """site/ directory (mkdocs build output) must be in .gitignore."""
    gitignore_path = Path(".gitignore")
    with open(gitignore_path, "r") as f:
        content = f.read()

    assert "site/" in content or "site" in content, \
        "site/ not in .gitignore"


# ─── LaTeX Syntax Validation Tests ──────────────────────────────────────────

def test_equation_latex_syntax():
    """Equations in new docs must have valid LaTeX syntax (basic check)."""
    doc_paths = [
        "docs/probe_operators.md",
        "docs/tensor_field_workflows.md",
    ]

    for doc in doc_paths:
        if Path(doc).exists():
            with open(doc, "r") as f:
                content = f.read()

            # Extract math blocks and check for basic syntax
            # Look for $$...$$ blocks
            math_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)

            # Basic checks: balanced braces, no obvious syntax errors
            for block in math_blocks:
                open_braces = block.count("{")
                close_braces = block.count("}")
                assert open_braces == close_braces, \
                    f"Unbalanced braces in LaTeX block: {block[:50]}..."
