#!/usr/bin/env python3
"""Generate 0.4.13 stable-surface contract under artifacts/developer/.

This is a development/PRP artifact — not public documentation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import jaxfne as jtfne

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "developer" / "surface_contract_v0413.json"

DOC_GRAMMAR = {
    "README": None,
    "Mathematics": [
        "TFNE",
        "NeuronalTensor",
        "H-state / HDP",
        "Source → Field → Probe",
        "Objective → Optimization",
    ],
    "API": [
        "Circuit",
        "Dynamics",
        "Observation",
        "Optimization",
        "Evidence",
    ],
    "Tutorials": [
        "MCC / minimal system",
        "structured laminar",
        "H-state adaptation",
        "optimization",
    ],
    "Reference": [
        "installation",
        "configuration",
        "citation",
        "changelog",
    ],
}

# Explicit overrides beat heuristics.
SYMBOL_OVERRIDES: dict[str, str] = {
    # REMOVE — root stubs that raise or mislead
    "read_nwb": "REMOVE",
    "write_nwb": "REMOVE",
    "GLIFEmitter": "REMOVE",
    "LIFEmitter": "REMOVE",
    # NAMESPACED — must not read as second grammar
    "build_tutorial_laminar_column": "NAMESPACED",
    "select_neurons": "NAMESPACED",
    "kappa_synchrony": "NAMESPACED",
    "rate_synchrony_targets": "NAMESPACED",
    "spectrolaminar_motif_score": "NAMESPACED",
    "LaminarColumnConfig": "NAMESPACED",
    "CellTypePreset": "NAMESPACED",
    "make_cell_dist": "NAMESPACED",
    "make_cell_type_catalog": "NAMESPACED",
    "cell_catalog_frame": "NAMESPACED",
    "make_laminar_column_config": "NAMESPACED",
    "config_summary_frame": "NAMESPACED",
    "make_izhikevich_control_panel": "NAMESPACED",
    "collect_izhikevich_control": "NAMESPACED",
    "make_stimulus": "NAMESPACED",
    "build_laminar_connections": "NAMESPACED",
    "select_cells": "NAMESPACED",
    "simulate_laminar_trials": "NAMESPACED",
    "spectrolaminar_from_trials": "NAMESPACED",
    "summarize_spectrolaminar_similarity": "NAMESPACED",
    "export_tutorial_artifacts": "NAMESPACED",
    "save_figure": "NAMESPACED",
    "save_figures": "NAMESPACED",
    "export_report": "NAMESPACED",
    "plot_raster": "NAMESPACED",
    "plot_spectrolaminar_suite": "NAMESPACED",
    "plot_stdp_adaptation_suite": "NAMESPACED",
    "vis": "NAMESPACED",
    # EXPERIMENTAL
    "solve_volume_conductor_experimental": "EXPERIMENTAL",
    "get_sharding_context": "EXPERIMENTAL",
    "make_candidate_sharding": "EXPERIMENTAL",
    "make_replicated_sharding": "EXPERIMENTAL",
    # COMPATIBILITY
    "Config": "COMPATIBILITY",
    "Net": "COMPATIBILITY",
    "Signal": "COMPATIBILITY",
    "RuntimeConfiguration": "COMPATIBILITY",
    "AGSDR": "COMPATIBILITY",
    # STABLE core (explicit anchors)
    "Configuration": "STABLE",
    "Model": "STABLE",
    "Signals": "STABLE",
    "RuntimeConfig": "STABLE",
    "NeuronalTensor": "STABLE",
    "construct": "STABLE",
    "simulate": "STABLE",
    "Objective": "STABLE",
    "objective": "STABLE",
    "rate_targets": "STABLE",
    "EdgeParameterSpec": "STABLE",
    "MatrixParameterSpec": "STABLE",
    "edge_parameter": "STABLE",
    "matrix_parameter": "STABLE",
    "TuneResult": "STABLE",
    "Paradigm": "STABLE",
    "manifest": "STABLE",
    "validate_model": "STABLE",
    "DEFAULT_HDP": "STABLE",
}

STABLE_PREFIXES = (
    "validate_",
    "compile_connection",
    "neuronal_tensor",
    "construct_",
    "merge_neuronal",
    "load_neuronal",
    "save_neuronal",
    "rate_",
    "objective",
    "manifest",
    "run_receipt",
    "provenance_receipt",
    "json_safe",
    "config_hash",
    "enable_x64",
    "simulation",
    "runtime_report",
)

STABLE_EXACT = {
    "configuration",
    "connect",
    "construct",
    "simulate",
    "compute_fields",
    "objective",
    "rate_targets",
    "agsdr",
    "random_search",
    "gsdr",
    "gsgd",
    "optax_adam",
    "optax_sgd",
    "paradigm",
    "omission_oddball_paradigm",
    "manifest",
    "validate_model",
    "validate_neuronal_tensor",
    "validate_runtime_config",
    "DEFAULT_HDP",
    "EdgeList",
    "EdgeParameterSpec",
    "MatrixParameterSpec",
    "edge_parameter",
    "matrix_parameter",
    "Objective",
    "TuneResult",
    "Paradigm",
    "ParadigmCondition",
    "ParadigmEvent",
    "Probe",
    "Signals",
    "Simulation",
    "RuntimeConfig",
    "Model",
    "Configuration",
    "NeuronalTensor",
    "ContinuationState",
    "ObjectiveReport",
    "RunReceipt",
    "Manifest",
    "spectrolaminar_psd_jax",
    "bandpower_jax",
    "spectrolaminar_readout_kernel_jax",
    "spectrolaminar_similarity_kernel_jax",
}

DOC_OVERRIDES: dict[str, dict[str, str]] = {
    "docs/v047_refactor_audit.md": {"action": "ARCHIVE", "merge_target": None},
    "docs/fullroadmap.md": {"action": "ARCHIVE", "merge_target": None},
    "docs/for_ai_agents.md": {"action": "ARCHIVE", "merge_target": None},
    "docs/STDP_HOMEOSTATIC_REPORT.md": {"action": "MERGE", "merge_target": "docs/reports/stdp.md"},
    "docs/STDP_CLOSED_LOOP_REPORT.md": {"action": "MERGE", "merge_target": "docs/reports/stdp.md"},
    "docs/STDP_GLOBAL_SCALE_REPORT.md": {"action": "MERGE", "merge_target": "docs/reports/stdp.md"},
    "docs/STDP_LOWRATE_REGIME_REPORT.md": {"action": "MERGE", "merge_target": "docs/reports/stdp.md"},
    "docs/STDP_REAL_TEST_REPORT.md": {"action": "MERGE", "merge_target": "docs/reports/stdp.md"},
    "docs/releases/v0.2.0.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/releases/v0.2.1.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/releases/v0.2.3.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/releases/v0.2.10.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/releases/v0.2.18.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/releases/v0.3.4.md": {"action": "MERGE", "merge_target": "docs/changelog.md"},
    "docs/index.md": {"action": "KEEP", "grammar_bucket": "README"},
    "docs/install.md": {"action": "KEEP", "grammar_bucket": "Reference/installation"},
    "docs/citation.md": {"action": "KEEP", "grammar_bucket": "Reference/citation"},
    "docs/changelog.md": {"action": "KEEP", "grammar_bucket": "Reference/changelog"},
    "docs/source_field_equations.md": {"action": "KEEP", "grammar_bucket": "Mathematics/TFNE"},
    "docs/api/neuronal_tensor.md": {"action": "KEEP", "grammar_bucket": "Mathematics/NeuronalTensor"},
    "docs/guides/hdp.md": {"action": "KEEP", "grammar_bucket": "Mathematics/H-state / HDP"},
    "docs/guides/tensor_field_workflows.md": {"action": "KEEP", "grammar_bucket": "Mathematics/Source → Field → Probe"},
    "docs/api/objectives.md": {"action": "KEEP", "grammar_bucket": "Mathematics/Objective → Optimization"},
    "docs/quickstart.md": {"action": "KEEP", "grammar_bucket": "Tutorials/MCC / minimal system"},
    "docs/scope_and_status.md": {"action": "KEEP", "grammar_bucket": "Reference/configuration"},
    "docs/operator_doctrine.md": {"action": "KEEP", "grammar_bucket": "Mathematics/TFNE"},
}


def classify_symbol(name: str) -> str:
    if name in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[name]
    lower = name.lower()
    if lower.startswith("experimental") or "experimental" in lower:
        return "EXPERIMENTAL"
    if name.startswith("suite2_") or name in {
        "build_laminar_column",
        "build_multi_area_columns",
        "laminar_cortex_config",
        "default_cortical_column_config",
        "default_complete_configuration",
    }:
        return "STABLE"
    if name in STABLE_EXACT or any(name.startswith(p) for p in STABLE_PREFIXES):
        return "STABLE"
    if name.endswith("State") and name in {
        "AGSDRState",
        "GSDRState",
        "GSGDState",
        "SDRState",
        "STDPState",
    }:
        return "STABLE"
    if re.match(r"^(plot_|save_)", name):
        return "NAMESPACED"
    if name in {"SanityDeltaConfig", "SanityDeltaModel", "HierarchicalOddballParadigm"}:
        return "EXPERIMENTAL"
    return "COMPATIBILITY"


def classify_doc(path: str) -> dict[str, str | None]:
    if path in DOC_OVERRIDES:
        return dict(DOC_OVERRIDES[path])
    if path.startswith("docs/_generated/"):
        return {"action": "KEEP", "grammar_bucket": "generated"}
    if path.startswith("docs/tutorials/"):
        return {"action": "KEEP", "grammar_bucket": "Tutorials"}
    if path.startswith("docs/api/"):
        return {"action": "KEEP", "grammar_bucket": "API"}
    if path.startswith("docs/guides/"):
        return {"action": "KEEP", "grammar_bucket": "Guides"}
    if path.startswith("docs/releases/"):
        return {"action": "MERGE", "merge_target": "docs/changelog.md"}
    if path.startswith("docs/notes/"):
        return {"action": "ARCHIVE", "merge_target": None}
    if "REPORT" in path or "CHECKLIST" in path or "CHARACTERIZATION" in path:
        return {"action": "MERGE", "merge_target": "docs/reports/index.md"}
    return {"action": "KEEP", "grammar_bucket": "Appendix"}


def main() -> None:
    names = sorted(jtfne.__all__)
    symbols = {name: classify_symbol(name) for name in names}
    docs = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "docs").rglob("*.md")
    )
    doc_actions = {path: classify_doc(path) for path in docs}

    counts: dict[str, int] = {}
    for value in symbols.values():
        counts[value] = counts.get(value, 0) + 1

    doc_counts: dict[str, int] = {}
    for meta in doc_actions.values():
        action = meta["action"]
        doc_counts[action] = doc_counts.get(action, 0) + 1

    payload = {
        "schema_version": "jaxfne.surface_contract.v0.4.13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess_head(),
        "objective": "minimize public/contextual complexity subject to preserving complete frozen TFNE grammar",
        "phases": {
            "0.4.13": "contraction + publication freeze",
            "0.4.14": "prove + package + release",
        },
        "documentation_grammar": DOC_GRAMMAR,
        "test_gates": {
            "dev": "make test-dev  →  scripts/run_test_gate.py dev",
            "broad": "make test-broad",
            "release": "make test-release",
            "publication": "make test-publication",
        },
        "public_api_snapshot": {
            "baseline_file": "artifacts/public_api_before.json",
            "semantics": "intentional compatibility baseline for test_public_api_matches_snapshot; not current API truth",
            "current_truth": "__all__ in jaxfne/__init__.py until contraction lands",
            "future": "generate public_api_current.json mechanically after 0.4.13 contraction",
        },
        "root_symbols": {
            "source": "__all__",
            "count": len(names),
            "by_class": counts,
            "stable_only_count": counts.get("STABLE", 0),
            "entries": symbols,
        },
        "documentation_pages": {
            "count": len(docs),
            "by_action": doc_counts,
            "entries": doc_actions,
        },
        "readme_constraint": {
            "target_lines": "50-80",
            "must_contain": [
                "mathematical definition",
                "operator composition",
                "NeuronalTensor",
                "H-state dynamics",
                "minimal example",
                "installation",
                "documentation",
                "citation",
            ],
            "must_not_contain": [
                "development history",
                "roadmap",
                "agent discussion",
                "unfinished API inventory",
                "defensive claim language",
                "package-health commentary",
                "feature marketing",
            ],
            "rewrite_when": "after stable root surface is contracted",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("symbol classes:", counts)
    print("doc actions:", doc_counts)


def subprocess_head() -> str | None:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


if __name__ == "__main__":
    main()
