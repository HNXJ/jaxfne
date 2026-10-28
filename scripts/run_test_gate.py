#!/usr/bin/env python3
"""Executable test gates for jaxfne.

Gate vocabulary (single source of truth for local runs and CI references):

  dev          Curated architectural gate (~1 minute)
  report       All audit checks without the pytest sweeps (~1 minute)
  docs         The documentation subset of the audit checks (~30 seconds)
  broad        Repository-wide ``-m "not slow"`` (~8 minutes). The CLI reuses a passing
               run's stamp when the code tree is unchanged (report audits only);
               --no-stamp forces the full run
  release      Broad + slow markers + release examples/docs checks
  rc           Full release-candidate verification (PRE_RELEASE_GATE >= RELEASE_CI_GATE).
               Pytest sweeps that Release CI ran on this tree are reused from its JUnit when
               that run passed; every other family runs here
  publication  Frozen scientific experiments and artifact validation
  examples     The RELEASE_EXAMPLES smoke list only (owner of the list)
  paper-lanes  The paper-lane pytest sweeps only, the broad gate's lane list
               (--lane NAME, repeatable, selects some lanes; CI job paper-lanes)
  wheel-smoke  The canonical isolated-wheel smoke in a given venv
               (owner of the smoke body; needs --venv-python)

Documentation must refer to these gate names and this script (or Makefile
targets), not duplicate file lists in prose.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYTEST_ENV = {
    **os.environ,
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONPATH": str(ROOT),
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}

# Curated dev gate — owned here, not in docs/reference/project/ci_policy.md prose.
DEV_PYTEST_TARGETS = [
    "tests/test_api_smoke.py",
    "tests/test_root_import_lightweight.py",
    "tests/test_signals_get_v0329.py",
    "tests/test_jnwb_view.py",
    "tests/test_neuronal_tensor_connectivity.py",
    "tests/test_neuronal_tensor.py",
    "tests/test_connection_rule_compile_v0330.py",
    "tests/test_continuation_contract.py",
    "tests/test_mcc.py",
    # TFNE/2 adoption. The specification layer alone is not enough: TFNE-PARAM-01
    # was a total substitution of the declared weight that every edge-count test
    # passed through, so the curated gate holds realization/execution
    # equivalence over topology AND executable parameters, not counts or prose.
    "tests/test_tfne_algebra.py",
    "tests/test_tfne_execution.py",
    "tests/test_tfne_parameter_transfer.py",
    # An atlas regeneration resets figure_state, which silently dropped the
    # canonical atlas from docs/start/gallery.md twice; dev CI never collected this.
    "tests/test_generate_gallery.py",
    # Docs atlas registry and .html link resolution; 34 of 47 docs iframes resolved
    # nowhere under directory URLs before this gate.
    "tests/test_build_docs_assets.py",
    # Docs currency: no published current page states an older minor version as current.
    "tests/test_docs_version_currency.py",
    # Prose hygiene ("X, not Y", em dashes); a docs edit with three hits passed CI Fast
    # and the docs gate on 2026-10-27.
    "tests/test_public_docs_hygiene.py",
    # Canary chains: one seeded run per core data path, frozen in tests/canary_register.json.
    "tests/test_canary_chains.py",
]

# 0.5.1-4b(4): parallelize the independent pytest sweeps with xdist (already
# a dev dependency). PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 in PYTEST_ENV means the
# plugin must be loaded explicitly with -p. The notebook sweep stays serial:
# its scoped paths exist for Windows kernel/zmq stability under load.
PYTEST_XDIST_ARGS = ["-p", "xdist", "-n", "auto"]

# Marker selectors for the pytest sweeps.
#
# Together these MUST be exhaustive over the (slow, notebook) marker algebra.
# Release CI on main runs `pytest tests` with no marker filter, so any
# combination not covered here is a test that release CI executes and the RC
# gate does not -- which silently breaks PRE_RELEASE_GATE >= RELEASE_CI_GATE at
# the test-population level even while every check-family name lines up.
# That is exactly how the `notebook` set (30 node ids) went uncovered.
# tests/test_release_gate_hierarchy.py proves exhaustiveness mechanically.
BROAD_MARKER_EXPR = "not slow"
SLOW_MARKER_EXPR = "slow and not notebook"
NOTEBOOK_MARKER_EXPR = "notebook"
RC_MARKER_EXPRS = (BROAD_MARKER_EXPR, SLOW_MARKER_EXPR, NOTEBOOK_MARKER_EXPR)

# Test modules excluded from the broad/release/rc pytest sweeps.
#
# Empty since 0.4.20. The two multi-area modules were excluded in d4e3f72
# because they referenced functions that did not exist yet
# (synaptic_resonance_source, spectrolaminar_similarity). Those symbols are now
# part of jaxfne.fields and both modules pass, so the exclusion was stale and
# has been retired. Adding an entry here removes a test module from every
# blocking gate, so it requires a stated, checkable reason.
BROAD_PYTEST_IGNORE: list[str] = []

# Notebook execution tests only; scoped paths avoid collecting the full 3800+
# test tree before the notebook sweep (Windows kernel/zmq stability under load).
NOTEBOOK_PYTEST_TARGETS = [
    "tests/test_neuronal_tensor_notebook_execution.py",
    "tests/test_notebook_execution_suite.py",
    "tests/test_suite_no1_notebook_execution.py",
    "tests/test_suite_no4_notebook_execution.py",
]

RELEASE_EXAMPLES = [
    "examples/00_minimal_column.py",
    "examples/01_source_field_manifest.py",
    "examples/01_generalized_readout_smoke.py",
    "examples/02_generalized_vis_smoke.py",
    "examples/02_omission_scaffold.py",
    "examples/03_objective_and_tune_smoke.py",
    "examples/04_blackbox_tuning_loop.py",
    "examples/05_dataset_bridge_manifest.py",
    "examples/06_edge_list_recurrent_backend.py",
    "examples/07_jaxley_trace_bridge.py",
    "examples/00_generalized_izhikevich_3d_smoke.py",
    "examples/02_spectrolaminar_oddball_scaffold.py",
    "examples/03_jaxley_bridge_smoke.py",
    "examples/05_network_100_ei_multimodal.py",
    "examples/v031_single_izhikevich_neuron.py",
    "examples/v032_single_neuron_parameter_sweep.py",
    "examples/v033_two_neuron_ei_multimodal.py",
    "examples/08_neuronal_tensor_first.py",
]

CHECK_FAMILIES = {
    "environment_parity",
    "junit_parity",
    "compileall",
    "pytest_dev",
    "pytest_docs",
    "lint_ruff",
    "docs_language_audit",
    "notebook_grammar_audit",
    "vocabulary_audit",
    "docs_orphans_check",
    "doc_code_integrity_audit",
    "docs_figures_check",
    "record_integrity",
    "replication_audit",
    "etude_no_resim",
    "harness_integrity",
    "agent_refs",
    "atlas_inheritance",
    "claim_ledger",
    "memory_hygiene",
    "report_hygiene",
    "release_artifacts",
    "release_freeze",
    "notebooks_assets",
    "release_metadata",
    "tree_size_audit",
    "docs_build_strict",
    "pytest_broad",
    "pytest_lanes",
    "pytest_slow",
    "pytest_notebook",
    "examples_smoke",
    "package_build",
    "twine_check",
    "isolated_wheel_smoke",
    "publication_equivalence",
    # The broad stamp's reuse path: the tests that name a stack file, run after a stamp match.
    "pytest_stack_readers",
}

RELEASE_CI_GATE_FAMILIES = {
    "compileall",
    "lint_ruff",
    "docs_language_audit",
    "notebook_grammar_audit",
    "docs_orphans_check",
    "doc_code_integrity_audit",
    "tree_size_audit",
    "docs_build_strict",
    "pytest_broad",
    "pytest_slow",
    # Release CI on main runs `pytest tests` unfiltered, so it executes the
    # notebook set too. The RC gate must therefore execute it as well.
    "pytest_notebook",
    "examples_smoke",
    "package_build",
    "twine_check",
    "isolated_wheel_smoke",
}

GATE_CHECK_FAMILIES = {
    "dev": {
        "compileall",
        "pytest_dev",
        "docs_language_audit",
        "vocabulary_audit",
        # CI on dev runs `scripts/run_test_gate.py dev` only (ci.yml), so the
        # audit families below must run in the dev gate too, or nothing
        # executes them before the broad gate on main.
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "agent_refs",
    },
    "broad": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "docs_figures_check",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "agent_refs",
        "atlas_inheritance",
        "claim_ledger",
        "memory_hygiene",
        "report_hygiene",
        "release_artifacts",
        "release_freeze",
        "notebooks_assets",
        "release_metadata",
        "tree_size_audit",
        "pytest_broad",
        "pytest_lanes",
    },
    "report": {
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "docs_figures_check",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "agent_refs",
        "atlas_inheritance",
        "claim_ledger",
        "memory_hygiene",
        "report_hygiene",
        "release_artifacts",
        "release_freeze",
        "notebooks_assets",
        "release_metadata",
        "tree_size_audit",
    },
    "docs": {
        "pytest_docs",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
    },
    "release": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "docs_figures_check",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "agent_refs",
        "atlas_inheritance",
        "claim_ledger",
        "memory_hygiene",
        "report_hygiene",
        "release_artifacts",
        "release_freeze",
        "notebooks_assets",
        "release_metadata",
        "tree_size_audit",
        "docs_build_strict",
        "pytest_broad",
        "pytest_lanes",
        "pytest_slow",
        "pytest_notebook",
        "examples_smoke",
    },
    "rc": {
        "environment_parity",
        "junit_parity",
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "docs_figures_check",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "agent_refs",
        "atlas_inheritance",
        "claim_ledger",
        "memory_hygiene",
        "report_hygiene",
        "release_artifacts",
        "release_freeze",
        "notebooks_assets",
        "release_metadata",
        "tree_size_audit",
        "docs_build_strict",
        "pytest_broad",
        "pytest_lanes",
        "pytest_slow",
        "pytest_notebook",
        "examples_smoke",
        "package_build",
        "twine_check",
        "isolated_wheel_smoke",
    },
}


# --- Observed execution ledger -------------------------------------------------
#
# The RC attestation records what was *observed to execute*, never what was
# declared. Every check family accumulates one record carrying the fields the
# authorization contract requires: family, command, started, completed,
# exit_code, status, evidence. A family that is declared but never executed
# simply has no record, and therefore cannot appear in observed_pass_families.

ATTESTATION_SCHEMA: str = "jaxfne.rc_gate_attestation.v2"

# Untracked by design: an attestation that lives in the tree changes the SHA it
# attests to. See .gitignore (artifacts/attestations/).
DEFAULT_ATTESTATION_PATH = ROOT / "artifacts" / "attestations" / "rc_gate_attestation.json"

_OBSERVATIONS: dict[str, dict] = {}


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reset_observations() -> None:
    """Clear the observed-execution ledger (used by tests)."""
    _OBSERVATIONS.clear()


def record_execution(
    family: str,
    argv: list[str],
    started: str,
    completed: str,
    exit_code: int,
) -> None:
    """Append one observed command result to ``family``'s record."""
    rec = _OBSERVATIONS.get(family)
    if rec is None:
        rec = {
            "family": family,
            "command": "",
            "commands": [],
            "started": started,
            "completed": completed,
            "exit_code": 0,
            "status": "PASS",
            "evidence": [],
        }
        _OBSERVATIONS[family] = rec
    joined = " ".join(argv)
    rec["commands"].append({"argv": list(argv), "exit_code": int(exit_code)})
    rec["completed"] = completed
    rec["command"] = (
        joined
        if len(rec["commands"]) == 1
        else f"{len(rec['commands'])} commands; first: {' '.join(rec['commands'][0]['argv'])}"
    )
    rec["evidence"].append(f"{started} -> {completed} exit={int(exit_code)}: {joined}")
    if int(exit_code) != 0:
        rec["exit_code"] = int(exit_code)
        rec["status"] = "FAIL"


def record_reuse(family: str, reuse: dict) -> None:
    """Record ``family`` as executed on Release CI for this tree, not by a local run.

    The record names the CI run and tree it rests on. The reconciler reads it like any
    successful execution; ``reused_from`` shows that no local run produced it.
    """
    run_id, tree = reuse["run_id"], reuse["tree_sha"]
    workflow = reuse["workflow"]
    _OBSERVATIONS[family] = {
        "family": family,
        "command": f"reused from {workflow} run {run_id} (tree {tree})",
        "commands": [
            {"argv": ["reused", workflow, f"run={run_id}", f"tree={tree}"], "exit_code": 0}
        ],
        "started": reuse["created"],
        "completed": reuse["updated"],
        "exit_code": 0,
        "status": "PASS",
        "evidence": [
            f"reused: {workflow} run {run_id} concluded success on tree {tree}; "
            f"JUnit {', '.join(reuse['junit'])}"
        ],
        "reused_from": {
            "workflow": workflow,
            "run_id": run_id,
            "head_sha": reuse["head_sha"],
            "tree_sha": tree,
        },
    }


def observed_pass_families() -> set[str]:
    """Families with at least one executed command and no non-zero exit."""
    return {
        rec["family"]
        for rec in _OBSERVATIONS.values()
        if rec["status"] == "PASS" and rec["exit_code"] == 0 and rec["commands"]
    }


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    family: str | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    started = _utcnow()
    proc = subprocess.run(cmd, cwd=cwd or ROOT, check=False, env=env or PYTEST_ENV)
    completed = _utcnow()
    if family is not None:
        if family not in CHECK_FAMILIES:
            raise RuntimeError(f"Unknown check family {family!r}; add it to CHECK_FAMILIES")
        record_execution(family, cmd, started, completed, proc.returncode)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# --- Shared CI-owned checks ----------------------------------------------------
#
# RELEASE_EXAMPLES and the isolated-wheel smoke body below are owned here.
# .github/workflows/ci.yml, .github/workflows/release_ci.yml and
# scripts/release_rehearsal.sh must call the ``examples`` / ``wheel-smoke``
# gates instead of carrying their own copies.


def wheel_smoke_code(expected_version: str) -> str:
    """Canonical isolated-wheel smoke program.

    Union of the bodies formerly in ci.yml, release_ci.yml,
    release_rehearsal.sh and gate_rc: every assertion each of them ran.
    The emitter/field spellings are the values construct realizes for the
    bare calls (``cortical_eig`` is the only accepted preset; the field keys
    accept only these values), so they pin the declaration, not new physics.
    """
    return (
        "import json, jaxfne as jtfne\n"
        "assert 'site-packages' in jtfne.__file__,"
        " f'Not in site-packages: {jtfne.__file__}'\n"
        f"assert jtfne.__version__ == {expected_version!r},"
        f" f'version={{jtfne.__version__}}, expected {expected_version}'\n"
        "cfg = (jtfne.configuration()\n"
        "    .network(n=8).emitter(family='izhikevich', preset='cortical_eig')\n"
        "    .field(domain='laminar_column', conductivity='proxy',\n"
        "           boundary='mean_zero_neumann', gauge='mean_zero')\n"
        "    .probe(name='lp', n_contacts=8))\n"
        "model = jtfne.construct(cfg)\n"
        "signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0))\n"
        "readouts = model.compute_readout(signals, [jtfne.readout_spec('r', 'spike_rate_hz')])\n"
        "manifest = model.manifest(signals, readouts)\n"
        "json.dumps(manifest, allow_nan=False)\n"
        "assert manifest['physical_amplitude_calibrated'] is False\n"
        "print('wheel smoke OK:', jtfne.__version__)\n"
    )


def _local_package_version() -> str:
    """Expected wheel version, read from the tree (pyproject, not the env)."""
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def run_release_examples() -> None:
    """Run every RELEASE_EXAMPLES entry (family ``examples_smoke``)."""
    try:
        for example in RELEASE_EXAMPLES:
            _run([sys.executable, example], family="examples_smoke")
            print(f"PASS: {example}", flush=True)
    finally:
        # Restore static docs figures that examples mirror to keep tree clean, also after a failure
        subprocess.run(
            ["git", "checkout", "--", "docs/archive/tutorials_v030/_static/figures"], cwd=ROOT, check=False
        )


def run_isolated_wheel_smoke(
    venv_python: str | Path, expected_version: str, *, cwd: str | Path
) -> None:
    """Run the canonical wheel smoke in ``venv_python`` (family ``isolated_wheel_smoke``).

    ``cwd`` must hold no importable ``jaxfne`` package: ``python -c`` puts it
    at the front of ``sys.path``, which would shadow the installed wheel the
    ``site-packages`` assertion guards against.
    """
    isolated_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    isolated_env["PYTHONIOENCODING"] = "utf-8"
    isolated_env["PYTHONUTF8"] = "1"
    print("+ Running isolated wheel smoke simulation...", flush=True)
    _run(
        [str(venv_python), "-c", wheel_smoke_code(expected_version)],
        cwd=Path(cwd),
        env=isolated_env,
        family="isolated_wheel_smoke",
    )


def gate_examples() -> None:
    """The RELEASE_EXAMPLES smoke list only (CI calls this; it owns the list)."""
    run_release_examples()


def gate_wheel_smoke(
    venv_python: str | None = None, expected_version: str | None = None
) -> None:
    """Run the canonical wheel smoke in an existing venv (CI calls this)."""
    if not venv_python:
        raise RuntimeError("wheel-smoke needs --venv-python <path to the venv python>")
    version = expected_version or _local_package_version()
    with tempfile.TemporaryDirectory(prefix="jaxfne_wheel_smoke_cwd_") as tmp_cwd:
        run_isolated_wheel_smoke(venv_python, version, cwd=tmp_cwd)


def gate_dev() -> None:
    _run(
        [sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts"],
        family="compileall",
    )
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            *DEV_PYTEST_TARGETS,
            "-q",
            "-m",
            "not slow and not release",
            "--tb=short",
            *PYTEST_XDIST_ARGS,
        ],
        family="pytest_dev",
    )
    _run(
        [sys.executable, "scripts/audit_public_docs_language.py", "--check"],
        family="docs_language_audit",
    )
    _run([sys.executable, "scripts/audit_vocabulary.py", "--check"], family="vocabulary_audit")
    # CI on dev runs this gate only (ci.yml), so the audit families in the dev
    # set run here, from AUDIT_STEPS (the commands' one owner).
    for family, cmd in AUDIT_STEPS:
        if family in DEV_AUDIT_FAMILIES:
            _run(cmd, family=family)


# Audit families the dev gate runs from AUDIT_STEPS (the docs-language and vocabulary
# audits run above with the dev gate's own ordering).
DEV_AUDIT_FAMILIES = (
    "record_integrity",
    "replication_audit",
    "etude_no_resim",
    "harness_integrity",
    "agent_refs",
)


# The audit checks the broad gate runs before its pytest sweep, owned here and
# shared with the report and docs gates. Order matters only for readability.
AUDIT_STEPS: list[tuple[str, list[str]]] = [
    ("lint_ruff", [sys.executable, "-m", "ruff", "check", "jaxfne/"]),
    (
        "lint_ruff",
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "scripts/",
            "tests/",
            "examples/",
            "--select",
            "E711,E712,E721,E722,F601,F632,F811,F821,F823,F402,F401,F841,E741,E402",
        ],
    ),
    (
        "docs_language_audit",
        [sys.executable, "scripts/audit_public_docs_language.py", "--check"],
    ),
    (
        "notebook_grammar_audit",
        [sys.executable, "scripts/audit_notebook_grammar.py", "--check"],
    ),
    ("vocabulary_audit", [sys.executable, "scripts/audit_vocabulary.py", "--check"]),
    ("docs_orphans_check", [sys.executable, "scripts/check_docs_orphans.py"]),
    (
        "doc_code_integrity_audit",
        [sys.executable, "scripts/audit_doc_code_integrity.py", "--check"],
    ),
    (
        "docs_figures_check",
        [sys.executable, "scripts/build_docs_assets.py", "--docs-figures-check"],
    ),
    ("tree_size_audit", [sys.executable, "scripts/check_tree_size.py", "--check"]),
    ("record_integrity", [sys.executable, "scripts/check_record_integrity.py", "--check"]),
    (
        "replication_audit",
        [sys.executable, "scripts/check_replications.py", "--check"],
    ),
    (
        "etude_no_resim",
        [sys.executable, "scripts/gate_etude_no_resim.py", "--check"],
    ),
    (
        "harness_integrity",
        [sys.executable, "scripts/harness/check_harness_integrity.py"],
    ),
    ("agent_refs", [sys.executable, "scripts/harness/check_agent_refs.py"]),
    ("atlas_inheritance", [sys.executable, "scripts/check_atlas_inheritance.py"]),
    ("claim_ledger", [sys.executable, "scripts/check_claim_ledger_055.py"]),
    ("memory_hygiene", [sys.executable, "scripts/harness/check_memory_hygiene.py"]),
    ("report_hygiene", [sys.executable, "scripts/report_hygiene_check.py"]),
    (
        "release_artifacts",
        [
            sys.executable,
            "scripts/release/validate_release_artifacts.py",
            "--version",
            _local_package_version(),
            "--allow-missing-dist",
        ],
    ),
    ("release_freeze", [sys.executable, "scripts/release/assert_release_freeze.py"]),
    (
        "notebooks_assets",
        [sys.executable, "scripts/audit_notebooks_and_assets.py", "--check"],
    ),
    (
        "release_metadata",
        [sys.executable, "scripts/sync_release_metadata.py"],
    ),
]

# The documentation subset: what a docs edit needs checked before pushing.
DOCS_AUDIT_FAMILIES = {
    "docs_language_audit",
    "notebook_grammar_audit",
    "vocabulary_audit",
    "docs_orphans_check",
    "doc_code_integrity_audit",
    "docs_figures_check",
}


def gate_report() -> None:
    """The audit checks only: a gate status report without the pytest sweeps."""
    for family, cmd in AUDIT_STEPS:
        _run(cmd, family=family)


def gate_docs() -> None:
    """Quick docs gate: the documentation subset of the audit checks, then the prose hygiene tests."""
    for family, cmd in AUDIT_STEPS:
        if family in DOCS_AUDIT_FAMILIES:
            _run(cmd, family=family)
    _run(
        [sys.executable, "-m", "pytest", "tests/test_public_docs_hygiene.py", "-q", "--tb=short"],
        family="pytest_docs",
    )


# --- Paper-lane pytest sweeps (family ``pytest_lanes``) -------------------------
#
# ``testpaths`` is ``tests`` only, so the lane tests under PAPER_LANES_DIR ran in
# no gate. Each lane runs in its own pytest process: caminos enables jax x64 at
# import, and a pcl guard aborts a shared collection.
PAPER_LANES_DIR = "artifacts/etudes/papers"

# Lanes that fail, keyed by lane folder, with the summary line that showed it. A
# listed lane that still fails is printed as skipped; a listed lane that passes
# fails the family (stale entry), so an entry cannot outlive its fault.
LANE_PYTEST_KNOWN_FAILING: dict[str, str] = {}


def discover_pytest_lanes() -> list[str]:
    """Lane folders: directories under PAPER_LANES_DIR holding a tracked ``test_*.py`` at any depth."""
    lanes: set[str] = set()
    prefix = PAPER_LANES_DIR + "/"
    for path in _git_out("ls-files", "--", PAPER_LANES_DIR).splitlines():
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix) :].split("/")
        if len(parts) >= 2 and parts[-1].startswith("test_") and parts[-1].endswith(".py"):
            lanes.add(parts[0])
    return sorted(lanes)


# Lanes whose tests run under xdist (profiled 2026-10-27: pcl 542 s serial, 121 s with
# xdist; caminos 247 s, 65 s; both pass the same tests). Short lanes stay serial, where
# worker start-up would cost more than it saves.
XDIST_LANES = {"pcl", "caminos"}


def _lane_argv(lane: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *(PYTEST_XDIST_ARGS if lane in XDIST_LANES else []),
        f"{PAPER_LANES_DIR}/{lane}",
    ]


def _run_lane(lane: str) -> int:
    """Run one lane in its own pytest process; return its exit code."""
    cmd = _lane_argv(lane)
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False, env=PYTEST_ENV).returncode


def gate_pytest_lanes(lanes: list[str] | None = None) -> None:
    """One pytest process per paper lane (family ``pytest_lanes``).

    ``lanes`` selects some lanes by folder name; the default is every discovered lane,
    which is what the broad gate runs. An unknown name raises before any lane runs.

    Exit code 5 (no tests collected) fails: a lane is discovered only when it holds a
    tracked ``test_*.py``, so collecting nothing hides its tests. A known-failing lane
    that still fails is recorded in the ledger as exit 0 and named in its printed SKIP
    line; one that passes or collects nothing is a stale entry and fails the family.
    """
    known = discover_pytest_lanes()
    selected = known if lanes is None else list(lanes)
    unknown = sorted(set(selected) - set(known))
    if unknown:
        raise ValueError(f"unknown paper lane(s) {unknown}; known lanes: {known}")
    failures: list[str] = []
    for lane in selected:
        argv = _lane_argv(lane)
        started = _utcnow()
        rc = _run_lane(lane)
        completed = _utcnow()
        known = LANE_PYTEST_KNOWN_FAILING.get(lane)
        if known is not None and rc in (0, 5):
            failures.append(f"{lane}: listed in LANE_PYTEST_KNOWN_FAILING but passes (stale entry)")
            print(f"FAIL lane {lane}: stale LANE_PYTEST_KNOWN_FAILING entry", flush=True)
            record_execution("pytest_lanes", argv, started, completed, 1)
        elif known is not None:
            print(f"SKIP known-failing lane {lane} (rc={rc}): {known}", flush=True)
            record_execution("pytest_lanes", argv, started, completed, 0)
        elif rc == 0:
            print(f"PASS lane {lane} (rc=0)", flush=True)
            record_execution("pytest_lanes", argv, started, completed, 0)
        else:
            failures.append(f"{lane}: rc={rc}" + (" (no tests collected)" if rc == 5 else ""))
            print(f"FAIL lane {lane} (rc={rc})", flush=True)
            record_execution("pytest_lanes", argv, started, completed, rc)
    if failures:
        raise RuntimeError("pytest_lanes failed: " + "; ".join(failures))


# --- Code-tree stamp (broad CLI only) ------------------------------------------
#
# A passing ``broad`` run writes a stamp keyed by the tree it tested: the index entries of every
# tracked path except the six stack files, the interpreter, and no modified or untracked file
# outside those six. The next ``broad`` whose key matches runs the report audits and the tests
# that name a stack file (their results depend on the stacks, which are excluded from the key).
# Any doubt runs everything: no stamp, an unreadable or mismatched stamp, a dirty tree, no git
# answer, or no stack-reading test. Direct calls (release, rc, tests) never read or write the stamp.

BROAD_STAMP_SCHEMA = "jaxfne.broad_gate_stamp.v1"
# The owner's state files. Edits to them do not change the key; the tests that read them rerun.
STAMP_STATE_FILES = (
    "artifacts/todo_stack.md",
    "artifacts/open_stack.md",
    "artifacts/state.md",
    "artifacts/memory.md",
    "artifacts/goal_stack.md",
    "artifacts/fact_stack.md",
)


def _stamp_pathspec() -> list[str]:
    """Every tracked path, minus the exact state files (exclude magic, same in key and dirty check)."""
    return [".", *(f":(exclude){path}" for path in STAMP_STATE_FILES)]


def _stack_reader_tests() -> list[str]:
    """Tracked .py files under tests/ that name a stack file, found by git grep at run time."""
    args = [arg for path in STAMP_STATE_FILES for arg in ("-e", Path(path).name)]
    try:
        out = _git_out("grep", "-l", "-F", *args, "--", "tests")
    except (OSError, subprocess.CalledProcessError):  # git grep exits 1 when nothing matches
        return []
    return sorted(line for line in out.splitlines() if line.endswith(".py"))


def broad_stamp_path() -> Path:
    """Gitignored (artifacts/_temp/): the receipt of the last passing ``broad`` run."""
    return ROOT / "artifacts" / "_temp" / "broad_gate_stamp.json"


def _installed_versions() -> str:
    """Installed distributions as sorted ``name==version`` lines (a dependency upgrade changes the key)."""
    from importlib import metadata

    return "\n".join(
        sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()})
    )


def _interpreter_id() -> str:
    return f"{sys.version}\n{sys.executable}\n{_installed_versions()}"


def _code_tree_key() -> str | None:
    """sha256 over the tracked tree's index entries (minus the state files) and the interpreter; None if git cannot say."""
    try:
        listing = _git_out("ls-files", "-s", "--", *_stamp_pathspec())
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256()
    digest.update(listing.encode("utf-8"))
    digest.update(b"\n" + _interpreter_id().encode("utf-8"))
    return digest.hexdigest()


def _tree_dirty() -> str | None:
    """None when no tracked or untracked file outside the state files differs; otherwise why not."""
    try:
        out = _git_out(
            "status", "--porcelain", "--untracked-files=all", "--", *_stamp_pathspec()
        )
    except (OSError, subprocess.CalledProcessError) as e:
        return f"git status failed ({e})"
    if out:
        return f"modified or untracked files outside the state files ({out.splitlines()[0].strip()})"
    return None


def _stamp_mismatch(key: str) -> str | None:
    """None when the stamp on disk is a readable broad stamp for ``key``; otherwise why not."""
    path = broad_stamp_path()
    if not path.is_file():
        return "no broad stamp"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"unreadable broad stamp ({e})"
    if not isinstance(data, dict) or data.get("schema") != BROAD_STAMP_SCHEMA:
        return "broad stamp has an unknown schema"
    if data.get("key") != key:
        return "the code tree changed since the stamp was written"
    return None


def _write_broad_stamp(key: str) -> Path:
    path = broad_stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = {
        "schema": BROAD_STAMP_SCHEMA,
        "key": key,
        "written_utc": _utcnow(),
        "head": _git_out("rev-parse", "HEAD"),
    }
    path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return path


def _run_or_reuse(family: str, cmd: list[str], reuse: dict | None) -> None:
    """Run one pytest sweep, or record it as executed on Release CI when ``reuse`` covers it."""
    if reuse is not None and family in RELEASE_CI_REUSABLE_FAMILIES:
        record_reuse(family, reuse)
        print(
            f"+ {family}: reused from {RELEASE_CI_WORKFLOW} run {reuse['run_id']} "
            f"(tree {reuse['tree_sha']}); not rerun here",
            flush=True,
        )
        return
    _run(cmd, family=family)


def gate_broad(*, stamp: str | None = None, reuse: dict | None = None) -> None:
    """The broad gate: report audits, the broad pytest sweep, then the paper lanes.

    ``stamp`` is for the ``broad`` CLI only. None (release, rc, direct calls) reads and writes
    no stamp. "reuse" runs the report audits and the stack-reading tests when the stamp matches
    the tree (family ``pytest_stack_readers``). "force"
    (--no-stamp) runs everything. A full run in a clean tree writes the stamp.
    ``reuse`` is the Release CI evidence the rc gate found (see ``_release_ci_reuse``).
    """
    key = dirty = None
    if stamp is not None:
        key = _code_tree_key()
        dirty = _tree_dirty()
    if stamp == "reuse":
        block = dirty or ("code-tree key unavailable" if key is None else _stamp_mismatch(key))
        stack_tests = _stack_reader_tests() if block is None else []
        if block is None and not stack_tests:
            block = "no tracked test file names a stack file"
        if block is None:
            print(
                f"+ broad stamp reused: code-tree key {key} matches a passing broad run; "
                f"running the report audits and {len(stack_tests)} stack-reading test file(s)",
                flush=True,
            )
            gate_report()
            _run(
                [sys.executable, "-m", "pytest", *stack_tests, "-q", "-m", BROAD_MARKER_EXPR,
                 "--tb=short", "-rs"],
                family="pytest_stack_readers",
            )
            return
        print(f"+ broad stamp not reused: {block}; running the full broad gate", flush=True)
    _run(
        [sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts", "examples"],
        family="compileall",
    )
    gate_report()
    # Per-node evidence: each sweep writes JUnit into the gitignored
    # artifacts/attestations/ dir (same provenance home as the RC
    # attestation) so RC outcomes can be compared node-by-node against
    # blocking CI JUnit via scripts/compare_pytest_junit.py. `-rs`
    # keeps skip/xfail reasons in the console log as well.
    junit_dir = ROOT / "artifacts" / "attestations"
    junit_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        BROAD_MARKER_EXPR,
        "--tb=short",
        "-rs",
        *PYTEST_XDIST_ARGS,
        f"--junitxml={junit_dir / 'rc-pytest-broad.xml'}",
    ]
    for path in BROAD_PYTEST_IGNORE:
        cmd.extend(["--ignore", path])
    _run_or_reuse("pytest_broad", cmd, reuse)
    gate_pytest_lanes()
    if stamp is not None and key is not None and dirty is None and reuse is None:
        print(f"+ broad stamp written: {_write_broad_stamp(key)}", flush=True)


def gate_release(*, reuse: dict | None = None) -> None:
    gate_broad(reuse=reuse)
    junit_dir = ROOT / "artifacts" / "attestations"
    junit_dir.mkdir(parents=True, exist_ok=True)
    slow_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        SLOW_MARKER_EXPR,
        "--tb=short",
        "-rs",
        *PYTEST_XDIST_ARGS,
        f"--junitxml={junit_dir / 'rc-pytest-slow.xml'}",
    ]
    for path in BROAD_PYTEST_IGNORE:
        slow_cmd.extend(["--ignore", path])
    _run_or_reuse("pytest_slow", slow_cmd, reuse)
    notebook_cmd = [
        sys.executable,
        "-m",
        "pytest",
        *NOTEBOOK_PYTEST_TARGETS,
        "-q",
        "-m",
        NOTEBOOK_MARKER_EXPR,
        "--tb=short",
        "-rs",
        f"--junitxml={junit_dir / 'rc-pytest-notebook.xml'}",
    ]
    for path in BROAD_PYTEST_IGNORE:
        notebook_cmd.extend(["--ignore", path])
    _run_or_reuse("pytest_notebook", notebook_cmd, reuse)
    _run([sys.executable, "-m", "mkdocs", "build", "--strict"], family="docs_build_strict")
    run_release_examples()


# --- Release CI evidence for the rc gate (pytest family reuse) ------------------
#
# release_ci.yml runs the unfiltered pytest sweep on main, on its schedule and on dispatch.
# When the run for HEAD passed on this exact tree, its JUnit stands in for the three pytest
# families below and the rc gate does not rerun them. Lanes stay local: Release CI runs no
# paper lanes. Any doubt runs the full rc, and the reason is printed.

RELEASE_CI_WORKFLOW = "release_ci.yml"
# The release-test matrix legs in release_ci.yml (tests/test_run_test_gate_stamp_reuse.py pins them).
RELEASE_CI_LEGS = ("3.11", "3.14")
RELEASE_CI_REUSABLE_FAMILIES = ("pytest_broad", "pytest_slow", "pytest_notebook")
RELEASE_CI_RUN_FIELDS = "databaseId,headSha,headBranch,event,status,conclusion,createdAt,updatedAt"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _gh_json(*args: str) -> tuple[object | None, str]:
    """Parsed JSON from one gh call, or (None, why)."""
    try:
        res = _gh(*args)
    except OSError as e:
        return None, f"gh is unavailable ({e})"
    if res.returncode != 0:
        return None, f"gh failed ({res.stderr.strip()})"
    try:
        return json.loads(res.stdout), ""
    except ValueError as e:
        return None, f"gh printed no JSON ({e})"


def _runs_unfiltered_sweep(run: dict) -> bool:
    """Mirror release_ci.yml: the unfiltered sweep runs on any dispatch, and on push or schedule on main."""
    event = run.get("event")
    return event == "workflow_dispatch" or (
        event in ("push", "schedule") and run.get("headBranch") == "main"
    )


def _download_release_ci_junit(run_id: str, sha: str) -> list[Path]:
    """Download the run's pytest JUnit into artifacts/attestations/ci-junit/<sha>, cleared first."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"{sha!r} is not a full commit SHA")
    target = ROOT / "artifacts" / "attestations" / "ci-junit" / sha
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    res = _gh("run", "download", run_id, "--pattern", "pytest-results-*", "--dir", str(target))
    if res.returncode != 0:
        raise RuntimeError(f"gh run download failed ({res.stderr.strip()})")
    found = sorted(target.rglob("pytest-results-*.xml"))
    if not found:
        raise RuntimeError(f"run {run_id} has no pytest-results-*.xml artifact")
    return found


def _release_ci_junit_problem(paths: list[Path]) -> str | None:
    """None when every matrix leg's JUnit reads, holds test cases and records no failure or error."""
    names = {p.name for p in paths}
    missing = [
        f"pytest-results-{leg}.xml"
        for leg in RELEASE_CI_LEGS
        if f"pytest-results-{leg}.xml" not in names
    ]
    if missing:
        return f"JUnit missing for {missing}"
    for path in paths:
        try:
            cases = list(ET.parse(path).getroot().iter("testcase"))
        except (ET.ParseError, OSError) as e:
            return f"JUnit {path.name} cannot be read ({e})"
        if not cases:
            return f"JUnit {path.name} holds no test cases"
        red = sum(1 for c in cases if c.find("failure") is not None or c.find("error") is not None)
        if red:
            return f"JUnit {path.name} records {red} failure(s) or error(s)"
    return None


def _release_ci_reuse() -> tuple[dict | None, str]:
    """The Release CI run that stands in for the local pytest sweeps, or (None, why not)."""
    try:
        head = _git_out("rev-parse", "HEAD")
        tree = _git_out("rev-parse", "HEAD^{tree}")
    except (OSError, subprocess.CalledProcessError) as e:
        return None, f"git failed ({e})"
    runs, why = _gh_json(
        "run", "list", "--workflow", RELEASE_CI_WORKFLOW, "--commit", head,
        "--status", "success", "--limit", "1", "--json", RELEASE_CI_RUN_FIELDS,
    )
    if runs is None:
        return None, why
    if not isinstance(runs, list) or not runs or not isinstance(runs[0], dict):
        return None, f"no successful {RELEASE_CI_WORKFLOW} run for HEAD"
    run = runs[0]
    run_id = str(run.get("databaseId", ""))
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return None, f"run {run_id} concluded {run.get('conclusion')!r}, not 'success'"
    if not _runs_unfiltered_sweep(run):
        return None, (
            f"run {run_id} is a {run.get('event')} run on {run.get('headBranch')!r}; "
            "its pytest sweep is filtered"
        )
    head_sha = str(run.get("headSha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        return None, f"run {run_id} has no full head SHA"
    try:
        run_tree = _git_out("rev-parse", f"{head_sha}^{{tree}}")
    except (OSError, subprocess.CalledProcessError):
        return None, f"run {run_id} head {head_sha} is not a local commit"
    if run_tree != tree:
        return None, f"run {run_id} tree {run_tree} differs from HEAD^{{tree}} {tree}"
    if not run.get("createdAt") or not run.get("updatedAt"):
        return None, f"run {run_id} has no timestamps"
    try:
        paths = _download_release_ci_junit(run_id, head_sha)
    except (OSError, RuntimeError) as e:
        return None, str(e)
    problem = _release_ci_junit_problem(paths)
    if problem is not None:
        return None, problem
    reuse = {
        "workflow": RELEASE_CI_WORKFLOW,
        "run_id": run_id,
        "head_sha": head_sha,
        "tree_sha": tree,
        "event": run.get("event"),
        "head_branch": run.get("headBranch"),
        "conclusion": "success",
        "created": run["createdAt"],
        "updated": run["updatedAt"],
        "junit": sorted(p.name for p in paths),
    }
    return reuse, ""


def _reuse_summary(reuse: dict | None) -> dict | None:
    if reuse is None:
        return None
    return {
        **reuse,
        "families": sorted(RELEASE_CI_REUSABLE_FAMILIES),
        # No local RC sweep ran, so there is no RC JUnit for the per-node comparison.
        "skipped_locally": ["junit_parity"],
    }


def gate_rc() -> None:
    """Full Release Candidate verification gate.

    Guarantees the invariant:
        PRE_RELEASE_GATE >= RELEASE_CI_GATE
    for all blocking test and check families.

    Executes:
    0. Release CI reuse, when its run for HEAD passed on this tree: the three pytest
       sweep families are recorded from its JUnit, not rerun (the reason is printed otherwise)
    1. gate_release() (broad + slow + mkdocs strict + release examples)
    2. Isolated package build (wheel and sdist)
    3. Twine check
    4. Isolated virtualenv wheel install and smoke simulation
    5. Clean working tree check
    6. Authoritative machine-readable receipt generation
    """
    print("=== STARTING RELEASE CANDIDATE (RC) GATE ===", flush=True)
    # Precondition, not a test: a missing optional dependency makes modules skip
    # at collection, so their tests produce no node IDs and the gate silently
    # under-covers release CI while still reporting PASS. Checked first so the
    # failure is loud and cheap.
    _run([sys.executable, "scripts/check_environment_parity.py"], family="environment_parity")
    reuse, why = _release_ci_reuse()
    if reuse is None:
        print(f"+ Release CI reuse off: {why}. Running the full RC sweeps.", flush=True)
        gate_release()

        # Per-node RC-vs-CI comparison. gate_release() has just written the three
        # RC JUnit sweeps; this compares every node ID against the blocking CI
        # evidence for this exact commit. Advisory until now, which meant a
        # divergence could pass unread -- a check nobody runs is not a check.
        _run([sys.executable, "scripts/check_junit_parity.py"], family="junit_parity")
    else:
        print(
            f"+ Release CI reuse on: run {reuse['run_id']} (tree {reuse['tree_sha']}) supplies "
            f"{', '.join(sorted(RELEASE_CI_REUSABLE_FAMILIES))}; those sweeps are not rerun",
            flush=True,
        )
        gate_release(reuse=reuse)

    with tempfile.TemporaryDirectory(prefix="jaxfne_rc_dist_") as tmp_dist:
        print(f"+ Building distribution artifacts in {tmp_dist}...", flush=True)
        _run(
            [sys.executable, "-m", "build", "--outdir", tmp_dist, str(ROOT)], family="package_build"
        )

        print("+ Running twine check on candidate artifacts...", flush=True)
        whl_sdist = [str(p) for p in Path(tmp_dist).glob("*") if p.suffix in (".whl", ".gz")]
        _run([sys.executable, "-m", "twine", "check", *whl_sdist], family="twine_check")

        whls = list(Path(tmp_dist).glob("*.whl"))
        if not whls:
            raise RuntimeError(f"No wheel built in {tmp_dist}")
        candidate_whl = whls[0]

        with tempfile.TemporaryDirectory(prefix="jaxfne_rc_venv_") as tmp_venv:
            print(f"+ Creating isolated test venv in {tmp_venv}...", flush=True)
            _run([sys.executable, "-m", "venv", tmp_venv])

            venv_py = (
                Path(tmp_venv)
                / ("Scripts" if os.name == "nt" else "bin")
                / ("python.exe" if os.name == "nt" else "python")
            )
            print("+ Installing candidate wheel into isolated venv...", flush=True)
            _run(
                [str(venv_py), "-m", "pip", "install", "--upgrade", "pip"],
                family="isolated_wheel_smoke",
            )
            _run(
                [str(venv_py), "-m", "pip", "install", str(candidate_whl)],
                family="isolated_wheel_smoke",
            )

            run_isolated_wheel_smoke(venv_py, _local_package_version(), cwd=tmp_venv)

    attestation_out = write_rc_attestation(reuse=reuse)
    print(f"\nRC GATE PASS: attestation written to {attestation_out}", flush=True)


def _git_out(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def build_rc_attestation(reuse: dict | None = None) -> dict:
    """Assemble the RC attestation strictly from observed execution results.

    Nothing here is asserted as a literal: ``observed_pass_families`` is derived
    from the execution ledger, ``pre_release_subsumes_ci`` is derived from that
    set, and ``status`` is derived from both. A family that was declared but
    never executed has no ledger record and therefore cannot appear. A family
    reused from Release CI has a record that names its run and tree
    (``reused_from``); ``release_ci_reuse`` names the run for the whole gate.
    """
    families = [_OBSERVATIONS[k] for k in sorted(_OBSERVATIONS)]
    observed = observed_pass_families()
    subsumes = RELEASE_CI_GATE_FAMILIES.issubset(observed)
    all_passed = all(rec["status"] == "PASS" for rec in families)
    return {
        "schema": ATTESTATION_SCHEMA,
        "gate": "rc",
        "commit_sha": _git_out("rev-parse", "HEAD"),
        "tree_sha": _git_out("rev-parse", "HEAD^{tree}"),
        "working_tree_clean": not bool(_git_out("status", "--porcelain")),
        "timestamp_utc": _utcnow(),
        "required_families": sorted(RELEASE_CI_GATE_FAMILIES),
        "declared_families": sorted(GATE_CHECK_FAMILIES["rc"]),
        "observed_pass_families": sorted(observed),
        "families": families,
        # DERIVED from the ledger, never asserted as a literal.
        "pre_release_subsumes_ci": bool(subsumes),
        "release_ci_reuse": _reuse_summary(reuse),
        "status": "PASS" if (subsumes and all_passed) else "FAIL",
    }


def attestation_path() -> Path:
    """Untracked by design: a tracked attestation changes the SHA it attests to."""
    override = os.environ.get("JAXFNE_RC_ATTESTATION")
    return Path(override) if override else DEFAULT_ATTESTATION_PATH


def write_rc_attestation(out_path: Path | None = None, *, reuse: dict | None = None) -> Path:
    out = out_path or attestation_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_rc_attestation(reuse), indent=2) + "\n", encoding="utf-8")
    return out


def gate_publication() -> None:
    """The frozen publication record: frozen-path hashes, then Figures 1-7 re-rendered and
    compared pixel-for-pixel with their frozen PNGs, then the repository state snapshot.

    Render and report go under the gitignored ``artifacts/_temp/``, so the tracked
    ``artifacts/publication/equivalence_report.json`` is not rewritten by a gate run.
    """
    for family, cmd in AUDIT_STEPS:
        if family in PUBLICATION_AUDIT_FAMILIES:
            _run(cmd, family=family)
    out = ROOT / "artifacts" / "_temp" / "publication_gate"
    _run(
        [
            sys.executable,
            "scripts/publication_figures/equivalence_gate.py",
            "--render-dir",
            str(out / "render"),
            "--report",
            str(out / "equivalence_report.json"),
        ],
        family="publication_equivalence",
    )
    _run([sys.executable, "scripts/repo_state_snapshot.py"])


# Frozen-path checks the publication gate runs from AUDIT_STEPS.
PUBLICATION_AUDIT_FAMILIES = ("record_integrity", "harness_integrity")


GATES = {
    "dev": gate_dev,
    "report": gate_report,
    "docs": gate_docs,
    "broad": gate_broad,
    "release": gate_release,
    "rc": gate_rc,
    "release-candidate": gate_rc,
    "publication": gate_publication,
    "examples": gate_examples,
    "paper-lanes": gate_pytest_lanes,
    "wheel-smoke": gate_wheel_smoke,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gate",
        choices=sorted(GATES),
        help="which validation gate to run",
    )
    parser.add_argument(
        "--venv-python",
        default=None,
        help="venv python for the wheel-smoke gate",
    )
    parser.add_argument(
        "--expected-version",
        default=None,
        help="expected jaxfne version for the wheel-smoke gate (default: pyproject version)",
    )
    parser.add_argument(
        "--lane",
        action="append",
        default=None,
        help="paper-lanes gate only: run this lane folder (repeatable; default: every discovered lane)",
    )
    parser.add_argument(
        "--no-stamp",
        action="store_true",
        help="broad gate only: run every family even when a passing run's stamp matches the code tree",
    )
    args = parser.parse_args(argv)
    if args.lane is not None and args.gate != "paper-lanes":
        parser.error("--lane applies to the paper-lanes gate only")
    if args.no_stamp and args.gate != "broad":
        parser.error("--no-stamp applies to the broad gate only")
    if args.gate == "wheel-smoke":
        gate_wheel_smoke(venv_python=args.venv_python, expected_version=args.expected_version)
    elif args.gate == "paper-lanes":
        gate_pytest_lanes(args.lane)
    elif args.gate == "broad":
        gate_broad(stamp="force" if args.no_stamp else "reuse")
    else:
        GATES[args.gate]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
