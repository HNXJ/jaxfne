#!/usr/bin/env python3
"""Executable test gates for jaxfne.

Gate vocabulary (single source of truth for local runs and CI references):

  dev          Curated architectural gate (~1 minute)
  report       All audit checks without the pytest sweeps (~1 minute)
  docs         The documentation subset of the audit checks (~30 seconds)
  broad        Repository-wide ``-m "not slow"`` (~8 minutes)
  release      Broad + slow markers + release examples/docs checks
  rc           Full release-candidate verification (PRE_RELEASE_GATE >= RELEASE_CI_GATE)
  publication  Frozen scientific experiments and artifact validation
  examples     The RELEASE_EXAMPLES smoke list only (owner of the list)
  wheel-smoke  The canonical isolated-wheel smoke in a given venv
               (owner of the smoke body; needs --venv-python)

Documentation must refer to these gate names and this script (or Makefile
targets), not duplicate file lists in prose.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
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
    "lint_ruff",
    "docs_language_audit",
    "notebook_grammar_audit",
    "vocabulary_audit",
    "docs_orphans_check",
    "doc_code_integrity_audit",
    "record_integrity",
    "replication_audit",
    "etude_no_resim",
    "harness_integrity",
    "tree_size_audit",
    "docs_build_strict",
    "pytest_broad",
    "pytest_slow",
    "pytest_notebook",
    "examples_smoke",
    "package_build",
    "twine_check",
    "isolated_wheel_smoke",
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
    },
    "broad": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "tree_size_audit",
        "pytest_broad",
    },
    "report": {
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "doc_code_integrity_audit",
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "tree_size_audit",
    },
    "docs": {
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
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "tree_size_audit",
        "docs_build_strict",
        "pytest_broad",
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
        "record_integrity",
        "replication_audit",
        "etude_no_resim",
        "harness_integrity",
        "tree_size_audit",
        "docs_build_strict",
        "pytest_broad",
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
    for example in RELEASE_EXAMPLES:
        _run([sys.executable, example], family="examples_smoke")
        print(f"PASS: {example}", flush=True)
    # Restore static docs figures that examples mirror to keep tree clean
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
DEV_AUDIT_FAMILIES = ("record_integrity", "replication_audit", "etude_no_resim", "harness_integrity")


# The audit checks the broad gate runs before its pytest sweep, owned here and
# shared with the report and docs gates. Order matters only for readability.
AUDIT_STEPS: list[tuple[str, list[str]]] = [
    ("lint_ruff", [sys.executable, "-m", "ruff", "check", "jaxfne/"]),
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
]

# The documentation subset: what a docs edit needs checked before pushing.
DOCS_AUDIT_FAMILIES = {
    "docs_language_audit",
    "notebook_grammar_audit",
    "vocabulary_audit",
    "docs_orphans_check",
    "doc_code_integrity_audit",
}


def gate_report() -> None:
    """The audit checks only: a gate status report without the pytest sweeps."""
    for family, cmd in AUDIT_STEPS:
        _run(cmd, family=family)


def gate_docs() -> None:
    """Quick docs gate: the documentation subset of the audit checks."""
    for family, cmd in AUDIT_STEPS:
        if family in DOCS_AUDIT_FAMILIES:
            _run(cmd, family=family)


def gate_broad() -> None:
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
    _run(cmd, family="pytest_broad")


def gate_release() -> None:
    gate_broad()
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
    _run(slow_cmd, family="pytest_slow")
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
    _run(notebook_cmd, family="pytest_notebook")
    _run([sys.executable, "-m", "mkdocs", "build", "--strict"], family="docs_build_strict")
    run_release_examples()


def gate_rc() -> None:
    """Full Release Candidate verification gate.

    Guarantees the invariant:
        PRE_RELEASE_GATE >= RELEASE_CI_GATE
    for all blocking test and check families.

    Executes:
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
    gate_release()

    # Per-node RC-vs-CI comparison. gate_release() has just written the three
    # RC JUnit sweeps; this compares every node ID against the blocking CI
    # evidence for this exact commit. Advisory until now, which meant a
    # divergence could pass unread -- a check nobody runs is not a check.
    _run([sys.executable, "scripts/check_junit_parity.py"], family="junit_parity")

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

    attestation_out = write_rc_attestation()
    print(f"\nRC GATE PASS: attestation written to {attestation_out}", flush=True)


def _git_out(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def build_rc_attestation() -> dict:
    """Assemble the RC attestation strictly from observed execution results.

    Nothing here is asserted as a literal: ``observed_pass_families`` is derived
    from the execution ledger, ``pre_release_subsumes_ci`` is derived from that
    set, and ``status`` is derived from both. A family that was declared but
    never executed has no ledger record and therefore cannot appear.
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
        "status": "PASS" if (subsumes and all_passed) else "FAIL",
    }


def attestation_path() -> Path:
    """Untracked by design: a tracked attestation changes the SHA it attests to."""
    override = os.environ.get("JAXFNE_RC_ATTESTATION")
    return Path(override) if override else DEFAULT_ATTESTATION_PATH


def write_rc_attestation(out_path: Path | None = None) -> Path:
    out = out_path or attestation_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_rc_attestation(), indent=2) + "\n", encoding="utf-8")
    return out


def gate_publication() -> None:
    """Placeholder hook for frozen experiment manifests.

    Extend this gate when publication-facing receipts have a single entrypoint.
    """
    _run([sys.executable, "scripts/repo_state_snapshot.py"])
    print(
        "publication gate: repo_state_snapshot only; "
        "add frozen experiment runners here when contracted.",
        flush=True,
    )


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
    args = parser.parse_args(argv)
    if args.gate == "wheel-smoke":
        gate_wheel_smoke(venv_python=args.venv_python, expected_version=args.expected_version)
    else:
        GATES[args.gate]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
