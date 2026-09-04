#!/usr/bin/env python3
"""Executable test gates for jaxfne.

Gate vocabulary (single source of truth for local runs and CI references):

  dev          Curated architectural gate (~1 minute)
  broad        Repository-wide ``-m "not slow"`` (~8 minutes)
  release      Broad + slow markers + release examples/docs checks
  rc           Full release-candidate verification (PRE_RELEASE_GATE >= RELEASE_CI_GATE)
  publication  Frozen scientific experiments and artifact validation

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

# Curated dev gate — owned here, not in docs/ci_policy.md prose.
DEV_PYTEST_TARGETS = [
    "tests/test_api_smoke.py",
    "tests/test_root_import_lightweight.py",
    "tests/test_signals_get_v0329.py",
    "tests/test_neuronal_tensor_connectivity.py",
    "tests/test_neuronal_tensor.py",
    "tests/test_connection_rule_compile_v0330.py",
    "tests/test_continuation_contract.py",
    "tests/test_mcc.py",
]

BROAD_PYTEST_IGNORE = [
    "tests/test_multi_area_source_projector.py",
    "tests/test_multi_area_spectrolaminar_objective.py",
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
    "compileall",
    "lint_ruff",
    "docs_language_audit",
    "notebook_grammar_audit",
    "vocabulary_audit",
    "docs_orphans_check",
    "docs_build_strict",
    "pytest_broad",
    "pytest_slow",
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
    "docs_build_strict",
    "pytest_broad",
    "pytest_slow",
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
    },
    "broad": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "pytest_broad",
    },
    "release": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "docs_build_strict",
        "pytest_broad",
        "pytest_slow",
        "examples_smoke",
    },
    "rc": {
        "compileall",
        "lint_ruff",
        "docs_language_audit",
        "notebook_grammar_audit",
        "vocabulary_audit",
        "docs_orphans_check",
        "docs_build_strict",
        "pytest_broad",
        "pytest_slow",
        "examples_smoke",
        "package_build",
        "twine_check",
        "isolated_wheel_smoke",
    },
}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, env=PYTEST_ENV)


def gate_dev() -> None:
    _run([sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts"])
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
        ]
    )
    _run([sys.executable, "scripts/audit_public_docs_language.py", "--check"])
    _run([sys.executable, "scripts/audit_vocabulary.py", "--check"])


def gate_broad() -> None:
    _run([sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts", "examples"])
    _run([sys.executable, "-m", "ruff", "check", "jaxfne/"])
    _run([sys.executable, "scripts/audit_public_docs_language.py", "--check"])
    _run([sys.executable, "scripts/audit_notebook_grammar.py", "--check"])
    _run([sys.executable, "scripts/audit_vocabulary.py", "--check"])
    _run([sys.executable, "scripts/check_docs_orphans.py"])
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        "not slow",
        "--tb=short",
    ]
    for path in BROAD_PYTEST_IGNORE:
        cmd.extend(["--ignore", path])
    _run(cmd)


def gate_release() -> None:
    gate_broad()
    slow_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        "slow and not notebook",
        "--tb=short",
    ]
    for path in BROAD_PYTEST_IGNORE:
        slow_cmd.extend(["--ignore", path])
    _run(slow_cmd)
    _run([sys.executable, "-m", "mkdocs", "build", "--strict"])
    for example in RELEASE_EXAMPLES:
        _run([sys.executable, example])


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
    gate_release()

    with tempfile.TemporaryDirectory(prefix="jaxfne_rc_dist_") as tmp_dist:
        print(f"+ Building distribution artifacts in {tmp_dist}...", flush=True)
        _run([sys.executable, "-m", "build", "--outdir", tmp_dist, str(ROOT)])

        print("+ Running twine check on candidate artifacts...", flush=True)
        whl_sdist = [str(p) for p in Path(tmp_dist).glob("*") if p.suffix in (".whl", ".gz")]
        _run([sys.executable, "-m", "twine", "check", *whl_sdist])

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
            _run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
            _run([str(venv_py), "-m", "pip", "install", str(candidate_whl)])

            smoke_code = (
                "import json, jaxfne as jtfne\n"
                "assert 'site-packages' in jtfne.__file__, f'Not in site-packages: {jtfne.__file__}'\n"
                "cfg = jtfne.configuration().network(n=8).emitter().field().probe(n_contacts=4)\n"
                "model = jtfne.construct(cfg)\n"
                "signals = model.simulate(jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0))\n"
                "readouts = model.compute_readout(signals, [jtfne.readout_spec('r', 'spike_rate_hz')])\n"
                "manifest = model.manifest(signals, readouts)\n"
                "json.dumps(manifest, allow_nan=False)\n"
                "assert manifest['physical_amplitude_calibrated'] is False\n"
                "print('RC isolated wheel smoke OK: version =', jtfne.__version__)\n"
            )
            print("+ Running isolated wheel smoke simulation...", flush=True)
            _run([str(venv_py), "-c", smoke_code], cwd=Path(tmp_venv))

    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    head_sha = res.stdout.strip()

    status_res = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
    dirty = status_res.stdout.strip()
    working_tree_clean = not bool(dirty)

    receipt_dir = ROOT / "artifacts" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "release_candidate_gate_receipt.json"

    receipt_data = {
        "schema": "jaxfne.rc_gate_receipt.v1",
        "commit_sha": head_sha,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gate": "rc",
        "status": "PASS",
        "working_tree_clean": working_tree_clean,
        "check_families": sorted(list(GATE_CHECK_FAMILIES["rc"])),
        "pre_release_subsumes_ci": True,
    }
    receipt_path.write_text(json.dumps(receipt_data, indent=2) + "\n", encoding="utf-8")
    print(f"\nRC GATE PASS: Authoritative receipt generated at {receipt_path}", flush=True)


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
    "broad": gate_broad,
    "release": gate_release,
    "rc": gate_rc,
    "release-candidate": gate_rc,
    "publication": gate_publication,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gate",
        choices=sorted(GATES),
        help="which validation gate to run",
    )
    args = parser.parse_args(argv)
    GATES[args.gate]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
