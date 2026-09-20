#!/usr/bin/env python3
"""Frozen-bundle no-resimulation gate (Batch B5).

Etude figure/observation generation must satisfy Δsimulation = 0: every
function in an etude script except the declared simulation entry points
(`main`, `*_config`, `build_*`, `simulate_*`) must contain no
simulate/construct calls. Observation helpers stay pure functions of arrays.

Usage:
    python scripts/gate_etude_no_resim.py --check
"""

from __future__ import annotations

import argparse
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SCRIPTS = [
    "scripts/run_multiscale_observation_etude.py",
    "scripts/run_heterogeneous_emitters_etude.py",
    "scripts/run_experiment_a.py",
    "scripts/consolidate_hdp_controllability_etude.py",
]

# Functions allowed to drive simulation (entry points + builders).
SIM_ENTRY_PREFIXES = ("main", "simulate_", "run_condition", "run_")
SIM_ENTRY_EXACT = {"build_config", "build_manifest", "izh_config",
                   "hei_config", "mcc3_config", "mcc3_runtime", "mcc3_specs"}

SIM_CALLS = {"simulate", "construct", "construct_neuronal_tensor",
             "simulate_homeostatic_ei", "simulate_edge_recurrent_izhikevich_hdp"}


def _is_entry(name: str) -> bool:
    if name in SIM_ENTRY_EXACT:
        return True
    return name == "main" or any(name.startswith(p) for p in SIM_ENTRY_PREFIXES)


def _calls_sim(node: ast.AST) -> list[str]:
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in SIM_CALLS:
                found.append(name)
    return found


def audit() -> list[dict]:
    violations = []
    for rel in SCRIPTS:
        path = ROOT / rel
        if not path.exists():
            violations.append({"file": rel, "function": None,
                               "issue": "script missing"})
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_entry(node.name):
                    continue
                hits = _calls_sim(node)
                if hits:
                    violations.append(
                        {"file": rel, "function": node.name,
                         "issue": f"non-entry function calls {sorted(set(hits))}"})
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    violations = audit()
    if not violations:
        print(f"etude no-resim: pass ({len(SCRIPTS)} scripts)")
        return 0
    print(f"etude no-resim: {len(violations)} violations")
    for v in violations:
        print(f"  {v['file']}:{v['function']}: {v['issue']}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
