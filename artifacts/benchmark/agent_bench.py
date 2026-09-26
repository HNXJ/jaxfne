"""Frozen Atlas task set and scorer for the skill/tool benchmark (0.5.5 item 5d).

Agent-native step 7 (canonical tasks with frozen expected properties) and the
scorer for step 9 (skill+tools vs direct repository use). A task is one
Atlas AT: the agent receives ``intent`` (runner docstring) and ``spec``
(``at_manifest.at_spec``) and must return a bundle in the ``at_bundle``
convention, ``{arm: {"model", "signals", "hdp"}}``. The expected properties
are extracted from the reference runner's own bundle, so the reference scores
full marks by construction and every property is read from executed objects.

Properties are permutation-invariant (sorted multisets, degree sequences), so
an equivalent network built through another path scores the same. Classes:

    model        n_neurons, emitter, n_edges, weights, mechanisms, delays, degrees
    execution    n_steps, dt_ms, finite
    observation  recorded outputs
    plasticity   whether W changed during the arm (from the arm's own hdp)
    dynamics     sorted per-neuron spike counts and mean V_m (atol 1e-3)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

TASK_SET = Path(__file__).with_name("agent_tasks_055.json")
CLASSES: dict[str, tuple[str, ...]] = {
    "model": ("n_neurons", "emitter", "n_edges", "weights", "mechanisms", "delays", "degrees"),
    "execution": ("n_steps", "dt_ms", "finite"),
    "observation": ("recorded",),
    "plasticity": ("plastic",),
    # Structure alone scored a differently driven model 1.0 in the 5d pilot
    # (direct arm, AT-02: 12 spikes vs 8); dynamics close that blind spot.
    "dynamics": ("spike_counts", "mean_vm"),
}
_VM_ATOL = 1e-3  # float32 means; exact equality would bind the set to one platform


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def arm_properties(arm: dict[str, Any]) -> dict[str, Any]:
    """Permutation-invariant properties of one executed arm."""
    from jaxfne import agent

    model, signals, hdp = arm["model"], arm["signals"], arm.get("hdp")
    rows = model.edge_table()
    w0 = np.asarray([r["weight"] for r in rows], dtype=float)
    t = np.asarray(signals.time_ms, dtype=float)
    recorded = agent._recorded(signals)
    finite = all(np.isfinite(np.asarray(signals.get(k), dtype=float)).all() for k in recorded)
    plastic = False
    if hdp and hdp.get("w_final") is not None:
        wf = np.asarray(hdp["w_final"], dtype=float).ravel()
        plastic = wf.shape != w0.shape or not np.allclose(wf, w0)
    out_deg = Counter(Counter(r["pre"] for r in rows).values())
    in_deg = Counter(Counter(r["post"] for r in rows).values())
    return {
        "n_neurons": len(model.neuron_table()),
        "emitter": type(model.params.get("emitter")).__name__,
        "n_edges": len(rows),
        "weights": _digest(sorted(round(float(x), 6) for x in w0)),
        # TFNE-compiled receptors carry an internal suffix (AMPA__tfne__0); the
        # base name is the mechanism, as in jaxfne.agent.compare.
        "mechanisms": dict(sorted(
            Counter(str(r["receptor_type"]).split("__")[0] for r in rows).items())),
        "delays": _digest(sorted(Counter(int(r["delay_steps"]) for r in rows).items())),
        "degrees": _digest([sorted(out_deg.items()), sorted(in_deg.items())]),
        "n_steps": int(t.shape[0]),
        "dt_ms": round(float(t[1] - t[0]), 9) if t.shape[0] > 1 else None,
        "finite": bool(finite),
        "recorded": sorted(recorded),
        "plastic": bool(plastic),
        "spike_counts": sorted(int(x) for x in np.asarray(signals.spikes).sum(axis=0)),
        "mean_vm": sorted(round(float(x), 4)
                          for x in np.asarray(signals.V_m, dtype=float).mean(axis=0)),
    }


def _same(key: str, got: Any, want: Any) -> bool:
    if key == "mean_vm":
        return len(got) == len(want) and bool(np.allclose(got, want, rtol=0.0, atol=_VM_ATOL))
    return got == want


def freeze_task(at_id: str) -> dict[str, Any]:
    """One task: intent, spec, and expected per-arm properties from the reference bundle."""
    from artifacts.atlas.at_bundle import bundle
    from artifacts.atlas.at_manifest import at_spec, resolve_runner, spec_digest

    spec = at_spec(at_id)
    doc = (resolve_runner(at_id).__doc__ or "").strip().split("\n\n")[0]
    arms = bundle(at_id)
    return {
        "at_id": at_id,
        "intent": " ".join(doc.split()),
        "spec": json.loads(json.dumps(spec, default=str)),
        "spec_digest": spec_digest(spec),
        "arms": {name: arm_properties(arm) for name, arm in arms.items()},
    }


def score(task: dict[str, Any], candidate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-class fraction of expected properties the candidate bundle reproduces."""
    got = {name: arm_properties(arm) for name, arm in candidate.items()}
    classes: dict[str, dict[str, Any]] = {}
    for cls, keys in CLASSES.items():
        failed = []
        total = 0
        for arm, want in task["arms"].items():
            for k in keys:
                total += 1
                if arm not in got or not _same(k, got[arm][k], want[k]):
                    failed.append(f"{arm}.{k}")
        classes[cls] = {"score": (total - len(failed)) / total if total else 1.0, "failed": failed}
    return {
        "at_id": task["at_id"],
        "missing_arms": sorted(set(task["arms"]) - set(got)),
        "extra_arms": sorted(set(got) - set(task["arms"])),
        "classes": classes,
    }


# Benchmark arms (step 9). Both arms work in a disposable linked worktree with
# the reference answers removed; the direct arm also loses the skills and the
# agent surface, so it works from the package and repository alone.
ARMS = ("skills", "direct")
_ANSWER_PATHS = (
    "artifacts/atlas", "artifacts/publication/atlas", "artifacts/benchmark",
    "scripts/generate_atlas_figures.py", "scripts/benchmark_051_matrix.py",
    "artifacts/programme/atlas_gap_052.md", "artifacts/programme/atlas_gap_053.md",
    "artifacts/programme/atlas_gap_054.md", "tests/test_atlas_*.py", "tests/test_agent_bench_055.py",
)
_SKILL_PATHS = ("artifacts/skills", "jaxfne/agent.py", "tests/test_agent_*.py")
SOLUTION = "bench_solution.py"


def load_task(at_id: str) -> dict[str, Any]:
    tasks = json.loads(TASK_SET.read_text(encoding="utf-8"))["tasks"]
    return next(t for t in tasks if t["at_id"] == at_id)


def prepare_worktree(root: Path, arm: str) -> list[str]:
    """Delete answer paths (and, for the direct arm, the skill surface) from a disposable checkout.

    The checkout must be a git repository other than this one. Use a fresh
    single-commit repository built from ``git archive`` after this call: a
    clone or worktree still carries the answers in its history.
    """
    import shutil

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    source = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists() or root.resolve() == source:
        raise ValueError(f"{root} is not a disposable git checkout; refusing to delete files")
    removed = []
    for pattern in _ANSWER_PATHS + (_SKILL_PATHS if arm == "direct" else ()):
        for p in sorted(root.glob(pattern)):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(p.relative_to(root).as_posix())
    return removed


def packet(task: dict[str, Any], arm: str) -> str:
    """The task text a benchmark worker receives (identical except the arm's route)."""
    spec = {k: v for k, v in task["spec"].items() if k not in ("runner", "cited")}
    route = {
        "skills": "Start at `artifacts/skills/jaxfne-core/SKILL.md`; use the task skills it "
                  "routes to and the `jaxfne.agent` operations they name.",
        "direct": "Use the `jaxfne` package and the repository directly.",
    }[arm]
    return f"""# Benchmark task {task['at_id']}

## Goal
Build and run, with jaxfne, the model this task describes, and return the
executed objects. {route}

## Intent
{task['intent']}

## Specification (JSON)
```json
{json.dumps(spec, indent=1)}
```

## Arms
Produce exactly these arms: {', '.join(task['arms'])}. Infer what separates
them from the intent and specification.

## Deliverable
Write `{SOLUTION}` at the repository root defining `bundle()`, which returns
`{{arm_name: {{"model": Model, "signals": Signals, "hdp": dict | None}}}}`:
the constructed jaxfne Model, the Signals its simulation returned, and for
plastic arms the HDP diagnostics captured right after that arm ran
(`model.last_hdp_diagnostics()`), else None.

## Rules
- The specification lists every input that differs from jaxfne's library
  defaults. Anything it does not state takes the default; add no input it
  does not declare (no extra drive, stimulus, noise or plasticity).
- Do not edit anything under `jaxfne/`; do not read git history.
- `bundle()` must finish in under 10 minutes.

## Acceptance
Write a throwaway script that imports `{SOLUTION}`, calls `bundle()` and
prints each arm's neuron count, edge count and number of time steps; run it
and include its output in your report.
"""


def score_solution(at_id: str, solution: Path) -> dict[str, Any]:
    """Import a worker's solution file by path, run ``bundle()``, score it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_solution", solution)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return score(load_task(at_id), mod.bundle())


def main(argv: list[str] | None = None) -> int:
    """freeze (write-once) | packet AT ARM | prepare WORKTREE ARM | score AT SOLUTION."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out", type=Path, default=TASK_SET)
    k = sub.add_parser("packet")
    k.add_argument("at_id")
    k.add_argument("arm", choices=ARMS)
    w = sub.add_parser("prepare")
    w.add_argument("worktree", type=Path)
    w.add_argument("arm", choices=ARMS)
    s = sub.add_parser("score")
    s.add_argument("at_id")
    s.add_argument("solution", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "freeze":
        from artifacts.atlas.at_manifest import REGISTRY

        if args.out.exists():
            raise FileExistsError(f"{args.out} is frozen; remove it deliberately to re-freeze")
        tasks = [freeze_task(at_id) for at_id in REGISTRY]
        args.out.write_text(json.dumps({"schema": 1, "tasks": tasks}, indent=1) + "\n",
                            encoding="utf-8")
        print(f"froze {len(tasks)} tasks -> {args.out}")
    elif args.cmd == "packet":
        print(packet(load_task(args.at_id), args.arm))
    elif args.cmd == "prepare":
        print("\n".join(prepare_worktree(args.worktree, args.arm)))
    else:
        print(json.dumps(score_solution(args.at_id, args.solution), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
