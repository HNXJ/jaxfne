#!/usr/bin/env python3
"""Re-record the MCC-3 retune=True checkpoint after the multi-parameter seed fix.

Runs the same calls as the retune=True path of
``scripts/rerun_etude_figures.py::mode_hdp_mcc3`` (the AGSDR tune, then condition B
at the tuned theta), and writes the replacement digest beside the post-P-048 one:

    artifacts/runs/mcc3_10s_checkpoint/mcc3_10s_post_seedfix_digest.json

Keys that do not depend on the tune (``firing_frozen_theta``,
``abc_table_frozen_theta``) are carried forward unchanged from
``mcc3_10s_post_p048_digest.json``; ``agsdr.theta_hat``, ``firing.B`` and
``abc_table.total_spikes.B`` hold the retune=True values from this run.
Frozen files are never touched.

Usage:
    python scripts/write_mcc3_seedfix_digest.py

Exit codes:
    0  digest written
    1  runtime failure (traceback on stderr)
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jaxfne as J  # noqa: E402
import scripts.mcc3_10s_scientific_checkpoint as M3  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

OUT = ROOT / "artifacts" / "runs" / "mcc3_10s_checkpoint" / "mcc3_10s_post_seedfix_digest.json"
POST_P048 = ROOT / "artifacts" / "runs" / "mcc3_10s_checkpoint" / "mcc3_10s_post_p048_digest.json"


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    specs = M3.mcc3_specs()
    objective = J.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": M3.TARGET_HZ},
        burn_in_ms=M3.BURN_IN_MS,
    )
    base = J.construct(M3.mcc3_config())
    tune_sim = J.simulation(
        duration_ms=M3.TUNE_DURATION_MS,
        dt_ms=M3.DT_MS,
        seed=M3.SIM_SEED,
        runtime=M3.mcc3_runtime(True),
    )
    optimizer = J.agsdr(
        parameters=specs,
        generations=M3.AGSDR_GENERATIONS,
        population_size=M3.AGSDR_POPULATION,
        seed=M3.OPT_SEED,
    )
    print("Running canonical AGSDR (100 ms training sim)...", flush=True)
    tune_result = base.tune(objectives=objective, optimizer=optimizer, simulation=tune_sim)
    thetah = {k: float(v) for k, v in dict(tune_result.best_parameters).items()}

    sim_10s_on = J.simulation(
        duration_ms=M3.DURATION_MS, dt_ms=M3.DT_MS, seed=M3.SIM_SEED, runtime=M3.mcc3_runtime(True)
    )
    print("Running B (theta_hat, HDP on) 10 s...", flush=True)
    B = M3.run_condition(base, thetah, specs, sim_10s_on, objective, "B")

    post_p048 = json.loads(POST_P048.read_text(encoding="utf-8"))

    digest = {
        "schema": "jaxfne.mcc3_10s_post_seedfix_digest.v1",
        "note": (
            "Re-recorded retune=True condition-B fields after 9ade2f8f (Hamm 2026-10-29: "
            "multi-parameter Model.tune seeds each candidate from the optimizer seed and "
            "(gen, row)), an intended optimizer change. The AGSDR tune is re-run under the "
            "new candidate seeding (it reaches the same uniform optimum on this clone); "
            "condition B is recorded under the tuned theta (retune=True CLI path: "
            "agsdr/firing/abc_table). The frozen-theta section is carried forward "
            "unchanged from the post-P-048 digest for the retune=False figure-site path. "
            "Frozen files stay byte-identical. Computed on this clone by the code path "
            "of scripts/rerun_etude_figures.py::mode_hdp_mcc3."
        ),
        "regenerated_after_commit": git_head(),
        "supersedes_fields_of": (
            "artifacts/runs/mcc3_10s_checkpoint/mcc3_10s_post_p048_digest.json"
        ),
        "agsdr": {"theta_hat": thetah},
        "firing": {
            "B": {
                "population_rate_hz": B["population_rate_hz"],
                "E_rate_hz": B["E_rate_hz"],
                "I_rate_hz": B["I_rate_hz"],
            }
        },
        "abc_table": {"total_spikes": {"B": B["spike_total"]}},
        "firing_frozen_theta": post_p048["firing_frozen_theta"],
        "abc_table_frozen_theta": post_p048["abc_table_frozen_theta"],
    }

    OUT.write_text(json.dumps(digest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)
    print(json.dumps({"theta_hat": thetah, "firing_B": digest["firing"]["B"],
                      "spikes_B": B["spike_total"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
