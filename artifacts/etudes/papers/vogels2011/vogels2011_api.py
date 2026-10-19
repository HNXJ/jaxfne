"""vogels2011 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``Configuration -> construct -> Model -> simulate -> Signals`` (checklist
E1). Estimators come from ``jaxfne.analysis`` (checklist E3). No scored run:
outputs land in the caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype and seeds (checklist P4).
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jaxfne as J
from scripts import lane_kit


def build_spec(n=20, seed=0):
    """The lane network built only from the public API (fill: port rows)."""
    cfg = (
        J.Configuration()
        .network(name="vogels2011", n=int(n), cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
    )
    return cfg.connectivity(edge_seed=int(seed))


def build_sim(T_s=1.0, dt_ms=0.1):
    """The ``Simulation`` for a run (fill: duration, dt, drive)."""
    return J.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=float(dt_ms))


def run(seed, tag, out_dir, *, T_s=1.0, signed=False):
    """Build, run and write the raster plus the ``manifest.json`` the scorer reads."""
    import numpy as np

    lane_kit.refuse_scored_dir(out_dir, signed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = "spikes_seed%d_%s.npz" % (int(seed), tag)
    sha = lane_kit.git_sha(_repo_root)
    prior = lane_kit.refuse_existing(out, fname, sha, signed=signed)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    model = J.construct(build_spec(seed=seed))
    sim = build_sim(T_s=T_s)
    spikes = np.zeros((sim.n_steps, model.n), dtype=bool)  # fill: simulate
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
              "steps": int(spikes.shape[0]), "n": int(spikes.shape[1])}
    entries[fname] = entry
    manifest = {"git_sha": sha, "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps}


def score(spikes):
    """Score a raster with ``jaxfne.analysis`` estimators (fill: estimands)."""
    from jaxfne.analysis import activity_regime as _activity_regime

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    return _activity_regime(arr, 0.1)
