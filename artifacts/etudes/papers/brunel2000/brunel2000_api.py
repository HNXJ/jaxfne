"""brunel2000 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``NeuronalTensor + RuntimeConfiguration(emitter="lif") -> construct ->
Model -> simulate -> Signals`` (checklist E1). Estimators come from
``jaxfne.analysis`` (checklist E3). No scored run: outputs land in the
caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype, seeds and the activity-regime block (checklist P4).

Port decisions live in ``PORT_SPEC.md``; digits denote Fig. 8 of Brunel (2000).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jaxfne as J
assert "site-packages" not in J.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
from jaxfne import neuronal_tensor as nt
from scripts import lane_kit

# Paper constants (SOURCE_P1.md section 2; Fig. 8 set).
J_MV = 0.1
D_MS = 1.5
DT_MS = 0.5
TAU_MS = 20.0
THETA_MV = 20.0
VRESET_MV = 10.0
CONNECTIVITY_P = 0.1
NUTHR_HZ = 10.0
E_FRACTION = 0.8

#: Fig. 8 (g, nuext/nuthr) points (SOURCE_P1.md section 3).
FIG8_POINTS = {
    "A": (3.0, 2.0),
    "B": (6.0, 4.0),
    "C": (5.0, 2.0),
    "D": (4.5, 0.9),
}

#: Brunel-zeroed LIF extras: plain LIF with absolute reset, no PCL
#: adaptation tail (PORT_SPEC.md row 6); hard refractory taurp = 2 ms via
#: ``t_ref_ms`` (PORT_SPEC.md row 3).
_LIF_BRNEL = dict(
    tau_m_ms=TAU_MS,
    v_thresh_mV=THETA_MV,
    v_reset_mV=VRESET_MV,
    v_min_mV=-1000.0,
    eta_rp_mV=0.0,
    tau_rp_ms=5.0,
    tau_sra_ms=100.0,
    delta_sra_mV=0.0,
    t_ref_ms=2.0,
)


def _realized_counts(n):
    """Realized (NE, NI) cell counts at ``n`` neurons under the lane split.

    Mirrors ``jaxfne._config._counts_from_fractions`` for the lane's
    ``{E: 0.8, PV: 0.2}`` declaration (fractions sum to 1, E declared
    first, so ``NE = round(0.8*n)`` and NI takes the remainder); the lane
    needs the realized counts to declare exact CE/CI fan-in.
    """
    n = int(n)
    ne = int(round(E_FRACTION * n))
    return ne, n - ne


def external_rate_hz(point="D", n=1250):
    """Total external Poisson rate per cell: ``Cext * nuext`` (Hz)."""
    _, eta = FIG8_POINTS[str(point)]
    ne, _ = _realized_counts(n)
    cext = int(round(CONNECTIVITY_P * ne))
    return float(cext) * float(eta) * NUTHR_HZ


def build_spec(point="D", n=1250, seed=0):
    """The lane network built only from the public API (PORT_SPEC.md rows)."""
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    g, _ = FIG8_POINTS[str(point)]
    n = int(n)
    w = J_MV * math.sqrt(n)
    ne, ni = _realized_counts(n)
    ce = int(round(CONNECTIVITY_P * ne))
    ci = int(round(CONNECTIVITY_P * ni))
    lif = dict(_LIF_BRNEL)
    tensor = nt.NeuronalTensor(
        name="brunel2000-%s" % str(point),
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0",
            n_neurons=n,
            neuron_types=[
                nt.NeuronType.make("E", fraction=E_FRACTION, emitter="lif",
                                   emitter_params=dict(lif), drive=0.0),
                nt.NeuronType.make("PV", fraction=1.0 - E_FRACTION, emitter="lif",
                                   emitter_params=dict(lif), drive=0.0),
            ],
        )])],
        # Same-area AreaConnections: InterConnection has no probability
        # (all-to-all), AreaConnection carries the exact per-cell fan-in
        # (PORT_SPEC 8, 16): CE inputs per cell from E, CI from PV.
        area_connections=[
            nt.AreaConnection(
                source_area="net", source_layer="L0", source_neuron_type=src,
                target_area="net", target_layer="L0", target_neuron_type=tgt,
                mechanism=("AMPA" if src == "E" else "GABA_A"),
                static=nt.StaticParams(dT_ms=2.0),
                plastic=nt.PlasticParams(w_mech=(w if src == "E" else float(g) * w)),
                delay_ms=D_MS,
                sampling="fixed_indegree",
                in_degree=(ce if src == "E" else ci),
            )
            for src in ("E", "PV")
            for tgt in ("E", "PV")
        ],
    )
    return tensor


def build_sim(point="D", n=1250, T_s=0.5, dt_ms=DT_MS, seed=0):
    """The ``Simulation`` for a run (duration, dt, drive)."""
    _, eta = FIG8_POINTS[str(point)]
    nuext = float(eta) * NUTHR_HZ
    ne, _ = _realized_counts(n)
    cext = int(round(CONNECTIVITY_P * ne))
    prob = float(nuext) * float(dt_ms) / 1000.0
    if prob > 1.0:
        raise ValueError(
            "brunel2000 point %s at n=%d needs per-train rate %.0f Hz "
            "(p=%.2f per %.2g ms step): the current-route Poisson drive "
            "allows at most one event per train per step (PORT_SPEC.md row 12)"
            % (str(point), int(n), nuext, prob, float(dt_ms))
        )
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
        seed=int(seed),
        poisson_drive={"rate_hz": nuext, "amplitude": J_MV, "counts": cext,
                       "seed": int(seed)},
        runtime=J.RuntimeConfig(recurrent_backend="edge_list"),
    )


def run(seed, tag, out_dir, *, point="D", n=1250, T_s=0.5, burn_in_ms=0.0, dt_ms=DT_MS,
        freq_res_hz=None, signed=False):
    """Build, run and write the raster plus the ``manifest.json`` the scorer reads."""
    import time

    import numpy as np

    lane_kit.refuse_scored_dir(out_dir, signed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = "spikes_seed%d_%s.npz" % (int(seed), tag)
    sha = lane_kit.git_sha(_repo_root)
    prior = lane_kit.refuse_existing(out, fname, sha, signed=signed)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    t0 = time.perf_counter()
    model = J.construct(
        build_spec(point=point, n=n, seed=seed),
        nt.RuntimeConfiguration(seed=int(seed), duration_ms=float(T_s) * 1000.0,
                                dt_ms=float(dt_ms), emitter="lif"),
    )
    t_construct = time.perf_counter() - t0
    sim = build_sim(point=point, n=n, T_s=T_s, dt_ms=dt_ms, seed=seed)
    signals = model.simulate(sim)
    t_simulate = time.perf_counter() - t0 - t_construct
    spikes = np.asarray(signals.spikes, dtype=bool)
    assert spikes.shape == (sim.n_steps, sum(
        1 for _ in model.neuron_table())), "raster must cover every neuron"
    assert bool(np.isfinite(np.asarray(signals.V_m)).all()), "V must stay finite"
    regime = score(spikes, dt_ms, burn_in_ms=burn_in_ms, freq_res_hz=freq_res_hz)
    import jax

    backend = jax.default_backend()
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname, "point": str(point),
             "n": int(n), "steps": int(spikes.shape[0]), "dt_ms": float(dt_ms),
             "duration_ms": float(T_s) * 1000.0, "dtype": "float32",
             "backend": backend, "burn_in_ms": float(burn_in_ms), "freq_res_hz": freq_res_hz,
             "poisson_drive": {"rate_hz": float(sim.poisson_drive["rate_hz"]),
                               "amplitude": J_MV,
                               "counts": int(sim.poisson_drive["counts"]),
                               "seed": int(seed)},
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": backend,
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps, "regime": regime,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def score(spikes, dt_ms=DT_MS, burn_in_ms=0.0, freq_res_hz=None):
    """Score a raster after ``burn_in_ms`` with ``jaxfne.analysis`` estimators (checklist E3).

    Labels use the classifier's default Welch grid. ``freq_res_hz`` reads
    ``spectral_peak_hz`` from a second Welch grid with that bin width (the
    global frequency of Fig. 8); None keeps the default grid.
    """
    from jaxfne.analysis import activity_regime as _activity_regime
    from jaxfne.analysis import wang_buzsaki_kappa as _kappa

    import numpy as np

    if not float(burn_in_ms) >= 0.0:
        raise ValueError(f"burn_in_ms must be >= 0, got {burn_in_ms!r}")
    arr = np.asarray(spikes, dtype=bool)[int(round(float(burn_in_ms) / float(dt_ms))):]
    assert arr.shape[0] > 0, "burn_in_ms must leave steps to score"
    regime = _activity_regime(arr, float(dt_ms))
    peak_hz = regime["spectral_peak_hz"]
    if freq_res_hz is not None:
        if not float(freq_res_hz) > 0.0:
            raise ValueError(f"freq_res_hz must be > 0, got {freq_res_hz!r}")
        nperseg = int(round(1000.0 / (float(dt_ms) * float(freq_res_hz))))
        peak_hz = _activity_regime(
            arr, float(dt_ms), thresholds={"welch_nperseg": nperseg})["spectral_peak_hz"]
    return {"rate_hz": float(regime["rate_hz"]),
            "cv_isi": float(regime["cv_isi"]),
            "kappa": float(_kappa(arr, float(dt_ms))),
            "active_fraction": float(regime["active_fraction"]),
            "spectral_peak_excess": float(regime["spectral_peak_excess"]),
            "spectral_peak_hz": float(peak_hz),
            "label": str(regime["label"])}
