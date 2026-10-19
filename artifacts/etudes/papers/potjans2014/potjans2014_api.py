"""potjans2014 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``NeuronalTensor + RuntimeConfiguration(emitter="lif") -> construct ->
Model -> simulate -> Signals`` (checklist E1). Estimators come from
``jaxfne.analysis`` (checklist E3). No scored run: outputs land in the
caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype, seeds and the activity-regime block (checklist P4).

Port decisions live in ``PORT_SPEC.md``; stand-ins are declared deviations
there (density wiring, constant weights/delays, uniform drive, mV scale).
"""

from __future__ import annotations

import math
import sys
import time
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

# Paper constants (SOURCE_P1.md section 2). Voltages are shifted +65 mV
# from the paper origin (PORT_SPEC D-VSHIFT): the lif kernel leaks toward
# 0 mV, so the origin sits at V_reset; every stated gap is exact
# (theta - V_reset = 15 mV, tau_m = 10 ms, tau_ref = 2 ms).
TAU_M_MS = 10.0
THETA_MV = 15.0  # paper theta = -50 mV
VRESET_MV = 0.0  # paper V_reset = -65 mV
TREF_MS = 2.0
DT_MS = 0.5
W_E_MV = 0.1  # stand-in mV scale (PORT_SPEC D-SCALE); paper w = 87.8 pA
G_REL = 4.0  # paper g (SOURCE_P1.md section 2)
D_E_MS = 1.5  # paper excitatory mean delay
D_I_MS = 1.0  # stand-in for paper 0.8 ms (PORT_SPEC D-GRID)
NU_BG_HZ = 8.0  # paper background rate per train
DRIVE_COUNTS = 2000  # stand-in uniform train count (PORT_SPEC D-EXT)
DRIVE_AMPLITUDE_MV = 0.1
DT_INERT_MS = 2.0  # P-023-required dT, inert on the lif route

#: Cortical population sizes, paper Table 5 (SOURCE_P1.md section 2).
POP_N = {
    ("L23", "E"): 20683,
    ("L23", "PV"): 5834,
    ("L4", "E"): 21915,
    ("L4", "PV"): 5479,
    ("L5", "E"): 4850,
    ("L5", "PV"): 1065,
    ("L6", "E"): 14395,
    ("L6", "PV"): 2948,
}
LAYERS = ("L23", "L4", "L5", "L6")

#: Connection probabilities, paper Table 5 connectivity block
#: ((to_layer, to_type), (from_layer, from_type)) -> p (SOURCE_P1.md section 2).
#: Zero entries are absent (no rule); the Th column is folded into the
#: uniform drive (PORT_SPEC D-TH).
PROB = {
    (("L23", "E"), ("L23", "E")): 0.101,
    (("L23", "E"), ("L23", "PV")): 0.169,
    (("L23", "E"), ("L4", "E")): 0.044,
    (("L23", "E"), ("L4", "PV")): 0.082,
    (("L23", "E"), ("L5", "E")): 0.032,
    (("L23", "E"), ("L6", "E")): 0.008,
    (("L23", "PV"), ("L23", "E")): 0.135,
    (("L23", "PV"), ("L23", "PV")): 0.137,
    (("L23", "PV"), ("L4", "E")): 0.032,
    (("L23", "PV"), ("L4", "PV")): 0.052,
    (("L23", "PV"), ("L5", "E")): 0.075,
    (("L23", "PV"), ("L6", "E")): 0.004,
    (("L4", "E"), ("L23", "E")): 0.008,
    (("L4", "E"), ("L23", "PV")): 0.006,
    (("L4", "E"), ("L4", "E")): 0.050,
    (("L4", "E"), ("L4", "PV")): 0.135,
    (("L4", "E"), ("L5", "E")): 0.007,
    (("L4", "E"), ("L5", "PV")): 0.0003,
    (("L4", "E"), ("L6", "E")): 0.045,
    (("L4", "PV"), ("L23", "E")): 0.069,
    (("L4", "PV"), ("L23", "PV")): 0.003,
    (("L4", "PV"), ("L4", "E")): 0.079,
    (("L4", "PV"), ("L4", "PV")): 0.160,
    (("L4", "PV"), ("L5", "E")): 0.003,
    (("L4", "PV"), ("L6", "E")): 0.106,
    (("L5", "E"), ("L23", "E")): 0.100,
    (("L5", "E"), ("L23", "PV")): 0.062,
    (("L5", "E"), ("L4", "E")): 0.051,
    (("L5", "E"), ("L4", "PV")): 0.006,
    (("L5", "E"), ("L5", "E")): 0.083,
    (("L5", "E"), ("L5", "PV")): 0.373,
    (("L5", "E"), ("L6", "E")): 0.020,
    (("L5", "PV"), ("L23", "E")): 0.055,
    (("L5", "PV"), ("L23", "PV")): 0.027,
    (("L5", "PV"), ("L4", "E")): 0.026,
    (("L5", "PV"), ("L4", "PV")): 0.002,
    (("L5", "PV"), ("L5", "E")): 0.060,
    (("L5", "PV"), ("L5", "PV")): 0.316,
    (("L5", "PV"), ("L6", "E")): 0.009,
    (("L6", "E"), ("L23", "E")): 0.016,
    (("L6", "E"), ("L23", "PV")): 0.007,
    (("L6", "E"), ("L4", "E")): 0.021,
    (("L6", "E"), ("L4", "PV")): 0.017,
    (("L6", "E"), ("L5", "E")): 0.057,
    (("L6", "E"), ("L5", "PV")): 0.020,
    (("L6", "E"), ("L6", "E")): 0.040,
    (("L6", "E"), ("L6", "PV")): 0.225,
    (("L6", "PV"), ("L23", "E")): 0.036,
    (("L6", "PV"), ("L23", "PV")): 0.001,
    (("L6", "PV"), ("L4", "E")): 0.003,
    (("L6", "PV"), ("L4", "PV")): 0.001,
    (("L6", "PV"), ("L5", "E")): 0.028,
    (("L6", "PV"), ("L5", "PV")): 0.008,
    (("L6", "PV"), ("L6", "E")): 0.066,
    (("L6", "PV"), ("L6", "PV")): 0.144,
}

#: Potjans-zeroed LIF extras: plain LIF with absolute reset, no PCL
#: adaptation tail (PORT_SPEC row 6); hard refractory tau_ref = 2 ms via
#: ``t_ref_ms`` (PORT_SPEC row 3).
_LIF_POTJANS = dict(
    tau_m_ms=TAU_M_MS,
    v_thresh_mV=THETA_MV,
    v_reset_mV=VRESET_MV,
    v_min_mV=-1000.0,
    eta_rp_mV=0.0,
    tau_rp_ms=5.0,
    tau_sra_ms=100.0,
    delta_sra_mV=0.0,
    t_ref_ms=TREF_MS,
)


def _layer_sizes(n):
    """Layer neuron counts at ``n`` cells, paper Table 5 fractions."""
    total = float(sum(POP_N.values()))
    sizes = {}
    for layer in LAYERS:
        frac = float(POP_N[(layer, "E")] + POP_N[(layer, "PV")]) / total
        sizes[layer] = int(round(frac * int(n)))
    sizes[LAYERS[-1]] += int(n) - sum(sizes.values())
    return sizes


def _pop_counts(n):
    """Realized (layer, type) cell counts at ``n`` neurons."""
    sizes = _layer_sizes(n)
    counts = {}
    for layer in LAYERS:
        e_frac = float(POP_N[(layer, "E")]) / float(
            POP_N[(layer, "E")] + POP_N[(layer, "PV")]
        )
        ne = int(round(e_frac * sizes[layer]))
        counts[(layer, "E")] = ne
        counts[(layer, "PV")] = sizes[layer] - ne
    return counts


def _edge_magnitude_mv(src_layer, src_type, tgt_layer, tgt_type):
    """Stand-in edge magnitude in mV (PORT_SPEC D-SCALE, D-L4)."""
    mag = W_E_MV * G_REL if src_type == "PV" else W_E_MV
    if (src_layer, src_type) == ("L4", "E") and (tgt_layer, tgt_type) == ("L23", "E"):
        mag = 2.0 * W_E_MV
    return float(mag)


def build_spec(n=800, seed=0):
    """The lane network built only from the public API (PORT_SPEC.md rows)."""
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    n = int(n)
    total = float(n)
    layers = []
    for layer in LAYERS:
        e_frac = float(POP_N[(layer, "E")]) / float(
            POP_N[(layer, "E")] + POP_N[(layer, "PV")]
        )
        layers.append(
            nt.Layer(
                name=layer,
                n_neurons=int(_layer_sizes(n)[layer]),
                neuron_types=[
                    nt.NeuronType.make("E", fraction=e_frac, emitter="lif",
                                       emitter_params=dict(_LIF_POTJANS),
                                       drive=0.0),
                    nt.NeuronType.make("PV", fraction=1.0 - e_frac,
                                       emitter="lif",
                                       emitter_params=dict(_LIF_POTJANS),
                                       drive=0.0),
                ],
            )
        )
    area_conns = []
    for (tgt, src), p in sorted(PROB.items()):
        (tgt_layer, tgt_type), (src_layer, src_type) = tgt, src
        mag = _edge_magnitude_mv(src_layer, src_type, tgt_layer, tgt_type)
        area_conns.append(
            nt.AreaConnection(
                source_area="mc", source_layer=src_layer,
                source_neuron_type=src_type,
                target_area="mc", target_layer=tgt_layer,
                target_neuron_type=tgt_type,
                mechanism=("AMPA" if src_type == "E" else "GABA_A"),
                static=nt.StaticParams(dT_ms=DT_INERT_MS),
                plastic=nt.PlasticParams(w_mech=mag * math.sqrt(total)),
                delay_ms=(D_E_MS if src_type == "E" else D_I_MS),
                sampling="density",
                probability=float(p),
            )
        )
    return nt.NeuronalTensor(
        name="potjans2014",
        areas=[nt.Area(name="mc", layers=layers)],
        area_connections=area_conns,
    )


def build_sim(T_s=1.0, dt_ms=DT_MS, seed=0):
    """The ``Simulation`` for a run (duration, dt, uniform drive)."""
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
        seed=int(seed),
        poisson_drive={"rate_hz": NU_BG_HZ, "amplitude": DRIVE_AMPLITUDE_MV,
                       "counts": DRIVE_COUNTS, "seed": int(seed)},
        runtime=J.RuntimeConfig(recurrent_backend="edge_list"),
    )


def _pop_index(model):
    """Per-neuron population ids ``(layer, cell_type)`` in raster order."""
    return [(str(r["layer"]), str(r["cell_type"])) for r in model.neuron_table()]


def score(spikes, dt_ms=DT_MS):
    """Score a raster with ``jaxfne.analysis`` estimators (checklist E3)."""
    from jaxfne.analysis import activity_regime as _activity_regime
    from jaxfne.analysis import wang_buzsaki_kappa as _kappa

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    regime = _activity_regime(arr, float(dt_ms))
    return {"rate_hz": float(regime["rate_hz"]),
            "cv_isi": float(regime["cv_isi"]),
            "kappa": float(_kappa(arr, float(dt_ms))),
            "active_fraction": float(regime["active_fraction"]),
            "spectral_peak_excess": float(regime["spectral_peak_excess"]),
            "label": str(regime["label"])}


def score_by_population(spikes, pop_ids, dt_ms=DT_MS):
    """Per-population rates/CV/kappa (C1/C4/C5 readout shape)."""
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    out = {}
    for pop in sorted(set(pop_ids)):
        cols = np.array([i for i, q in enumerate(pop_ids) if q == pop])
        sub = arr[:, cols]
        reg = score(sub, dt_ms)
        reg["n"] = int(len(cols))
        out["%s%s" % pop] = reg
    return out


def run(seed, tag, out_dir, *, n=800, T_s=1.0, signed=False):
    """Build, run and write the raster plus the ``manifest.json`` the scorer reads."""
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
        build_spec(n=n, seed=seed),
        nt.RuntimeConfiguration(seed=int(seed),
                                duration_ms=float(T_s) * 1000.0,
                                dt_ms=DT_MS, emitter="lif"),
    )
    t_construct = time.perf_counter() - t0
    sim = build_sim(T_s=T_s, seed=seed)
    signals = model.simulate(sim)
    t_simulate = time.perf_counter() - t0 - t_construct
    spikes = np.asarray(signals.spikes, dtype=bool)
    assert spikes.shape == (sim.n_steps, sum(
        1 for _ in model.neuron_table())), "raster must cover every neuron"
    assert bool(np.isfinite(np.asarray(signals.V_m)).all()), "V must stay finite"
    pops = _pop_index(model)
    by_pop = score_by_population(spikes, pops)
    regime = score(spikes)
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "n": int(n), "steps": int(spikes.shape[0]), "dt_ms": DT_MS,
             "duration_ms": float(T_s) * 1000.0, "dtype": "float32",
             "backend": "cpu",
             "poisson_drive": {"rate_hz": NU_BG_HZ,
                               "amplitude": DRIVE_AMPLITUDE_MV,
                               "counts": DRIVE_COUNTS, "seed": int(seed)},
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime, "pops": by_pop}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps, "regime": regime,
            "pops": by_pop,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}
