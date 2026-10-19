"""mejias2016 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``NeuronalTensor + RuntimeConfiguration(emitter="lif") -> construct ->
Model -> simulate -> Signals`` (checklist E1). Estimators come from
``jaxfne.analysis`` (checklist E3). No scored run: outputs land in the
caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype and seeds (checklist P4).

Port decisions live in ``PORT_SPEC.md``. The paper's unit is a Wilson-Cowan
rate pair per laminar module (gap G1: no rate emitter in the core); ``run``
executes the declared spiking stand-in S1 (lif E-PV pairs per module) and
labels it as such. Digits denote Figs. 2-3 of Mejias et al. (2016).
"""

from __future__ import annotations

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

# Stand-in constants (PORT_SPEC.md rows S1, 8-10; paper values in SOURCE_P1.md).
DT_MS = 0.5
D_MS = 1.0
CONNECTIVITY_P = 0.5
POISSON_RATE_HZ = 400.0
POISSON_AMPLITUDE_MV = 2.0
DRIVE_E_MV = 0.0
DRIVE_I_MV = 0.0
#: FB infra:supra weight ratio, declared (PORT_SPEC.md row 6; value under G6).
FB_INFRA_SUPRA_RATIO = 2.0

#: Zeroed lif extras: plain lif with absolute reset, no PCL
#: refractory/adaptation tail (PORT_SPEC.md row S1).
_LIF_MEJIAS = dict(
    tau_m_ms=20.0,
    v_thresh_mV=20.0,
    v_reset_mV=10.0,
    v_min_mV=-1000.0,
    eta_rp_mV=0.0,
    tau_rp_ms=5.0,
    tau_sra_ms=100.0,
    delta_sra_mV=0.0,
)

_LAYERS = ("L23", "L56")


def _lif_types(drive_e=DRIVE_E_MV, drive_i=DRIVE_I_MV):
    lif = dict(_LIF_MEJIAS)
    return [
        nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                           emitter_params=dict(lif), drive=float(drive_e)),
        nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                           emitter_params=dict(lif), drive=float(drive_i)),
    ]


def _local_rules(area, w_mech=1.0):
    """Intra-module E/I rules plus the Fig. 3A interlaminar motif."""
    rules = []
    for lay in _LAYERS:
        for src in ("E", "PV"):
            for tgt in ("E", "PV"):
                rules.append(nt.AreaConnection(
                    source_area=area, source_layer=lay, source_neuron_type=src,
                    target_area=area, target_layer=lay, target_neuron_type=tgt,
                    mechanism=("AMPA" if src == "E" else "GABA_A"),
                    static=nt.StaticParams(dT_ms=2.0),
                    plastic=nt.PlasticParams(w_mech=float(w_mech)),
                    delay_ms=D_MS,
                    probability=CONNECTIVITY_P,
                ))
    # Fig. 3A motif: L23E to L5E; L5E to L2/3I (PORT_SPEC.md row 4).
    rules.append(nt.AreaConnection(
        source_area=area, source_layer="L23", source_neuron_type="E",
        target_area=area, target_layer="L56", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=float(w_mech)),
        delay_ms=D_MS,
        probability=CONNECTIVITY_P,
    ))
    rules.append(nt.AreaConnection(
        source_area=area, source_layer="L56", source_neuron_type="E",
        target_area=area, target_layer="L23", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=float(w_mech)),
        delay_ms=D_MS,
        probability=CONNECTIVITY_P,
    ))
    return rules


def build_spec(n=20, seed=0):
    """The lane network built only from the public API (PORT_SPEC.md rows).

    Single-area two-layer local module (Figs. 2-3): lif E-PV pairs per
    layer with the Fig. 3A interlaminar motif. The Wilson-Cowan rate pair
    is gap G1; this is the declared stand-in S1.
    """
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    n = int(n)
    tensor = nt.NeuronalTensor(
        name="mejias2016-local",
        areas=[nt.Area(name="V1", layers=[
            nt.Layer(name=lay, n_neurons=n, neuron_types=_lif_types())
            for lay in _LAYERS
        ])],
        area_connections=_local_rules("V1"),
    )
    return tensor


def build_two_area(n=20, seed=0):
    """Two areas with the interareal motifs (PORT_SPEC.md rows 5-6).

    FF: V1 L23E to V4 L23E. FB: V4 L56E to all four V1 populations,
    infra-weighted by FB_INFRA_SUPRA_RATIO (declared; value under G6).
    """
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    n = int(n)
    rules = _local_rules("V1") + _local_rules("V4")
    rules.append(nt.AreaConnection(
        source_area="V1", source_layer="L23", source_neuron_type="E",
        target_area="V4", target_layer="L23", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.0),
        delay_ms=D_MS,
        probability=CONNECTIVITY_P,
    ))
    for tgt_layer in _LAYERS:
        for tgt_type in ("E", "PV"):
            w = FB_INFRA_SUPRA_RATIO if tgt_layer == "L56" else 1.0
            rules.append(nt.AreaConnection(
                source_area="V4", source_layer="L56", source_neuron_type="E",
                target_area="V1", target_layer=tgt_layer,
                target_neuron_type=tgt_type,
                mechanism="AMPA",
                static=nt.StaticParams(dT_ms=2.0),
                plastic=nt.PlasticParams(w_mech=float(w)),
                delay_ms=D_MS,
                probability=CONNECTIVITY_P,
            ))
    tensor = nt.NeuronalTensor(
        name="mejias2016-pair",
        areas=[nt.Area(name=area, layers=[
            nt.Layer(name=lay, n_neurons=n, neuron_types=_lif_types())
            for lay in _LAYERS
        ]) for area in ("V1", "V4")],
        area_connections=rules,
    )
    return tensor


def build_sim(T_s=1.0, dt_ms=DT_MS, seed=0):
    """The ``Simulation`` for a run (duration, dt, noise drive)."""
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
        seed=int(seed),
        poisson_drive={"rate_hz": POISSON_RATE_HZ,
                       "amplitude": POISSON_AMPLITUDE_MV, "seed": int(seed)},
        runtime=J.RuntimeConfig(recurrent_backend="edge_list"),
    )


def run(seed, tag, out_dir, *, n=20, T_s=1.0, signed=False):
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
        build_spec(n=n, seed=seed),
        nt.RuntimeConfiguration(seed=int(seed), duration_ms=float(T_s) * 1000.0,
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
    regime = score(spikes, DT_MS)
    layer_rates = _layer_rates(spikes, model)
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "engine": "lif-stand-in-S1",
             "n": int(spikes.shape[1]), "steps": int(spikes.shape[0]),
             "dt_ms": DT_MS, "duration_ms": float(T_s) * 1000.0,
             "dtype": "float32", "backend": "cpu",
             "poisson_drive": {"rate_hz": POISSON_RATE_HZ,
                               "amplitude": POISSON_AMPLITUDE_MV,
                               "seed": int(seed)},
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime, "layer_rates_hz": layer_rates}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "engine": "lif-stand-in-S1 (PORT_SPEC.md; gap G1)",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps, "regime": regime,
            "layer_rates_hz": layer_rates,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def _layer_rates(spikes, model):
    """Mean per-neuron rate in Hz per layer (PORT_SPEC.md row S2)."""
    import numpy as np

    layers = np.array([str(r["layer"]) for r in model.neuron_table()])
    out = {}
    for lay in ("L23", "L56"):
        m = layers == lay
        out[str(lay)] = float(spikes[:, m].mean() / (DT_MS / 1000.0)) if m.any() else 0.0
    return out


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