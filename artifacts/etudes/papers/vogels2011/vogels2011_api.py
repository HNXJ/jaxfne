"""vogels2011 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it in two
segments (``PORT_SPEC.md``): segment A through ``NeuronalTensor +
RuntimeConfiguration(emitter="adex") -> construct -> Model -> simulate ->
Signals`` (checklist E1; static kernel, I-to-E held at its init), segment
B through ``simulate_adex_hdp_registered`` on the compiled kernel
arguments with the registered ``istdp_vogels2011`` rule (the adex
``Model.simulate`` path runs the static kernel only). The conductance
(adex) route is the only core path with reversal-potential synapses; the
LIF route carries reversals as metadata only. Estimators come from
``jaxfne.analysis`` (checklist E3). No scored run: outputs land in the
caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). Every run writes a manifest with code SHA,
backend, dtype, seeds and the activity-regime block (checklist P4).

Port decisions live in ``PORT_SPEC.md``; digits denote the recurrent network
of Vogels et al. (2011), Fig. 4 (parameters in ``SOURCE_P1.md`` section 2).
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

# Paper constants (SOURCE_P1.md section 2, isp_brian.py parameter block).
NE_FRAC = 0.8
EPSILON = 0.02
DT_MS = 0.1
TAU_AMPA_MS = 5.0
TAU_GABA_MS = 10.0
E_REV_MV = 0.0
I_REV_MV = -80.0
W_E_NS = 0.3
W_II_NS = 3.0
# PORT_SPEC.md row 8: I-to-E is plastic under the registered iSTDP rule,
# starting at the paper init; the static segment holds this value.
W_IE_NS = 1e-10
BG_CURRENT_PA = 200.0
DELAY_MS = 0.1
# iSTDP paper values (SOURCE_P1.md section 2): eta = 1e-2, tau = 20 ms,
# alpha = 2*rho0*tau with rho0 = 3 Hz, cap gmax = 100. All four ride the
# rule defaults in `jaxfne.istdp_vogels2011`.
ISTDP_RULE = "istdp_vogels2011"

#: Conductance-LIF cells: AdEx at the LIF limit (Delta_mV = 0.0) with
#: adaptation zeroed, leak and threshold at the paper values
#: (PORT_SPEC.md rows 1-4).
_LIF_LIKE = dict(
    EL_mV=-60.0,
    VT_mV=-50.0,
    Vreset_mV=-60.0,
    Delta_mV=0.0,
    a_nS=0.0,
    b_pA=0.0,
)


def _w_mech(w_ns, n):
    """Paper weight in nS as the declared gain (realized ``/sqrt(N)``)."""
    return float(w_ns) * math.sqrt(int(n))


def build_spec(n=20, seed=0):
    """The lane network built only from the public API (PORT_SPEC.md rows)."""
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    n = int(n)
    lif = dict(_LIF_LIKE)
    return nt.NeuronalTensor(
        name="vogels2011",
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0",
            n_neurons=n,
            neuron_types=[
                nt.NeuronType.make("E", fraction=NE_FRAC, emitter="adex",
                                   emitter_params=dict(lif), drive=BG_CURRENT_PA),
                nt.NeuronType.make("PV", fraction=1.0 - NE_FRAC, emitter="adex",
                                   emitter_params=dict(lif), drive=BG_CURRENT_PA),
            ],
        )])],
        # Same-area AreaConnections: InterConnection has no probability
        # (all-to-all); AreaConnection carries probability=epsilon
        # (PORT_SPEC 16). Receptors are user-named on the adex route; tau
        # and reversal ride StaticParams (PORT_SPEC 6).
        area_connections=[
            nt.AreaConnection(
                source_area="net", source_layer="L0", source_neuron_type=src,
                target_area="net", target_layer="L0", target_neuron_type=tgt,
                mechanism=("ampa" if src == "E" else "gaba"),
                static=nt.StaticParams(
                    dT_ms=(TAU_AMPA_MS if src == "E" else TAU_GABA_MS),
                    reversal_potentials_mV={
                        ("ampa" if src == "E" else "gaba"):
                        (E_REV_MV if src == "E" else I_REV_MV)},
                ),
                plastic=nt.PlasticParams(w_mech=_w_mech(
                    (W_E_NS if src == "E"
                     else (W_IE_NS if tgt == "E" else W_II_NS)), n)),
                delay_ms=DELAY_MS,
                probability=EPSILON,
            )
            for src in ("E", "PV")
            for tgt in ("E", "PV")
        ],
    )


def build_sim(T_s=1.0, dt_ms=DT_MS, seed=0):
    """The ``Simulation`` for a run (duration, dt; drive is per-triple)."""
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
        seed=int(seed),
    )


def run(seed, tag, out_dir, *, n=200, T_s=1.0, signed=False):
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
        nt.RuntimeConfiguration(seed=int(seed),
                                duration_ms=float(T_s) * 1000.0,
                                dt_ms=DT_MS, emitter="adex"),
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
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "n": int(n), "steps": int(spikes.shape[0]), "dt_ms": DT_MS,
             "duration_ms": float(T_s) * 1000.0, "dtype": "float32",
             "backend": "cpu",
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps, "regime": regime,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def run_plastic_segment(seed, tag, out_dir, *, n=200, T_s=1.0, signed=False):
    """Segment B: the registered iSTDP kernel from rest (PORT_SPEC rows 8-9).

    The adex ``Model.simulate`` path runs the static kernel only, so the
    plastic leg calls ``simulate_adex_hdp_registered`` directly on the
    compiled kernel arguments, with an I-to-E ``plasticity_mask`` (E-to-all
    and I-to-I edges frozen exactly). It starts from rest at the paper
    I-to-E init, like the authors' plasticity-onset leg.
    """
    import time

    import numpy as np

    from jaxfne._construct_adex import adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex_hdp_registered
    from jaxfne.istdp_vogels2011 import ISTDP_RULE_NAME, istdp_plasticity_mask

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
                                dt_ms=DT_MS, emitter="adex"),
    )
    t_construct = time.perf_counter() - t0
    kw = adex_kernel_args(model, DT_MS)
    rows = model.neuron_table()
    lab = np.array([r["cell_type"] for r in rows])
    pre = np.asarray(kw["edge_pre"])
    post = np.asarray(kw["edge_post"])
    mask = np.asarray(istdp_plasticity_mask(lab[pre] == "PV", lab[post] == "E"))
    w_init = np.abs(np.asarray(kw["edge_weight_nS"], dtype=float))
    n_steps = int(round(float(T_s) * 1000.0 / DT_MS))
    result = simulate_adex_hdp_registered(
        **kw,
        n_steps=n_steps,
        dt_ms=DT_MS,
        hdp_rule=ISTDP_RULE_NAME,
        plasticity_mask=mask,
        seed=int(seed),
    )
    t_simulate = time.perf_counter() - t0 - t_construct
    spikes = np.asarray(result["spikes"], dtype=bool)
    assert spikes.shape == (n_steps, sum(
        1 for _ in model.neuron_table())), "raster must cover every neuron"
    w_final = np.asarray(result["w_final_nS"], dtype=float)
    assert bool(np.all(w_final[~mask] == w_init[~mask])), \
        "frozen edges must stay bit-identical"
    regime = score(spikes, DT_MS)
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "n": int(n), "steps": int(spikes.shape[0]), "dt_ms": DT_MS,
             "duration_ms": float(T_s) * 1000.0, "dtype": "float32",
             "backend": "cpu",
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime,
             "plastic": {"rule": ISTDP_RULE_NAME,
                        "w_ie_init_nS": float(w_init[mask].max(initial=0.0)) if int(mask.sum()) else 0.0,
                        "w_ie_final_mean_nS": float(w_final[mask].mean()) if int(mask.sum()) else 0.0,
                        "n_plastic": int(mask.sum())}}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": n_steps, "regime": regime,
            "w_ie_final_mean_nS": entry["plastic"]["w_ie_final_mean_nS"],
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def score(spikes, dt_ms=DT_MS):
    """Score a raster with ``jaxfne.analysis`` estimators (checklist E3)."""
    from jaxfne.analysis import activity_regime as _activity_regime

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    regime = _activity_regime(arr, float(dt_ms))
    return {"rate_hz": float(regime["rate_hz"]),
            "cv_isi": float(regime["cv_isi"]),
            "kappa": float(regime["kappa"]),
            "active_fraction": float(regime["active_fraction"]),
            "spectral_peak_excess": float(regime["spectral_peak_excess"]),
            "label": str(regime["label"])}
