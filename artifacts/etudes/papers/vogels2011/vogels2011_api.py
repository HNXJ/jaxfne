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
from jaxfne.istdp_vogels2011 import ETA_DEFAULT as ISTDP_ETA_DEFAULT
from jaxfne.istdp_vogels2011 import RHO0_HZ_DEFAULT as ISTDP_RHO0_HZ_DEFAULT
from jaxfne.istdp_vogels2011 import TAU_STDP_MS_DEFAULT as ISTDP_TAU_MS_DEFAULT

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
    e_cols = np.array([r["cell_type"] == "E"
                       for r in model.neuron_table()], dtype=bool)
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
            "e_cols": e_cols,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def run_plastic_segment(seed, tag, out_dir, *, n=200, T_s=1.0, signed=False,
                        rho0_hz=ISTDP_RHO0_HZ_DEFAULT, eta=ISTDP_ETA_DEFAULT):
    """Segment B: the registered iSTDP kernel from rest (PORT_SPEC rows 8-9).

    The adex ``Model.simulate`` path runs the static kernel only, so the
    plastic leg calls ``simulate_adex_hdp_registered`` directly on the
    compiled kernel arguments, with an I-to-E ``plasticity_mask`` (E-to-all
    and I-to-I edges frozen exactly). It starts from rest at the paper
    I-to-E init, like the authors' plasticity-onset leg.

    ``rho0_hz`` and ``eta`` ride the rule defaults (3 Hz, 1e-2); an
    unrealizable value (negative, non-finite) is refused by
    ``resolve_istdp_params`` before anything runs, never stored unused.
    """
    import time

    import numpy as np

    from jaxfne._construct_adex import adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex_hdp_registered
    from jaxfne.istdp_vogels2011 import (
        ISTDP_RULE_NAME,
        istdp_plasticity_mask,
        resolve_istdp_params,
    )

    # Fail closed before any work: unrealizable values raise here, and the
    # resolved triple is exactly what the kernel receives (defaults when
    # the caller passes none, bit-identical to the unparameterized call).
    rule_cfg = resolve_istdp_params({"eta": eta, "rho0_hz": rho0_hz,
                                     "tau_stdp_ms": ISTDP_TAU_MS_DEFAULT})

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
        hdp_rule_params={"eta": rule_cfg["eta"],
                         "rho0_hz": rule_cfg["rho0_hz"],
                         "tau_stdp_ms": rule_cfg["tau_stdp_ms"]},
        plasticity_mask=mask,
        seed=int(seed),
    )
    t_simulate = time.perf_counter() - t0 - t_construct
    spikes = np.asarray(result["spikes"], dtype=bool)
    assert spikes.shape == (n_steps, sum(
        1 for _ in model.neuron_table())), "raster must cover every neuron"
    e_cols = np.array([r["cell_type"] == "E"
                       for r in model.neuron_table()], dtype=bool)
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
                        "eta": rule_cfg["eta"],
                        "rho0_hz": rule_cfg["rho0_hz"],
                        "tau_stdp_ms": rule_cfg["tau_stdp_ms"],
                        "alpha": rule_cfg["alpha"],
                        "w_ie_init_nS": float(w_init[mask].max(initial=0.0)) if int(mask.sum()) else 0.0,
                        "w_ie_final_mean_nS": float(w_final[mask].mean()) if int(mask.sum()) else 0.0,
                        "n_plastic": int(mask.sum())}}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": n_steps, "regime": regime,
            "e_cols": e_cols,
            "w_ie_final_mean_nS": entry["plastic"]["w_ie_final_mean_nS"],
            "eta": rule_cfg["eta"], "rho0_hz": rule_cfg["rho0_hz"],
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


#: Prereg P3 v2 estimand constants.
R_E_Y_HZ = 3.0
R_E_BAND = 0.20
O_DROP_RATIO = 2.0
STATIC_S = 1.0


def _e_rate_hz(spikes, e_cols, start_step, end_step):
    """E-cell mean rate over a step window via the prereg estimator."""
    from jaxfne.analysis import activity_regime as _activity_regime

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)[:, np.asarray(e_cols, dtype=bool)]
    regime = _activity_regime(arr, DT_MS,
                              kept_window=(int(start_step), int(end_step)))
    return float(regime["rate_hz"])


def pilot(seed, n, T_s, out_dir, *, rho0_hz=ISTDP_RHO0_HZ_DEFAULT,
          eta=ISTDP_ETA_DEFAULT):
    """B8 pilot: 1 s static then ``T_s`` plastic, with the prereg estimands.

    Unscored always (``signed=False``): R-E is the E mean rate over the last
    1 s of the plastic leg against y = 3 Hz (e = |z-y|/y, close at e <= 0.20)
    and O-DROP is the static-leg 0.8-1.0 s E rate over R-E's z (pass at
    ratio >= 2). Windows shorter than 1 s clamp to the leg with a note.
    Writes ``pilot_receipt.json`` (estimands, "unscored pilot", code SHA).
    """
    import time

    import numpy as np

    t0 = time.perf_counter()
    a = run(int(seed), "segA", out_dir, n=int(n), T_s=STATIC_S)
    b = run_plastic_segment(int(seed), "segB", out_dir, n=int(n),
                            T_s=float(T_s), rho0_hz=float(rho0_hz),
                            eta=float(eta))
    assert bool((np.asarray(a["e_cols"]) == np.asarray(b["e_cols"])).all()), \
        "static and plastic legs must share E columns"
    e_cols = a["e_cols"]
    # Static "before" window: 0.8-1.0 s of the 1 s static leg.
    full_static = int(round(STATIC_S * 1000.0 / DT_MS))
    before_end = min(int(a["steps"]), full_static)
    before_start = max(0, before_end - int(round(0.2 * 1000.0 / DT_MS)))
    before_hz = _e_rate_hz(a["spikes"], e_cols, before_start, before_end)
    n_plastic = int(b["steps"])
    last_s = int(round(1.0 * 1000.0 / DT_MS))
    after_start = max(0, n_plastic - last_s)
    z = _e_rate_hz(b["spikes"], e_cols, after_start, n_plastic)
    e = abs(z - R_E_Y_HZ) / R_E_Y_HZ
    ratio = (before_hz / z) if z > 0 else (
        float("inf") if before_hz > 0 else float("nan"))
    receipt = {
        "verdict": "unscored pilot",
        "seed": int(seed), "n": int(n), "T_s": float(T_s),
        "static_s": STATIC_S, "dt_ms": DT_MS,
        "eta": float(b["eta"]), "rho0_hz": float(b["rho0_hz"]),
        "static_file": "spikes_seed%d_segA.npz" % int(seed),
        "plastic_file": "spikes_seed%d_segB.npz" % int(seed),
        "R-E": {"y_hz": R_E_Y_HZ, "z_hz": z, "e": e,
                "criterion": "e <= %.2f" % R_E_BAND,
                "window_steps": [after_start, n_plastic],
                "window_clamped": bool(n_plastic < last_s)},
        "O-DROP": {"before_hz": before_hz, "z_hz": z, "ratio": ratio,
                   "criterion": "ratio >= %.1f" % O_DROP_RATIO,
                   "window_steps": [before_start, before_end]},
        "W-IE": {"w_ie_final_mean_nS": float(b["w_ie_final_mean_nS"])},
        "wall_s": {"pilot": time.perf_counter() - t0,
                   "static": a["wall_s"], "plastic": b["wall_s"]},
    }
    sha = lane_kit.git_sha(_repo_root)
    lane_kit.write_manifest(out_dir, "pilot_receipt.json", receipt, sha=sha)
    return receipt


def main(argv=None):
    """Command-line entry: ``pilot --seed S --n N --T-s T --out DIR``."""
    import argparse

    p = argparse.ArgumentParser(description="vogels2011 P3 v2 B8 pilot driver")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot", help="run the 1 s static + T_s plastic pilot")
    pp.add_argument("--seed", type=int, required=True)
    pp.add_argument("--n", type=int, required=True)
    pp.add_argument("--T-s", type=float, required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--rho0-hz", type=float, default=ISTDP_RHO0_HZ_DEFAULT)
    pp.add_argument("--eta", type=float, default=ISTDP_ETA_DEFAULT)
    args = p.parse_args(argv)
    if args.cmd == "pilot":
        receipt = pilot(args.seed, args.n, getattr(args, "T_s"), args.out,
                        rho0_hz=args.rho0_hz, eta=args.eta)
        print("R-E z=%.3f Hz e=%.3f | O-DROP ratio=%.3f | %s"
              % (receipt["R-E"]["z_hz"], receipt["R-E"]["e"],
                 receipt["O-DROP"]["ratio"], receipt["verdict"]))
        return 0
    raise SystemExit("unknown command %r" % (args.cmd,))


if __name__ == "__main__":
    raise SystemExit(main())
