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

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jaxfne as J
from jaxfne import neuronal_tensor as nt
from jaxfne.istdp_vogels2011 import ETA_DEFAULT as ISTDP_ETA_DEFAULT
from jaxfne.istdp_vogels2011 import RHO0_HZ_DEFAULT as ISTDP_RHO0_HZ_DEFAULT
from jaxfne.istdp_vogels2011 import TAU_STDP_MS_DEFAULT as ISTDP_TAU_MS_DEFAULT
from scripts import lane_kit

assert "site-packages" not in J.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

_repo_root = Path(__file__).resolve().parents[4]

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


def build_sim(T_s=1.0, dt_ms=DT_MS):
    """The ``Simulation`` for a run (duration, dt; the static adex route owns
    no RNG, so the lane seed rides construct-time wiring, never this sim)."""
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
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
    sim = build_sim(T_s=T_s)
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
                        rho0_hz=ISTDP_RHO0_HZ_DEFAULT, eta=ISTDP_ETA_DEFAULT,
                        plastic_segment_s=None):
    """Segment B: the registered iSTDP kernel from rest (PORT_SPEC rows 8-9).

    The adex ``Model.simulate`` path runs the static kernel only, so the
    plastic leg calls ``simulate_adex_hdp_registered`` directly on the
    compiled kernel arguments, with an I-to-E ``plasticity_mask`` (E-to-all
    and I-to-I edges frozen exactly). It starts from rest at the paper
    I-to-E init, like the authors' plasticity-onset leg.

    ``rho0_hz`` and ``eta`` ride the rule defaults (3 Hz, 1e-2); an
    unrealizable value (negative, non-finite) is refused by
    ``resolve_istdp_params`` before anything runs, never stored unused.

    ``plastic_segment_s`` runs the leg in segments of that length chained
    by the kernel state: each segment after the first resumes the prior
    segment's returned state, reproducing the one-segment run bit-for-bit
    while holding only one segment plus per-step E tallies in memory. The
    saved raster covers the last 1 s only (the R-E window); the receipt
    carries the per-second E rate course (``e_rate_per_s_hz``, E columns
    only) over each full second of the leg. ``None`` is one segment: the
    full raster is saved, exactly as before.
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
    e_cols = np.array([r["cell_type"] == "E"
                       for r in model.neuron_table()], dtype=bool)
    n_e = int(e_cols.sum())
    steps_per_s = int(round(1000.0 / DT_MS))
    rule_kwargs = dict(
        hdp_rule=ISTDP_RULE_NAME,
        hdp_rule_params={"eta": rule_cfg["eta"],
                         "rho0_hz": rule_cfg["rho0_hz"],
                         "tau_stdp_ms": rule_cfg["tau_stdp_ms"]},
        plasticity_mask=mask,
        seed=int(seed),
        # The per-step |w| trace is n_steps x n_edges (about 720 GB at n = 10000, 9 s); only w_final is read.
        record_w_trace=False,
    )
    if plastic_segment_s is None:
        result = simulate_adex_hdp_registered(
            **kw,
            n_steps=n_steps,
            dt_ms=DT_MS,
            **rule_kwargs,
        )
        t_simulate = time.perf_counter() - t0 - t_construct
        spikes = np.asarray(result["spikes"], dtype=bool)
        saved_start = 0
        e_per_step = spikes[:, e_cols].sum(axis=1).astype(np.int64)
        w_final = np.asarray(result["w_final_nS"], dtype=float)
        seg_s = None
    else:
        from collections import deque

        seg_s = float(plastic_segment_s)
        if not (math.isfinite(seg_s) and seg_s > 0.0):
            raise ValueError(
                "plastic_segment_s must be a finite duration in s > 0; "
                f"got {plastic_segment_s!r}"
            )
        seg_steps = max(1, int(round(seg_s * 1000.0 / DT_MS)))
        last_s = min(n_steps, steps_per_s)
        seg_state = None
        e_tallies = []
        keep = deque()
        kept = 0
        for start in range(0, n_steps, seg_steps):
            ln = min(seg_steps, n_steps - start)
            result = simulate_adex_hdp_registered(
                **kw,
                n_steps=ln,
                dt_ms=DT_MS,
                state=seg_state,
                return_state=True,
                **rule_kwargs,
            )
            seg_sp = np.asarray(result["spikes"], dtype=bool)
            e_tallies.append(seg_sp[:, e_cols].sum(axis=1).astype(np.int64))
            keep.append(seg_sp)
            kept += ln
            while kept - keep[0].shape[0] >= last_s:
                kept -= keep.popleft().shape[0]
            seg_state = result["state"]
        t_simulate = time.perf_counter() - t0 - t_construct
        tail = np.concatenate(list(keep), axis=0)
        spikes = tail[-last_s:]
        saved_start = n_steps - int(spikes.shape[0])
        e_per_step = np.concatenate(e_tallies, axis=0)
        w_final = np.asarray(result["w_final_nS"], dtype=float)
    assert spikes.shape == (n_steps - saved_start, sum(
        1 for _ in model.neuron_table())), "raster must cover every neuron"
    n_full_s = n_steps // steps_per_s
    e_rate_per_s_hz = ([float(e_per_step[i * steps_per_s:(i + 1) * steps_per_s].sum())
                        / float(n_e) for i in range(n_full_s)] if n_e else [])
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
                        "record_w_trace": False,
                        "plastic_segment_s": seg_s,
                        "spikes_window_steps": [saved_start, n_steps],
                        "e_rate_per_s_hz": list(e_rate_per_s_hz),
                        "w_ie_init_nS": float(w_init[mask].max(initial=0.0)) if int(mask.sum()) else 0.0,
                        "w_ie_final_mean_nS": float(w_final[mask].mean()) if int(mask.sum()) else 0.0,
                        "n_plastic": int(mask.sum())}}
    entries[fname] = entry
    manifest = {"git_sha": sha, "dtype": "float32", "backend": "cpu",
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": n_steps, "regime": regime,
            "e_cols": e_cols,
            "e_rate_per_s_hz": list(e_rate_per_s_hz),
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
          eta=ISTDP_ETA_DEFAULT, plastic_segment_s=None, signed=False):
    """B8 pilot: 1 s static then ``T_s`` plastic, with the prereg estimands.

    Unscored always (``signed=False``): R-E is the E mean rate over the last
    1 s of the plastic leg against y = 3 Hz (e = |z-y|/y, close at e <= 0.20)
    and O-DROP is the static-leg 0.8-1.0 s E rate over R-E's z (pass at
    ratio >= 2). Windows shorter than 1 s clamp to the leg with a note.
    ``plastic_segment_s`` chains the plastic leg in segments of that length
    (the saved plastic raster covers the R-E window only); R-E and O-DROP
    read the same spikes through the same estimator as the one-segment run.
    Writes ``pilot_receipt.json`` (estimands, "unscored pilot", code SHA).
    """
    import time

    import numpy as np

    t0 = time.perf_counter()
    a = run(int(seed), "segA", out_dir, n=int(n), T_s=STATIC_S,
            signed=signed)
    b = run_plastic_segment(int(seed), "segB", out_dir, n=int(n),
                            T_s=float(T_s), rho0_hz=float(rho0_hz),
                            eta=float(eta),
                            plastic_segment_s=plastic_segment_s,
                            signed=signed)
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
    saved_start = n_plastic - int(np.asarray(b["spikes"]).shape[0])
    z = _e_rate_hz(b["spikes"], e_cols, after_start - saved_start,
                   n_plastic - saved_start)
    e = abs(z - R_E_Y_HZ) / R_E_Y_HZ
    ratio = (before_hz / z) if z > 0 else (
        float("inf") if before_hz > 0 else float("nan"))
    receipt = {
        "verdict": "unscored pilot",
        "seed": int(seed), "n": int(n), "T_s": float(T_s),
        "static_s": STATIC_S, "dt_ms": DT_MS,
        "eta": float(b["eta"]), "rho0_hz": float(b["rho0_hz"]),
        "plastic_segment_s": None if plastic_segment_s is None else float(plastic_segment_s),
        "static_file": "spikes_seed%d_segA.npz" % int(seed),
        "plastic_file": "spikes_seed%d_segB.npz" % int(seed),
        "e_rate_per_s_hz": list(b["e_rate_per_s_hz"]),
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


#: P3 v4 (signed) scored protocol: seeds, scale and the prereg record.
SCORED_SEEDS = (4, 5, 6, 7, 8)
SCORED_N = 10000
SCORED_T_S = 150.0
SCORED_SEGMENT_S = 5.0
PREREG_SIGNED_REL = ("artifacts/etudes/papers/vogels2011/prereg/"
                     "P3_v4_signed.md")
CONV_SLOPE_HZ_S = 0.01
CONV_T_CRIT = 2.101  # t(0.975, 18): P3 v4 signed, 20 bins, n - 2 dof
CONV_WINDOW_BINS = 20


def prereg_signed_path():
    """Filesystem path of the signed P3 v4 prereg (read at run time)."""
    return Path(__file__).resolve().parent / "prereg" / "P3_v4_signed.md"


def prereg_sha256():
    """sha256 of the signed prereg's bytes, computed at run time."""
    import hashlib

    return hashlib.sha256(prereg_signed_path().read_bytes()).hexdigest()


def _t_crit_975(df):
    """Two-sided 0.975 t quantile at ``df`` (scipy; the v4 pin on failure)."""
    try:
        from scipy.stats import t as _t

        return float(_t.ppf(0.975, int(df)))
    except Exception:
        return CONV_T_CRIT


def conv_fit(rates, window=CONV_WINDOW_BINS):
    """Least-squares slope of the per-second E rate over its last bins.

    P3 v4 signed CONV (Change from v3): slope b over the last ``window``
    one-second bins (20 at T 150 s: 130-150 s) with its standard error from
    the residual sd on n - 2 degrees of freedom. The seed is "not converged"
    only when |b| >= 0.01 Hz/s and |b| / se > t(0.975, n - 2). Short series
    fit their last bins; the scorer refuses them, the small receipt notes it.
    """
    import numpy as np

    y = np.asarray(list(rates), dtype=float)
    if y.size < 3:
        raise ValueError(
            "conv_fit needs at least 3 one-second bins, got %d" % y.size)
    w = min(int(window), y.size)
    seg = y[y.size - w:]
    n = int(w)
    x = np.arange(n, dtype=float)
    xbar = float(x.mean())
    sxx = float(((x - xbar) ** 2).sum())
    b = float(((x - xbar) * (seg - float(seg.mean()))).sum() / sxx)
    resid = seg - (float(seg.mean()) + b * (x - xbar))
    sse = float((resid ** 2).sum())
    se = math.sqrt(sse / (n - 2)) / math.sqrt(sxx)
    if se > 0.0:
        t = abs(b) / se
    else:
        t = float("inf") if b != 0.0 else 0.0
    crit = _t_crit_975(n - 2)
    converged = not (abs(b) >= CONV_SLOPE_HZ_S and t > crit)
    return {"b": b, "se": float(se), "t": float(t),
            "t_crit": float(crit), "converged": bool(converged),
            "n_bins": n, "window_s": [int(y.size - w), int(y.size)]}


def scored_run(seed, out_dir, *, n=SCORED_N, T_s=SCORED_T_S,
               plastic_segment_s=SCORED_SEGMENT_S,
               rho0_hz=ISTDP_RHO0_HZ_DEFAULT, eta=ISTDP_ETA_DEFAULT,
               signed=False):
    """One signed scored seed under P3 v4 signed (``score --seed S --signed``).

    Refuses seeds outside 4-8 and unsigned calls before any compute. Runs the
    pilot protocol with ``signed=True`` at the signed parameters (n 10000,
    1 s static then T 150 s plastic, 5 s plastic segments, default rho0/eta;
    small n/T_s only in tests) and writes ``score_receipt.json``: the pilot
    fields plus ``status: "signed"``, the prereg path and its run-time
    sha256, the CONV fit (clamped with a note when T_s < 150 s) and the AI
    regime of the saved plastic window.
    """
    import numpy as np

    if not signed:
        raise SystemExit(
            "error: scored run needs --signed (P3 v4 signed only)")
    if int(seed) not in SCORED_SEEDS:
        raise SystemExit(
            "error: scored seeds are 4-8, got %r" % (seed,))
    receipt = pilot(int(seed), int(n), float(T_s), out_dir,
                    rho0_hz=float(rho0_hz), eta=float(eta),
                    plastic_segment_s=plastic_segment_s, signed=True)
    rates = list(receipt["e_rate_per_s_hz"])
    fit = conv_fit(rates)
    if len(rates) < SCORED_T_S:
        window_note = (
            "CONV window clamped to the last %d one-second bins "
            "(T_s %.1f s < 150 s; the signed window is 130-150 s)"
            % (fit["n_bins"], float(T_s)))
    else:
        window_note = None
    spikes = np.load(
        str(Path(out_dir) / receipt["plastic_file"]))["spikes"]
    scored = dict(receipt)
    scored.update({
        "verdict": "signed scored seed",
        "status": "signed",
        "prereg": PREREG_SIGNED_REL,
        "prereg_sha256": prereg_sha256(),
        "CONV": {"b_hz_s": fit["b"], "se_hz_s": fit["se"], "t": fit["t"],
                 "t_crit": fit["t_crit"], "converged": fit["converged"],
                 "window_s": fit["window_s"], "window_note": window_note,
                 "criterion": ("not converged only when |b| >= 0.01 Hz/s "
                               "and |b|/se > 2.101")},
        "AI": score(spikes, DT_MS),
    })
    sha = lane_kit.git_sha(_repo_root)
    lane_kit.write_manifest(out_dir, "score_receipt.json", scored, sha=sha)
    return scored


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
    pp.add_argument("--plastic-segment-s", type=float, default=None)
    ps = sub.add_parser("score", help="run one signed scored seed (P3 v4)")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--signed", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "score":
        receipt = scored_run(args.seed, args.out, signed=args.signed)
        print("R-E z=%.3f Hz e=%.3f | O-DROP ratio=%.3f | "
              "CONV b=%+.4f (se %.4f) converged=%s | %s"
              % (receipt["R-E"]["z_hz"], receipt["R-E"]["e"],
                 receipt["O-DROP"]["ratio"], receipt["CONV"]["b_hz_s"],
                 receipt["CONV"]["se_hz_s"], receipt["CONV"]["converged"],
                 receipt["status"]))
        return 0
    if args.cmd == "pilot":
        receipt = pilot(args.seed, args.n, getattr(args, "T_s"), args.out,
                        rho0_hz=args.rho0_hz, eta=args.eta,
                        plastic_segment_s=args.plastic_segment_s)
        print("R-E z=%.3f Hz e=%.3f | O-DROP ratio=%.3f | %s"
              % (receipt["R-E"]["z_hz"], receipt["R-E"]["e"],
                 receipt["O-DROP"]["ratio"], receipt["verdict"]))
        return 0
    raise SystemExit("unknown command %r" % (args.cmd,))


if __name__ == "__main__":
    raise SystemExit(main())
