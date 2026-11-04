"""potjans2014 replication, library-path runner.

Builds the network only from the jaxfne public API and runs it through
``NeuronalTensor + RuntimeConfiguration(emitter="lif") -> construct ->
Model -> simulate -> Signals`` (checklist E1). Estimators come from
``jaxfne.analysis`` (checklist E3). Pilot runs are unscored: outputs land in
the caller's directory, never under a ``scored`` path
(``lane_kit.refuse_scored_dir``). The ``score`` subcommand runs one signed
scored seed (4-8) at the frozen protocol under a ``scored`` path. Every run
writes a manifest with code SHA, backend, dtype, seeds and the
activity-regime block (checklist P4); signed runs also record the prereg path
and its sha256.

Port decisions live in ``PORT_SPEC.md``; declared deviations are there too
(unfiltered drive kick, sub-step delay floor, spread proportional to the mean).
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import jax

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
DT_MS = 0.1  # paper step h = 0.1 ms (SOURCE_P1.md section 1)
TAU_SYN_MS = 0.5  # paper tau_syn (SOURCE_P1.md section 2)
C_M_PF = 250.0  # paper C_m (SOURCE_P1.md section 2)
W_PA = 87.8  # paper excitatory w (SOURCE_P1.md section 2)
W_SD_PA = 8.8  # paper delta_w (SOURCE_P1.md section 2)
G_REL = -4.0  # paper g (SOURCE_P1.md section 2)
D_E_MS, D_E_SD_MS = 1.5, 0.75  # paper d_e +- delta_d_e (section 2)
D_I_MS, D_I_SD_MS = 0.8, 0.4  # paper d_i +- delta_d_i (section 2)
NU_BG_HZ = 8.0  # paper background rate per train (SOURCE_P1.md section 2)
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

#: Reference external in-degree k_ext per population, paper Table 5
#: (SOURCE_P1.md section 2). Each entry drives one population (row 12).
K_EXT = {
    ("L23", "E"): 1600,
    ("L23", "PV"): 1500,
    ("L4", "E"): 2100,
    ("L4", "PV"): 1900,
    ("L5", "E"): 2000,
    ("L5", "PV"): 1900,
    ("L6", "E"): 2900,
    ("L6", "PV"): 2100,
}

#: Connection probabilities, paper Table 5 connectivity block
#: ((to_layer, to_type), (from_layer, from_type)) -> p (SOURCE_P1.md section 2).
#: Zero entries are absent (no rule); the Th column is not wired (PORT_SPEC D-TH).
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

#: Population-name aliases for lesions: paper-style names (``L5e``, ``L5i``,
#: ``L2/3e``) map to the lane's ``(layer, cell_type)`` populations.
#: Inhibitory aliases end in ``i`` or ``pv``; excitatory aliases end in ``e``.
LESION_POP_ALIASES = {
    "l23e": ("L23", "E"), "l2/3e": ("L23", "E"),
    "l23i": ("L23", "PV"), "l2/3i": ("L23", "PV"),
    "l23pv": ("L23", "PV"), "l2/3pv": ("L23", "PV"),
    "l4e": ("L4", "E"),
    "l4i": ("L4", "PV"), "l4pv": ("L4", "PV"),
    "l5e": ("L5", "E"),
    "l5i": ("L5", "PV"), "l5pv": ("L5", "PV"),
    "l6e": ("L6", "E"),
    "l6i": ("L6", "PV"), "l6pv": ("L6", "PV"),
}

#: Paper Table 6 reference rates (Hz) for the P3_v2 primary rows R-*.
P3V2_REFERENCE_HZ = {
    "L23E": 0.86,
    "L4E": 4.45,
    "L5E": 7.59,
    "L6E": 1.09,
}
P3V2_CLOSE_E = 0.20

#: Signed protocol record: the frozen prereg (repo-relative) and the scored
#: protocol parameters (prereg Seeds and Scored/unscored split sections).
PREREG_P3V2_SIGNED = (
    "artifacts/etudes/papers/potjans2014/prereg/P3_v2_signed.md"
)
SCORED_SEEDS = (4, 5, 6, 7, 8)
SCORE_SCALE = 0.1
SCORE_T_S = 10.0
SCORE_SEGMENT_S = 1.0


def _prereg_sha256():
    """sha256 of the signed prereg file, computed at run time from the file."""
    return hashlib.sha256(
        (_repo_root / PREREG_P3V2_SIGNED).read_bytes()).hexdigest()


def _resolve_lesion_pop(name, *, role):
    """Map a lesion population name to a ``(layer, cell_type)`` pair."""
    if isinstance(name, (tuple, list)) and len(name) == 2:
        layer, ctype = str(name[0]), str(name[1])
        key = (layer, "PV" if ctype.upper() in ("PV", "I") else ctype.upper())
        if key in POP_N:
            return key
        raise ValueError("unknown lesion %s population %r" % (role, name))
    key = str(name).strip().lower().replace("_", "")
    if key in LESION_POP_ALIASES:
        return LESION_POP_ALIASES[key]
    raise ValueError("unknown lesion %s population %r" % (role, name))


def _normalize_lesion(lesion):
    """Normalize ``lesion`` to a list of ``((layer, type), (layer, type))`` pairs.

    One projection is a 2-tuple of names, e.g. ``("L5i", "L5e")``; several
    projections are a list of such pairs.
    """
    if lesion is None:
        return []
    if (isinstance(lesion, tuple) and len(lesion) == 2
            and all(isinstance(x, str) for x in lesion)):
        items = [lesion]
    elif (isinstance(lesion, tuple) and len(lesion) == 2
            and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in lesion)):
        items = [tuple(lesion)]
    else:
        items = list(lesion)
    pairs = []
    for item in items:
        pre_name, post_name = item
        pairs.append((_resolve_lesion_pop(pre_name, role="source"),
                      _resolve_lesion_pop(post_name, role="target")))
    return pairs
#: adaptation tail (PORT_SPEC row 6); hard refractory tau_ref = 2 ms via
#: ``t_ref_ms`` (PORT_SPEC row 3); first-order synaptic filter tau_syn = 0.5 ms
#: (PORT_SPEC row 11).
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
    tau_syn_ms=TAU_SYN_MS,
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


def _scale_counts(scale):
    """Cell counts at ``scale`` of every POP_N: half-up, at least one cell."""
    s = float(scale)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError("scale must be a finite number > 0; got %r" % (scale,))
    return {pop: max(1, int(math.floor(POP_N[pop] * s + 0.5))) for pop in POP_N}


def _deposit_mv_per_pa():
    """Total mV deposited by one arrival per pA of weight: tau_syn / C_m (row 14)."""
    return TAU_SYN_MS / C_M_PF


def _step_mv_per_pa():
    """Per-step edge weight per pA: (1/C_m) tau_syn (1 - exp(-dt/tau_syn)).

    The core's filter adds each arrival to ``g``, which decays by exp(-dt/tau)
    per step and enters V directly, so this increment deposits
    ``w tau_syn / C_m`` in total over the decay (row 14).
    """
    return TAU_SYN_MS * (1.0 - math.exp(-DT_MS / TAU_SYN_MS)) / C_M_PF


def _validate_segment_s(segment_s):
    """A segment length in seconds, or None for one call; refused by name."""
    if segment_s is None:
        return None
    try:
        s = float(segment_s)
    except (TypeError, ValueError):
        raise ValueError(
            "segment_s must be a finite number of seconds > 0; got %r"
            % (segment_s,))
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(
            "segment_s must be a finite number of seconds > 0; got %r"
            % (segment_s,))
    return s


def mean_in_degree(src, tgt):
    """Mean in-degree k of projection src -> tgt: K_full / N_post_full (row 8).

    K_full = ln(1 - C) / ln(1 - 1/(N_pre N_post)) draws with replacement give
    the paper's connection probability C at full scale; k is held at every scale.
    """
    c = PROB[(tgt, src)]
    n_pre, n_post = POP_N[src], POP_N[tgt]
    k_full = math.log1p(-c) / math.log1p(-1.0 / (n_pre * n_post))
    return k_full / n_post


def _edge_mean_mv(src, tgt):
    """Mean edge weight magnitude in mV per arrival (rows 9 and 14).

    Excitatory w = 87.8 pA; inhibitory |g| w; L4e -> L2/3e doubled. The sign
    comes from the mechanism.
    """
    pa = W_PA * (abs(G_REL) if src[1] == "PV" else 1.0)
    if src == ("L4", "E") and tgt == ("L23", "E"):
        pa *= 2.0
    return pa * _step_mv_per_pa()


def build_spec(n=800, seed=0, scale=None, lesion=None):
    """The lane network built only from the public API (PORT_SPEC.md rows).

    ``scale`` takes every population at that fraction of POP_N (half-up, at
    least one cell); without it the cell counts are the fractions of ``n``.
    ``lesion`` removes whole projections through the public API, e.g.
    ``lesion=("L5i", "L5e")`` drops every L5i -> L5e connection rule; unknown
    population names are refused. The default None wires every projection.
    """
    del seed  # wiring draws seed from RuntimeConfiguration at construct time
    lesions = _normalize_lesion(lesion)
    counts = _scale_counts(scale) if scale is not None else _pop_counts(int(n))
    total = float(sum(counts.values()))
    layers = []
    for layer in LAYERS:
        n_e, n_pv = counts[(layer, "E")], counts[(layer, "PV")]
        e_frac = float(n_e) / float(n_e + n_pv)
        layers.append(
            nt.Layer(
                name=layer,
                n_neurons=int(n_e + n_pv),
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
    for tgt, src in sorted(PROB):
        if (src, tgt) in lesions:
            continue
        (tgt_layer, tgt_type), (src_layer, src_type) = tgt, src
        mean_mv = _edge_mean_mv(src, tgt)
        area_conns.append(
            nt.AreaConnection(
                source_area="mc", source_layer=src_layer,
                source_neuron_type=src_type,
                target_area="mc", target_layer=tgt_layer,
                target_neuron_type=tgt_type,
                mechanism=("AMPA" if src_type == "E" else "GABA_A"),
                static=nt.StaticParams(dT_ms=DT_INERT_MS),
                # The core scales the rule mean by 1/sqrt(total_n) but takes
                # w_std raw, so the mean carries sqrt(total_n) and the spread
                # is passed in mV as realized. The spread scales with the mean
                # (delta_w / w on every rule): g w and 2 w are amplitude distributions.
                plastic=nt.PlasticParams(
                    w_mech=mean_mv * math.sqrt(total),
                    w_std=mean_mv * W_SD_PA / W_PA,
                ),
                delay_ms=(D_E_MS if src_type == "E" else D_I_MS),
                delay_std_ms=(D_E_SD_MS if src_type == "E" else D_I_SD_MS),
                sampling="multinomial",
                in_degree=mean_in_degree(src, tgt),
            )
        )
    return nt.NeuronalTensor(
        name="potjans2014",
        areas=[nt.Area(name="mc", layers=layers)],
        area_connections=area_conns,
    )


def build_drive(seed=0):
    """One ``poisson_drive`` entry per population (row 12).

    Each entry is k_ext trains of NU_BG_HZ onto its own (area, layer, cell_type)
    triple. The amplitude is the full deposit w tau_syn / C_m of one arrival;
    the core adds it unfiltered (declared deviation, row 12).
    """
    amp = W_PA * _deposit_mv_per_pa()
    return [
        {"rate_hz": NU_BG_HZ, "amplitude": amp, "counts": int(K_EXT[pop]),
         "seed": int(seed), "target": ("mc", pop[0], pop[1])}
        for pop in POP_N
    ]


def build_sim(T_s=1.0, dt_ms=DT_MS, seed=0):
    """The ``Simulation`` for a run (duration, dt, one drive entry per population)."""
    return J.Simulation(
        duration_ms=float(T_s) * 1000.0,
        dt_ms=float(dt_ms),
        seed=int(seed),
        poisson_drive=build_drive(seed),
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


def p3v2_estimands(spikes, pop_ids, dt_ms=DT_MS, t_s=None):
    """P3_v2 estimands from a raster, estimators from ``jaxfne.analysis`` only.

    Returns the primary rows R-* (relative error against Table 6), O-IE,
    O-ORD and CV, plus the secondary SYN row: ``fano_factor`` per population
    (up to 100 cells, 3 ms bins; the core estimator is the per-neuron mean
    ratio, ``metrics.fano_factor``). Rates and CV read the kept window (first
    0.5 s dropped when the run is longer than that); the window is recorded.
    SYN is reported, never scored.
    """
    from jaxfne.analysis import fano_factor as _fano_factor

    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    drop_steps = int(round(500.0 / float(dt_ms))) if t_s is None or t_s > 0.5 else 0
    drop_steps = min(drop_steps, arr.shape[0] - 1)
    kept = arr[drop_steps:]
    rates, cvs = {}, {}
    for pop in sorted(set(pop_ids)):
        cols = np.array([i for i, q in enumerate(pop_ids) if q == pop])
        sub = kept[:, cols]
        n_spikes = int(sub.sum())
        dur_s = float(sub.shape[0]) * float(dt_ms) / 1000.0
        rates["%s%s" % pop] = (n_spikes / (len(cols) * dur_s)) if dur_s > 0 else 0.0
        cvs["%s%s" % pop] = float(score(sub, dt_ms)["cv_isi"])
    rows = {}
    for key, y in sorted(P3V2_REFERENCE_HZ.items()):
        z = float(rates.get(key, 0.0))
        e = abs(z - y) / abs(y)
        rows["R-%s" % key] = {"y_hz": y, "z_hz": z, "e": e,
                              "pass": bool(e <= P3V2_CLOSE_E)}
    oi_e = {}
    for layer in LAYERS:
        e_rate = float(rates.get("%sE" % layer, 0.0))
        i_rate = float(rates.get("%sPV" % layer, 0.0))
        oi_e[layer] = {"e_hz": e_rate, "i_hz": i_rate,
                       "pass": bool(i_rate > e_rate)}
    ord_row = {"L5E_highest": bool(
        rates.get("L5E", 0.0) > max(rates.get(k, 0.0)
                                    for k in ("L23E", "L4E", "L6E"))),
               "L23E_below_L4E": bool(rates.get("L23E", 0.0) < rates.get("L4E", 0.0)),
               "L6E_below_L4E": bool(rates.get("L6E", 0.0) < rates.get("L4E", 0.0))}
    ord_row["pass"] = bool(all(ord_row.values()))
    cv_row = {"cv_isi": {k: cvs[k] for k in sorted(cvs)},
              "pass": bool(all(v > 0.8 for v in cvs.values()))}
    syn = {}
    for pop in sorted(set(pop_ids)):
        cols = [i for i, q in enumerate(pop_ids) if q == pop][:100]
        sub = np.asarray(kept[:, cols].T, dtype=float)
        syn["%s%s" % pop] = float(_fano_factor(sub, bin_size_ms=3.0, dt_ms=float(dt_ms)))
    return {"window_s": {"dropped_s": drop_steps * float(dt_ms) / 1000.0,
                         "kept_s": float(kept.shape[0]) * float(dt_ms) / 1000.0},
            "rates_hz": rates, "R": rows, "O-IE": oi_e,
            "O-ORD": ord_row, "CV": cv_row,
            "SYN_fano_3ms": syn,
            "SYN_note": "reported, never scored; core fano_factor is the "
                        "per-neuron mean ratio over up to 100 cells"}


def run(seed, tag, out_dir, *, n=800, scale=None, T_s=1.0, signed=False,
        lesion=None, pilot=False, segment_s=None):
    """Build, run and write the raster plus the ``manifest.json`` the scorer reads.

    ``segment_s`` chains ``simulate`` calls of that many seconds (the last one
    shorter when it does not divide the run): the concatenated raster scores
    exactly as one call, while only it grows. The manifest records
    ``segment_s``, the segment count and each segment's Poisson receipt. The
    default None is one call, unchanged.
    """
    import numpy as np

    seg_s = _validate_segment_s(segment_s)
    lane_kit.refuse_scored_dir(out_dir, signed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = "spikes_seed%d_%s.npz" % (int(seed), tag)
    sha = lane_kit.git_sha(_repo_root)
    prior = lane_kit.refuse_existing(out, fname, sha, signed=signed)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    lesions = _normalize_lesion(lesion)
    t0 = time.perf_counter()
    model = J.construct(
        build_spec(n=n, seed=seed, scale=scale, lesion=lesion),
        nt.RuntimeConfiguration(seed=int(seed),
                                duration_ms=float(T_s) * 1000.0,
                                dt_ms=DT_MS, emitter="lif"),
    )
    t_construct = time.perf_counter() - t0
    sim = build_sim(T_s=T_s, seed=seed)
    seg_receipts = None
    if seg_s is None:
        signals = model.simulate(sim)
        t_simulate = time.perf_counter() - t0 - t_construct
        spikes = np.asarray(signals.spikes, dtype=bool)
        assert spikes.shape == (sim.n_steps, sum(
            1 for _ in model.neuron_table())), "raster must cover every neuron"
        assert bool(np.isfinite(np.asarray(signals.V_m)).all()), "V must stay finite"
    else:
        seg_steps = int(round(seg_s * 1000.0 / DT_MS))
        if seg_steps <= 0:
            raise ValueError(
                "segment_s=%r holds no dt_ms step; pass at least %s s"
                % (segment_s, DT_MS / 1000.0))
        t_seg = time.perf_counter()
        parts, seg_receipts, state, remaining = [], [], None, sim.n_steps
        while remaining > 0:
            k = min(seg_steps, remaining)
            seg_sim = J.Simulation(
                duration_ms=float(k) * DT_MS, dt_ms=DT_MS, seed=int(seed),
                poisson_drive=build_drive(seed),
                runtime=J.RuntimeConfig(recurrent_backend="edge_list"))
            if state is None:
                seg_signals, state = model.simulate(seg_sim, return_state=True)
            else:
                seg_signals, state = model.simulate(
                    seg_sim, continuation=state, return_state=True)
            assert bool(np.isfinite(np.asarray(seg_signals.V_m)).all()), \
                "segment V must stay finite"
            parts.append(np.asarray(seg_signals.spikes, dtype=bool))
            seg_receipts.append(seg_signals.metadata["poisson_drive"])
            remaining -= k
            del seg_signals
        t_simulate = time.perf_counter() - t_seg
        spikes = np.concatenate(parts, axis=0)
        del parts
        assert spikes.shape == (sim.n_steps, sum(
            1 for _ in model.neuron_table())), "raster must cover every neuron"
    pops = _pop_index(model)
    by_pop = score_by_population(spikes, pops)
    regime = score(spikes)
    estimands = p3v2_estimands(spikes, pops, DT_MS, T_s)
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes)
    import os

    os.replace(tmp, out / fname)
    backend = jax.default_backend()  # the backend the run used, never a fixed label
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "status": "unscored pilot" if pilot else ("signed" if signed else "unscored"),
             "n": int(spikes.shape[1]), "scale": scale,
             "lesion": [[list(pre), list(post)] for pre, post in lesions],
             "total_synapses": int(np.asarray(model.params["edge_list"].pre).shape[0]),
             "steps": int(spikes.shape[0]), "dt_ms": DT_MS,
             "duration_ms": float(T_s) * 1000.0, "dtype": "float32",
             "backend": backend,
             "poisson_drive": build_drive(seed),
             "wall_s": {"construct": t_construct, "simulate": t_simulate},
             "regime": regime, "pops": by_pop, "estimands_p3v2": estimands}
    if seg_receipts is not None:
        entry["segment_s"] = float(seg_s)
        entry["n_segments"] = int(len(seg_receipts))
        entry["segment_poisson_receipts"] = seg_receipts
    entries[fname] = entry
    if signed and not pilot:
        entry["prereg"] = PREREG_P3V2_SIGNED
        entry["prereg_sha256"] = _prereg_sha256()
    manifest = {"git_sha": sha, "dtype": "float32", "backend": backend,
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return {"spikes": spikes, "steps": sim.n_steps, "regime": regime,
            "pops": by_pop, "estimands": estimands,
            "wall_s": {"construct": t_construct, "simulate": t_simulate}}


def _pilot(args):
    """Run one unscored pilot and print its P3_v2 estimands."""
    lesion = tuple(args.lesion) if args.lesion else None
    tag = "pilot" + ("_lesion_%s_%s" % tuple(args.lesion) if args.lesion else "")
    result = run(args.seed, tag, args.out, scale=args.scale, T_s=args.T_s,
                 lesion=lesion, pilot=True, segment_s=args.segment_s)
    try:
        import resource

        peak_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        peak_kb = None
    est = result["estimands"]
    print("status: unscored pilot (seed %d, scale %s, T_s %s, lesion %s, segment_s %s)"
          % (args.seed, args.scale, args.T_s, lesion, args.segment_s))
    print("wall_s: construct %.1f simulate %.1f"
          % (result["wall_s"]["construct"], result["wall_s"]["simulate"]))
    if peak_kb is not None:
        print("peak_rss_MB: %.1f" % (peak_kb / 1024.0,))
    print("L5E rate_hz: %.4f (reference 7.59 Hz, e=%.3f)"
          % (est["rates_hz"]["L5E"], est["R"]["R-L5E"]["e"]))
    for key in sorted(est["R"]):
        row = est["R"][key]
        print("%s: z=%.4f y=%.2f e=%.3f %s"
              % (key, row["z_hz"], row["y_hz"], row["e"],
                 "PASS" if row["pass"] else "FAIL"))
    for layer, row in sorted(est["O-IE"].items()):
        print("O-IE %s: e=%.4f i=%.4f %s"
              % (layer, row["e_hz"], row["i_hz"], "PASS" if row["pass"] else "FAIL"))
    print("O-ORD: %s" % ("PASS" if est["O-ORD"]["pass"] else "FAIL"))
    print("CV: %s" % ("PASS" if est["CV"]["pass"] else "FAIL"))
    print("SYN_fano_3ms (reported, never scored): %s"
          % ", ".join("%s=%.3f" % (k, v) for k, v in sorted(est["SYN_fano_3ms"].items())))
    print("receipt: %s/manifest.json" % args.out)
    return 0


def _score(args):
    """Run one signed scored seed at the frozen protocol (prereg line 61)."""
    if not args.signed:
        raise SystemExit("error: score needs --signed; unscored runs use pilot")
    if int(args.seed) not in SCORED_SEEDS:
        raise SystemExit(
            "error: score seed must be one of 4-8 (held-out scored seeds); "
            "got %r" % (args.seed,))
    parts = Path(args.out).resolve().parts
    if not any(lane_kit._is_scored_part(part) for part in parts):
        raise SystemExit(
            "error: refusing scored run outside a scored/ path: %s" % args.out)
    tag = "scored"
    result = run(int(args.seed), tag, args.out, scale=SCORE_SCALE,
                 T_s=SCORE_T_S, signed=True, lesion=None, pilot=False,
                 segment_s=SCORE_SEGMENT_S)
    est = result["estimands"]
    print("status: signed (seed %d, scale %s, T_s %s, lesion None, segment_s %s)"
          % (int(args.seed), SCORE_SCALE, SCORE_T_S, SCORE_SEGMENT_S))
    print("wall_s: construct %.1f simulate %.1f"
          % (result["wall_s"]["construct"], result["wall_s"]["simulate"]))
    for key in sorted(est["R"]):
        row = est["R"][key]
        print("%s: z=%.4f y=%.2f e=%.3f %s"
              % (key, row["z_hz"], row["y_hz"], row["e"],
                 "PASS" if row["pass"] else "FAIL"))
    for layer, row in sorted(est["O-IE"].items()):
        print("O-IE %s: e=%.4f i=%.4f %s"
              % (layer, row["e_hz"], row["i_hz"], "PASS" if row["pass"] else "FAIL"))
    print("O-ORD: %s" % ("PASS" if est["O-ORD"]["pass"] else "FAIL"))
    print("CV: %s" % ("PASS" if est["CV"]["pass"] else "FAIL"))
    print("SYN_fano_3ms (reported, never scored): %s"
          % ", ".join("%s=%.3f" % (k, v) for k, v in sorted(est["SYN_fano_3ms"].items())))
    print("receipt: %s/manifest.json" % args.out)
    return 0


def main(argv=None):
    """Command line: ``pilot ...`` (unscored, seed 0) or ``score ...`` (signed, seeds 4-8)."""
    import argparse

    parser = argparse.ArgumentParser(prog="potjans2014_api.py")
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="run one unscored pilot (seed 0 only)")
    pilot.add_argument("--seed", type=int, required=True)
    pilot.add_argument("--scale", type=float, required=True)
    pilot.add_argument("--T-s", type=float, required=True)
    pilot.add_argument("--out", required=True)
    pilot.add_argument("--lesion", nargs=2, default=None, metavar=("PRE", "POST"))
    pilot.add_argument("--segment-s", type=float, default=None)
    score = sub.add_parser("score", help="run one signed scored seed (4-8)")
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--signed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "pilot":
        return _pilot(args)
    if args.command == "score":
        return _score(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
