"""Tests for the potjans2014 lane API (P2 route: PORT_SPEC rows 8-14).

Each passing test reads the realized model through its accessors (neuron
table, edge list, emitter columns, signals receipts) and compares it with
values restated from SOURCE_P1.md section 2. Gap tests (G-MULT, G-TAUSYN,
G-POPDRIVE, G-WDIST, G-DDIST) run the closed core surfaces and must keep
constructing.
"""

import functools
import math
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jaxfne as jtfne
assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
from artifacts.etudes.papers.potjans2014 import potjans2014_api as L

SCALE_SMALL = 0.002  # 155 cells, about 6e5 synapses
SCALE_CHECK = 0.01  # 772 cells: the per-projection count check (seed 0)
DT = L.DT_MS

# Values restated from SOURCE_P1.md section 2 (not read from the lane module).
PAPER_W_PA = 87.8
PAPER_W_SD_PA = 8.8
PAPER_G = -4.0
PAPER_TAU_SYN_MS = 0.5
PAPER_C_M_PF = 250.0
PAPER_D_E = (1.5, 0.75)
PAPER_D_I = (0.8, 0.4)
PAPER_K_EXT = {
    ("L23", "E"): 1600, ("L23", "PV"): 1500, ("L4", "E"): 2100, ("L4", "PV"): 1900,
    ("L5", "E"): 2000, ("L5", "PV"): 1900, ("L6", "E"): 2900, ("L6", "PV"): 2100,
}


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def _step_mv_per_pa():
    """Per-step edge weight per pA: (1/C_m) tau (1 - exp(-dt/tau)), restated."""
    return PAPER_TAU_SYN_MS * (1.0 - math.exp(-DT / PAPER_TAU_SYN_MS)) / PAPER_C_M_PF


def _intended_mean_mv(src, tgt):
    pa = PAPER_W_PA * (abs(PAPER_G) if src[1] == "PV" else 1.0)
    if src == ("L4", "E") and tgt == ("L23", "E"):
        pa *= 2.0
    return pa * _step_mv_per_pa()


def _intended_k(src, tgt):
    """Mean in-degree K_full / N_post_full with K_full = ln(1-C)/ln(1-1/(N_pre N_post))."""
    c = L.PROB[(tgt, src)]
    n_pre, n_post = L.POP_N[src], L.POP_N[tgt]
    return (math.log1p(-c) / math.log1p(-1.0 / (n_pre * n_post))) / n_post


def _folded_sd(mu, sd):
    """SD of |X| for X ~ N(mu, sd): the core folds negative draws at zero."""
    z = mu / sd
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    cdf_neg = 0.5 * (1.0 + math.erf(-z / math.sqrt(2.0)))
    mean_abs = mu * (1.0 - 2.0 * cdf_neg) + 2.0 * sd * phi
    return math.sqrt(mu * mu + sd * sd - mean_abs * mean_abs)


@functools.lru_cache(maxsize=None)
def _lif_model(scale=SCALE_SMALL, seed=0):
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        L.build_spec(scale=scale, seed=seed),
        nt.RuntimeConfiguration(seed=seed, duration_ms=20.0, dt_ms=DT,
                                emitter="lif"),
    )


def _edges(model):
    el = model.params["edge_list"]
    rows = model.neuron_table()
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in rows]
    return (np.asarray(el.pre), np.asarray(el.post), np.asarray(el.weight),
            np.asarray(el.delay_steps), lab)


def test_build_spec_smoke():
    """The lane spec builds: 55 projections, each on the multinomial in-degree route."""
    spec = L.build_spec(scale=SCALE_SMALL, seed=0)
    assert spec is not None, "fixture must build the lane spec"
    assert len(spec.area_connections) == 55, "55 nonzero paper projections"
    for ac in spec.area_connections:
        assert ac.sampling == "multinomial", "rules sample multinomial (row 8)"
        assert ac.probability is None, "in_degree replaces the probability form"
        assert ac.in_degree is not None and ac.in_degree > 0, "mean in-degree declared"
        assert ac.delay_std_ms is not None, "per-rule delay spread declared (row 10)"
        assert ac.plastic.w_std is not None, "per-rule weight spread declared (row 9)"


def test_eight_population_counts():
    """Realized counts equal round_half_up(POP_N * scale) per population (PORT_SPEC 1)."""
    model = _lif_model()
    rows = model.neuron_table()
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in rows]
    want = {p: max(1, _round_half_up(L.POP_N[p] * SCALE_SMALL)) for p in L.POP_N}
    assert len(rows) == sum(want.values()), "every declared cell is realized"
    for pop, n_want in sorted(want.items()):
        got = sum(1 for q in lab if q == pop)
        assert got == n_want, "population %r realizes its share" % (pop,)
        assert got > 0, "every paper population is present at this scale"


def test_n_path_counts_still_build():
    """The fraction-of-n path still realizes the Table 5 fractions of n (PORT_SPEC 1)."""
    from jaxfne import neuronal_tensor as nt

    model = jtfne.construct(
        L.build_spec(n=200, seed=0),
        nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT, emitter="lif"))
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in model.neuron_table()]
    want = L._pop_counts(200)
    assert len(lab) == 200, "n=200 realizes every cell"
    for pop, n_want in sorted(want.items()):
        assert sum(1 for q in lab if q == pop) == n_want, "n path population share"


def test_lif_params_exact():
    """Realized LIF columns carry the paper Table 5 values (PORT_SPEC 1-3)."""
    model = _lif_model()
    em = model.params["emitter"]
    n = len(np.asarray(em.drive))
    assert n == len(model.neuron_table()), "emitter covers every neuron"
    assert np.allclose(np.asarray(em.tau_m_ms), 10.0), "tau_m = 10 ms"
    assert np.allclose(np.asarray(em.t_ref_ms), 2.0), "tau_ref = 2 ms"
    v_th = np.asarray(em.v_thresh_mV)
    v_re = np.asarray(em.v_reset_mV)
    assert np.allclose(v_th - v_re, 15.0), "paper theta-V_reset gap exact"
    assert np.allclose(v_re, 0.0), "origin at V_reset (PORT_SPEC D-VSHIFT)"
    assert np.allclose(np.asarray(em.eta_rp_mV), 0.0), "no PCL refractory tail"
    assert np.allclose(np.asarray(em.delta_sra_mV), 0.0), "no adaptation"
    assert np.allclose(np.asarray(em.drive), 0.0), "lane drive is 0.0"


def test_tau_syn_realized():
    """Every neuron carries the paper synaptic filter tau_syn = 0.5 ms (PORT_SPEC 11)."""
    em = _lif_model().params["emitter"]
    assert np.allclose(np.asarray(em.tau_syn_ms), PAPER_TAU_SYN_MS), "tau_syn = 0.5 ms"


def test_in_degree_exact_totals():
    """Each projection realizes round_half_up(k * n_post) synapses at scale 0.01,
    seed 0, with k the paper's full-scale mean in-degree held at every scale (PORT_SPEC 8)."""
    model = _lif_model(scale=SCALE_CHECK, seed=0)
    pre, post, _, _, lab = _edges(model)
    lab_arr = np.array(lab)
    spec = L.build_spec(scale=SCALE_CHECK, seed=0)
    assert len(spec.area_connections) == 55, "55 nonzero paper projections"
    for ac in spec.area_connections:
        src_q = (ac.source_layer, ac.source_neuron_type)
        tgt_q = (ac.target_layer, ac.target_neuron_type)
        n_post = sum(1 for q in lab if q == tgt_q)
        k = _intended_k(src_q, tgt_q)
        assert ac.in_degree == pytest.approx(k, rel=1e-12), "declared k is the paper k"
        want = _round_half_up(k * n_post)
        got = int(((lab_arr == np.array(src_q)).all(axis=1)[pre]
                   & (lab_arr == np.array(tgt_q)).all(axis=1)[post]).sum())
        assert got == want, \
            "rule %s%s->%s%s realizes round_half_up(k n_post)" % (src_q + tgt_q)


def test_weights_per_rule():
    """Realized |w| mean and SD per rule match the intended per-step values (PORT_SPEC 9, 14).

    Tolerance: the mean within 5 standard errors (SD/sqrt(K)), the SD within 5
    standard errors of a sample SD (SD/sqrt(2K)), K the realized edge count.
    The spread scales with the mean: delta_w / w on every rule, so |g| w and 2 w carry
    |g| delta_w and 2 delta_w (declared reading, PORT_SPEC row 9).
    """
    model = _lif_model(scale=SCALE_CHECK, seed=0)
    pre, post, weight, _, lab = _edges(model)
    lab_arr = np.array(lab)
    spec = L.build_spec(scale=SCALE_CHECK, seed=0)
    for ac in spec.area_connections:
        src_q = (ac.source_layer, ac.source_neuron_type)
        tgt_q = (ac.target_layer, ac.target_neuron_type)
        sel = ((lab_arr == np.array(src_q)).all(axis=1)[pre]
               & (lab_arr == np.array(tgt_q)).all(axis=1)[post])
        k_real = int(sel.sum())
        assert k_real > 1, "fixture: projection must hold edges"
        mags = np.abs(weight[sel]).astype(np.float64)
        mean_want = _intended_mean_mv(src_q, tgt_q)
        sd_want = mean_want * PAPER_W_SD_PA / PAPER_W_PA
        assert abs(mags.mean() - mean_want) <= 5 * sd_want / math.sqrt(k_real), \
            "rule %s%s->%s%s mean weight" % (src_q + tgt_q)
        assert abs(mags.std(ddof=1) - sd_want) <= 5 * sd_want / math.sqrt(2 * k_real), \
            "rule %s%s->%s%s weight SD" % (src_q + tgt_q)
        signs = np.sign(weight[sel])
        want_sign = 1.0 if src_q[1] == "E" else -1.0
        assert bool((signs == want_sign).all()), "sign follows the source type"


def test_delays_per_rule():
    """Realized delays per rule snap to dt; the mean matches the paper and the SD
    matches the core's folded Gaussian (PORT_SPEC 10).

    Tolerance: the mean within 4 standard errors plus 0.01 ms (snap and sub-step
    floor); the SD within 4 standard errors of a sample SD plus 0.01 ms.
    """
    model = _lif_model(scale=SCALE_CHECK, seed=0)
    pre, post, _, delay, lab = _edges(model)
    lab_arr = np.array(lab)
    assert delay.min() >= 1, "every delay is at least one step"
    spec = L.build_spec(scale=SCALE_CHECK, seed=0)
    for ac in spec.area_connections:
        src_q = (ac.source_layer, ac.source_neuron_type)
        tgt_q = (ac.target_layer, ac.target_neuron_type)
        sel = ((lab_arr == np.array(src_q)).all(axis=1)[pre]
               & (lab_arr == np.array(tgt_q)).all(axis=1)[post])
        k_real = int(sel.sum())
        mu, sd = PAPER_D_E if src_q[1] == "E" else PAPER_D_I
        d_ms = delay[sel].astype(np.float64) * DT
        assert abs(d_ms.mean() - mu) <= 4 * sd / math.sqrt(k_real) + 0.01, \
            "rule %s%s->%s%s mean delay" % (src_q + tgt_q)
        assert abs(d_ms.std(ddof=1) - _folded_sd(mu, sd)) <= \
            4 * _folded_sd(mu, sd) / math.sqrt(2 * k_real) + 0.01, \
            "rule %s%s->%s%s delay SD (folded Gaussian)" % (src_q + tgt_q)


def test_external_drive_per_population():
    """Eight drive entries, one per population: its triple, its k_ext trains at
    8 Hz, and the full-deposit amplitude w tau_syn / C_m (PORT_SPEC 12)."""
    from jaxfne._signals import resolve_poisson_drive_target

    entries = list(L.build_sim(T_s=0.02, seed=0).poisson_drive)
    assert len(entries) == 8, "one drive entry per population"
    want_amp = PAPER_W_PA * PAPER_TAU_SYN_MS / PAPER_C_M_PF
    model = _lif_model()
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in model.neuron_table()]
    for e in entries:
        pop = (e["target"][1], e["target"][2])
        assert tuple(e["target"]) == ("mc",) + pop, "entry targets its own triple"
        assert int(e["counts"]) == PAPER_K_EXT[pop], "entry carries its k_ext"
        assert float(e["rate_hz"]) == pytest.approx(8.0), "8 Hz per train"
        assert float(e["amplitude"]) == pytest.approx(want_amp, rel=1e-12), \
            "amplitude is the full deposit per arrival"
        idx = resolve_poisson_drive_target(model, tuple(e["target"]))
        assert len(idx) == sum(1 for q in lab if q == pop), "triple resolves to its population"
    signals = model.simulate(L.build_sim(T_s=0.02, seed=1))
    receipts = [(tuple(r["target"]), int(r["counts"]))
                for r in signals.metadata["poisson_drive"]]
    declared = [(tuple(e["target"]), int(e["counts"])) for e in L.build_drive(seed=1)]
    assert receipts == declared, "declared entries reach the run receipt"


def test_drive_entry_changes_only_its_population():
    """With no recurrent projections the input is the drive alone, so changing one
    entry's counts changes only that population's spikes (PORT_SPEC 12)."""
    from jaxfne import neuronal_tensor as nt

    spec = L.build_spec(scale=SCALE_SMALL, seed=0)
    silent = nt.NeuronalTensor(name="drive-only", areas=spec.areas, area_connections=[])
    model = jtfne.construct(
        silent, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT, emitter="lif"))
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in model.neuron_table()]
    base = L.build_drive(seed=0)
    changed = [dict(e) for e in base]
    j = [i for i, e in enumerate(base) if tuple(e["target"]) == ("mc", "L4", "E")][0]
    changed[j]["counts"] = 100  # L4e: 2100 -> 100 trains

    def _spikes(entries):
        sim = jtfne.Simulation(
            duration_ms=500.0, dt_ms=DT, seed=0, poisson_drive=entries,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
        return np.asarray(model.simulate(sim).spikes)

    s0, s1 = _spikes(base), _spikes(changed)
    for pop in PAPER_K_EXT:
        cols = [i for i, q in enumerate(lab) if q == pop]
        if pop == ("L4", "E"):
            assert int(s1[:, cols].sum()) < int(s0[:, cols].sum()), \
                "fewer trains drive fewer L4e spikes"
        else:
            assert np.array_equal(s0[:, cols], s1[:, cols]), \
                "population %r unchanged by another entry" % (pop,)


def test_seed_changes_edges():
    """Seed 0 and seed 1 wire different edges (PORT_SPEC 8)."""
    a = _lif_model(scale=SCALE_SMALL, seed=0).params["edge_list"]
    b = _lif_model(scale=SCALE_SMALL, seed=1).params["edge_list"]
    same = (np.array_equal(np.asarray(a.pre), np.asarray(b.pre))
            and np.array_equal(np.asarray(a.post), np.asarray(b.post)))
    assert not same, "seed 0 and seed 1 must give different edges"


def test_zero_drive_silent():
    """Declared drive=0.0 holds the network silent with no input (PORT_SPEC 13)."""
    model = _lif_model()
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=DT, seed=0,
                           runtime=jtfne.RuntimeConfig(
                               recurrent_backend="edge_list"))
    spikes = np.asarray(model.simulate(sim).spikes)
    assert int(spikes.sum()) == 0, "no-drive network must stay silent"


def test_score_keys():
    """The lane scorer returns the analysis estimands (checklist E3)."""
    regime = L.score(np.zeros((100, 10), dtype=bool), DT)
    assert set(regime) == {"rate_hz", "cv_isi", "kappa", "active_fraction",
                           "spectral_peak_excess", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"
    pops = L.score_by_population(np.zeros((100, 10), dtype=bool),
                                 [("L23", "E")] * 5 + [("L4", "E")] * 5, DT)
    assert set(pops) == {"L23E", "L4E"}, "per-population keys"
    assert pops["L23E"]["n"] == 5, "per-population sizes"


def test_lesion_none_matches_baseline():
    """``lesion=None`` realizes exactly the baseline edges (bit-identical)."""
    from jaxfne import neuronal_tensor as nt

    def _wired(**kw):
        return jtfne.construct(
            L.build_spec(n=800, seed=0, **kw),
            nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                    emitter="lif"))

    a, b = _wired(), _wired(lesion=None)
    for arr_a, arr_b in zip(_edges(a)[:2], _edges(b)[:2]):
        assert np.array_equal(np.asarray(arr_a), np.asarray(arr_b)), \
            "lesion=None must realize the baseline edges"


def _pair_counts(model):
    """Realized edge counts keyed ``(source, target)`` population pairs."""
    pre, post, _, _, lab = _edges(model)
    lab_arr = np.array(lab)
    counts = {}
    for tgt, src in L.PROB:
        sel = ((lab_arr == np.array(src)).all(axis=1)[pre]
               & (lab_arr == np.array(tgt)).all(axis=1)[post])
        counts[(src, tgt)] = int(sel.sum())
    return counts


def test_lesion_removes_only_named_projection():
    """``lesion=("L5i", "L5e")`` zeroes exactly the L5i -> L5e edges at n=800."""
    from jaxfne import neuronal_tensor as nt

    def _wired(**kw):
        return jtfne.construct(
            L.build_spec(n=800, seed=0, **kw),
            nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                    emitter="lif"))

    base, cut = _wired(), _wired(lesion=("L5i", "L5e"))
    base_counts, cut_counts = _pair_counts(base), _pair_counts(cut)
    assert base_counts[(("L5", "PV"), ("L5", "E"))] > 0, \
        "fixture: the baseline must wire L5i -> L5e"
    assert cut_counts[(("L5", "PV"), ("L5", "E"))] == 0, \
        "the lesion must remove every L5i -> L5e edge"
    for pair in base_counts:
        if pair == (("L5", "PV"), ("L5", "E")):
            continue
        assert cut_counts[pair] == base_counts[pair], \
            "lesion must leave projection %s%s->%s%s untouched" % (pair[0] + pair[1])
    spec = L.build_spec(n=800, seed=0, lesion=("L5i", "L5e"))
    assert len(spec.area_connections) == 54, "one of the 55 rules is dropped"


def test_lesion_unknown_refused():
    """Unknown lesion population names are refused with ValueError."""
    with pytest.raises(ValueError):
        L.build_spec(n=800, seed=0, lesion=("L5x", "L5e"))
    with pytest.raises(ValueError):
        L.build_spec(n=800, seed=0, lesion=("L5i", "L9e"))
    with pytest.raises(ValueError):
        L.build_spec(n=800, seed=0, lesion=[("L5i", "L5e"), ("L4x", "L4E")])
    ok = L.build_spec(n=800, seed=0,
                       lesion=[("L5i", "L5e"), (("L4", "PV"), ("L4", "E"))])
    assert len(ok.area_connections) == 53, "two named projections are dropped"


def test_gap_multapse_sampler():
    """Paper wiring draws K synapses per projection with replacement
    (multapses allowed, SOURCE_P1.md section 1); the core draws distinct
    pairs only. Proposed: sampling="multinomial" with n_synapses=K."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gmult",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0),
            delay_ms=1.5, sampling="multinomial", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "multapse sampler must construct"


def test_gap_tau_syn_filter():
    """Paper PSCs decay with tau_syn = 0.5 ms (SOURCE_P1.md section 1); the
    lif route sums delta arrivals with no filter. Proposed: tau_syn_ms LIF
    param consumed by the kernel."""
    from jaxfne import neuronal_tensor as nt

    lif = dict(L._LIF_POTJANS, tau_syn_ms=0.5)
    tensor = nt.NeuronalTensor(
        name="potjans-gtausyn",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=2,
            neuron_types=[nt.NeuronType.make(
                "E", fraction=1.0, emitter="lif", emitter_params=dict(lif),
                drive=0.0)])])],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "tau_syn_ms must construct and filter arrivals"


def test_gap_per_population_drive():
    """Paper external input is per-population k_ext trains (SOURCE_P1.md
    section 2: 1600-2900 per cell); poisson_drive target="all" is the only
    path (`resolve_poisson_drive_target` in `jaxfne/_signals.py`). Proposed: a triple target."""
    model = _lif_model()
    sim = jtfne.Simulation(
        duration_ms=20.0, dt_ms=DT, seed=0,
        poisson_drive={"rate_hz": 8.0, "amplitude": 0.1, "counts": 1600,
                       "seed": 0, "target": ("mc", "L23", "E")},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    assert np.asarray(signals.spikes).shape[0] == sim.n_steps, \
        "targeted drive must simulate"


def test_gap_weight_distribution():
    """Paper weights are Gaussian w +- delta_w, sign-preserving
    (SOURCE_P1.md section 2); the core realizes one constant per rule.
    Proposed: PlasticParams(w_std=delta_w)."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gwdist",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0, w_std=0.1),
            delay_ms=1.5, sampling="density", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "weight spread must construct"


def test_gap_delay_distribution():
    """Paper delays are Gaussian d +- delta_d, sign-preserving and snapped
    to the grid (SOURCE_P1.md section 2); the core fans one constant per
    rule out. Proposed: AreaConnection delay_std_ms."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gddist",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0),
            delay_ms=1.5, delay_std_ms=0.75,
            sampling="density", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "delay spread must construct"


SEG_T_S = 0.02  # 200 steps: the segmented-run fixture window
SEG_S = 0.005  # 50 steps: four even segments over the fixture window


def test_segmented_raster_shape_and_chunk0_prefix(tmp_path):
    """The chained raster covers the grid; chunk 0 equals the one-call prefix."""
    seg = L.run(0, "seg", str(tmp_path / "seg"), scale=SCALE_SMALL,
                T_s=SEG_T_S, pilot=True, segment_s=SEG_S)
    whole = L.run(0, "whole", str(tmp_path / "whole"), scale=SCALE_SMALL,
                  T_s=SEG_T_S, pilot=True)
    n_steps = int(round(SEG_T_S * 1000.0 / DT))
    assert seg["spikes"].shape == (n_steps, whole["spikes"].shape[1]), \
        "segmented raster covers every step and neuron"
    first_rows = int(round(SEG_S * 1000.0 / DT))
    assert int(seg["spikes"][:first_rows].sum()) > 0, \
        "fixture must spike in chunk 0"
    assert np.array_equal(seg["spikes"][:first_rows],
                          whole["spikes"][:first_rows]), \
        "chunk 0 equals the unsegmented prefix bit-for-bit"


def test_segmented_uneven_tail(tmp_path):
    """A segment length missing the run end leaves one shorter last segment."""
    import json

    seg_s = 0.007  # 70 steps over 200: 70 + 70 + 60
    out = tmp_path / "tail"
    result = L.run(0, "seg", str(out), scale=SCALE_SMALL,
                   T_s=SEG_T_S, pilot=True, segment_s=seg_s)
    assert result["spikes"].shape[0] == int(round(SEG_T_S * 1000.0 / DT))
    entry = json.loads((out / "manifest.json").read_text())["runs"][0]
    assert entry["n_segments"] == 3, "70 + 70 + 60 steps over 200"
    assert len(entry["segment_poisson_receipts"]) == 3


def test_segmented_manifest_receipts(tmp_path):
    """The manifest records segment_s, the count and one receipt per segment."""
    import json

    from jaxfne._signals import poisson_chunk_seed

    out = tmp_path / "segman"
    L.run(0, "seg", str(out), scale=SCALE_SMALL,
          T_s=SEG_T_S, pilot=True, segment_s=SEG_S)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["runs"]) == 1, "one run recorded"
    entry = manifest["runs"][0]
    assert entry["segment_s"] == pytest.approx(SEG_S)
    assert entry["n_segments"] == 4
    receipts = entry["segment_poisson_receipts"]
    assert len(receipts) == 4, "one receipt per segment"
    for i, receipt in enumerate(receipts):
        assert isinstance(receipt, list) and len(receipt) == 8, \
            "segment %d receipts cover all eight populations" % i
        for e in receipt:
            assert e["chunk_index"] == i, "segment %d names its chunk" % i
            assert e["chunk_seed"] == poisson_chunk_seed(int(e["seed"]), i), \
                "segment %d chunk seed derives from its chunk" % i


def test_segment_s_refused(tmp_path):
    """A non-finite or non-positive segment_s is refused by name."""
    for bad in (0, 0.0, -1.5, float("nan"), float("inf"), "abc"):
        with pytest.raises(ValueError, match="segment_s"):
            L.run(0, "bad", str(tmp_path), scale=SCALE_SMALL,
                  T_s=SEG_T_S, pilot=True, segment_s=bad)


def test_default_manifest_carries_no_segment_keys(tmp_path):
    """The default one-call manifest records no segment keys."""
    import json

    out = tmp_path / "plain"
    L.run(0, "plain", str(out), scale=SCALE_SMALL, T_s=SEG_T_S, pilot=True)
    entry = json.loads((out / "manifest.json").read_text())["runs"][0]
    for key in ("segment_s", "n_segments", "segment_poisson_receipts"):
        assert key not in entry, "default call records no %s" % key


def test_manifest_records_the_realized_backend(tmp_path, monkeypatch):
    """The manifest records jax.default_backend(); a fixed "cpu" label fails here."""
    import json

    monkeypatch.setattr(L.jax, "default_backend", lambda: "gpu")
    out = tmp_path / "plain"
    L.run(0, "plain", str(out), scale=SCALE_SMALL, T_s=SEG_T_S, pilot=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["backend"] == "gpu", "top-level backend is the realized one"
    assert manifest["runs"][0]["backend"] == "gpu", "per-run backend is the realized one"


def test_score_refuses_out_of_band_seeds(tmp_path):
    """score takes the held-out scored seeds 4-8 only (prereg Seeds)."""
    for seed in (0, 1, 2, 3, 9):
        with pytest.raises(SystemExit, match="4-8"):
            L.main(["score", "--seed", str(seed),
                    "--out", str(tmp_path / "scored" / ("seed%d" % seed)),
                    "--signed"])


def test_score_refuses_without_signed(tmp_path):
    """score without --signed is refused (prereg scored/unscored split)."""
    with pytest.raises(SystemExit, match="--signed"):
        L.main(["score", "--seed", "4",
                "--out", str(tmp_path / "scored" / "seed4")])


def test_score_refuses_outside_scored_path(tmp_path):
    """Signed runs land under a scored/ path (prereg split, lane_kit rule)."""
    with pytest.raises(SystemExit, match="scored"):
        L.main(["score", "--seed", "4",
                "--out", str(tmp_path / "plain"), "--signed"])


def test_signed_run_manifest_records_prereg(tmp_path):
    """A signed run writes status signed and the prereg sha256 (P4 manifest)."""
    import hashlib
    import json

    out = tmp_path / "scored"
    L.run(4, "scored", str(out), scale=SCALE_SMALL, T_s=SEG_T_S,
          signed=True, segment_s=SEG_S)
    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["runs"][0]
    assert entry["status"] == "signed", "signed run records its status"
    assert entry["seed"] == 4
    assert entry["prereg"] == L.PREREG_P3V2_SIGNED
    want = hashlib.sha256(
        (L._repo_root / L.PREREG_P3V2_SIGNED).read_bytes()).hexdigest()
    assert entry["prereg_sha256"] == want, "prereg sha256 computed at run time"
