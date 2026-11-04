"""Cellrep set through the jaxfne core (``caminos_api.run_core_raster``).

Every manipulation runs through the public API only (construct + simulate);
no ``caminos_jax`` oracle code is reached. Oracle draws the suite compares
against are the :func:`caminos_jax.manip_mask` subsets, which coincide with
the core's by construction (same fold-in tags over E | PV | SOM order).
CPU only; the T_s = 1.0 s smoke net (200/40/40) keeps each run small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

CORE_SIZES = {"NE": 200, "NPV": 40, "NSOM": 40}
CORE_T_S = 1.0
CORE_P = 0.05
CORE_EXT = 50
CORE_SOM_HZ = 4.0


def _core_kw(**over):
    kw = {"T_s": CORE_T_S, "sizes": dict(CORE_SIZES), "p": CORE_P,
          "ext_sources": CORE_EXT, "som_ext_rate_hz": CORE_SOM_HZ}
    kw.update(over)
    return kw


def _emitter(swap=False):
    import jaxfne as jtfne

    import caminos_api as ca

    model = jtfne.construct(ca.build_cellrep_spec(p=CORE_P, seed=1, sizes=CORE_SIZES,
                                                  swap=swap))
    return model.params["emitter"]


def test_core_overrides_only_the_differing_adex_fields():
    import numpy as np

    em = _emitter()
    lab = list(em.labels)
    e, pv, som = lab.index("E"), lab.index("PV"), lab.index("SOM")
    el, b = np.asarray(em.EL_mV), np.asarray(em.b_pA)
    assert float(el[e]) == pytest.approx(-60.0)
    assert float(b[e]) == pytest.approx(130.0)
    assert float(el[pv]) == pytest.approx(-60.0)
    assert float(np.asarray(em.Delta_mV)[som]) == pytest.approx(1.5)
    assert float(b[som]) == pytest.approx(25.0)
    assert float(np.asarray(em.a_nS)[pv]) == pytest.approx(0.0)


def test_core_swap_exchanges_the_pv_som_neuronal_values():
    import numpy as np

    base, swapped = _emitter(), _emitter(swap=True)
    for field in ("EL_mV", "Delta_mV", "a_nS", "b_pA"):
        a, c = np.asarray(getattr(base, field)), np.asarray(getattr(swapped, field))
        lab = list(base.labels)
        for t in ("PV", "SOM"):
            other = "SOM" if t == "PV" else "PV"
            assert (c[[i for i, lab_i in enumerate(lab) if lab_i == t]]
                    == a[[i for i, lab_i in enumerate(lab) if lab_i == other]]).all()


def test_core_realizes_four_receptors_weights_and_one_step_delay():
    import numpy as np

    import jaxfne as jtfne

    import caminos_api as ca
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(ca.build_cellrep_spec(p=CORE_P, seed=1, sizes=CORE_SIZES))
    kw = adex_kernel_args(model, ca.DT_MS)
    assert np.asarray(kw["receptor_tau_ms"]).tolist() == pytest.approx([5.0, 1.0, 2.0, 5.0])
    assert np.asarray(kw["receptor_reversal_mV"]).tolist() == pytest.approx([0.0, 0.0, 0.0, -80.0])
    assert sorted(set(np.asarray(kw["edge_weight_nS"]).tolist())) == pytest.approx(
        [1.25, 2.5, 3.75, 5.0, 6.0])
    assert np.asarray(kw["edge_delay_steps"]).size > 0
    assert (np.asarray(kw["edge_delay_steps"]) == 1).all()
    rec, w = np.asarray(kw["edge_receptor"]), np.asarray(kw["edge_weight_nS"])
    e_pv = int(((rec == 1) & (w == 3.75)).sum())
    e_som = int(((rec == 2) & (w == 2.5)).sum())
    assert e_pv > 1.5 * e_som, "E -> PV connects at twice p: %d vs %d" % (e_pv, e_som)


def test_core_drive_entries_use_per_group_seeds_and_mechanisms():
    import caminos_api as ca

    entries = ca.cellrep_drive_entries(CORE_P, 200, 1, CORE_EXT, CORE_SOM_HZ)
    assert [(e["mechanism"], e["seed"]) for e in entries] == [
        ("ampa_e", 1), ("ampa_pv", 2), ("ampa_som", 3)]
    assert [e["target_cell_types"] for e in entries] == [["E"], ["PV"], ["SOM"]]
    plain = ca.cellrep_drive_entries(CORE_P, 200, 1, CORE_EXT, 0.0)
    assert [e["mechanism"] for e in plain] == ["ampa_e", "ampa_pv"]
    with pytest.raises(ValueError):
        ca.cellrep_drive_entries(CORE_P, 200, 1, CORE_EXT, CORE_SOM_HZ,
                                 som_ext_rate_schedule=[1.0])


@pytest.mark.parametrize("bad", [0.0, -0.01, 0.5 + 1e-9, 1.0])
def test_core_p_outside_zero_half_is_refused(bad):
    import caminos_api as ca

    with pytest.raises(ValueError, match=r"\(0, 0\.5\]"):
        ca.build_cellrep_spec(p=bad, seed=1, sizes=CORE_SIZES)


@pytest.mark.parametrize("key", ["indegree", "ext", "step_order", "param_set"])
def test_core_unknown_keywords_are_refused_by_name(key):
    import caminos_api as ca

    with pytest.raises(ValueError, match=key):
        ca.run_core_raster(1, **_core_kw(**{key: "fixed"}))


def test_core_silence_onset_without_silence_is_refused():
    import caminos_api as ca

    with pytest.raises(ValueError, match="silence_onset_s"):
        ca.run_core_raster(1, **_core_kw(silence_onset_s=6.04))


def test_core_replace_without_rates_is_refused():
    import caminos_api as ca

    with pytest.raises(ValueError, match="replace_rate_hz"):
        ca.run_core_raster(1, **_core_kw(replace={"PV": 0.5}))


@pytest.fixture(scope="module")
def base_raster():
    import caminos_api as ca

    return ca.run_core_raster(1, **_core_kw())


def test_core_raster_shape_dtype_and_firing(base_raster):
    assert base_raster.dtype == bool
    assert base_raster.shape == (10000, 280)
    assert int(base_raster.sum()) > 0, "precondition: the smoke net must fire"
    assert int(base_raster[:, :200].sum()) > 0, "precondition: E must fire"


def test_core_raster_is_deterministic(base_raster):
    import numpy as np

    import caminos_api as ca

    again = ca.run_core_raster(1, **_core_kw())
    assert np.array_equal(again, base_raster)


def test_core_silence_subset_coincides_with_the_oracle():
    import numpy as np

    import jax
    import jaxfne as jtfne

    import caminos_api as ca
    import caminos_jax as cj
    from jaxfne._construct_adex import adex_silence_kwargs

    model = jtfne.construct(ca.build_cellrep_spec(p=CORE_P, seed=1, sizes=CORE_SIZES))
    sim = ca.build_cellrep_sim(
        1, CORE_T_S, ca.cellrep_drive_entries(CORE_P, 200, 1, CORE_EXT, CORE_SOM_HZ),
        silence={"PV": 0.5})
    mask, _, _ = adex_silence_kwargs(model, sim)
    oracle = np.asarray(cj.manip_mask(jax.random.PRNGKey(1), {"PV": 0.5},
                                       {"E": 200, "PV": 40, "SOM": 40}, "silence"))
    assert np.array_equal(np.asarray(mask), oracle)


def test_core_replace_subset_coincides_with_the_oracle():
    import numpy as np

    import jax
    import jaxfne as jtfne

    import caminos_api as ca
    import caminos_jax as cj
    from jaxfne._construct_adex import adex_replace_kwargs

    model = jtfne.construct(ca.build_cellrep_spec(p=CORE_P, seed=1, sizes=CORE_SIZES))
    sim = ca.build_cellrep_sim(
        1, CORE_T_S, ca.cellrep_drive_entries(CORE_P, 200, 1, CORE_EXT, CORE_SOM_HZ),
        replace={"SOM": 0.5}, replace_rate_hz={"SOM": 4.0})
    mask, _, _, _ = adex_replace_kwargs(model, sim)
    oracle = np.asarray(cj.manip_mask(jax.random.PRNGKey(1), {"SOM": 0.5},
                                       {"E": 200, "PV": 40, "SOM": 40}, "replace"))
    assert np.array_equal(np.asarray(mask), oracle)


def test_core_full_som_silence_zeros_som_spikes(base_raster):
    import caminos_api as ca

    assert int(base_raster[:, 240:].sum()) > 0, "precondition: baseline SOM must fire"
    silenced = ca.run_core_raster(1, **_core_kw(silence={"SOM": 1.0}))
    assert int(silenced[:, 240:].sum()) == 0
    assert int(silenced[:, :200].sum()) > 0


def test_core_route_is_lazy_and_defaults_to_oracle():
    import ast

    import cellrep_figures as cf

    assert cf.ROUTE == "oracle"
    tree = ast.parse(Path(cf.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [n for n in top if "caminos_jax" in ast.dump(n)], \
        "caminos_jax must stay a lazy import inside _oracle()"
    assert "caminos_jax" not in dir(cf)


def test_core_run_cell_smoke():
    import cellrep_figures as cf

    out = cf.run_cell(1, T_s=3.0, sizes=dict(CORE_SIZES), ext_sources=CORE_EXT,
                      som_ext_rate_hz=CORE_SOM_HZ, route="core")
    assert out["sizes"] == {"E": 200, "PV": 40, "SOM": 40}
    assert sum(out["counts"].values()) > 0, "precondition: the core smoke must fire"
    assert out["counts"]["E"] > 0
    with pytest.raises(ValueError, match="route must be"):
        cf.run_cell(1, T_s=3.0, sizes=dict(CORE_SIZES), ext_sources=CORE_EXT,
                    som_ext_rate_hz=CORE_SOM_HZ, route="bogus")
