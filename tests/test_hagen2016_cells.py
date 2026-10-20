"""Reduced Hagen 2016 cells (jaxfne.fields.hagen2016_cells): profiles, depth map, refusals."""

import numpy as np
import pytest

from jaxfne.fields import hagen2016_cells as hc
from jaxfne.fields.hagen2016_cells import hagen2016_cell

SRC = np.array(hc.SOURCE_BOUNDS_UM)


def _profile_area(names):
    tot = 0.0
    for n in names:
        p = hc.PROFILES[n]
        tot += np.sum(p["up_area_um2"]) + np.sum(p["down_area_um2"]) + np.pi * p["soma_diam_um"] * p["soma_length_um"]
    return tot / len(names)


@pytest.mark.parametrize("pop", list(hc.POPULATIONS))
@pytest.mark.parametrize("merge", [1, 3])
def test_cell_keeps_the_mean_membrane_area(pop, merge):
    for tb in (SRC, SRC * 1.7 + 5.0):
        cell = hagen2016_cell(pop, tb, merge=merge)
        np.testing.assert_allclose(cell.area_cm2.sum() * 1e8, _profile_area(hc.POPULATIONS[pop]), rtol=1e-9)


def test_tufts_reach_the_pia_in_source_and_target():
    tgt = np.array([0.0, 130.0, 1200.0, 1470.0, 1700.0, 2000.0])
    for pop, files in [("L23E", None), ("L5E", None)]:
        for tb in (SRC, tgt):
            cell = hagen2016_cell(pop, tb)
            top = min(cell.z0.min(), cell.z1.min())
            scale = (tb[1] - tb[0]) / (SRC[1] - SRC[0])
            assert abs(top) <= hc.BIN_UM * max(1.0, scale) + 1e-9, (pop, top)
    # L6E averages a cell that reaches the pia with one that ends in L2/3; its profile still reaches L1
    cell = hagen2016_cell("L6E", tgt)
    assert min(cell.z0.min(), cell.z1.min()) < tgt[1]


def test_soma_sits_at_the_target_layer_midpoint_and_profiles_map_by_layer():
    tgt = np.array([0.0, 130.0, 1200.0, 1470.0, 1700.0, 2000.0])
    for pop, li in [("L23E", 1), ("L4E", 2), ("L5I", 3), ("L6E", 4)]:
        cell = hagen2016_cell(pop, tgt)
        np.testing.assert_allclose(cell.mid[0], 0.5 * (tgt[li] + tgt[li + 1]), rtol=1e-12)
    # L5E apical membrane in source L2/3 lands in target L2/3, with the same area
    for tb in (SRC, tgt):
        cell = hagen2016_cell("L5E", tb)
        m = (cell.mid > tb[1]) & (cell.mid < tb[2])
        if tb is SRC:
            ref = cell.area_cm2[m].sum()
        else:
            np.testing.assert_allclose(cell.area_cm2[m].sum(), ref, rtol=0.05)
    assert ref > 0


def test_compartment_diameter_is_the_rall_equivalent():
    p = hc.PROFILES["L23E_oi24rpy1"]
    cell = hagen2016_cell("L23E", SRC)
    n_up = len(p["up_area_um2"])
    np.testing.assert_allclose(cell.diam[1:1 + n_up], np.asarray(p["up_rall_d32"]) ** (2 / 3), rtol=1e-12)
    np.testing.assert_allclose(cell.diam[1 + n_up:], np.asarray(p["down_rall_d32"]) ** (2 / 3), rtol=1e-12)


def test_profiles_are_positive_where_defined():
    for name, p in hc.PROFILES.items():
        for key in ("up_area_um2", "up_rall_d32", "down_area_um2", "down_rall_d32"):
            assert np.all(np.asarray(p[key]) > 0), (name, key)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        hagen2016_cell("L1I", SRC)
    with pytest.raises(ValueError):
        hagen2016_cell("L5E", SRC[::-1])
    with pytest.raises(ValueError):
        hagen2016_cell("L5E", SRC[:5])
    with pytest.raises(ValueError):
        hagen2016_cell("L5E", SRC, merge=0)
