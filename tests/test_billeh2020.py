"""Billeh 2020 V1 connectivity table and its distance rule (jaxfne.billeh2020)."""

import numpy as np
import pytest

from jaxfne import billeh2020 as b


def test_table_is_complete_and_spot_values_match_the_release():
    assert len(b.POPULATIONS) == 17
    for table in (b.A_LITERATURE, b.SIGMA_UM, b.B_RATIO, b.NSYN_RANGE):
        assert set(table) == {(p, q) for p in b.POPULATIONS for q in b.POPULATIONS}
    # cells read independently from v1_conn_props.json
    assert b.R0_UM == 75.0
    assert b.A_LITERATURE[("e2", "i2P")] == 0.395
    assert b.A_LITERATURE[("i2S", "i2P")] == 0.857
    assert b.A_LITERATURE[("i5S", "e5")] == 0.317
    assert b.A_LITERATURE[("e6", "e2")] == 0.0
    assert b.SIGMA_UM[("e2", "e2")] == 114.0 and b.SIGMA_UM[("i5S", "e5")] == 85.0
    assert b.B_RATIO[("e2", "e5")] == 0.8 and b.B_RATIO[("e2", "i2P")] is None
    assert b.NSYN_RANGE[("e2", "i2P")] == (3, 8)


@pytest.mark.parametrize("pair", [("e2", "e2"), ("e4", "i4P"), ("i5S", "e5"), ("i1H", "i2H")])
def test_disk_mean_of_p_equals_a_literature(pair):
    assert b.peak_probability(*pair) < 1.0, "fixture must stay below the cap"
    r = (np.arange(200000) + 0.5) * b.R0_UM / 200000
    mean = np.sum(b.connection_probability(*pair, r) * 2 * r) * (b.R0_UM / 200000) / b.R0_UM ** 2
    np.testing.assert_allclose(mean, b.A_LITERATURE[pair], rtol=1e-8)


def test_probability_is_capped_and_decays():
    pair = ("i2S", "i2P")
    assert b.peak_probability(*pair) > 1.0, "fixture must exceed the cap"
    p = b.connection_probability(*pair, np.array([0.0, 50.0, 400.0]))
    assert p[0] == 1.0 and p[2] < p[1] <= 1.0
    np.testing.assert_allclose(b.connection_probability("e2", "e2", 114.0),
                               b.peak_probability("e2", "e2") * np.exp(-1.0), rtol=1e-12)


def test_unknown_pair_and_bad_distance_are_refused():
    with pytest.raises(ValueError):
        b.connection_probability("e3", "e2", 0.0)
    with pytest.raises(ValueError):
        b.connection_probability("e2", "e2", -1.0)
    with pytest.raises(ValueError):
        b.connection_probability("e2", "e2", np.nan)
