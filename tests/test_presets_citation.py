"""The cell-type presets cite what Izhikevich (2003) actually contains.

The paper has no table. Its Fig. 2 gives RS, FS and LTS parameters; it has no VIP or IS cell
(`artifacts/etudes/papers/izhikevich2003/SOURCE_P1.md`).
"""

import pytest

from jaxfne.presets import CELL_TYPE_PRESETS

# (a, b, c, d) from Izhikevich 2003, Fig. 2.
FIG2 = {
    "E_RS": (0.02, 0.2, -65.0, 8.0),
    "PV_FS": (0.1, 0.2, -65.0, 2.0),
    "SST_LTS": (0.02, 0.25, -65.0, 2.0),
}


def test_no_preset_cites_a_table():
    for name, preset in CELL_TYPE_PRESETS.items():
        assert "Table" not in preset["source"], name


@pytest.mark.parametrize("name", sorted(FIG2))
def test_fig2_presets_cite_fig2_and_match_it(name):
    preset = CELL_TYPE_PRESETS[name]
    assert preset["source"].startswith("Izhikevich 2003 Fig. 2"), preset["source"]
    got = tuple(preset[k] for k in "abcd")
    assert got == pytest.approx(FIG2[name], rel=1e-12)


def test_vip_is_does_not_claim_the_paper():
    source = CELL_TYPE_PRESETS["VIP_IS"]["source"]
    assert "not in Izhikevich 2003" in source, source
