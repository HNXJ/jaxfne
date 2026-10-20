"""Every cell-type catalog sign reads the one table, CELL_TYPE_SIGN."""

import jaxfne.emitters as emitters
import jaxfne.presets as presets
from jaxfne.presets import CELL_TYPE_PRESETS
from jaxfne.tutorial_utils import make_cell_type_catalog


def test_emitters_name_is_the_presets_table():
    assert emitters.CELL_TYPE_SIGN is presets.CELL_TYPE_SIGN


def test_presets_signs_equal_table():
    assert CELL_TYPE_PRESETS, "fixture must build entries"
    for name, preset in CELL_TYPE_PRESETS.items():
        assert preset["sign"] == presets.CELL_TYPE_SIGN[preset["label"]], name


def test_tutorial_catalog_signs_equal_table():
    catalog = make_cell_type_catalog()
    assert catalog, "fixture must build entries"
    for label, preset in catalog.items():
        assert preset.sign == presets.CELL_TYPE_SIGN[label], label


def test_flipped_table_entry_changes_tutorial_catalog(monkeypatch):
    before = make_cell_type_catalog()["E"].sign
    monkeypatch.setitem(presets.CELL_TYPE_SIGN, "E", -1.0)
    after = make_cell_type_catalog()["E"].sign
    assert before == 1.0
    assert after == -1.0
