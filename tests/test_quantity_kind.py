"""D28: public QuantityKind vocabulary for the B, R, E, T quantity kinds."""

from jaxfne.hdp_rule import QuantityKind


def test_quantity_kind_members_are_b_r_e_t():
    assert {m.value for m in QuantityKind} == {"B", "R", "E", "T"}


def test_quantity_kind_lookup_returns_member():
    assert QuantityKind("R") is QuantityKind.R


def test_quantity_kind_value_equals_name():
    for member in QuantityKind:
        assert member.value == member.name


def test_quantity_kind_importable_from_module():
    import jaxfne.hdp_rule as hdp_rule_module

    assert hdp_rule_module.QuantityKind is QuantityKind
