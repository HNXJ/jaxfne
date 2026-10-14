"""Pin the per-NeuronType emitter declaration registry (Lane 6).

Covers: built-in names, register/get/list roundtrip, refusal of unknown
names, duplicate and malformed registrations, and per-emitter param
validation (adex consumes the AdEx cell fields, lif the LIF cell fields;
izhikevich/glif consume nothing, so any declared params there are refused
as unconsumed).
"""

import pytest

from jaxfne import emitter_registry as reg
from jaxfne.emitters_adex import _PARAM_FIELDS as ADEX_FIELDS
from jaxfne.emitters_lif import _PARAM_FIELDS as LIF_FIELDS


def test_builtin_emitter_names():
    assert reg.list_registered_emitters() == ("adex", "glif", "izhikevich", "lif")


def test_get_roundtrip_and_is_registered():
    for name in ("izhikevich", "lif", "glif", "adex"):
        assert reg.is_registered_emitter(name)
        assert reg.get_emitter(name).name == name
    assert not reg.is_registered_emitter("hodgkin_huxley")
    assert not reg.is_registered_emitter(None)


def test_get_unknown_names_the_known_set():
    with pytest.raises(ValueError, match="known"):
        reg.get_emitter("morris_lecar")


def test_adex_descriptor_consumes_the_adex_cell_fields():
    assert tuple(reg.get_emitter("adex").params) == tuple(ADEX_FIELDS)


def test_check_adex_params_accepts_a_subset_and_returns_a_copy():
    params = {"C_pF": 150.0, "b_pA": 20.0}
    checked = reg.check_emitter_params("adex", params)
    assert checked == params
    assert checked is not params


def test_check_adex_params_none_reads_as_empty():
    assert reg.check_emitter_params("adex", None) == {}


def test_check_adex_params_refuses_unconsumed_keys():
    with pytest.raises(ValueError, match="does not consume"):
        reg.check_emitter_params("adex", {"C_pF": 150.0, "tau_m": 10.0})


def test_lif_descriptor_consumes_the_lif_cell_fields():
    assert tuple(reg.get_emitter("lif").params) == tuple(LIF_FIELDS)


def test_check_lif_params_accepts_a_subset_and_returns_a_copy():
    params = {"tau_m_ms": 18.0, "v_thresh_mV": 10.0}
    checked = reg.check_emitter_params("lif", params)
    assert checked == params
    assert checked is not params


def test_check_lif_params_none_reads_as_empty():
    assert reg.check_emitter_params("lif", None) == {}


def test_check_lif_params_refuses_unconsumed_keys():
    with pytest.raises(ValueError, match="does not consume"):
        reg.check_emitter_params("lif", {"tau_m_ms": 18.0, "C_pF": 150.0})


def test_check_params_refuses_unknown_emitter():
    with pytest.raises(ValueError, match="unknown"):
        reg.check_emitter_params("morris_lecar", None)


def test_check_params_refuses_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        reg.check_emitter_params("adex", [("C_pF", 150.0)])


@pytest.mark.parametrize("family", ["izhikevich", "glif"])
def test_non_adex_emitters_consume_no_params(family):
    assert reg.check_emitter_params(family, None) == {}
    with pytest.raises(ValueError, match="does not consume"):
        reg.check_emitter_params(family, {"C_pF": 150.0})


def test_register_custom_roundtrip():
    name = "test_lane6_custom_emitter"
    assert not reg.is_registered_emitter(name)
    reg.register_emitter(reg.EmitterDescriptor(name=name, params=("gain",)))
    try:
        assert reg.is_registered_emitter(name)
        assert reg.check_emitter_params(name, {"gain": 2.0}) == {"gain": 2.0}
        with pytest.raises(ValueError, match="does not consume"):
            reg.check_emitter_params(name, {"other": 1.0})
    finally:
        reg._REGISTRY.pop(name, None)
    assert not reg.is_registered_emitter(name)


def test_register_refuses_duplicates_empty_names_and_bad_keys():
    with pytest.raises(ValueError, match="already registered"):
        reg.register_emitter(reg.EmitterDescriptor(name="izhikevich"))
    with pytest.raises(ValueError, match="non-empty"):
        reg.register_emitter(reg.EmitterDescriptor(name=""))
    with pytest.raises(ValueError, match="param key"):
        reg.register_emitter(reg.EmitterDescriptor(name="test_lane6_bad", params=("",)))
    with pytest.raises(TypeError, match="EmitterDescriptor"):
        reg.register_emitter("izhikevich")
