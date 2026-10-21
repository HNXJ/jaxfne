"""T-18: one presynaptic sign source (`CELL_TYPE_SIGN`).

The Izhikevich "sign" entries and `LIF_LABEL_SIGN` both derive from
`jaxfne.emitters.CELL_TYPE_SIGN`; unlisted LIF labels are refused (no silent
+1 fallback); the tutorial same-layer rule keeps its PV/SST/VIP scope under
a named constant; the two small-model spike pins below are byte-identical
before/after (baseline sha256 recorded 2026-10-18, CPU, float32 default).
"""

import hashlib

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.emitters import CELL_TYPE_SIGN, IZHIKEVICH_CELL_TYPE_DEFAULTS
from jaxfne.emitters_lif import (
    LIF_LABEL_SIGN,
    lif_network_params_from_labels,
    lif_sign_of,
)
from jaxfne.tutorial_utils import SAME_LAYER_ZEROED_INHIBITORY

DT_MS = 0.5
DUR_MS = 20.0

# Baseline pins (recorded on the pre-T-18 tree; see _temp_pin_baseline.py run).
EXPECTED_IZHI_SPIKES_SHA256 = "a678f7994df54fe5175f415ae2b1979aba2c7a96218fd368127f040e7cb6dccb"
EXPECTED_LIF_SPIKES_SHA256 = "f555501deeaab2c885bcb6e0a18320815b331ef919021af47016ab839ccfcc03"


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _tensor():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        delay_ms=1.0,
    )
    return nt.NeuronalTensor(
        name="t18-pin",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=6,
                neuron_types=[_nt("E"), _nt("PV")])],
            inter_connections=[conn],
        )],
    )


def _spike_sha256(spikes) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(spikes)).tobytes()).hexdigest()


def test_izhi_signs_equal_cell_type_sign_restricted():
    """Every Izhikevich "sign" entry reads the single sign table."""
    for label, row in IZHIKEVICH_CELL_TYPE_DEFAULTS.items():
        assert row["sign"] == CELL_TYPE_SIGN[label] == pytest.approx(row["sign"])
    assert {k for k in CELL_TYPE_SIGN if k in IZHIKEVICH_CELL_TYPE_DEFAULTS} == set(
        IZHIKEVICH_CELL_TYPE_DEFAULTS)


def test_lif_label_sign_equals_cell_type_sign_restricted():
    """`LIF_LABEL_SIGN` is the single table restricted to the LIF-signed labels."""
    assert set(LIF_LABEL_SIGN) == {
        "simple", "complex", "E", "PV", "SST", "VIP", "Inl", "Ing", "I"}
    for label, sign in LIF_LABEL_SIGN.items():
        assert sign == CELL_TYPE_SIGN[label]
    assert CELL_TYPE_SIGN["simple"] == 1.0 and CELL_TYPE_SIGN["complex"] == 1.0


def test_lif_unknown_label_refused():
    """An unlisted LIF label names itself and the known set instead of signing +1."""
    with pytest.raises(ValueError, match=r"'SOM'.*known|expected one of"):
        lif_sign_of("SOM")
    with pytest.raises(ValueError, match=r"'SOM'"):
        lif_network_params_from_labels(["E", "SOM"])


def test_same_layer_zeroing_scope_is_pv_sst_vip_only():
    """The tutorial same-layer rule keeps its stated scope (no Inl/Ing/I)."""
    assert tuple(SAME_LAYER_ZEROED_INHIBITORY) == ("PV", "SST", "VIP")


def test_izhi_small_model_spikes_byte_identical():
    """Byte-identity pin: Izhikevich spikes match the pre-T-18 baseline hash."""
    model = jtfne.construct(
        _tensor(), nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="izhikevich"))
    assert _spike_sha256(jtfne.simulate(model).spikes) == EXPECTED_IZHI_SPIKES_SHA256


def test_lif_small_model_spikes_byte_identical():
    """Byte-identity pin: LIF spikes match the pre-T-18 baseline hash."""
    model = jtfne.construct(
        _tensor(), nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="lif"))
    assert _spike_sha256(jtfne.simulate(model).spikes) == EXPECTED_LIF_SPIKES_SHA256
