"""v0.3.29 Signals.get() and get_signal() tests.

Signals.get() resolves key aliases, applies selectors to neuron-indexed signals
only, rejects laminar field readouts for selection, rejects an ambiguous trial
argument, and requires neuron_metadata for selection rather than guessing.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.core import _SIGNALS_GET_KEY_ALIASES


def _sim(config_fn=None, **sim_kwargs):
    cfg_fn = config_fn or jtfne.suite2_four_celltype_config
    cfg = cfg_fn(seed=0)
    model = jtfne.construct(cfg)
    sim_kwargs.setdefault("duration_ms", 10)
    sim_kwargs.setdefault("dt_ms", 0.1)
    sim_kwargs.setdefault("seed", 0)
    sig = jtfne.simulate(model, **sim_kwargs)
    return model, sig


def test_get_vm_aliases():
    _, sig = _sim()
    base = np.asarray(sig.get("V_m"))
    assert np.array_equal(np.asarray(sig.get("vm")), base)
    assert np.array_equal(np.asarray(sig.get("voltage")), base)


def test_get_spike_aliases():
    _, sig = _sim()
    base = np.asarray(sig.get("spikes"))
    assert np.array_equal(np.asarray(sig.get("spk")), base)
    assert np.array_equal(np.asarray(sig.get("raster")), base)


def test_get_unknown_key_raises_keyerror_with_available():
    _, sig = _sim()
    with pytest.raises(KeyError) as exc:
        sig.get("does_not_exist")
    assert "Available" in str(exc.value)


def test_probe_only_aliases_normalize_to_proxy_names():
    # `*_like` is fully retired (no aliases): proxy vocabulary only.
    assert _SIGNALS_GET_KEY_ALIASES["emm"] == "emm_proxy"
    assert _SIGNALS_GET_KEY_ALIASES["eeg"] == "eeg_proxy"
    assert _SIGNALS_GET_KEY_ALIASES["meg"] == "meg_proxy"
    for retired in ("eeg_like", "meg_like", "emm_like", "lfp_like", "csd_like"):
        assert retired not in _SIGNALS_GET_KEY_ALIASES


@pytest.mark.parametrize(
    ("key", "resolved"),
    [
        ("eeg", "eeg_proxy"),
        ("meg", "meg_proxy"),
        ("emm", "emm_proxy"),
    ],
)
def test_probe_only_aliases_fail_explicitly_when_unavailable(key, resolved):
    """EEG/MEG/EMM are not on core FieldOutput; aliases must not synthesize data."""
    _, sig = _sim()
    with pytest.raises(KeyError, match=resolved):
        sig.get(key)


@pytest.mark.parametrize("retired", ["lfp_like", "csd_like", "eeg_like", "meg_like", "emm_like"])
def test_retired_like_keys_raise(retired):
    """`*_like` signal keys are fully retired and must not resolve."""
    _, sig = _sim()
    with pytest.raises(KeyError):
        sig.get(retired)


def test_get_trial_raises_not_implemented():
    _, sig = _sim()
    with pytest.raises(NotImplementedError, match="trial"):
        sig.get("vm", trial=0)


def test_get_mixed_selector_style_raises():
    _, sig = _sim()
    with pytest.raises(ValueError, match="not both"):
        sig.get("vm", selector=jtfne.SelectorSpec(cell_type="E"), area="V1")


def test_get_as_numpy_returns_ndarray():
    _, sig = _sim()
    arr = sig.get("vm", as_numpy=True)
    assert isinstance(arr, np.ndarray)


def test_get_with_selector_slices_neuron_axis():
    model, sig = _sim()
    idx = np.asarray(model.select(cell_type="E"))
    vm_e = sig.get("vm", cell_type="E")
    assert vm_e.shape[-1] == len(idx)
    assert np.allclose(np.asarray(vm_e), np.asarray(sig.V_m)[..., idx])


def test_get_with_selectorspec_object():
    model, sig = _sim()
    spec = jtfne.SelectorSpec(cell_type="E")
    vm_e = sig.get("vm", selector=spec)
    idx = np.asarray(model.select(cell_type="E"))
    assert vm_e.shape[-1] == len(idx)


def test_get_selector_on_field_readout_raises():
    """Laminar field readouts have no neuron axis; selectors must be rejected."""
    _, sig = _sim()
    if sig.field is None:
        pytest.skip("field not recorded in this run")
    with pytest.raises(ValueError, match="no declared neuron axis"):
        sig.get("lfp", cell_type="E")


def test_get_field_readout_without_selector_ok():
    _, sig = _sim()
    if sig.field is None:
        pytest.skip("field not recorded in this run")
    lfp = sig.get("lfp")
    assert np.all(np.isfinite(np.asarray(lfp)))


def test_get_sources_without_record_raises():
    """Requesting sources when not recorded raises a clear ValueError."""
    _, sig = _sim(record_sources=False)
    if sig.sources is not None:
        pytest.skip("sources recorded by default in this config")
    with pytest.raises(ValueError, match="sources not recorded"):
        sig.get("sources")


def test_get_selector_requires_neuron_metadata():
    """When neuron_metadata is absent, selection raises rather than guessing."""
    _, sig = _sim(config_fn=jtfne.suite2_single_neuron_config)
    if sig.metadata.get("neuron_metadata") is not None:
        pytest.skip("this config provides neuron_metadata")
    with pytest.raises(ValueError, match="neuron_metadata"):
        sig.get("vm", cell_type="E")


def test_get_signal_free_function_delegates():
    _, sig = _sim()
    assert np.array_equal(
        np.asarray(jtfne.get_signal(sig, "vm")), np.asarray(sig.V_m)
    )


def test_get_signal_type_guard():
    with pytest.raises(TypeError, match="Signals"):
        jtfne.get_signal({"vm": 1}, "vm")
