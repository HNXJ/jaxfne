"""0.5.5 ENGINE item 2b: per-AT in-memory data bundles.

Each canonical runner in ``artifacts/atlas/at_manifest.py:REGISTRY``
accepts ``keep_bundle`` (default False: output unchanged) and returns the
executed ``model``/``signals`` beside the summary under ``"bundle"`` when
True. ``artifacts/atlas/at_bundle.py:bundle`` is the consumer entry point.
"""

import numpy as np
import pytest

from artifacts.atlas import at_bundle
from artifacts.atlas.at_manifest import REGISTRY, resolve_runner

# Cost keys vary run to run; everything else must reproduce exactly at
# fixed seeds (same idea as tests/test_atlas_manifest_055.py:12-24).
_COST_KEYS = ("wall_s", "mem_peak_b")


def _without_cost(node):
    if isinstance(node, dict):
        return {k: _without_cost(v) for k, v in node.items() if k not in _COST_KEYS}
    if isinstance(node, list):
        return [_without_cost(v) for v in node]
    return node


def _summary_only(out):
    return _without_cost({k: v for k, v in out.items() if k != "bundle"})


def _check_summary_unchanged(at_id):
    run = resolve_runner(at_id)
    with_bundle = run(keep_bundle=True)
    assert "bundle" in with_bundle, at_id
    assert set(with_bundle["bundle"]) == set(at_bundle.BUNDLE_ARM_NAMES[at_id]), at_id
    plain = run(keep_bundle=False)
    assert "bundle" not in plain, at_id
    assert _summary_only(with_bundle) == _summary_only(plain), at_id
    return with_bundle["bundle"]


def _check_bundle_arms(at_id, b):
    assert len(b) >= 1, at_id
    for arm_name, arm in b.items():
        model, sig = arm["model"], arm["signals"]
        spikes = np.asarray(sig.spikes)
        assert spikes.ndim == 2, (at_id, arm_name, spikes.shape)
        n = len(model.neuron_table())
        assert spikes.shape == (spikes.shape[0], n), (at_id, arm_name, spikes.shape, n)
        vm = np.asarray(sig.V_m)
        assert vm.shape[0] == spikes.shape[0] and vm.shape[-1] == n, (
            at_id,
            arm_name,
            vm.shape,
            n,
        )


_CHEAP_ATS = ("AT-02", "AT-05", "AT-10")


@pytest.mark.parametrize("at_id", _CHEAP_ATS)
def test_bundle_request_leaves_summary_unchanged(at_id):
    _check_summary_unchanged(at_id)


@pytest.mark.parametrize("at_id", _CHEAP_ATS)
def test_bundle_arms_hold_model_and_signals(at_id):
    _check_bundle_arms(at_id, at_bundle.bundle(at_id))


def test_bundle_fields_contract():
    assert set(at_bundle.BUNDLE_FIELDS) == {
        "signals.spikes",
        "signals.V_m",
        "signals.sources",
        "signals.field",
        "hdp",
    }
    assert set(at_bundle.BUNDLE_ARM_NAMES) == set(REGISTRY)
    with pytest.raises(KeyError):
        at_bundle.bundle("AT-99")


@pytest.mark.slow
def test_shared_model_arms_keep_their_own_hdp():
    # AT-08 arms share one Model, whose last_hdp_diagnostics() is the last
    # arm's; each bundle arm must carry the diagnostics of its own run.
    b = at_bundle.bundle("AT-08")
    assert b["fixed"]["hdp"] is None
    full, local = b["adapt_full"]["hdp"], b["adapt_local"]["hdp"]
    assert not np.array_equal(full["w_final"], local["w_final"])
    last = b["adapt_local"]["model"].last_hdp_diagnostics()
    assert not np.array_equal(full["w_final"], np.asarray(last["w_final"]))


@pytest.mark.slow
@pytest.mark.parametrize("at_id", sorted(REGISTRY))
def test_bundle_request_leaves_summary_unchanged_all(at_id):
    b = _check_summary_unchanged(at_id)
    _check_bundle_arms(at_id, b)
