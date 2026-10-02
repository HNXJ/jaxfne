"""0.5.3 item 6 (ENGINE W2): causal intervention grammar (H4/H5/H8).

One declarative object = identical realized network + one-mechanism
intervention + declared observation difference. In-memory behavior and
serialization roundtrip tested separately (H4); invalid grammars refused
(H5); d_H=1 default throughout (H8).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.intervene import (
    Intervention,
    intervention_from_manifest,
    network_spec_digest,
    run_intervention,
)

BASE_HP = {
    "K_HDP": 0.05,
    "K_ctrl": 0.15,
    "K_w_ctrl": 0.002,
    "alpha": 0.05,
    "tau_0_ms": 5.0,
    "noise_scale": 0.0,
}
NET = {
    "builder": "suite2_net1",
    "build": {"seed": 1, "n": 8},
    "run": {"duration_ms": 10.0, "dt_ms": 0.5, "seed": 4},
    "hdp_params": dict(BASE_HP),
}
N_EDGES = 56


def _disable_iv(**over):
    kw = dict(
        name="iv-disable",
        network={k: (dict(v) if isinstance(v, dict) else v) for k, v in NET.items()},
        control={"mechanism": "plasticity", "mode": "disable", "mask": None, "value": None},
        observation={
            "signal": "w_final",
            "subset": "all",
            "expect": "different",
            "tolerance": None,
        },
    )
    kw.update(over)
    return Intervention(**kw)


# --- execution: declared differences ------------------------------------------------


def test_disable_w_final_differs_PASS():
    res = run_intervention(_disable_iv())
    assert res["verdict"] == "PASS"
    assert res["spec_digest"] == network_spec_digest(_disable_iv())


def test_allplastic_clamp_equal_PASS_and_wrong_declaration_FAIL():
    mask = [1.0] * N_EDGES
    eq = _disable_iv(
        name="iv-eq",
        control={"mechanism": "plasticity", "mode": "clamp", "mask": mask, "value": 0.3},
        observation={"signal": "w_final", "subset": "all", "expect": "equal", "tolerance": None},
    )
    assert run_intervention(eq)["verdict"] == "PASS"
    ne = _disable_iv(
        name="iv-ne",
        control={"mechanism": "plasticity", "mode": "clamp", "mask": mask, "value": 0.3},
        observation={
            "signal": "w_final",
            "subset": "all",
            "expect": "different",
            "tolerance": None,
        },
    )
    assert run_intervention(ne)["verdict"] == "FAIL"


def test_clamp_subset_differs_PASS():
    mask = [0.0] * (N_EDGES // 2) + [1.0] * (N_EDGES - N_EDGES // 2)
    iv = _disable_iv(
        name="iv-clamp",
        control={"mechanism": "plasticity", "mode": "clamp", "mask": mask, "value": 0.3},
        observation={
            "signal": "w_final",
            "subset": "clamped",
            "expect": "different",
            "tolerance": None,
        },
    )
    res = run_intervention(iv)
    assert res["verdict"] == "PASS"
    w_int = np.asarray(res["intervened"]["observed"]["values"])
    assert np.array_equal(w_int, np.full(w_int.shape, np.float32(0.3)))


def test_hdp_column_builder_runs():
    net = {k: (dict(v) if isinstance(v, dict) else v) for k, v in NET.items()}
    net["builder"] = "hdp_column"
    net["build"] = {"n_neurons": 8, "duration_ms": 10.0, "dt_ms": 0.5, "seed": 6}
    iv = Intervention(
        name="iv-col",
        network=net,
        control={"mechanism": "plasticity", "mode": "disable", "mask": None, "value": None},
        observation={
            "signal": "w_final",
            "subset": "all",
            "expect": "different",
            "tolerance": None,
        },
    )
    assert run_intervention(iv)["verdict"] == "PASS"


# --- arm manifests: first-class, additive-only ---------------------------------------


def test_arm_manifests_carry_intervention_additively():
    res = run_intervention(_disable_iv())
    for arm in ("baseline", "intervened"):
        m = res[arm]["manifest"]
        assert m["intervention"]["arm"] == arm
        assert m["intervention"]["network"]["spec_digest"] == res["spec_digest"]
        assert m["intervention"]["name"] == "iv-disable"
        assert "backend_metadata" in m  # standard surface intact


def test_plain_manifest_has_no_intervention_key():
    model = jtfne.construct(jtfne.suite2_net1_config(seed=1, n=8))
    sig = jtfne.simulate(
        model,
        duration_ms=4.0,
        dt_ms=0.5,
        seed=4,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(BASE_HP)),
    )
    assert "intervention" not in model.manifest(sig)


# --- H4: serialization boundary --------------------------------------------------------


def test_manifest_json_roundtrip_reproduces_object():
    iv = _disable_iv()
    m = iv.to_manifest()
    restored = intervention_from_manifest(json.loads(json.dumps(m)))
    assert restored == iv
    assert restored.to_manifest() == m


def test_result_verdict_preserved_in_manifest_record():
    res = run_intervention(_disable_iv())
    assert res["verdict"] == "PASS"
    for arm in ("baseline", "intervened"):
        assert res[arm]["manifest"]["intervention"]["observation"]["expect"] == "different"


# --- H5: grammar refusals -----------------------------------------------------------------


def test_second_mechanism_refused():
    with pytest.raises(ValueError, match="exactly one mechanism"):
        _disable_iv(control={"mechanism": "drive", "mode": "disable", "mask": None, "value": None})


def test_control_extra_key_refused():
    with pytest.raises(ValueError, match="unknown keys"):
        _disable_iv(
            control={
                "mechanism": "plasticity",
                "mode": "disable",
                "mask": None,
                "value": None,
                "strength": 2.0,
            }
        )


def test_manifest_extra_key_refused():
    m = _disable_iv().to_manifest()
    m["extra"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        intervention_from_manifest(m)


def test_unknown_builder_refused():
    net = dict(NET)
    net["builder"] = "cortical_sheet"
    with pytest.raises(ValueError, match="unknown network builder"):
        _disable_iv(network=net)


def test_subset_without_mask_refused():
    with pytest.raises(ValueError, match="needs signal 'w_final' and a control mask"):
        _disable_iv(
            observation={
                "signal": "w_final",
                "subset": "clamped",
                "expect": "different",
                "tolerance": None,
            }
        )


def test_mask_length_mismatch_refused_before_running():
    iv = _disable_iv(
        control={"mechanism": "plasticity", "mode": "disable", "mask": [1.0, 0.0], "value": None},
    )
    with pytest.raises(ValueError, match="does not match realized edge count"):
        run_intervention(iv)


def test_baseline_mask_refused():
    net = {k: (dict(v) if isinstance(v, dict) else v) for k, v in NET.items()}
    net["hdp_params"] = dict(BASE_HP, plasticity_mask=[1.0] * N_EDGES)
    with pytest.raises(ValueError, match="must not carry a plasticity_mask"):
        run_intervention(_disable_iv(network=net))


def test_clamp_requires_mask_and_finite_value():
    with pytest.raises(ValueError, match="clamp requires a mask"):
        _disable_iv(
            control={"mechanism": "plasticity", "mode": "clamp", "mask": None, "value": 0.3}
        )
    with pytest.raises(ValueError, match="finite"):
        _disable_iv(
            control={
                "mechanism": "plasticity",
                "mode": "clamp",
                "mask": [1.0] * N_EDGES,
                "value": float("nan"),
            }
        )
