"""AUG-4 small augmentation suite: ladder, commutation, stochastic replay.

One test per behaviour; fixtures mirror tests/test_augment.py. Every
comparison that matters is checked on the realized model (construct ->
simulate), not only on the tensor.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.augment import (
    AugmentationSpec,
    GeometryTransform,
    PoseEdit,
    ScaleN,
    ThetaC,
    ThetaX,
    W0,
    H0,
    augment,
)


def _tiny_tensor(n=10):
    layer = nt.Layer(
        name="L4",
        n_neurons=n,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
    )
    area = nt.Area(name="V1", layers=[layer], inter_connections=[conn])
    return nt.NeuronalTensor(areas=[area], name="suite_tiny")


def _build(tensor, seed=0):
    return nt.construct_neuronal_tensor(tensor, seed=seed, duration_ms=20.0, dt_ms=0.5)


def _rule_weights(model):
    return [float(rule["weight"]) for rule in model.cfg.metadata["circuit"]["connections"]]


def _neuron_count(model):
    return int(model.params["emitter"].n_neurons)


def _mechanism_taus(model):
    return [
        (mech["name"], float(mech["params"]["tau_ms"]))
        for mech in model.cfg.metadata["circuit"]["mechanisms"]
    ]


def test_ladder_n10_100_1000_configured_realized_executed():
    base = jtfne.make_minimal_ei_tensor(n=10)
    assert sum(layer.n_neurons for area in base.areas for layer in area.layers) == 10
    rungs = {}
    for name, factor, expect in (("n100", 10.0, 100), ("n1000", 100.0, 1000)):
        out, record = augment(base, AugmentationSpec(transforms=[ScaleN(factor)]))
        assert record.realized_order == ("N",)
        # Configured: tensor layer counts sum to the rung size.
        total = sum(layer.n_neurons for area in out.areas for layer in area.layers)
        assert total == expect
        model = _build(out, seed=0)
        # Realized: the constructed model holds that many neurons.
        assert _neuron_count(model) == expect
        assert len(model.neuron_table()) == expect
        # Executed: spikes have one column per neuron and are finite.
        signals = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
        spikes = jnp.asarray(signals.spikes)
        assert spikes.shape[1] == expect
        assert bool(jnp.all(jnp.isfinite(spikes)))
        rungs[name] = _rule_weights(model)
    base_weights = _rule_weights(_build(base, seed=0))
    assert len(base_weights) == len(rungs["n100"]) == len(rungs["n1000"]) > 0
    # Realized edge weight follows w/sqrt(N): the same declared connection
    # gives weight(N1)/weight(N0) == sqrt(N0/N1) at both rungs.
    for weights, n1 in ((rungs["n100"], 100), (rungs["n1000"], 1000)):
        for w0, w1 in zip(base_weights, weights):
            assert w1 / w0 == pytest.approx(math.sqrt(10 / n1))


def _sequential(tensor, first, second):
    """Apply two single-record specs one at a time, in the given order."""
    mid, _ = augment(tensor, AugmentationSpec(transforms=[first]))
    return augment(mid, AugmentationSpec(transforms=[second]))


def _assert_combined_equals_sequential(tensor, first, second, sequential_out):
    """The single combined spec, listed in either order, equals ``sequential_out``."""
    both, rec_both = augment(tensor, AugmentationSpec(transforms=[first, second]))
    swapped, rec_swapped = augment(tensor, AugmentationSpec(transforms=[second, first]))
    assert both.to_dict() == sequential_out.to_dict()
    assert swapped.to_dict() == sequential_out.to_dict()
    assert rec_both.spec_digest == rec_swapped.spec_digest
    return both


def test_commute_n_g_positions_and_counts_equal():
    tensor = _tiny_tensor(n=10)
    scale = ScaleN(10.0)
    geom = GeometryTransform(pose_edits=[PoseEdit(area="V1", translation=(1.0, 2.0, 3.0))])
    ab, _ = _sequential(tensor, scale, geom)
    ba, _ = _sequential(tensor, geom, scale)
    # Both edits took effect (a double no-op would also compare equal).
    assert ab.areas[0].layers[0].n_neurons == 100
    assert ab.areas[0].pose.translation == (1.0, 2.0, 3.0)
    assert ab.to_dict() == ba.to_dict()
    model_ab = _build(ab, seed=0)
    model_ba = _build(ba, seed=0)
    # Realized model quantities, not only the tensor.
    assert _neuron_count(model_ab) == _neuron_count(model_ba) == 100
    assert bool(jnp.array_equal(model_ab.params["positions"], model_ba.params["positions"]))
    assert model_ab.cfg.metadata["circuit"]["connections"] == (
        model_ba.cfg.metadata["circuit"]["connections"]
    )
    # The pose edit moved realized positions relative to the N-only model.
    model_n, _ = augment(tensor, AugmentationSpec(transforms=[scale]))
    shift = model_ab.params["positions"] - _build(model_n, seed=0).params["positions"]
    assert bool(jnp.all(jnp.isfinite(shift)))
    assert float(jnp.abs(shift).max()) > 0.0
    _assert_combined_equals_sequential(tensor, scale, geom, ab)


def test_commute_n_w0_realized_weights_equal():
    tensor = _tiny_tensor(n=10)
    scale = ScaleN(10.0)
    gain = W0(factor=1.5)
    ab, _ = _sequential(tensor, scale, gain)
    ba, _ = _sequential(tensor, gain, scale)
    # The gain took effect on the tensor (a double no-op would also compare equal).
    assert ab.areas[0].inter_connections[0].plastic.w_mech == pytest.approx(3.0)
    assert ab.to_dict() == ba.to_dict()
    weights_ab = _rule_weights(_build(ab, seed=0))
    weights_ba = _rule_weights(_build(ba, seed=0))
    # Realized weights are equal across orders, and carry both transforms:
    # 1.5x the N-only model's weights.
    assert weights_ab == pytest.approx(weights_ba)
    model_n, _ = augment(tensor, AugmentationSpec(transforms=[scale]))
    for scaled, plain in zip(weights_ab, _rule_weights(_build(model_n, seed=0))):
        assert scaled == pytest.approx(1.5 * plain)
    _assert_combined_equals_sequential(tensor, scale, gain, ab)


def test_commute_g_thetax_tau_and_geometry_equal():
    tensor = _tiny_tensor(n=10)
    geom = GeometryTransform(pose_edits=[PoseEdit(area="V1", translation=(1.0, 2.0, 3.0))])
    taux = ThetaX(tau_factor=2.0)
    ab, _ = _sequential(tensor, geom, taux)
    ba, _ = _sequential(tensor, taux, geom)
    # Both edits took effect (a double no-op would also compare equal).
    assert ab.areas[0].pose.translation == (1.0, 2.0, 3.0)
    assert ab.areas[0].inter_connections[0].static.dT_ms == pytest.approx(4.0)
    assert ab.to_dict() == ba.to_dict()
    model_ab = _build(ab, seed=0)
    model_ba = _build(ba, seed=0)
    # Realized tau and geometry are equal across orders.
    assert _mechanism_taus(model_ab) == _mechanism_taus(model_ba)
    assert [tau for _, tau in _mechanism_taus(model_ab)] == [4.0]
    assert [tau for _, tau in _mechanism_taus(model_ba)] == [4.0]
    assert bool(jnp.array_equal(model_ab.params["positions"], model_ba.params["positions"]))
    _assert_combined_equals_sequential(tensor, geom, taux, ab)


_REPLAY_SCRIPT = """\
import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.augment import (
    AugmentationSpec, GeometryTransform, PoseEdit, ScaleN,
    ThetaC, ThetaX, W0, H0, augment,
)

def _layer():
    return nt.Layer(
        name="L4", n_neurons=10,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )

def _conn(delay):
    return nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
        delay_ms=delay,
    )

tensor = nt.NeuronalTensor(
    areas=[
        nt.Area(name="A0", layers=[_layer()], inter_connections=[_conn(1.0)]),
        nt.Area(name="A1", layers=[_layer()], inter_connections=[_conn(1.0)]),
    ],
    area_connections=[nt.AreaConnection(
        source_area="A0", source_layer="L4", source_neuron_type="E",
        target_area="A1", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=0.5),
        delay_ms=2.0, probability=0.5,
    )],
    name="suite_replay",
)
spec = AugmentationSpec(
    transforms=[
        ScaleN(2.0),
        GeometryTransform(pose_edits=[PoseEdit(area="A1", translation=(1.0, 2.0, 3.0))]),
        ThetaC(delay_factor=1.0, delay_jitter=0.2,
              probability_factor=1.0, probability_jitter=0.1),
        ThetaX(g_factor=1.0, g_jitter=0.2, tau_factor=1.0, tau_jitter=0.1),
        W0(factor=1.0, jitter=0.2),
        H0(offset=0.5, jitter=0.1),
    ],
    k_v=11,
)
out, record = augment(tensor, spec)
print(record.spec_digest)
print(repr([(e.address, e.axis, e.before, e.after, e.origin) for e in record.changes]))
"""


def _run_replay_script(script_path):
    import jaxfne

    root = Path(jaxfne.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=300,
    )


def test_stochastic_six_axis_replay_bit_identical_across_processes(tmp_path):
    script = tmp_path / "_temp_aug_suite_replay.py"
    script.write_text(_REPLAY_SCRIPT, encoding="utf-8")
    try:
        first = _run_replay_script(script)
        assert first.returncode == 0, (
            f"replay script failed\nstdout:\n{first.stdout}\nstderr:\n{first.stderr}"
        )
        second = _run_replay_script(script)
        assert second.returncode == 0, (
            f"replay script failed\nstdout:\n{second.stdout}\nstderr:\n{second.stderr}"
        )
    finally:
        script.unlink(missing_ok=True)
    assert not script.exists()
    # Bit-identical changes and digest across the two processes.
    assert first.stdout == second.stdout
    digest, changes_repr = first.stdout.strip().splitlines()
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    # Genuinely six-axis: every axis realized at least one change, and the
    # stochastic axes drew under K_V.
    for axis in ("N", "G", "Theta_C", "Theta_X", "W_0", "H_0"):
        assert f"'{axis}'" in changes_repr, f"axis {axis} realized no change"
    assert "augment-sampled" in changes_repr
