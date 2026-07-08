"""Tests for Model/NeuronalTensor-targeted paradigm helpers."""

import jaxfne as jtfne


def test_paradigm_target_indices_from_model_suite2():
    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    model = jtfne.construct(cfg)
    idx = jtfne.paradigm_target_indices_from_model(model, cell_type="E")
    assert len(idx) >= 1
    all_e = model.select(cell_type="E")
    assert set(idx) == set(int(i) for i in all_e.tolist())


def test_coop_omission_oddball_for_model_sets_target_indices():
    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    model = jtfne.construct(cfg)
    p = jtfne.coop_omission_oddball_for_model(
        model, target_cell_type="PV", duration_ms=500.0, seed=1,
    )
    cond = p.conditions[0]
    drive_events = [e for e in cond.events if e.metadata.get("drive_amplitude", 0) != 0]
    assert drive_events
    targets = drive_events[0].metadata.get("target_indices")
    assert targets is not None
    assert set(targets) == set(jtfne.paradigm_target_indices_from_model(model, cell_type="PV"))


def test_coop_omission_oddball_for_neuronal_tensor_smoke():
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    p = jtfne.coop_omission_oddball_for_neuronal_tensor(
        tensor,
        target_layer="L4",
        target_cell_type="E",
        construct_duration_ms=50.0,
        duration_ms=100.0,
        seed=2,
    )
    assert p.name == "coop_omission_oddball"
    assert len(p.conditions) == 1
