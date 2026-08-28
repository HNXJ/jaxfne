"""Regression test for canonical compact summary Θ=Θ_static⊕X⊕H⊕W.

Δscience=0, no overhead when unused, uses existing API only.
"""
import json

import jaxfne as jtfne


def test_canonical_compact_summary_off_hot_path():
    # no kernel change, no overhead when unused — summary is not called in construct/simulate hot path
    assert hasattr(jtfne.util, "canonical_compact_summary")
    assert hasattr(jtfne, "canonical_compact_summary")
    cfg = jtfne.default_cortical_column_config(n=10, duration_ms=10, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sigs_before = model.simulate(jtfne.Simulation(duration_ms=10, dt_ms=0.5, seed=0))
    summ = jtfne.canonical_compact_summary(model, sigs_before)
    sigs_after = model.simulate(jtfne.Simulation(duration_ms=10, dt_ms=0.5, seed=0))
    # Δscience=0: simulate output unchanged by summary call
    assert float(sigs_before.V_m[0, 0]) == float(sigs_after.V_m[0, 0])
    # JSON-safe
    json.dumps(summ, allow_nan=False)
    # Theta checks: N=10 → N_static=6*N+1=61, X per_step=20, canonical_4N=40
    assert summ["N_static"] == 6 * 10 + 1
    assert summ["Theta"]["X"]["per_step"] == 20
    assert summ["Theta"]["X"]["canonical_4N"] == 40
    assert summ["output_basis"]["STATE"]["V_m"]["shape"] == [20, 10]
    assert "STATE" in summ["output_basis"] and "SOURCE" in summ["output_basis"]
    assert "FIELD" in summ["output_basis"] and "PROBE" in summ["output_basis"] and "DERIVED" in summ["output_basis"]
    # text bundle present and like jaxfne summary
    assert "Θ=Θ_static" in summ["text_bundle"]
    assert summ["Δscience"] == 0


def test_canonical_1000n_counts():
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    model = jtfne.construct(
        tensor, jtfne.neuronal_tensor.RuntimeConfiguration(duration_ms=1000, dt_ms=0.5, seed=0)
    )
    sigs = model.simulate(jtfne.Simulation(duration_ms=1000, dt_ms=0.5, seed=0))
    summ = jtfne.canonical_compact_summary(model, sigs, tensor)
    # configured vs realized vs effective
    assert summ["counts"]["configured"]["n_neurons"] == 1000
    assert summ["counts"]["realized"]["n_neurons"] == 1000
    assert summ["counts"]["effective"]["n_neurons"] == 1000
    # populations: declared 23 detailed, effective EI-collapsed 12 matches task example
    assert summ["counts"]["realized"]["n_populations_detailed"] == 23
    assert summ["counts"]["realized"]["populations_inventory"]["effective_EI_collapsed"] == 12
    # edges: 48 rules → realized 215785 (p=1.0 bipartite); task ~79k is illustrative with sparser p
    assert summ["counts"]["configured"]["n_connection_rules"] == 48
    assert summ["counts"]["realized"]["n_edges"] == 215785
    assert summ["counts"]["effective"]["n_steps"] == 2000  # 1000ms/0.5ms
    assert summ["counts"]["effective"]["dt_ms"] == 0.5
    # Theta decomposition
    assert summ["Theta"]["Theta_static"]["N_static"] == 6001  # 6*1000+1
    assert summ["Theta"]["W"]["n_edges"] == 215785
    assert summ["Theta"]["H"]["size"] == 0  # HDP disabled
    assert summ["Theta"]["X"]["per_step"] == 2000
    assert summ["Theta"]["X"]["canonical_4N"] == 4000
    # provenance uses existing API only
    assert summ["provenance"]["config_hash"] is not None
    assert summ["provenance"]["tensor_identity"] is not None
    # output basis minimal independent, not flattened
    assert summ["output_basis"]["STATE"]["V_m"]["shape"] == [2000, 1000]
    assert summ["output_basis"]["FIELD"]["lfp_proxy"]["shape"] == [2000, 16]
    json.dumps(summ, allow_nan=False)
