"""v0.0.9 edge-list recurrent backend smoke example.

This example compares the dense baseline path with the sparse EdgeList path.
Both remain proxy readout scaffolds: no PDE solve, no calibrated amplitudes,
and no mechanism claim.
"""

import json

import jaxfne as jtfne


def main() -> None:
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8, cell_types={"E": 0.7, "PV": 0.2, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    edge_payload = model.params["edge_list"].to_dict()
    sim = jtfne.simulation(
        duration_ms=5.0,
        dt_ms=0.1,
        seed=9,
        runtime=jtfne.runtime(jit=True, vmap=True, recurrent_backend="edge_list", seed=9),
    )
    signals = model.simulate(sim)
    batch = model.simulate_batch(sim, n_seeds=3)
    summary = signals.summary()
    report = {
        "edge_list": edge_payload,
        "signals_summary": summary,
        "batch_metadata": batch["metadata"],
        "truth_mode": "truth_safe_unverified",
        "field_solver_status": "laminar_proxy_no_pde",
        "field_claim_level": "proxy_readout_only",
        "physical_amplitude_claim_allowed": False,
        "mechanism_claim_status": "not_claimed",
    }
    json.dumps(report, allow_nan=False)
    print("edge backend:", edge_payload["backend"])
    print("n_edges:", edge_payload["n_edges"])
    print("V_m shape:", tuple(signals.V_m.shape))
    print("batch shape:", tuple(batch["V_m"].shape))
    print("field_claim_level:", report["field_claim_level"])
    print("physical_amplitude_claim_allowed:", report["physical_amplitude_claim_allowed"])


if __name__ == "__main__":
    main()
