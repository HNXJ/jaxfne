"""Example 04: v0.0.6 black-box tuning loop smoke.

This example runs a tiny CPU-safe random-search loop over one source-scale
parameter. It is a computational scaffold only: no biological, calibrated, or
mechanistic claim is made from the selected candidate.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jaxfne as jtfne


def main():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    obj = jtfne.objective().loss("rate", target=10.0, metric="spike_rate_hz_mean")
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=3)
    tuned, report = model.tune(
        obj,
        optimizer=jtfne.random_search(),
        steps=3,
        seed=3,
        simulation=sim,
        parameter="source_scale",
        bounds=(0.5, 1.5),
    )
    print("tuning_status:", report["tuning_status"])
    print("candidate_count:", len(report["candidate_history"]))
    print("best_parameter_value:", report["best_parameter_value"])
    print("physical_amplitude_claim_allowed:", report["physical_amplitude_claim_allowed"])
    print("mechanism_claim_status:", report["mechanism_claim_status"])
    json.dumps(report, allow_nan=False)
    assert tuned is not None


if __name__ == "__main__":
    main()
