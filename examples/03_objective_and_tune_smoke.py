"""Example 03: objective evaluation and tune metadata smoke.

Demonstrates the v0.0.5-P2/P3 Objective/evaluate/tune API.  The simulation
uses a small network (n=8, 5ms) so the example runs quickly on CPU.
No Optax is required: the tuning smoke uses the GSDR blackbox path.

Model.tune() in v0.0.5 is a metadata-only scaffold:
  - No parameter mutation occurs.
  - The returned model is the same object (same_model is model).
  - tuning_status = 'metadata_only_v0.0.5'

Scientific truth status:
  truth_mode: truth_safe_unverified
  claim_level: computational_scaffold
  field_claim_level: proxy_readout_only
  physical_amplitude_claim_allowed: false
  empirical_validation_status: not_empirically_validated
  mechanism_claim_status: not_claimed

Gate pass/fail is a computational diagnostic only.  It does not imply
empirical validation or biological mechanism proof.
"""

import json

import jaxfne as jtfne


def main():
    # --- Build a small model ---
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)

    # --- Simulate (CPU-safe, 5ms) ---
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])

    print("=== Simulation ===")
    print(f"  V_m shape: {signals.V_m.shape}")
    print(f"  CSD proxy shape: {signals.field.csd_proxy.shape}")

    # --- Build objective ---
    paradigm = jtfne.standard_visual_omission()
    aaax_meta = {"condition": "AAAX", "omission_position": "p4"}

    obj = (
        jtfne.Objective(name="omission_scaffold_v005")
        .loss("spike_rate_target", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .regularizer("vm_reg", target=-65.0, weight=0.1, metric="mean_V_m")
        .gate("rate_gate", threshold=500.0, criterion="below",
              metric="spike_rate_hz_mean", metadata=aaax_meta)
    )

    print("\n=== Objective ===")
    print(f"  name: {obj.name}")
    print(f"  losses: {[l['name'] for l in obj.losses]}")
    print(f"  regularizers: {[r['name'] for r in obj.regularizers]}")
    print(f"  gates: {[g['name'] for g in obj.gates]}")

    # --- Evaluate ---
    eval_report = model.evaluate(signals, obj)

    print("\n=== Evaluation report ===")
    print(f"  evaluation_status: {eval_report['evaluation_status']!r}")
    print(f"  total_loss: {eval_report['total_loss']}")
    print(f"  all_gates_pass: {eval_report['all_gates_pass']}")
    print(f"  acceptance_decision: {eval_report['acceptance_decision']!r}")
    if eval_report["losses"]:
        lr = eval_report["losses"][0]
        print(f"  loss[spike_rate_target]: value={lr.get('value'):.4f}  "
              f"weighted={lr.get('weighted_value'):.4f}")
    if eval_report["gates"]:
        gr = eval_report["gates"][0]
        print(f"  gate[rate_gate]: value={gr.get('value'):.4f}  pass={gr.get('pass')}")

    # Verify JSON safety
    eval_json = json.dumps(eval_report, allow_nan=False)
    assert isinstance(eval_json, str)

    # --- Tune metadata smoke (GSDR blackbox, strict=False) ---
    same_model, tune_report = model.tune(obj, optimizer="GSDR", steps=1, strict=False)

    print("\n=== Tuning metadata report ===")
    print(f"  tuning_status: {tune_report['tuning_status']!r}")
    print(f"  acceptance_decision: {tune_report['acceptance_decision']!r}")
    print(f"  same_model_unchanged: {tune_report['same_model_unchanged']}")
    print(f"  optimizer: {tune_report['optimizer']['optimizer']!r}")

    # The contract: model is never mutated
    assert same_model is model, "Model must be unchanged after tune()"

    # Verify JSON safety
    tune_json = json.dumps(tune_report, allow_nan=False)
    assert isinstance(tune_json, str)

    # --- Full manifest with v0.0.5 extension fields ---
    full_manifest = model.manifest(
        signals=signals,
        readout=readout,
        paradigm=paradigm.to_dict(),
        objective={"name": obj.name, "losses": obj.losses,
                   "regularizers": obj.regularizers, "gates": obj.gates},
        evaluation=eval_report,
        tuning=tune_report,
    )

    print("\n=== Full manifest (truth gates) ===")
    print(f"  manifest_schema_version: {full_manifest['manifest_schema_version']!r}")
    print(f"  truth_mode: {full_manifest['truth_mode']!r}")
    print(f"  field_claim_level: {full_manifest['source_field_status']['field_claim_level']!r}")
    print(f"  physical_amplitude_claim_allowed: "
          f"{full_manifest['source_field_status']['physical_amplitude_claim_allowed']}")
    print(f"  v005_claim_labels: {full_manifest.get('v005_claim_labels', {})}")

    manifest_json = json.dumps(full_manifest, allow_nan=False)
    assert isinstance(manifest_json, str)

    print("\n=== Scientific truth status ===")
    print("  truth_mode: truth_safe_unverified")
    print("  claim_level: computational_scaffold")
    print("  field_claim_level: proxy_readout_only")
    print("  physical_amplitude_claim_allowed: false")
    print("  empirical_validation_status: not_empirically_validated")
    print("  mechanism_claim_status: not_claimed")
    print("  Gate pass/fail is a computational diagnostic only.")
    print("  Model.tune() in v0.0.5 is a metadata scaffold; no parameters were changed.")


if __name__ == "__main__":
    main()
