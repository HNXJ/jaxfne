"""Example 05: v0.0.7 dataset and bridge schema manifest smoke."""

import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jaxfne as jtfne
from jaxfne.bridges import JaxleyEmitterBridge


def main():
    paradigm = jtfne.standard_visual_omission()
    condition_map = {c.name: list(c.condition_numbers) for c in paradigm.conditions}
    dataset = (
        jtfne.dataset_spec(
            name="sequential_visual_omission_schema",
            modality="SPK_MUAe_LFP_BHV",
            source_format="NWB_or_MonkeyLogic_export",
            alignment_label="p1",
            alignment_code=101,
            sampling_rate_hz=1000.0,
            units="declared_by_modality",
            trial_filter={"TrialError": 0},
        )
        .with_condition_map(condition_map)
        .with_quality_gate("correct_trials_only", True)
    )
    bridge = JaxleyEmitterBridge(morphology="future_compartment_template").to_spec().to_dict()
    payload = {
        "paradigm": paradigm.to_dict(),
        "dataset": dataset.to_dict(),
        "bridge": bridge,
        "truth_mode": "truth_safe_unverified",
        "mechanism_claim_status": "not_claimed",
    }
    print("dataset_status:", payload["dataset"]["dataset_status"])
    print("bridge_status:", payload["bridge"]["status"])
    print("AAAB condition numbers:", payload["dataset"]["condition_map"]["AAAB"])
    json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    main()
