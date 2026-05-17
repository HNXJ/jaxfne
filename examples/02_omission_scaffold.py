"""Example 02: standard visual omission task paradigm scaffold.

Demonstrates the standard_visual_omission() paradigm object introduced in
jaxfne v0.0.5-P1.  No simulation is run here — this example shows the
task-flow grammar (conditions, event codes, analysis windows) and verifies
that the full paradigm serializes to strict JSON.

Scientific truth status:
  truth_mode: truth_safe_unverified
  claim_level: computational_scaffold
  mechanism_claim_status: not_claimed
  empirical_validation_status: not_empirically_validated

No biological mechanism is implied by the condition structure.  The paradigm
object is a computational schema for organizing trial types, not a validated
model of neural processing.
"""

import json

import jaxfne as jtfne


def main():
    paradigm = jtfne.standard_visual_omission()

    print("=== Paradigm name ===")
    print(paradigm.name)

    print("\n=== Condition names (12 total) ===")
    print(paradigm.condition_names())

    print("\n=== Omission conditions (9 with omission_position set) ===")
    for cond in paradigm.omission_conditions():
        print(f"  {cond.name:6s}  omission_position={cond.omission_position!r:5s}  "
              f"condition_numbers={list(cond.condition_numbers)}")

    print("\n=== Non-omission conditions ===")
    for cond in paradigm.conditions:
        if not cond.has_omission():
            print(f"  {cond.name:6s}  condition_numbers={list(cond.condition_numbers)}")

    print("\n=== Event codes ===")
    for label, code in paradigm.event_codes.items():
        print(f"  {label:4s}: {code}")

    print("\n=== Analysis windows (ms) ===")
    for window, (lo, hi) in paradigm.analysis_windows.items():
        print(f"  {window:12s}: [{lo}, {hi}]")

    print(f"\n=== Alignment ===")
    print(f"  alignment_label: {paradigm.alignment_label!r}")
    print(f"  alignment_code:  {paradigm.alignment_code}")
    print(f"  pre_stimulus_buffer_ms: {paradigm.pre_stimulus_buffer_ms}")

    print("\n=== Condition detail: AAAX ===")
    aaax = paradigm.condition("AAAX")
    print(f"  name: {aaax.name}")
    print(f"  sequence: {aaax.sequence}")
    print(f"  omission_position: {aaax.omission_position!r}")
    print(f"  condition_numbers: {aaax.condition_numbers}")
    print(f"  events:")
    for evt in aaax.events:
        omission_flag = " [OMISSION]" if evt.is_omission else ""
        print(f"    {evt.label}: onset={evt.onset_ms}ms  code={evt.code}  "
              f"stimulus={evt.stimulus!r}{omission_flag}")

    print("\n=== JSON serialization (allow_nan=False) ===")
    paradigm_dict = paradigm.to_dict()
    json_str = json.dumps(paradigm_dict, allow_nan=False, indent=2, sort_keys=True)
    # Print only the top-level keys to keep output manageable
    top_keys = list(json.loads(json_str).keys())
    print(f"  Top-level keys: {top_keys}")
    print(f"  Conditions serialized: {len(paradigm_dict['conditions'])}")
    print(f"  JSON byte length: {len(json_str.encode('utf-8'))}")

    print("\n=== Scientific truth status ===")
    print("  truth_mode: truth_safe_unverified")
    print("  claim_level: computational_scaffold")
    print("  mechanism_claim_status: not_claimed")
    print("  empirical_validation_status: not_empirically_validated")
    print("  This paradigm object is a computational schema only.")
    print("  No biological mechanism is implied by condition structure.")


if __name__ == "__main__":
    main()
