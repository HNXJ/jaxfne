import json

with open("tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb") as f:
    nb = json.load(f)

replacements = {
    "Claim discipline (immutable throughout):": "Scope status:",
    "**Claim discipline (immutable throughout):**": "**Scope status:**",
    "`native current-like` (proxy amplitude), `LFP-proxy` , `CSD-proxy` , `proxy` , `computational_scaffold` .": "`uncalibrated_izhikevich_native_current` (proxy amplitude), `LFP-proxy`, `CSD-proxy`, `EEG-proxy`, `MEG-proxy`, `EMM-proxy`, `computational_scaffold`.",
    "they are not proofs of biological mechanism.": "they are tools for *generating and testing hypotheses* in a clean educational sandbox.",
    "Model outputs (spikes, source proxy, LFP-proxy signals) require explicit empirical validation before any biological claim can be made.": "Model outputs (spikes, source proxy, LFP-proxy signals) function as uncalibrated educational proxies.",
    "**Claim gate:**": "**Scope status:**",
    "Claim gate:": "Scope status:",
    "**Claim Gate:**": "**Scope status:**",
    "Claim Gate:": "Scope status:",
    "Immutable claim gates": "Scope Fields",
    "Immutable claim gates (frozen in `tutorial_manifest.json`):": "Scope Fields (Immutable):",
    "Claim Gates": "Scope Fields",
    "claim gates": "scope fields",
    "Claim discipline": "Scope discipline",
    "claim discipline": "scope discipline",
    "> Claim: This is a **computational scaffold** for learning circuit concepts. Population sizes are educational, not empirically calibrated.": "> **Scope status:** This is a **computational scaffold** for learning circuit concepts with educational population sizes.",
    "> **Claim gate:** This is a **demo-scale optimization** (15 steps). It acts as a demo-scale optimization.": "> **Scope status:** This is a **demo-scale optimization** (15 steps).",
    "toy_scale_not_empirical": "uncalibrated_izhikevich_native_current",
    "tutorial_exploratory_not_biological_truth": "tutorial_exploratory_computational_scaffold"
}

for cell in nb.get("cells", []):
    source_text = "".join(cell.get("source", []))
    for k, v in replacements.items():
        source_text = source_text.replace(k, v)
    # Split back into lines, keeping the line endings
    lines = source_text.splitlines(keepends=True)
    cell["source"] = lines

with open("tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook updated successfully via joined-string replacement.")
