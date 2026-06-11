# v0.3.33 Colab/Gemini Orientation Tutorial

This page documents the purpose, execution pathways, and strict metadata constraints of the Google Colab orientation notebook designed for AI onboarding.

## Purpose
The orientation notebook allows an agent or compiler to inspect the currently installed `jaxfne` package boundary, check which API capabilities are present, and run a safe smoke simulation matching the installed version.

It safely bridges:
- **v0.3.31 (Stable PyPI Path)**: Uses `suite2_four_celltype_config` and `simulate()`.
- **v0.3.32-alpha (Dev Branch Path)**: Employs the decoupled `SanityDeltaConfig` and `sanity_runtime` full-mode simulation.

## Key Design Rules
1. **Dynamic Installation**: Supports both standard PyPI install and direct Git tracking via the branch configuration at the top.
2. **Canonical Imports**: Only loads `import jaxfne as jtfne` to ensure package namespace integrity.
3. **No Uncalibrated Physical Claims**: Simulated readouts are strictly labeled as computational proxies (`lfp_proxy`, `csd_proxy`, `eeg_proxy`, `meg_proxy`), preserving default truth boundaries.
4. **Structured Orientation Report**: Automatically inspects public members and writes verification diagnostics to `outputs/v0333_colab_gemini_orientation/orientation_report.json`.

## Run Pathways
- **v031_stable_smoke**: Executed automatically when `SanityDeltaConfig` is absent.
- **v032_delta_full_smoke**: Executed automatically when the dev-mode components are detected.
