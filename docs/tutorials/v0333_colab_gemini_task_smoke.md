# v0.3.33 Colab/Gemini Task Smoke Tutorial

This page documents the purpose, execution pathways, and validation of the task smoke notebook designed for Colab.

## Purpose
The task smoke notebook executes actual neural network simulations dynamically based on package version checks:
- **v031_stable_task_smoke**: Standard rate-synchrony models.
- **v032_delta_task_smoke**: Advanced hierarchical oddball simulation under the modular decoupled structure.

## Configuration
Use the selector variable `INSTALL_MODE` in the first cell:
- `"local"`: Runs using the local workspace checkout without modifying dependencies.
- `"pypi"`: Downloads standard `jaxfne`.
- `"dev"`: Pulls from Git dev branch.

## Output Structure
Reports and metrics are written to `outputs/v0333_colab_gemini_task_smoke/`. All output checks validate that readouts match strict JAX proxy criteria and retain `physical_amplitude_claim_allowed: False`.
