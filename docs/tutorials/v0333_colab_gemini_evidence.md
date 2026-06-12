# v0.3.33 Colab/Gemini Evidence Tutorial

This page documents the execution of the full evidence notebook showing oddball simulation pathways and notebook-local figures.

## Purpose
The evidence notebook runs the dev `SanityDelta` full-runtime simulation path, generates strict JSON reports, and produces a small notebook-local figure set (raster, mean membrane potential by area, and proxy readouts).

## Configuration
The `INSTALL_MODE` selector allows running using local checkout (`"local"`) or installing from GitHub dev branch (`"dev"`).

## Output Structure
Outputs are saved to `outputs/v0333_colab_gemini_evidence/`. All output checks validate that simulated activities and proxy readouts retain `physical_amplitude_claim_allowed: False` and `biological_learning_claim: False`.
