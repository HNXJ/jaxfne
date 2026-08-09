"""Fields subpackage for jaxfne.

Isolates localized laminar projection computations from higher-level multimodal probe transformations.
"""
# Field solver acceptance checklist (maintainer contract, Phase F).
# A field solver may join the public API surface only when all five hold:
#   1. finite field outputs for any finite inputs;
#   2. additive (linear) superposition -- project(a + b) equals
#      project(a) + project(b) within the platform tolerance;
#   3. callable under jax.jit, verified by executing on valid array inputs;
#   4. carries the field metadata keys on its output surface
#      (field_claim_level, field_solver_status, physical_amplitude_calibrated);
#   5. amplitude truth gate -- physical_amplitude_calibrated is False
#      (proxy-readout state, never a physical-amplitude statement).
# Reference: tests/test_phaseF_solver_acceptance.py.
from __future__ import annotations

from .proxy import (
    FieldOutput,
    project_laminar_sources,
    project_sources_to_laminar_field,
    validate_source_field_status,
    compute_conservation_proxy_diagnostics,
    probe_laminar_modes,
    make_laminar_connectivity,
    exponential_synaptic_trace,
    synaptic_current,
    filtered_spike_source,
    teaching_control_spectrolaminar_resonance_source,
    spectrolaminar_psd,
    spectrolaminar_bandpower,
    spectrolaminar_readout,
    multi_area_spectrolaminar_readout,
    LinearReadout,
    construct_source_tensor,
    synaptic_resonance_source,
    combined_multi_area_source,
    spectrolaminar_similarity,
    spectrolaminar_objective,
    LegacyMultiAreaSpectrolaminarObjective,
    cable_filter_tau,
    cable_filter_sources,
    cable_filter_report,
    csd_tensor,
)
from .probes import (
    ProbeReadout,
    create_probe,
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
    eeg_proxy_transform,
    meg_proxy_transform,
    emm_proxy_transform,
)
from .diagnostics import (
    validate_projection_invariants,
    _make_field_solution_report,
)
from .solvers import (
    experimental_poisson_1d,
    experimental_poisson_1d_from_neuron_table,
)

__all__ = [
    "FieldOutput",
    "experimental_poisson_1d",
    "experimental_poisson_1d_from_neuron_table",
    "project_laminar_sources",
    "project_sources_to_laminar_field",
    "validate_projection_invariants",
    "validate_source_field_status",
    "compute_conservation_proxy_diagnostics",
    "probe_laminar_modes",
    "make_laminar_connectivity",
    "exponential_synaptic_trace",
    "synaptic_current",
    "filtered_spike_source",
    "teaching_control_spectrolaminar_resonance_source",
    "spectrolaminar_psd",
    "spectrolaminar_bandpower",
    "spectrolaminar_readout",
    "multi_area_spectrolaminar_readout",
    "LinearReadout",
    "construct_source_tensor",
    "synaptic_resonance_source",
    "combined_multi_area_source",
    "spectrolaminar_similarity",
    "spectrolaminar_objective",
    "LegacyMultiAreaSpectrolaminarObjective",
    "cable_filter_tau",
    "cable_filter_sources",
    "cable_filter_report",
    "csd_tensor",
    "_make_field_solution_report",
    "create_probe",
    "ProbeReadout",
    "spk_probe",
    "vm_probe",
    "source_probe",
    "lfp_proxy_probe",
    "csd_proxy_probe",
    "eeg_proxy_probe",
    "meg_proxy_probe",
    "emm_proxy_probe",
    "eeg_proxy_transform",
    "meg_proxy_transform",
    "emm_proxy_transform",
]
