"""Preset registries for cell types and receptor kinetics.

These are metadata only — no biological calibration or empirical validation.
Truth status: truth_safe_unverified, computational_scaffold.
"""

from __future__ import annotations


# Cell-type presets for Izhikevich parameters
CELL_TYPE_PRESETS = {
    "E_RS": {
        "a": 0.02,
        "b": 0.20,
        "c": -65.0,
        "d": 8.0,
        "drive": 5.0,
        "sign": 1.0,
        "label": "E",
        "description": "Regular-spiking excitatory",
        "source": "Izhikevich 2003 Table 1",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_claim_allowed": False,
    },
    "PV_FS": {
        "a": 0.10,
        "b": 0.20,
        "c": -65.0,
        "d": 2.0,
        "drive": 3.0,
        "sign": -1.0,
        "label": "PV",
        "description": "Fast-spiking parvalbumin+",
        "source": "Izhikevich 2003 Table 1",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_claim_allowed": False,
    },
    "SST_LTS": {
        "a": 0.02,
        "b": 0.25,
        "c": -65.0,
        "d": 2.0,
        "drive": 3.5,
        "sign": -1.0,
        "label": "SST",
        "description": "Low-threshold somatostatin+",
        "source": "Izhikevich 2003 Table 1 (LTS variant)",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_claim_allowed": False,
    },
    "VIP_IS": {
        "a": 0.02,
        "b": -0.10,
        "c": -55.0,
        "d": 6.0,
        "drive": 3.0,
        "sign": -1.0,
        "label": "VIP",
        "description": "Intrinsic spiking / bursting, VIP+ disinhibitory",
        "source": "Izhikevich 2003 Table 1 (IS variant)",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_claim_allowed": False,
    },
}


# Receptor kinetics metadata
RECEPTOR_KINETICS = {
    "AMPA": {
        "name": "AMPA",
        "receptor_index": 0,
        "sign": 1,
        "tau_ms": 2.0,
        "reversal_mV": 0.0,
        "description": "Alpha-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid",
        "source": "Standard neuroscience literature",
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_claim_allowed": False,
    },
    "NMDA": {
        "name": "NMDA",
        "receptor_index": 2,
        "sign": 1,
        "tau_ms": 100.0,
        "reversal_mV": 0.0,
        "description": "N-methyl-D-aspartate; slow timescale (no Mg-block implemented)",
        "source": "Standard neuroscience literature",
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_claim_allowed": False,
    },
    "GABA_A": {
        "name": "GABA_A",
        "receptor_index": 1,
        "sign": -1,
        "tau_ms": 5.0,
        "reversal_mV": -80.0,
        "description": "Ionotropic GABA receptor, fast timescale",
        "source": "Standard neuroscience literature",
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_claim_allowed": False,
    },
    "GABA_B": {
        "name": "GABA_B",
        "receptor_index": 3,
        "sign": -1,
        "tau_ms": 150.0,
        "reversal_mV": -95.0,
        "description": "Metabotropic GABA receptor, slow timescale (no G-protein cascade implemented)",
        "source": "Standard neuroscience literature",
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_claim_allowed": False,
    },
}


# Module-level constants for spike impulse gain
DEFAULT_SPIKE_IMPULSE_GAIN = 20.0
"""Default amplitude contribution per spike in source proxy computation.

This is a scaling factor, not a biophysical calibration.
No empirical basis; used for numerical stability and range.
"""
