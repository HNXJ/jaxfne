"""Preset registries for cell types and receptor kinetics.

These are metadata only; no biological calibration or empirical validation.
Truth status: computational_scaffold.
"""

from __future__ import annotations


# Single source of truth for presynaptic sign (+1 excitatory, -1 inhibitory),
# keyed by cell-type label. It lives here, below jaxfne.emitters in the import
# order, so every catalog can read it; jaxfne.emitters re-exports it as
# `emitters.CELL_TYPE_SIGN`. The Izhikevich rows, the presets below, the tutorial
# catalog and `jaxfne.emitters_lif.LIF_LABEL_SIGN` (LIF-only labels "simple",
# "complex", both excitatory) read their signs from it. "SOM" is deliberately
# absent: it names an AdEx-table type, and adding a label is a model decision.
CELL_TYPE_SIGN: dict[str, float] = {
    "E": 1.0,
    "PV": -1.0,
    "Inl": -1.0,
    "SST": -1.0,
    "Ing": -1.0,
    "VIP": -1.0,
    "I": -1.0,
    "simple": 1.0,
    "complex": 1.0,
}


# Cell-type presets for Izhikevich parameters.
# Own a/b/c/d and drive values; signs from CELL_TYPE_SIGN.
CELL_TYPE_PRESETS = {
    "E_RS": {
        "a": 0.02,
        "b": 0.20,
        "c": -65.0,
        "d": 8.0,
        "drive": 5.0,
        "sign": CELL_TYPE_SIGN["E"],
        "label": "E",
        "description": "Regular-spiking excitatory",
        "source": "Izhikevich 2003 Fig. 2 (RS)",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
    },
    "PV_FS": {
        "a": 0.10,
        "b": 0.20,
        "c": -65.0,
        "d": 2.0,
        "drive": 3.0,
        "sign": CELL_TYPE_SIGN["PV"],
        "label": "PV",
        "description": "Fast-spiking parvalbumin+",
        "source": "Izhikevich 2003 Fig. 2 (FS)",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
    },
    "SST_LTS": {
        "a": 0.02,
        "b": 0.25,
        "c": -65.0,
        "d": 2.0,
        "drive": 3.5,
        "sign": CELL_TYPE_SIGN["SST"],
        "label": "SST",
        "description": "Low-threshold somatostatin+",
        "source": "Izhikevich 2003 Fig. 2 (LTS)",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
    },
    "VIP_IS": {
        "a": 0.02,
        "b": -0.10,
        "c": -55.0,
        "d": 6.0,
        "drive": 3.0,
        "sign": CELL_TYPE_SIGN["VIP"],
        "label": "VIP",
        "description": "Intrinsic spiking / bursting, VIP+ disinhibitory",
        "source": "not in Izhikevich 2003; origin not recorded",
        "truth_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
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
        "physical_amplitude_calibrated": False,
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
        "physical_amplitude_calibrated": False,
    },
    "GABA_A": {
        "name": "GABA_A",
        "receptor_index": 1,
        "sign": -1,
        "tau_ms": 5.0,
        "reversal_mV": -80.0,
        "description": "Ionotropic GABA receptor, fast timescale",
        "source": "Xiang, Huguenard & Prince 1998, J Physiol 506:715-730 (adult rat visual cortex LV pyramidal sIPSC monoexponential tau_D 5.0 ms)",
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_calibrated": False,
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
        "physical_amplitude_calibrated": False,
    },
}


# Module-level constants for spike impulse gain
DEFAULT_SPIKE_IMPULSE_GAIN = 20.0
"""Default amplitude contribution per spike in source proxy computation.

This is a scaling factor, not a biophysical calibration.
No empirical basis; used for numerical stability and range.
"""
