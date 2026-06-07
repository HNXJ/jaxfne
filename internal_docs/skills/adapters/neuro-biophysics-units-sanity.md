# neuro-biophysics-units-sanity

**Triggers:** mV, ms, S/cm2, conductance, Vm, physiological, units, biophysics, LFP, CSD, proxy amplitude.

**Purpose:** Block silent garbage from unit mismatches before trusting biophysical outputs.

**jaxley canon (typical):** Vm and reversal potentials in **mV**; `delta_t` in **ms**; conductance density in **S/cm²**; capacitance **µF/cm²**; length/radius **µm**.

**Fail conditions:** |Vm| > ~150 mV, NaN/Inf, conductance values orders of magnitude off (e.g. 120 instead of 0.12 S/cm²).

**Full skill:** user-installed `neuro-biophysics-units-sanity`. Claim scope: proxy outputs stay proxy unless calibration evidence exists.
