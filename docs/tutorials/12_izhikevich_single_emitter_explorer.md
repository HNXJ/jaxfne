# jaxfne TFNE-Izhikevich Single Emitter Explorer

Interactive browser-side preview for exploring reduced-emitter waveform dynamics before configuring a jaxfne scaffold.

## Run status

```yaml
artifact_class: interactive_tool
run_status: browser_euler_preview
model_status: parameter_intuition_only
field_solver_status: not_applicable
amplitude_status: native_unscaled
solver: browser_euler_dt_0.25ms
```

This tool runs entirely in the browser. It is a parameter intuition aid — not a jaxfne execution. All outputs are browser-computed previews for qualitative exploration only.

## What this tool does

The explorer lets you sweep Izhikevich reduced-emitter parameters in real time and observe the resulting waveform, phase portrait, and nullcline structure. Eleven biological presets are included: E-Regular, E-Bursting, E-Chattering, E-Wide, PV, SST, VIP, FS, LTS, RZ (Resonator), and TC (Thalamocortical).

Use it to:

1. Develop intuition for how `a`, `b`, `c`, `d` shape spike patterns before a jaxfne run.
2. Compare E vs PV vs SST vs VIP waveform classes side by side.
3. Understand the phase portrait (v vs u nullclines) for each cell type.
4. Choose a starting drive level `I` before setting `drive_gain` in AGSDR tuning.

## Transition to jaxfne

After exploring parameters here, use the package-native path for any actual simulation:

```python
import jaxfne as jtfne

# Configure with the presets that match your exploration
cfg = jtfne.suite2_v1_v4_config(seed=42, n_per_area=80, duration_ms=1000.0, dt_ms=0.1)
model = jtfne.construct(cfg)
bundle = jtfne.suite2_run_bundle(model, seed=42, duration_ms=1000.0, dt_ms=0.1)
```

The emitter presets in the explorer correspond to `jtfne.suite2_celltype_presets()`. The drive level `I` maps to the `drive_gain` tunable parameter in `jtfne.agsdr()`.

## Scope

- Browser Euler solver at dt=0.25 ms. Not the jaxfne JAX kernel.
- Outputs are waveform previews only. No source projection, no field readout.
- `amplitude_status: native_unscaled` — arbitrary units throughout.
- No biological mechanism is implied by the parameter sweep.

## Interactive panel

<div class="admonition note">
<p class="admonition-title">Browser-only preview</p>
<p>The panel below runs a forward-Euler Izhikevich solver in JavaScript. It requires no server, no Python, and no JAX. Adjust sliders to explore — then use jaxfne for any run that produces evidence.</p>
</div>

<iframe
  src="../../assets/interactive/izhikevich_single_emitter_explorer.html"
  width="100%"
  height="900px"
  style="border: 1px solid #2e2e3a; border-radius: 8px; background: #0d0d10;"
  title="jaxfne TFNE-Izhikevich Single Emitter Explorer"
  loading="lazy"
></iframe>

---

`run_status: browser_euler_preview`  
`model_status: parameter_intuition_only`  
`amplitude_status: native_unscaled`
