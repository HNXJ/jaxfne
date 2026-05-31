"""Build Etude No. 1 v2 — legacy-inspired hardcore mode.

Reads the existing notebook, inserts new cells (centralized config,
3D circuit visualization, activity suites, spectrolaminar baseline/stimulus),
and writes back the upgraded notebook.

Run from the repo root:
    python scripts/build_etude1_v2.py
"""
import json, uuid
from pathlib import Path

NB_PATH  = Path("tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb")
BASE_PATH = Path("tutorials/etudes/jaxfne_etude_no_1_base.ipynb")  # 87-cell source of truth


def _id():
    return uuid.uuid4().hex[:8]


def _to_source_list(s: str) -> list:
    """Split into lines, re-adding newlines (nbformat list-of-lines convention)."""
    lines = s.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": _to_source_list(source),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _id(),
        "metadata": {},
        "source": _to_source_list(source),
        "outputs": [],
        "execution_count": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL CONTENT BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

CFG_MD = """\
## Centralized Full-Detail Config

All knobs in one place. Every subsequent cell derives its values from `config`.
This cell may exceed the normal code-cell line preference — it is the main edit anchor."""

CFG_CODE = """\
from types import SimpleNamespace
from pathlib import Path as _Path

config = SimpleNamespace(
    # === Runtime ===
    SEED          = 20260530,
    DT_MS         = 0.1,
    T_MS_DEFAULT  = 1000.0,
    SMOKE_T_MS    = 300.0,
    DTYPE         = "float32",
    # === Geometry (declarative; proxy scaffold only) ===
    CX_M          = 1.0e-3,
    CY_M          = 1.0e-3,
    CZ_M          = 1.6e-3,
    UM_PER_M      = 1.0e6,
    COLUMN_RADIUS_MM = 0.25,
    # === Areas & Columns ===
    AREA_ORDER    = ["V1", "V4"],
    N_PER_AREA    = 80,
    N_PER_AREA_SMOKE = 40,
    # === Layers ===
    LAYERS        = ["L1", "L2/3", "L4", "L5", "L6"],
    LAYER_FRACTIONS = {
        "L1":  (0.00, 0.10),
        "L2/3":(0.10, 0.30),
        "L4":  (0.30, 0.45),
        "L5":  (0.45, 0.70),
        "L6":  (0.70, 1.00),
    },
    # === Cell Types ===
    CELL_TYPES    = ["E", "PV", "SST", "VIP"],
    CELL_COLORS   = {"E": "#e69500", "PV": "#0072ce", "SST": "#ffbf00", "VIP": "#7b3294"},
    CELL_SIGNS    = {"E": 1.0, "PV": -1.0, "SST": -1.0, "VIP": -1.0},
    CELL_TYPE_FRACTIONS = {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07},
    LAYER_CELL_FRACTIONS = {
        "L1":   {"E": 0.85, "PV": 0.00, "SST": 0.00, "VIP": 0.15},
        "L2/3": {"E": 0.78, "PV": 0.08, "SST": 0.07, "VIP": 0.07},
        "L4":   {"E": 0.60, "PV": 0.25, "SST": 0.10, "VIP": 0.05},
        "L5":   {"E": 0.70, "PV": 0.12, "SST": 0.12, "VIP": 0.06},
        "L6":   {"E": 0.72, "PV": 0.10, "SST": 0.08, "VIP": 0.10},
    },
    # === Drive & Noise (proxy metadata; not calibrated physical drive) ===
    DRIVE_BASELINE = {"E": 5.0, "PV": 3.0, "SST": 3.5, "VIP": 3.0},
    DRIVE_RANGE    = {
        "E":   (4.5, 9.0),
        "PV":  (2.5, 7.0),
        "SST": (2.5, 6.5),
        "VIP": (2.5, 6.5),
    },
    NOISE_SCALE    = 0.5,
    # === Inter-column connectivity (declarative metadata) ===
    INTER_P_FF     = 0.3,
    INTER_P_FB     = 0.2,
    INTER_FF_W     = (0.5, 2.0),
    INTER_FB_W     = (0.3, 1.5),
    # === Field / Probe / Readout ===
    N_CONTACTS     = 16,
    FIELD_SOLVER   = "laminar_proxy_no_pde",
    FREQ_MIN_HZ    = 1.0,
    FREQ_MAX_HZ    = 150.0,
    FREQ_COUNT     = 128,
    PSD_NPERSEG    = 128,
    SPECTRO_FIGSIZE = (14, 5),
    SPECTRO_CMAP   = "viridis",
    ALPHA_BETA_HZ  = (8.0, 25.0),
    GAMMA_HZ       = (40.0, 150.0),
    # === AGSDR Optimizer ===
    AGSDR_ALPHA    = 0.7,
    AGSDR_EXPL     = 0.05,
    AGSDR_DESEL    = 2.0,
    AGSDR_GEN      = 3,
    AGSDR_POP      = 2,
    AGSDR_PARAM    = {"drive_gain": (0.1, 1.5)},
    # === Objective ===
    TARGET_RATE_HZ = 3.5,
    TARGET_KAPPA   = 0.0,
    RATE_WEIGHT    = 1.0,
    KAPPA_WEIGHT   = 0.25,
    # === Stimulus ===
    STIM_AREA      = "V1",
    STIM_LAYER     = "L4",
    STIM_CTYPE     = "E",
    STIM_ONSET_MS  = 100.0,
    STIM_DUR_MS    = 100.0,
    STIM_AMP       = 1.5,
    # === Visualization ===
    FIG_DPI        = 150,
    ACT_FIGSIZE    = (14, 8),
    CIRC_FIGSIZE   = (12, 5),
    # === Output paths ===
    OUTPUT_DIR     = _Path("outputs/etude_no_1"),
    # === Truth gates ===
    TRUTH_MODE     = "truth_safe_unverified",
    CLAIM_LEVEL    = "computational_scaffold",
    FIELD_SOLVER_STATUS = "laminar_proxy_no_pde",
    PHYSICAL_AMPLITUDE_CLAIM_ALLOWED = False,
)"""

FINALIZE_MD = """\
## Finalize Config (derived paths)

Creates output directories and attaches derived `FIG_DIR` and `PLOTLY_DIR`."""

FINALIZE_CODE = """\
def finalize_config(cfg):
    cfg.FIG_DIR    = cfg.OUTPUT_DIR / "figures"
    cfg.PLOTLY_DIR = cfg.OUTPUT_DIR / "plotly"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTLY_DIR.mkdir(parents=True, exist_ok=True)
    return cfg

config = finalize_config(config)"""

RUNTIME_MD = """\
## Runtime Constants (derived from config)

`TFNE_SMOKE=1` selects a fast smoke configuration. All values come from the
centralized `config` object above."""

RUNTIME_CODE = """\
SMOKE       = os.environ.get("TFNE_SMOKE", "0") == "1"
SEED        = config.SEED
DT_MS       = config.DT_MS
DTYPE       = config.DTYPE
DURATION_MS = config.SMOKE_T_MS if SMOKE else config.T_MS_DEFAULT
N_PER_AREA  = config.N_PER_AREA_SMOKE if SMOKE else config.N_PER_AREA
AREAS       = config.AREA_ORDER
OUTPUT_DIR  = config.OUTPUT_DIR
FIG_DIR     = config.FIG_DIR"""

DOMAINS_MD = """\
## Runtime & Column Domains (editable via config)

These dicts expose jaxfne-API-facing settings. Edit the centralized `config` cell
above to change values — the dicts here derive from it."""

DOMAINS_CODE = """\
runtime = {"seed": SEED, "duration_ms": DURATION_MS, "dt_ms": DT_MS, "dtype": DTYPE}
LAYERS  = config.LAYERS
columns = {"areas": AREAS, "n_per_area": N_PER_AREA, "layers": LAYERS}
cell_types = config.CELL_TYPE_FRACTIONS"""

DRIVE_MD = """\
## Drive, Inter-Column, Field (derived from config)"""

DRIVE_CODE = """\
drive     = {"baseline": config.DRIVE_BASELINE, "noise": config.NOISE_SCALE}
inter_conn = {"source": AREAS[0], "target": AREAS[-1],
              "p_ff": config.INTER_P_FF, "p_fb": config.INTER_P_FB}
field     = {"solver": config.FIELD_SOLVER, "domain": "laminar_column"}"""

PROBES_MD = """\
## Probes, Objective, Optimizer (derived from config)"""

PROBES_CODE = """\
probes    = {"types": ["spikes", "V_m", "source", "LFP", "CSD"],
             "n_contacts": config.N_CONTACTS}
objective = {"rate_hz": config.TARGET_RATE_HZ, "kappa": config.TARGET_KAPPA,
             "rate_w": config.RATE_WEIGHT, "kappa_w": config.KAPPA_WEIGHT}
optimizer = {"family": "AGSDR", "alpha": config.AGSDR_ALPHA,
             "exploration": config.AGSDR_EXPL, "deselect_factor": config.AGSDR_DESEL,
             "parameters": config.AGSDR_PARAM, "gen": config.AGSDR_GEN,
             "pop": config.AGSDR_POP, "seed": SEED}"""

VIZ_HELPER_MD = """\
## Visualization Helpers

`visualize_circuit` renders the 3D proxy geometry; `plot_activity_suite` shows
raster, firing-rate trace, LFP proxy, and CSD proxy for any `signals` object.
Both helpers display the figure in the notebook **and** save a PNG artifact.
These are setup cells and may exceed the normal workflow-cell line preference."""

CIRCUIT_HELPER_CODE = """\
def visualize_circuit(mdl, cfg):
    import pandas as pd
    neurons = pd.DataFrame(mdl.neuron_table())
    fig = plt.figure(figsize=cfg.CIRC_FIGSIZE, dpi=cfg.FIG_DPI)
    ax  = fig.add_subplot(111, projection="3d")
    for ct in cfg.CELL_TYPES:
        sub = neurons[neurons["cell_type"] == ct]
        ax.scatter(sub["x"], sub["y"], sub["z"], s=8,
                   c=cfg.CELL_COLORS[ct], label=ct, alpha=0.7, edgecolors="none")
    ax.set_xlabel("x (a.u.)"); ax.set_ylabel("y (a.u.)"); ax.set_zlabel("depth (a.u.)")
    ax.set_title("Cortical circuit geometry — proxy scaffold (laminar_proxy_no_pde)")
    ax.legend(loc="upper left", fontsize=8, markerscale=2)
    out = cfg.FIG_DIR / "cortical_circuit_network.png"
    fig.savefig(out, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.show()
    return fig"""

ACTIVITY_HELPER_CODE = """\
def plot_activity_suite(signals, label, cfg, fname_suffix=None):
    spikes = np.asarray(signals.spikes)
    t_ms   = np.asarray(signals.time_ms)
    lfp    = np.asarray(signals.field.lfp_proxy)
    csd    = np.asarray(signals.field.csd_proxy)
    depths = np.asarray(signals.field.contact_depths)
    n_steps, n_units = spikes.shape
    rate_hz = 1000.0 * spikes.mean(axis=1) / cfg.DT_MS
    fig, axes = plt.subplots(2, 2, figsize=cfg.ACT_FIGSIZE, dpi=cfg.FIG_DPI)
    fig.suptitle(f"Activity Suite — {label} (proxy · truth_safe_unverified)", fontsize=11)
    ax = axes[0, 0]; si, ti = np.where(spikes.T)
    ax.scatter(t_ms[ti], si, s=0.5, c="k", alpha=0.4)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Neuron"); ax.set_title("Spike raster")
    ax = axes[0, 1]
    ax.plot(t_ms, rate_hz, lw=1, color="#e69500")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Rate (Hz)"); ax.set_title("Mean firing rate")
    ax = axes[1, 0]; vmax = float(np.abs(lfp).max()) or 1.0
    ax.imshow(lfp.T, aspect="auto", origin="lower",
              extent=[t_ms[0], t_ms[-1], float(depths[0]), float(depths[-1])],
              cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Depth"); ax.set_title("LFP proxy")
    ax = axes[1, 1]; vmax = float(np.abs(csd).max()) or 1.0
    ax.imshow(csd.T, aspect="auto", origin="lower",
              extent=[t_ms[0], t_ms[-1], float(depths[0]), float(depths[-1])],
              cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Depth"); ax.set_title("CSD proxy")
    fig.tight_layout()
    suf = fname_suffix or label.lower().replace(" ", "_")
    out = cfg.FIG_DIR / f"activity_suite_{suf}.png"
    fig.savefig(out, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.show()
    return fig"""

SPECTRO_HELPER_MD = """\
## Spectrolaminar Suite Helper (3-panel: cell density / power heatmap / band profiles)

`plot_spectrolaminar_etude1` produces the 3-panel layout:
- **A** Mean Relative Distribution of Cells (E/PV/SST/VIP vs cortical position from L4)
- **B** Mean Relative Power Spectrum (frequency × depth heatmap)
- **C** Alpha-beta / Gamma cross (band profiles vs depth)

Y-axis convention: negative = superficial (above L4), positive = deep (below L4).
This is a setup cell and may exceed the normal workflow-cell line preference."""

SPECTRO_HELPER_CODE = """\
def plot_spectrolaminar_etude1(signals, mdl, label, cfg, fname_suffix=None):
    import pandas as pd
    from scipy.signal import welch
    from scipy.ndimage import gaussian_filter1d
    neurons = pd.DataFrame(mdl.neuron_table())
    lfp    = np.asarray(signals.field.lfp_proxy)         # (n_steps, n_contacts)
    depths = np.asarray(signals.field.contact_depths)    # ∈ [0, 1]
    l4_c   = (cfg.LAYER_FRACTIONS["L4"][0] + cfg.LAYER_FRACTIONS["L4"][1]) / 2.0
    pos    = depths - l4_c                               # signed: superficial < 0
    fs     = 1000.0 / cfg.DT_MS
    fgrid  = np.linspace(cfg.FREQ_MIN_HZ, cfg.FREQ_MAX_HZ, cfg.FREQ_COUNT)
    psd    = np.zeros((len(depths), len(fgrid)))
    for ci in range(len(depths)):
        f, px = welch(lfp[:, ci], fs=fs, nperseg=min(cfg.PSD_NPERSEG, lfp.shape[0] // 2))
        psd[ci] = np.interp(fgrid, f, px)
    plog  = np.log10(psd + 1e-12)
    vlo, vhi = np.percentile(plog, 5), np.percentile(plog, 95)
    rel   = np.clip((plog - vlo) / (vhi - vlo + 1e-12), 0, 1) * 0.5 + 0.5
    ab_m  = (fgrid >= cfg.ALPHA_BETA_HZ[0]) & (fgrid <= cfg.ALPHA_BETA_HZ[1])
    gm_m  = (fgrid >= cfg.GAMMA_HZ[0])      & (fgrid <= cfg.GAMMA_HZ[1])
    ab_p  = rel[:, ab_m].mean(axis=1); ab_p = (ab_p - ab_p.min()) / (ab_p.max() - ab_p.min() + 1e-12)
    gm_p  = rel[:, gm_m].mean(axis=1); gm_p = (gm_p - gm_p.min()) / (gm_p.max() - gm_p.min() + 1e-12)
    z     = neurons["z"].values
    neurons = neurons.copy()
    neurons["pos"] = (z - z.min()) / (z.max() - z.min() + 1e-12) - l4_c
    bins  = np.linspace(pos[0] - 0.02, pos[-1] + 0.02, 33)
    ctrs  = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=cfg.SPECTRO_FIGSIZE, dpi=cfg.FIG_DPI,
        gridspec_kw={"width_ratios": [0.85, 1.75, 0.85]}, sharey=True)
    for ct in cfg.CELL_TYPES:
        vals, _ = np.histogram(neurons.loc[neurons["cell_type"] == ct, "pos"], bins=bins)
        vals    = gaussian_filter1d(vals.astype(float), 1.2)
        if vals.max() > 0: vals = vals / vals.max()
        ax0.plot(vals, ctrs, lw=2.0, color=cfg.CELL_COLORS[ct], label=ct)
    ax0.axhline(0.0, color="k", lw=1.2)
    ax0.set_title("A Mean Relative Dist of Cells")
    ax0.set_xlabel("Relative Count"); ax0.set_ylabel("Cortical Position from L4")
    ax0.legend(fontsize=7); ax0.set_ylim(pos[-1] + 0.05, pos[0] - 0.05)
    im = ax1.imshow(rel, aspect="auto", origin="upper", cmap=cfg.SPECTRO_CMAP,
                    extent=[fgrid[0], fgrid[-1], pos[-1], pos[0]], vmin=0.5, vmax=1.0)
    ax1.axhline(0.0, color="k", lw=1.2)
    ax1.set_title("B Mean Relative Power Spectrum"); ax1.set_xlabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax1, label="Rel Pow")
    ax2.plot(ab_p, pos, color="blue", lw=3.0, label="Alpha-beta")
    ax2.plot(gm_p, pos, color="red",  lw=3.0, label="Gamma")
    ax2.axhline(0.0, color="k", lw=1.2)
    ax2.set_title("C Alpha-beta / Gamma cross"); ax2.set_xlabel("Relative power")
    ax2.legend(fontsize=8)
    fig.suptitle(f"{label} — spectrolaminar proxy (truth_safe_unverified · laminar_proxy_no_pde)")
    fig.tight_layout()
    suf = fname_suffix or label.lower().replace(" ", "_")
    out = cfg.FIG_DIR / f"spectrolaminar_{suf}.png"
    fig.savefig(out, dpi=cfg.FIG_DPI, bbox_inches="tight"); plt.show()
    return fig"""

CIRCUIT_CALL_MD = """\
## 3D Cortical Circuit Visualization

Proxy scaffold geometry: neuron positions colored by cell type. The `z` axis
represents laminar depth. Positions are jaxfne-internal proxy coordinates, not
calibrated anatomical millimetres."""

CIRCUIT_CALL_CODE = """\
fig_circuit = visualize_circuit(model, config)
plt.close(fig_circuit)"""

ACT_BASE_MD = """\
## Activity Suite — Baseline

Raster, mean firing rate, LFP-proxy, and CSD-proxy for the pre-stimulus
baseline simulation. Proxy readouts only; not calibrated LFP/CSD amplitudes."""

ACT_BASE_CODE = """\
fig_act_base = plot_activity_suite(signals_baseline, "Baseline", config)
plt.close(fig_act_base)"""

SPEC_BASE_MD = """\
## Spectrolaminar Suite — Baseline

Three-panel spectrolaminar readout (cell density / power heatmap / band profiles)
for the baseline condition. Proxy-scale only (truth_safe_unverified; laminar_proxy_no_pde)."""

SPEC_BASE_CODE = """\
fig_spec_base = plot_spectrolaminar_etude1(signals_baseline, model, "Baseline", config)
plt.close(fig_spec_base)"""

ACT_STIM_MD = """\
## Activity Suite — Stimulus

Same visualization suite for the stimulus condition. Compare with baseline
to verify that native-drive stimulation changes the population response."""

ACT_STIM_CODE = """\
fig_act_stim = plot_activity_suite(signals_stim, "Stimulus", config)
plt.close(fig_act_stim)"""

SPEC_STIM_MD = """\
## Spectrolaminar Suite — Stimulus

Three-panel spectrolaminar readout for the stimulus condition."""

SPEC_STIM_CODE = """\
fig_spec_stim = plot_spectrolaminar_etude1(signals_stim, model, "Stimulus", config)
plt.close(fig_spec_stim)"""

ACT_TUNED_MD = """\
## Activity Suite — Tuned

Activity suite after AGSDR tuning. Compare with baseline and stimulus to
confirm that `drive_gain` adjustment changed the population dynamics."""

ACT_TUNED_CODE = """\
fig_act_tuned = plot_activity_suite(signals_tuned, "Tuned", config)
plt.close(fig_act_tuned)"""

# ─────────────────────────────────────────────────────────────────────────────
# MODIFIED EXISTING CELLS (replacement content)
# ─────────────────────────────────────────────────────────────────────────────

# Cell 55 (Viz Arguments MD) — update description
FIG_MOD_MD = """\
## Spectrolaminar Band Definitions (derived from config)

`config.ALPHA_BETA_HZ` and `config.GAMMA_HZ` define the band ranges used in
panel C of the spectrolaminar suite. Edit these in the centralized config."""

FIG_MOD_CODE = """\
ALPHA_BETA_HZ = config.ALPHA_BETA_HZ   # e.g. (8.0, 25.0)
GAMMA_HZ      = config.GAMMA_HZ        # e.g. (40.0, 150.0)
SPECTRO_CMAP  = config.SPECTRO_CMAP    # "viridis\""""

# Cell 57 (spectrolaminar figure header) — rename to "Tuned"
SPEC_TUNED_MD = """\
## Spectrolaminar Suite — Tuned

Three-panel spectrolaminar readout after AGSDR tuning. Use panel C to confirm
that the alpha-beta and gamma laminar profiles changed relative to baseline."""

# Cell 58 (spectrolaminar save) — use new helper
SPEC_TUNED_CODE = """\
fig_spec_tuned = plot_spectrolaminar_etude1(signals_tuned, tuned_model, "Tuned", config)
plt.close(fig_spec_tuned)"""

# STIM dict — derive from config
STIM_MD_NEW = """\
## Stimulus Target (derived from config)

Edit `config.STIM_AREA`, `config.STIM_LAYER`, `config.STIM_CTYPE`,
`config.STIM_AMP` etc. to change the targeted stimulus."""

STIM_CODE_NEW = """\
STIM = {"target_area": config.STIM_AREA, "target_layer": config.STIM_LAYER,
        "target_cell_type": config.STIM_CTYPE, "onset_ms": config.STIM_ONSET_MS,
        "duration_ms": config.STIM_DUR_MS, "amplitude": config.STIM_AMP,
        "label": "custom_target_drive"}"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def build():
    # Always read from the 87-cell base (source of truth); write to NB_PATH
    src = BASE_PATH if BASE_PATH.exists() else NB_PATH
    with open(src) as f:
        nb = json.load(f)

    old = nb["cells"]
    assert len(old) == 87, f"Base notebook must have 87 cells, got {len(old)}"

    # Helper to normalize source to list-with-newlines (nbformat convention)
    def src_list(c):
        s = c["source"]
        if isinstance(s, list):
            # Join and re-split to ensure proper \n handling
            s = "".join(s)
        c["source"] = _to_source_list(s)
        return c

    old = [src_list(c) for c in old]

    def get_old(i):
        """Return old cell i with a fresh ID."""
        c = dict(old[i])
        c["id"] = _id()
        return c

    # Identify key insertion points in the original notebook
    # Cells 00-06: header/install/imports → keep verbatim
    # Cell 07: "Runtime Constants" markdown → REPLACE with new Runtime MD
    # Cell 08: SMOKE/SEED code → REPLACE with derived-from-config version
    # Cells 09-21: warmup → keep verbatim
    # Cells 22-23: Part 2 header + Runtime & Column MD → keep
    # Cell 24: runtime/AREAS/columns code → REPLACE (derive from config)
    # Cell 25: Drive MD → KEEP
    # Cell 26: drive/inter_conn/field code → REPLACE
    # Cell 27: Probes MD → KEEP
    # Cell 28: probes/objective/optimizer code → REPLACE
    # Cells 29-31: CFG_KWARGS MD + code + Construct MD → keep
    # Cell 32: construct code → keep
    # After cell 32: INSERT circuit helper + call
    # Cells 33-34: Sim Setup MD + code → keep
    # Cells 35-36: Baseline MD + code → keep
    # After cell 36: INSERT act_base + spec_base
    # Cell 37: Stimulus Target MD → REPLACE
    # Cell 38: STIM code → REPLACE
    # Cells 39-44: select/stim/simulate stim → keep
    # After cell 44: INSERT act_stim + spec_stim
    # Cells 45-52: obj/opt/tune/tuned → keep
    # After cell 52: INSERT act_tuned
    # Cell 53: Condition Comparison MD → keep
    # Cell 54: df code → keep
    # Cell 55: Viz Arguments MD → REPLACE
    # Cell 56: FIG code → REPLACE
    # Cell 57: Spectrolaminar MD → REPLACE
    # Cell 58: spectrolaminar code → REPLACE
    # Cells 59-86: exports → keep

    new_cells = []

    # 0-6: header, installs, imports
    for i in range(7):
        new_cells.append(get_old(i))

    # INSERTED: Centralized config (MD + C)
    new_cells.append(md(CFG_MD))
    new_cells.append(code(CFG_CODE))

    # INSERTED: Finalize config (MD + C)
    new_cells.append(md(FINALIZE_MD))
    new_cells.append(code(FINALIZE_CODE))

    # REPLACED: Runtime Constants (MD + C)
    old[7]["source"] = _to_source_list(RUNTIME_MD)  # original MD
    c7 = get_old(7)
    c7["source"] = _to_source_list(RUNTIME_MD)
    new_cells.append(c7)
    c8 = get_old(8)
    c8["source"] = _to_source_list(RUNTIME_CODE)
    new_cells.append(c8)

    # 9-21: warmup (keep verbatim)
    for i in range(9, 22):
        new_cells.append(get_old(i))

    # 22: Part 2 header MD — keep
    new_cells.append(get_old(22))

    # 23: Runtime & Column MD — REPLACE text
    c23 = get_old(23)
    c23["source"] = _to_source_list(DOMAINS_MD)
    new_cells.append(c23)

    # 24: runtime/AREAS/columns code — REPLACE
    c24 = get_old(24)
    c24["source"] = _to_source_list(DOMAINS_CODE)
    new_cells.append(c24)

    # 25: Drive MD — REPLACE text
    c25 = get_old(25)
    c25["source"] = _to_source_list(DRIVE_MD)
    new_cells.append(c25)

    # 26: drive code — REPLACE
    c26 = get_old(26)
    c26["source"] = _to_source_list(DRIVE_CODE)
    new_cells.append(c26)

    # 27: Probes MD — REPLACE text
    c27 = get_old(27)
    c27["source"] = _to_source_list(PROBES_MD)
    new_cells.append(c27)

    # 28: probes code — REPLACE
    c28 = get_old(28)
    c28["source"] = _to_source_list(PROBES_CODE)
    new_cells.append(c28)

    # 29-32: CFG_KWARGS MD, code, Construct MD, construct code
    for i in range(29, 33):
        new_cells.append(get_old(i))

    # INSERTED after model construction: visualization helpers + circuit call
    new_cells.append(md(VIZ_HELPER_MD))
    new_cells.append(code(CIRCUIT_HELPER_CODE))
    new_cells.append(md("## Activity Suite Helper"))
    new_cells.append(code(ACTIVITY_HELPER_CODE))
    new_cells.append(md(SPECTRO_HELPER_MD))
    new_cells.append(code(SPECTRO_HELPER_CODE))
    new_cells.append(md(CIRCUIT_CALL_MD))
    new_cells.append(code(CIRCUIT_CALL_CODE))

    # 33-36: Sim Setup MD + code, Baseline MD + code
    for i in range(33, 37):
        new_cells.append(get_old(i))

    # INSERTED after baseline: activity suite + spectrolaminar
    new_cells.append(md(ACT_BASE_MD))
    new_cells.append(code(ACT_BASE_CODE))
    new_cells.append(md(SPEC_BASE_MD))
    new_cells.append(code(SPEC_BASE_CODE))

    # 37: Stimulus Target MD — REPLACE
    c37 = get_old(37)
    c37["source"] = _to_source_list(STIM_MD_NEW)
    new_cells.append(c37)

    # 38: STIM code — REPLACE
    c38 = get_old(38)
    c38["source"] = _to_source_list(STIM_CODE_NEW)
    new_cells.append(c38)

    # 39-44: select neurons, stim schedule, stim simulation
    for i in range(39, 45):
        new_cells.append(get_old(i))

    # INSERTED after stim: activity suite + spectrolaminar
    new_cells.append(md(ACT_STIM_MD))
    new_cells.append(code(ACT_STIM_CODE))
    new_cells.append(md(SPEC_STIM_MD))
    new_cells.append(code(SPEC_STIM_CODE))

    # 45-52: objective, optimizer, tune, tuned simulation
    for i in range(45, 53):
        new_cells.append(get_old(i))

    # INSERTED after tuned simulation: activity suite
    new_cells.append(md(ACT_TUNED_MD))
    new_cells.append(code(ACT_TUNED_CODE))

    # 53-54: Condition Comparison MD + code
    for i in range(53, 55):
        new_cells.append(get_old(i))

    # 55: Viz Arguments MD — REPLACE
    c55 = get_old(55)
    c55["source"] = _to_source_list(FIG_MOD_MD)
    new_cells.append(c55)

    # 56: FIG dict code — REPLACE
    c56 = get_old(56)
    c56["source"] = _to_source_list(FIG_MOD_CODE)
    new_cells.append(c56)

    # 57: Spectrolaminar Suite MD — REPLACE
    c57 = get_old(57)
    c57["source"] = _to_source_list(SPEC_TUNED_MD)
    new_cells.append(c57)

    # 58: spectrolaminar code — REPLACE (save as spectrolaminar_tuned.png)
    c58 = get_old(58)
    c58["source"] = _to_source_list(SPEC_TUNED_CODE)
    new_cells.append(c58)

    ARTIFACT_FILES_CODE = """\
artifact_files = [
    OUTPUT_DIR / 'manifest.json', OUTPUT_DIR / 'validation_report.json',
    OUTPUT_DIR / 'metrics.json',
    FIG_DIR / 'cortical_circuit_network.png',
    FIG_DIR / 'activity_suite_baseline.png',
    FIG_DIR / 'activity_suite_stimulus.png',
    FIG_DIR / 'activity_suite_tuned.png',
    FIG_DIR / 'spectrolaminar_baseline.png',
    FIG_DIR / 'spectrolaminar_stimulus.png',
    FIG_DIR / 'spectrolaminar_tuned.png',
]
hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in artifact_files if f.exists()}
jtfne.save_json(hashes, OUTPUT_DIR / 'asset_hashes.json')"""

    # 59-86: all export/manifest/completion cells — keep most verbatim
    # Old cell 83 (artifact_files) → replace with updated figure list
    for i in range(59, len(old)):
        c = get_old(i)
        raw_src = "".join(old[i]["source"])
        if "spectrolaminar.png" in raw_src and "artifact_files" in raw_src:
            # Replace with updated artifact_files that includes all new figures
            c["source"] = _to_source_list(ARTIFACT_FILES_CODE)
        elif "FIG" in raw_src and "visualization" in raw_src and "editable_inputs" in raw_src:
            # Replace the visualization export: FIG no longer exists, use config
            c["source"] = _to_source_list(
                "manifest['editable_inputs']['stimulus'] = STIM\n"
                "manifest['editable_inputs']['visualization'] = {\n"
                "    'freq_min_hz': config.FREQ_MIN_HZ, 'freq_max_hz': config.FREQ_MAX_HZ,\n"
                "    'freq_count': config.FREQ_COUNT, 'psd_nperseg': config.PSD_NPERSEG,\n"
                "    'figsize': list(config.SPECTRO_FIGSIZE), 'dpi': config.FIG_DPI,\n"
                "    'alpha_beta_hz': list(config.ALPHA_BETA_HZ),\n"
                "    'gamma_hz': list(config.GAMMA_HZ), 'cmap': config.SPECTRO_CMAP,\n"
                "}"
            )
        new_cells.append(c)

    # Verify no consecutive code cells
    for idx in range(len(new_cells) - 1):
        if new_cells[idx]["cell_type"] == "code" and new_cells[idx + 1]["cell_type"] == "code":
            print(f"WARNING: consecutive code cells at {idx} and {idx + 1}")
            print("  [%d] %s" % (idx, str(new_cells[idx]["source"])[:60]))
            print("  [%d] %s" % (idx + 1, str(new_cells[idx + 1]["source"])[:60]))

    nb["cells"] = new_cells
    nb["nbformat_minor"] = 5

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    n_code = sum(1 for c in new_cells if c["cell_type"] == "code")
    n_md = sum(1 for c in new_cells if c["cell_type"] == "markdown")
    print(f"Written {NB_PATH}: {len(new_cells)} cells ({n_md} md / {n_code} code)")


if __name__ == "__main__":
    build()
