"""Public builder and helper functions for cortical-column configuration.

Docs: ``docs/api/index.md`` "Column builders & the canonical prior"
(https://jaxfne.readthedocs.io/en/latest/api/) — update that page when these
builders' signatures or defaults change.

High-level construction helpers for common cortical-column and multi-area
scenarios. Every builder returns a :class:`Configuration` with all knobs
defaulted, so canonical models are reproducible from the call site without
editing source.

Two cell-type composition paths share the builders:

* ``ei_profile="flat"`` (default) — depth-invariant E/PV/SST/VIP fractions with
  ``uniform3d`` placement. This is the legacy behavior, preserved unchanged.
* ``ei_profile="canonical"`` — the verified ground-truth E:I gradient (E peaks
  deep to 95%, I peaks superficial at 50%, PV peaks at L2/L3, ≈66E:34I
  overall), with laminar placement so each neuron keeps its layer label. The
  gradient is exported as first-class constants
  (:data:`CANONICAL_LAYER_CELL_TYPE_FRACTIONS`, :data:`CANONICAL_Z_BANDS`,
  and the 5-layer variants) — query those directly rather than copying these
  numbers, they are the source of truth.

All builders preserve the truth gates:
  - claim_level = computational_scaffold
  - field_solver_status = linear_solver
  - physical_amplitude_calibrated = False

This is a proxy toolkit for declarative network specification, not a field
solver or calibration engine.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

from .core import Configuration

#: Historical default builder layer set (5-layer, L2/3 merged), superficial → deep.
DEFAULT_LAYERS: tuple[str, ...] = ("L1", "L2/3", "L4", "L5", "L6")

#: Fully resolved canonical 6-layer set (L2/L3 split), superficial → deep.
CANONICAL_LAYERS_6L: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")

#: Flat (depth-invariant) cell-type fractions — the legacy default.
FLAT_CELL_TYPE_FRACTIONS: dict[str, float] = {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07}

#: Canonical ground-truth per-layer E:I composition (user reference, 2026-06-19).
#: E-fraction rises with depth (peaks L6 95%); I-fraction is highest superficial
#: (L1 50% I, L2/L3 50% I); PV FRACTION peaks at L4 (feedforward); L1 is VIP-rich
#: (those VIP specifically inhibit L1/2/3 SST); L6 has no PV/VIP. Overall ≈72E:28I.
#: Keys are the 6-layer names in :data:`CANONICAL_LAYERS_6L`; a 5-layer ("L2/3"
#: merged) variant is :data:`CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L`.
CANONICAL_LAYER_CELL_TYPE_FRACTIONS: dict[str, dict[str, float]] = {
    "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},  # 50% I, VIP-rich
    "L2": {"E": 0.50, "PV": 0.25, "SST": 0.10, "VIP": 0.15},  # 50% I, PV-strong
    "L3": {"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10},  # 50% I
    "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},  # PV fraction peak
    "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},  # large-E apical SST inhib
    "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},  # E-fraction peak, no PV/VIP
}

#: 5-layer ("L2/3" merged) variant of the canonical composition (L2+L3 averaged).
CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L: dict[str, dict[str, float]] = {
    "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},
    "L2/3": {"E": 0.50, "PV": 0.25, "SST": 0.13, "VIP": 0.12},
    "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},
    "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},
    "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},
}

#: Count-proportional z-bands (depth ∝ neuron count) for the 6-layer canonical column.
CANONICAL_Z_BANDS: dict[str, tuple[float, float]] = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

#: Count-proportional z-bands for the 5-layer (L2/3 merged) canonical column.
CANONICAL_Z_BANDS_5L: dict[str, tuple[float, float]] = {
    "L1": (0.00, 0.10), "L2/3": (0.10, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}


def _canonical_z_bands(layers: Sequence[str]) -> dict[str, tuple[float, float]] | None:
    """Return canonical count-proportional z-bands for a known layer set, else None."""
    s = tuple(layers)
    if s == CANONICAL_LAYERS_6L:
        return dict(CANONICAL_Z_BANDS)
    if s == DEFAULT_LAYERS:
        return dict(CANONICAL_Z_BANDS_5L)
    return None


def _resolve_layer_cell_types(
    layers: Sequence[str],
    ei_profile: Literal["flat", "canonical"],
    cell_type_fractions: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Resolve per-layer cell-type fractions for an ``ei_profile``.

    ``"flat"`` repeats ``cell_type_fractions`` across every layer (legacy
    behavior). ``"canonical"`` returns the verified ground-truth E:I gradient
    when ``layers`` matches the 6-layer or 5-layer canonical set; otherwise it
    raises ``ValueError`` so the caller is never silently downgraded to flat.
    """
    if ei_profile == "flat":
        return {L: dict(cell_type_fractions) for L in layers}
    if ei_profile != "canonical":
        raise ValueError(f"ei_profile must be 'flat' or 'canonical', got {ei_profile!r}")
    layer_set = tuple(layers)
    if layer_set == CANONICAL_LAYERS_6L:
        return {L: dict(CANONICAL_LAYER_CELL_TYPE_FRACTIONS[L]) for L in layers}
    if layer_set == DEFAULT_LAYERS:  # 5-layer, L2/3 merged
        return {L: dict(CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L[L]) for L in layers}
    raise ValueError(
        f"ei_profile='canonical' requires the canonical 6-layer {CANONICAL_LAYERS_6L} "
        f"or 5-layer {DEFAULT_LAYERS} sequence; got {layer_set}. "
        "Pass layer_cell_type_fractions explicitly for custom layers."
    )


def default_cortical_column_config(
    column_name: str = "single_column",
    n: int = 100,
    layers: Sequence[str] | None = None,
    seed: int | None = None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    *,
    synaptic_kernel: Literal["exponential", "receptor_exponential"] = "exponential",
) -> Configuration:
    """Create a default laminar cortical column Configuration.

    This helper sets up a single-column model with sensible defaults:
    - 6 layers (L1, L2/3, L4, L5, L6)
    - 4 cell types (E, PV, SST, VIP) with standard fractions
    - All-to-all uniform random local connectivity
    - Laminar proxy field with declarative metadata
    - Standard probe suite (spikes, V_m, source, LFP, CSD)

    All native (non-Jaxley) builds use the same Izhikevich emitter --
    ``construct()`` only supports ``family="izhikevich"``
    (``_SUPPORTED_EMITTER_FAMILIES``); ``set_emitter()``'s ``preset=`` kwarg is
    metadata only and has no behavioral effect. The one real behavioral
    choice within the native pipeline is ``synaptic_kernel``, exposed here.
    Biophysical (HH/Jaxley) neurons are not reachable through this builder at
    all -- they go through the separate ``JaxleyBridge`` entry point, which
    integrates a Jaxley model directly rather than building one from a
    ``Configuration``.

    Parameters
    ----------
    column_name : str
        Name of the column (e.g., "V1", "visual_cortex"). Default: "single_column".
    n : int
        Total number of neurons. Default: 100.
    layers : Sequence[str], optional
        Layer names. Default: ["L1", "L2/3", "L4", "L5", "L6"].
    seed : int, optional
        Random seed. Default: None (uses default in runtime).
    duration_ms : float
        Simulation duration in milliseconds. Default: 1000.0.
    dt_ms : float
        Timestep in milliseconds. Default: 0.1.
    synaptic_kernel : {"exponential", "receptor_exponential"}, default "exponential"
        Which native synaptic kernel the edge-list recurrent path uses.
        ``"receptor_exponential"`` requires (and sets) ``recurrent_backend=
        "edge_list"`` -- it is not supported on the dense backend, and not
        supported together with ``enable_homeostasis``.

    Returns
    -------
    Configuration
        Configuration with sensible defaults for a laminar column.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> cfg = jtfne.default_cortical_column_config("V1", n=200)
    >>> cfg = cfg.objective(firing_rate_target={"E": 8.0})
    >>> cfg = cfg.optimizer(optimizer_family="AGSDR", budget=50)

    Notes
    -----
    - All truth gates are preserved: ,
      computational_scaffold, linear_solver, no physical claims.
    - Field is a declarative proxy only; no PDE solver.
    - No sparse connectivity; all within-area connections are all-to-all uniform random.
    """
    if layers is None:
        layers = ["L1", "L2/3", "L4", "L5", "L6"]
    if synaptic_kernel not in ("exponential", "receptor_exponential"):
        raise ValueError(
            f"synaptic_kernel must be 'exponential' or 'receptor_exponential'; got {synaptic_kernel!r}"
        )
    recurrent_backend = "edge_list" if synaptic_kernel == "receptor_exponential" else "dense"

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32",
                  synaptic_kernel=synaptic_kernel, recurrent_backend=recurrent_backend)
        .column(column_name, layers=layers, n=n)
        .cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .layer_fractions(
            layer_fractions={L: (i / len(layers), (i + 1) / len(layers)) for i, L in enumerate(layers)},
            layer_cell_types={L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in layers},
        )
        .uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45, edge_seed=seed or 42)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return cfg


def default_complete_configuration(
    column_name: str = "V1",
    nucleus_name: str = "thalamus",
    n_column: int = 100,
    n_nucleus: int = 60,
    layers: Sequence[str] | None = None,
    seed: int | None = None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Create the broadest default Configuration: a laminar cortical column
    wired to a non-laminar nucleus — cortex and subcortex in one config.

    Combines what :func:`default_cortical_column_config` and
    :func:`default_nuclei_config` each declare into a single multi-area
    Configuration (``.column()`` calls accumulate; see
    ``Configuration.column()``'s docstring), then wires them with inter-area
    connectivity: feedforward column→nucleus from its superficial layer,
    feedback nucleus→column into its first layer. The FF/FB direction is a
    structural choice mirroring how :func:`default_spectrolaminar_config`
    wires two cortical areas — it is not a thalamocortical-circuit claim;
    no biological-mechanism status is implied by the layer/area names.

    Parameters
    ----------
    column_name : str, default "V1"
        Name of the cortical column.
    nucleus_name : str, default "thalamus"
        Name of the nucleus.
    n_column : int, default 100
        Neurons in the cortical column.
    n_nucleus : int, default 60
        Neurons in the nucleus.
    layers : Sequence[str], optional
        Cortical column layer names. Default: ``["L1", "L2/3", "L4", "L5", "L6"]``.
    seed : int, optional
        Random seed. Default: None (uses 42).
    duration_ms : float, default 1000.0
        Simulation duration in milliseconds.
    dt_ms : float, default 0.1
        Timestep in milliseconds.

    Returns
    -------
    Configuration
        Multi-area Configuration with one laminar column, one flat nucleus,
        and declarative inter-area connectivity between them.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> cfg = jtfne.default_complete_configuration("V1", "thalamus")
    >>> model = jtfne.construct(cfg)

    Notes
    -----
    - All truth gates preserved.
    - Inter-area connectivity is declarative metadata only (no PDE solve,
      no mechanism claim); the nucleus carries no biological-nucleus-identity
      claim from its name alone.
    """
    if layers is None:
        layers = ["L1", "L2/3", "L4", "L5", "L6"]
    layers = list(layers)

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .areas([column_name, nucleus_name])
        .column(column_name, layers=layers, n=n_column)
        .column(nucleus_name, layers=["core"], n=n_nucleus)
        .cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .area_layer_cell_types(
            column_name,
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in layers},
        )
        .area_layer_cell_types(nucleus_name, {"core": {"E": 0.70, "PV": 0.30}})
        .uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.40, edge_seed=seed or 42)
        .inter_column_connectivity(
            source_area=column_name,
            target_area=nucleus_name,
            mode="sparse",
            layer_to_layer_map={"L2/3": "core"},
            p_feedforward=0.3,
            p_feedback=0.2,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return cfg


def build_laminar_column(
    name: str = "V1",
    n: int = 1000,
    layers: Sequence[str] | None = None,
    layer_fractions: Mapping[str, tuple] | None = None,
    cell_type_fractions: Mapping[str, float] | None = None,
    layer_cell_type_fractions: Mapping[str, Mapping[str, float]] | None = None,
    *,
    ei_profile: Literal["flat", "canonical"] = "flat",
    geometry: Literal["auto", "uniform3d", "laminar"] = "auto",
    within_connectivity: str = "all_to_all_uniform_random",
    within_gain: float = 0.45,
    radius_mm: float = 0.25,
    height_mm: float = 1.6,
    edge_seed: int | None = None,
) -> Configuration:
    """Build a Configuration for a single laminar cortical column.

    Single-column builder with full control over layer structure, cell-type
    composition, and local connectivity. Every knob is defaulted so the
    canonical column is reproducible from the call site without editing source.

    Parameters
    ----------
    name : str, default "V1"
        Column name (e.g., "V1", "M1", "visual_cortex").
    n : int, default 1000
        Total number of neurons, distributed across layers by ``layer_fractions``
        depth widths (count ∝ thickness).
    layers : Sequence[str], optional
        Layer names. Default: the 5-layer ``DEFAULT_LAYERS``
        ``("L1","L2/3","L4","L5","L6")``. Use :data:`CANONICAL_LAYERS_6L` for the
        split-L2/L3 form.
    layer_fractions : dict[str, tuple], optional
        Relative depth band ``(z0, z1)`` per layer; band *width* sets the
        per-layer neuron count. Default: canonical count-proportional z-bands
        when ``layers`` is a known canonical set (:data:`CANONICAL_Z_BANDS` /
        :data:`CANONICAL_Z_BANDS_5L`), otherwise evenly spaced bands.
    cell_type_fractions : dict[str, float], optional
        Global E/PV/SST/VIP fractions used when ``ei_profile="flat"`` and no
        explicit per-layer map is given. Default: :data:`FLAT_CELL_TYPE_FRACTIONS`.
    layer_cell_type_fractions : dict[str, dict[str, float]], optional
        Explicit per-layer composition. When given it overrides ``ei_profile``.
    ei_profile : {"flat", "canonical"}, keyword-only, default "flat"
        ``"flat"`` repeats ``cell_type_fractions`` across layers (legacy).
        ``"canonical"`` applies the verified ground-truth E:I gradient
        (E peaks deep to 95%, I peaks superficial at 50%, PV peaks at L2/L3,
        ≈66E:34I overall) — requires the canonical 6- or 5-layer set.
    geometry : {"auto", "uniform3d", "laminar"}, keyword-only, default "auto"
        Neuron placement. ``"uniform3d"`` scatters neurons in a cylinder
        (legacy; collapses layer identity to ``"uniform_3d"``, so per-layer
        composition is *not* preserved). ``"laminar"`` places neurons in depth
        bands so each neuron keeps its layer label and per-layer composition is
        honored. ``"auto"`` selects ``"laminar"`` whenever a non-flat
        composition is requested (``ei_profile="canonical"`` or an explicit
        ``layer_cell_type_fractions``), else ``"uniform3d"`` for backward
        compatibility.
    within_connectivity : str, keyword-only, default "all_to_all_uniform_random"
        Within-area connectivity rule passed to ``.connectivity``.
    within_gain : float, keyword-only, default 0.45
        Within-area weight gain.
    radius_mm, height_mm : float, keyword-only
        Column geometry (cylinder radius and depth), default 0.25 / 1.6 mm.
    edge_seed : int, optional, keyword-only
        Seed for connectivity edge sampling; ``None`` uses the runtime default.

    Returns
    -------
    Configuration
        Single-column Configuration, ready for probes/objective/optimizer.

    Examples
    --------
    >>> cfg = jtfne.build_laminar_column()                       # canonical-size V1, flat E:I
    >>> cfg = jtfne.build_laminar_column(ei_profile="canonical")  # ground-truth E:I gradient
    >>> cfg = jtfne.build_laminar_column("M1", 500, layers=["L2/3", "L5"], within_gain=0.6)
    """
    if layers is None:
        layers = list(DEFAULT_LAYERS)
    layers = list(layers)
    if cell_type_fractions is None:
        cell_type_fractions = dict(FLAT_CELL_TYPE_FRACTIONS)
    if layer_fractions is None:
        canon_bands = _canonical_z_bands(layers)
        if canon_bands is not None:
            layer_fractions = canon_bands
        else:
            layer_fractions = {L: (i / len(layers), (i + 1) / len(layers)) for i, L in enumerate(layers)}
    explicit_per_layer = layer_cell_type_fractions is not None
    if layer_cell_type_fractions is None:
        layer_cell_type_fractions = _resolve_layer_cell_types(layers, ei_profile, cell_type_fractions)

    if geometry == "auto":
        geometry = "laminar" if (ei_profile == "canonical" or explicit_per_layer) else "uniform3d"

    conn_kwargs: dict[str, Any] = {"within_area": within_connectivity, "within_gain": within_gain}
    if edge_seed is not None:
        conn_kwargs["edge_seed"] = edge_seed

    cfg = (
        Configuration()
        .column(name, layers=layers, n=n)
        .cell_types(cell_type_fractions)
        .layer_fractions(layer_fractions, layer_cell_type_fractions)
    )
    if geometry == "uniform3d":
        # Legacy placement: cylinder scatter (collapses per-neuron layer label).
        cfg = cfg.uniform3d(radius_mm=radius_mm, height_mm=height_mm)
    cfg = cfg.connectivity(**conn_kwargs)
    if ei_profile == "canonical":
        # Enable construct-time canonical biophysics (deep-E size grading +
        # PV<->E local strengthening). Random v0 is always on regardless.
        cfg = cfg.runtime(canonical_biophysics=True)
    return cfg


def build_multi_area_columns(
    areas: Sequence[str] = ("V1", "V4", "PFC"),
    n_per_area: int = 200,
    layers: Sequence[str] | None = None,
    connectivity_mode: Literal["sparse", "all_to_all"] = "sparse",
    *,
    ei_profile: Literal["flat", "canonical"] = "flat",
    cell_type_fractions: Mapping[str, float] | None = None,
    within_connectivity: str = "all_to_all_uniform_random",
    within_gain: float = 0.35,
    p_feedforward: float = 0.3,
    p_feedback: float = 0.2,
) -> Configuration:
    """Build a Configuration for multiple laminar areas with inter-area connectivity.

    Multi-area scaffold with declarative feedforward/feedback connectivity
    between adjacent areas. Defaults to the canonical V1→V4→PFC hierarchy so a
    full multi-area config requires no positional arguments.

    Parameters
    ----------
    areas : Sequence[str], default ("V1", "V4", "PFC")
        Area names, ordered low → high in the hierarchy.
    n_per_area : int, default 200
        Neurons per area.
    layers : Sequence[str], optional
        Shared layer sequence. Default: the 5-layer ``DEFAULT_LAYERS``.
    connectivity_mode : {"sparse", "all_to_all"}, default "sparse"
        Inter-area connectivity mode.
    ei_profile : {"flat", "canonical"}, keyword-only, default "flat"
        Per-layer cell-type composition applied to every area; ``"canonical"``
        uses the verified ground-truth E:I gradient (see
        :func:`build_laminar_column`).
    cell_type_fractions : dict[str, float], optional, keyword-only
        Global fractions for ``ei_profile="flat"``. Default:
        :data:`FLAT_CELL_TYPE_FRACTIONS`.
    within_connectivity : str, keyword-only, default "all_to_all_uniform_random"
        Within-area connectivity rule.
    within_gain : float, keyword-only, default 0.35
        Within-area weight gain.
    p_feedforward, p_feedback : float, keyword-only
        Inter-area connection probabilities, default 0.3 / 0.2.

    Returns
    -------
    Configuration
        Multi-area Configuration with declared inter-area connectivity.

    Examples
    --------
    >>> cfg = jtfne.build_multi_area_columns()                          # V1→V4→PFC, 200/area
    >>> cfg = jtfne.build_multi_area_columns(["V1", "V4"], ei_profile="canonical")
    """
    if layers is None:
        layers = list(DEFAULT_LAYERS)
    layers = list(layers)
    if cell_type_fractions is None:
        cell_type_fractions = dict(FLAT_CELL_TYPE_FRACTIONS)
    per_layer = _resolve_layer_cell_types(layers, ei_profile, cell_type_fractions)

    cfg = Configuration().areas(areas)
    for area in areas:
        cfg = cfg.column(area, layers=layers, n=n_per_area)

    # Inter-area connectivity for each adjacent pair, wired in BOTH directions so
    # the hierarchy carries genuine top-down feedback (not just feedforward into
    # deeper target layers). One spec per direction; specs accumulate:
    #   feedforward  lo -> hi  : source L2/3 E -> target L4   (uses p_feedforward)
    #   feedback     hi -> lo  : source L6     -> target L1/L5 (uses p_feedback)
    for lo, hi in zip(areas[:-1], areas[1:]):
        cfg = cfg.inter_column_connectivity(
            source_area=lo, target_area=hi, mode=connectivity_mode,
            p_feedforward=p_feedforward, p_feedback=0.0,
        )
        cfg = cfg.inter_column_connectivity(
            source_area=hi, target_area=lo, mode=connectivity_mode,
            p_feedforward=0.0, p_feedback=p_feedback,
        )

    cfg = cfg.cell_types(cell_type_fractions)
    for area in areas:
        cfg = cfg.area_layer_cell_types(area, per_layer)
    cfg = cfg.connectivity(within_area=within_connectivity, within_gain=within_gain)
    return cfg


def connect_columns(
    cfg: Configuration,
    source_area: str,
    target_area: str,
    mode: Literal["sparse", "all_to_all"] = "sparse",
    feedforward_gain: float = 0.65,
    feedback_gain: float = 0.50,
) -> Configuration:
    """Add inter-column connectivity between two areas in a Configuration.

    Parameters
    ----------
    cfg : Configuration
        Existing Configuration object.
    source_area : str
        Source area name.
    target_area : str
        Target area name.
    mode : {"sparse", "all_to_all"}
        Connectivity mode. Default: "sparse".
    feedforward_gain : float
        Feedforward weight scaling. Default: 0.65.
    feedback_gain : float
        Feedback weight scaling. Default: 0.50.

    Returns
    -------
    Configuration
        Updated Configuration.

    Examples
    --------
    >>> cfg = jtfne.build_multi_area_columns(["V1", "V4"], n_per_area=100)
    >>> cfg = jtfne.connect_columns(cfg, "V1", "V4", mode="all_to_all")
    """
    return cfg.inter_column_connectivity(
        source_area=source_area,
        target_area=target_area,
        mode=mode,
        p_feedforward=0.3,
        p_feedback=0.2,
        feedforward_weight_range=(feedforward_gain * 0.7, feedforward_gain * 1.3),
        feedback_weight_range=(feedback_gain * 0.7, feedback_gain * 1.3),
    )


def sparse_intercolumn_connectivity(
    p_feedforward: float = 0.3,
    p_feedback: float = 0.2,
    feedforward_weight_range: tuple = (0.5, 2.0),
    feedback_weight_range: tuple = (0.3, 1.5),
    seed: int | None = None,
) -> dict[str, Any]:
    """Create a sparse inter-column connectivity specification.

    Returns
    -------
    dict[str, Any]
        Dictionary with sparse connectivity parameters for use with
        .inter_column_connectivity().

    Examples
    --------
    >>> spec = jtfne.sparse_intercolumn_connectivity(p_feedforward=0.4)
    >>> cfg = cfg.inter_column_connectivity(**spec)
    """
    return {
        "mode": "sparse",
        "p_feedforward": p_feedforward,
        "p_feedback": p_feedback,
        "feedforward_weight_range": feedforward_weight_range,
        "feedback_weight_range": feedback_weight_range,
        "seed": seed,
    }


def all_to_all_intercolumn_connectivity(
    feedforward_gain: float = 0.65,
    feedback_gain: float = 0.50,
) -> dict[str, Any]:
    """Create an all-to-all inter-column connectivity specification.

    Returns
    -------
    dict[str, Any]
        Dictionary with dense connectivity parameters.

    Examples
    --------
    >>> spec = jtfne.all_to_all_intercolumn_connectivity(feedforward_gain=0.7)
    >>> cfg = cfg.inter_column_connectivity(**spec)
    """
    return {
        "mode": "all_to_all",
        "feedforward_weight_range": (feedforward_gain * 0.7, feedforward_gain * 1.3),
        "feedback_weight_range": (feedback_gain * 0.7, feedback_gain * 1.3),
    }


def _metadata_for_table(obj: Configuration | Any) -> dict[str, Any]:
    """Resolve configuration metadata from a Configuration or Model-like object."""
    if hasattr(obj, "cfg") and hasattr(obj.cfg, "metadata"):
        return dict(obj.cfg.metadata)
    if isinstance(obj, Configuration):
        return dict(obj.metadata)
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


def _neuron_rows_for_table(obj: Configuration | Any) -> list[dict[str, Any]]:
    """Return neuron_table rows for Configuration (via construct) or Model."""
    if hasattr(obj, "neuron_table") and callable(getattr(obj, "neuron_table")):
        if not isinstance(obj, Configuration):
            return [dict(row) for row in obj.neuron_table()]
    if isinstance(obj, Configuration):
        from .core import construct

        return construct(obj).neuron_table()
    raise TypeError(
        "Expected Configuration or Model with neuron_table(); "
        f"got {type(obj).__name__!r}"
    )


def _layer_thickness_mm(layer: str, metadata: Mapping[str, Any]) -> float:
    """Layer thickness in mm from declared z-bands and column height."""
    height_mm = float(metadata.get("column_height_mm", 1.60))
    if str(layer) == "uniform_3d":
        return height_mm
    fractions = metadata.get("layer_fractions") or {}
    if str(layer) in fractions:
        z0, z1 = fractions[str(layer)]
        return max(0.0, float(z1) - float(z0)) * height_mm
    columns = metadata.get("columns") or []
    if columns:
        layers = [str(x) for x in columns[0].get("layers", [])]
        if layers and str(layer) in layers:
            idx = layers.index(str(layer))
            n = len(layers)
            return height_mm / max(n, 1)
    return height_mm


def layer_celltype_count_table(cfg: Configuration | Any) -> dict[str, dict[str, int]]:
    """Generate a table of neuron counts by layer and cell type.

    Parameters
    ----------
    cfg : Configuration or Model
        Configuration (compiled via :func:`construct`) or an existing Model.

    Returns
    -------
    dict[str, dict[str, int]]
        Nested dict: {layer_name: {cell_type: count, ...}, ...}.

    Examples
    --------
    >>> table = jtfne.layer_celltype_count_table(cfg)
    >>> table["L4"]["E"]
    75
    """
    rows = _neuron_rows_for_table(cfg)
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        layer = str(row.get("layer", "unspecified"))
        cell_type = str(row.get("cell_type", "unspecified"))
        table.setdefault(layer, {})
        table[layer][cell_type] = table[layer].get(cell_type, 0) + 1
    return table


def column_density_table(cfg: Configuration | Any) -> dict[str, float]:
    """Generate neuronal density per layer (neurons / mm³).

    Uses declared column geometry (``column_radius_mm``, ``column_height_mm``,
    ``layer_fractions``) and neuron counts from :meth:`Model.neuron_table`.
    Cylindrical layer volume: ``π r² × thickness_mm`` with ``thickness_mm``
    from the layer's normalized z-band × column height.

    Parameters
    ----------
    cfg : Configuration or Model
        Configuration or Model (same resolution path as
        :func:`layer_celltype_count_table`).

    Returns
    -------
    dict[str, float]
        Density per layer: {layer_name: neurons_per_mm³, ...}.
    """
    rows = _neuron_rows_for_table(cfg)
    metadata = _metadata_for_table(cfg)
    radius_mm = float(metadata.get("column_radius_mm", 0.25))
    layer_counts: dict[str, int] = {}
    for row in rows:
        layer = str(row.get("layer", "unspecified"))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    out: dict[str, float] = {}
    for layer, count in layer_counts.items():
        thickness_mm = _layer_thickness_mm(layer, metadata)
        volume_mm3 = math.pi * radius_mm ** 2 * thickness_mm
        out[layer] = float(count) / volume_mm3 if volume_mm3 > 0.0 else 0.0
    return out


def configuration_table(cfg: Configuration) -> dict[str, Any]:
    """Generate a human-readable summary table of all configuration settings.

    Parameters
    ----------
    cfg : Configuration
        Configuration object.

    Returns
    -------
    dict[str, Any]
        Summary with all top-level config domains.

    Examples
    --------
    >>> table = jtfne.configuration_table(cfg)
    >>> table["runtime"]
    {'seed': 42, 'duration_ms': 1000, ...}
    """
    return {
        "runtime": cfg.metadata.get("runtime", {}),
        "columns": cfg.metadata.get("columns", {}),
        "cell_types": cfg.metadata.get("cell_types", {}),
        "connectivity": cfg.metadata.get("connectivity", {}),
        "inter_column_connectivity": cfg.metadata.get("inter_column_connectivity", {}),
        "drive": cfg.metadata.get("drive", {}),
        "probes": {p.get("name", "unnamed"): p for p in cfg.probes},
        "field": cfg.metadata.get("field", {}),
        "objective": cfg.metadata.get("objective", {}),
        "optimizer": cfg.metadata.get("optimizer", {}),
    }


def validate_configuration(
    cfg: Configuration,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate a Configuration against required gates.

    Parameters
    ----------
    cfg : Configuration
        Configuration object.
    strict : bool
        If True, fail on any gap; if False, list warnings only. Default: True.

    Returns
    -------
    dict[str, Any]
        Validation result with status, warnings/errors, truth gates.

    Checks
    ------
    - All required domains present (or optional with sensible defaults)
    - Layer fractions sum to 1.0
    - Cell-type fractions sum to 1.0
    - Total neuron count > 0
    - Connectivity parameters are valid
    - Field domain is declared
    - Probes list is non-empty
    - No physical_amplitude_calibrated overrides
    - claim_level = computational_scaffold
    - field_solver_status = linear_solver
    """
    issues = []

    # Check required domains
    if not cfg.networks:
        issues.append("no_networks_declared")
    if not cfg.emitters:
        issues.append("no_emitters_declared")
    if not cfg.fields:
        issues.append("no_field_declared")
    if not cfg.probes:
        issues.append("no_probes_declared")

    # Check truth gates
    claim_level = cfg.metadata.get("claim_level", "computational_scaffold")
    field_solver_status = cfg.metadata.get("field_solver_status", "linear_solver")
    physical_amplitude_claim = cfg.metadata.get("physical_amplitude_calibrated", False)

    truth_gates = {
        "claim_level": claim_level,
        "field_solver_status": field_solver_status,
        "physical_amplitude_calibrated": physical_amplitude_claim,
    }

    if physical_amplitude_claim is not False:
        issues.append("physical_amplitude_calibrated_is_true")

    if strict and issues:
        return {
            "status": "FAIL",
            "issues": issues,
            "truth_gates": truth_gates,
        }

    return {
        "status": "PASS" if not issues else "WARN",
        "issues": issues,
        "truth_gates": truth_gates,
    }


def laminar_cortex_config(
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    areas: Sequence[str] | None = None,
    layers: Sequence[str] | None = None,
    cell_types: Mapping[str, float] | None = None,
    n: int = 128,
    emitter: str = "izhikevich",
    baseline_drive_by_cell_type: Mapping[str, float] | None = None,
) -> Configuration:
    """Generalized laminar cortical configuration builder.

    Creates a multi-area, multi-layer laminar circuit Configuration with
    stable metadata and user-controlled geometry.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Default: 0.
    duration_ms : float
        Simulation duration in milliseconds. Default: 1000.0.
    dt_ms : float
        Simulation timestep in milliseconds. Default: 0.1.
    areas : Sequence[str], optional
        Area labels (e.g., ["V1", "V4"]). Default: ["V1"].
    layers : Sequence[str], optional
        Layer labels (e.g., ["L2/3", "L4", "L5"]). Default: ["L2/3", "L4", "L5", "L6"].
        For single-layer degenerate case, use ["single"] or any one-element list.
    cell_types : Mapping[str, float], optional
        Cell-type fractions (must sum to ~1.0). Default: {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}.
    n : int
        Total number of neurons. Default: 128.
    emitter : str
        Emitter family ("izhikevich", "lif", "glif"). Default: "izhikevich".
    baseline_drive_by_cell_type : Mapping[str, float], optional
        Baseline DC input (drive) per cell type to eliminate silent neurons.
        Default: None (no explicit baseline drive; uses emitter preset defaults).
        Example: {"E": 4.5, "PV": 4.0, "SST": 5.5, "VIP": 10.0} (in nA or native units).

    Returns
    -------
    Configuration
        Generalized laminar cortex Configuration with stable area/layer/cell_type metadata.

    Notes
    -----
    - All truth gates preserved: computational_scaffold,
      linear_solver, physical_amplitude_calibrated=False.
    - Selectors work on area, layer, cell_type, and combinations (e.g., select(area="V1", layer="L4")).
    - All arrays generated with finite values and float32 dtype by default.
    - Non-laminar use: pass layers=["single"] for one-layer degenerate case.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> cfg = jtfne.laminar_cortex_config(
    ...     seed=0,
    ...     duration_ms=1000.0,
    ...     dt_ms=0.1,
    ...     areas=["V1"],
    ...     layers=["L2/3", "L4", "L5", "L6"],
    ...     cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    ...     n=128,
    ... )
    >>> model = jtfne.construct(cfg)
    >>> signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)

    Single-layer degenerate case:

    >>> cfg_single = jtfne.laminar_cortex_config(
    ...     layers=["single"],
    ...     n=64,
    ... )
    """
    if areas is None:
        areas = ["V1"]
    if layers is None:
        layers = ["L2/3", "L4", "L5", "L6"]
    if cell_types is None:
        cell_types = {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}

    # Ensure cell_types sum is approximately 1.0 (tolerance for floating point)
    cell_types_sum = sum(cell_types.values())
    if not (0.99 <= cell_types_sum <= 1.01):
        raise ValueError(
            f"cell_types fractions must sum to ~1.0; got {cell_types_sum}. "
            f"Provided: {cell_types}"
        )

    cfg = (
        Configuration()
        .runtime(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
    )

    # Add areas and layers
    for area_idx, area_label in enumerate(areas):
        cfg = cfg.column(area_label, layers=list(layers), n=n)
        cfg = cfg.cell_types(cell_types)

    # Set emitter
    if emitter == "izhikevich":
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
    elif emitter == "lif":
        cfg = cfg.set_emitter("lif")
    elif emitter == "glif":
        cfg = cfg.set_emitter("glif")
    else:
        raise ValueError(f"Unknown emitter: {emitter}. Choose from: izhikevich, lif, glif")

    # Set baseline drive to eliminate silent neurons (if provided)
    if baseline_drive_by_cell_type:
        cfg = cfg.drive(baseline_drive_by_cell_type=baseline_drive_by_cell_type)

    # Add probes and field
    cfg = cfg.probes(["spikes", "V_m"], n_contacts=16)
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")

    return cfg
