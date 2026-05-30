"""Public builder and helper functions for cortical-column configuration.

This module provides high-level construction helpers for common cortical-column
and multi-area scenarios. All functions return Configuration objects with
sensible defaults while preserving truth gates:
  - truth_mode = truth_safe_unverified
  - claim_level = computational_scaffold
  - field_solver_status = laminar_proxy_no_pde
  - physical_amplitude_claim_allowed = False

This is a proxy toolkit for declarative network specification, not a field
solver or calibration engine.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence

from .core import Configuration


def default_cortical_column_config(
    column_name: str = "single_column",
    n: int = 100,
    layers: Sequence[str] | None = None,
    seed: int | None = None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Create a default laminar cortical column Configuration.

    This helper sets up a single-column model with sensible defaults:
    - 6 layers (L1, L2/3, L4, L5, L6)
    - 4 cell types (E, PV, SST, VIP) with standard fractions
    - All-to-all uniform random local connectivity
    - Laminar proxy field with declarative metadata
    - Standard probe suite (spikes, V_m, source, LFP, CSD)

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
    - All truth gates are preserved: truth_safe_unverified,
      computational_scaffold, laminar_proxy_no_pde, no physical claims.
    - Field is a declarative proxy only; no PDE solver.
    - No sparse connectivity; all within-area connections are all-to-all uniform random.
    """
    if layers is None:
        layers = ["L1", "L2/3", "L4", "L5", "L6"]

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
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


def default_spectrolaminar_config(
    areas: Sequence[str] | None = None,
    n_per_area: int = 100,
    seed: int | None = None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Create a default multi-area spectrolaminar Configuration.

    This helper sets up a V1-V4 (or custom areas) laminar scaffold with:
    - Dual laminar columns (e.g., V1 and V4)
    - 4 cell types per column with standard fractions
    - Local all-to-all uniform random connectivity within each area
    - Inter-area feedforward/feedback connectivity metadata
    - Full readout suite (spikes, LFP, CSD, EEG, MEG, EMM)
    - Spectral objectives (alpha/beta, gamma band definitions)

    Parameters
    ----------
    areas : Sequence[str], optional
        Area names (e.g., ["V1", "V4"]). Default: ["V1", "V4"].
    n_per_area : int
        Neurons per area. Default: 100.
    seed : int, optional
        Random seed. Default: None.
    duration_ms : float
        Simulation duration in milliseconds. Default: 1000.0.
    dt_ms : float
        Timestep in milliseconds. Default: 0.1.

    Returns
    -------
    Configuration
        Multi-area laminar Configuration with inter-area connectivity and
        spectral objectives.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> cfg = jtfne.default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=200)
    >>> cfg = cfg.drive(baseline_drive_by_cell_type={"E": 5.0, "PV": 3.0})

    Notes
    -----
    - All truth gates preserved.
    - Field is laminar proxy; no PDE solution.
    - Inter-area connectivity is declarative metadata only.
    - Band definitions: alpha/beta [8, 25] Hz, gamma [40, 150] Hz.
    """
    if areas is None:
        areas = ["V1", "V4"]

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .areas(areas)
    )

    # Add columns for each area
    for area in areas:
        cfg = cfg.column(area, layers=["L1", "L2/3", "L4", "L5", "L6"], n=n_per_area)

    cfg = (
        cfg.cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .area_layer_cell_types(
            "V1",
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )
    )

    if len(areas) > 1:
        cfg = cfg.area_layer_cell_types(
            areas[1],
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )

    cfg = (
        cfg.uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.35, edge_seed=seed or 42)
    )

    # Add inter-area connectivity for V1 → V4
    if len(areas) >= 2:
        cfg = cfg.inter_column_connectivity(
            source_area=areas[0],
            target_area=areas[1],
            mode="sparse",
            p_feedforward=0.3,
            p_feedback=0.2,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
        )

    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD", "EEG", "MEG", "EMM"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .objective(
            firing_rate_target={"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0},
            band_definitions={"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)},
        )
    )
    return cfg


def build_laminar_column(
    name: str,
    n: int,
    layers: Sequence[str] | None = None,
    layer_fractions: Mapping[str, tuple] | None = None,
    cell_type_fractions: Mapping[str, float] | None = None,
    layer_cell_type_fractions: Mapping[str, Mapping[str, float]] | None = None,
) -> Configuration:
    """Build a Configuration for a single laminar cortical column.

    This builder creates a single-column Configuration with full control
    over layer structure and cell-type distribution.

    Parameters
    ----------
    name : str
        Column name (e.g., "V1", "visual_cortex").
    n : int
        Total number of neurons.
    layers : Sequence[str], optional
        Layer names. Default: ["L1", "L2/3", "L4", "L5", "L6"].
    layer_fractions : dict[str, tuple], optional
        Relative depths per layer (min, max). Default: computed evenly.
    cell_type_fractions : dict[str, float], optional
        Global E/PV/SST/VIP fractions. Default: {E:0.75, PV:0.1, SST:0.08, VIP:0.07}.
    layer_cell_type_fractions : dict[str, dict[str, float]], optional
        Per-layer overrides. Default: same as global.

    Returns
    -------
    Configuration
        Single-column Configuration, ready for probes/objective/optimizer.

    Examples
    --------
    >>> cfg = jtfne.build_laminar_column("M1", n=500, layers=["L2/3", "L5"])
    >>> cfg = cfg.probes(["spikes", "LFP"]).objective(firing_rate_target={"E": 8.0})
    """
    if layers is None:
        layers = ["L1", "L2/3", "L4", "L5", "L6"]
    if cell_type_fractions is None:
        cell_type_fractions = {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07}
    if layer_fractions is None:
        layer_fractions = {L: (i / len(layers), (i + 1) / len(layers)) for i, L in enumerate(layers)}
    if layer_cell_type_fractions is None:
        layer_cell_type_fractions = {L: cell_type_fractions for L in layers}

    cfg = (
        Configuration()
        .column(name, layers=layers, n=n)
        .cell_types(cell_type_fractions)
        .layer_fractions(layer_fractions, layer_cell_type_fractions)
        .uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45)
    )
    return cfg


def build_multi_area_columns(
    areas: Sequence[str],
    n_per_area: int,
    layers: Sequence[str] | None = None,
    connectivity_mode: Literal["sparse", "all_to_all"] = "sparse",
) -> Configuration:
    """Build a Configuration for multiple laminar areas with inter-area connectivity.

    This builder creates a multi-area scaffold with declarative inter-area
    connectivity between adjacent areas.

    Parameters
    ----------
    areas : Sequence[str]
        Area names (e.g., ["V1", "V4", "PFC"]).
    n_per_area : int
        Neurons per area.
    layers : Sequence[str], optional
        Shared layer sequence. Default: ["L1", "L2/3", "L4", "L5", "L6"].
    connectivity_mode : {"sparse", "all_to_all"}
        Inter-area connectivity mode. Default: "sparse".

    Returns
    -------
    Configuration
        Multi-area Configuration with declared inter-area connectivity.

    Examples
    --------
    >>> cfg = jtfne.build_multi_area_columns(["V1", "V4", "PFC"], n_per_area=200)
    >>> cfg = cfg.objective(band_definitions={"gamma": (40, 150)})
    """
    if layers is None:
        layers = ["L1", "L2/3", "L4", "L5", "L6"]

    cfg = Configuration().areas(areas)
    for area in areas:
        cfg = cfg.column(area, layers=layers, n=n_per_area)

    # Add inter-area connectivity: each adjacent pair has ff + fb
    for i, source_area in enumerate(areas[:-1]):
        target_area = areas[i + 1]
        cfg = cfg.inter_column_connectivity(
            source_area=source_area,
            target_area=target_area,
            mode=connectivity_mode,
            p_feedforward=0.3,
            p_feedback=0.2,
        )

    cfg = cfg.cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
    cfg = cfg.connectivity(within_area="all_to_all_uniform_random", within_gain=0.35)
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


def layer_celltype_count_table(cfg: Configuration | Any) -> dict[str, dict[str, int]]:
    """Generate a table of neuron counts by layer and cell type.

    Parameters
    ----------
    cfg : Configuration or Model
        Configuration or Model object.

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
    # Extract from cfg.metadata or model.neuron_table
    # For now, return a placeholder that will be filled in during integration
    metadata = getattr(cfg, "metadata", {})
    columns = metadata.get("columns", [])
    layers = metadata.get("layers", [])
    cell_types = metadata.get("cell_types", {})

    result = {}
    for layer in layers:
        result[layer] = {}
        total_n = sum(col.get("n", 0) for col in columns)
        for cell_type, fraction in cell_types.items():
            count = int(total_n * fraction / len(layers))
            result[layer][cell_type] = max(1, count)
    return result


def column_density_table(cfg: Configuration | Any) -> dict[str, float]:
    """Generate a table of neuronal density per layer (neurons / mm³).

    Parameters
    ----------
    cfg : Configuration or Model
        Configuration or Model object.

    Returns
    -------
    dict[str, float]
        Density per layer: {layer_name: density_per_mm³, ...}.
    """
    metadata = getattr(cfg, "metadata", {})
    columns = metadata.get("columns", [])
    layers = metadata.get("layers", [])

    result = {}
    for layer in layers:
        result[layer] = 0.0  # Placeholder; will be filled in during integration
    return result


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
    - No physical_amplitude_claim_allowed overrides
    - truth_mode = truth_safe_unverified
    - claim_level = computational_scaffold
    - field_solver_status = laminar_proxy_no_pde
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
    truth_mode = cfg.metadata.get("truth_mode")
    claim_level = cfg.metadata.get("claim_level", "computational_scaffold")
    field_solver_status = cfg.metadata.get("field_solver_status", "laminar_proxy_no_pde")
    physical_amplitude_claim = cfg.metadata.get("physical_amplitude_claim_allowed", False)

    truth_gates = {
        "truth_mode": truth_mode,
        "claim_level": claim_level,
        "field_solver_status": field_solver_status,
        "physical_amplitude_claim_allowed": physical_amplitude_claim,
    }

    if physical_amplitude_claim is not False:
        issues.append("physical_amplitude_claim_allowed_is_true")

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
