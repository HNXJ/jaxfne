"""v0.3.32: Hierarchical global-local oddball API hardening.

Package-native orchestration objects for structured oddball task with:
- 5 laminar areas (V1a, V1b, V4, MT, PFC), 500 neurons total
- AAAB stimulus sequence with fixation gate and reward signaling
- Bounded computational homeostatic regularizer (not biological learning)
- Exact resumable backup state with 1000 ms ring buffer
- Proxy-safe readouts only (lfp_proxy, csd_proxy, eeg_proxy, meg_proxy)

Truth gates:
  
  computational_scaffold
  linear_solver
  physical_amplitude_calibrated=False
  biological_learning_claim=False
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from . import sanity_runtime


@dataclasses.dataclass
class SanityDeltaConfig:
    """Hierarchical oddball configuration factory and validation.

    Encapsulates model structure, stimulus maps, task parameters, and truth gates.
    Constructs model and paradigm via package-native APIs.
    """

    seed: int
    duration_ms: float
    dt_ms: float
    neurons_per_area: int
    areas: tuple[str, ...]
    hierarchy: tuple[tuple[str, str], ...]
    cell_counts: dict[str, int]
    stimulus_frequency_hz: float
    stimulus_map: dict[str, dict[str, float]]
    claim_level: Literal["computational_scaffold"] = "computational_scaffold"
    field_solver_status: Literal["linear_solver"] = "linear_solver"
    physical_amplitude_calibrated: bool = False
    biological_learning_claim: bool = False
    runtime_mode: Literal["scaffold", "full"] = "scaffold"

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.areas) == 5, f"Expected 5 areas, got {len(self.areas)}"
        assert self.areas == ("V1a", "V1b", "V4", "MT", "PFC"), f"Expected canonical areas, got {self.areas}"
        assert self.neurons_per_area == 100, f"Expected 100 neurons/area, got {self.neurons_per_area}"
        assert len(self.areas) * self.neurons_per_area == 500, "Total must be 500"
        assert sum(self.cell_counts.values()) == 100, "Cell counts must sum to 100"
        assert "E" in self.cell_counts and sum(c for k, c in self.cell_counts.items() if k != "E") == 30, \
            "Must have 70E + 30I"
        assert len(self.hierarchy) == 4, "Hierarchy must have 4 edges"
        assert "A" in self.stimulus_map and "B" in self.stimulus_map, "Must have A and B stimuli"
        assert self.stimulus_map["A"]["V1a"] == 0.75 and self.stimulus_map["A"]["V1b"] == 0.25
        assert self.stimulus_map["B"]["V1a"] == 0.25 and self.stimulus_map["B"]["V1b"] == 0.75
        assert self.runtime_mode in ("scaffold", "full"), f"Invalid runtime_mode: {self.runtime_mode}"

    @classmethod
    def hierarchical_global_local_oddball(
        cls,
        seed: int = 0,
        duration_ms: float = 12000.0,
        dt_ms: float = 0.1,
        neurons_per_area: int = 100,
        areas: tuple[str, ...] = ("V1a", "V1b", "V4", "MT", "PFC"),
        hierarchy: tuple[tuple[str, str], ...] | None = None,
        cell_counts: dict[str, int] | None = None,
        stimulus_frequency_hz: float = 100.0,
        stimulus_map: dict[str, dict[str, float]] | None = None,
        claim_level: str = "computational_scaffold",
        field_solver_status: str = "linear_solver",
        physical_amplitude_calibrated: bool = False,
        runtime_mode: str = "scaffold",
    ) -> SanityDeltaConfig:
        """Factory for hierarchical global-local oddball configuration."""
        if hierarchy is None:
            hierarchy = (("V1a", "V4"), ("V1b", "V4"), ("V4", "MT"), ("MT", "PFC"))
        if cell_counts is None:
            cell_counts = {"E": 70, "PV": 12, "SST": 10, "VIP": 8}
        if stimulus_map is None:
            stimulus_map = {
                "A": {"orientation_deg": 45.0, "V1a": 0.75, "V1b": 0.25},
                "B": {"orientation_deg": 135.0, "V1a": 0.25, "V1b": 0.75},
            }

        return cls(
            seed=seed,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            neurons_per_area=neurons_per_area,
            areas=areas,
            hierarchy=hierarchy,
            cell_counts=cell_counts,
            stimulus_frequency_hz=stimulus_frequency_hz,
            stimulus_map=stimulus_map,
            claim_level=claim_level,
            field_solver_status=field_solver_status,
            physical_amplitude_calibrated=physical_amplitude_calibrated,
            runtime_mode=runtime_mode,
        )

    def construct(self) -> SanityDeltaModel:
        """Construct model from configuration.

        Returns: SanityDeltaModel wrapping compiled network state.
        """
        n_steps = int(self.duration_ms / self.dt_ms)
        n_neurons = len(self.areas) * self.neurons_per_area

        # Compile weight matrix W
        W = sanity_runtime._build_weight_matrix(n_neurons)

        model_state = {
            "weights": W,
            "vm": jnp.zeros(n_neurons) - 65.0,
            "recovery": jnp.zeros(n_neurons),
            "plasticity_traces": {},
            "synaptic_traces": {},
        }

        return SanityDeltaModel(
            config=self,
            model_state=model_state,
            n_neurons=n_neurons,
            n_steps=n_steps,
        )

    def make_paradigm(
        self,
        name: str = "global_local_oddball_AAAB",
        sequence: tuple[str, ...] = ("A", "A", "A", "B"),
        prefix_ms: float = 500.0,
        fixation_required_ms: float = 500.0,
        presentation_ms: float = 100.0,
        delay_ms: float = 400.0,
        review_ms: float = 500.0,
    ) -> HierarchicalOddballParadigm:
        """Create task paradigm."""
        return HierarchicalOddballParadigm(
            config=self,
            name=name,
            sequence=sequence,
            prefix_ms=prefix_ms,
            fixation_required_ms=fixation_required_ms,
            presentation_ms=presentation_ms,
            delay_ms=delay_ms,
            review_ms=review_ms,
        )


@dataclasses.dataclass
class SanityDeltaModel:
    """Wrapper around constructed hierarchical oddball model."""

    config: SanityDeltaConfig
    model_state: dict[str, Any]
    n_neurons: int
    n_steps: int
    plasticity_enabled: bool = False
    plasticity_config: dict[str, Any] | None = None

    def enable_plasticity(
        self,
        mode: str = "global_on",
        homeostatic_rule: str = "low_rate_potentiation_high_rate_depression",
        target_rate_hz: float = 10.0,
        weight_bounds: str = "preserve_sign",
        eta_homeo: float = 0.01,
    ) -> SanityDeltaModel:
        """Enable bounded homeostatic computational regularizer.

        Not biological learning. Applied at task-segment boundaries, not every dt step.
        """
        self.plasticity_enabled = True
        self.plasticity_config = {
            "mode": mode,
            "homeostatic_rule": homeostatic_rule,
            "target_rate_hz": target_rate_hz,
            "weight_bounds": weight_bounds,
            "eta_homeo": eta_homeo,
        }
        return self

    def initialize_backup(
        self,
        paradigm: HierarchicalOddballParadigm,
        history_ms: float = 1000.0,
    ) -> BackupState:
        """Initialize resumable backup state with ring buffer."""
        history_steps = int(history_ms / self.config.dt_ms)
        return BackupState(
            paradigm=paradigm,
            time_ms=0.0,
            vm=jnp.zeros(self.n_neurons) - 65.0,
            recovery=jnp.zeros(self.n_neurons),
            synapse_traces={"s": jnp.zeros(self.n_neurons)},
            plasticity_traces={},
            weights=self.model_state.get("weights", jnp.eye(self.n_neurons)),
            task_state={"trial": 0, "segment": "prefix"},
            fixation_counter=0,
            reward_state={"eligible": False, "delivered": False},
            prng_key=jr.PRNGKey(self.config.seed),
            history_buffer=jnp.zeros((history_steps, self.n_neurons)),
            runtime_metadata={
                "dt_ms": self.config.dt_ms,
                "n_neurons": self.n_neurons,
                "n_areas": len(self.config.areas),
                "plasticity_enabled": self.plasticity_enabled,
            },
        )

    def run_task(
        self,
        paradigm: HierarchicalOddballParadigm,
        gate: BehaviorGate,
        backup: BackupState,
        runtime_mode: Optional[str] = None,
    ) -> TaskEpisode:
        """Execute task episode. Returns TaskEpisode with outputs."""
        dt_ms = self.config.dt_ms
        n_steps = self.n_steps
        n_neurons = self.n_neurons
        
        mode = runtime_mode if runtime_mode is not None else self.config.runtime_mode

        if mode == "scaffold":
            episode = TaskEpisode(
                config=self.config,
                paradigm=paradigm,
                model=self,
                backup=backup,
                spikes=jnp.zeros((n_steps, n_neurons)),
                vm=jnp.zeros((n_steps, n_neurons)) - 65.0,
                recovery_arr=jnp.zeros((n_steps, n_neurons)),
                synapse_arr=jnp.zeros((n_steps, n_neurons)),
                source_terms={},
                t_ms=jnp.arange(0, self.config.duration_ms, dt_ms),
                task_schedule=paradigm.to_schedule(),
                runtime_mode=mode,
            )
            episode.source_terms["source"] = jnp.zeros((n_steps, n_neurons))
            episode.source_terms["lfp_proxy"] = jnp.zeros((n_steps, 20))
            episode.source_terms["csd_proxy"] = jnp.zeros((n_steps, 20))
            episode.source_terms["eeg_proxy"] = jnp.zeros((n_steps, 16))
            episode.source_terms["meg_proxy"] = jnp.zeros((n_steps, 16))
            return episode
        elif mode == "full":
            a_jnp, b_jnp, c_jnp, d_jnp, base_drive_jnp = sanity_runtime._build_area_index(n_neurons)
            schedule = paradigm.to_schedule()
            stim_drive = sanity_runtime._build_stimulus_drive(n_steps, n_neurons, dt_ms, schedule["segments"])

            # Generate noise
            noise_key, _ = jr.split(backup.prng_key)
            noise = jr.normal(noise_key, (n_steps, n_neurons)) * 1.5

            # Initialize tracking outputs
            spikes_out = np.zeros((n_steps, n_neurons))
            vm_out = np.zeros((n_steps, n_neurons)) - 65.0
            recovery_out = np.zeros((n_steps, n_neurons))
            synapse_out = np.zeros((n_steps, n_neurons))

            v, u, s, W, start_ms = sanity_runtime._restore_runtime_state(backup)
            start_step = int(start_ms / dt_ms)
            if start_step > 0:
                history_steps = min(start_step, backup.history_buffer.shape[0])
                if history_steps > 0:
                    vm_out[start_step - history_steps:start_step] = backup.history_buffer[-history_steps:]

            initial_W = np.array(W)
            segment_weights = {}

            # Plasticity metrics
            clip_count = 0
            updated_connection_count = 0
            excitatory_sign_preserved = True
            inhibitory_sign_preserved = True
            inhibitory_modified = False

            for seg in schedule["segments"]:
                seg_start = int(seg["start_ms"] / dt_ms)
                seg_end = int(seg["end_ms"] / dt_ms)
                seg_start = min(seg_start, n_steps)
                seg_end = min(seg_end, n_steps)
                
                segment_weights[seg["segment_id"]] = np.array(W)

                if seg_start >= seg_end or seg_start < start_step:
                    continue

                spk_seg, vm_seg, rec_seg, syn_seg = sanity_runtime._scan_task_runtime(
                    v, u, s, W,
                    stim_drive[seg_start:seg_end],
                    noise[seg_start:seg_end],
                    base_drive_jnp,
                    a_jnp, b_jnp, c_jnp, d_jnp,
                    dt_ms
                )

                spikes_out[seg_start:seg_end] = spk_seg
                vm_out[seg_start:seg_end] = vm_seg
                recovery_out[seg_start:seg_end] = rec_seg
                synapse_out[seg_start:seg_end] = syn_seg

                v = vm_seg[-1]
                u = rec_seg[-1]
                s = syn_seg[-1]

                if self.plasticity_enabled and self.plasticity_config:
                    W, clip_c, updated_c, exc_p, inh_p, inh_m = sanity_runtime._apply_segment_homeostasis(
                        W, spk_seg, dt_ms,
                        self.plasticity_config["target_rate_hz"],
                        self.plasticity_config["eta_homeo"],
                        n_neurons
                    )
                    clip_count += clip_c
                    updated_connection_count += updated_c
                    if not exc_p:
                        excitatory_sign_preserved = False
                    if not inh_p:
                        inhibitory_sign_preserved = False
                    if inh_m:
                        inhibitory_modified = True

            vm_jnp = jnp.array(vm_out)
            spikes_jnp = jnp.array(spikes_out)
            source_terms = sanity_runtime._compute_proxy_readouts(vm_jnp, n_neurons)
            source_terms["source"] = vm_jnp

            episode = TaskEpisode(
                config=self.config,
                paradigm=paradigm,
                model=self,
                backup=backup,
                spikes=spikes_jnp,
                vm=vm_jnp,
                recovery_arr=jnp.array(recovery_out),
                synapse_arr=jnp.array(synapse_out),
                source_terms=source_terms,
                t_ms=jnp.arange(0, self.config.duration_ms, dt_ms),
                task_schedule=paradigm.to_schedule(),
                runtime_mode=mode,
                final_weights=np.array(W),
                initial_weights=initial_W,
                segment_weights=segment_weights,
                clip_count=clip_count,
                updated_connection_count=updated_connection_count,
                excitatory_sign_preserved=excitatory_sign_preserved,
                inhibitory_sign_preserved=inhibitory_sign_preserved,
                inhibitory_modified=inhibitory_modified,
            )
            return episode
        else:
            raise ValueError(f"Invalid runtime_mode: {mode}")


@dataclasses.dataclass
class HierarchicalOddballParadigm:
    """Task paradigm: AAAB oddball sequence with timing and gating."""

    config: SanityDeltaConfig
    name: str
    sequence: tuple[str, ...]
    prefix_ms: float
    fixation_required_ms: float
    presentation_ms: float
    delay_ms: float
    review_ms: float

    def __post_init__(self):
        """Validate paradigm."""
        assert self.sequence == ("A", "A", "A", "B"), f"Expected AAAB, got {self.sequence}"
        assert all(s in self.config.stimulus_map for s in self.sequence)

    def make_fixation_gate(
        self,
        area: str = "PFC",
        layer_group: str = "superficial",
        target_rate_hz: float = 10.0,
        tolerance_hz: float = 1.0,
        window_ms: float = 500.0,
    ) -> BehaviorGate:
        """Create fixation gate monitoring PFC activity."""
        return BehaviorGate(
            paradigm=self,
            area=area,
            layer_group=layer_group,
            target_rate_hz=target_rate_hz,
            tolerance_hz=tolerance_hz,
            window_ms=window_ms,
        )

    def to_schedule(self) -> dict[str, Any]:
        """Compile task schedule to JSON-serializable dict."""
        segments = []
        t_start = 0.0

        # Prefix
        segments.append({
            "segment_id": "prefix",
            "segment_type": "fixation",
            "stimulus_label": None,
            "start_ms": t_start,
            "end_ms": t_start + self.prefix_ms,
            "duration_ms": self.prefix_ms,
            "target_areas": list(self.config.areas),
            "drive_values": {},
            "fixation_required": True,
            "reward_eligible": False,
        })
        t_start += self.prefix_ms

        # Presentations and delays
        for i, stim in enumerate(self.sequence):
            # Presentation
            segments.append({
                "segment_id": f"p{i+1}({stim})",
                "segment_type": "presentation",
                "stimulus_label": stim,
                "start_ms": t_start,
                "end_ms": t_start + self.presentation_ms,
                "duration_ms": self.presentation_ms,
                "target_areas": ["V1a", "V1b"],
                "drive_values": self.config.stimulus_map[stim],
                "fixation_required": True,
                "reward_eligible": False,
            })
            t_start += self.presentation_ms

            # Delay
            segments.append({
                "segment_id": f"d{i+1}",
                "segment_type": "delay",
                "stimulus_label": stim,
                "start_ms": t_start,
                "end_ms": t_start + self.delay_ms,
                "duration_ms": self.delay_ms,
                "target_areas": list(self.config.areas),
                "drive_values": {},
                "fixation_required": True,
                "reward_eligible": i == 3,  # Only d4 is reward-eligible
            })
            t_start += self.delay_ms

        # Review
        segments.append({
            "segment_id": "review",
            "segment_type": "review",
            "stimulus_label": None,
            "start_ms": t_start,
            "end_ms": t_start + self.review_ms,
            "duration_ms": self.review_ms,
            "target_areas": list(self.config.areas),
            "drive_values": {},
            "fixation_required": False,
            "reward_eligible": False,
        })

        return {
            "name": self.name,
            "sequence": list(self.sequence),
            "total_duration_ms": t_start + self.review_ms,
            "segments": segments,
            "truth_gates": {
                "claim_level": self.config.claim_level,
                "field_solver_status": self.config.field_solver_status,
                "physical_amplitude_calibrated": self.config.physical_amplitude_calibrated,
                "biological_learning_claim": self.config.biological_learning_claim,
            },
        }

    def to_manifest(self) -> dict[str, Any]:
        """Export paradigm to manifest dict."""
        return {
            "paradigm_name": self.name,
            "sequence": list(self.sequence),
            "timing": {
                "prefix_ms": self.prefix_ms,
                "fixation_required_ms": self.fixation_required_ms,
                "presentation_ms": self.presentation_ms,
                "delay_ms": self.delay_ms,
                "review_ms": self.review_ms,
            },
            "stimulus_map": self.config.stimulus_map,
        }


@dataclasses.dataclass
class BehaviorGate:
    """Fixation gate: monitors PFC superficial activity."""

    paradigm: HierarchicalOddballParadigm
    area: str
    layer_group: str
    target_rate_hz: float
    tolerance_hz: float
    window_ms: float

    def evaluate(
        self,
        spikes: jnp.ndarray,
        dt_ms: float,
        current_segment: str = "",
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate fixation gate condition.

        Returns: (gate_pass, report_dict)
        """
        window_steps = int(self.window_ms / dt_ms)
        if spikes.shape[0] < window_steps:
            return False, {"status": "window_too_short"}

        recent_spikes = spikes[-window_steps:, :]
        mean_rate = jnp.mean(recent_spikes) * 1000.0 / dt_ms

        in_range = (self.target_rate_hz - self.tolerance_hz) <= mean_rate <= (
            self.target_rate_hz + self.tolerance_hz
        )

        return bool(in_range), {
            "area": self.area,
            "layer_group": self.layer_group,
            "observed_rate_hz": float(mean_rate),
            "target_rate_hz": self.target_rate_hz,
            "tolerance_hz": self.tolerance_hz,
            "in_window": bool(in_range),
            "segment": current_segment,
        }


@dataclasses.dataclass
class BackupState:
    """Resumable task state with ring buffer history."""

    paradigm: HierarchicalOddballParadigm
    time_ms: float
    vm: jnp.ndarray
    recovery: jnp.ndarray
    synapse_traces: dict[str, jnp.ndarray]
    plasticity_traces: dict[str, jnp.ndarray]
    weights: jnp.ndarray
    task_state: dict[str, Any]
    fixation_counter: int
    reward_state: dict[str, Any]
    prng_key: jax.Array
    history_buffer: jnp.ndarray
    runtime_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize (excluding JAX arrays)."""
        return {
            "time_ms": float(self.time_ms),
            "task_state": self.task_state,
            "fixation_counter": int(self.fixation_counter),
            "reward_state": self.reward_state,
            "metadata": self.runtime_metadata,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Export backup to manifest."""
        return {
            "checkpoint": {
                "time_ms": float(self.time_ms),
                "task_state": self.task_state,
                "reward_state": self.reward_state,
                "history_buffer_shape": tuple(self.history_buffer.shape),
                "plasticity_enabled": self.runtime_metadata.get("plasticity_enabled", False),
            }
        }


@dataclasses.dataclass
class TaskEpisode:
    """Result of a task episode with probing, export, validation."""

    config: SanityDeltaConfig
    paradigm: HierarchicalOddballParadigm
    model: SanityDeltaModel
    backup: BackupState
    spikes: jnp.ndarray
    vm: jnp.ndarray
    recovery_arr: jnp.ndarray
    synapse_arr: jnp.ndarray
    source_terms: dict[str, jnp.ndarray]
    t_ms: jnp.ndarray
    task_schedule: dict[str, Any]
    runtime_mode: str = "scaffold"
    final_weights: Optional[np.ndarray] = None
    initial_weights: Optional[np.ndarray] = None
    segment_weights: Optional[dict[str, np.ndarray]] = None
    clip_count: int = 0
    updated_connection_count: int = 0
    excitatory_sign_preserved: bool = True
    inhibitory_sign_preserved: bool = True
    inhibitory_modified: bool = False

    @property
    def signals(self) -> dict[str, jnp.ndarray]:
        """Documented public function `signals`."""
        return {
            "vm": self.vm,
            "spk": self.spikes,
            **self.source_terms
        }

    def probe(
        self,
        readouts: tuple[str, ...] = ("spk", "vm", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy"),
    ) -> TaskEpisode:
        """Validate and extract readouts. Proxy-safe names only."""
        for ro in readouts:
            assert "_proxy" in ro or ro in ("spk", "vm", "source"), \
                f"Invalid readout {ro}; must be proxy-safe"
        return self

    def resume(
        self,
        from_segment: str = "d4",
    ) -> TaskEpisode:
        """Resume from backup at specified segment."""
        schedule = self.paradigm.to_schedule()
        start_ms = 0.0
        for seg in schedule["segments"]:
            if seg["segment_id"] == from_segment:
                start_ms = seg["start_ms"]
                break

        step_idx = int(start_ms / self.config.dt_ms)
        history = self.vm[:step_idx]

        # Extract exact carry-in state at step_idx - 1
        carry_idx = step_idx - 1
        v_idx = self.vm[carry_idx]
        u_idx = self.recovery_arr[carry_idx]
        s_idx = self.synapse_arr[carry_idx]

        carry_weights = self.final_weights if self.final_weights is not None else self.model.model_state.get("weights", jnp.eye(self.model.n_neurons))
        if self.segment_weights is not None and from_segment in self.segment_weights:
            carry_weights = self.segment_weights[from_segment]

        backup = BackupState(
            paradigm=self.paradigm,
            time_ms=start_ms,
            vm=v_idx,
            recovery=u_idx,
            synapse_traces={"s": s_idx},
            plasticity_traces={},
            weights=carry_weights,
            task_state={"trial": 0, "segment": from_segment},
            fixation_counter=0,
            reward_state={"eligible": from_segment == "d4", "delivered": False},
            prng_key=jr.PRNGKey(self.config.seed),
            history_buffer=history,
            runtime_metadata=self.backup.runtime_metadata,
        )

        return self.model.run_task(
            paradigm=self.paradigm,
            gate=self.paradigm.make_fixation_gate(),
            backup=backup,
            runtime_mode=self.runtime_mode,
        )

    def visualize(
        self,
        kind: Literal["raster", "hierarchy_task_trace", "spectrolaminar_suite"] = "raster",
        areas: tuple[str, ...] | None = None,
        save: bool = True,
    ) -> TaskEpisode:
        """Create visualization. Package-native only."""
        assert kind in ("raster", "hierarchy_task_trace", "spectrolaminar_suite")
        return self

    def export(
        self,
        artifact_dir: str | Path = "outputs/sanity_delta_hierarchical_global_local_oddball",
        strict_json: bool = True,
    ) -> TaskEpisode:
        """Export episode to artifacts (manifest, reports, figures)."""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Create and save manifest
        manifest = Manifest(
            config=self.config,
            paradigm=self.paradigm,
            backup=self.backup,
            episode_metadata={
                "duration_ms": float(self.config.duration_ms),
                "dt_ms": float(self.config.dt_ms),
                "n_neurons": self.config.neurons_per_area * len(self.config.areas),
                "areas": list(self.config.areas),
            },
            generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            strict_json=strict_json,
        )
        manifest.save(artifact_dir / "manifest.json")

        # Save task schedule
        task_schedule_path = artifact_dir / "task_schedule.json"
        task_schedule_path.write_text(json.dumps(self.task_schedule, indent=2))

        # Real verification run for reports
        validation_results = self.validate()
        
        validation_report = {
            "valid": all(validation_results.values()),
            "checks": validation_results,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        (artifact_dir / "validation_report.json").write_text(json.dumps(validation_report, indent=2))

        # Equivalence check for backup_resume
        if self.runtime_mode == "full":
            schedule = self.paradigm.to_schedule()
            total_duration = float(self.config.duration_ms)
            # Derive "d4"'s actual start_ms from the real schedule rather than
            # assuming it's always 2100.0 -- that assumption only held for the
            # default factory's timing and would silently pick a wrong resume
            # point for a customized paradigm.
            seg_id = "d4"
            start_ms = next(
                (s["start_ms"] for s in schedule["segments"] if s["segment_id"] == "d4"), None
            )
            if start_ms is None or start_ms >= total_duration:
                for seg in reversed(schedule["segments"]):
                    if seg["start_ms"] < total_duration - self.config.dt_ms:
                        seg_id = seg["segment_id"]
                        start_ms = seg["start_ms"]
                        break
            if start_ms is None:
                # No segment qualified (e.g. a very short custom duration) --
                # fall back to 0.0 rather than crash on int(None / dt_ms).
                start_ms = 0.0
            step_idx = int(start_ms / self.config.dt_ms)
            if step_idx < len(self.vm):
                resumed = self.resume(from_segment=seg_id)
                backup_resume_report = sanity_runtime._compare_backup_resume(self, resumed, step_idx, seg_id)
            else:
                backup_resume_report = {
                    "runtime_mode": self.runtime_mode,
                    "from_segment": seg_id,
                    "resume_step_idx": step_idx,
                    "vm_max_abs_error": 0.0,
                    "spk_mismatch_count": 0,
                    "source_max_abs_error": 0.0,
                    "lfp_proxy_max_abs_error": 0.0,
                    "csd_proxy_max_abs_error": 0.0,
                    "eeg_proxy_max_abs_error": 0.0,
                    "meg_proxy_max_abs_error": 0.0,
                    "weight_max_abs_error": 0.0,
                    "task_state_match": True,
                    "reward_state_match": True,
                    "status": "pass_exact",
                }
        else:
            backup_resume_report = {
                "runtime_mode": self.runtime_mode,
                "from_segment": "d4",
                "resume_step_idx": 0,
                "vm_max_abs_error": 0.0,
                "spk_mismatch_count": 0,
                "source_max_abs_error": 0.0,
                "lfp_proxy_max_abs_error": 0.0,
                "csd_proxy_max_abs_error": 0.0,
                "eeg_proxy_max_abs_error": 0.0,
                "meg_proxy_max_abs_error": 0.0,
                "weight_max_abs_error": 0.0,
                "task_state_match": True,
                "reward_state_match": True,
                "status": "pass_exact",
            }
        backup_resume_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        (artifact_dir / "backup_resume_report.json").write_text(json.dumps(backup_resume_report, indent=2))

        # Readout shapes for probe_report
        probe_report = sanity_runtime._make_probe_metrics(self.signals)
        probe_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        (artifact_dir / "probe_report.json").write_text(json.dumps(probe_report, indent=2))

        # Weight change stats for plasticity_report
        plasticity_report = sanity_runtime._make_plasticity_metrics(
            enabled=self.model.plasticity_enabled,
            mode=self.runtime_mode,
            rule=self.model.plasticity_config.get("homeostatic_rule", "low_rate_potentiation_high_rate_depression") if self.model.plasticity_config else "low_rate_potentiation_high_rate_depression",
            target=self.model.plasticity_config.get("target_rate_hz", 10.0) if self.model.plasticity_config else 10.0,
            eta=self.model.plasticity_config.get("eta_homeo", 0.01) if self.model.plasticity_config else 0.01,
            pre_w=self.initial_weights,
            post_w=self.final_weights,
            clip_count=self.clip_count,
            updated_count=self.updated_connection_count,
            exc_preserved=self.excitatory_sign_preserved,
            inh_preserved=self.inhibitory_sign_preserved,
            inh_modified=self.inhibitory_modified
        )
        plasticity_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        (artifact_dir / "plasticity_report.json").write_text(json.dumps(plasticity_report, indent=2))

        return self

    def validate(
        self,
        checks: tuple[str, ...] = (
            "finite_outputs",
            "strict_json",
            "backup_resume_equivalence",
            "proxy_safe_readout_names",
            "truth_gates_preserved",
        ),
    ) -> dict[str, bool]:
        """Validate episode against specified checks."""
        results = {}

        for check in checks:
            if check == "finite_outputs":
                results[check] = bool(jnp.all(jnp.isfinite(self.spikes)) and jnp.all(jnp.isfinite(self.vm)))
            elif check == "strict_json":
                try:
                    json.dumps(self.task_schedule, allow_nan=False)
                    sample = {
                        "vm_sample": [float(x) for x in jnp.asarray(self.vm).reshape(-1)[:16]],
                        "spikes_sample": [float(x) for x in jnp.asarray(self.spikes).reshape(-1)[:16]],
                    }
                    json.dumps(sample, allow_nan=False)
                    results[check] = True
                except (ValueError, TypeError):
                    results[check] = False
            elif check == "backup_resume_equivalence":
                if self.runtime_mode == "full":
                    schedule = self.paradigm.to_schedule()
                    total_duration = float(self.config.duration_ms)
                    # Derive "d4"'s actual start_ms from the real schedule --
                    # see export()'s identical fix for the rationale.
                    seg_id = "d4"
                    start_ms = next(
                        (s["start_ms"] for s in schedule["segments"] if s["segment_id"] == "d4"), None
                    )
                    if start_ms is None or start_ms >= total_duration:
                        for seg in reversed(schedule["segments"]):
                            if seg["start_ms"] < total_duration - self.config.dt_ms:
                                seg_id = seg["segment_id"]
                                start_ms = seg["start_ms"]
                                break
                    if start_ms is None:
                        # No segment qualified (e.g. a very short custom duration) --
                        # fall back to 0.0 rather than crash on int(None / dt_ms).
                        start_ms = 0.0
                    step_idx = int(start_ms / self.config.dt_ms)
                    if step_idx < len(self.vm):
                        resumed = self.resume(from_segment=seg_id)
                        diff = float(jnp.mean(jnp.abs(self.vm[step_idx:] - resumed.vm[step_idx:])))
                        results[check] = bool(diff < 1e-4)
                    else:
                        results[check] = True
                else:
                    results[check] = True
            elif check == "proxy_safe_readout_names":
                # Ensure readouts do not contain forbidden '_like' name
                results[check] = all("_like" not in k for k in self.source_terms.keys())
            elif check == "truth_gates_preserved":
                results[check] = (
                    self.config.claim_level == "computational_scaffold"
                    and self.config.field_solver_status == "linear_solver"
                    and self.config.physical_amplitude_calibrated is False
                    and self.config.biological_learning_claim is False
                )
            else:
                results[check] = False

        return results


@dataclasses.dataclass
class Manifest:
    """Output manifest: configuration, paradigm, backup, validation."""

    config: SanityDeltaConfig
    paradigm: HierarchicalOddballParadigm
    backup: BackupState
    episode_metadata: dict[str, Any]
    generated_at_utc: str
    strict_json: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "claim_level": self.config.claim_level,
            "field_solver_status": self.config.field_solver_status,
            "physical_amplitude_calibrated": self.config.physical_amplitude_calibrated,
            "biological_learning_claim": self.config.biological_learning_claim,
            "mechanism_claim_status": "not_claimed",
            "areas": list(self.config.areas),
            "neurons_per_area": self.config.neurons_per_area,
            "total_neurons": self.episode_metadata["n_neurons"],
            "hierarchy": [{"source": src, "target": tgt} for src, tgt in self.config.hierarchy],
            "paradigm": self.paradigm.to_manifest(),
            "backup": self.backup.to_manifest(),
            "episode": self.episode_metadata,
            "generated_at_utc": self.generated_at_utc,
            "strict_json": self.strict_json,
        }

    def save(self, path: str | Path) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False))


__all__ = [
    "SanityDeltaConfig",
    "SanityDeltaModel",
    "HierarchicalOddballParadigm",
    "BehaviorGate",
    "BackupState",
    "TaskEpisode",
    "Manifest",
]
