"""Atlas spec registry, manifests, and inheritance check (0.5.5 ENGINE items 3+4).

One declarative spec per Atlas simulation. Authority for the stage sets is
artifacts/project_sources/8_atlas.md:61-65 (S1: X, Q, Phi; S2-4: + W, B, r,
interaction; S5-7: + N, rho, collective state; S8-9: + area hierarchy, long
delays; S10: + G, D). This module is data over the runner modules, never over
the engine: it performs no simulation and edits no runner.

Reading rule: every value that exists as a runner module constant is read by
attribute at call time (import + getattr), never retyped. Values that are
inline literals in a run function are recorded verbatim and listed under the
per-spec "transcribed" key with their file:line, so the gap stays visible.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

_ATLAS_PKG = "artifacts.atlas"

# The only place Atlas stage components are written. Each stage holds the full
# cumulative set (S1 subset S2-4 subset S5-7 subset S8-9 subset S10).
STAGE_COMPONENTS: dict[str, list[str]] = {
    "S1": ["X", "Q", "Phi"],
    "S2-4": ["X", "Q", "Phi", "W", "B", "r", "interaction"],
    "S5-7": ["X", "Q", "Phi", "W", "B", "r", "interaction", "N", "rho", "collective_state"],
    "S8-9": [
        "X",
        "Q",
        "Phi",
        "W",
        "B",
        "r",
        "interaction",
        "N",
        "rho",
        "collective_state",
        "area_hierarchy",
        "long_delays",
    ],
    "S10": [
        "X",
        "Q",
        "Phi",
        "W",
        "B",
        "r",
        "interaction",
        "N",
        "rho",
        "collective_state",
        "area_hierarchy",
        "long_delays",
        "G",
        "D",
    ],
}

STAGE_ORDER: list[str] = ["S1", "S2-4", "S5-7", "S8-9", "S10"]

_STAGE_PARENT: dict[str, str | None] = {
    "S1": None,
    "S2-4": "S1",
    "S5-7": "S2-4",
    "S8-9": "S5-7",
    "S10": "S8-9",
}

# Canonical runner per AT id. AT-10 is the toy composition pattern; the real
# 20-area AT-10 is future work, hence "toy".
REGISTRY: dict[str, dict[str, str]] = {
    "AT-01": {"runner": "at01_at06_052:run_at01", "status": "canonical", "stage": "S1"},
    "AT-02": {"runner": "at01_at06_052:run_at02", "status": "canonical", "stage": "S2-4"},
    "AT-03": {"runner": "at01_at06_052:run_at03", "status": "canonical", "stage": "S2-4"},
    "AT-04": {"runner": "at01_at06_052:run_at04", "status": "canonical", "stage": "S2-4"},
    "AT-05": {"runner": "at01_at06_052:run_at05", "status": "canonical", "stage": "S5-7"},
    "AT-06": {"runner": "at01_at06_052:run_at06", "status": "canonical", "stage": "S5-7"},
    "AT-07": {"runner": "at07_at04_053:run_at07", "status": "canonical", "stage": "S5-7"},
    "AT-04-R2": {"runner": "at07_at04_053:run_at04r2", "status": "canonical", "stage": "S2-4"},
    "AT-08": {"runner": "at08_at09_054:run_at08", "status": "canonical", "stage": "S8-9"},
    "AT-09": {"runner": "at08_at09_054:run_at09", "status": "canonical", "stage": "S8-9"},
    "AT-10": {"runner": "at01_at10_toy:run_at10", "status": "toy", "stage": "S10"},
}


def _runner_module(at_id: str) -> tuple[Any, str]:
    """Import the runner module for an AT id; return (module, function name)."""
    runner = REGISTRY[at_id]["runner"]
    mod_name, func_name = runner.split(":")
    return importlib.import_module(f"{_ATLAS_PKG}.{mod_name}"), func_name


def _freeze(value: Any) -> Any:
    """Convert module constants to JSON-safe structures (tuples become lists)."""
    if isinstance(value, dict):
        return {str(k): _freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_freeze(v) for v in value]
    return value


def _const(mod: Any, name: str) -> Any:
    """Read one runner module constant by attribute (never retyped here)."""
    return _freeze(getattr(mod, name))


def resolve_runner(at_id: str) -> Any:
    """Return the runner callable for an AT id (raises KeyError/AttributeError)."""
    mod, func_name = _runner_module(at_id)
    func = getattr(mod, func_name)
    if not callable(func):
        raise TypeError(f"runner {REGISTRY[at_id]['runner']!r} is not callable")
    return func


def _spec_at01(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT01_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": "suite2_single_neuron_config",
            "n_neurons": 1,
            "hh_current_ua": _const(mod, "AT01_HH_CURRENT_UA"),
        },
        "recording": {"spikes": "signals.spikes", "V_m": "signals.V_m", "field": "none"},
        "hdp_params": None,
        "transcribed": [
            "inputs.builder: at01_at06_052.py:321",
            "inputs.n_neurons: at01_at06_052.py:321",
            "recording.spikes: at01_at06_052.py:326",
            "recording.V_m: at01_at06_052.py:327",
            "recording.field: at01_at06_052.py:320 (no field probes on the reduced arm)",
        ],
    }


def _pair_recording(mod: Any) -> tuple[dict[str, Any], list[str]]:
    """Field/probe block shared by the TFNE pair runs (literals live in the helper)."""
    rec = {
        "field_domain": "laminar_column",
        "field_conductivity": "proxy",
        "probe": "e1",
        "modes": ["spikes", "V_m", "source", "LFP-proxy"],
        "n_contacts": _const(mod, "AT02_N_CONTACTS"),
    }
    cited = [
        "recording.field_domain: at01_at06_052.py:448",
        "recording.field_conductivity: at01_at06_052.py:448",
        "recording.probe: at01_at06_052.py:449",
        "recording.modes: at01_at06_052.py:451",
    ]
    return rec, cited


def _spec_at02(mod: Any) -> dict[str, Any]:
    rec, cited = _pair_recording(mod)
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT02_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "tfne_spec": _const(mod, "AT02_SPEC_DELAY"),
            "absent_delay_spec": _const(mod, "AT02_SPEC_ABSENT"),
            "delay_ms": _const(mod, "DELAY_MS"),
            "arms": {
                "delayed": "delay_ms=DELAY_MS",
                "zero": "delay_ms=0.0",
                "absent": "no delay key",
            },
        },
        "recording": rec,
        "hdp_params": None,
        "transcribed": [
            "inputs.arms: at01_at06_052.py:526 (arm names; zero arm delay 0.0: :527)",
            *cited,
        ],
    }


def _spec_at03(mod: Any) -> dict[str, Any]:
    rec, cited = _pair_recording(mod)
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT03_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {"tfne_spec": _const(mod, "AT03_SPEC"), "delay_ms": _const(mod, "DELAY_MS")},
        "recording": rec,
        "hdp_params": None,
        "transcribed": [*cited],
    }


def _spec_at04(mod: Any) -> dict[str, Any]:
    rec, cited = _pair_recording(mod)
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT04_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "tfne_spec": _const(mod, "AT04_SPEC"),
            "geometry_arms": _const(mod, "AT04_ARMS"),
            "delay_ms": _const(mod, "DELAY_MS"),
        },
        "recording": rec,
        "hdp_params": None,
        "transcribed": [*cited],
    }


def _spec_at05(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT05_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": "suite2_net1_config",
            "arms": _const(mod, "AT05_ARMS"),
            "ratio_margin": _const(mod, "AT05_RATIO_MARGIN"),
        },
        "recording": {
            "field_domain": "laminar_column",
            "field_conductivity": "proxy",
            "probe": "e1",
            "modes": ["spikes", "V_m", "source", "LFP-proxy"],
            "n_contacts": _const(mod, "AT06_N_CONTACTS"),
        },
        "hdp_params": None,
        "transcribed": [
            "inputs.builder: at01_at06_052.py:841",
            "recording.field_domain: at01_at06_052.py:842",
            "recording.field_conductivity: at01_at06_052.py:842",
            "recording.probe: at01_at06_052.py:843",
            "recording.modes: at01_at06_052.py:844",
        ],
    }


def _spec_at06(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT06_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "column": "V1 [L2/3, L4] x8",
            "cell_types": "E 0.75 / PV 0.25",
            "connectivity": "laminar_signed_metadata recurrent",
            "emitter": "izhikevich/cortical_eig",
            "radii_frac": _const(mod, "AT06_RADII_FRAC"),
            "bands_hz": _const(mod, "AT06_BANDS_HZ"),
        },
        "recording": {
            "field_domain": "laminar_column",
            "field_conductivity": "proxy",
            "probe": "e1",
            "modes": ["spikes", "V_m", "source", "LFP-proxy"],
            "n_contacts": _const(mod, "AT06_N_CONTACTS"),
            "position": "[0.0, 1.0/3.0, 2.0/3.0, 1.0]",
            "reference": "common_average (record-only)",
            "filter": "bandpass 8-25 Hz (record-only)",
        },
        "hdp_params": None,
        "transcribed": [
            "inputs.column: at01_at06_052.py:933",
            "inputs.cell_types: at01_at06_052.py:934",
            "inputs.connectivity: at01_at06_052.py:935",
            "inputs.emitter: at01_at06_052.py:936",
            "recording.field_domain: at01_at06_052.py:937",
            "recording.field_conductivity: at01_at06_052.py:937",
            "recording.probe: at01_at06_052.py:938",
            "recording.modes: at01_at06_052.py:940",
            "recording.position: at01_at06_052.py:941",
            "recording.reference: at01_at06_052.py:942",
            "recording.filter: at01_at06_052.py:943",
        ],
    }


def _spec_at07(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": "suite2_net1_config",
            "n_neurons": _const(mod, "N_NEURONS"),
            "arms": ["hebbian", "fixed", "noisy", "clamp"],
            "clamp_value": _const(mod, "CLAMP_VALUE"),
        },
        "recording": {
            "H_trace": "full trajectory + budgeted decimation",
            "w_trace": "full trajectory + budgeted decimation",
            "record_weight_trace": True,
            "record_stride": _const(mod, "BUDGET_STRIDE"),
            "record_h_subset": _const(mod, "BUDGET_H_SUBSET"),
            "record_w_subset": [0, 1, 2],
        },
        "hdp_params": {"baseline": _const(mod, "BASE_HP"), "noisy": _const(mod, "NOISY_HP")},
        "transcribed": [
            "inputs.builder: at07_at04_053.py:184",
            "inputs.arms: at07_at04_053.py:399",
            "inputs.clamp_mask_values: at07_at04_053.py:392 (first half pinned, second half free)",
            "recording.record_weight_trace: at07_at04_053.py:467",
            "recording.record_w_subset: at07_at04_053.py:470",
        ],
    }


def _spec_at04r2(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": "suite2_net1_config",
            "n_neurons": _const(mod, "N_NEURONS"),
            "H0_baseline": _const(mod, "H0_BASELINE"),
            "H0_perturbed": _const(mod, "H0_PERTURBED"),
            "arms": "baseline/perturbed (HDP engaged) + disabled pair (W bit-fixed)",
            "disable_control": "disable_plasticity",
        },
        "recording": {
            "summary": "spike_count/rate/H_final/w_final/Q/Phi/kappa per arm",
        },
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "transcribed": [
            "inputs.builder: at07_at04_053.py:184",
            "inputs.arms: at07_at04_053.py:584 (H0 arrays; identical W0, matched stimulation)",
            "inputs.disable_control: at07_at04_053.py:566",
            "recording.summary: at07_at04_053.py:235 (_arm_summary signals read)",
        ],
    }


def _ensemble_inputs(mod: Any) -> tuple[dict[str, Any], list[str]]:
    """Two-area composition inputs shared by AT-08/AT-09 (edge literals cited)."""
    inputs = {
        "members": "2x Configuration columns A1/A2",
        "member_build_seeds": [_const(mod, "SEED_BUILD") + i for i in range(2)],
        "members_cell_types": {"E": 0.8, "PV": 0.2},
        "members_emitter": "izhikevich (family default preset)",
        "n_per_area": _const(mod, "N_PER_AREA"),
        "cross_delay_ms": _const(mod, "CROSS_DELAY_MS"),
        "edges": "A1->A2 + A2->A1, E source, probability 0.5, weight 0.5, excitatory",
        "stimulus": "repeated pulses to A1",
        "stim_period_ms": _const(mod, "STIM_PERIOD_MS"),
        "stim_duration_ms": _const(mod, "STIM_DUR_MS"),
        "stim_amplitude": _const(mod, "STIM_AMP"),
    }
    cited = [
        "inputs.members: at08_at09_054.py:173 (A1/A2 loop) + :179 (column layers/n)",
        "inputs.member_build_seeds: at08_at09_054.py:177 (SEED_BUILD + i per area)",
        "inputs.members_cell_types: at08_at09_054.py:180",
        "inputs.members_emitter: at08_at09_054.py:182",
        "inputs.edges: at08_at09_054.py:197 (rule order: rule 0 = A1->A2, rule 1 = A2->A1)",
    ]
    return inputs, cited


def _ensemble_recording(mod: Any) -> tuple[dict[str, Any], list[str]]:
    rec = {
        "probe_modes": ["spikes", "V_m", "LFP", "CSD"],
        "n_contacts": 8,
        "field_domain": "laminar_column",
        "field_conductivity": "proxy",
        "field_boundary": "mean_zero_neumann",
    }
    cited = [
        "recording.probe_modes: at08_at09_054.py:183",
        "recording.n_contacts: at08_at09_054.py:183",
        "recording.field_domain: at08_at09_054.py:184",
        "recording.field_conductivity: at08_at09_054.py:184",
        "recording.field_boundary: at08_at09_054.py:184",
    ]
    return rec, cited


def _spec_at08(mod: Any) -> dict[str, Any]:
    inputs, in_cited = _ensemble_inputs(mod)
    inputs["arms"] = ["fixed", "adapt_full", "adapt_local"]
    rec, rec_cited = _ensemble_recording(mod)
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": inputs,
        "recording": rec,
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "transcribed": [
            *in_cited,
            *rec_cited,
            "inputs.arms: at08_at09_054.py:440",
            "inputs.masks: at08_at09_054.py:416 (cross/member ranges from ensemble_edge_ownership)",
        ],
    }


def _spec_at09(mod: Any) -> dict[str, Any]:
    inputs, in_cited = _ensemble_inputs(mod)
    inputs["arms"] = ["plastic_cross", "plastic_member", "frozen"]
    rec, rec_cited = _ensemble_recording(mod)
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": inputs,
        "recording": rec,
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "transcribed": [
            *in_cited,
            *rec_cited,
            "inputs.arms: at08_at09_054.py:518",
            "inputs.masks: at08_at09_054.py:416 (cross_only/member_only/frozen)",
        ],
    }


def _spec_at10(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "TOY_SEED"), "run": _const(mod, "TOY_SEED")},
        "run": {"duration_ms": 10.0, "dt_ms": _const(mod, "TOY_DT_MS")},
        "inputs": {
            "builder": "build_multi_area_columns",
            "areas": ["A1", "A2", "A3"],
            "n_per_area": 2,
            "emitter": "izhikevich/cortical_eig",
            "genome_development": "OMITTED",
        },
        "recording": {"probes": ["spikes", "V_m"]},
        "hdp_params": None,
        "transcribed": [
            "inputs.builder: at01_at10_toy.py:254",
            "inputs.areas: at01_at10_toy.py:254",
            "inputs.n_per_area: at01_at10_toy.py:254",
            "run.duration_ms: at01_at10_toy.py:256 (also :261 in the _run_config call)",
            "inputs.emitter: at01_at10_toy.py:257",
            "recording.probes: at01_at10_toy.py:259",
            "inputs.genome_development: at01_at10_toy.py:265 (toy owns no G->D)",
        ],
    }


_BUILDERS = {
    "AT-01": _spec_at01,
    "AT-02": _spec_at02,
    "AT-03": _spec_at03,
    "AT-04": _spec_at04,
    "AT-05": _spec_at05,
    "AT-06": _spec_at06,
    "AT-07": _spec_at07,
    "AT-04-R2": _spec_at04r2,
    "AT-08": _spec_at08,
    "AT-09": _spec_at09,
    "AT-10": _spec_at10,
}


def at_spec(at_id: str) -> dict[str, Any]:
    """Declarative JSON-safe spec for one Atlas simulation (no simulation)."""
    entry = REGISTRY[at_id]
    mod, _ = _runner_module(at_id)
    stage = entry["stage"]
    body = _BUILDERS[at_id](mod)
    spec = {
        "at_id": at_id,
        "runner": entry["runner"],
        "status": entry["status"],
        "stage": stage,
        **body,
        "components": sorted(STAGE_COMPONENTS[stage]),
        "parent_stage": _STAGE_PARENT[stage],
    }
    return json.loads(json.dumps(spec))


def spec_digest(spec: dict[str, Any]) -> str:
    """sha256 over the canonical JSON encoding of a spec."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pkg_version(dist: str) -> str:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return "UNKNOWN"


def _git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=root,
        )
        sha = out.stdout.strip()
        return sha if sha else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def manifest(at_id: str) -> dict[str, Any]:
    """Spec + digest + lineage + environment for one Atlas simulation (no simulation)."""
    spec = at_spec(at_id)
    return {
        "spec": spec,
        "spec_digest": spec_digest(spec),
        "lineage": {"parent_stage": spec["parent_stage"], "runner": spec["runner"]},
        "environment": {
            "python": platform.python_version(),
            "jax": _pkg_version("jax"),
            "jaxlib": _pkg_version("jaxlib"),
            "numpy": _pkg_version("numpy"),
            "jaxfne": _pkg_version("jaxfne"),
            "git_sha": _git_sha(),
        },
    }


def inheritance_check() -> list[str]:
    """Mechanical check over declared-object sets; empty list means pass."""
    violations: list[str] = []
    try:
        sets = {stage: set(STAGE_COMPONENTS[stage]) for stage in STAGE_ORDER}
    except KeyError as exc:
        return [f"STAGE_COMPONENTS missing stage {exc}"]
    for parent, child in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        dropped = sets[parent] - sets[child]
        if dropped:
            violations.append(f"{child} drops parent components {sorted(dropped)} from {parent}")
        if not sets[child] - sets[parent]:
            violations.append(f"{child} adds no component over {parent}")
    for at_id, entry in REGISTRY.items():
        try:
            got = at_spec(at_id)["components"]
        except Exception as exc:
            violations.append(f"{at_id}: spec unreadable ({exc!r})")
            continue
        want = sorted(STAGE_COMPONENTS[entry["stage"]])
        if got != want:
            violations.append(f"{at_id}: components {got} != stage {entry['stage']} set {want}")
    return violations
