"""Atlas spec registry, manifests, and inheritance check (0.5.5 ENGINE items 3+4).

One declarative spec per Atlas simulation. Authority for the stage sets is
artifacts/project_sources/8_atlas.md:61-65 (S1: X, Q, Phi; S2-4: + W, B, r,
interaction; S5-7: + N, rho, collective state; S8-9: + area hierarchy, long
delays; S10: + G, D). This module is data over the runner modules, never over
the engine: it performs no simulation and edits no runner.

Reading rule: every run input is a runner module constant read by attribute
at call time (import + getattr), never retyped, so the spec is the run's
inputs. What does not drive the run (arm names, masks, library defaults)
is listed under the per-spec "cited" key with a verbatim anchor string that
tests/test_atlas_manifest_055.py requires to occur in the cited file.
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


def _cite(field: str, file: str, anchor: str, why: str) -> dict[str, str]:
    """A spec value that does not drive the run: code structure or a library default.

    ``anchor`` is a verbatim substring of ``file`` (checked by the test suite),
    so a citation cannot go stale silently the way a line number does.
    """
    return {"field": field, "file": file, "anchor": anchor, "why": why}


_R052 = "artifacts/atlas/at01_at06_052.py"
_R053 = "artifacts/atlas/at07_at04_053.py"
_R054 = "artifacts/atlas/at08_at09_054.py"
_RTOY = "artifacts/atlas/at01_at10_toy.py"
_PRESETS = "jaxfne/_construct_presets.py"


def _field_recording(mod: Any, n_contacts: str) -> dict[str, Any]:
    """Field/probe block shared by the field-probed 052 runs (module constants)."""
    return {
        "field_domain": _const(mod, "FIELD_DOMAIN"),
        "field_conductivity": _const(mod, "FIELD_CONDUCTIVITY"),
        "probe": _const(mod, "PROBE_NAME"),
        "modes": _const(mod, "PROBE_MODES"),
        "n_contacts": _const(mod, n_contacts),
    }


def _spec_at01(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT01_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": _const(mod, "AT01_BUILDER"),
            "hh_current_ua": _const(mod, "AT01_HH_CURRENT_UA"),
        },
        "recording": {"reads": ["spikes", "V_m"]},
        "hdp_params": None,
        "cited": [
            _cite(
                "inputs.n_neurons",
                _PRESETS,
                '.column("single", layers=["uniform_3d"], n=1)',
                "builder default: one neuron",
            ),
            _cite(
                "recording.field",
                _PRESETS,
                ".probes(_SUITE2_PROXY_MODES, n_contacts=4)",
                "builder-default probes; the runner adds none",
            ),
            _cite(
                "recording.reads",
                _R052,
                "vm = np.asarray(sig.V_m).ravel()",
                "Signals attributes the reduced arm reads",
            ),
        ],
    }


def _spec_at02(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT02_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "tfne_spec": _const(mod, "AT02_SPEC_DELAY"),
            "absent_delay_spec": _const(mod, "AT02_SPEC_ABSENT"),
            "delay_ms": _const(mod, "DELAY_MS"),
            "zero_delay_ms": _const(mod, "AT02_ZERO_DELAY_MS"),
        },
        "recording": _field_recording(mod, "AT02_N_CONTACTS"),
        "hdp_params": None,
        "cited": [
            _cite("arms", _R052, '"zero": _tfne_pair_run(', "arm names delayed/zero/absent"),
        ],
    }


def _spec_at03(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT03_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {"tfne_spec": _const(mod, "AT03_SPEC"), "delay_ms": _const(mod, "DELAY_MS")},
        "recording": _field_recording(mod, "AT02_N_CONTACTS"),
        "hdp_params": None,
        "cited": [],
    }


def _spec_at04(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT04_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "tfne_spec": _const(mod, "AT04_SPEC"),
            "geometry_arms": _const(mod, "AT04_ARMS"),
            "delay_ms": _const(mod, "DELAY_MS"),
        },
        "recording": _field_recording(mod, "AT02_N_CONTACTS"),
        "hdp_params": None,
        "cited": [],
    }


def _spec_at05(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT05_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": _const(mod, "AT05_BUILDER"),
            "arms": _const(mod, "AT05_ARMS"),
            "ratio_margin": _const(mod, "AT05_RATIO_MARGIN"),
        },
        "recording": _field_recording(mod, "AT06_N_CONTACTS"),
        "hdp_params": None,
        "cited": [],
    }


def _spec_at06(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED"), "run": _const(mod, "SEED")},
        "run": {"duration_ms": _const(mod, "AT06_DURATION_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "area": _const(mod, "AT06_AREA"),
            "layers": _const(mod, "AT06_LAYERS"),
            "n": _const(mod, "AT06_N"),
            "cell_types": _const(mod, "AT06_CELL_TYPES"),
            "connectivity": _const(mod, "AT06_CONNECTIVITY"),
            "emitter": _const(mod, "AT06_EMITTER"),
            "radii_frac": _const(mod, "AT06_RADII_FRAC"),
            "bands_hz": _const(mod, "AT06_BANDS_HZ"),
        },
        "recording": {
            **_field_recording(mod, "AT06_N_CONTACTS"),
            "position": _const(mod, "AT06_PROBE_POSITION"),
            "reference": _const(mod, "AT06_REFERENCE"),
            "filter": _const(mod, "AT06_FILTER"),
        },
        "hdp_params": None,
        "cited": [],
    }


def _spec_at07(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": _const(mod, "BUILDER"),
            "n_neurons": _const(mod, "N_NEURONS"),
            "clamp_value": _const(mod, "CLAMP_VALUE"),
        },
        "recording": {
            "H_trace": "full trajectory + budgeted decimation",
            "w_trace": "full trajectory + budgeted decimation",
            "record_weight_trace": _const(mod, "BUDGET_WEIGHT_TRACE"),
            "record_stride": _const(mod, "BUDGET_STRIDE"),
            "record_h_subset": _const(mod, "BUDGET_H_SUBSET"),
            "record_w_subset": _const(mod, "BUDGET_W_SUBSET"),
        },
        "hdp_params": {"baseline": _const(mod, "BASE_HP"), "noisy": _const(mod, "NOISY_HP")},
        "cited": [
            _cite(
                "arms",
                _R053,
                '"hebbian": _run_arm(models[0], BASE_HP),',
                "arm names hebbian/fixed/noisy/clamp",
            ),
            _cite(
                "clamp_mask",
                _R053,
                "values=[0.0] * (n_edges // 2) + [1.0] * (n_edges - n_edges // 2)",
                "first half of edges pinned, second half free",
            ),
        ],
    }


def _spec_at04r2(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": {
            "builder": _const(mod, "BUILDER"),
            "n_neurons": _const(mod, "N_NEURONS"),
            "H0_baseline": _const(mod, "H0_BASELINE"),
            "H0_perturbed": _const(mod, "H0_PERTURBED"),
        },
        "recording": {
            "summary": "spike_count/rate/H_final/w_final/Q/Phi/kappa per arm",
        },
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "cited": [
            _cite(
                "arms",
                _R053,
                "model_dis_pert = built[3].with_hdp_initial_state(",
                "baseline/perturbed (HDP engaged) + disabled pair (W bit-fixed)",
            ),
            _cite(
                "disable_control",
                _R053,
                "hp_off = J.hdp_network.disable_plasticity(",
                "disabled arms use the public disable_plasticity control",
            ),
            _cite("recording.summary", _R053, "def _arm_summary(", "per-arm summary reads"),
        ],
    }


def _ensemble_inputs(mod: Any) -> dict[str, Any]:
    """Two-area composition inputs shared by AT-08/AT-09 (module constants)."""
    return {
        "member_areas": _const(mod, "MEMBER_AREAS"),
        "member_layers": _const(mod, "MEMBER_LAYERS"),
        "member_build_seeds": [_const(mod, "SEED_BUILD") + i for i in range(2)],
        "members_cell_types": _const(mod, "MEMBER_CELL_TYPES"),
        "members_emitter_family": _const(mod, "MEMBER_EMITTER_FAMILY"),
        "n_per_area": _const(mod, "N_PER_AREA"),
        "cross_delay_ms": _const(mod, "CROSS_DELAY_MS"),
        "edge_source_cell_type": _const(mod, "EDGE_SOURCE_CELL_TYPE"),
        "edge_probability": _const(mod, "EDGE_PROBABILITY"),
        "edge_weight": _const(mod, "EDGE_WEIGHT"),
        "edge_sign": _const(mod, "EDGE_SIGN"),
        "stimulus": "repeated pulses to A1",
        "stim_period_ms": _const(mod, "STIM_PERIOD_MS"),
        "stim_duration_ms": _const(mod, "STIM_DUR_MS"),
        "stim_amplitude": _const(mod, "STIM_AMP"),
    }


def _ensemble_recording(mod: Any) -> dict[str, Any]:
    return {
        "probe_modes": _const(mod, "PROBE_MODES"),
        "n_contacts": _const(mod, "N_CONTACTS"),
        "field_domain": _const(mod, "FIELD_DOMAIN"),
        "field_conductivity": _const(mod, "FIELD_CONDUCTIVITY"),
        "field_boundary": _const(mod, "FIELD_BOUNDARY"),
    }


_ENSEMBLE_CITED = [
    _cite("edges", _R054, 'target={"model": 1, "area": "A2"},', "rule 0 = A1->A2, rule 1 = A2->A1"),
    _cite(
        "members_emitter_preset",
        _R054,
        ".set_emitter(family=MEMBER_EMITTER_FAMILY)",
        "family default preset (no preset argument)",
    ),
]


def _spec_at08(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": _ensemble_inputs(mod),
        "recording": _ensemble_recording(mod),
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "cited": [
            *_ENSEMBLE_CITED,
            _cite(
                "arms",
                _R054,
                '"adapt_full": _run_arm(model, dict(BASE_HP), stim),',
                "arm names fixed/adapt_full/adapt_local",
            ),
            _cite(
                "masks",
                _R054,
                'return {"cross_only": cross, "member_only": member',
                "cross/member ranges from ensemble_edge_ownership",
            ),
        ],
    }


def _spec_at09(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "SEED_BUILD"), "run": _const(mod, "RUN_SEED")},
        "run": {"duration_ms": _const(mod, "DUR_MS"), "dt_ms": _const(mod, "DT_MS")},
        "inputs": _ensemble_inputs(mod),
        "recording": _ensemble_recording(mod),
        "hdp_params": {"baseline": _const(mod, "BASE_HP")},
        "cited": [
            *_ENSEMBLE_CITED,
            _cite(
                "arms",
                _R054,
                '"plastic_cross": _run_arm(',
                "arm names plastic_cross/plastic_member/frozen",
            ),
            _cite(
                "masks",
                _R054,
                'return {"cross_only": cross, "member_only": member',
                "cross_only/member_only/frozen",
            ),
        ],
    }


def _spec_at10(mod: Any) -> dict[str, Any]:
    return {
        "seeds": {"build": _const(mod, "TOY_SEED"), "run": _const(mod, "TOY_SEED")},
        "run": {"duration_ms": _const(mod, "AT10_DURATION_MS"), "dt_ms": _const(mod, "TOY_DT_MS")},
        "inputs": {
            "builder": _const(mod, "AT10_BUILDER"),
            "areas": _const(mod, "AT10_AREAS"),
            "n_per_area": _const(mod, "AT10_N_PER_AREA"),
            "emitter": _const(mod, "AT10_EMITTER"),
        },
        "recording": {"field": _const(mod, "AT10_FIELD"), "probes": _const(mod, "AT10_PROBES")},
        "hdp_params": None,
        "cited": [
            _cite(
                "genome_development", _RTOY, '"genome_development": "OMITTED', "toy owns no G->D"
            ),
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
