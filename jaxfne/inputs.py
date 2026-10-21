"""Frozen input-source declarations (T-24).

An ``XSource`` is a frozen record of one configured external input: a
constant drive channel or a Poisson drive stream. The numeric apply paths
compile the existing declarations (``Configuration.drive()``, per-type
``cell_params(drive=...)``, ``NeuronType.drive`` bridged as
``tensor_triple_drive`` metadata, emitter-table defaults, and
``Simulation(poisson_drive=...)``) and read their values back out in the
same order and dtype as before, without changing numerics;
``Model.x_sources`` derives the constant entries at access from the
realized per-neuron drive, grouped by the population's own labels.

Units: ``"native_current"`` is the Izhikevich/LIF reduced-model drive unit;
``"pA"`` is the AdEx external-current unit (one declared drive unit reads
as one pA there); Poisson ``value`` is always the rate in ``"Hz"``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping


_CONSTANT_UNITS = ("native_current", "pA")
_POISSON_UNIT = "Hz"

# Drive() vocabulary order: default entries are pinned in this order with
# emitter-table values.
DRIVE_VOCABULARY_ORDER = ("E", "PV", "SST", "VIP")

# Schedule keys carried on Poisson entries: the exact existing mappings.
# Current route: (("amplitude", float),), plus ("counts", int) when the
# declaration names it. AdEx route: n_sources,
# weight_nS, mechanism, delay_ms in this order, plus optionally the
# t_start_ms/t_stop_ms window pair (both or neither).
_POISSON_CURRENT_SCHEDULE_KEYS = ("amplitude", "counts")
_POISSON_ADEX_SCHEDULE_KEYS = ("n_sources", "weight_nS", "mechanism", "delay_ms")
_POISSON_WINDOW_KEYS = ("t_start_ms", "t_stop_ms")
_POISSON_SCHEDULE_KEYS = frozenset(
    (*_POISSON_CURRENT_SCHEDULE_KEYS, *_POISSON_ADEX_SCHEDULE_KEYS, *_POISSON_WINDOW_KEYS)
)


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class XSource:
    """One frozen configured input source.

    ``targets`` is a non-empty tuple of strings: a single cell type for a
    baseline entry (``("E",)``), an ``(area, layer, cell_type)`` triple for a
    tensor-triple entry, ``cell_type``/``"layer=X"`` tokens for a
    ``cell_params(drive=...)`` entry, or the drive target(s) for Poisson.
    ``value`` is the constant drive (constant) or the rate in Hz (poisson).
    ``schedule`` is None or a tuple of ``(key, value)`` pairs in the fixed
    order documented above; a window pair must satisfy
    ``0 <= t_start_ms < t_stop_ms``. ``seed`` is None (constant) or the
    declared Poisson seed (poisson). ``n_neurons`` is None or the count of
    neurons carrying exactly this entry's target and value (constant entries
    derived from a realized model always carry it, so the per-neuron drive
    is reconstructible as each value repeated ``n_neurons`` times).
    """

    kind: str
    targets: tuple[str, ...]
    value: float
    unit: str
    schedule: tuple[tuple[str, Any], ...] | None = None
    seed: int | None = None
    n_neurons: int | None = None

    def __post_init__(self) -> None:
        kind = self.kind
        if kind not in ("constant", "poisson"):
            raise ValueError(f"unknown XSource kind {kind!r}; expected 'constant' or 'poisson'")
        targets = self.targets
        if (
            not isinstance(targets, tuple)
            or not targets
            or any(not isinstance(t, str) or not t for t in targets)
        ):
            raise ValueError(f"XSource targets must be a non-empty tuple of strings; got {targets!r}")
        if not _is_finite_real(self.value):
            raise ValueError(f"XSource({kind}) value takes a finite real number; got {self.value!r}")
        if kind == "poisson" and float(self.value) < 0.0:
            raise ValueError(f"poisson XSource rate_hz={self.value!r} must be >= 0")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError(f"XSource unit must be a non-empty string; got {self.unit!r}")
        if self.n_neurons is not None and (
            isinstance(self.n_neurons, bool)
            or not isinstance(self.n_neurons, int)
            or self.n_neurons < 1
        ):
            raise ValueError(
                f"XSource({kind}) n_neurons={self.n_neurons!r} takes an int >= 1"
            )
        if kind == "constant":
            if self.unit not in _CONSTANT_UNITS:
                raise ValueError(
                    f"constant XSource unit {self.unit!r} is not realized: "
                    f"expected one of {list(_CONSTANT_UNITS)}"
                )
            if self.schedule is not None:
                raise ValueError(
                    f"constant XSource takes no schedule; got {self.schedule!r}"
                )
            if self.seed is not None:
                raise ValueError(f"constant XSource takes no seed; got {self.seed!r}")
            return
        # poisson below.
        if self.unit != _POISSON_UNIT:
            raise ValueError(
                f"poisson XSource unit {self.unit!r} is not realized: expected {_POISSON_UNIT!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"poisson XSource seed must be an int; got {self.seed!r}")
        schedule = self.schedule
        if schedule is None:
            return
        if not isinstance(schedule, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str)
            for pair in schedule
        ):
            raise ValueError(
                f"poisson XSource schedule takes (key, value) pairs; got {schedule!r}"
            )
        keys = [pair[0] for pair in schedule]
        unknown = [k for k in keys if k not in _POISSON_SCHEDULE_KEYS]
        if unknown:
            raise ValueError(
                f"poisson XSource schedule has unknown keys {unknown!r}; "
                f"expected a subset of {sorted(_POISSON_SCHEDULE_KEYS)}"
            )
        table = dict(schedule)
        if "amplitude" in table and set(keys) - {"amplitude", "counts"}:
            raise ValueError(
                f"poisson XSource schedule {schedule!r} is not realized: 'amplitude' "
                "carries the current-route mapping, alone or with 'counts'"
            )
        if "counts" in table:
            counts = table["counts"]
            if (
                isinstance(counts, bool)
                or not isinstance(counts, numbers.Integral)
                or int(counts) < 1
            ):
                raise ValueError(
                    f"poisson XSource counts={counts!r} must be a positive integer"
                )
        if "amplitude" in table and not _is_finite_real(table["amplitude"]):
            raise ValueError(
                f"poisson XSource amplitude={table['amplitude']!r} takes a finite real number"
            )
        adex_keys = [k for k in keys if k in _POISSON_ADEX_SCHEDULE_KEYS]
        if adex_keys and set(adex_keys) != set(_POISSON_ADEX_SCHEDULE_KEYS):
            raise ValueError(
                f"poisson XSource schedule {schedule!r} is not realized: the AdEx mapping "
                f"carries exactly {list(_POISSON_ADEX_SCHEDULE_KEYS)}"
            )
        if adex_keys:
            n_sources = table["n_sources"]
            if isinstance(n_sources, bool) or not isinstance(n_sources, numbers.Real):
                raise ValueError(
                    f"poisson XSource n_sources={n_sources!r} must be an int >= 1"
                )
            if float(n_sources) != float(int(n_sources)) or int(n_sources) < 1:
                raise ValueError(
                    f"poisson XSource n_sources={n_sources!r} must be an int >= 1"
                )
            if not _is_finite_real(table["weight_nS"]) or float(table["weight_nS"]) < 0.0:
                raise ValueError(
                    f"poisson XSource weight_nS={table['weight_nS']!r} must be >= 0"
                )
            mechanism = table["mechanism"]
            if not isinstance(mechanism, str) or not mechanism:
                raise ValueError(
                    f"poisson XSource mechanism={mechanism!r} must be a non-empty string"
                )
            if not _is_finite_real(table["delay_ms"]):
                raise ValueError(
                    f"poisson XSource delay_ms={table['delay_ms']!r} takes a finite real number"
                )
        window = [k for k in keys if k in _POISSON_WINDOW_KEYS]
        if window and set(window) != set(_POISSON_WINDOW_KEYS):
            raise ValueError(
                f"poisson XSource schedule {schedule!r} declares one window bound without "
                "the other; declare t_start_ms and t_stop_ms together"
            )
        if window:
            start, stop = float(table["t_start_ms"]), float(table["t_stop_ms"])
            if not (
                _is_finite_real(table["t_start_ms"])
                and _is_finite_real(table["t_stop_ms"])
                and start >= 0.0
                and stop >= 0.0
            ):
                raise ValueError(
                    f"poisson XSource window {(table['t_start_ms'], table['t_stop_ms'])!r} "
                    "takes finite bounds >= 0 in ms"
                )
            if not start < stop:
                raise ValueError(
                    f"poisson XSource t_start_ms={table['t_start_ms']!r} must be below "
                    f"t_stop_ms={table['t_stop_ms']!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (targets as a list, schedule as a dict or None)."""
        schedule = None if self.schedule is None else {k: v for k, v in self.schedule}
        value = self.value
        if isinstance(value, bool):
            text = str(value)
        elif isinstance(value, numbers.Real):
            text = float(value)
        else:
            text = value
        out: dict[str, Any] = {
            "kind": str(self.kind),
            "targets": [str(t) for t in self.targets],
            "value": text,
            "unit": str(self.unit),
            "schedule": schedule,
            "seed": None if self.seed is None else int(self.seed),
            "n_neurons": None if self.n_neurons is None else int(self.n_neurons),
        }
        # json_safe handles the rest at the manifest layer; keep raw types here
        # but normalized (floats as float, schedule pairs as a dict).
        if self.schedule is not None:
            safe: dict[str, Any] = {}
            for k, v in self.schedule:
                if isinstance(v, bool):
                    safe[str(k)] = v
                elif isinstance(v, numbers.Real):
                    safe[str(k)] = float(v) if isinstance(v, float) or "." in repr(v) or "e" in repr(v).lower() else v
                    if isinstance(safe[str(k)], float):
                        pass
                    else:
                        safe[str(k)] = int(v)
                else:
                    safe[str(k)] = v
            out["schedule"] = safe
        return out


def xsource_to_dict(source: XSource) -> dict[str, Any]:
    """Return the manifest row for one entry (kind, targets, value, unit, schedule, seed, n_neurons)."""
    return source.to_dict()


def _default_drive_values() -> dict[str, float]:
    # Late import: emitters is heavy; inputs stays import-light.
    from .emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS

    return {
        cell_type: float(IZHIKEVICH_CELL_TYPE_DEFAULTS[cell_type]["drive"])
        for cell_type in DRIVE_VOCABULARY_ORDER
    }


def baseline_drive_entries(
    metadata: Mapping[str, Any] | None,
    *,
    unit: str = "native_current",
) -> list[XSource]:
    """Baseline ``drive()`` entries, defaults-filled in vocabulary order.

    Absent metadata (no ``drive()`` call) yields the four ``drive()``
    vocabulary entries with emitter-table values, in
    ``(E, PV, SST, VIP)`` order. An explicit map yields its cell types in
    vocabulary order first, then any extra cell types sorted. Values are
    validated as finite reals here; population membership is refused at
    construction, which sees the built labels.
    """
    spec = (metadata or {}).get("drive") if isinstance(metadata, Mapping) else None
    baseline = spec.get("baseline_drive_by_cell_type") if isinstance(spec, dict) else None
    if baseline is None:
        items = _default_drive_values()
        ordered = [(cell_type, items[cell_type]) for cell_type in DRIVE_VOCABULARY_ORDER]
    else:
        declared = dict(baseline)
        for cell_type, value in declared.items():
            if not _is_finite_real(value):
                raise ValueError(
                    f"drive(baseline_drive_by_cell_type={{{cell_type!r}: {value!r}}}) is not realized: "
                    "it takes a finite real number"
                )
        ordered = [
            (cell_type, float(declared[cell_type]))
            for cell_type in DRIVE_VOCABULARY_ORDER
            if cell_type in declared
        ]
        ordered.extend(
            (cell_type, float(declared[cell_type]))
            for cell_type in sorted(declared, key=str)
            if cell_type not in DRIVE_VOCABULARY_ORDER
        )
    return [
        XSource(kind="constant", targets=(str(cell_type),), value=float(value), unit=unit)
        for cell_type, value in ordered
    ]


def triple_drive_entries(
    metadata: Mapping[str, Any] | None,
    *,
    unit: str = "native_current",
) -> list[XSource]:
    """Per-triple ``tensor_triple_drive`` entries in metadata order."""
    entries = (metadata or {}).get("tensor_triple_drive") if isinstance(metadata, Mapping) else None
    if not entries:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            f"tensor_triple_drive {entries!r} is not realized: it takes "
            "a list of {area, layer, cell_type, drive}"
        )
    out: list[XSource] = []
    for entry in entries:
        try:
            area = str(entry["area"])
            layer = str(entry["layer"])
            cell_type = str(entry["cell_type"])
            value = entry["drive"]
        except (KeyError, TypeError):
            raise ValueError(
                f"tensor_triple_drive entry {entry!r} is not realized: it takes "
                "{area, layer, cell_type, drive}"
            )
        if not _is_finite_real(value):
            raise ValueError(
                f"tensor_triple_drive(area={area!r}, layer={layer!r}, "
                f"cell_type={cell_type!r}, drive={value!r}) is not realized: "
                "it takes a finite real number"
            )
        out.append(
            XSource(
                kind="constant",
                targets=(area, layer, cell_type),
                value=float(value),
                unit=unit,
            )
        )
    return out


# Keys each Poisson route consumes. The current route (``_make_poisson_drive``:
# whole-population Bernoulli noise) reads only these five; the AdEx route
# (``adex_external_drive_kwargs``) reads its seven plus the optional window.
_CURRENT_POISSON_KEYS = ("rate_hz", "amplitude", "target", "seed", "counts")
# Run-bookkeeping receipts the simulate path writes into
# ``signals.metadata["poisson_drive"]`` alongside the declaration; they are
# not declarations and are stripped before any consumed-key check.
_POISSON_RECEIPT_KEYS = ("status", "chunk_index", "chunk_seed")
_ADEX_POISSON_KEYS = (
    "n_sources", "rate_hz", "weight_nS", "mechanism",
    "target_cell_types", "delay_ms", "seed",
)


def check_current_route_poisson_drive(poisson: Mapping[str, Any]) -> None:
    """Refuse current-route ``poisson_drive`` keys no numeric path consumes.

    A ``t_start_ms``/``t_stop_ms`` window and every AdEx-only key
    (``n_sources``, ``weight_nS``, ``mechanism``, ``target_cell_types``,
    ``delay_ms``) are silently dropped by ``_make_poisson_drive``; the
    window is consumed only by the AdEx route, as a pair naming
    ``target_cell_types``. Anything outside
    ``("rate_hz", "amplitude", "target", "seed", "counts")`` is refused here so a
    misrouted declaration fails loudly at the public entry point instead
    of running with part of its drive missing. A declared ``counts`` must be
    a positive integer (bools, non-integral numbers and values < 1 refused);
    it sums that many independent trains per step per cell.
    """
    seed = poisson.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"poisson_drive seed={seed!r} must be an int")
    if "counts" in poisson:
        counts = poisson["counts"]
        if isinstance(counts, bool) or not isinstance(counts, numbers.Integral):
            raise ValueError(
                f"poisson_drive counts={counts!r} must be a positive integer"
            )
        if int(counts) < 1:
            raise ValueError(
                f"poisson_drive counts={counts!r} must be a positive integer"
            )
    extra = [k for k in poisson if k not in _CURRENT_POISSON_KEYS]
    if not extra:
        return
    raise ValueError(
        f"poisson_drive {dict(poisson)!r} declares keys {extra!r} the current "
        "route does not consume: it reads only "
        f"{list(_CURRENT_POISSON_KEYS)} (no drive window). The window "
        "(t_start_ms and t_stop_ms together) and "
        f"{list(_ADEX_POISSON_KEYS)} are consumed only by the AdEx route "
        "(a mapping naming target_cell_types)."
    )


def compile_poisson_sources(
    poisson: Mapping[str, Any] | None,
    *,
    route: str = "auto",
) -> tuple[XSource, ...]:
    """Compile one ``Simulation(poisson_drive=...)`` mapping to a Poisson entry.

    ``route="auto"`` detects the AdEx mapping by its ``target_cell_types``
    key; ``"current"`` and ``"adex"`` force a route. The existing numeric
    paths (``ext_lambda``/``_make_poisson_drive``) are unchanged; this only
    records the declaration.
    """
    if poisson is None:
        return ()
    if not isinstance(poisson, Mapping):
        raise ValueError(f"poisson_drive {poisson!r} is not realized: it takes a mapping")
    detected = "adex" if "target_cell_types" in poisson else "current"
    if route == "auto":
        route = detected
    if route not in ("current", "adex"):
        raise ValueError(f"unknown poisson route {route!r}; expected 'current', 'adex' or 'auto'")
    if route == "current":
        check_current_route_poisson_drive(poisson)
        rate = poisson.get("rate_hz", 2.0)
        amplitude = poisson.get("amplitude", 0.5)
        target = str(poisson.get("target", "all"))
        seed = poisson.get("seed", 0)
        if not _is_finite_real(rate) or float(rate) < 0.0:
            raise ValueError(f"poisson_drive rate_hz={rate!r} must be >= 0")
        if not _is_finite_real(amplitude):
            raise ValueError(f"poisson_drive amplitude={amplitude!r} takes a finite real number")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError(f"poisson_drive seed={seed!r} must be an int")
        current_schedule: list[tuple[str, Any]] = [("amplitude", float(amplitude))]
        if "counts" in poisson:
            # Recorded only when declared so default schedules stay unchanged;
            # validated by check_current_route_poisson_drive above.
            current_schedule.append(("counts", int(poisson["counts"])))
        entry = XSource(
            kind="poisson",
            targets=(target,),
            value=float(rate),
            unit="Hz",
            schedule=tuple(current_schedule),
            seed=int(seed),
        )
        return (entry,)
    # AdEx route: exactly the _DRIVE_KEYS mapping plus the optional window.
    required = ("n_sources", "rate_hz", "weight_nS", "mechanism", "target_cell_types", "delay_ms", "seed")
    missing = [k for k in required if k not in poisson]
    if missing:
        raise ValueError(f"poisson_drive {dict(poisson)!r} is missing keys {missing}")
    targets = poisson["target_cell_types"]
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ValueError(
            f"poisson_drive target_cell_types={targets!r} must be a non-empty list of cell types"
        )
    schedule: list[tuple[str, Any]] = [
        ("n_sources", int(poisson["n_sources"])),
        ("weight_nS", float(poisson["weight_nS"])),
        ("mechanism", str(poisson["mechanism"])),
        ("delay_ms", float(poisson["delay_ms"])),
    ]
    if poisson.get("t_start_ms") is not None or poisson.get("t_stop_ms") is not None:
        schedule.extend(
            [
                ("t_start_ms", float(poisson["t_start_ms"])),
                ("t_stop_ms", float(poisson["t_stop_ms"])),
            ]
        )
    entry = XSource(
        kind="poisson",
        targets=tuple(str(t) for t in targets),
        value=float(poisson["rate_hz"]),
        unit="Hz",
        schedule=tuple(schedule),
        seed=int(poisson["seed"]),
    )
    return (entry,)


def compile_poisson_sources_from_signals_metadata(
    signals_metadata: Mapping[str, Any] | None,
) -> tuple[XSource, ...]:
    """Manifest-time Poisson entries from ``signals.metadata["poisson_drive"]``.

    Route auto-detection reads the recorded mapping: a ``target_cell_types``
    key is the AdEx mapping, anything else the current route.
    """
    recorded = (signals_metadata or {}).get("poisson_drive") if isinstance(signals_metadata, Mapping) else None
    if recorded is None:
        return ()
    if not isinstance(recorded, Mapping):
        return ()
    declared = {k: v for k, v in dict(recorded).items() if k not in _POISSON_RECEIPT_KEYS}
    return compile_poisson_sources(declared, route="auto")
