"""F-017 sweep: find rho_passive for DEFAULT_HDP and DEFAULT_HDP_DESYNC.

Goal: restore H_mean ≈ 1.0 ± 0.03, H_std < 0.05 over full simulation
duration at each preset's verified N, measuring only the steady-state
second half of the run to skip transient.

Usage
-----
    python scripts/hdp_v2_rho_sweep.py            # full sweep (both presets)
    python scripts/hdp_v2_rho_sweep.py --preset default
    python scripts/hdp_v2_rho_sweep.py --preset desync
    python scripts/hdp_v2_rho_sweep.py --fast     # 2s/2seeds smoke test

Outputs
-------
    artifacts/hdp_v2_rho_sweep/results_default.csv
    artifacts/hdp_v2_rho_sweep/results_desync.csv
    artifacts/hdp_v2_rho_sweep/summary.txt

Do NOT put simulator or optimizer logic here. This script only
configures and calls package APIs.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.hdp_network import (
    HDPColumnConfig,
    BASE_HDP_KWARGS_DEFAULT,
    DEFAULT_HDP,
    DEFAULT_HDP_DESYNC,
    DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT,
    DRIVE_SCALE_DESYNC,
    LAYER_SIZE_SCALE_DEFAULT,
    build_model,
)

# -----------------------------------------------------------------------
# Sweep grid and acceptance thresholds
# -----------------------------------------------------------------------

RHO_CANDIDATES: list[float] = np.geomspace(0.005, 2.0, 15).tolist()

H_MEAN_TARGET = 1.0
H_MEAN_TOL    = 0.03   # |H_mean - 1.0| must be <= this
H_STD_MAX     = 0.05   # H_std over steady-state must be <= this

# -----------------------------------------------------------------------
# Per-preset sweep parameters
# -----------------------------------------------------------------------

PRESET_SPECS: dict[str, dict] = {
    "default": dict(
        label="DEFAULT_HDP",
        n_neurons=250,
        duration_ms=20_000.0,
        n_seeds=5,
        hdp_kwargs={**DEFAULT_HDP},
        base_kwargs={**BASE_HDP_KWARGS_DEFAULT},
        drive_scale=1.0,
        drive_correction={**DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT},
        layer_size_scale={**LAYER_SIZE_SCALE_DEFAULT},
    ),
    "desync": dict(
        label="DEFAULT_HDP_DESYNC",
        n_neurons=500,
        duration_ms=20_000.0,
        n_seeds=5,
        hdp_kwargs={**DEFAULT_HDP_DESYNC},
        base_kwargs={**BASE_HDP_KWARGS_DEFAULT},
        drive_scale=DRIVE_SCALE_DESYNC,
        drive_correction={**DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT},
        layer_size_scale={**LAYER_SIZE_SCALE_DEFAULT},
    ),
}

FAST_OVERRIDE = dict(duration_ms=2_000.0, n_seeds=2)
OUT_DIR = Path("artifacts/hdp_v2_rho_sweep")


# -----------------------------------------------------------------------
# Single-run helper
# -----------------------------------------------------------------------

def _run_one(preset_key: str, rho: float, seed: int, overrides: dict) -> dict:
    spec = {**PRESET_SPECS[preset_key], **overrides}

    hdp_kw = {**spec["hdp_kwargs"], "rho_passive": rho}
    combined_kw = {**hdp_kw, **spec["base_kwargs"]}

    cfg = HDPColumnConfig(
        n_neurons=spec["n_neurons"],
        duration_ms=spec["duration_ms"],
        dt_ms=0.5,
        seed=seed,
    )

    # Build model from config
    model = build_model(cfg)

    # Extract params and edges from model
    izhikevich_params = model.params["emitter"]
    edges = model.params["edge_list"]

    # Simulate with the custom HDP parameters
    t0 = time.perf_counter()
    _, sig, _, diagnostics = jtfne.emitters.simulate_edge_recurrent_izhikevich_hdp(
        params=izhikevich_params,
        edges=edges,
        n_steps=int(spec["duration_ms"] / cfg.dt_ms),
        dt_ms=cfg.dt_ms,
        key=jax.random.PRNGKey(seed),
        **combined_kw,
    )
    elapsed = time.perf_counter() - t0

    # H_trace: (n_steps, n_neurons) — from diagnostics
    H_trace = np.array(diagnostics["H_trace"])   # (T, N)
    spikes  = np.array(sig)                       # (T, N) bool

    # Measure steady-state: second half only
    T = H_trace.shape[0]
    half = T // 2
    H_ss = H_trace[half:]

    H_mean       = float(np.mean(H_ss))
    H_std        = float(np.std(H_ss))
    H_min_obs    = float(np.min(H_ss))
    H_max_obs    = float(np.max(H_ss))
    H_floor_frac = float(np.mean(H_ss <= 0.12))   # near H_min clamp
    H_ceil_frac  = float(np.mean(H_ss >= 9.8))    # near H_max clamp

    dt_ms = cfg.dt_ms
    mean_rate_hz = float(np.mean(spikes[half:]) / (dt_ms * 1e-3))

    passes = int(
        abs(H_mean - H_MEAN_TARGET) <= H_MEAN_TOL
        and H_std <= H_STD_MAX
    )

    return dict(
        preset=spec["label"], rho_passive=rho, seed=seed,
        H_mean=H_mean, H_std=H_std,
        H_min_obs=H_min_obs, H_max_obs=H_max_obs,
        H_floor_frac=H_floor_frac, H_ceil_frac=H_ceil_frac,
        mean_rate_hz=mean_rate_hz, elapsed_s=elapsed, passes=passes,
    )


# -----------------------------------------------------------------------
# Preset sweep
# -----------------------------------------------------------------------

def sweep_preset(preset_key: str, overrides: dict, out_csv: Path) -> list[dict]:
    spec = {**PRESET_SPECS[preset_key], **overrides}
    label = spec["label"]
    print(f"\n{'='*60}")
    print(f"Sweeping {label}  N={spec['n_neurons']}  "
          f"{spec['duration_ms']/1000:.0f}s  seeds={spec['n_seeds']}")
    print(f"rho candidates: {[f'{r:.4f}' for r in RHO_CANDIDATES]}")
    print(f"{'='*60}")

    rows: list[dict] = []
    for rho in RHO_CANDIDATES:
        seed_rows: list[dict] = []
        for seed in range(spec["n_seeds"]):
            print(f"  rho={rho:.5f}  seed={seed} ...", end=" ", flush=True)
            try:
                row = _run_one(preset_key, rho, seed, overrides)
                seed_rows.append(row)
                print(
                    f"H_mean={row['H_mean']:.4f}  "
                    f"H_std={row['H_std']:.4f}  "
                    f"rate={row['mean_rate_hz']:.1f}Hz  "
                    f"{'PASS' if row['passes'] else 'fail'}  "
                    f"({row['elapsed_s']:.1f}s)"
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                seed_rows.append(dict(
                    preset=label, rho_passive=rho, seed=seed,
                    H_mean=float("nan"), H_std=float("nan"),
                    H_min_obs=float("nan"), H_max_obs=float("nan"),
                    H_floor_frac=float("nan"), H_ceil_frac=float("nan"),
                    mean_rate_hz=float("nan"), elapsed_s=0.0, passes=0,
                ))
        rows.extend(seed_rows)

        all_pass = all(r["passes"] for r in seed_rows)
        if all_pass:
            print(f"  -> All {spec['n_seeds']} seeds PASS at rho={rho:.5f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  CSV written: {out_csv}")

    return rows


# -----------------------------------------------------------------------
# Recommendation and summary
# -----------------------------------------------------------------------

def recommend(all_rows: list[dict]) -> dict[str, float | None]:
    """Return smallest rho that achieves all-seed PASS per preset."""
    by_preset: dict[str, list[dict]] = {}
    for r in all_rows:
        by_preset.setdefault(r["preset"], []).append(r)

    rec: dict[str, float | None] = {}
    for preset, rows in sorted(by_preset.items()):
        by_rho: dict[float, list[dict]] = {}
        for r in rows:
            by_rho.setdefault(r["rho_passive"], []).append(r)

        best: float | None = None
        for rho in sorted(by_rho):
            seed_rows = by_rho[rho]
            if all(r["passes"] for r in seed_rows):
                means = [r["H_mean"] for r in seed_rows]
                stds  = [r["H_std"]  for r in seed_rows]
                print(
                    f"  {preset}: rho={rho:.5f}  "
                    f"H_mean={np.mean(means):.4f}±{np.std(means):.4f}  "
                    f"H_std={np.mean(stds):.4f}  ALL SEEDS PASS"
                )
                if best is None:
                    best = rho
        if best is None:
            print(f"  {preset}: NO candidate passed all seeds — widen range")
        rec[preset] = best
    return rec


def write_summary(
    all_rows: list[dict],
    rec: dict[str, float | None],
    out_path: Path,
) -> None:
    lines = [
        "F-017 rho_passive sweep — summary",
        "=" * 60,
        f"Date: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"Acceptance: |H_mean-1.0|<={H_MEAN_TOL}, H_std<={H_STD_MAX}",
        "",
        "Recommendations",
        "-" * 40,
    ]
    for preset, rho in sorted(rec.items()):
        if rho is not None:
            lines.append(f"  {preset}: rho_passive = {rho:.5f}")
            for r in all_rows:
                if r["preset"] == preset and abs(r["rho_passive"] - rho) < 1e-9:
                    lines.append(
                        f"    seed={r['seed']}  "
                        f"H_mean={r['H_mean']:.4f}  "
                        f"H_std={r['H_std']:.4f}  "
                        f"H=[{r['H_min_obs']:.3f},{r['H_max_obs']:.3f}]  "
                        f"rate={r['mean_rate_hz']:.1f}Hz"
                    )
        else:
            lines.append(f"  {preset}: NO CANDIDATE FOUND")

    lines += [
        "",
        "Post-sweep steps (coder agent)",
        "-" * 40,
        "  1. In jaxfne/hdp_network.py:",
        "     - Set DEFAULT_HDP['rho_passive']       = <value from above>",
        "     - Set DEFAULT_HDP_DESYNC['rho_passive'] = <value from above>",
        "     - Replace the K_ctrl transition comment with a freeze comment:",
        "       '# Verified (HDP v2, F-017): rho_passive=X.XXXXX,",
        "       #  H_mean≈1.0, H_std<0.05 over 20s, 5 seeds. SHA: <commit>'",
        "  2. In skills/FRICTIONS_STACK.md:",
        "     - Move F-017 from Open to Resolved with today's date and SHA.",
        "  3. Commit:",
        "     git add jaxfne/hdp_network.py skills/FRICTIONS_STACK.md",
        "     git commit -m 'fix(hdp): retune rho_passive; resolve F-017'",
        "  4. Run full test suite: python -m pytest tests/ -q",
        "  5. Report audit: paste exact git log --oneline -3 and",
        "     grep -n rho_passive jaxfne/hdp_network.py as evidence.",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written: {out_path}")


# -----------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=["default", "desync", "both"], default="both"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="2s / 2 seeds — smoke test only"
    )
    args = parser.parse_args()

    overrides = FAST_OVERRIDE if args.fast else {}
    if args.fast:
        print("[--fast] 2s duration, 2 seeds per candidate.")

    keys = ["default", "desync"] if args.preset == "both" else [args.preset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for pk in keys:
        rows = sweep_preset(pk, overrides, OUT_DIR / f"results_{pk}.csv")
        all_rows.extend(rows)

    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    rec = recommend(all_rows)
    write_summary(all_rows, rec, OUT_DIR / "summary.txt")


if __name__ == "__main__":
    main()
