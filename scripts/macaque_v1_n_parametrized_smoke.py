#!/usr/bin/env python3
"""N-parametrized smoke pipeline for the default_macaque_V1 config.

Implements the 7-step pipeline (user spec, 2026-06-30): load the area config
JSON, override total N, build the network, simulate with HDP + AMPA/GABA/NMDA
on, extract LFP-proxy + raster, visualize raster + spectrolaminar suite.

N-override uses jaxfne's existing NeuronalTensor/Area/Layer/NeuronType
dataclasses directly (rescale each layer's n_neurons proportionally to the
loaded reference tensor's own layer split, keep each layer's per-cell-type
fractions and the area's inter_connections unchanged) rather than a new
public rescale-by-N helper -- per explicit user decision, 2026-06-30.

Claim status: computational scaffold / proxy readout, not a calibrated
physical recording. N=10 is a smoke-test size; several layers will have 0
neurons at this scale (some cell types within a layer round to 0 at 2-5% of
~1-2 neurons) -- accepted for test purposes per the original spec.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jaxfne as jtfne
from jaxfne.neuronal_tensor import Area, Layer, NeuronalTensor, NeuronType, RuntimeConfiguration

CONFIG_PATH = Path(__file__).resolve().parent.parent / "jaxfne" / "configs" / "default_macaque_V1.json"


def load_and_rescale(config_path: Path, n_total: int) -> NeuronalTensor:
    """Step 1+2: load the reference-N tensor, rebuild at an arbitrary n_total.

    Each layer's share of n_total is proportional to its share of the
    reference tensor's total N (i.e. layer thickness is preserved); each
    layer's per-cell-type fractions and the area's inter_connections are
    carried over unchanged.
    """
    ref = jtfne.load(str(config_path))
    ref_area = ref.areas[0]
    ref_total = sum(l.n_neurons for l in ref_area.layers)

    new_layers = []
    for layer in ref_area.layers:
        share = layer.n_neurons / ref_total
        n_new = round(n_total * share)
        new_layers.append(replace(layer, n_neurons=n_new))

    new_area = replace(ref_area, layers=new_layers)
    return replace(ref, areas=[new_area])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="total neuron count to rescale to")
    parser.add_argument("--duration-ms", type=float, default=1000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("tutorial_outputs/macaque_v1_smoke"))
    args = parser.parse_args()

    jtfne.enable_x64()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Steps 1-2: load config, set N.
    tensor = load_and_rescale(CONFIG_PATH, args.n)
    layer_counts = {l.name: l.n_neurons for l in tensor.areas[0].layers}
    print(f"N={args.n} -> per-layer counts: {layer_counts}")

    # Step 3-4: build network (NeuronalTensor IS the tensor; construct() builds the Model/"network").
    model = jtfne.construct(tensor, RuntimeConfiguration(seed=args.seed, duration_ms=args.duration_ms, dt_ms=args.dt_ms))

    # Step 5: simulate with HDP on (runtime= override, per jaxfne NeuronalTensor+HDP doctrine).
    from jaxfne.hdp_network import DEFAULT_HDP

    sig = jtfne.simulate(
        model,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        seed=args.seed,
        record_fields=True,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(DEFAULT_HDP)),
    )

    # Step 6: get LFP and raster.
    n_spikes = int(sig.spikes.sum()) if sig.spikes is not None else 0
    has_lfp = sig.field is not None and getattr(sig.field, "lfp_proxy", None) is not None
    print(f"n_spikes={n_spikes}  has_lfp_proxy={has_lfp}")
    if not has_lfp:
        raise RuntimeError("sig.field.lfp_proxy is missing -- record_fields=True did not produce field output")

    # Step 7: visualize raster + spectrolaminar suite.
    fig_raster = jtfne.vis.raster(sig)
    fig_raster.savefig(args.out_dir / f"raster_n{args.n}.png", dpi=100)

    fig_suite = jtfne.vis.spectrolaminar_suite(sig)
    fig_suite.savefig(args.out_dir / f"spectrolaminar_suite_n{args.n}.png", dpi=100)

    summary = {
        "n_total": args.n,
        "layer_counts": layer_counts,
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "seed": args.seed,
        "n_spikes": n_spikes,
        "has_lfp_proxy": has_lfp,
        "claim_level": "computational_scaffold_proxy_not_calibrated_physical",
    }
    with (args.out_dir / f"summary_n{args.n}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
