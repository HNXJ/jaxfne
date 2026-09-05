#!/usr/bin/env python3
"""Regenerate the canonical atlas: interactive docs panels and README stills.

Both public presentations of the atlas come from one canonical implementation
(:mod:`jaxfne.vis.atlas_suite`) and one declared configuration, so the README's
static PNGs and the documentation's interactive Plotly panels cannot drift apart
and are reproducible from a clean checkout:

* ``docs/_static/atlas/*.html``  — interactive panels + index, published by
  ReadTheDocs.
* ``docs/assets/readme/*.png``   — static stills GitHub can render inline,
  rasterized from the *same* figure objects via kaleido.

The README images were previously hand-added binaries with no generator, so
nothing could re-derive or drift-check them. This script is that generator.

Canonical configuration (pinned; changing it changes the published atlas):

    tensor  = canonical-v1-column-1000n
    runtime = RuntimeConfiguration(seed=0, duration_ms=200.0, dt_ms=0.5)
    -> config_hash 4b0d96456d56bc1a, 1000 neurons, 215785 edges, 400 steps

Usage:
    python scripts/generate_readme_atlas.py             # html + png
    python scripts/generate_readme_atlas.py --html-only
    python scripts/generate_readme_atlas.py --check     # verify config only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_TENSOR = "canonical-v1-column-1000n"
CANONICAL_SEED = 0
CANONICAL_DURATION_MS = 200.0
CANONICAL_DT_MS = 0.5
CANONICAL_TITLE = "Canonical V1 Column (1000n)"
EXPECTED_CONFIG_HASH = "4b0d96456d56bc1a"

HTML_DIR = ROOT / "docs" / "_static" / "atlas"
PNG_DIR = ROOT / "docs" / "assets" / "readme"

# Rasterized at 2x for legibility on high-DPI displays in the GitHub README.
PNG_SCALE = 2


def build_canonical_model():
    """Realize the pinned canonical column and simulate it."""
    import jaxfne as jtfne

    tensor = jtfne.load_canonical_neuronal_tensor(CANONICAL_TENSOR)
    model = jtfne.construct(
        tensor,
        jtfne.RuntimeConfiguration(
            seed=CANONICAL_SEED,
            duration_ms=CANONICAL_DURATION_MS,
            dt_ms=CANONICAL_DT_MS,
        ),
    )
    observed_hash = str(model.summary().get("config_hash", ""))
    if observed_hash != EXPECTED_CONFIG_HASH:
        raise SystemExit(
            f"canonical config drift: config_hash {observed_hash!r} != "
            f"{EXPECTED_CONFIG_HASH!r}. The published atlas is pinned to this "
            "configuration; update EXPECTED_CONFIG_HASH deliberately if the "
            "canonical column really changed."
        )
    signals = jtfne.simulate(model)
    return model, signals


def write_png_stills(model, signals) -> list[Path]:
    """Rasterize each atlas panel from the canonical figure builders.

    Uses the same figure objects ``build_atlas`` writes to HTML, so the stills
    carry the canonical theme and titles by construction rather than by
    convention.
    """
    from jaxfne.vis import canonical as C
    from jaxfne.vis.atlas_suite import _build_state_summary_fig

    builders = {
        "network_3d": lambda: C.plot_network_3d(model, backend="plotly"),
        "connectivity": lambda: C.plot_connectivity(model, backend="plotly"),
        "raster": lambda: C.plot_raster(signals, model, backend="plotly"),
        "traces": lambda: C.plot_membrane_potentials(signals, model, backend="plotly"),
        "spectral": lambda: C.plot_psd(signals, backend="plotly"),
        "state_summary": lambda: _build_state_summary_fig(model, signals)[0],
    }

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, make_fig in builders.items():
        fig = make_fig()
        # Match the atlas HTML theme so the README stills and the interactive
        # panels read as one system.
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
        out = PNG_DIR / f"{name}.png"
        fig.write_image(str(out), scale=PNG_SCALE)
        written.append(out)
        print(f"  wrote {out.relative_to(ROOT)}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true", help="skip PNG rasterization")
    parser.add_argument("--png-only", action="store_true", help="skip interactive HTML panels")
    parser.add_argument(
        "--check",
        action="store_true",
        help="realize the canonical config and verify its hash, writing nothing",
    )
    args = parser.parse_args(argv)

    model, signals = build_canonical_model()
    print(f"canonical model OK: config_hash={EXPECTED_CONFIG_HASH}")
    if args.check:
        return 0

    if not args.png_only:
        from jaxfne.vis.atlas_suite import build_atlas

        manifest = build_atlas(
            model,
            signals,
            out_dir=str(HTML_DIR),
            title=CANONICAL_TITLE,
        )
        print(f"interactive atlas -> {HTML_DIR.relative_to(ROOT)} (sha256 {manifest['sha256']})")
        for panel in manifest["panels"]:
            print(f"  {panel['file']}: {panel['evidence']} {panel['status']}")

    if not args.html_only:
        try:
            import kaleido  # noqa: F401
        except ImportError:
            print(
                "kaleido is required to rasterize README stills; "
                'install with pip install ".[viz]"',
                file=sys.stderr,
            )
            return 1
        print("README stills:")
        write_png_stills(model, signals)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
