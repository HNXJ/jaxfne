#!/usr/bin/env python3
"""Post-process spectrolaminar suite figures to smooth middle panel (PSD) in frequency dimension.

Loads rendered PNG figures, applies Gaussian blur to the middle panel (mean relative power),
and re-saves with smoothed visualization.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
FIGS = HERE / "figs"

DC_LEVELS = list(range(0, 105, 5))  # 0, 5, 10, ..., 100


def smooth_middle_panel_psd(fig_path: Path, sigma: float = 2.0) -> bool:
    """Load spectrolaminar PNG, smooth middle panel (PSD profile) in frequency axis, re-save.

    The spectrolaminar_suite_3panel renders 3 panels horizontally:
    1. Depth x Frequency heatmap (left)
    2. Mean relative power (frequency axis) - THIS ONE GETS SMOOTHED (middle)
    3. Frequency profile (right)

    Applies Gaussian blur to the middle region to smooth frequency-domain noise.
    """
    if not fig_path.exists():
        print(f"[smooth] {fig_path.name} not found, skipping")
        return False

    try:
        # Load the PNG
        img = Image.open(fig_path)
        img_array = np.asarray(img)

        # For spectrolaminar_suite_3panel (3 horizontal panels):
        # Left = depth x freq heatmap, Middle = mean power, Right = freq profile
        # Estimate middle panel boundaries (typically ~33-67% of width for 3-panel layout)
        h, w = img_array.shape[:2]
        left_bound = int(w * 0.32)
        right_bound = int(w * 0.68)

        # Extract middle panel
        middle_slice = img_array[:, left_bound:right_bound]

        # Convert to PIL, apply Gaussian blur (sigma=2 is gentle; higher = more blur)
        middle_pil = Image.fromarray(middle_slice.astype(np.uint8))
        smoothed_middle = middle_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Reconstruct full image
        result_array = img_array.copy()
        result_array[:, left_bound:right_bound] = np.asarray(smoothed_middle)

        # Save smoothed version
        result_img = Image.fromarray(result_array.astype(np.uint8))
        result_img.save(fig_path, dpi=(150, 150))
        print(f"[smooth] {fig_path.name} — frequency smoothed (sigma={sigma})", flush=True)
        return True
    except Exception as e:
        print(f"[smooth] {fig_path.name} — failed: {e}", flush=True)
        return False


def main():
    success_count = 0
    for dc in DC_LEVELS:
        fname = FIGS / f"spectrolaminar_suite_v2_dc{int(dc)}.png"
        if smooth_middle_panel_psd(fname, sigma=2.0):
            success_count += 1

    print(f"\n[smooth] DONE. {success_count}/{len(DC_LEVELS)} figures smoothed.", flush=True)


if __name__ == "__main__":
    main()
