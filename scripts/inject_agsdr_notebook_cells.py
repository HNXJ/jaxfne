#!/usr/bin/env python3
"""Inject AGSDR cells into the delta-test notebook."""

import json
from pathlib import Path

notebook_path = Path("tutorials/jaxfne-sanity-checker-notebook-01.ipynb")

# Read notebook
with open(notebook_path) as f:
    nb = json.load(f)

# Create new cells for AGSDR
agsdr_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6a: Compute baseline firing rates from simulation\n",
            "baseline_rates_all = np.zeros((GLOBAL['N_PER_COLUMN'] * len(GLOBAL['areas']),))\n",
            "baseline_mean_rate_hz = 0.0\n",
            "baseline_min_rate_hz = 1e6\n",
            "\n",
            "neuron_idx = 0\n",
            "for area in GLOBAL['areas']:\n",
            "    signals = signals_by_area[area]\n",
            "    spk = signals.get('spikes')\n",
            "    \n",
            "    # Count spikes per neuron\n",
            "    spike_counts = np.sum(spk, axis=0)\n",
            "    duration_sec = GLOBAL['duration_ms'] / 1000.0\n",
            "    rates_hz = spike_counts / duration_sec\n",
            "    \n",
            "    baseline_rates_all[neuron_idx:neuron_idx+len(rates_hz)] = rates_hz\n",
            "    neuron_idx += len(rates_hz)\n",
            "\n",
            "baseline_mean_rate_hz = float(np.mean(baseline_rates_all))\n",
            "baseline_min_rate_hz = float(np.min(baseline_rates_all[baseline_rates_all > 0.1]))\n",
            "\n",
            "print(f'Baseline rates: mean={baseline_mean_rate_hz:.2f} Hz, min={baseline_min_rate_hz:.2f} Hz')\n",
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6b: AGSDR connectivity-gain tuning\n",
            "# Define AGSDR parameters from GLOBAL\n",
            "AGSDR_PARAMS = {\n",
            "    'agsdr_target_mean_rate_hz': 7.5,\n",
            "    'agsdr_mean_rate_tolerance_hz': 1.5,\n",
            "    'agsdr_min_neuron_rate_hz': 1.0,\n",
            "    'n_candidates': 8,  # smoke mode\n",
            "}\n",
            "\n",
            "# Grid of connectivity gains\n",
            "candidate_gains = np.linspace(0.2, 3.0, AGSDR_PARAMS['n_candidates'])\n",
            "\n",
            "results = []\n",
            "\n",
            "print(f'AGSDR grid search: {len(candidate_gains)} candidates')\n",
            "print(f'Target rate: {AGSDR_PARAMS[\"agsdr_target_mean_rate_hz\"]} ± {AGSDR_PARAMS[\"agsdr_mean_rate_tolerance_hz\"]} Hz')\n",
            "\n",
            "for gain in candidate_gains:\n",
            "    tuned_rates = baseline_rates_all * gain\n",
            "    mean_rate = float(np.mean(tuned_rates))\n",
            "    min_rate = float(np.min(tuned_rates[tuned_rates > 0.05]))\n",
            "    \n",
            "    mean_error_hz = abs(mean_rate - AGSDR_PARAMS['agsdr_target_mean_rate_hz'])\n",
            "    min_rate_violation_hz = max(0.0, AGSDR_PARAMS['agsdr_min_neuron_rate_hz'] - min_rate)\n",
            "    gain_deviation = abs(gain - 1.0)\n",
            "    \n",
            "    score = mean_error_hz + 10.0 * min_rate_violation_hz + 0.01 * gain_deviation\n",
            "    \n",
            "    results.append({\n",
            "        'candidate_gain': float(gain),\n",
            "        'mean_rate_hz': mean_rate,\n",
            "        'min_rate_hz': min_rate,\n",
            "        'score': float(score),\n",
            "    })\n",
            "    \n",
            "    print(f'  gain={gain:.2f}: mean={mean_rate:.2f} Hz, min={min_rate:.2f} Hz, score={score:.3f}')\n",
            "\n",
            "# Find best result\n",
            "best_result = min(results, key=lambda r: r['score'])\n",
            "print(f'\\nBest gain: {best_result[\"candidate_gain\"]:.2f} (score={best_result[\"score\"]:.3f})')\n",
            "print(f'  mean rate: {best_result[\"mean_rate_hz\"]:.2f} Hz')\n",
            "print(f'  min rate: {best_result[\"min_rate_hz\"]:.2f} Hz')\n",
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6c: Baseline vs tuned comparison\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "fig.suptitle('Baseline vs AGSDR-Tuned Firing Rates')\n",
            "\n",
            "# Panel 1: baseline vs tuned mean/min\n",
            "categories = ['Mean Rate', 'Min Rate']\n",
            "baseline_vals = [baseline_mean_rate_hz, baseline_min_rate_hz]\n",
            "tuned_vals = [best_result['mean_rate_hz'], best_result['min_rate_hz']]\n",
            "\n",
            "x_pos = np.arange(len(categories))\n",
            "width = 0.35\n",
            "\n",
            "axes[0].bar(x_pos - width/2, baseline_vals, width, label='baseline', alpha=0.8)\n",
            "axes[0].bar(x_pos + width/2, tuned_vals, width, label='tuned', alpha=0.8)\n",
            "axes[0].axhline(y=AGSDR_PARAMS['agsdr_min_neuron_rate_hz'], color='r', linestyle='--', label='min gate (1.0 Hz)')\n",
            "axes[0].set_ylabel('Firing Rate (Hz)')\n",
            "axes[0].set_title('Mean and Min Firing Rates')\n",
            "axes[0].set_xticks(x_pos)\n",
            "axes[0].set_xticklabels(categories)\n",
            "axes[0].legend()\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Panel 2: gain vs score\n",
            "gains_plot = [r['candidate_gain'] for r in results]\n",
            "scores_plot = [r['score'] for r in results]\n",
            "\n",
            "axes[1].plot(gains_plot, scores_plot, 'o-', linewidth=2, markersize=8)\n",
            "axes[1].axvline(x=best_result['candidate_gain'], color='r', linestyle='--', label=f'best gain={best_result[\"candidate_gain\"]:.2f}')\n",
            "axes[1].set_xlabel('Connectivity Gain')\n",
            "axes[1].set_ylabel('Objective Score')\n",
            "axes[1].set_title('Gain Search')\n",
            "axes[1].legend()\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "fig_path = output_dir / 'agsdr_baseline_vs_tuned.png'\n",
            "plt.savefig(fig_path, dpi=100, bbox_inches='tight')\n",
            "print(f'✓ Baseline vs tuned figure: {fig_path}')\n",
            "plt.show()\n",
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6d: Export AGSDR results\n",
            "optimizer_report = {\n",
            "    'optimizer_name': 'agsdr_connectivity_gain_grid_search',\n",
            "    'best_parameters': {'connectivity_gain': float(best_result['candidate_gain'])},\n",
            "    'best_score': float(best_result['score']),\n",
            "    'mean_rate_error_hz': float(abs(best_result['mean_rate_hz'] - AGSDR_PARAMS['agsdr_target_mean_rate_hz'])),\n",
            "    'tuning_status': 'smoke_test',  # smoke mode with short duration\n",
            "    \n",
            "    'biological_learning_claim': False,\n",
            "    'mechanism_claim_status': 'not_claimed',\n",
            "    'baseline_mean_rate_hz': baseline_mean_rate_hz,\n",
            "    'baseline_min_rate_hz': baseline_min_rate_hz,\n",
            "    'best_mean_rate_hz': float(best_result['mean_rate_hz']),\n",
            "    'best_min_rate_hz': float(best_result['min_rate_hz']),\n",
            "    'n_candidates': len(candidate_gains),\n",
            "}\n",
            "\n",
            "optimizer_path = output_dir / 'optimizer_report.json'\n",
            "jtfne.save_json(optimizer_report, optimizer_path)\n",
            "print(f'✓ Optimizer report: {optimizer_path}')\n",
            "\n",
            "# Also update metrics.json with AGSDR summary\n",
            "agsdr_metrics = {\n",
            "    'agsdr_tuning_enabled': True,\n",
            "    'agsdr_best_gain': float(best_result['candidate_gain']),\n",
            "    'agsdr_baseline_mean_rate_hz': baseline_mean_rate_hz,\n",
            "    'agsdr_tuned_mean_rate_hz': float(best_result['mean_rate_hz']),\n",
            "    'agsdr_target_mean_rate_hz': AGSDR_PARAMS['agsdr_target_mean_rate_hz'],\n",
            "    'agsdr_tolerance_hz': AGSDR_PARAMS['agsdr_mean_rate_tolerance_hz'],\n",
            "}\n",
            "\n",
            "# Merge with existing metrics\n",
            "metrics_path = output_dir / 'metrics.json'\n",
            "if metrics_path.exists():\n",
            "    existing_metrics = jtfne.load_json(metrics_path)\n",
            "    existing_metrics.update(agsdr_metrics)\n",
            "    jtfne.save_json(existing_metrics, metrics_path)\n",
            "    print(f'✓ Updated metrics.json with AGSDR summary')\n",
        ]
    },
]

# Find the insertion point: after cell 6 (baseline simulation), before cell 7
# The cells are indexed 0-17 in the notebook
# We want to insert after index 6 (cell 6)

insertion_point = 7  # Insert before current cell 7 (raster)

# Insert new cells
for i, cell in enumerate(agsdr_cells):
    nb["cells"].insert(insertion_point + i, cell)

# Write updated notebook
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Injected {len(agsdr_cells)} AGSDR cells into {notebook_path}")
print(f"  Inserted at position {insertion_point}")
print(f"  Total cells now: {len(nb['cells'])}")
