#!/usr/bin/env python3
"""Plot per-metric bar charts from hidden-state evaluation JSON results.

Usage:
     python plot_metrics.py --input-dir results/ --output-dir my_plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_dir: Path) -> list[dict]:
    """Load every results_*.json file in *input_dir* and return sorted list."""
    records = []
    for path in sorted(input_dir.glob("results_*.json")):
        with open(path) as f:
            data = json.load(f)

        cfg = data.get("config", {})
        layer = cfg.get("target_layer", "?")
        alpha = cfg.get("alpha", "?")
        label = f"L{layer}_a{alpha}"

        records.append({
            "path": path.name,
            "label": label,
            "layer": layer,
            "alpha": alpha,
            "metrics": data.get("metrics", {}),
        })

    # Sort by (layer, alpha) so the x-axis is in a natural order
    records.sort(key=lambda r: (r["layer"], r["alpha"]))
    return records


def plot_metric(records, metric_key, output_dir):
    """Create and save a single bar chart for *metric_key*."""
    labels = [r["label"] for r in records]
    values = [r["metrics"].get(metric_key, 0) for r in records]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))

    bars_model = ax.bar(x - width / 2, values, width, label="Model", color="#4A90D9", edgecolor="white", linewidth=0.6)

    # Value annotations on the model bars
    for bar, val in zip(bars_model, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    ax.set_xlabel("Condition  (Layer_Alpha)", fontsize=11)
    ax.set_ylabel(metric_key, fontsize=11)
    ax.set_title(f"{metric_key}  by Target Layer / Alpha", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    # ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"{metric_key.replace('@', '_at_')}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",  type=Path, default=Path("."), help="Folder containing results_*.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("output_plots"), help="Folder for saved bar-plot PNGs")
    args = parser.parse_args()

    records = load_results(args.input_dir)
    if not records:
        print(f"No results_*.json files found in {args.input_dir.resolve()}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(records)} result files:")
    for r in records:
        print(f"  {r['path']}  →  {r['label']}")

    # Gather every metric key that appears across all files
    all_metric_keys = sorted(
        {k for r in records for k in r["metrics"] if k != "n"},
        key=lambda k: (k.split("@")[0], int(k.split("@")[1]) if "@" in k else 0),
    )

    print(f"\nPlotting {len(all_metric_keys)} metrics: {', '.join(all_metric_keys)}")
    for key in all_metric_keys:
        plot_metric(records, key, args.output_dir)

    print(f"\nDone — all plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()