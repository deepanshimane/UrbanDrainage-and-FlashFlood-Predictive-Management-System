"""Runner and evaluation for B7 & B8 Optimizer.

Runs on 3 test storms:
- No control (before)
- Rule-based optimizer (B7)
- Search optimizer (B8)
Generates:
- results/optimizer_results.csv
- results/optimizer_before_after.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io import load_graph, load_dataset
from src.opt.optimize import optimize


def run_optimizer_suite(data_dir: str = None):
    if data_dir is None:
        data_dir = "data/hindmata" if Path("data/hindmata/nodes.csv").exists() else "data/sample"
    nodes, edges = load_graph(data_dir)
    ds = load_dataset(data_dir)

    with open("results/split.json") as f:
        test_storms = json.load(f)["test"]

    # Pick 3 test storms
    selected_storms = test_storms[:3]
    print(f"Running Optimizer evaluation on 3 test storms: {selected_storms}...\n")

    records = []
    for s_idx, s in enumerate(selected_storms):
        rain = ds["rain"][s]
        river = ds["river"][s]

        # 1. Rules
        res_rules = optimize(nodes, edges, rain, river, method="rules")
        cost_before = res_rules["scenario_cost_before"]
        cost_rules = res_rules["scenario_cost_after"]
        red_rules = res_rules["reduction_pct"]

        # 2. Search
        res_search = optimize(nodes, edges, rain, river, method="search", seed=42)
        cost_search = res_search["scenario_cost_after"]
        red_search = res_search["reduction_pct"]

        print(f"--- Storm {s} (Test #{s_idx + 1}) ---")
        print(f"Cost Before (No Control): {cost_before:.2f} m-h")
        print(f"Cost After (Rules)      : {cost_rules:.2f} m-h ({red_rules:.1f}% reduction)")
        print(f"Cost After (Search)     : {cost_search:.2f} m-h ({red_search:.1f}% reduction)\n")

        records.append({
            "storm_id": s,
            "cost_before": cost_before,
            "cost_rules": cost_rules,
            "reduction_rules_pct": red_rules,
            "cost_search": cost_search,
            "reduction_search_pct": red_search,
        })

    df = pd.DataFrame(records)
    csv_out = Path("results/optimizer_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved optimizer results table -> {csv_out}")

    # Generate before/after chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(selected_storms))
    width = 0.25

    rects1 = ax.bar(x - width, df["cost_before"], width, label="No Control (Pumps OFF)", color="#e41a1c", alpha=0.85)
    rects2 = ax.bar(x, df["cost_rules"], width, label="Rules Optimizer (B7)", color="#377eb8", alpha=0.85)
    rects3 = ax.bar(x + width, df["cost_search"], width, label="Search Optimizer (B8)", color="#4daf4a", alpha=0.85)

    ax.set_ylabel("Flood Cost (Metre-Hours, Lower is Better)")
    ax.set_title("Flood Prevention: Before vs After Intelligent Control")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Storm {s}" for s in selected_storms])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            val = bar.get_height()
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    chart_out = Path("results/optimizer_before_after.png")
    fig.savefig(chart_out, dpi=160)
    plt.close(fig)
    print(f"Saved before/after chart -> {chart_out}")

    return df


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_optimizer_suite(data_dir=target)
