"""B10 - Final Scenario Generation & Pitch Benchmarks.

Generates 3 canonical scenarios for Person C (Dashboard/API) and the pitch:
1. 'light':   rain peak ~ 25 mm/h  (controlled rainfall)
2. 'heavy':   rain peak ~ 60 mm/h  (severe monsoon thunderstorm)
3. 'extreme': rain peak ~ 110 mm/h (cloudburst / flash flood event)

For each scenario:
- Generates and saves rain [T, N] and river [T] arrays to results/scenarios/
- Computes baseline flood cost (pumps off, gates open)
- Computes optimized flood cost with AI search control
- Calculates flooded node counts and reduction %
- Saves results/scenario_summary.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.common.io import load_graph, HISTORY_STEPS, HORIZON_STEPS
from src.sim.simulator import simulate
from src.ml.predict import predict
from src.opt.optimize import optimize

SCENARIOS_DIR = Path("results/scenarios")


def make_controlled_storm(nodes: pd.DataFrame, peak_mmh: float, seed: int = 42, T: int = 72):
    """Generate a reproducible storm with a specified peak rainfall intensity."""
    rng = np.random.default_rng(seed)
    N = len(nodes)
    x = (nodes.lon - nodes.lon.min()) / (nodes.lon.max() - nodes.lon.min())
    y = (nodes.lat - nodes.lat.min()) / (nodes.lat.max() - nodes.lat.min())

    sigma = 0.4
    start, dur = 10, 16
    c0 = np.array([0.5, 0.4])
    vel = np.array([0.005, 0.008])

    t = np.arange(T)
    env = peak_mmh * np.exp(-0.5 * ((t - (start + dur / 2)) / (dur / 2.5)) ** 2)
    cx, cy = c0[0] + vel[0] * t, c0[1] + vel[1] * t
    d2 = (x.to_numpy()[None, :] - cx[:, None]) ** 2 + (y.to_numpy()[None, :] - cy[:, None]) ** 2
    rain = (env[:, None] * np.exp(-d2 / (2 * sigma ** 2))).astype(np.float32)

    river = np.zeros(T, np.float32)
    for k in range(1, T):
        river[k] = 0.95 * river[k - 1] + 0.01 * rain[k].mean()

    return rain, np.clip(river, 0, 6)


def run_scenarios(data_dir: str = "data/sample"):
    nodes, edges = load_graph(data_dir)
    N = len(nodes)
    T = 72
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("light", 25.0, 101),
        ("heavy", 60.0, 202),
        ("extreme", 110.0, 303),
    ]

    summary = {}
    table_rows = []

    print("Generating and evaluating the 3 canonical scenarios (light, heavy, extreme)...\n")

    for name, peak, seed in configs:
        rain, river = make_controlled_storm(nodes, peak_mmh=peak, seed=seed, T=T)

        # 1. Save arrays for Person C
        rain_file = SCENARIOS_DIR / f"{name}_rain.npy"
        river_file = SCENARIOS_DIR / f"{name}_river.npy"
        np.save(rain_file, rain)
        np.save(river_file, river)

        # 2. Baseline No Control
        no_pump = np.zeros((T, N), dtype=np.float32)
        all_open = np.ones((T, len(edges)), dtype=np.float32)
        base_sim = simulate(nodes, edges, rain, river, no_pump, all_open)
        cost_before = float(base_sim["flood_cost"])
        flooded_nodes_before = int((base_sim["depth"] > 0.10).any(axis=0).sum())
        max_depth_before = float(base_sim["depth"].max())

        # 3. Optimize with Search
        opt_res = optimize(nodes, edges, rain, river, method="search", seed=seed)
        cost_after = opt_res["scenario_cost_after"]
        red_pct = opt_res["reduction_pct"]

        # Run simulation with optimized arrays to get post-optimization node stats
        opt_pump = np.load("results/opt_pump_on.npy")
        opt_gate = np.load("results/opt_gate_open.npy")
        opt_sim = simulate(nodes, edges, rain, river, opt_pump, opt_gate)
        flooded_nodes_after = int((opt_sim["depth"] > 0.10).any(axis=0).sum())
        max_depth_after = float(opt_sim["depth"].max())

        # 4. Prediction Accuracy on early storm window
        recent = {
            "rain": rain[10:18],
            "fill": base_sim["fill"][10:18],
            "river": river[10:18],
            "pump_on": no_pump[10:18],
        }
        pred_res = predict(nodes, edges, recent)
        pred_flooded_3h = pred_res["flooded_nodes_3h"]

        summary[name] = {
            "peak_rainfall_mmh": peak,
            "rain_file": str(rain_file),
            "river_file": str(river_file),
            "flood_cost_before_mh": cost_before,
            "flood_cost_after_mh": cost_after,
            "reduction_pct": red_pct,
            "flooded_nodes_before": flooded_nodes_before,
            "flooded_nodes_after": flooded_nodes_after,
            "prevented_flooded_junctions": max(0, flooded_nodes_before - flooded_nodes_after),
            "max_water_depth_before_m": round(max_depth_before, 2),
            "max_water_depth_after_m": round(max_depth_after, 2),
            "early_warning_3h_detected_nodes": pred_flooded_3h,
        }

        table_rows.append({
            "Scenario": name.capitalize(),
            "Peak Rain": f"{peak:.0f} mm/h",
            "Cost Before (m-h)": f"{cost_before:.1f}",
            "Cost After (m-h)": f"{cost_after:.1f}",
            "Reduction %": f"{red_pct:.1f}%",
            "Flooded Junctions (Before -> After)": f"{flooded_nodes_before} -> {flooded_nodes_after}",
            "Max Depth (Before -> After)": f"{max_depth_before:.2f}m -> {max_depth_after:.2f}m",
        })

    # Save summary JSON
    summary_path = Path("results/scenario_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved scenario summary JSON -> {summary_path}")

    df_table = pd.DataFrame(table_rows)
    print("\n" + "="*80)
    print("PITCH-READY SCENARIO SUMMARY TABLE")
    print("="*80)
    print(df_table.to_string(index=False))
    print("="*80 + "\n")

    return summary


if __name__ == "__main__":
    run_scenarios()
