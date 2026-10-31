"""Check if known flood spots (Hindmata, etc.) flood more often than average in heavy storms.
Also compares real storms under random vs worst-case tide alignments (the Tide Story).
"""
import sys, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

area = sys.argv[1] if len(sys.argv) > 1 else "data/hindmata"
nodes = pd.read_csv(f"{area}/nodes.csv")
meta = pd.read_csv(f"{area}/dataset/meta.csv")
X = np.load(f"{area}/dataset/X.npy", mmap_mode="r")
spots = pd.read_csv(f"{area}/known_spots.csv")

# Filter for heavy storms (total rain > 100mm)
heavy = meta[(meta.total_mm > 100) & (meta.split == "test")].storm_id.values
if len(heavy) == 0:
    heavy = meta[meta.total_mm > 100].storm_id.values

tree = cKDTree(np.c_[nodes.lat, nodes.lon])

# Average flood rate across all nodes in heavy storms
avg = np.mean([(X[k, :, :, 2].max(axis=0) > 0.10).mean() for k in heavy])

print("==========================================================")
print(f"KNOWN FLOOD SPOTS VALIDATION ({len(heavy)} heavy storms)")
print("==========================================================")
print(f"City Average Node: floods in {avg * 100:.1f}% of heavy storms\n")

results = []
for _, s in spots.iterrows():
    _, i = tree.query([s.lat, s.lon])
    hit = np.mean([X[k, :, i, 2].max() > 0.10 for k in heavy])
    elev = nodes.loc[i, "elevation"]
    max_d = np.mean([X[k, :, i, 2].max() for k in heavy])
    results.append({
        "name": s["name"],
        "node_id": i,
        "elevation_m": elev,
        "flood_freq_pct": hit * 100,
        "avg_depth_m": max_d
    })
    print(f"  {s['name']:25s} (Node {i:3d}, elev {elev:5.2f}m) -> Floods in {hit * 100:5.1f}% of heavy storms (avg peak depth: {max_d:.2f}m)")

res_df = pd.DataFrame(results)
res_df.to_csv(f"{area}/known_spot_validation.csv", index=False)
print(f"\nSaved validation results to {area}/known_spot_validation.csv")

# Tide Story Comparison on Real Storms
print("\n==========================================================")
print("THE TIDE STORY (Real Storms: Random vs Worst-Case High Tide)")
print("==========================================================")
real_storms = meta[meta.source.str.startswith("real:")].copy()
grouped = real_storms.groupby("source")
for src, g in grouped:
    rnd = g[g.tide_mode == "random"].iloc[0]
    wrst = g[g.tide_mode == "worst"].iloc[0]
    print(f"Storm: {src}")
    print(f"  - Low/Random Tide:  Max Depth = {rnd.max_depth_m:.2f}m | Flooded Nodes = {rnd.flooded_nodes}")
    print(f"  - High Tide Peak:   Max Depth = {wrst.max_depth_m:.2f}m | Flooded Nodes = {wrst.flooded_nodes}")
    delta_nodes = wrst.flooded_nodes - rnd.flooded_nodes
    print(f"  --> High tide increases flooded nodes by +{delta_nodes}\n")

