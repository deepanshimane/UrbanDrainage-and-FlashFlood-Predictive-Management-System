"""Run this any time: python -m src.common.check_contract data/sample
It checks that a dataset follows CONTRACT.md. If it prints OK, B and C can trust the data."""
import sys
import numpy as np
from src.common.io import load_graph, load_dataset

NODE_COLS = ["node_id","lat","lon","elevation_m","capacity_m3","catchment_m2","runoff_coeff","has_pump","pump_rate_m3s","is_outlet"]
EDGE_COLS = ["edge_id","source","target","capacity_m3s","length_m","is_gate"]

def main(d):
    nodes, edges = load_graph(d); ds = load_dataset(d)
    assert list(nodes.columns) == NODE_COLS, f"nodes.csv columns must be {NODE_COLS}"
    assert list(edges.columns) == EDGE_COLS, f"edges.csv columns must be {EDGE_COLS}"
    N, E = len(nodes), len(edges); S, T = ds["rain"].shape[:2]
    want = dict(rain=(S,T,N), river=(S,T), pump_on=(S,T,N), gate_open=(S,T,E), fill=(S,T,N), depth=(S,T,N))
    for k, shp in want.items():
        assert ds[k].shape == shp, f"{k}.npy shape {ds[k].shape} should be {shp}"
        assert np.isfinite(ds[k]).all(), f"{k}.npy has NaN/inf"
    assert (nodes.node_id.to_numpy() == np.arange(N)).all(), "node_id must be 0..N-1 in order"
    assert edges.source.between(0, N-1).all() and edges.target.between(0, N-1).all(), "edge ids out of range"
    assert (ds["depth"] >= 0).all(), "depth must be >= 0"
    print(f"OK: {S} storms, {T} steps, {N} nodes, {E} edges")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
