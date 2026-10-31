"""Shared helpers. Everyone uses these so we all read data the SAME way."""
from pathlib import Path
import numpy as np
import pandas as pd

DT_SECONDS = 900          # one time step = 15 minutes
HISTORY_STEPS = 8         # model looks at the last 8 steps (2 hours)
HORIZON_STEPS = 12        # model predicts the next 12 steps (3 hours)

def load_graph(data_dir):
    d = Path(data_dir)
    return pd.read_csv(d / "nodes.csv"), pd.read_csv(d / "edges.csv")

def load_dataset(data_dir):
    """Returns dict of arrays. Supports both sample format and real area dataset/ format."""
    d = Path(data_dir)
    # Check for Person A's real area dataset format (dataset/X.npy)
    ds_dir = d / "dataset" if (d / "dataset" / "X.npy").exists() else d
    if (ds_dir / "X.npy").exists():
        X = np.load(ds_dir / "X.npy")  # [S, T, N, 4]
        # Features: 0=rain_mm, 1=tide_m, 2=depth_m, 3=pump_on
        S, T, N, _ = X.shape
        rain = X[..., 0].astype(np.float32)
        # Tide is identical for all nodes, extract [S, T]
        tide = X[..., 0, 1].astype(np.float32)
        depth = X[..., 2].astype(np.float32)
        pump_on = X[..., 3].astype(np.float32)
        
        gate_open = np.load(ds_dir / "gate_open.npy").astype(np.float32) if (ds_dir / "gate_open.npy").exists() else np.ones((S, T), dtype=np.float32)
        pump_sched = np.load(ds_dir / "pump_sched.npy").astype(np.float32) if (ds_dir / "pump_sched.npy").exists() else np.zeros((S, T, 1), dtype=np.float32)

        # Approximate fill from depth and street ponding
        fill = depth / 0.10

        return {
            "rain": rain,
            "river": tide,  # in Mumbai, tidal gauge is the river/outlet boundary
            "pump_on": pump_on,
            "gate_open": gate_open,
            "fill": fill,
            "depth": depth,
            "pump_sched": pump_sched,
            "X": X,
        }

    names = ["rain", "river", "pump_on", "gate_open", "fill", "depth"]
    return {n: np.load(d / f"{n}.npy") for n in names}


def build_adjacency(nodes, edges):
    """Symmetric, normalised N x N adjacency (with self loops) for the GNN."""
    n = len(nodes)
    a = np.zeros((n, n), dtype=np.float32)
    src = edges.source.to_numpy(int)
    dst = edges.target.to_numpy(int)
    a[src, dst] = 1.0
    a = np.maximum(a, a.T) + np.eye(n, dtype=np.float32)
    deg = a.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-6))
    return (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
