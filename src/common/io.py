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
    """Returns dict of arrays. S=storms, T=time steps, N=nodes, E=edges."""
    d = Path(data_dir)
    names = ["rain", "river", "pump_on", "gate_open", "fill", "depth"]
    return {n: np.load(d / f"{n}.npy") for n in names}

def build_adjacency(nodes, edges):
    """Symmetric, normalised N x N adjacency (with self loops) for the GNN."""
    n = len(nodes)
    a = np.zeros((n, n), dtype=np.float32)
    a[edges.source.to_numpy(int), edges.target.to_numpy(int)] = 1.0
    a = np.maximum(a, a.T) + np.eye(n, dtype=np.float32)
    deg = a.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    return (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
