"""B5 - Prediction interface for Person C (Backend API / Dashboard).

Follows CONTRACT.md:
predict(nodes, edges, recent, model_path="models/best.pt")
Returns PREDICTION JSON dict.
Caches model in memory for sub-second responses.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

from src.common.io import build_adjacency, load_graph, load_dataset
from src.ml.model import STGNN

# Module-level cache for high performance
_CACHE = {
    "model_path": None,
    "model": None,
    "meta": None,
    "adj": None,
}

LEAD_INDICES = {
    "1h": 3,   # 4th step (index 3)
    "2h": 7,   # 8th step (index 7)
    "3h": 11,  # 12th step (index 11)
}


def _load_cached_model(nodes: pd.DataFrame, edges: pd.DataFrame, model_path: str = "models/best.pt"):
    """Load model and metadata once, caching for future calls."""
    global _CACHE
    model_path = str(model_path)

    if _CACHE["model"] is not None and _CACHE["model_path"] == model_path:
        return _CACHE["model"], _CACHE["meta"]

    p = Path(model_path)
    meta_path = p.parent / "best_meta.json"
    if not meta_path.exists():
        # Fallback to default metadata
        meta = {
            "norm_mean": [5.84, 0.095, 1.07, 0.019],
            "norm_std": [13.30, 0.61, 1.16, 0.14],
            "hidden_dim": 48,
        }
    else:
        with open(meta_path) as f:
            meta = json.load(f)

    num_nodes = len(nodes)
    adj_np = build_adjacency(nodes, edges)
    adj_tensor = torch.from_numpy(adj_np).float()

    hidden_dim = meta.get("hidden_dim", 48)
    model = STGNN(
        num_nodes=num_nodes,
        in_channels=4,
        hidden_dim=hidden_dim,
        history_steps=8,
        horizon_steps=12,
        adj=adj_tensor,
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    _CACHE["model_path"] = model_path
    _CACHE["model"] = model
    _CACHE["meta"] = meta
    _CACHE["adj"] = adj_tensor

    return model, meta


def predict(nodes: pd.DataFrame, edges: pd.DataFrame, recent: dict, model_path: str = "models/best.pt") -> dict:
    """Predict flood depth for each node 1h, 2h, and 3h ahead.

    Parameters
    ----------
    nodes : pd.DataFrame
        Nodes dataframe matching nodes.csv schema.
    edges : pd.DataFrame
        Edges dataframe matching edges.csv schema.
    recent : dict
        Dict with keys:
          'rain':    [8, N] in mm/h
          'fill':    [8, N] stored/capacity
          'river':   [8] in metres
          'pump_on': [8, N] binary 0/1
    model_path : str
        Path to PyTorch checkpoint.

    Returns
    -------
    dict
        PREDICTION JSON format matching CONTRACT.md:
        {
          "horizons_h": [1, 2, 3],
          "nodes": {
            "0": {"1h": float, "2h": float, "3h": float},
            ...
          },
          "flooded_nodes_3h": int,
          "worst_node": {"node_id": int, "depth_m": float, "at": str}
        }
    """
    model, meta = _load_cached_model(nodes, edges, model_path)
    N = len(nodes)

    rain = np.asarray(recent["rain"], dtype=np.float32)
    fill = np.asarray(recent["fill"], dtype=np.float32)
    river = np.asarray(recent["river"], dtype=np.float32)
    pump_on = np.asarray(recent["pump_on"], dtype=np.float32)

    # Broadcast river to all nodes [8, N]
    river_bc = np.repeat(river[:, None], N, axis=1)

    # Stack channels -> [8, N, 4]
    x_raw = np.stack([rain, fill, river_bc, pump_on], axis=-1)

    # Normalise
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std = np.array(meta["norm_std"], dtype=np.float32)
    x_norm = (x_raw - mean) / std

    # Add batch dim -> [1, 8, N, 4]
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float()

    with torch.no_grad():
        pred_12 = model(x_tensor).squeeze(0).numpy()  # [12, N]

    pred_12 = np.maximum(0.0, pred_12)  # depth cannot be negative

    # Extract 1h, 2h, 3h
    d_1h = pred_12[LEAD_INDICES["1h"]]  # [N]
    d_2h = pred_12[LEAD_INDICES["2h"]]
    d_3h = pred_12[LEAD_INDICES["3h"]]

    nodes_dict = {}
    flooded_3h_count = 0
    worst_id = 0
    worst_depth = -1.0
    worst_horizon = "1h"

    for i in range(N):
        v1 = float(round(d_1h[i], 2))
        v2 = float(round(d_2h[i], 2))
        v3 = float(round(d_3h[i], 2))

        nodes_dict[str(i)] = {"1h": v1, "2h": v2, "3h": v3}

        if v3 > 0.10:
            flooded_3h_count += 1

        # Check worst across all horizons
        for val, h in [(v1, "1h"), (v2, "2h"), (v3, "3h")]:
            if val > worst_depth:
                worst_depth = val
                worst_id = i
                worst_horizon = h

    if worst_depth < 0:
        worst_depth = 0.0

    return {
        "horizons_h": [1, 2, 3],
        "nodes": nodes_dict,
        "flooded_nodes_3h": int(flooded_3h_count),
        "worst_node": {
            "node_id": int(worst_id),
            "depth_m": float(worst_depth),
            "at": worst_horizon,
        },
    }


def demo(data_dir: str = None, model_path: str = "models/best.pt"):
    """Command-line demo: picks test storm, runs predict, prints valid JSON."""
    if data_dir is None:
        data_dir = "data/hindmata" if Path("data/hindmata/nodes.csv").exists() else "data/sample"
    nodes, edges = load_graph(data_dir)
    ds = load_dataset(data_dir)

    split_file = Path("results/split.json")
    if split_file.exists():
        with open(split_file) as f:
            test_storms = json.load(f)["test"]
    else:
        test_storms = [0]

    storm_id = test_storms[0]
    # Pick a mid-storm window with active rain/fill
    t_start = 15
    recent = {
        "rain": ds["rain"][storm_id, t_start: t_start + 8],
        "fill": ds["fill"][storm_id, t_start: t_start + 8],
        "river": ds["river"][storm_id, t_start: t_start + 8],
        "pump_on": ds["pump_on"][storm_id, t_start: t_start + 8],
    }

    result = predict(nodes, edges, recent, model_path=model_path)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    target = None
    for arg in sys.argv[1:]:
        if arg != "--demo" and not arg.startswith("-"):
            target = arg
    demo(data_dir=target)
