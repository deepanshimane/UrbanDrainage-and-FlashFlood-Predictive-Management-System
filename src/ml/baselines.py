"""B3 - Baselines to beat.

Before claiming an AI model is good, we must compare it to simple baselines:
1. Persistence: Assume flood depth stays exactly what it is right now.
   We estimate current depth from the last observed fill sensor reading:
   depth_now = max(0, (fill - 1.0) * capacity) / pond_area.
2. No-Graph Model (Node-wise MLP):
   Looks at each node's own 8-step history (8 steps * 4 features = 32 inputs),
   completely ignoring neighboring pipes and streets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.io import load_graph, load_dataset, HISTORY_STEPS, HORIZON_STEPS
from src.ml.dataset import build_loaders, DrainageDataset
from src.ml.metrics import evaluate_lead_times, evaluate_flooded_only, FLOOD_THRESH

# ── Baseline 1: Persistence ───────────────────────────────────────────────────

def run_persistence_baseline(ds, val_idx, nodes):
    """Predict future 12 depths as equal to the current estimated depth.

    Current depth estimate:
    fill = stored / capacity
    if fill > 1.0: excess = (fill - 1.0) * capacity
    depth = min(3.0, excess / (0.02 * catchment_m2))
    """
    cap = nodes.capacity_m3.to_numpy(float)
    area = nodes.catchment_m2.to_numpy(float)
    pond = 0.02 * area
    N = len(nodes)

    all_preds = []
    all_targets = []

    for s in val_idx:
        fill = ds["fill"][s]    # [T, N]
        depth = ds["depth"][s]  # [T, N]
        T = fill.shape[0]

        for t in range(T - HISTORY_STEPS - HORIZON_STEPS + 1):
            hi = t + HISTORY_STEPS
            # last observed fill at step hi - 1
            curr_fill = fill[hi - 1]
            excess = np.maximum(0.0, (curr_fill - 1.0) * cap)
            curr_depth = np.minimum(3.0, excess / pond).astype(np.float32)  # [N]

            # persistence: repeat for all 12 horizon steps
            pred_12 = np.repeat(curr_depth[None, :], HORIZON_STEPS, axis=0)  # [12, N]
            target_12 = depth[hi: hi + HORIZON_STEPS]                         # [12, N]

            all_preds.append(pred_12)
            all_targets.append(target_12)

    y_pred = np.stack(all_preds, axis=0)      # [Samples, 12, N]
    y_true = np.stack(all_targets, axis=0)    # [Samples, 12, N]
    return y_pred, y_true


# ── Baseline 2: No-Graph Model (Node-wise MLP) ────────────────────────────────

class NodeWiseMLP(nn.Module):
    """Small neural net that predicts depth for each node in isolation.

    Input: [B, N, 32] (8 steps * 4 normalized features)
    Output: [B, N, 12] (12 horizon steps depth)
    """

    def __init__(self, in_features: int = 32, hidden_dim: int = 64, out_steps: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_steps),
            nn.ReLU(),  # depths are non-negative
        )

    def forward(self, x):
        # x is [B, 8, N, 4]
        B, H, N, C = x.shape
        # permute to [B, N, H, C] then flatten to [B, N, H*C]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, H * C)
        out = self.net(x_flat)  # [B * N, 12]
        # reshape back to [B, 12, N]
        out = out.view(B, N, -1).permute(0, 2, 1)
        return out


def train_no_graph_baseline(loaders, epochs: int = 15, lr: float = 1e-3, device: str = "cpu"):
    """Train the node-wise MLP baseline."""
    model = NodeWiseMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Weighted loss: penalty is 10x higher when actual depth > 0.10m
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_b, y_b in loaders["train"]:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)

            # Weighted MSE to avoid predicting all zeros
            weight = torch.where(y_b > FLOOD_THRESH, 10.0, 1.0)
            loss = torch.mean(weight * (pred - y_b) ** 2)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

    return model


def evaluate_model_on_val(model, val_loader, device: str = "cpu"):
    """Get all predictions and targets from val_loader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            pred = model(x_b).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y_b.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_pred, y_true


# ── Run and save comparison table ─────────────────────────────────────────────

def run_baselines(data_dir: str = "data/sample", out_csv: str = "results/baseline_metrics.csv"):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_graph(data_dir)
    ds = load_dataset(data_dir)
    loaders, meta = build_loaders(data_dir, batch_size=32)

    val_idx = meta["val_idx"]

    # 1. Evaluate Persistence
    print("Evaluating Baseline 1: Persistence...")
    p_pred, p_true = run_persistence_baseline(ds, val_idx, nodes)
    p_lead = evaluate_lead_times(p_pred, p_true)
    p_flood = evaluate_flooded_only(p_pred, p_true)

    # 2. Train and Evaluate No-Graph MLP
    print("Training Baseline 2: No-Graph Node-wise MLP...")
    mlp_model = train_no_graph_baseline(loaders, epochs=15)
    mlp_pred, mlp_true = evaluate_model_on_val(mlp_model, loaders["val"])
    mlp_lead = evaluate_lead_times(mlp_pred, mlp_true)
    mlp_flood = evaluate_flooded_only(mlp_pred, mlp_true)

    # Assemble rows for CSV
    rows = []
    for model_name, lead_res, flood_res in [
        ("Persistence", p_lead, p_flood),
        ("No-Graph MLP", mlp_lead, mlp_flood),
    ]:
        for horizon in ["1h", "2h", "3h"]:
            m = lead_res[horizon]
            f_f1 = flood_res[horizon].get("f1", 0.0)
            rows.append({
                "model": model_name,
                "lead_time": horizon,
                "mae_cm": m["mae_cm"],
                "rmse_cm": m["rmse_cm"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_overall": m["f1"],
                "f1_flooded_storms": f_f1,
                "iou": m["iou"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nBaseline metrics saved -> {out_csv}")
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    run_baselines(data_dir=target_dir)
