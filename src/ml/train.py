"""B4 - Training pipeline for the ST-GNN model.

Trains the ST-GNN using:
- Adam optimizer
- Weighted MSE Loss (penalty is 10x higher for flooded nodes so the AI doesn't ignore floods)
- Early stopping based on validation loss
- Fixed random seed for reproducibility
- Saves best checkpoint to models/best.pt and metadata to models/best_meta.json
- Saves training curve chart to results/training_curve.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless mode for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.common.io import load_graph, build_adjacency
from src.ml.dataset import build_loaders, SEED
from src.ml.model import STGNN
from src.ml.metrics import evaluate_lead_times, evaluate_flooded_only, FLOOD_THRESH


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, flood_weight: float = 12.0):
    """Weighted Mean Squared Error.

    Why? In urban drainage, 98% of the streets are dry during most hours.
    A naive model that predicts '0 water' everywhere achieves 98% accuracy but is useless.
    By penalizing errors on flooded streets 12x more, we force the AI to pay attention to floods.
    """
    weight = torch.where(target > FLOOD_THRESH, flood_weight, 1.0)
    return torch.mean(weight * (pred - target) ** 2)


def train_stgnn(
    data_dir: str = "data/sample",
    hidden_dim: int = 48,
    epochs: int = 15,
    lr: float = 1.5e-3,
    flood_weight: float = 12.0,
    patience: int = 4,
    batch_size: int = 64,
    device: str = "cpu",
):
    # Fix seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # 1. Load data & graph
    print(f"Loading data and graph from {data_dir}...")
    nodes, edges = load_graph(data_dir)
    adj_np = build_adjacency(nodes, edges)
    adj_tensor = torch.from_numpy(adj_np).float().to(device)

    loaders, meta = build_loaders(data_dir, batch_size=batch_size)
    num_nodes = meta["N"]

    # 2. Initialize Model
    model = STGNN(
        num_nodes=num_nodes,
        in_channels=4,
        hidden_dim=hidden_dim,
        history_steps=8,
        horizon_steps=12,
        adj=adj_tensor,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Path("models/best.pt")

    print(f"Starting training ST-GNN (hidden={hidden_dim}, max_epochs={epochs}, flood_weight={flood_weight})...\n")

    for epoch in range(1, epochs + 1):
        # ── Train loop ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_train_batches = 0
        for x_b, y_b in loaders["train"]:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = weighted_mse_loss(pred, y_b, flood_weight=flood_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # ── Validation loop ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_v, y_v in loaders["val"]:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v)
                v_loss = weighted_mse_loss(pred_v, y_v, flood_weight=flood_weight)
                val_loss += v_loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        improved = avg_val_loss < best_val_loss
        flag = "*" if improved else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {flag}")

        if improved:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    print(f"\nBest model saved to {best_model_path} with Val Loss = {best_val_loss:.4f}")

    # 3. Save training curve plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss (Weighted MSE)", color="#1f77b4", lw=2)
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss (Weighted MSE)", color="#ff7f0e", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ST-GNN Training Curve")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    curve_path = Path("results/training_curve.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Training curve chart saved -> {curve_path}")

    # 4. Load best checkpoint and evaluate on validation set
    print("\nEvaluating best ST-GNN checkpoint on Validation Storms...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_v, y_v in loaders["val"]:
            x_v = x_v.to(device)
            p = model(x_v).cpu().numpy()
            all_preds.append(p)
            all_trues.append(y_v.numpy())

    val_preds = np.concatenate(all_preds, axis=0)
    val_trues = np.concatenate(all_trues, axis=0)

    val_metrics = evaluate_lead_times(val_preds, val_trues)
    val_flood_metrics = evaluate_flooded_only(val_preds, val_trues)

    print("\n" + "="*50)
    print("ST-GNN VALIDATION PERFORMANCE")
    print("="*50)
    for lead in ["1h", "2h", "3h"]:
        m = val_metrics[lead]
        print(f"[{lead}] MAE: {m['mae_cm']:.2f} cm | RMSE: {m['rmse_cm']:.2f} cm | Flood F1: {m['f1']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f}")

    # 5. Save best_meta.json
    meta_path = Path("models/best_meta.json")
    save_meta = {
        "model_type": "ST-GNN",
        "num_nodes": num_nodes,
        "hidden_dim": hidden_dim,
        "history_steps": 8,
        "horizon_steps": 12,
        "norm_mean": meta["norm_mean"],
        "norm_std": meta["norm_std"],
        "best_val_loss": round(float(best_val_loss), 5),
        "validation_metrics": val_metrics,
        "validation_flooded_metrics": val_flood_metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(save_meta, f, indent=2)
    print(f"Metadata and normalisation numbers saved -> {meta_path}")

    return model, val_metrics


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else ("data/hindmata" if Path("data/hindmata/nodes.csv").exists() else "data/sample")
    train_stgnn(data_dir=target_dir)
