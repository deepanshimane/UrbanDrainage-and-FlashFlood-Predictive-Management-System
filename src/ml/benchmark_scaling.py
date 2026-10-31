"""B11 - Scalability benchmark for urban drainage graph neural network.

Measures how prediction inference latency and training step time scale
as the city network grows from 200 nodes to 400 and 800 nodes.
Saves results to results/scalability_benchmark.csv.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.ml.model import STGNN


def benchmark():
    Path("results").mkdir(parents=True, exist_ok=True)
    node_sizes = [200, 400, 800]
    records = []

    print("Running scalability benchmark on graph sizes: 200, 400, 800 nodes...\n")

    for n in node_sizes:
        # Create synthetic normalized adjacency with degree ~ 4
        adj_dense = torch.eye(n)
        for i in range(n):
            for offset in [1, -1, 2, -2]:
                j = (i + offset) % n
                adj_dense[i, j] = 1.0
        # Normalize
        deg = adj_dense.sum(1)
        d_inv_sqrt = 1.0 / torch.sqrt(deg)
        adj_norm = (adj_dense * d_inv_sqrt.unsqueeze(1)) * d_inv_sqrt.unsqueeze(0)

        model = STGNN(
            num_nodes=n,
            in_channels=4,
            hidden_dim=48,
            history_steps=8,
            horizon_steps=12,
            adj=adj_norm,
        )
        model.eval()

        # Measure single prediction inference latency
        # Batch size 1 (real-time streaming query)
        x_single = torch.randn(1, 8, n, 4)
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x_single)

        latencies = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                _ = model(x_single)
                latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        mean_latency_ms = float(np.mean(latencies))
        std_latency_ms = float(np.std(latencies))

        # Measure training step time (batch size 32)
        model.train()
        x_batch = torch.randn(32, 8, n, 4)
        y_batch = torch.randn(32, 12, n)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_step_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            optimizer.zero_grad()
            out = model(x_batch)
            loss = torch.mean((out - y_batch) ** 2)
            loss.backward()
            optimizer.step()
            train_step_times.append((time.perf_counter() - t0) * 1000.0)

        mean_train_step_ms = float(np.mean(train_step_times))

        records.append({
            "nodes": n,
            "inference_latency_ms": round(mean_latency_ms, 2),
            "inference_std_ms": round(std_latency_ms, 2),
            "train_step_ms": round(mean_train_step_ms, 2),
            "real_time_fps": round(1000.0 / mean_latency_ms, 1),
        })

        print(f"Nodes: {n:3d} | Single Prediction Latency: {mean_latency_ms:5.1f} ms | Train Step (B=32): {mean_train_step_ms:5.1f} ms")

    df = pd.DataFrame(records)
    csv_path = Path("results/scalability_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved scalability benchmark -> {csv_path}")
    return df


if __name__ == "__main__":
    benchmark()
