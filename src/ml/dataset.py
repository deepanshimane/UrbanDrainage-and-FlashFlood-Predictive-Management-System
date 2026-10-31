"""B1 – Data loader and training pieces.

Think of a "storm" like a 18-hour rainfall movie for the whole city.
We cut each movie into overlapping 20-step clips (8 steps input + 12 steps target).
Like cutting a YouTube video into short clips to teach the AI.

Inputs per clip  : rain, fill, river (copied to every node), pump_on  -> shape [8, N, 4]
Target per clip  : flood depth for the NEXT 12 steps                  -> shape [12, N]
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.common.io import load_graph, load_dataset, HISTORY_STEPS, HORIZON_STEPS

# ── reproducible split ────────────────────────────────────────────────────────
SEED = 42
FLOOD_THRESH = 0.10   # metres – matches CONTRACT.md


def split_storms(n_storms: int, seed: int = SEED):
    """Split storm indices 70/15/15 by storm (never by step)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_storms)
    n_train = int(0.70 * n_storms)
    n_val   = int(0.15 * n_storms)
    train   = sorted(idx[:n_train].tolist())
    val     = sorted(idx[n_train:n_train + n_val].tolist())
    test    = sorted(idx[n_train + n_val:].tolist())
    return train, val, test


# ── normalisation ─────────────────────────────────────────────────────────────

def compute_norm_stats(ds, storm_idx):
    """Compute mean and std for the 4 input channels using TRAIN storms only.

    Channel order: [rain, fill, river-copied, pump_on]
    We compute stats across all train storms, all time steps, all nodes.
    """
    rain    = ds["rain"][storm_idx]       # [S_tr, T, N]
    fill    = ds["fill"][storm_idx]       # [S_tr, T, N]
    river   = ds["river"][storm_idx]      # [S_tr, T]
    pump_on = ds["pump_on"][storm_idx]    # [S_tr, T, N]

    # broadcast river -> [S_tr, T, N]
    N = rain.shape[2]
    river_bc = np.repeat(river[:, :, None], N, axis=2)

    # stack -> [S_tr, T, N, 4]
    stacked = np.stack([rain, fill, river_bc, pump_on], axis=-1)
    flat = stacked.reshape(-1, 4)

    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0   # avoid div-by-zero for near-constant channels
    return mean, std         # both shape [4]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class DrainageDataset(Dataset):
    """One sample = one (8-step input window, 12-step depth target) pair.

    We slide a window one step at a time inside each storm.
    A storm of T=72 steps gives (72 - 8 - 12 + 1) = 53 windows.
    """

    def __init__(self, ds, storm_idx, norm_mean, norm_std):
        self.samples = []   # list of (x, y) tensors
        N = ds["rain"].shape[2]

        for s in storm_idx:
            rain    = ds["rain"][s]     # [T, N]
            fill    = ds["fill"][s]
            river   = ds["river"][s]    # [T]
            pump_on = ds["pump_on"][s]
            depth   = ds["depth"][s]    # [T, N]  ← target

            river_bc = np.repeat(river[:, None], N, axis=1)  # [T, N]

            # slide window
            T = rain.shape[0]
            for t in range(T - HISTORY_STEPS - HORIZON_STEPS + 1):
                # input: last 8 steps, shape [8, N, 4]
                hi = t + HISTORY_STEPS
                x = np.stack([
                    rain[t:hi],
                    fill[t:hi],
                    river_bc[t:hi],
                    pump_on[t:hi],
                ], axis=-1).astype(np.float32)  # [8, N, 4]

                # normalise
                x = (x - norm_mean) / norm_std  # broadcast over [8, N, 4]

                # target: next 12 steps depth, shape [12, N]
                y = depth[hi: hi + HORIZON_STEPS].astype(np.float32)

                self.samples.append((
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── convenience builder ───────────────────────────────────────────────────────

def build_loaders(data_dir: str, batch_size: int = 32, save_split: bool = True):
    """Load data, split, normalise, return train/val/test DataLoaders + metadata.

    Returns
    -------
    loaders   : dict with keys 'train', 'val', 'test'
    meta      : dict with norm_mean, norm_std, split indices, N, E
    """
    data_dir = Path(data_dir)
    nodes, edges = load_graph(data_dir)
    ds = load_dataset(data_dir)

    S = ds["rain"].shape[0]
    meta_csv_path = data_dir / "dataset" / "meta.csv" if (data_dir / "dataset" / "meta.csv").exists() else data_dir / "meta.csv"
    if meta_csv_path.exists():
        meta_df = pd.read_csv(meta_csv_path)
        train_idx = meta_df.index[meta_df["split"] == "train"].tolist()
        val_idx = meta_df.index[meta_df["split"] == "val"].tolist()
        test_idx = meta_df.index[meta_df["split"] == "test"].tolist()
    else:
        train_idx, val_idx, test_idx = split_storms(S)

    # save split so we never accidentally leak test storms
    if save_split:
        split_path = Path("results/split.json")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": test_idx,
                       "seed": SEED, "n_storms": S}, f, indent=2)
        print(f"Split saved -> {split_path}  "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    norm_mean, norm_std = compute_norm_stats(ds, train_idx)

    train_ds = DrainageDataset(ds, train_idx, norm_mean, norm_std)
    val_ds   = DrainageDataset(ds, val_idx,   norm_mean, norm_std)
    test_ds  = DrainageDataset(ds, test_idx,  norm_mean, norm_std)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }

    meta = {
        "norm_mean": norm_mean.tolist(),
        "norm_std":  norm_std.tolist(),
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "test_idx":  test_idx,
        "N": len(nodes),
        "E": len(edges),
        "n_storms": S,
    }
    return loaders, meta


# ── quick smoke-test when run directly ───────────────────────────────────────

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else ("data/hindmata" if Path("data/hindmata/nodes.csv").exists() else "data/sample")
    loaders, meta = build_loaders(target)
    x, y = next(iter(loaders["train"]))
    print(f"Dataset target: {target}")
    print(f"Input batch shape  : {tuple(x.shape)}  (batch, history, nodes, channels)")
    print(f"Target batch shape : {tuple(y.shape)}  (batch, horizon, nodes)")
    print(f"Norm mean (4 channels): {meta['norm_mean']}")
    print(f"Norm std  (4 channels): {meta['norm_std']}")
