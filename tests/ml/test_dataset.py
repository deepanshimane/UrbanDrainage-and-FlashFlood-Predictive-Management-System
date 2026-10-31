"""Tests for B1 – dataset loader.

Run with:  .venv\\Scripts\\pytest tests/ml/test_dataset.py -v
"""
import json
import numpy as np
import torch
import pytest
from pathlib import Path

from src.ml.dataset import split_storms, build_loaders, SEED

DATA_DIR = "data/sample"


def test_split_no_overlap():
    """No storm index should appear in more than one split."""
    train, val, test = split_storms(60)
    all_sets = [set(train), set(val), set(test)]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = all_sets[i] & all_sets[j]
            assert len(overlap) == 0, f"Splits {i} and {j} share storms: {overlap}"


def test_split_covers_all():
    """All 60 storms must appear somewhere."""
    train, val, test = split_storms(60)
    all_idx = sorted(train + val + test)
    assert all_idx == list(range(60))


def test_split_sizes():
    """70/15/15 split should give 42 / 9 / 9 storms for 60 total."""
    train, val, test = split_storms(60)
    assert len(train) == 42
    assert len(val) == 9
    assert len(test) == 9


def test_split_reproducible():
    """Same seed must give the same split every time."""
    t1, v1, te1 = split_storms(60, seed=SEED)
    t2, v2, te2 = split_storms(60, seed=SEED)
    assert t1 == t2 and v1 == v2 and te1 == te2


def test_batch_shapes():
    """Input shape must be [B, 8, N, 4] and target [B, 12, N]."""
    loaders, meta = build_loaders(DATA_DIR, batch_size=4, save_split=False)
    x, y = next(iter(loaders["train"]))
    B, H, N, C = x.shape
    assert H == 8,  f"Expected 8 history steps, got {H}"
    assert C == 4,  f"Expected 4 channels, got {C}"
    assert N == meta["N"], f"Node count mismatch"
    B2, Ho, N2 = y.shape
    assert Ho == 12, f"Expected 12 horizon steps, got {Ho}"
    assert N2 == N
    assert B == B2 == 4


def test_no_test_leak_in_train():
    """Not a single test-storm sample should appear in the train loader.

    We check by verifying test_idx has zero overlap with train_idx (already
    checked in split tests), and that the split.json saved file is consistent.
    """
    # build with save so split.json is written
    _, meta = build_loaders(DATA_DIR, batch_size=4, save_split=True)
    split_path = Path("results/split.json")
    assert split_path.exists(), "split.json was not saved"
    with open(split_path) as f:
        saved = json.load(f)
    assert set(saved["train"]) & set(saved["test"]) == set()
    assert set(saved["train"]) & set(saved["val"]) == set()


def test_target_non_negative():
    """Depth targets must be >= 0 (no negative flood depth)."""
    loaders, _ = build_loaders(DATA_DIR, batch_size=16, save_split=False)
    x, y = next(iter(loaders["train"]))
    assert (y >= 0).all(), "Found negative depth in target"


if __name__ == "__main__":
    loaders, meta = build_loaders(DATA_DIR, batch_size=4, save_split=True)
    x, y = next(iter(loaders["train"]))
    print(f"[OK] Input batch : {tuple(x.shape)}")
    print(f"[OK] Target batch: {tuple(y.shape)}")
    print(f"[OK] Norm mean   : {meta['norm_mean']}")
    print(f"[OK] Norm std    : {meta['norm_std']}")
