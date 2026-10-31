"""Tests for B5 - predict(...) interface.

Validates that:
- Keys match CONTRACT.md PREDICTION JSON format exactly
- Every node 0..N-1 is present in the dictionary
- All predicted depths are >= 0.0 metres
- Repeated calls are cached and execute in < 1 second
"""

import time
import numpy as np
import pytest

from src.common.io import load_graph, load_dataset
from src.ml.predict import predict

DATA_DIR = "data/sample"
MODEL_PATH = "models/best.pt"


@pytest.fixture
def graph_and_recent():
    nodes, edges = load_graph(DATA_DIR)
    ds = load_dataset(DATA_DIR)
    N = len(nodes)
    recent = {
        "rain": ds["rain"][0, :8],
        "fill": ds["fill"][0, :8],
        "river": ds["river"][0, :8],
        "pump_on": ds["pump_on"][0, :8],
    }
    return nodes, edges, recent


def test_predict_keys_and_schema(graph_and_recent):
    nodes, edges, recent = graph_and_recent
    res = predict(nodes, edges, recent, model_path=MODEL_PATH)

    # Check top-level keys
    assert "horizons_h" in res
    assert res["horizons_h"] == [1, 2, 3]
    assert "nodes" in res
    assert "flooded_nodes_3h" in res
    assert isinstance(res["flooded_nodes_3h"], int)
    assert "worst_node" in res
    assert "node_id" in res["worst_node"]
    assert "depth_m" in res["worst_node"]
    assert "at" in res["worst_node"]
    assert res["worst_node"]["at"] in ["1h", "2h", "3h"]


def test_predict_all_nodes_present(graph_and_recent):
    nodes, edges, recent = graph_and_recent
    res = predict(nodes, edges, recent, model_path=MODEL_PATH)

    N = len(nodes)
    assert len(res["nodes"]) == N
    for i in range(N):
        key = str(i)
        assert key in res["nodes"], f"Node {key} missing from predictions"
        entry = res["nodes"][key]
        for h in ["1h", "2h", "3h"]:
            assert h in entry
            assert entry[h] >= 0.0, f"Negative depth predicted for node {key} at {h}"


def test_predict_cached_speed(graph_and_recent):
    nodes, edges, recent = graph_and_recent
    # Warm up / load cache
    predict(nodes, edges, recent, model_path=MODEL_PATH)

    # Second call should be lightning fast (< 0.5s)
    t0 = time.time()
    predict(nodes, edges, recent, model_path=MODEL_PATH)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"Cached predict took {elapsed:.3f}s, expected < 1s"
