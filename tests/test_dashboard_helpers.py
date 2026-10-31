"""Unit tests for pure dashboard helpers and data access layer.
"""
import pytest
import numpy as np
import pandas as pd
from dashboard.helpers import (
    step_to_clock,
    horizon_steps,
    depth_band,
    build_recent,
    rank_storms_by_peak_rain,
    haversine_distance_m,
    nearest_node,
    risk_table,
)
from dashboard.data_access import load_area, list_areas, Dataset


def test_step_to_clock():
    assert step_to_clock(0, 900) == "T+00:00"
    assert step_to_clock(1, 900) == "T+00:15"
    assert step_to_clock(4, 900) == "T+01:00"
    assert step_to_clock(71, 900) == "T+17:45"
    # Non-15-min dt (e.g. 10m = 600s)
    assert step_to_clock(6, 600) == "T+01:00"


def test_horizon_steps():
    assert horizon_steps(1.0, 900) == 4
    assert horizon_steps(2.0, 900) == 8
    assert horizon_steps(3.0, 900) == 12
    # 10m steps (600s)
    assert horizon_steps(1.0, 600) == 6


def test_depth_band_boundaries():
    label, rgba = depth_band(0.0)
    assert label == "Safe"
    assert rgba[0] == 56

    label, _ = depth_band(9.9)
    assert label == "Safe"

    label, _ = depth_band(10.0)
    assert label == "Minor"

    label, _ = depth_band(29.9)
    assert label == "Minor"

    label, _ = depth_band(30.0)
    assert label == "Moderate"

    label, _ = depth_band(59.9)
    assert label == "Moderate"

    label, _ = depth_band(60.0)
    assert label == "Severe"

    label, _ = depth_band(100.0)
    assert label == "Severe"

    label, _ = depth_band(100.1)
    assert label == "Extreme"


def test_haversine_distance():
    # Distance from Dadar to CST (~8.5 km)
    d = haversine_distance_m(19.0178, 72.8478, 18.9400, 72.8353)
    assert 8000 < d < 10000


def test_nearest_node():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lat": [19.00, 19.01, 19.02],
        "lon": [72.80, 72.81, 72.82],
    })
    nid, dist = nearest_node(19.009, 72.809, nodes_df)
    assert nid == 1
    assert dist < 200.0


def test_rank_storms_by_peak_rain():
    # 3 storms, 10 steps, 2 nodes
    rain = np.array([
        [[10, 15], [5, 2]],
        [[30, 45], [20, 10]],
        [[80, 110], [50, 40]],
    ])
    ranked = rank_storms_by_peak_rain(rain)
    assert ranked["Extreme cloudburst"] == 2


def test_build_recent():
    dummy_ds = Dataset(
        name="test",
        nodes=pd.DataFrame(),
        edges=pd.DataFrame(),
        rain=np.ones((2, 20, 5), dtype=np.float32),
        river=np.ones((2, 20), dtype=np.float32),
        fill=np.ones((2, 20, 5), dtype=np.float32),
        pump_on=np.zeros((2, 20, 5), dtype=np.float32),
        depth=np.zeros((2, 20, 5), dtype=np.float32),
    )
    # At t=5 (< 8 steps history), should return None
    assert build_recent(dummy_ds, 0, 5, history_steps=8) is None

    # At t=10, should return [8, 5] arrays
    recent = build_recent(dummy_ds, 0, 10, history_steps=8)
    assert recent is not None
    assert recent["rain"].shape == (8, 5)
    assert recent["river"].shape == (8,)


def test_risk_table_ordering():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lat": [19.0, 19.01, 19.02],
        "lon": [72.8, 72.81, 72.82],
        "elevation_m": [10.0, 5.0, 2.0],
        "has_pump": [0, 1, 0],
    })
    depths = np.array([0.05, 0.45, 0.20])
    rt = risk_table(depths, nodes_df, top_k=3)
    assert rt.iloc[0]["node_id"] == 1
    assert rt.iloc[0]["depth_cm"] == 45.0
    assert rt.iloc[1]["node_id"] == 2
    assert rt.iloc[2]["node_id"] == 0


def test_load_sample_area_contract():
    ds, err = load_area("sample")
    assert err is None
    assert ds is not None
    assert ds.N == 200
    assert ds.T == 72
    assert ds.S == 60
    assert ds.is_model_compatible is False


def test_load_hindmata_area():
    ds, err = load_area("hindmata")
    assert err is None
    assert ds is not None
    assert ds.N == 89
    assert ds.E == 140
    assert ds.is_model_compatible is True
