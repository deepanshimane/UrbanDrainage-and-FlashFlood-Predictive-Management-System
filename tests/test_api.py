"""Unit and integration tests for FloodSense Person C FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["nodes_loaded"] > 0
    assert data["edges_loaded"] > 0


def test_areas_endpoint():
    resp = client.get("/areas")
    assert resp.status_code == 200
    areas = resp.json().get("areas", [])
    names = [a["name"] for a in areas]
    assert "sample" in names
    assert "hindmata" in names
    assert "kurla" in names


def test_scenarios_endpoint():
    resp = client.get("/scenarios")
    assert resp.status_code == 200
    scens = resp.json()
    assert "light" in scens
    assert "heavy" in scens
    assert "extreme" in scens
    assert len(scens["heavy"]["rain_profile"]) > 0


def test_predict_endpoint_contract():
    # Test primary Hindmata area (ST-GNN model inference)
    resp = client.post("/predict", json={"area": "hindmata", "scenario": "heavy"})
    assert resp.status_code == 200
    data = resp.json()
    assert "horizons_h" in data
    assert data["horizons_h"] == [1, 2, 3]
    assert "nodes" in data
    assert len(data["nodes"]) == 89
    assert "worst_node" in data
    assert data["source"] == "model"
    assert "latency_ms" in data

    # Test sample benchmark area fallback
    resp_sample = client.post("/predict", json={"area": "sample", "scenario": "heavy"})
    assert resp_sample.status_code == 200
    assert len(resp_sample.json()["nodes"]) == 200

    # Test Kurla area (dedicated ST-GNN model inference)
    resp_kurla = client.post("/predict", json={"area": "kurla", "scenario": "heavy"})
    assert resp_kurla.status_code == 200
    assert len(resp_kurla.json()["nodes"]) == 91
    assert resp_kurla.json()["source"] == "model"


def test_optimize_endpoint_contract():
    resp = client.post("/optimize", json={"area": "hindmata", "scenario": "heavy", "method": "search", "seed": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert "scenario_cost_before" in data
    assert "scenario_cost_after" in data
    assert "reduction_pct" in data
    assert "pump_schedule" in data
    assert "gate_schedule" in data
    assert data["reduction_pct"] >= 0.0
    assert data["source"] in ("model", "cache")
    assert "latency_ms" in data


def test_optimize_cache_hits():
    # Cold / Cached check
    resp1 = client.post("/optimize", json={"area": "hindmata", "scenario": "light", "method": "search", "seed": 0})
    assert resp1.status_code == 200
    resp2 = client.post("/optimize", json={"area": "hindmata", "scenario": "light", "method": "search", "seed": 0})
    assert resp2.status_code == 200
    assert resp2.json()["source"] == "cache"


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    if data["available"]:
        assert len(data["models"]) > 0


def test_sensor_stream_endpoint():
    resp = client.get("/sensor-stream?step=16&scenario=heavy&area=hindmata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["step"] == 16
    assert "time_str" in data
    assert "current_rain_mmh" in data
    assert "river_level_m" in data


def test_invalid_scenario_error():
    resp = client.get("/scenario/non_existent_storm")
    assert resp.status_code == 404
