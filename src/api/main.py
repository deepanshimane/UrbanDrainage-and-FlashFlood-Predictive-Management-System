"""FastAPI backend for Urban Drainage and Flash Flood Predictive Management System.
Person C Implementation.

Endpoints:
  GET  /
  GET  /health
  GET  /areas
  GET  /scenarios
  GET  /scenario/{name}
  POST /predict
  POST /optimize
  GET  /cache/status
  GET  /sensor-stream
  GET  /graph
  GET  /known-spots
  GET  /metrics
"""
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import DT_SECONDS, HISTORY_STEPS, HORIZON_STEPS
from dashboard.data_access import list_areas, load_area, get_model_constraints
from dashboard.helpers import step_to_clock

app = FastAPI(
    title="FloodSense AI - Urban Drainage & Flash Flood Management API",
    description="Person C backend providing flood prediction, pump/gate dispatch optimization, and IoT sensor streams. All data simulated.",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_SAMPLE_DIR = REPO_ROOT / "data" / "sample"
RESULTS_DIR = REPO_ROOT / "results"

# In-memory optimizer cache: key -> dict
OPTIMIZER_CACHE: Dict[str, Any] = {}
CACHE_STATS = {"hits": 0, "misses": 0, "warmed": False, "warmup_duration_s": 0.0}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class WorstNodeInfo(BaseModel):
    node_id: int
    depth_m: float
    at: str


class PredictionResponse(BaseModel):
    horizons_h: List[int] = [1, 2, 3]
    nodes: Dict[str, Dict[str, float]]
    flooded_nodes_3h: int
    worst_node: WorstNodeInfo
    source: str = "model"               # "model" or "fallback"
    latency_ms: float = 0.0


class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    area: Optional[str] = Field("sample", description="Area name (e.g. sample)")
    scenario: Optional[str] = Field("heavy", description="Scenario name: light, heavy, extreme")
    model_path: Optional[str] = Field("models/best.pt", description="Path to trained model checkpoint")
    recent_rain: Optional[List[List[float]]] = Field(None, description="Recent 8 steps of rain [8, N]")
    recent_fill: Optional[List[List[float]]] = Field(None, description="Recent 8 steps of fill [8, N]")
    recent_river: Optional[List[float]] = Field(None, description="Recent 8 steps of river level [8]")
    recent_pump_on: Optional[List[List[float]]] = Field(None, description="Recent 8 steps of pump states [8, N]")


class PumpScheduleItem(BaseModel):
    step: int
    node_id: int
    on: int


class GateScheduleItem(BaseModel):
    step: int
    edge_id: int
    open: int


class OptimizeResponse(BaseModel):
    scenario_cost_before: float
    scenario_cost_after: float
    reduction_pct: float
    method: str
    pump_schedule: List[PumpScheduleItem]
    gate_schedule: List[GateScheduleItem]
    pump_on: str
    gate_open: str
    source: str = "model"               # "model", "cache", or "fallback"
    latency_ms: float = 0.0


class OptimizeRequest(BaseModel):
    area: Optional[str] = Field("sample", description="Area name (e.g. sample)")
    scenario: Optional[str] = Field("heavy", description="Scenario name: light, heavy, extreme")
    method: Optional[str] = Field("search", description="Optimization algorithm ('rules', 'search')")
    seed: Optional[int] = Field(0, description="Random seed")


# ---------------------------------------------------------------------------
# Background Warm-Up
# ---------------------------------------------------------------------------
def _warmup_optimizer():
    """Pre-computes and caches optimizer results for standard scenarios on startup."""
    global CACHE_STATS
    t0 = time.perf_counter()
    ds, err = load_area("sample")
    if ds is not None and err is None:
        try:
            from src.opt.optimize import optimize as run_opt
            for scen_name in ["light", "heavy", "extreme"]:
                key_name = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get(scen_name, "Heavy storm")
                storm_idx = ds.scenarios.get(key_name, 0)
                rain_T = ds.rain[storm_idx]
                river_T = ds.river[storm_idx]
                
                cache_key = f"sample:{scen_name}:search:0"
                if cache_key not in OPTIMIZER_CACHE:
                    res = run_opt(ds.nodes, ds.edges, rain_T, river_T, method="search", seed=0)
                    OPTIMIZER_CACHE[cache_key] = res
            CACHE_STATS["warmed"] = True
        except Exception as e:
            pass
    CACHE_STATS["warmup_duration_s"] = round(time.perf_counter() - t0, 2)


@app.on_event("startup")
async def startup_event():
    # Run warmup asynchronously
    asyncio.get_event_loop().run_in_executor(None, _warmup_optimizer)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "FloodSense AI API",
        "description": "Urban Drainage and Flash Flood Predictive Management System (All data simulated)",
        "team_role": "Person C (Dashboard & Backend API)",
        "docs_url": "/docs",
        "endpoints": [
            "GET /health", "GET /areas", "GET /scenarios", "GET /scenario/{name}",
            "POST /predict", "POST /optimize", "GET /cache/status",
            "GET /sensor-stream", "GET /graph", "GET /known-spots", "GET /metrics"
        ],
    }


@app.get("/health")
def health():
    ds, err = load_area("sample")
    return {
        "status": "healthy",
        "default_area": "sample",
        "nodes_loaded": ds.N if ds else 0,
        "edges_loaded": ds.E if ds else 0,
        "time_step_sec": ds.dt_seconds if ds else DT_SECONDS,
        "model_compatible": ds.is_model_compatible if ds else False,
        "cache_warmed": CACHE_STATS["warmed"],
    }


@app.get("/areas")
def get_areas():
    """Lists all detected urban study areas and their formats/compatibility."""
    return {"areas": list_areas()}


@app.get("/scenarios")
def get_scenarios():
    """Returns the 3 master scenarios with peak rain and storm hydrographs."""
    ds, err = load_area("sample")
    if ds is None:
        raise HTTPException(status_code=500, detail="Sample dataset unavailable.")
        
    res = {}
    for label, s_idx in ds.scenarios.items():
        key = label.lower().split()[0] # "light", "heavy", "extreme"
        rain_profile = [round(float(v), 2) for v in ds.rain[s_idx].mean(axis=-1).tolist()]
        river_profile = [round(float(v), 2) for v in ds.river[s_idx].tolist()]
        res[key] = {
            "name": key,
            "label": label,
            "storm_index": s_idx,
            "peak_rainfall_mmh": round(float(ds.rain[s_idx].max()), 1),
            "rain_profile": rain_profile,
            "river_profile": river_profile,
            "duration_hours": round(ds.T * ds.dt_seconds / 3600.0, 1),
            "steps": ds.T,
        }
    return res


@app.get("/scenario/{name}")
def get_scenario(name: str):
    """Returns hydrograph and metadata for a specific scenario."""
    scens = get_scenarios()
    normalized = name.lower().strip()
    if normalized not in scens:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found. Valid: {list(scens.keys())}")
    return scens[normalized]


@app.post("/predict", response_model=PredictionResponse)
def post_predict(req: Optional[PredictionRequest] = None):
    """Predicts flood depth at +1h, +2h, and +3h horizons for every junction node.
    Calls Person B's ST-GNN model in src.ml.predict if compatible; otherwise uses fallback.
    """
    req = req or PredictionRequest()
    t0 = time.perf_counter()
    area_name = req.area or "sample"
    
    ds, err = load_area(area_name)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Area '{area_name}' could not be loaded: {err}")
        
    # Check if we can run Person B's real model
    if ds.is_model_compatible:
        try:
            from src.ml.predict import predict as ml_predict
            
            # Prepare 'recent' dict
            if req.recent_rain and req.recent_fill and req.recent_river and req.recent_pump_on:
                recent = {
                    "rain": np.array(req.recent_rain, dtype=np.float32),
                    "fill": np.array(req.recent_fill, dtype=np.float32),
                    "river": np.array(req.recent_river, dtype=np.float32),
                    "pump_on": np.array(req.recent_pump_on, dtype=np.float32),
                }
            else:
                # Build recent from scenario storm at t=20
                s_key = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get((req.scenario or "heavy").lower(), "Heavy storm")
                storm_idx = ds.scenarios.get(s_key, 0)
                t_step = min(20, ds.T - 1)
                t_start = max(0, t_step - 8)
                recent = {
                    "rain": ds.rain[storm_idx, t_start:t_start+8],
                    "fill": ds.fill[storm_idx, t_start:t_start+8] if ds.fill is not None else np.zeros((8, ds.N), np.float32),
                    "river": ds.river[storm_idx, t_start:t_start+8],
                    "pump_on": ds.pump_on[storm_idx, t_start:t_start+8] if ds.pump_on is not None else np.zeros((8, ds.N), np.float32),
                }
                
            res = ml_predict(ds.nodes, ds.edges, recent, model_path=req.model_path or "models/best.pt")
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            return PredictionResponse(
                horizons_h=res.get("horizons_h", [1, 2, 3]),
                nodes=res.get("nodes", {}),
                flooded_nodes_3h=int(res.get("flooded_nodes_3h", 0)),
                worst_node=WorstNodeInfo(**res.get("worst_node", {"node_id": 0, "depth_m": 0.0, "at": "1h"})),
                source="model",
                latency_ms=latency_ms,
            )
        except Exception as e:
            pass  # Fall back to calibrated stub
            
    # Fallback estimation
    mult = {"light": 0.2, "heavy": 1.0, "extreme": 2.2}.get((req.scenario or "heavy").lower(), 1.0)
    elev_col = "elevation_m" if "elevation_m" in ds.nodes.columns else "elevation"
    elev = ds.nodes[elev_col].to_numpy(float) if elev_col in ds.nodes.columns else np.zeros(ds.N)
    elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    
    nodes_pred = {}
    flooded_count_3h = 0
    worst_id = 0
    worst_depth = 0.0
    
    for i in range(ds.N):
        topo_risk = (1.0 - elev_norm[i]) ** 1.8
        is_outlet = bool(ds.nodes.iloc[i].get("is_outlet", 0)) if "is_outlet" in ds.nodes.columns else False
        d1 = float(max(0.0, round(topo_risk * 0.08 * mult, 2)))
        d2 = float(max(0.0, round(topo_risk * 0.24 * mult + (0.04 if is_outlet else 0.0), 2)))
        d3 = float(max(0.0, round(topo_risk * 0.42 * mult + (0.09 if is_outlet else 0.0), 2)))
        nodes_pred[str(i)] = {"1h": d1, "2h": d2, "3h": d3}
        if d3 > 0.10:
            flooded_count_3h += 1
        if d3 > worst_depth:
            worst_depth = d3
            worst_id = i
            
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return PredictionResponse(
        horizons_h=[1, 2, 3],
        nodes=nodes_pred,
        flooded_nodes_3h=flooded_count_3h,
        worst_node=WorstNodeInfo(node_id=worst_id, depth_m=worst_depth, at="3h"),
        source="fallback",
        latency_ms=latency_ms,
    )


@app.post("/optimize", response_model=OptimizeResponse)
def post_optimize(req: Optional[OptimizeRequest] = None):
    """Optimizes pump & sluice gate schedules using Person B's search/rules optimizer."""
    req = req or OptimizeRequest()
    t0 = time.perf_counter()
    area_name = req.area or "sample"
    scen_name = (req.scenario or "heavy").lower()
    method = req.method or "search"
    seed = req.seed or 0
    
    cache_key = f"{area_name}:{scen_name}:{method}:{seed}"
    if cache_key in OPTIMIZER_CACHE:
        CACHE_STATS["hits"] += 1
        cached = OPTIMIZER_CACHE[cache_key]
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return OptimizeResponse(
            scenario_cost_before=cached["scenario_cost_before"],
            scenario_cost_after=cached["scenario_cost_after"],
            reduction_pct=cached["reduction_pct"],
            method=cached["method"],
            pump_schedule=[PumpScheduleItem(**p) for p in cached["pump_schedule"]],
            gate_schedule=[GateScheduleItem(**g) for g in cached["gate_schedule"]],
            pump_on=cached["pump_on"],
            gate_open=cached["gate_open"],
            source="cache",
            latency_ms=latency_ms,
        )
        
    CACHE_STATS["misses"] += 1
    ds, err = load_area(area_name)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Area '{area_name}' unavailable: {err}")
        
    if ds.is_model_compatible:
        try:
            from src.opt.optimize import optimize as run_opt
            s_key = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get(scen_name, "Heavy storm")
            storm_idx = ds.scenarios.get(s_key, 0)
            rain_T = ds.rain[storm_idx]
            river_T = ds.river[storm_idx]
            
            res = run_opt(ds.nodes, ds.edges, rain_T, river_T, method=method, seed=seed)
            OPTIMIZER_CACHE[cache_key] = res
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            return OptimizeResponse(
                scenario_cost_before=res["scenario_cost_before"],
                scenario_cost_after=res["scenario_cost_after"],
                reduction_pct=res["reduction_pct"],
                method=res["method"],
                pump_schedule=[PumpScheduleItem(**p) for p in res["pump_schedule"]],
                gate_schedule=[GateScheduleItem(**g) for g in res["gate_schedule"]],
                pump_on=res["pump_on"],
                gate_open=res["gate_open"],
                source="model",
                latency_ms=latency_ms,
            )
        except Exception as e:
            pass
            
    # Fallback stub
    costs = {"light": (14.2, 7.8, 45.1), "heavy": (41.2, 27.9, 32.3), "extreme": (98.6, 68.2, 30.8)}
    b, a, r = costs.get(scen_name, (41.2, 27.9, 32.3))
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return OptimizeResponse(
        scenario_cost_before=b,
        scenario_cost_after=a,
        reduction_pct=r,
        method=method,
        pump_schedule=[PumpScheduleItem(step=10, node_id=5, on=1)],
        gate_schedule=[GateScheduleItem(step=12, edge_id=40, open=0)],
        pump_on="results/opt_pump_on.npy",
        gate_open="results/opt_gate_open.npy",
        source="fallback",
        latency_ms=latency_ms,
    )


@app.get("/cache/status")
def get_cache_status():
    return {
        "cached_entries": list(OPTIMIZER_CACHE.keys()),
        "stats": CACHE_STATS,
    }


@app.get("/sensor-stream")
def get_sensor_stream(
    step: int = Query(24, ge=0, le=71),
    scenario: str = Query("heavy"),
    area: str = Query("sample"),
):
    """Simulated IoT stream for river gauge, rainfall, and node telemetry."""
    ds, err = load_area(area)
    if ds is None:
        raise HTTPException(status_code=404, detail="Area not found")
        
    s_key = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get(scenario.lower(), "Heavy storm")
    storm_idx = ds.scenarios.get(s_key, 0)
    step = min(step, ds.T - 1)
    
    current_rain = float(ds.rain[storm_idx, step].mean())
    current_river = float(ds.river[storm_idx, step])
    d_step = ds.depth[storm_idx, step]
    
    flooded_count = int((d_step > 0.10).sum())
    worst_d = float(d_step.max())
    
    return {
        "stream_type": "SIMULATED IoT Stream",
        "area": area,
        "scenario": scenario,
        "step": step,
        "time_str": step_to_clock(step, ds.dt_seconds),
        "current_rain_mmh": round(current_rain, 1),
        "river_level_m": round(current_river, 2),
        "river_max_threshold_m": 6.0,
        "flooded_nodes": flooded_count,
        "worst_depth_m": round(worst_d, 2),
        "total_nodes": ds.N,
    }


@app.get("/graph")
def get_graph(area: str = Query("sample")):
    """Returns nodes and edges geometry."""
    ds, err = load_area(area)
    if ds is None:
        raise HTTPException(status_code=404, detail="Area not found")
        
    return {
        "nodes": ds.nodes.to_dict(orient="records"),
        "edges": ds.edges.to_dict(orient="records"),
        "total_nodes": ds.N,
        "total_edges": ds.E,
    }


@app.get("/known-spots")
def get_known_spots(area: str = Query("hindmata")):
    """Returns historical flood spots if available."""
    ds, err = load_area(area)
    if ds and ds.known_spots is not None:
        return {"area": area, "known_spots": ds.known_spots.to_dict(orient="records")}
    return {"area": area, "known_spots": []}


@app.get("/metrics")
def get_metrics():
    """Reads and parses metrics from results/ directory produced by Person B."""
    final_metrics_path = RESULTS_DIR / "final_metrics.json"
    scenario_path = RESULTS_DIR / "scenario_summary.json"
    
    res = {"available": False, "models": [], "scenarios": {}}
    if final_metrics_path.exists():
        try:
            with open(final_metrics_path) as f:
                res["models"] = json.load(f)
            res["available"] = True
        except Exception:
            pass
            
    if scenario_path.exists():
        try:
            with open(scenario_path) as f:
                res["scenarios"] = json.load(f)
        except Exception:
            pass
            
    return res
