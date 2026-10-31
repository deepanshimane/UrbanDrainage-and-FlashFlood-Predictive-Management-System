# FloodSense: Multi-Modal AI Urban Drainage & Flash Flood Predictive Management System
**Study Area:** Mumbai (Dadar-Hindmata-Parel Catchment; Scalability Study: Kurla-Mithi River)  
**Hackathon Problem Statement 11 | AI for Environmental and Water Resources Engineering**

> ⚠️ **DATA HONESTY STATEMENT:** Real city underground drainage conduit maps, real-time sluice gate telemetry, and street inundation sensors are not publicly accessible due to municipal security constraints. In accordance with competition guidelines, all rainfall runoff, river surge, drainage flows, and flood labels are **SIMULATED** using an engineering hydraulic simulator. Digital Elevation Models (DEM) and street graph topology are sourced from OpenStreetMap.

---

## 🏗️ Architecture & Dataflow
```text
  [OpenStreetMap + DEM]         [Simulated Radar Storms]        [IoT Gauge Level]
           │                                 │                          │
           └────────────────────────┬────────┴──────────────────────────┘
                                    ▼
                      Physics Simulator (src/sim/)
                      - Street gravity routing
                      - Outfall tide backwater
                      - Pump drawdown dynamics
                                    │
                       Dataset (.npy / CSV)
                                    │
                                    ▼
                 Spatio-Temporal GNN Model (src/ml/)
                 - 4 features: [Rain, Fill, River, Pump]
                 - 2h History -> 3h Multi-Horizon Forecast
                                    │
                    FastAPI Service (src/api/)
                    - Endpoints: /predict, /optimize, /sensor-stream
                    - Sub-50ms Cached Schedule Optimizer
                                    │
                                    ▼
                   Interactive Streamlit Dashboard
                   - Pydeck 3D/2D Topographic Flood Map
                   - +1h, +2h, +3h Lead Time Comparison
                   - Automated Sluice Gate & Pump Dispatch
```

---

## 🚀 Quickstart & Running the Demo

### 1. Requirements
Ensure Python 3.10+ is installed:
```powershell
pip install -r requirements.txt
```

### 2. Generate Simulated Sample Dataset
```powershell
python -m src.sim.make_fake_data data/sample 60
python -m src.common.check_contract data/sample
```

### 3. Launch Both Services Automatically
Run the integrated launcher:
```powershell
.\run_demo.ps1
```
Or start manually in separate terminals:
```powershell
# Terminal 1: Backend API
python -m uvicorn src.api.main:app --port 8000

# Terminal 2: Streamlit Dashboard
python -m streamlit run dashboard/app.py --server.port 8501
```

---

## 📡 API Endpoints (Person C)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health, node count, and model readiness status |
| `GET` | `/areas` | Discovers available urban catchments and compatibility |
| `GET` | `/scenarios` | Returns Light, Heavy, and Extreme storm hydrographs |
| `POST` | `/predict` | Ingests 2h sensor history; outputs multi-horizon flood depths |
| `POST` | `/optimize` | Computes optimal pump schedule & gate throttling |
| `GET` | `/sensor-stream` | Live IoT river gauge and rainfall stream feed |
| `GET` | `/graph` | Network junction nodes and pipe conduit geometries |
| `GET` | `/metrics` | Empirical validation metrics (ST-GNN vs baselines) |

---

## 🎛️ Dashboard Guide (dashboard/app.py)

1. **Interactive Flood Map:** Visualizes street junction nodes color-coded by fixed severity bands (<10cm Safe, 10-30cm Minor, 30-60cm Moderate, 60-100cm Severe, >100cm Extreme). Displays pump station rings and river outfalls.
2. **Multi-Horizon AI Forecast:** Step-by-step forecast for +1h, +2h, and +3h lead times compared against simulator ground truth with real MAE (cm) tracking.
3. **Risk & Alerts:** Automated prioritized table of most inundated junctions with CSV export.
4. **Pump & Gate Optimizer:** One-click dispatch optimization displaying flood cost reduction before/after and an actuation timeline.
5. **IoT Sensor Gauge:** River water level telemetry chart with critical backwater thresholds.
6. **Model Performance:** Real empirical evaluation scores (F1, MAE, RMSE) comparing ST-GNN against Persistence and No-Graph baselines.
7. **About & Honesty:** Complete transparency on simulated components.
