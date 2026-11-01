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
                   - Top Executive Alert Banner & Master Slider
                   - 5 Human-Centric Operational Tabs
                   - Bilingual Emergency Broadcasts (EN & HI)
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
| `GET` | `/areas` | Discovers available urban catchments (`hindmata`, `kurla`, `sample`) |
| `GET` | `/scenarios` | Returns Light, Heavy, and Extreme storm hydrographs |
| `POST` | `/predict` | Ingests 2h sensor history; outputs multi-horizon flood depths |
| `POST` | `/optimize` | Computes optimal pump schedule & gate throttling |
| `GET` | `/sensor-stream` | Live IoT river gauge and rainfall stream feed |
| `GET` | `/graph` | Network junction nodes and pipe conduit geometries |
| `GET` | `/metrics` | Empirical validation metrics (ST-GNN vs baselines) |

---

## 🎛️ Dashboard Guide (dashboard/app.py)

- **3-Second Executive Status Banner:** Prominent top-of-screen status bar providing instant situation awareness (🚨 Critical Alert, ⚠️ Moderate Warning, 🟢 All Clear) with automated mitigation actions and real street landmark names.
- **Global Timeline Controller:** Master slider and live clock (`00:00` to `16:00`) that synchronously controls all map layers, risk rankings, and alerts.
- **Tab 1: 🗺️ Live Situation & AI Prediction:** Pydeck dark-matter flood map with physical depth bands (Ankle-Deep, Knee-Deep, Waist-Deep), pump halo rings, and $+1\text{h}$, $+2\text{h}$, $+3\text{h}$ ST-GNN forecast cards (91.9% recall).
- **Tab 2: ⚡ Automated Pump & Gate Action:** Dual baselines optimization comparison showing proactive dispatch improvement against both Passive Drainage and Current Municipal Practice (9.4% mean reduction across test storms), the Mumbai "Tide Story" explainer, and the automated pump dispatch timetable.
- **Tab 3: 📢 Civic Alerts & Citizen Broadcast:** Bilingual emergency broadcast cards in English and Hindi (हिंदी) ready for public ward transmission, and prioritized flood hotspot table with 1-click CSV download.
- **Tab 4: 📊 AI Accuracy & IoT Telemetry:** Benchmark comparison against Persistence and No-Graph baselines ($1.08\text{ cm}$ MAE), plus real-time simulated ultrasonic outfall gauge chart.
- **Tab 5: ℹ️ Engineering Assumptions & Honesty:** Complete transparency on real OSM/DEM/reanalysis data vs. simulated internal pipe hydraulics.
