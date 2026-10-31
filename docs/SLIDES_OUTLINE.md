# FloodSense: Presentation Slides Outline (8 Slides)

### Slide 1: Title & Team
- **Title:** FloodSense: AI-Powered Urban Drainage & Flash Flood Predictive Management System
- **Subtitle:** Spatial Graph Neural Networks & Autonomous Dispatch Optimization for Mumbai
- **Team Roles:**
  - Person A: Topographic Graph & Physics Simulator Engine
  - Person B: Spatio-Temporal Graph Neural Network & Dispatch Optimizer
  - Person C: Backend API, Decision Dashboard & System Integration
- **Badge:** All flood telemetry and labels simulated for hackathon evaluation.

### Slide 2: The Urban Flood Crisis
- **Challenge:** Urban flash floods overwhelm stormwater networks within minutes.
- **Vulnerability:** Bottleneck catchments (e.g., Dadar-Hindmata-Parel) sit in natural topographic depressions.
- **The Core Flaw:** Conventional municipal management is reactive (pumps switched on after water rises).
- **Goal:** Shift from reactive recovery to proactive pre-emptive dispatch.

### Slide 3: System Architecture & Workflow
- **Data Ingestion:** Rainfall radar intensity, IoT river gauges, junction water fill.
- **Topological Backbone:** OpenStreetMap street network turned into a directional drainage graph.
- **Prediction:** Spatio-Temporal GNN (ST-GNN) propagating spatial runoff over time.
- **Actuation:** Heuristic rules + stochastic schedule search for pumps and sluice gates.
- **Interfaces:** Real-time FastAPI backend + interactive Streamlit command center.

### Slide 4: AI Model: Spatio-Temporal Graph Neural Network
- **Input:** 4 temporal features (`rain_mmh`, `water_fill`, `river_level`, `pump_status`) over last 2 hours (8 steps).
- **Spatio-Temporal Convolution:** Spatial graph convolutions over normalized adjacency matrix $A$ capturing downhill flow.
- **Output:** Multi-horizon street inundation depth at +1h, +2h, and +3h lead times.
- **Inference Speed:** **[FILL FROM REAL RESULTS: 8.0 ms inference latency]** (source: `results/scalability_benchmark.csv`).

### Slide 5: Empirical Results & Baseline Comparison
- **Test Set:** Evaluated on held-out test storms unseen during training.
- **Metrics Table:**
  - *Lead Time +1h:* ST-GNN MAE **[FILL: 1.98 cm]**, F1 **[FILL: 0.408]** (vs No-Graph 0.342).
  - *Lead Time +2h:* ST-GNN MAE **[FILL: 2.56 cm]**, F1 **[FILL: 0.314]** (vs No-Graph 0.132 — >2x improvement).
  - *Lead Time +3h:* ST-GNN MAE **[FILL: 4.07 cm]**, F1 **[FILL: 0.158]** (vs No-Graph 0.057).
  *(Source: `results/final_metrics.json`)*

### Slide 6: Smart Dispatch Optimizer: Before vs After
- **Objective:** Minimize cumulative flood cost $\int \text{depth} \, dt$ (metre-hours).
- **Interventions:** Pre-emptive pump drawdown + protective downstream sluice throttling.
- **Results:**
  - Light Rain: **[FILL: 29.1% reduction]** (source: `results/scenario_summary.json`).
  - Heavy Storm: **[FILL: 32.8% reduction]** ($40.52 \to 27.23\text{ m·h}$).
  - Extreme Cloudburst: **[FILL: 14.8% reduction]** ($1189.4 \to 1013.1\text{ m·h}$).

### Slide 7: Scalability & Real-World Handoff
- **Transferability:** Graph structure accepts any urban catchment with elevation DEM and OSM road centerlines.
- **Second Study Area:** Tested scaling on Kurla-Mithi catchment network.
- **Latency Scaling:** Sub-50ms for 200 nodes; sub-100ms for 800 nodes.
- **API Integration:** Ready-to-deploy FastAPI microservice with automated in-memory caching.

### Slide 8: Data Honesty, Limitations & Next Steps
- **Transparent Honesty:** Underground pipe volumes and flood ground truth are simulated; topography and rain curves are grounded in reality.
- **Limitations:** Does not model solid waste debris clogging drains.
- **Next Steps:** Integration with BMC automatic weather stations and city SCADA PLCs.
