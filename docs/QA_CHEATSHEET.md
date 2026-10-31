# FloodSense: Technical Q&A Cheatsheet for Judges (20 Questions)

### 1. What data is real vs. simulated?
**Answer:** Real data includes the Mumbai street network geometry from OpenStreetMap, elevations from Digital Elevation Models (DEM), and realistic rainfall intensity distributions. Underground sewer pipe internal capacities, IoT river level sensors, and street flood labels are simulated using hydraulic approximations because municipal drain networks are not public records.

### 2. Why use a Graph Neural Network (ST-GNN) instead of an LSTM or CNN?
**Answer:** Water flows through physical pipe networks that form a graph, not a regular Euclidean grid. CNNs distort spatial connectivity along irregular street centerlines. LSTMs ignore spatial topography. ST-GNN models both time-series dynamics and spatial water flow downhill along connected graph edges.

### 3. How does your ST-GNN beat standard baselines?
**Answer:** On unseen test storms, ST-GNN achieved an F1 of 0.314 at +2h lead time compared to 0.132 for a No-Graph MLP baseline—more than double the performance. At +3h, the No-Graph model collapses to 0.057 while ST-GNN maintains 0.158.

### 4. What are your prediction lead times and error rates?
**Answer:** We predict at +1h, +2h, and +3h horizons. Mean Absolute Error (MAE) across the network is 1.98 cm at +1h, 2.56 cm at +2h, and 4.07 cm at +3h.

### 5. How long does model inference take?
**Answer:** Inference on CPU takes approximately 8.0 ms for a 200-node network, making it suitable for real-time edge deployment.

### 6. How does the dispatch optimizer work?
**Answer:** It uses a two-stage approach: first, reactive heuristic rules turn on pumps early when basin fill exceeds 35% and throttle gates if downstream junctions saturate; second, stochastic local search mutates pump on/off time intervals to minimize cumulative flood cost $\int \text{depth} \, dt$.

### 7. What flood cost reduction does the optimizer achieve?
**Answer:** Measured on test storms: 29.1% reduction for Light Rain, 32.8% reduction for Heavy Storms ($40.52 \to 27.23\text{ m·h}$), and 14.8% for Extreme Cloudbursts where infrastructure capacity is physically overwhelmed.

### 8. What is the definition of a "flooded node"?
**Answer:** Per our master engineering contract (`CONTRACT.md`), any junction with standing water depth exceeding 0.10 metres (10 cm) is classified as flooded.

### 9. What happens if the river level rises above the outfalls?
**Answer:** Our simulator models tidal backwater: gravity discharge rate scales as $\max(0, 1 - \text{river} / 6.0)$. When the river gauge reaches 6.0 m, outfalls stop draining completely, requiring active pumping.

### 10. How does the system scale to larger cities?
**Answer:** The GNN scales linearly with the number of edges via sparse adjacency multiplication. Benchmarks show inference takes 8 ms for 200 nodes and 43.7 ms for 800 nodes.

### 11. Can the model run if some IoT sensors fail?
**Answer:** Yes. The feature stack can impute missing sensors using neighbor averages across the graph adjacency matrix.

### 12. Why does extreme storm reduction drop to 14.8%?
**Answer:** During extreme 110 mm/h cloudbursts, total rainfall volume exceeds the physical pump and conduit capacity. Pumping helps, but cannot completely prevent flooding when storage is saturated.

### 13. How fast is the optimizer in production?
**Answer:** Cold stochastic search takes ~3 to 8 seconds. We implement in-memory caching and pre-computed warm-up on startup, reducing response latency to under 35 ms for cached scenarios.

### 14. Is the API compliant with standard REST conventions?
**Answer:** Yes, built with FastAPI, automatic Swagger documentation (`/docs`), Pydantic validation schemas, and CORS middleware for seamless dashboard and mobile client integration.

### 15. How does the dashboard communicate with the backend?
**Answer:** The dashboard operates in Auto-detect mode: it queries FastAPI on `localhost:8000` via HTTP. If the API is offline, it automatically falls back to direct module execution with zero downtime.

### 16. What is the time resolution of your system?
**Answer:** The simulation and forecast operate at 15-minute time steps ($DT = 900\text{ seconds}$), covering an 18-hour storm duration ($T = 72\text{ steps}$).

### 17. How is user interaction designed for emergency operators?
**Answer:** Time sliders show clock time (`T+HH:MM`), severity bands are color-coded (<10cm safe to >100cm extreme), and a one-click risk table provides CSV export for field teams.

### 18. Does the model require GPU infrastructure?
**Answer:** No. With PyTorch CPU runtime, inference is 8 ms, and the entire trained weight checkpoint (`models/best.pt`) is only 238 KB.

### 19. What are the key limitations of your system?
**Answer:** It does not model solid plastic waste clogging drain grates or structural pipe collapse. Future iterations would incorporate municipal maintenance logs.

### 20. How would municipal authorities deploy this in Mumbai?
**Answer:** By connecting BMC's automatic weather stations (AWS) and Mithi river radar gauges into our `/predict` endpoint, the system can trigger automated SCADA pump control signals directly.
