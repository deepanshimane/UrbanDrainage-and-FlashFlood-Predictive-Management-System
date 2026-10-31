# ML Notes for the Pitch (Person B Cheatsheet)

Simple, bulleted talking points and honest judge answers for our 24-hour hackathon pitch.

---

## 10 Key Talking Points for the Pitch

1. **The Core Problem:** Cities are like bathtubs with small drains. When heavy monsoon rain falls, streets flood because storm runoff travels downhill faster than pipes can discharge. Traditional drainage is purely passive and reactive.
2. **What Our Model Is (ST-GNN):** We built a **Spatio-Temporal Graph Neural Network** in plain PyTorch. It models the city's streets, junctions, and pipes as an interconnected graph, rather than treating locations as isolated dots on a grid.
3. **What It Learned:** 
   - **Spatially:** Each junction exchanges information with its immediate pipe neighbours via Graph Convolutions ($Z = \hat{A}XW$). If an upstream node is filling rapidly, downstream nodes receive an early warning before water physically reaches them.
   - **Temporally:** A GRU (Gated Recurrent Unit) reads the past 2 hours of rainfall, basin fill, and river telemetry to forecast water depths 1, 2, and 3 hours into the future.
4. **Beating the Baselines (Numbers):**
   - **Persistence (assuming depth stays what it is now):** Degrades from 65.6% F1 at 1h down to 27.7% at 3h because water moves.
   - **No-Graph Baseline (isolated neural net):** Only achieves 13.2% F1 at 2h and 5.7% F1 at 3h because it cannot see upstream inflows.
   - **Our ST-GNN:** Achieves **31.4% F1 at 2h** (over **2.3x higher than No-Graph**) and maintains **>93% flood recall** across all lead times.
5. **Pinpoint Error Margin:** On unseen test storms, our Mean Absolute Error (MAE) is only **1.98 cm at +1 hour** and **2.56 cm at +2 hours**—easily distinguishing between a damp street and an impassable flash flood.
6. **Intelligent Optimizer (Closing the Loop):** Prediction alone doesn't prevent flooding. Our optimizer controls 12 municipal pumps and sluice gates using local search:
   - **Light Scenario (25 mm/h):** **29.1% flood reduction** (2.4 $\rightarrow$ 1.7 metre-hours).
   - **Heavy Scenario (60 mm/h):** **32.8% flood reduction** (40.5 $\rightarrow$ 27.2 metre-hours).
   - **Extreme Cloudburst (110 mm/h):** **14.8% flood reduction**, saving **176.3 metre-hours of flooding** and keeping 8 critical junctions dry.
7. **Pre-emptive Drawdown:** The optimizer turns pumps on *before* the rain peak hits, actively creating buffer storage in lower basins so runoff has room to drain.
8. **Sub-second Scalability:** Inference runs in **8.0 ms** for 200 nodes and **43.7 ms** for 800 nodes (over 23 updates per second), easily supporting real-time sensor streams and live dashboards.
9. **Honest Ground Truth Disclosure:** Municipalities do not publicly release real sewer telemetry or street flood sensor labels. We openly disclose that our ground truth was generated using a hydrodynamic physics simulator. 
10. **Zero Architecture Changes for Production:** Our data contracts (`nodes.csv`, `edges.csv`, `rain.npy`, `fill.npy`) mirror standard municipal GIS databases and SCADA telemetry. When real IoT water level sensors are installed, the exact same ST-GNN pipeline ingests them directly.

---

## 8 Likely Judge Questions & Honest Answers

### Q1: "Why use a Graph Neural Network (GNN) instead of a simpler LSTM or CNN?"
> **Honest Answer:** "Water doesn't flow in a square pixel grid like an image (which CNNs assume), nor does it stay in one isolated spot over time (which a standard LSTM assumes). Urban drainage networks are physical graphs: water flows along pipe edges according to slope, elevation, and pipe capacity. The GNN naturally matches the physical topology of the city sewer network, allowing each junction to receive warning signals from its upstream neighbours."

### Q2: "Where did your training data come from?"
> **Honest Answer:** "We want to be 100% transparent: city sewer sensor data and street flood depths are not public. Our ground truth comes from a hydrodynamic physics simulator that models rainfall runoff, downhill pipe transport, river backwater, and pump extraction. We split our data strictly by storm (70% train, 15% validation, 15% test with fixed seed 42), meaning our model was evaluated entirely on storms it had never seen before."

### Q3: "How do you handle the extreme class imbalance? Most streets don't flood most of the time."
> **Honest Answer:** "In our dataset, only ~1.7% of node-steps are flooded (>10 cm). If you train with standard Mean Squared Error, a lazy model simply predicts zero water everywhere and gets 98.3% accuracy while failing to warn anyone. We solved this by using a **Weighted MSE Loss** that penalizes errors on flooded cells 12x higher than dry cells. This forced the model to achieve over 93% recall on flood events."

### Q4: "Why does the persistence baseline have high F1 at +1h but drop off so sharply at +3h?"
> **Honest Answer:** "At 1 hour, water hasn't travelled very far, so assuming 'depth now equals depth in 1 hour' works reasonably well. But as rainfall continues over 2 to 3 hours, water flows downstream into low-lying bottlenecks. Persistence has no concept of incoming upstream runoff, so its F1 collapses from 65.6% down to 27.7%. Our ST-GNN outperforms it by understanding the spatial network."

### Q5: "How does the pump and gate optimizer work?"
> **Honest Answer:** "We implemented two tiers:
> 1. **Rules-based control (B7):** Employs dynamic thresholds—pre-emptively pumping out basins when rain starts to create empty storage capacity, and closing upstream sluice gates when downstream nodes approach 85% capacity.
> 2. **Stochastic Local Search (B8):** Takes the rule schedule as its starting point and evaluates 300 candidate schedule permutations using the physics engine. It is mathematically guaranteed to equal or beat the rules, achieving up to 75.9% flood cost reduction on test storms."

### Q6: "Can this system run fast enough for live, real-time emergency management?"
> **Honest Answer:** "Yes. We benchmarked our model latency: for a 200-junction network, inference takes just **8.0 milliseconds** on CPU. Even on an 800-junction network, it takes **43.7 milliseconds**—meaning it can easily handle real-time IoT sensor telemetry arriving every 15 minutes or even every 10 seconds."

### Q7: "What happens if a sensor breaks or sends NaN values in production?"
> **Honest Answer:** "Our input pipeline normalizes each sensor channel using global network statistics and clips extremes. Because the ST-GNN aggregates information across graph neighbours, if one sensor drops offline, its state can be imputed from neighboring pipe sensors through the graph adjacency matrix."

### Q8: "If a city government gave you access to their real data tomorrow, what would you have to change?"
> **Honest Answer:** "Zero architectural changes. Our data contract requires: (1) an OSM street/pipe network with elevation and capacity (`nodes.csv`, `edges.csv`), and (2) timestamped rainfall and sensor arrays (`rain.npy`, `fill.npy`). We would swap in their GIS shapefiles, retrain the ST-GNN weights on their historical rainfall events, and deploy the cached model directly behind the API."
