# FloodSense: 5-Minute Live Presentation & Demo Script
**Team:** FloodSense | **Focus:** Urban Drainage & Flash Flood Management (Mumbai Catchment)

---

## ⏱️ Live Presentation Flow (5 Minutes)

### 0:00 - 0:45 | 1. The Urban Crisis (Dadar-Hindmata-Parel Context)
> "Good afternoon judges. In dense coastal cities like Mumbai—particularly low-lying bottlenecks like Hindmata and Dadar TT Circle—intense monsoon cloudbursts coincide with high river tides, completely submerging streets under 1 to 3 metres of water within hours.
> 
> Traditional municipal drainage operations are strictly reactive: pumps are switched on only after junctions are already drowned, and sluice gates are operated blindly.
> 
> We built **FloodSense**: an AI forecasting and automated dispatch platform that predicts street-level inundation hours in advance and pre-emptively activates smart pumps and throttles gates to avert disasters."

---

### 0:45 - 1:45 | 2. System Overview & The Flood Map
*(Action: Open http://localhost:8501, select 'Sample (Mumbai Ward)', select 'Heavy storm')*
> "Here on our live dashboard, you see our 200-junction urban drainage network. 
> Notice the color hierarchy: cyan is safe under 10 cm, amber is minor, and bright crimson nodes indicate severe street inundation exceeding 1 metre.
>
> As I drag the timeline slider to step 24 (Hour 6), the storm peaks. Watch the low-elevation nodes near the outfalls turn dark red as the river gauge approaches critical backwater levels."

---

### 1:45 - 2:45 | 3. ST-GNN Multi-Horizon Forecast (+1h, +2h, +3h)
*(Action: Switch to '🔮 Multi-Horizon AI Forecast' Tab, toggle '+2 Hours Forecast')*
> "Here is our core AI capability: Spatio-Temporal Graph Neural Networks (ST-GNN). 
> The model ingests the last 2 hours of rainfall, node water fill, river gauge, and pump status, propagating runoff across pipe connections.
>
> In real-time, the model predicts inundation depths at +1h, +2h, and +3h lead times with an average error of just 2.56 cm. This gives emergency responders a vital 2-hour window before street inundation peaks."

---

### 2:45 - 3:45 | 4. Automated Dispatch Optimizer (Before vs After)
*(Action: Switch to '⚡ Pump/Gate Optimizer' Tab, click 'Run Smart Dispatch Optimizer')*
> "Prediction alone doesn't stop flooding—action does. 
> When we trigger our Smart Dispatch Optimizer, it executes stochastic local search across pump and sluice schedules.
>
> Look at the verification chart: baseline flooding cost of 40.5 m·h drops to 27.2 m·h—a **32.8% reduction in flood damage**! 
> The schedule table below provides immediate actionable commands: pre-emptively running pumps at nodes #5 and #12 and closing gate #40 to protect downstream neighborhoods."

---

### 3:45 - 4:30 | 5. Empirical Validation & Scalability
*(Action: Switch to '📊 Model Performance' Tab)*
> "We evaluated our ST-GNN on unseen test storms against standard baselines. Our graph AI outperforms the standard No-Graph baseline by more than 2x in F1 score (0.314 vs 0.132 at +2h lead time).
>
> Furthermore, inference takes just 8 milliseconds, making it fully viable for real-time edge deployment across municipal SCADA systems."

---

### 4:30 - 5:00 | 6. Data Honesty & Conclusion
> "A note of complete honesty: real sewer pipe cross-sections and IoT sensor telemetry are not public in Indian cities. We trained on rigorous hydraulic physics simulations grounded on OpenStreetMap and Mumbai DEM topography. The pipeline is designed so real municipal SCADA feeds plug directly into our FastAPI backend.
>
> Thank you, we welcome your questions!"

---

## ⚡ 60-Second Elevator Pitch Version
1. **Problem:** Mumbai flood hotspots like Hindmata drown every monsoon because pump/gate operations are reactive.
2. **Solution:** FloodSense uses a Spatio-Temporal Graph Neural Network to forecast street depths 1 to 3 hours in advance.
3. **Action:** Our dispatch optimizer pre-emptively schedules pumps and gates, cutting total flood inundation by 32.8%.
4. **Validation:** 8ms inference latency, 2.56 cm MAE, and tested on unseen storm events.

