# 00 - FloodSense Project Plan (PS11, AI Manthan 2.0)

**Problem Statement 11:** AI-Powered Urban Drainage and Flash Flood Predictive Management System
**Area:** Mumbai - Hindmata / Dadar / Parel (main), Kurla / Mithi River (second area)
**Team:** 3 people (A: Data and Simulator, B: ML Model and Optimizer, C: Backend, Dashboard and Pitch)
**Time:** about 15 hours of build time

---

## 1. One-line idea

**FloodSense is a weather-and-water fortune teller that also runs the pumps.** It predicts which streets will flood in the next 1, 2 and 3 hours, and it decides when to turn pumps and tidal gates on or off to reduce the flooding.

## 2. Why Mumbai (our story)

- Judges know Mumbai floods. No need to explain why it matters.
- Mumbai floods because of **rain + weak drains + high tide together**, not because of slope. The old drains were built for about 25 mm of rain per hour, and high tide blocks the outfalls.
- BMC already has pumps and tidal gates (Hindmata, Gandhi Market, Brimstowad stations, Mithi floodgates). So pump and gate optimization is a real need.
- **Demo line:** "Same storm. At low tide, small flood. At high tide, big flood. FloodSense sees it coming and acts."
- **Risk:** flat terrain and noisy height data. We handle it by using tide and drain capacity as the main drivers, and by checking against known flood spots.

## 3. What the problem asks and what we build

| Problem statement asks | We build | Proof for judges |
|---|---|---|
| Ingest radar, river gauge, topography | Rain (real history + synthetic), simulated IoT gauge stream, tide, DEM heights, OpenStreetMap graph | Data flow diagram and live stream on the dashboard |
| Predict street-level flood maps hours ahead | ST-GNN model on the city graph, predicts depth at +1h, +2h, +3h | Map with time slider, metrics table |
| Optimize pumps and sluice gates | Optimizer that tries pump and gate schedules inside the simulator | Before/after: "flooded volume down by X%" |
| Hints: ST-GNN or ConvLSTM, OpenStreetMap | OSM graph + GCN + GRU model, baseline comparison | Model comparison table |
| End-to-end, scalable, real-world | One pipeline, second area runs with one name change | Second area demo |

## 4. System design

```
 Real rain history ─┐
 Synthetic storms ──┼─> SIMULATOR (Person A) ──> Dataset (X.npy, meta.csv)
 Tide (sine wave) ──┤        ^                          |
 OSM graph + DEM ───┘        |                          v
                             |                  ST-GNN MODEL (Person B)
                             |                          |
                     OPTIMIZER (Person B) <─────────────┘
                             |
                             v
                  FastAPI BACKEND (Person C) <── IoT stream (Person A script)
                             |
                             v
                   DASHBOARD (Person C): map, time slider,
                   alerts, pump/gate schedule, before/after
```

### Core pieces
1. **City graph:** nodes = merged road junctions (200-600), edges = road/drain links with capacity. Node has height, area, runoff coefficient, pump zone.
2. **Simulator (ground truth):** every 10 minutes: rain, local drains (reduced when tide is high and gates are closed), pumps, spreading by water-level difference. Fast NumPy code (about 6 ms per storm).
3. **Dataset:** 300+ storms, 96 steps each (16 hours), features `[rain_mm, tide_m, depth_m, pump_on]`. Includes 2-3 real heavy-rain days, with a worst-case copy where high tide hits at the rain peak. Split 70/15/15 **by storm**.
4. **Model (ST-GNN):** graph layer for space + GRU for time. Input: last 12 steps (2 hours). Output: depth at +6, +12, +18 steps.
5. **Optimizer:** first a rule-based schedule (pumps ON before the predicted flood). Then random search over about 200 schedules using the simulator. No reinforcement learning (too risky for 15 hours).
6. **Backend:** FastAPI endpoints `/predict`, `/optimize`, `/scenario/{name}`, `/sensor-stream`.
7. **Dashboard:** map with coloured nodes, time slider (+1h, +2h, +3h), scenario buttons (light rain, heavy storm, cloudburst), alerts, pump/gate table, before/after chart, top-10 risky streets.
8. **GenAI part (for the "Use of Generative AI" score):** an LLM writes the alert text in simple English and Hindi from the model's output ("Hindmata junction may flood in 2 hours. Turn on pumps now."). Optional: a small Q&A box for officials.

## 5. Roles

| Person | Owns | Main files |
|---|---|---|
| **A** (Achal) | Graph, DEM, simulator, dataset, validation, IoT stream, second area | `build_area.py`, `simulator.py`, `gen_dataset.py`, `iot_stream.py` |
| **B** | Baseline, ST-GNN, evaluation, `predict()`, optimizer, before/after numbers | `model.py`, `train.py`, `evaluate.py`, `optimizer.py` |
| **C** | Backend, dashboard, LLM alerts, README, slides, demo video, rehearsals | `api.py`, `dashboard.py`, `alerts.py` |

Detailed steps for A are in `01_PERSON_A_PLAN.md` and `02_PERSON_A_IMPLEMENTATION.md`. Shared file formats are in `03_DATA_CONTRACT.md`.

**Interfaces (do not change without telling everyone):**
- A to B: `X.npy`, `meta.csv`, `nodes.csv`, `edges.csv`, `pumps.csv`, and `from simulator import load_sim`.
- B to C: `predict(...)` returns JSON `{node_id: {1h, 2h, 3h}}`, and `optimize(...)` returns a pump/gate schedule plus before/after numbers.
- A to C: IoT stream messages, `scenarios.json`.

## 6. Timeline (15 hours)

| Hours | Everyone | A | B | C |
|---|---|---|---|---|
| 0-1 | Read problem, agree on data contract, make GitHub repo, pick roles | Setup, find pump coordinates | Setup Colab, read contract | Setup repo, Streamlit |
| 1-5 | Build in parallel | Graph, DEM, simulator, 50-storm dataset | Model code on fake data (baseline first, then ST-GNN) | Dashboard on fake data |
| **5-6** | **Checkpoint 1:** first real data goes to B | Hand over dataset | Confirm shapes load | Confirm formats match |
| 6-10 | Train, tune, connect | Full dataset, real storms, known-spot check, IoT stream | Train models, write optimizer | Connect API, add alerts |
| **10-12** | **Checkpoint 2:** full end-to-end run | Help fix bugs | Final numbers | Integrate and test |
| 12-13 | Final metrics, second area | Kurla/Mithi area | Final charts | Polish dashboard |
| **13** | **Code freeze.** Only bug fixes. | | | |
| 13-15 | Slides, README, backup video, 3 demo rehearsals | README data section | README model section | Slides, video, rehearsals |

Sleep in shifts: at least 2 to 3 hours each. Tired teams make the worst bugs.

## 7. Scope: what to build

**Must have**
1. OSM graph with heights, areas and pumps
2. Rain + tide generator
3. Flood simulator
4. Dataset with train/val/test split by storm
5. Baseline model, then ST-GNN
6. Evaluation script with metrics
7. Pump/gate optimizer
8. Backend API
9. Dashboard with map and time slider

**Should have**
10. Simulated IoT gauge stream
11. Scenario buttons
12. Before/after comparison
13. Top-10 risky streets
14. LLM alerts in English and Hindi

**Nice to have (only if time remains)**
15. Second area (Kurla-Mithi)
16. Downloadable alert report (CSV or PDF)
17. "What if a pump fails?" button

## 8. Metrics (our "acceptable accuracy")

- **MAE and RMSE** of predicted depth vs simulated depth (cm)
- **Flood/no-flood F1** per node (flooded means depth above 10 cm)
- **Accuracy by lead time** (+1h will be better than +3h, and that is normal, so show it)
- **Baseline comparison:** persistence ("same as now") and a simple model that ignores the graph. If ST-GNN does not win, say it honestly and use the better one.
- **Optimizer result:** total flooded volume or flooded nodes, with and without optimization, on 3 test storms
- **Known-spot check:** Hindmata and other known spots flood more often than the average node

Do not promise numbers before you measure them.

## 9. Honesty rules (very important)

- Real: OpenStreetMap streets, DEM heights, Open-Meteo rain history.
- Simulated: drain flows, tide, flood depth, IoT sensors, radar-style input.
- Real BMC drain maps, radar and live gauges are not public. Say it openly and show how real data plugs in through the same file format.
- The AI learns the simulator, so accuracy means "how well it copies the simulator on unseen storms." Never say it is proven on real Mumbai floods.
- Open-Meteo rain underestimates extreme rain. Mention it.
- Never claim "real-time radar". Say "radar-style rainfall input".

## 10. Judging criteria and our answer

| Criterion (10 each) | What we show |
|---|---|
| Innovation and Originality | Prediction + pump/gate control + tide in one system |
| Problem Understanding and Relevance | Mumbai facts: 25 mm/h drains, tide-blocked outfalls, Hindmata |
| Technical Approach and Implementation | OSM graph, simulator, ST-GNN with baseline comparison |
| Prototype Functionality and Completeness | Live end-to-end demo |
| Creativity and Design Excellence | Clean map dashboard with time slider and scenario buttons |
| Use of Technology / Generative AI | LLM alerts in English and Hindi, optional officials Q&A |
| Social Impact and Value Addition | Early warnings, fewer damaged shops and stranded commuters |
| Sustainability and Feasibility | Low-cost sensors, works with open data, runs on any area |
| Cross-Domain Integration | Civil, weather, IoT, AI, GIS, control |
| Presentation, Teamwork and Communication | Clear roles, rehearsed 5-minute demo, backup video |

## 11. Demo script (5 minutes)

1. **Problem (30 s):** one sentence plus a Mumbai flood photo. "Mumbai floods when rain, weak drains and high tide meet."
2. **Map (30 s):** the Hindmata area graph with pump locations.
3. **Heavy storm at low tide (45 s):** choose the scenario, show the flood map at +1h, +2h, +3h. Small flooding.
4. **Same storm at high tide (45 s):** much bigger flooding. This is the key moment.
5. **Live sensors and alert (45 s):** IoT stream, alert "Hindmata junction will flood in 2 hours".
6. **Optimize (60 s):** click "Optimize pumps and gates". Show before/after: "flooded volume down by X%".
7. **Numbers (45 s):** metrics table, ST-GNN vs baseline.
8. **Scale and real-world (30 s):** switch to the second area, explain how real BMC data plugs in.

Have a **recorded backup video** in case Wi-Fi or the app fails.

## 12. Slide outline (6-8 slides)

1. Title, team, one-line idea
2. The problem (Mumbai flood facts)
3. Our approach and architecture diagram
4. Data: what is real, what is simulated
5. Model and optimizer
6. Results: metrics and before/after
7. Impact, feasibility and next steps (real BMC data, more areas)
8. Team and thank you

## 13. Risks and fixes

| Risk | Fix |
|---|---|
| DEM too flat or noisy | Use FABDEM, smooth, and check with known flood spots. Document any calibration. |
| Simulator not ready by hour 5 | Hand over a crude version. Keep file formats the same. |
| ST-GNN is hard to train | Use plain PyTorch: one graph layer + GRU. Keep a baseline ready. |
| ST-GNN does not beat the baseline | Report honestly, use the better model |
| Optimizer shows no benefit | Check pump zones and capacity, and test on a storm at high tide |
| Integration fails at hour 12 | Everyone fixes it together. Cut "nice to have" items first. |
| Demo fails on stage | Pre-generated data and models saved locally, plus the backup video |
| Over-building | Code freeze at hour 13 |
| Hackathon rule: all work created during the event | Write and adapt code during the event, use only open-source libraries and public data, and ask organizers if unsure |

## 14. Tools

- **Language:** Python everywhere
- **Data and graph:** `osmnx`, `networkx`, `geopandas`, `rasterio`, `numpy`, `pandas`
- **Model:** PyTorch (plain GCN + GRU, or `torch_geometric_temporal`)
- **Optimizer:** random search with the simulator (`scipy` if needed)
- **Backend:** FastAPI
- **Dashboard:** Streamlit with `pydeck` or `folium`
- **GenAI:** an LLM API for alert text (keep API keys out of GitHub)
- **Training:** Google Colab free GPU, or laptop CPU for a small graph
- **Team:** one GitHub repo, one folder per person, commit often

## 15. Final checklist

- [ ] Data contract agreed (hour 0)
- [ ] First dataset to B (hour 5)
- [ ] Dashboard runs on fake data (hour 5)
- [ ] Baseline and ST-GNN trained and compared
- [ ] Optimizer shows before/after numbers on 3 storms
- [ ] End-to-end run works (hour 12)
- [ ] Second area runs
- [ ] Code freeze (hour 13)
- [ ] README, slides, backup video done
- [ ] 3 full demo rehearsals done
- [ ] Models and datasets saved locally for the demo
- [ ] Everyone can explain the "real vs simulated" answer
