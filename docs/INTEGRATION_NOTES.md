# FloodSense Integration Notes — Person C
**Author:** Person C | **Branch:** person-c/integration | **Updated:** 2026-10-31

---

## A. Branch Inventory

| Branch | Key Files | Status |
|---|---|---|
| `main` | CONTRACT.md, src/sim/simulator.py, src/common/, data/sample/ (60 storms, 200 nodes) | Baseline |
| `origin/person-b/ml-optimizer` | src/ml/predict.py, src/opt/optimize.py, models/best.pt, best_meta.json, results/*.json, results/scenarios/*.npy | COMPLETE (94%) |
| `origin/Person-A` | build_area.py, gen_dataset.py, simulator.py (class-based), data/hindmata/, data/sample/{nodes,edges}.csv | COMPLETE (hindmata dataset, DIFFERENT FORMAT) |

---

## B. Data Formats

### Format 1: "contract" — data/sample/ (master format, B's model trained on this)
- Time step: **900 s (15 min)**, T = 72 steps (18 h), S = 60 storms
- Nodes: N = 200, Edges: E = 358
- nodes.csv columns: `node_id, lat, lon, elevation_m, capacity_m3, catchment_m2, runoff_coeff, has_pump, pump_rate_m3s, is_outlet`
- edges.csv columns: `edge_id, source, target, capacity_m3s, length_m, is_gate`
- Arrays: `rain[S,T,N] mm/h`, `river[S,T] m`, `pump_on[S,T,N]`, `gate_open[S,T,E]`, `fill[S,T,N]`, `depth[S,T,N] m`
- All float32, no NaN, depth >= 0, clipped at 3.0 m
- Flood threshold: 0.10 m

### Format 2: "areaA" — data/hindmata/ (Person A's format)
- Time step: **600 s (10 min)**, T = 96 steps (16 h), S = ~150+ storms
- Nodes: N = 89, Edges: E = 140, Pumps: P = 3
- nodes.csv columns: `id, lat, lon, elevation, area_m2, runoff_coeff, pump_zone, has_pump` (no `node_id`, no `capacity_m3`, no `is_outlet`)
- edges.csv columns: `source, target, capacity_m3s, length_m` (no `edge_id`, no `is_gate`)
- pumps.csv: `pump_id, node_id, name, cap_m3s`
- Dataset in `data/hindmata/dataset/`:
  - `X.npy` shape `[S, 96, N, 4]` — features: `[rain_mm, tide_m, depth_m, pump_on]`
  - `gate_open.npy` shape `[S, 96]` (single gate, not per-edge)
  - `pump_sched.npy` shape `[S, 96, P]`
  - `meta.csv`, `scenarios.json`, `config.json`
- known_spots.csv columns: `name, lat, lon` (5 rows: Hindmata junction, Dadar TT circle, Parel TT junction, Gandhi Market, King's Circle)
- Scenarios: light_rain=143, heavy_storm=93, extreme_cloudburst=147

### Format 3: "unrecognised" — data/kurla/ (NOT FOUND on any branch)
- Kurla/Mithi area does not exist in the repo yet. Will be shown as disabled.

---

## C. Interface Summary

### Person B — predict(...)
```python
# src/ml/predict.py (origin/person-b/ml-optimizer)
predict(
    nodes: pd.DataFrame,    # nodes.csv schema (contract format)
    edges: pd.DataFrame,    # edges.csv schema (contract format)
    recent: dict,           # {"rain":[8,N], "fill":[8,N], "river":[8], "pump_on":[8,N]}
    model_path: str = "models/best.pt"
) -> dict  # CONTRACT PREDICTION JSON
```
- Returns: `{horizons_h:[1,2,3], nodes:{id:{1h,2h,3h},...}, flooded_nodes_3h:int, worst_node:{node_id,depth_m,at}}`
- History: 8 steps (2 h at 15 min/step), Horizon: 12 steps (3 h)
- Model trained on N=200 nodes. Incompatible with N=89 hindmata.

### Person B — optimize(...)
```python
# src/opt/optimize.py (origin/person-b/ml-optimizer)
optimize(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    rain: np.ndarray,    # [T, N]
    river: np.ndarray,   # [T]
    method: str = "search",   # "rules" or "search"
    seed: int = 0
) -> dict  # CONTRACT OPTIMIZER JSON
```
- Returns: `{scenario_cost_before, scenario_cost_after, reduction_pct, method, pump_schedule:[{step,node_id,on}], gate_schedule:[{step,edge_id,open}], pump_on:str, gate_open:str}`
- Internally calls simulate() from src/sim/simulator.py — must use CONTRACT format nodes/edges

### Person A — simulate(...)
```python
# src/sim/simulator.py (main branch, CONTRACT)
simulate(
    nodes: pd.DataFrame, edges: pd.DataFrame,
    rain: np.ndarray,      # [T,N] mm/h
    river: np.ndarray,     # [T] m
    pump_on: np.ndarray,   # [T,N] binary
    gate_open: np.ndarray, # [T,E] binary
    dt: float = 900.0
) -> {"fill": [T,N], "depth": [T,N], "flood_cost": float}
```
- DT=900.0, RIVER_MAX=6.0, OUTLET_RATE=4.0, depth clipped to [0, 3.0] m

### Model Constraints (from best_meta.json)
- num_nodes: 200 (contract format only)
- history_steps: 8, horizon_steps: 12
- Features: [rain, fill, river, pump_on] (indices 0-3)
- dt_seconds: 900 (inferred; not stored in meta — matches contract)
- **Model is NOT compatible with hindmata (N=89, dt=600, different features)**

---

## D. Model Performance (from results/final_metrics.json — real measured values)

| Model | Lead Time | MAE (cm) | RMSE (cm) | F1 |
|---|---|---|---|---|
| Persistence | +1h | 0.85 | 10.67 | 0.656 |
| Persistence | +2h | 1.33 | 14.83 | 0.425 |
| Persistence | +3h | 1.52 | 16.63 | 0.277 |
| No-Graph MLP | +1h | 2.09 | 12.19 | 0.342 |
| No-Graph MLP | +2h | 5.65 | 20.69 | 0.132 |
| No-Graph MLP | +3h | 10.46 | 28.41 | 0.057 |
| **ST-GNN (Ours)** | **+1h** | **1.98** | **13.36** | **0.408** |
| **ST-GNN (Ours)** | **+2h** | **2.56** | **15.09** | **0.314** |
| **ST-GNN (Ours)** | **+3h** | **4.07** | **18.25** | **0.158** |

---

## E. Decision Log

| # | Phase | Decision | Reason |
|---|---|---|---|
| 1 | 0 | Do NOT merge origin/Person-A — use `git show`/`git checkout` for read-only recon | Person A format is incompatible with B's model; merging would break src/sim/simulator.py |
| 2 | 0 | hindmata format: "areaA" — show map/depth, DISABLE forecast, DISABLE optimizer | N=89 vs model N=200; dt=600 vs model dt=900; features differ |
| 3 | 0 | kurla area: does not exist on any branch — show as "not available yet" | Not found in git ls-tree on any remote |
| 4 | 0 | Use `git checkout origin/person-b/ml-optimizer -- <path>` for B's files (not merge) | Avoids merge conflicts with main |
| 5 | 0 | best_meta.json does not store dt_seconds explicitly — infer from contract (900) | Consistent with training data format |


| 6 | 1 | Merge `origin/person-b/ml-optimizer` via `git merge --no-ff` | Clean merge via ORT strategy; models/best.pt, src/ml, src/opt, tests/ml all imported intact. |
| 7 | 1 | Smoke test B on sample dataset heavy storm | `predict()` took 270 ms cold, returned 200 nodes; `optimize()` took 3.39 s, achieved 12.9% reduction on heavy storm. |
| 8 | 1 | Ran B's test suite with pytest | 16 passed in 14.4s (`tests/ml/test_dataset.py`, `test_metrics.py`, `test_predict.py`). |

| 9 | 2 | Created `dashboard/helpers.py` | Unit-testable pure functions: `step_to_clock`, `depth_band`, `horizon_steps`, `build_recent`, `risk_table`, `nearest_node`. |
| 10 | 2 | Created `dashboard/data_access.py` | Data Access Layer with `Dataset` dataclass, `list_areas()`, `load_area()` handling contract format and areaA format with compatibility detection. |
