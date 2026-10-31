# CONTRACT: the rules all 3 of us follow

If you want to change anything in this file, tell the other two on chat FIRST, then edit, commit to `main`, and everyone pulls. This file is what stops "it works on my laptop" problems.

## 0. Names and units

| Thing | Value |
|---|---|
| Time step | 15 minutes (`DT_SECONDS = 900`) |
| Model input history | last 8 steps (2 hours) |
| Model output | next 12 steps (3 hours). We report +1h (step 4), +2h (step 8), +3h (step 12) |
| Rain | millimetres per hour (mm/h) |
| Water depth on street ("flood depth") | metres (m). Flooded means depth > 0.10 m |
| River level | metres (m) |
| Fill | stored water divided by node capacity. 1.0 = node is exactly full. More than 1.0 = flooding |
| Flood cost | total flood depth over all nodes and all time, in metre-hours. Lower is better |

S = number of storms, T = time steps per storm (72 = 18 hours), N = number of nodes, E = number of edges.

## 1. Files in `data/<area_name>/`

### `nodes.csv` (one row per junction, `node_id` must be 0..N-1 in order)
`node_id, lat, lon, elevation_m, capacity_m3, catchment_m2, runoff_coeff, has_pump, pump_rate_m3s, is_outlet`

### `edges.csv` (one row per pipe/road link, direction = water flows from `source` to `target`)
`edge_id, source, target, capacity_m3s, length_m, is_gate`
`is_gate = 1` means a sluice gate sits on this edge and the optimizer may open or close it.

### Big arrays (`.npy`, NOT committed to git, shared by Google Drive link)
| File | Shape | Meaning |
|---|---|---|
| `rain.npy` | [S, T, N] | rain at each node (mm/h) |
| `river.npy` | [S, T] | river gauge level (m) |
| `pump_on.npy` | [S, T, N] | 1 if the pump at that node is running |
| `gate_open.npy` | [S, T, E] | 1 if gate open (only matters where `is_gate = 1`) |
| `fill.npy` | [S, T, N] | stored water / capacity (the "water level" sensor value) |
| `depth.npy` | [S, T, N] | flood depth in metres. **This is what the model predicts** |

All are `float32`. No NaN. `depth >= 0`.

### Check any dataset with:
```
python -m src.common.check_contract data/sample
```
It must print `OK`. Person A runs it before sharing a dataset.

### Fake data for hour 0
`python -m src.sim.make_fake_data data/sample 60` makes a fake dataset so B and C can start right now. It is fake. Never put its numbers in the pitch.

## 2. Function contracts

### Person A provides: `src/sim/simulator.py`
```python
simulate(nodes, edges, rain, river, pump_on, gate_open, dt=900.0)
# rain [T,N]  river [T]  pump_on [T,N]  gate_open [T,E]
# returns {"fill": [T,N], "depth": [T,N], "flood_cost": float}
```
A may rewrite the inside as much as they like. Name, inputs and outputs stay the same.

### Person B provides: `src/ml/predict.py`
```python
predict(nodes, edges, recent, model_path="models/best.pt")
# recent = {"rain": [8,N], "fill": [8,N], "river": [8], "pump_on": [8,N]}
# returns a dict in the PREDICTION JSON format below
```

### Person B provides: `src/opt/optimize.py`
```python
optimize(nodes, edges, rain, river, method="search", seed=0)
# rain [T,N], river [T]
# returns a dict in the OPTIMIZER JSON format below
```

### Person C provides: the API in `src/api/` and the dashboard in `dashboard/`
Endpoints: `GET /scenario/{name}`, `POST /predict`, `POST /optimize`, `GET /sensor-stream`.
Scenario names: `light`, `heavy`, `extreme`.

## 3. JSON formats

### PREDICTION JSON
```json
{
  "horizons_h": [1, 2, 3],
  "nodes": {
    "12": {"1h": 0.00, "2h": 0.12, "3h": 0.31},
    "13": {"1h": 0.00, "2h": 0.00, "3h": 0.05}
  },
  "flooded_nodes_3h": 7,
  "worst_node": {"node_id": 12, "depth_m": 0.31, "at": "3h"}
}
```
Depths are in metres. A key for every node (0..N-1).

### OPTIMIZER JSON
```json
{
  "scenario_cost_before": 41.2,
  "scenario_cost_after": 27.9,
  "reduction_pct": 32.3,
  "method": "search",
  "pump_schedule": [{"step": 10, "node_id": 5, "on": 1}],
  "gate_schedule": [{"step": 12, "edge_id": 40, "open": 0}],
  "pump_on": "[T,N] array saved to results/opt_pump_on.npy",
  "gate_open": "[T,E] array saved to results/opt_gate_open.npy"
}
```
Costs are flood cost (metre-hours). `reduction_pct = (before - after) / before * 100`.

## 4. Git rules (so we don't destroy each other's work)

| Person | Branch | Only edits |
|---|---|---|
| A | `person-a/data-sim` | `src/sim/`, `data/`, `docs/data_notes.md` |
| B | `person-b/ml-optimizer` | `src/ml/`, `src/opt/`, `models/`, `results/`, `tests/ml/` |
| C | `person-c/dashboard-api` | `src/api/`, `dashboard/`, `README.md`, slides |

- Never edit another person's folder. Ask them.
- Edit `CONTRACT.md` only after telling the others.
- Commit small and often. Pull `main` into your branch at least every 2 hours.
- Merge into `main` at hours 6, 10 and 13 (checkpoints), and only when your part runs.
- The trained model file `models/best.pt` is small, so we DO commit it.
- Big data (`.npy`) goes on Google Drive. Put the link in `data/README_DATA.md`.

## 5. Honesty rules for the pitch

- Training data comes from our simulator because cities do not publish drain network data and flood labels. We say that out loud.
- Real inputs: OSM street network, elevation model, historical rainfall. Simulated: radar-style rain fields, IoT river gauge stream, flood ground truth.
- Always compare the model to a baseline. Always test on storms the model never saw.
