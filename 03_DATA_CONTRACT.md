# 03 - Data Contract (share this with Person B and Person C)

Agree on this at hour 0. If anything changes later, tell everyone first.

All files for one area live in `data/<area>/` (for example `data/hindmata/`).

## Units and basics
- Time step: **10 minutes** (600 s). One storm = **T = 96 steps** (16 hours).
- Length in metres, depth in metres, rain in **mm per 10-minute step**, tide in metres above mean sea level.
- Node ids are **0 to N-1**, with no gaps. Row number = id.
- Flooded means depth above **0.10 m**.

## Static files

### `nodes.csv`
| column | meaning |
|---|---|
| id | 0..N-1 |
| lat, lon | position (WGS84) |
| elevation | metres (smoothed DEM, relative height matters more than absolute) |
| area_m2 | land area that drains to this node |
| runoff_coeff | share of rain that becomes runoff (0.85) |
| pump_zone | index of the pump that serves this node, or -1 |
| has_pump | 1 if this node is a pump station location |

### `edges.csv` (undirected, each pair once)
| column | meaning |
|---|---|
| source, target | node ids |
| capacity_m3s | max flow along this link (m3/s) |
| length_m | length of the road |

### `pumps.csv`
| column | meaning |
|---|---|
| pump_id | 0..P-1 |
| node_id | node where the pump sits |
| name | station name |
| cap_m3s | pump capacity (m3/s) |

## Dataset files (`data/<area>/dataset/`)

| file | shape | meaning |
|---|---|---|
| `X.npy` | `[S, 96, N, 4]` float32 | per storm, time, node, feature |
| `gate_open.npy` | `[S, 96]` | 1 = outfall gate open, 0 = closed |
| `pump_sched.npy` | `[S, 96, P]` | 1 = pump ON, 0 = OFF |
| `meta.csv` | S rows | storm_id, source (synthetic or real), total_mm, peak_mm_per_hr, tide_amp, tide_mode, pump_mode, max_depth_m, flooded_nodes, **split** |
| `scenarios.json` | 3 keys | storm ids for dashboard buttons: light_rain, heavy_storm, extreme_cloudburst |
| `config.json` | - | T, dt, feature names, N, P |

**Feature order in `X[..., f]`:** `0 = rain_mm`, `1 = tide_m`, `2 = depth_m` (this is the target), `3 = pump_on` (for the node's pump zone).

**Split:** use the `split` column in `meta.csv` (train / val / test). The split is by **storm**, never by time step. Everyone must use the same split.

## Suggested model windows (Person B decides, but this is the default)
- Input: last **12 steps** (2 hours) of all 4 features + static node features (elevation, area).
- Output: depth at **+6, +12, +18 steps** (+1 h, +2 h, +3 h).

## Prediction output (B to C), JSON
```json
{
  "storm_id": 17,
  "t_now": 40,
  "nodes": {
    "0": {"1h": 0.02, "2h": 0.11, "3h": 0.25},
    "1": {"1h": 0.00, "2h": 0.04, "3h": 0.10}
  }
}
```
Depth in metres. Keys are node ids as strings.

## Using the simulator (for the optimizer, Person B)
```python
from simulator import load_sim
import numpy as np

sim, nodes, edges, pumps = load_sim("data/hindmata")
T, P = 96, sim.P
# rain: (T,) or (T, N) in mm per step ; tide: (T,) in m
depth = sim.simulate(rain, tide, gate_open=None, pump_on=np.ones((T, P)))
# depth has shape (T, N). Total flooding = (depth > 0.10).sum() or depth.sum()
```
- `gate_open=None` means automatic gates (open at low tide, closed at high tide).
- One call takes roughly 5-10 ms, so 200+ random schedules are fine.

## IoT stream message (A to C)
```json
{"t": 12, "sensor_id": "gauge_231", "water_level_m": 0.34, "rain_mm": 3.2, "tide_m": 1.7}
```

## Honesty note (for the README and slides)
Rain, tide and the water levels in the dataset come from a **simulator** built on real streets (OpenStreetMap), real height data (DEM) and real rainfall history (Open-Meteo). Real BMC drain maps, radar and sensor feeds are not public, so they are simulated. The same file format accepts real data later.
