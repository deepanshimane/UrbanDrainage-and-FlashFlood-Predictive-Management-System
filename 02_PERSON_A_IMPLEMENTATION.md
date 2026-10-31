# 02 - Person A Implementation Guide (Mumbai FloodSense)

All code below was written for you as a **reference**. `simulator.py`, `gen_dataset.py` and `test_sanity.py` were tested on a fake grid (they run without internet). `build_area.py` needs internet (OpenStreetMap) and was only syntax-checked, so **test it first on your laptop** and fix small errors.

Read `01_PERSON_A_PLAN.md` for the why, and `03_DATA_CONTRACT.md` for the file formats.

---

## Step 0 - Setup (hour 1)

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install numpy pandas scipy networkx osmnx geopandas rasterio pyproj matplotlib requests
```

Folder layout:
```
floodsense/
  simulator.py
  build_area.py
  gen_dataset.py
  test_sanity.py
  iot_stream.py
  dem/                      <- DEM tile(s)
  data/
    hindmata/
      pumps_input.csv       <- you create this
      known_spots.csv       <- you create this
      real_storms/          <- Open-Meteo CSVs
      nodes.csv edges.csv pumps.csv      <- made by build_area.py
      dataset/              <- made by gen_dataset.py
    kurla/                  <- same files for the second area
```

---

## Step 1 - Download the height data (DEM)

Mumbai is around latitude 19 N, longitude 72.8 E. You need the tile "N19 E072".

Options (pick the easiest):
1. **Copernicus GLO-30 on AWS** (no login). File name looks like `Copernicus_DSM_COG_10_N19_00_E072_00_DEM.tif`. Search "Copernicus DEM AWS open data" to get the current link.
2. **OpenTopography** (free account): choose Copernicus GLO-30, draw a box around Mumbai, download GeoTIFF.
3. **FABDEM** (buildings and trees removed, better for cities, free for non-commercial use).
4. **SRTM** from USGS EarthExplorer.

Save as `dem/mumbai_dem.tif`.

**Important:** these DEMs have errors of about 2 to 4 metres, but Mumbai's height range in your area may be only 5 to 10 metres. Use heights for "which places are relatively lower" only.

---

## Step 2 - Write the pump list

Create `data/hindmata/pumps_input.csv`. Find the coordinates in Google Maps (right-click on the place, click the coordinates to copy).

```csv
name,lat,lon,cap_m3s
Hindmata mini pumping station,<lat>,<lon>,3.0
Gandhi Market mini pumping station,<lat>,<lon>,3.0
Britannia pumping station,<lat>,<lon>,6.0
```

If a station is outside your radius, the code links it to the nearest node, which is fine. Say "virtual pump at the nearest node" in your notes.

Also create `data/hindmata/known_spots.csv` for validation:
```csv
name,lat,lon
Hindmata junction,<lat>,<lon>
Gandhi Market,<lat>,<lon>
King's Circle,<lat>,<lon>
```
(Only keep spots that are inside your radius, or increase `DIST` a bit.)

---

## Step 3 - Build the graph: `build_area.py`

What it does: downloads roads, merges close intersections into one node, reads height from the DEM, computes area per node, links pumps, writes the 3 CSV files.

Run:
```bash
python build_area.py "Hindmata, Mumbai, India" data/hindmata dem/mumbai_dem.tif
```

```python
"""Build nodes.csv / edges.csv / pumps.csv for one area.
Usage: python build_area.py "Hindmata, Mumbai, India" data/hindmata dem/mumbai_dem.tif
Needs: pip install osmnx networkx geopandas rasterio scipy pyproj pandas numpy matplotlib
"""
import sys, os
import numpy as np, pandas as pd, networkx as nx, osmnx as ox, rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer

PLACE, OUT, DEM = sys.argv[1], sys.argv[2], sys.argv[3]
DIST = 1000          # radius in metres (about 3 sq km). Keep nodes <= 600.
TOL = 35             # merge intersections closer than this (m). Raise TOL for fewer nodes.
os.makedirs(OUT, exist_ok=True)

# 1. street graph -> simplified graph
center = ox.geocode(PLACE)                               # (lat, lon)
G = ox.graph_from_point(center, dist=DIST, network_type="drive")
G = ox.consolidate_intersections(ox.project_graph(G), tolerance=TOL, rebuild_graph=True, dead_ends=False)
G = ox.project_graph(G, to_crs="EPSG:4326")
U = nx.Graph(G.to_undirected())
U = U.subgraph(max(nx.connected_components(U), key=len)).copy()
U = nx.convert_node_labels_to_integers(U)               # ids 0..N-1
N = U.number_of_nodes()
lon = np.array([U.nodes[i]["x"] for i in range(N)]); lat = np.array([U.nodes[i]["y"] for i in range(N)])
print("nodes:", N, "edges:", U.number_of_edges())

# 2. elevation from the DEM (+ one smoothing pass because 30 m DEMs are noisy in cities)
with rasterio.open(DEM) as src:
    z = np.array([v[0] for v in src.sample(zip(lon, lat))], float)
z[(z < -50) | (z > 500)] = np.nan
z = np.where(np.isnan(z), np.nanmedian(z), z)
nbr_mean = np.array([np.mean([z[j] for j in U.neighbors(i)]) if U.degree(i) else z[i] for i in range(N)])
z_s = 0.5 * z + 0.5 * nbr_mean
print("elevation min/max/range (m):", z_s.min().round(2), z_s.max().round(2), (z_s.max() - z_s.min()).round(2))

# 3. area each node drains (nearest-node on a 25 m grid inside the circle)
tr = Transformer.from_crs(4326, 32643, always_xy=True)   # UTM 43N covers Mumbai
X, Y = tr.transform(lon, lat); cx, cy = tr.transform(center[1], center[0])
g = np.arange(-DIST, DIST, 25.0) + 12.5
gx, gy = np.meshgrid(cx + g, cy + g)
inside = (gx - cx) ** 2 + (gy - cy) ** 2 <= DIST ** 2
_, near = cKDTree(np.c_[X, Y]).query(np.c_[gx[inside], gy[inside]])
area = np.bincount(near, minlength=N) * 625.0
area = np.maximum(area, 625.0)

# 4. pumps: create <OUT>/pumps_input.csv with columns name,lat,lon,cap_m3s; each node is served by the nearest pump
pin = pd.read_csv(os.path.join(OUT, "pumps_input.csv"))
px, py = tr.transform(pin.lon.values, pin.lat.values)
_, pnode = cKDTree(np.c_[X, Y]).query(np.c_[px, py])
_, zone = cKDTree(np.c_[X[pnode], Y[pnode]]).query(np.c_[X, Y])
pumps = pd.DataFrame(dict(pump_id=range(len(pin)), node_id=pnode, name=pin.name, cap_m3s=pin.cap_m3s))
has_pump = np.isin(np.arange(N), pnode).astype(int)

# 5. save
pd.DataFrame(dict(id=range(N), lat=lat, lon=lon, elevation=z_s.round(3), area_m2=area,
                  runoff_coeff=0.85, pump_zone=zone, has_pump=has_pump)).to_csv(f"{OUT}/nodes.csv", index=False)
CAP = {"motorway": 3.0, "trunk": 3.0, "primary": 2.0, "secondary": 1.5, "tertiary": 1.0}
rows = []
for u, v, d in U.edges(data=True):
    hw = d.get("highway", "residential"); hw = hw[0] if isinstance(hw, list) else hw
    rows.append(dict(source=u, target=v, capacity_m3s=CAP.get(hw, 0.5), length_m=float(d.get("length", 50))))
pd.DataFrame(rows).to_csv(f"{OUT}/edges.csv", index=False)
pumps.to_csv(f"{OUT}/pumps.csv", index=False)
print("saved to", OUT)
```

**Check the output:**
- It prints `nodes: N`. You want **200 to 600**. Too many? Increase `TOL` (35 to 50) or reduce `DIST`. Too few? Reduce `TOL`.
- It prints the elevation range. If the range is under 1 m, the DEM is too flat. Try FABDEM, or smooth less.
- Quick plot to see the graph and heights:

```python
import pandas as pd, matplotlib.pyplot as plt
n = pd.read_csv("data/hindmata/nodes.csv"); e = pd.read_csv("data/hindmata/edges.csv")
for _, r in e.iterrows():
    a, b = n.loc[int(r.source)], n.loc[int(r.target)]
    plt.plot([a.lon, b.lon], [a.lat, b.lat], c="lightgray", lw=0.6)
sc = plt.scatter(n.lon, n.lat, c=n.elevation, s=12, cmap="viridis")
p = n[n.has_pump == 1]; plt.scatter(p.lon, p.lat, c="red", s=80, marker="^", label="pump")
plt.colorbar(sc, label="elevation (m)"); plt.legend(); plt.axis("equal"); plt.show()
```

---

## Step 4 - The simulator: `simulator.py`

Every 10 minutes: rain, local drains (tide-dependent), pumps (zone-based), then spreading between neighbours. It is vectorized with NumPy, so one storm takes about 6 ms.

```python
"""FloodSense simulator v1 (Person A). Vectorized NumPy, 10-minute steps."""
import numpy as np
import pandas as pd

DT = 600.0  # seconds per step (10 min)

class FloodSim:
    def __init__(self, z, A, runoff, pump_zone, pump_cap, src, dst, ecap,
                 drain_mm_hr=25.0, k=1.0, gate_level=1.0,
                 blocked_factor=0.1, back_mm_hr=10.0):
        self.z, self.A, self.c = z, A, runoff
        self.pump_zone = pump_zone          # (n,) pump index serving each node, -1 = none
        self.pump_cap = np.asarray(pump_cap, float)   # (P,) m3/s per pump station
        self.P = len(self.pump_cap)
        self.src, self.dst, self.ecap = src, dst, ecap   # ecap m3/s
        self.n = len(z)
        self.drain_mm_hr, self.k = drain_mm_hr, k
        self.gate_level = gate_level        # tide (m above MSL) where outfalls get blocked
        self.blocked, self.back = blocked_factor, back_mm_hr
        zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)
        self.sens = 0.3 + 0.7 * (1 - zn)    # low nodes depend more on tidal outfalls

    def auto_gate(self, tide):
        """Automatic tidal gates: open at low tide, closed at high tide."""
        return (tide <= self.gate_level).astype(float)

    def step(self, V, rain_mm, tide, gate_open, pump_on):
        n, A = self.n, self.A
        V = V + self.c * A * rain_mm / 1000.0                      # 1 rain -> runoff
        cap = A * (self.drain_mm_hr / 1000.0 / 3600.0) * DT        # 2 local drains
        high = tide > self.gate_level
        if gate_open and not high:
            f = 1.0
        elif gate_open and high:                                    # open at high tide: sea water comes in
            f = 0.0
            V = V + self.sens * A * (self.back / 1000.0 / 3600.0) * DT
        else:
            f = self.blocked                                        # closed: drains mostly blocked
        V = V - np.minimum(V, cap * f * (0.3 + 0.7 * self.sens) if f > 0 else 0.0)
        if self.P:                                                  # 3 pumps pull from their zone
            m = self.pump_zone >= 0
            zid = np.where(m, self.pump_zone, 0)
            zV = np.bincount(zid, weights=np.where(m, V, 0.0), minlength=self.P)
            rem = np.minimum(zV, self.pump_cap * DT * pump_on)
            frac = np.where(zV > 0, rem / np.maximum(zV, 1e-9), 0.0)
            V = V * np.where(m, 1.0 - frac[zid], 1.0)
        H = self.z + V / A                                          # 4 spread by head difference
        q = np.clip(self.k * (H[self.src] - H[self.dst]) * DT,
                    -self.ecap * DT, self.ecap * DT)
        srcn = np.where(q > 0, self.src, self.dst)
        demand = np.bincount(srcn, weights=np.abs(q), minlength=n)
        scale = np.minimum(1.0, V / np.maximum(demand, 1e-9))
        q = q * scale[srcn]
        V = V - np.bincount(self.src, weights=q, minlength=n) \
              + np.bincount(self.dst, weights=q, minlength=n)
        return np.maximum(V, 0.0)

    def simulate(self, rain, tide, gate_open=None, pump_on=None):
        """rain (T,) or (T,n) mm/step; tide (T,) m; gate_open (T,) 0/1; pump_on (T,P) 0/1.
        Returns depth_m (T,n)."""
        T = len(tide)
        rain = np.broadcast_to(np.asarray(rain, float).reshape(T, -1), (T, self.n))
        if gate_open is None: gate_open = self.auto_gate(tide)
        if pump_on is None:   pump_on = np.ones((T, self.P))
        V = np.zeros(self.n); out = np.zeros((T, self.n))
        for t in range(T):
            V = self.step(V, rain[t], tide[t], gate_open[t], pump_on[t])
            out[t] = V / self.A
        return out


def load_sim(area_dir, **kw):
    """Build a FloodSim from <area_dir>/nodes.csv, edges.csv, pumps.csv (see 03_DATA_CONTRACT.md)."""
    nodes = pd.read_csv(f"{area_dir}/nodes.csv")
    edges = pd.read_csv(f"{area_dir}/edges.csv")
    pumps = pd.read_csv(f"{area_dir}/pumps.csv")
    return FloodSim(z=nodes.elevation.values.astype(float),
                    A=nodes.area_m2.values.astype(float),
                    runoff=nodes.runoff_coeff.values.astype(float),
                    pump_zone=nodes.pump_zone.values.astype(int),
                    pump_cap=pumps.cap_m3s.values.astype(float),
                    src=edges.source.values.astype(int),
                    dst=edges.target.values.astype(int),
                    ecap=edges.capacity_m3s.values.astype(float), **kw), nodes, edges, pumps
```

### Tuning knobs (change these in `FloodSim(...)` or via `load_sim(area, k=..., gate_level=...)`)
| Knob | Default | If you see... | Change |
|---|---|---|---|
| `drain_mm_hr` | 25 | Floods too easily | Raise to 30-40 |
| `k` | 1.0 | Water does not reach the pumps | Raise to 10-50 |
| `gate_level` | 1.0 | Tide blocks drains too often | Raise to 1.5 |
| `blocked_factor` | 0.1 | Closed gates too harsh | Raise to 0.2 |
| edge `capacity_m3s` | 0.5-3 | Water stuck | Raise |
| pump `cap_m3s` | 3 | Pumps too weak or too strong | Change in `pumps_input.csv` |

**Goal for believable results:**
- Light rain (about 40 mm): little or no flooding
- Heavy storm (about 120 mm): some low spots flood
- Cloudburst (about 250 mm) + high tide at the rain peak: big flooding
- Pumps ON clearly better than OFF

---

## Step 5 - Sanity tests: `test_sanity.py`

Run this **before** generating any dataset. It uses a fake grid, so no data is needed. After you build your real area, also re-run the same checks using `load_sim("data/hindmata")`.

```python
"""Run: python test_sanity.py   (uses a small fake 20x20 grid, no internet needed)"""
import numpy as np
from simulator import FloodSim

rng = np.random.default_rng(0); side = 20; n = side * side
src, dst = [], []
for i in range(side):
    for j in range(side):
        if i + 1 < side: src.append(i * side + j); dst.append((i + 1) * side + j)
        if j + 1 < side: src.append(i * side + j); dst.append(i * side + j + 1)
src, dst = np.array(src), np.array(dst)
z = 2 + rng.normal(0, 0.3, n); A = np.full(n, 7500.0); c = np.full(n, 0.85)
pump_nodes = np.argsort(z)[:3]
xy = np.array([[i // side, i % side] for i in range(n)], float)
zone = np.linalg.norm(xy[:, None] - xy[pump_nodes][None], axis=2).argmin(1)
sim = FloodSim(z, A, c, zone, np.array([3., 3., 3.]), src, dst, np.full(len(src), 1.0), k=10)

T = 72; t = np.arange(T); tide = 1.8 * np.sin(2 * np.pi * t * 10 / 60 / 12.42 + 1.0)
def storm(total, dur=18, peak=30):
    r = np.exp(-0.5 * ((t - peak) / (dur / 2.5)) ** 2); r[t > peak + dur] = 0
    return r / r.sum() * total

assert sim.simulate(np.zeros(T), tide).max() == 0, "zero rain must give zero flood"
m = [sim.simulate(storm(x), tide).max() for x in [20, 60, 120, 250]]
assert all(m[i] <= m[i + 1] for i in range(3)), "more rain must not give less flood"
on = sim.simulate(storm(120), tide)
off = sim.simulate(storm(120), tide, pump_on=np.zeros((T, 3)))
assert on.sum() < off.sum(), "pumps ON must give less flooding than OFF"
assert on.min() >= 0 and not np.isnan(on).any(), "no negative or NaN water"
closed = sim.simulate(storm(120), tide, gate_open=np.zeros(T))
assert closed.max() > on.max(), "gates stuck closed must be worse than automatic gates"
print("ALL SANITY TESTS PASSED")
```

```bash
python test_sanity.py     # must print: ALL SANITY TESTS PASSED
```

---

## Step 6 - Real storms (Open-Meteo)

Find the heaviest rain days in recent monsoons and save them as CSV (this makes 2-3 files for the dataset).

```python
import requests, pandas as pd, os
os.makedirs("data/hindmata/real_storms", exist_ok=True)
r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params=dict(
    latitude=19.0, longitude=72.84, start_date="2015-06-01", end_date="2026-09-30",
    hourly="precipitation", timezone="Asia/Kolkata")).json()
df = pd.DataFrame(r["hourly"]); df["time"] = pd.to_datetime(df["time"]); df = df.rename(columns={"precipitation": "precipitation"})
df["date"] = df.time.dt.date
top = df.groupby("date").precipitation.sum().sort_values(ascending=False).head(5)
print(top)                                    # pick 2-3 dates
for d in top.index[:3]:
    day = df[(df.time >= pd.Timestamp(d) - pd.Timedelta(hours=12)) & (df.time < pd.Timestamp(d) + pd.Timedelta(hours=36))]
    day[["precipitation"]].to_csv(f"data/hindmata/real_storms/{d}.csv", index=False)
```
Notes:
- This data is reanalysis, so it **underestimates extreme rain** in Mumbai. Say so in the README.
- `gen_dataset.py` finds the peak hour and cuts a 16-hour window around it. It also makes a "worst-case" copy where high tide happens exactly at the rain peak.

---

## Step 7 - Generate the dataset: `gen_dataset.py`

Run a small one first (hand over to B at hour 5), then the full one.

```bash
python gen_dataset.py data/hindmata 50      # quick version for B
python gen_dataset.py data/hindmata 300     # full version
```

The 300-storm file is about 180 MB for 400 nodes. If it is too big, reduce the storms or save as float16.

```python
"""Generate the training dataset. Usage: python gen_dataset.py data/hindmata 300"""
import sys, os, glob, json
import numpy as np, pandas as pd
from simulator import load_sim, DT

area_dir = sys.argv[1]
S_SYN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
T = 96                                   # 96 steps x 10 min = 16 hours
rng = np.random.default_rng(42)
sim, nodes, edges, pumps = load_sim(area_dir)
N, P = sim.n, sim.P
zone = nodes.pump_zone.values.astype(int)

# spatial rain pattern: tiny east-west / north-south gradient (area is small, so keep it mild)
xn = (nodes.lon - nodes.lon.min()) / max(nodes.lon.max() - nodes.lon.min(), 1e-9)
yn = (nodes.lat - nodes.lat.min()) / max(nodes.lat.max() - nodes.lat.min(), 1e-9)

def spatial_factor():
    g = 1 + rng.uniform(-0.12, 0.12) * (xn - 0.5) * 2 + rng.uniform(-0.12, 0.12) * (yn - 0.5) * 2
    return (g * (1 + rng.normal(0, 0.03, N))).values

def synthetic_rain():
    total = rng.uniform(30, 300)                      # mm
    dur = rng.integers(6, 37)                         # steps (1-6 h)
    peak = rng.integers(10, 50)
    t = np.arange(T)
    shape = np.exp(-0.5 * ((t - peak) / (dur / 2.5)) ** 2)
    shape[t > peak + dur] = 0; shape[t < peak - dur] = 0
    rain = shape / shape.sum() * total
    rain *= min(1.0, 25.0 / rain.max())               # cap at 25 mm per 10 min (150 mm/h)
    return rain, int(peak)

def real_rain(path):
    h = pd.read_csv(path)["precipitation"].values     # hourly mm
    k = int(np.argmax(h)); a = max(0, min(len(h) - 16, k - 5))
    rain = np.repeat(h[a:a + 16] / 6.0, 6)            # 16 h -> 96 steps of 10 min
    return rain, int(np.argmax(rain))

def tide_series(amp, phase):
    t = np.arange(T)
    return amp * np.sin(2 * np.pi * (t * 10 / 60) / 12.42 + phase)

def pump_schedule(mode):
    if mode == "all_on": return np.ones((T, P))
    blocks = (rng.random((T // 6 + 1, P)) < 0.85).astype(float)   # 1-hour blocks
    return np.repeat(blocks, 6, axis=0)[:T]

jobs = []                                              # (rain, peak, source, tide_mode)
for _ in range(S_SYN):
    r, p = synthetic_rain(); jobs.append((r, p, "synthetic", "random"))
for f in sorted(glob.glob(f"{area_dir}/real_storms/*.csv")):
    r, p = real_rain(f)
    jobs.append((r, p, "real:" + os.path.basename(f), "random"))
    jobs.append((r, p, "real:" + os.path.basename(f), "worst"))   # high tide exactly at rain peak

S = len(jobs)
X = np.zeros((S, T, N, 4), np.float32)                 # [rain_mm, tide_m, depth_m, pump_on]
GATE = np.zeros((S, T), np.float32); PUMP = np.zeros((S, T, P), np.float32)
meta = []
for i, (rain, peak, src_name, tide_mode) in enumerate(jobs):
    amp = rng.uniform(1.5, 2.2)
    phase = (np.pi / 2 - 2 * np.pi * (peak * 10 / 60) / 12.42) if tide_mode == "worst" else rng.uniform(0, 2 * np.pi)
    tide = tide_series(amp, phase)
    pmode = "all_on" if i % 3 == 0 else "random"
    sched = pump_schedule(pmode)
    gate = sim.auto_gate(tide)
    rain_node = rain[:, None] * spatial_factor()[None, :]
    depth = sim.simulate(rain_node, tide, gate_open=gate, pump_on=sched)
    node_pump = np.where(zone[None, :] >= 0, sched[:, np.clip(zone, 0, None)], 0.0)
    X[i, :, :, 0] = rain_node; X[i, :, :, 1] = tide[:, None]
    X[i, :, :, 2] = depth;     X[i, :, :, 3] = node_pump
    GATE[i] = gate; PUMP[i] = sched
    meta.append(dict(storm_id=i, source=src_name, total_mm=rain.sum(), peak_mm_per_hr=rain.max() * 6,
                     tide_amp=amp, tide_mode=tide_mode, pump_mode=pmode,
                     max_depth_m=depth.max(), flooded_nodes=int((depth.max(0) > 0.10).sum())))

meta = pd.DataFrame(meta)
order = rng.permutation(S); split = np.empty(S, object)
split[order[:int(.7 * S)]] = "train"; split[order[int(.7 * S):int(.85 * S)]] = "val"; split[order[int(.85 * S):]] = "test"
meta["split"] = split

out = f"{area_dir}/dataset"; os.makedirs(out, exist_ok=True)
np.save(f"{out}/X.npy", X); np.save(f"{out}/gate_open.npy", GATE); np.save(f"{out}/pump_sched.npy", PUMP)
meta.to_csv(f"{out}/meta.csv", index=False)

test = meta[meta.split == "test"]                       # scenario presets for the dashboard
pick = lambda mm: int(test.iloc[(test.total_mm - mm).abs().argsort().iloc[0]].storm_id)
json.dump({"light_rain": pick(40), "heavy_storm": pick(120), "extreme_cloudburst": pick(250)},
          open(f"{out}/scenarios.json", "w"))
json.dump({"T": T, "dt_seconds": DT, "features": ["rain_mm", "tide_m", "depth_m", "pump_on"],
           "n_nodes": N, "n_pumps": P, "flood_threshold_m": 0.10},
          open(f"{out}/config.json", "w"), indent=1)
print("saved", X.shape, "flooded storms:", int((meta.flooded_nodes > 0).sum()), "/", S)
print(meta[["total_mm", "max_depth_m", "flooded_nodes"]].describe().round(2))
```

What it makes: `X.npy`, `gate_open.npy`, `pump_sched.npy`, `meta.csv`, `scenarios.json`, `config.json` (see the data contract).

**Check the printed summary.** A good dataset has some storms with 0 flooded nodes (small rain) and some with many. If every storm floods badly, lower the rain range in `synthetic_rain()` (for example, `rng.uniform(15, 220)`) or raise `drain_mm_hr`. Better still, draw totals with `rng.lognormal(4.2, 0.6)` so small storms are more common.

---

## Step 8 - Known-spot check (your validation slide)

```python
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
area = "data/hindmata"
nodes = pd.read_csv(f"{area}/nodes.csv"); meta = pd.read_csv(f"{area}/dataset/meta.csv")
X = np.load(f"{area}/dataset/X.npy", mmap_mode="r")
spots = pd.read_csv(f"{area}/known_spots.csv")
heavy = meta[(meta.total_mm > 100) & (meta.split == "test")].storm_id.values
tree = cKDTree(np.c_[nodes.lat, nodes.lon])
avg = np.mean([(X[k, :, :, 2].max(0) > 0.10).mean() for k in heavy])
print(f"average node floods in {avg*100:.1f}% of heavy storms")
for _, s in spots.iterrows():
    _, i = tree.query([s.lat, s.lon])
    hit = np.mean([X[k, :, i, 2].max() > 0.10 for k in heavy])
    print(f"{s['name']:25s} node {i:4d}  floods in {hit*100:5.1f}% of heavy storms")
```
**Good result:** known spots flood more often than the average node. If not, lower those nodes a little (document it as calibration) or move the pump zones, then re-run.

**Tide story chart (great for the demo):** pick one real storm and compare the two copies in `meta.csv` (`tide_mode` = random vs worst). The worst-case copy should have a higher `max_depth_m` and more `flooded_nodes`.

---

## Step 9 - IoT sensor stream: `iot_stream.py`

Pretends to be a river gauge. Person C connects the dashboard to it.

```python
import json, time, sys, numpy as np
area, storm_id, node = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])   # node = a low node near the nallah
X = np.load(f"{area}/dataset/X.npy", mmap_mode="r")
rng = np.random.default_rng(0)
for t in range(X.shape[1]):
    msg = dict(t=t, sensor_id=f"gauge_{node}",
               water_level_m=float(X[storm_id, t, node, 2] + rng.normal(0, 0.005)),
               rain_mm=float(X[storm_id, t, node, 0]), tide_m=float(X[storm_id, t, node, 1]))
    print(json.dumps(msg), flush=True)          # or: requests.post("http://localhost:8000/sensor", json=msg)
    time.sleep(2)
```
```bash
python iot_stream.py data/hindmata 17 231
```

---

## Step 10 - Second area (scalability)

Same scripts, new place. Takes about 15 minutes if the first area works.

```bash
mkdir -p data/kurla
# create data/kurla/pumps_input.csv and known_spots.csv (look at the Mithi river outfalls / pump locations)
python build_area.py "Kurla, Mumbai, India" data/kurla dem/mumbai_dem.tif
python gen_dataset.py data/kurla 100
```
For the pitch: "The same pipeline runs on a second area by changing one name." Person C only needs to change the folder name in the dashboard.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ox.geocode` fails or finds the wrong place | Use coordinates: `center = (lat, lon)` in `build_area.py` |
| `consolidate_intersections` error | Update osmnx (`pip install -U osmnx`); function arguments changed between versions |
| `KeyError: 'x'` on nodes | After projecting back to EPSG:4326, nodes should have `x` (lon) and `y` (lat). Print `U.nodes(data=True)` to check names. |
| DEM `sample` returns nodata or huge numbers | The tile does not cover your area, or the CRS is not WGS84. Check with `rasterio.open(path).crs` and `.bounds`. |
| Elevation range under 1 m | Use FABDEM, or reduce smoothing (`0.5/0.5` to `0.8/0.2`) |
| Nothing floods | Lower `drain_mm_hr` or raise rain; check `pump_zone` is not all -1 |
| Everything floods | Raise `drain_mm_hr` and `k`, lower the rain totals |
| Pumps do nothing | Check `pumps.csv` `cap_m3s` and that nodes have `pump_zone >= 0` |
| Dataset file too big | Fewer storms or `X.astype(np.float16)` when saving (tell B) |
| Slow | Make sure you did not add Python loops over nodes; only the time loop is allowed |
