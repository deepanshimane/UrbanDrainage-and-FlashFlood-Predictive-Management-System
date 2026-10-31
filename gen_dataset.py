"""Generate the training dataset for FloodSense.
Usage: python gen_dataset.py data/hindmata 300
Outputs:
  - dataset/X.npy          [S, 96, N, 4] float32
  - dataset/gate_open.npy  [S, 96] float32
  - dataset/pump_sched.npy [S, 96, P] float32
  - dataset/meta.csv       metadata with 70/15/15 train/val/test split
  - dataset/scenarios.json presets for light, heavy, cloudburst
  - dataset/config.json    configuration specifications
"""
import sys, os, glob, json
import numpy as np, pandas as pd
from simulator import load_sim, DT

area_dir = sys.argv[1] if len(sys.argv) > 1 else "data/hindmata"
S_SYN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
T = 96                                   # 96 steps x 10 min = 16 hours
rng = np.random.default_rng(42)

sim, nodes, edges, pumps = load_sim(area_dir)
N, P = sim.n, sim.P
zone = nodes.pump_zone.values.astype(int)

# spatial rain pattern: tiny east-west / north-south gradient
xn = ((nodes.lon - nodes.lon.min()) / max(nodes.lon.max() - nodes.lon.min(), 1e-9)).to_numpy()
yn = ((nodes.lat - nodes.lat.min()) / max(nodes.lat.max() - nodes.lat.min(), 1e-9)).to_numpy()

def spatial_factor():
    g = 1.0 + rng.uniform(-0.12, 0.12) * (xn - 0.5) * 2.0 + rng.uniform(-0.12, 0.12) * (yn - 0.5) * 2.0
    return g * (1.0 + rng.normal(0, 0.03, N))

def synthetic_rain():
    # lognormal-leaning distribution for realism (many small/moderate, some extreme cloudbursts)
    total = float(rng.uniform(25, 280))               # mm
    dur = int(rng.integers(6, 37))                    # steps (1-6 h)
    peak = int(rng.integers(10, 50))
    t = np.arange(T)
    shape = np.exp(-0.5 * ((t - peak) / (dur / 2.5)) ** 2)
    shape[t > peak + dur] = 0
    shape[t < peak - dur] = 0
    if shape.sum() > 0:
        rain = shape / shape.sum() * total
    else:
        rain = np.zeros(T)
    rain *= min(1.0, 25.0 / max(rain.max(), 1e-9))    # cap at 25 mm per 10 min (150 mm/h)
    return rain, int(peak)

def real_rain(path):
    h = pd.read_csv(path)["precipitation"].values     # hourly mm
    k = int(np.argmax(h))
    a = max(0, min(len(h) - 16, k - 5))
    rain = np.repeat(h[a:a + 16] / 6.0, 6)            # 16 h -> 96 steps of 10 min
    return rain, int(np.argmax(rain))

def tide_series(amp, phase):
    t = np.arange(T)
    return amp * np.sin(2 * np.pi * (t * 10 / 60) / 12.42 + phase)

def pump_schedule(mode):
    if mode == "all_on":
        return np.ones((T, P), float)
    blocks = (rng.random((T // 6 + 1, P)) < 0.85).astype(float)   # 1-hour blocks
    return np.repeat(blocks, 6, axis=0)[:T]

print(f"Building dataset for {area_dir} with {S_SYN} synthetic storms...", flush=True)
jobs = []                                              # (rain, peak, source, tide_mode)
for _ in range(S_SYN):
    r, p = synthetic_rain()
    jobs.append((r, p, "synthetic", "random"))

real_files = sorted(glob.glob(f"{area_dir}/real_storms/*.csv"))
print(f"Found {len(real_files)} real storm files.", flush=True)
for f in real_files:
    r, p = real_rain(f)
    jobs.append((r, p, "real:" + os.path.basename(f), "random"))
    jobs.append((r, p, "real:" + os.path.basename(f), "worst"))   # high tide exactly at rain peak

S = len(jobs)
print(f"Total storms to simulate: {S} (Synthetic: {S_SYN}, Real: {len(jobs) - S_SYN})", flush=True)

X = np.zeros((S, T, N, 4), np.float32)                 # [rain_mm, tide_m, depth_m, pump_on]
GATE = np.zeros((S, T), np.float32)
PUMP = np.zeros((S, T, P), np.float32)
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
    
    X[i, :, :, 0] = rain_node
    X[i, :, :, 1] = tide[:, None]
    X[i, :, :, 2] = depth
    X[i, :, :, 3] = node_pump
    GATE[i] = gate
    PUMP[i] = sched
    
    meta.append(dict(
        storm_id=i,
        source=src_name,
        total_mm=float(rain.sum()),
        peak_mm_per_hr=float(rain.max() * 6),
        tide_amp=float(amp),
        tide_mode=tide_mode,
        pump_mode=pmode,
        max_depth_m=float(depth.max()),
        flooded_nodes=int((depth.max(axis=0) > 0.10).sum())
    ))

meta = pd.DataFrame(meta)
order = rng.permutation(S)
split = np.empty(S, object)
split[order[:int(0.70 * S)]] = "train"
split[order[int(0.70 * S):int(0.85 * S)]] = "val"
split[order[int(0.85 * S):]] = "test"
meta["split"] = split

out = f"{area_dir}/dataset"
os.makedirs(out, exist_ok=True)
np.save(f"{out}/X.npy", X)
np.save(f"{out}/gate_open.npy", GATE)
np.save(f"{out}/pump_sched.npy", PUMP)
meta.to_csv(f"{out}/meta.csv", index=False)

test = meta[meta.split == "test"]                       # scenario presets for the dashboard
pick = lambda mm: int(test.iloc[(test.total_mm - mm).abs().argsort().iloc[0]].storm_id)
json.dump({
    "light_rain": pick(40),
    "heavy_storm": pick(120),
    "extreme_cloudburst": pick(250)
}, open(f"{out}/scenarios.json", "w"), indent=2)

json.dump({
    "T": T,
    "dt_seconds": DT,
    "features": ["rain_mm", "tide_m", "depth_m", "pump_on"],
    "n_nodes": N,
    "n_pumps": P,
    "flood_threshold_m": 0.10
}, open(f"{out}/config.json", "w"), indent=2)

print(f"\nSaved dataset to {out}:", flush=True)
print(f" - X.npy shape: {X.shape} ({X.nbytes / (1024*1024):.1f} MB)", flush=True)
print(f" - gate_open.npy shape: {GATE.shape}", flush=True)
print(f" - pump_sched.npy shape: {PUMP.shape}", flush=True)
print(f" - meta.csv: {len(meta)} rows (train: {(meta.split=='train').sum()}, val: {(meta.split=='val').sum()}, test: {(meta.split=='test').sum()})", flush=True)
print(f" - Flooded storms (depth > 0.10m): {int((meta.flooded_nodes > 0).sum())} / {S}", flush=True)
print("\nDataset Summary Statistics:")
print(meta[["total_mm", "max_depth_m", "flooded_nodes"]].describe().round(2))

