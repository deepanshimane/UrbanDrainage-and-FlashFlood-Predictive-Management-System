"""Makes a FAKE but correctly-shaped dataset so B and C can start on hour 0.
Usage: python -m src.sim.make_fake_data data/sample 60
Person A later produces the REAL dataset in the same format (see CONTRACT.md)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from src.sim.simulator import simulate

def make_graph(rng, n=200):
    x, y = rng.random(n), rng.random(n)
    elev = 12 * (1 - y) + rng.normal(0, 1.2, n)          # high at y=0, low at y=1 (river side)
    nodes = pd.DataFrame({
        "node_id": np.arange(n), "lat": 19.00 + y * 0.018, "lon": 72.83 + x * 0.018,
        "elevation_m": elev.round(2), "capacity_m3": rng.uniform(200, 600, n).round(1),
        "catchment_m2": rng.uniform(8000, 30000, n).round(0), "runoff_coeff": rng.uniform(0.3, 0.9, n).round(2),
        "has_pump": 0, "pump_rate_m3s": 0.0, "is_outlet": 0})
    low = np.argsort(elev)[: int(n * 0.3)]
    pumps = rng.choice(low, 12, replace=False)
    nodes.loc[pumps, "has_pump"] = 1; nodes.loc[pumps, "pump_rate_m3s"] = 0.3
    nodes.loc[np.argsort(elev)[:6], "is_outlet"] = 1
    pts = np.c_[x, y]; _, nb = cKDTree(pts).query(pts, k=4)
    pairs = set()
    for i in range(n):
        for j in nb[i, 1:]:
            a, b = (i, int(j)) if elev[i] >= elev[j] else (int(j), i)   # flow goes downhill
            pairs.add((a, b))
    e = np.array(sorted(pairs))
    edges = pd.DataFrame({"edge_id": np.arange(len(e)), "source": e[:, 0], "target": e[:, 1],
                          "capacity_m3s": rng.uniform(0.3, 1.2, len(e)).round(3),
                          "length_m": rng.uniform(50, 250, len(e)).round(0),
                          "is_gate": (rng.random(len(e)) < 0.08).astype(int)})
    sinks = sorted(set(range(n)) - set(edges.source))          # nodes water cannot leave -> drain to river
    nodes.loc[sinks, "is_outlet"] = 1
    return nodes, edges

def make_storm(nodes, rng, T):
    x = (nodes.lon - nodes.lon.min()) / (nodes.lon.max() - nodes.lon.min())
    y = (nodes.lat - nodes.lat.min()) / (nodes.lat.max() - nodes.lat.min())
    peak, sigma = rng.uniform(30, 110), rng.uniform(0.25, 0.6)
    start, dur = rng.integers(4, 20), rng.integers(8, 20)
    c0, vel = rng.random(2), rng.uniform(-0.02, 0.02, 2)
    t = np.arange(T)
    env = peak * np.exp(-0.5 * ((t - (start + dur / 2)) / (dur / 2.5)) ** 2)
    cx, cy = c0[0] + vel[0] * t, c0[1] + vel[1] * t
    d2 = (x.to_numpy()[None, :] - cx[:, None]) ** 2 + (y.to_numpy()[None, :] - cy[:, None]) ** 2
    rain = (env[:, None] * np.exp(-d2 / (2 * sigma ** 2))).astype(np.float32)
    river = np.zeros(T, np.float32)
    for k in range(1, T):
        river[k] = 0.95 * river[k - 1] + 0.01 * rain[k].mean()
    return rain, np.clip(river, 0, 6)

def main(out, S=60, T=72, seed=0):
    rng = np.random.default_rng(seed); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    nodes, edges = make_graph(rng); N, E = len(nodes), len(edges)
    nodes.to_csv(out / "nodes.csv", index=False); edges.to_csv(out / "edges.csv", index=False)
    R = np.zeros((S, T, N), np.float32); RV = np.zeros((S, T), np.float32)
    P = np.zeros((S, T, N), np.float32); G = np.ones((S, T, E), np.float32)
    F = np.zeros((S, T, N), np.float32); D = np.zeros((S, T, N), np.float32)
    has_pump = nodes.has_pump.to_numpy()
    for s in range(S):
        R[s], RV[s] = make_storm(nodes, rng, T)
        on, off = rng.integers(0, 40), rng.integers(30, T)
        P[s, on:off] = has_pump[None, :] * (rng.random(N) < 0.7)[None, :]
        G[s] = (rng.random((1, E)) < 0.9).astype(np.float32).repeat(T, 0)
        r = simulate(nodes, edges, R[s], RV[s], P[s], G[s]); F[s], D[s] = r["fill"], r["depth"]
    for name, arr in dict(rain=R, river=RV, pump_on=P, gate_open=G, fill=F, depth=D).items():
        np.save(out / f"{name}.npy", arr)
    print(f"saved to {out}: storms={S} steps={T} nodes={N} edges={E}")
    print(f"share of node-steps flooded (depth>0.10 m): {(D > 0.10).mean():.3f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample", int(sys.argv[2]) if len(sys.argv) > 2 else 60)
