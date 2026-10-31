"""STUB simulator (Person A will replace/improve this file).
Keep the function name, inputs and outputs EXACTLY the same so B and C never break.
Crude physics: rain -> runoff -> water moves downhill along edges -> outlets drain to the
river (slower when river is high) -> pumps remove water -> extra water above capacity = flood.
"""
import numpy as np

DT = 900.0            # seconds per step
RIVER_MAX = 6.0       # metres; outlets stop draining when river reaches this
OUTLET_RATE = 4.0     # m3/s per outlet node when river is low

def simulate(nodes, edges, rain, river, pump_on, gate_open, dt=DT):
    """
    rain      [T,N]  mm/h          river [T]  metres
    pump_on   [T,N]  0/1           gate_open [T,E] 0/1 (only matters where edges.is_gate==1)
    returns dict: fill [T,N] (stored/capacity), depth [T,N] (flood depth m), flood_cost (float)
    flood_cost = sum of flood depth over all nodes and time, in metre-hours (lower = better)
    """
    T, N = rain.shape
    cap = nodes.capacity_m3.to_numpy(float)
    area = nodes.catchment_m2.to_numpy(float)
    runoff = nodes.runoff_coeff.to_numpy(float)
    elev = nodes.elevation_m.to_numpy(float)
    prate = (nodes.pump_rate_m3s * nodes.has_pump).to_numpy(float)
    outlet = nodes.is_outlet.to_numpy(float)
    src = edges.source.to_numpy(int); dst = edges.target.to_numpy(int)
    ecap = edges.capacity_m3s.to_numpy(float)
    is_gate = edges.is_gate.to_numpy(int) == 1
    pond = 0.02 * area                      # street area where water can pond
    storage = np.zeros(N)
    fill = np.zeros((T, N), np.float32); depth = np.zeros((T, N), np.float32)
    for t in range(T):
        storage += runoff * area * (rain[t] / 1000.0 / 3600.0) * dt
        head = elev + storage / cap
        g = np.where(is_gate, gate_open[t], 1.0)
        vol = ecap * g * np.clip(head[src] - head[dst] + 0.2, 0, 1) * dt
        out_sum = np.zeros(N); np.add.at(out_sum, src, vol)
        scale = np.minimum(1.0, storage / np.maximum(out_sum, 1e-9))
        vol = vol * scale[src]
        np.add.at(storage, src, -vol); np.add.at(storage, dst, vol)
        factor = max(0.0, 1.0 - float(river[t]) / RIVER_MAX)
        storage -= np.minimum(storage, outlet * OUTLET_RATE * factor * dt)
        storage -= np.minimum(storage, prate * pump_on[t] * dt)
        fill[t] = storage / cap
        depth[t] = np.minimum(3.0, np.maximum(0.0, storage - cap) / pond)
    return {"fill": fill, "depth": depth, "flood_cost": float(depth.sum() * dt / 3600.0)}
