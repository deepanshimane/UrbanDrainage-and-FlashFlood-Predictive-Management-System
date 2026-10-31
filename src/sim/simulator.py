"""Simulation interface supporting both FloodSim (real Mumbai area) and fallback schema."""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from simulator import FloodSim, load_sim, DT

def simulate(nodes, edges, rain, river, pump_on=None, gate_open=None, dt=DT):
    """Unified simulation entry point.
    Supports:
    - Real area schema (nodes: elevation, area_m2, pump_zone; edges: capacity_m3s)
    - Sample schema (nodes: elevation_m, capacity_m3, catchment_m2; edges: capacity_m3s)
    """
    T, N = rain.shape
    # If nodes has real area schema:
    if "area_m2" in nodes.columns and "pump_zone" in nodes.columns:
        pumps_path = Path("data/hindmata/pumps.csv")
        if pumps_path.exists():
            pumps_df = pd.read_csv(pumps_path)
            pump_cap = pumps_df.cap_m3s.values.astype(float)
        else:
            pump_cap = np.array([4.0], dtype=float)
        
        sim = FloodSim(
            z=nodes.elevation.values.astype(float),
            A=nodes.area_m2.values.astype(float),
            runoff=nodes.runoff_coeff.values.astype(float),
            pump_zone=nodes.pump_zone.values.astype(int),
            pump_cap=pump_cap,
            src=edges.source.values.astype(int),
            dst=edges.target.values.astype(int),
            ecap=edges.capacity_m3s.values.astype(float),
        )
        
        P = sim.P
        if pump_on is None:
            pump_mat = np.ones((T, P), dtype=float)
        elif pump_on.ndim == 2 and pump_on.shape[1] == N:
            pump_mat = np.zeros((T, P), dtype=float)
            for p_idx in range(P):
                zone_nodes = np.where(nodes.pump_zone.values == p_idx)[0]
                if len(zone_nodes) > 0:
                    pump_mat[:, p_idx] = np.mean(pump_on[:, zone_nodes], axis=1)
                else:
                    pump_mat[:, p_idx] = 1.0
        else:
            pump_mat = pump_on

        if gate_open is not None and gate_open.ndim == 2:
            gate_arr = np.mean(gate_open, axis=1)
        else:
            gate_arr = gate_open

        depth = sim.simulate(rain=rain, tide=river, gate_open=gate_arr, pump_on=pump_mat)
        fill = depth / 0.10
        flood_cost = float(depth.sum() * dt / 3600.0)
        return {"fill": fill, "depth": depth, "flood_cost": flood_cost}

    # Fallback to starter physics model for sample schema
    cap = nodes.capacity_m3.to_numpy(float)
    area = nodes.catchment_m2.to_numpy(float)
    runoff = nodes.runoff_coeff.to_numpy(float)
    elev = nodes.elevation_m.to_numpy(float)
    prate = (nodes.pump_rate_m3s * nodes.has_pump).to_numpy(float)
    outlet = nodes.is_outlet.to_numpy(float)
    src = edges.source.to_numpy(int)
    dst = edges.target.to_numpy(int)
    ecap = edges.capacity_m3s.to_numpy(float)
    is_gate = edges.is_gate.to_numpy(int) == 1
    pond = 0.02 * area
    storage = np.zeros(N)
    fill = np.zeros((T, N), np.float32)
    depth = np.zeros((T, N), np.float32)
    
    RIVER_MAX = 6.0
    OUTLET_RATE = 4.0
    
    for t in range(T):
        storage += runoff * area * (rain[t] / 1000.0 / 3600.0) * dt
        head = elev + storage / cap
        g = np.where(is_gate, gate_open[t], 1.0) if gate_open is not None else np.ones(len(edges))
        vol = ecap * g * np.clip(head[src] - head[dst] + 0.2, 0, 1) * dt
        out_sum = np.zeros(N)
        np.add.at(out_sum, src, vol)
        scale = np.minimum(1.0, storage / np.maximum(out_sum, 1e-9))
        vol = vol * scale[src]
        np.add.at(storage, src, -vol)
        np.add.at(storage, dst, vol)
        factor = max(0.0, 1.0 - float(river[t]) / RIVER_MAX)
        storage -= np.minimum(storage, outlet * OUTLET_RATE * factor * dt)
        if pump_on is not None:
            storage -= np.minimum(storage, prate * pump_on[t] * dt)
        fill[t] = storage / cap
        depth[t] = np.minimum(3.0, np.maximum(0.0, storage - cap) / pond)
        
    return {"fill": fill, "depth": depth, "flood_cost": float(depth.sum() * dt / 3600.0)}
