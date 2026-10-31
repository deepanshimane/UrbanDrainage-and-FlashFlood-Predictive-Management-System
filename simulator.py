"""FloodSense simulator v1 (Person A). Vectorized NumPy, 10-minute steps.
Predicts street-level flood depth based on rain, tide, drainage capacity, and pump operations.
"""
import numpy as np
import pandas as pd

DT = 600.0  # seconds per step (10 min)

class FloodSim:
    def __init__(self, z, A, runoff, pump_zone, pump_cap, src, dst, ecap,
                 drain_mm_hr=25.0, k=1.0, gate_level=1.0,
                 blocked_factor=0.1, back_mm_hr=10.0):
        self.z = np.asarray(z, float)
        self.A = np.asarray(A, float)
        self.c = np.asarray(runoff, float)
        self.pump_zone = np.asarray(pump_zone, int)         # (n,) pump index serving each node, -1 = none
        self.pump_cap = np.asarray(pump_cap, float)         # (P,) m3/s per pump station
        self.P = len(self.pump_cap)
        self.src = np.asarray(src, int)
        self.dst = np.asarray(dst, int)
        self.ecap = np.asarray(ecap, float)                 # ecap m3/s
        self.n = len(z)
        self.drain_mm_hr = float(drain_mm_hr)
        self.k = float(k)
        self.gate_level = float(gate_level)                 # tide (m above MSL) where outfalls get blocked
        self.blocked = float(blocked_factor)
        self.back = float(back_mm_hr)
        zn = (self.z - self.z.min()) / max(self.z.max() - self.z.min(), 1e-6)
        self.sens = 0.3 + 0.7 * (1.0 - zn)                  # low nodes depend more on tidal outfalls

    def auto_gate(self, tide):
        """Automatic tidal gates: open at low tide, closed at high tide."""
        return (np.asarray(tide, float) <= self.gate_level).astype(float)

    def step(self, V, rain_mm, tide, gate_open, pump_on):
        n, A = self.n, self.A
        
        # 1. Rain -> runoff volume addition
        V = V + self.c * A * (rain_mm / 1000.0)
        
        # 2. Local drain gravity discharge (tide dependent)
        cap = A * (self.drain_mm_hr / 1000.0 / 3600.0) * DT
        is_high = bool(tide > self.gate_level)
        is_gate_open = bool(gate_open > 0.5)
        
        if is_gate_open and not is_high:
            f = 1.0
        elif is_gate_open and is_high:
            # Sluice gate open at high tide: seawater backflow enters low-lying nodes
            f = 0.0
            V = V + self.sens * A * (self.back / 1000.0 / 3600.0) * DT
        else:
            # Gate closed at high tide: drain discharge severely blocked
            f = self.blocked
            
        if f > 0.0:
            drain_rate = cap * f * (0.3 + 0.7 * self.sens)
            V = V - np.minimum(V, drain_rate)
            
        # 3. Pumps pull water from their allocated zone
        if self.P > 0:
            pump_on_arr = np.broadcast_to(np.asarray(pump_on, float).ravel(), (self.P,))
            m = self.pump_zone >= 0
            zid = np.where(m, self.pump_zone, 0)
            zV = np.bincount(zid, weights=np.where(m, V, 0.0), minlength=self.P)
            rem = np.minimum(zV, self.pump_cap * DT * pump_on_arr)
            frac = np.where(zV > 0, rem / np.maximum(zV, 1e-9), 0.0)
            V = V * np.where(m, 1.0 - frac[zid], 1.0)
            
        # 4. Spreading by hydraulic head difference (elevation + water depth)
        H = self.z + V / A
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
        rain_arr = np.asarray(rain, float)
        if rain_arr.ndim == 1:
            rain_node = np.repeat(rain_arr[:, None], self.n, axis=1)
        elif rain_arr.shape == (T, self.n):
            rain_node = rain_arr
        else:
            rain_node = np.broadcast_to(rain_arr.reshape(T, -1), (T, self.n))
            
        if gate_open is None:
            gate_open = self.auto_gate(tide)
        gate_arr = np.asarray(gate_open, float)
        
        if pump_on is None:
            pump_on_mat = np.ones((T, self.P), float)
        else:
            pump_arr = np.asarray(pump_on, float)
            if pump_arr.ndim == 1 and len(pump_arr) == self.P:
                pump_on_mat = np.repeat(pump_arr[None, :], T, axis=0)
            elif pump_arr.ndim == 2:
                pump_on_mat = pump_arr
            else:
                pump_on_mat = np.broadcast_to(pump_arr, (T, self.P))
                
        V = np.zeros(self.n, float)
        out = np.zeros((T, self.n), float)
        for t in range(T):
            V = self.step(V, rain_node[t], tide[t], gate_arr[t], pump_on_mat[t])
            out[t] = V / self.A
            
        return out


def load_sim(area_dir, **kw):
    """Build a FloodSim from <area_dir>/nodes.csv, edges.csv, pumps.csv (see 03_DATA_CONTRACT.md)."""
    nodes = pd.read_csv(f"{area_dir}/nodes.csv")
    edges = pd.read_csv(f"{area_dir}/edges.csv")
    pumps = pd.read_csv(f"{area_dir}/pumps.csv")
    sim = FloodSim(z=nodes.elevation.values.astype(float),
                   A=nodes.area_m2.values.astype(float),
                   runoff=nodes.runoff_coeff.values.astype(float),
                   pump_zone=nodes.pump_zone.values.astype(int),
                   pump_cap=pumps.cap_m3s.values.astype(float),
                   src=edges.source.values.astype(int),
                   dst=edges.target.values.astype(int),
                   ecap=edges.capacity_m3s.values.astype(float), **kw)
    return sim, nodes, edges, pumps

