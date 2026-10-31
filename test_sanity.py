"""Run: python test_sanity.py
Performs physics sanity tests on both a synthetic 20x20 grid and the real Hindmata area.
"""
import os
import numpy as np
from simulator import FloodSim, load_sim

# -------------------------------------------------------------
# Part 1: Fake 20x20 Grid Tests
# -------------------------------------------------------------
print("Running sanity checks on 20x20 synthetic grid...")
rng = np.random.default_rng(0)
side = 20
n = side * side
src, dst = [], []
for i in range(side):
    for j in range(side):
        if i + 1 < side:
            src.append(i * side + j)
            dst.append((i + 1) * side + j)
        if j + 1 < side:
            src.append(i * side + j)
            dst.append(i * side + j + 1)
src, dst = np.array(src), np.array(dst)
z = 2 + rng.normal(0, 0.3, n)
A = np.full(n, 7500.0)
c = np.full(n, 0.85)
pump_nodes = np.argsort(z)[:3]
xy = np.array([[i // side, i % side] for i in range(n)], float)
zone = np.linalg.norm(xy[:, None] - xy[pump_nodes][None], axis=2).argmin(1)
sim = FloodSim(z, A, c, zone, np.array([3.0, 3.0, 3.0]), src, dst, np.full(len(src), 1.0), k=10)

T = 72
t = np.arange(T)
tide = 1.8 * np.sin(2 * np.pi * t * 10 / 60 / 12.42 + 1.0)

def storm(total, dur=18, peak=30):
    r = np.exp(-0.5 * ((t - peak) / (dur / 2.5)) ** 2)
    r[t > peak + dur] = 0
    return r / r.sum() * total

# Check 1: Zero rain -> Zero flood
zero_out = sim.simulate(np.zeros(T), tide)
assert zero_out.max() == 0, f"zero rain must give zero flood, got {zero_out.max()}"
print(" [PASS] Zero rain gives zero flood")

# Check 2: Monotonicity with rain volume
m = [sim.simulate(storm(x), tide).max() for x in [20, 60, 120, 250]]
assert all(m[i] <= m[i + 1] for i in range(3)), f"more rain must not give less flood: {m}"
print(f" [PASS] More rain yields more flood: max depths = {[round(x, 4) for x in m]} m")

# Check 3: Pumps ON vs Pumps OFF
on = sim.simulate(storm(120), tide)
off = sim.simulate(storm(120), tide, pump_on=np.zeros((T, 3)))
assert on.sum() < off.sum(), f"pumps ON must give less flooding than OFF: on={on.sum():.2f}, off={off.sum():.2f}"
print(f" [PASS] Pumps ON reduces flooding: on={on.sum():.2f} vs off={off.sum():.2f}")

# Check 4: No NaN or negative depth
assert on.min() >= 0 and not np.isnan(on).any(), "no negative or NaN water"
print(" [PASS] No negative or NaN water values")

# Check 5: Closed gates vs Auto gates
closed = sim.simulate(storm(120), tide, gate_open=np.zeros(T))
assert closed.max() > on.max(), "gates stuck closed must be worse than automatic gates"
print(f" [PASS] Gates stuck closed is worse: closed max={closed.max():.4f}m vs auto max={on.max():.4f}m")

# -------------------------------------------------------------
# Part 2: Real Hindmata Area Tests
# -------------------------------------------------------------
if os.path.exists("data/hindmata/nodes.csv"):
    print("\nRunning sanity checks on real Hindmata dataset...")
    sim_real, nodes, edges, pumps = load_sim("data/hindmata", k=10)
    T_real = 96
    t_real = np.arange(T_real)
    tide_real = 1.8 * np.sin(2 * np.pi * t_real * 10 / 60 / 12.42 + 1.0)
    
    def storm_real(total):
        r = np.exp(-0.5 * ((t_real - 35) / (20 / 2.5)) ** 2)
        return r / r.sum() * total
        
    zero_real = sim_real.simulate(np.zeros(T_real), tide_real)
    assert zero_real.max() == 0, "Hindmata: zero rain must give zero flood"
    print(" [PASS] Hindmata: Zero rain gives zero flood")
    
    on_real = sim_real.simulate(storm_real(120), tide_real)
    off_real = sim_real.simulate(storm_real(120), tide_real, pump_on=np.zeros((T_real, sim_real.P)))
    assert on_real.sum() <= off_real.sum(), "Hindmata: pumps ON must be <= OFF"
    print(f" [PASS] Hindmata: Pumps ON reduces flooding ({on_real.sum():.2f} vs {off_real.sum():.2f})")
    
    assert on_real.min() >= 0 and not np.isnan(on_real).any(), "Hindmata: valid numbers"
    print(" [PASS] Hindmata: No negative or NaN water values")

print("\nALL SANITY TESTS PASSED")

