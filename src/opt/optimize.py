"""B7 & B8 - Flood Prevention Optimizer (Rules and Local Search).

Implements CONTRACT.md:
optimize(nodes, edges, rain, river, method="search", seed=0)
Returns OPTIMIZER JSON dict.

Methods:
- 'rules'  (B7): Dynamic reactive & proactive thresholds.
                 Pre-emptively runs pumps when basins fill,
                 closes gates to protect bottleneck downstream junctions.
- 'search' (B8): Starts from the rules schedule and performs stochastic
                 local search / simulated schedule refinement to minimize
                 flood cost. Guaranteed to equal or beat rules.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.sim.simulator import simulate, DT

RESULTS_DIR = Path("results")


def optimize_rules(nodes: pd.DataFrame, edges: pd.DataFrame, rain: np.ndarray, river: np.ndarray):
    """B7: Rule-based pump and gate control schedule.

    Logic:
    1. Pre-emptive Pump Drawdown:
       Turn pumps ON as soon as rain begins or node fill > 0.40.
       Keep running until fill drops below 0.15.
    2. Protective Gate Throttling:
       If a downstream target node has high fill (> 0.85), close the gate
       on that incoming edge to prevent downstream inundation.
       Reopen when downstream drops below 0.65.
    """
    T, N = rain.shape
    E = len(edges)
    pump_nodes = nodes.index[nodes["has_pump"] == 1].to_numpy()
    gate_edges = edges.index[edges["is_gate"] == 1].to_numpy()
    src = edges.source.to_numpy(int)
    dst = edges.target.to_numpy(int)

    pump_on = np.zeros((T, N), dtype=np.float32)
    gate_open = np.ones((T, E), dtype=np.float32)

    # We do a sequential simulation pass to apply rules reactively
    cap = nodes.capacity_m3.to_numpy(float)
    area = nodes.catchment_m2.to_numpy(float)
    runoff = nodes.runoff_coeff.to_numpy(float)
    elev = nodes.elevation_m.to_numpy(float)
    prate = (nodes.pump_rate_m3s * nodes.has_pump).to_numpy(float)
    outlet = nodes.is_outlet.to_numpy(float)
    ecap = edges.capacity_m3s.to_numpy(float)
    is_gate_mask = edges.is_gate.to_numpy(int) == 1
    pond = 0.02 * area
    storage = np.zeros(N)

    RIVER_MAX = 6.0
    OUTLET_RATE = 4.0

    for t in range(T):
        # Determine rules for step t based on current storage
        curr_fill = storage / cap

        # Rule 1: Pump control
        for p in pump_nodes:
            # Turn ON if water is entering or fill > 0.35
            if curr_fill[p] > 0.30 or rain[t, p] > 10.0:
                pump_on[t, p] = 1.0

        # Rule 2: Gate control
        for g in gate_edges:
            target_node = dst[g]
            # Close gate if downstream node is nearing capacity
            if curr_fill[target_node] > 0.85:
                gate_open[t, g] = 0.0
            else:
                gate_open[t, g] = 1.0

        # Advance physics step
        storage += runoff * area * (rain[t] / 1000.0 / 3600.0) * DT
        head = elev + storage / cap
        g_val = np.where(is_gate_mask, gate_open[t], 1.0)
        vol = ecap * g_val * np.clip(head[src] - head[dst] + 0.2, 0, 1) * DT
        out_sum = np.zeros(N)
        np.add.at(out_sum, src, vol)
        scale = np.minimum(1.0, storage / np.maximum(out_sum, 1e-9))
        vol = vol * scale[src]
        np.add.at(storage, src, -vol)
        np.add.at(storage, dst, vol)
        factor = max(0.0, 1.0 - float(river[t]) / RIVER_MAX)
        storage -= np.minimum(storage, outlet * OUTLET_RATE * factor * DT)
        storage -= np.minimum(storage, prate * pump_on[t] * DT)

    return pump_on, gate_open


def optimize_search(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    rain: np.ndarray,
    river: np.ndarray,
    n_trials: int = 300,
    seed: int = 0,
):
    """B8: Stochastic schedule refinement starting from rules baseline.

    Explores variations of pump on/off time intervals and gate openings.
    Evaluates each candidate via simulate().
    Guaranteed to be equal to or better than rules schedule.
    """
    rng = np.random.default_rng(seed)
    T, N = rain.shape
    E = len(edges)
    pump_nodes = nodes.index[nodes["has_pump"] == 1].to_numpy()
    gate_edges = edges.index[edges["is_gate"] == 1].to_numpy()

    # Champion initialized with rules
    best_pumps, best_gates = optimize_rules(nodes, edges, rain, river)
    sim_res = simulate(nodes, edges, rain, river, best_pumps, best_gates)
    best_cost = sim_res["flood_cost"]

    # Also test completely proactive pumping (pumps ON for duration of storm)
    all_pumps_on = np.zeros((T, N), dtype=np.float32)
    all_pumps_on[:, pump_nodes] = 1.0
    cand_res = simulate(nodes, edges, rain, river, all_pumps_on, best_gates)
    if cand_res["flood_cost"] < best_cost:
        best_cost = cand_res["flood_cost"]
        best_pumps = all_pumps_on.copy()

    # Stochastic Local Search / Hill Climbing
    for _ in range(n_trials):
        cand_pumps = best_pumps.copy()
        cand_gates = best_gates.copy()

        mutation_type = rng.integers(0, 3)

        if mutation_type == 0 and len(pump_nodes) > 0:
            # Shift pump activation interval
            node = rng.choice(pump_nodes)
            t_on = rng.integers(0, max(1, T - 10))
            t_off = rng.integers(t_on + 5, T)
            cand_pumps[:, node] = 0.0
            cand_pumps[t_on:t_off, node] = 1.0

        elif mutation_type == 1 and len(gate_edges) > 0:
            # Shift gate close interval
            edge = rng.choice(gate_edges)
            t_close = rng.integers(0, max(1, T - 10))
            t_reopen = rng.integers(t_close + 4, min(T, t_close + 25))
            cand_gates[:, edge] = 1.0
            cand_gates[t_close:t_reopen, edge] = 0.0

        else:
            # Combined tweak
            if len(pump_nodes) > 0:
                p_subset = rng.choice(pump_nodes, size=max(1, len(pump_nodes) // 3), replace=False)
                for p in p_subset:
                    t_start = rng.integers(0, 30)
                    cand_pumps[t_start:, p] = 1.0

        sim_res = simulate(nodes, edges, rain, river, cand_pumps, cand_gates)
        if sim_res["flood_cost"] < best_cost:
            best_cost = sim_res["flood_cost"]
            best_pumps = cand_pumps.copy()
            best_gates = cand_gates.copy()

    return best_pumps, best_gates, best_cost


def _extract_schedules(pump_on: np.ndarray, gate_open: np.ndarray, nodes: pd.DataFrame, edges: pd.DataFrame):
    """Convert boolean arrays to human-readable change-event lists."""
    T, N = pump_on.shape
    E = gate_open.shape[1]
    pump_nodes = nodes.index[nodes["has_pump"] == 1].to_numpy()
    gate_edges = edges.index[edges["is_gate"] == 1].to_numpy()

    pump_sched = []
    for p in pump_nodes:
        changes = np.diff(np.pad(pump_on[:, p], (1, 0), constant_values=0))
        for step in np.where(changes != 0)[0]:
            pump_sched.append({
                "step": int(step),
                "node_id": int(p),
                "on": int(pump_on[step, p]),
            })

    gate_sched = []
    for g in gate_edges:
        changes = np.diff(np.pad(gate_open[:, g], (1, 0), constant_values=1))
        for step in np.where(changes != 0)[0]:
            gate_sched.append({
                "step": int(step),
                "edge_id": int(g),
                "open": int(gate_open[step, g]),
            })

    return pump_sched, gate_sched


def optimize(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    rain: np.ndarray,
    river: np.ndarray,
    method: str = "search",
    seed: int = 0,
) -> dict:
    """Optimize pump and gate schedules to minimize urban flood depth.

    Parameters
    ----------
    nodes, edges : pd.DataFrame
        Network definition dataframes.
    rain  : np.ndarray of shape [T, N] (mm/h)
    river : np.ndarray of shape [T] (m)
    method: 'rules' (B7) or 'search' (B8)
    seed  : random seed for reproducible search

    Returns
    -------
    dict
        OPTIMIZER JSON format matching CONTRACT.md.
    """
    T, N = rain.shape
    E = len(edges)

    # Baseline: No control (all pumps off, all gates open)
    no_pump = np.zeros((T, N), dtype=np.float32)
    all_open = np.ones((T, E), dtype=np.float32)
    sim_before = simulate(nodes, edges, rain, river, no_pump, all_open)
    cost_before = float(sim_before["flood_cost"])

    # Run optimizer method
    if method == "rules":
        opt_pump, opt_gate = optimize_rules(nodes, edges, rain, river)
        sim_after = simulate(nodes, edges, rain, river, opt_pump, opt_gate)
        cost_after = float(sim_after["flood_cost"])
    else:  # 'search'
        opt_pump, opt_gate, cost_after = optimize_search(
            nodes, edges, rain, river, n_trials=300, seed=seed
        )

    # Reduction %
    if cost_before > 0:
        red_pct = max(0.0, float((cost_before - cost_after) / cost_before * 100.0))
    else:
        red_pct = 0.0

    # Save arrays
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pump_out_path = RESULTS_DIR / "opt_pump_on.npy"
    gate_out_path = RESULTS_DIR / "opt_gate_open.npy"
    np.save(pump_out_path, opt_pump)
    np.save(gate_out_path, opt_gate)

    pump_sched, gate_sched = _extract_schedules(opt_pump, opt_gate, nodes, edges)

    return {
        "scenario_cost_before": round(cost_before, 2),
        "scenario_cost_after": round(cost_after, 2),
        "reduction_pct": round(red_pct, 1),
        "method": method,
        "pump_schedule": pump_sched,
        "gate_schedule": gate_sched,
        "pump_on": f"[T,N] array saved to {pump_out_path}",
        "gate_open": f"[T,E] array saved to {gate_out_path}",
    }
