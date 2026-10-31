"""Data Access Layer for FloodSense.
Scans data directories, loads contract and areaA datasets safely,
checks model compatibility, and provides structured dataset objects.
Does NOT import streamlit so it can be tested in CI.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class Dataset:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    rain: np.ndarray             # [S, T, N] mm/h
    river: np.ndarray            # [S, T] m
    pump_on: Optional[np.ndarray] = None   # [S, T, N]
    gate_open: Optional[np.ndarray] = None # [S, T, E]
    fill: Optional[np.ndarray] = None      # [S, T, N]
    depth: Optional[np.ndarray] = None     # [S, T, N] metres
    dt_seconds: float = 600.0
    T: int = 72
    N: int = 89
    E: int = 140
    S: int = 3
    is_model_compatible: bool = True
    incompatible_reason: str = ""
    notes: List[str] = field(default_factory=list)
    known_spots: Optional[pd.DataFrame] = None
    pumps: Optional[pd.DataFrame] = None
    scenarios: Dict[str, int] = field(default_factory=dict)


def get_model_constraints() -> Dict[str, Any]:
    """Reads constraints from models/best_meta.json."""
    meta_path = MODELS_DIR / "best_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"num_nodes": 89, "history_steps": 8, "horizon_steps": 12, "dt_seconds": 600.0}


def list_areas(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scans data/ directory and returns detected areas with format and compatibility."""
    d = base_dir or DATA_DIR
    meta = get_model_constraints()
    expected_n = meta.get("num_nodes", 89)
    results = []
    
    # 1. Hindmata area (Primary study area)
    hindmata_path = d / "hindmata"
    if hindmata_path.exists():
        has_nodes = (hindmata_path / "nodes.csv").exists()
        has_scenarios = (RESULTS_DIR / "scenarios" / "heavy_rain.npy").exists() or (hindmata_path / "dataset" / "X.npy").exists()
        is_compat = (expected_n == 89)
        results.append({
            "name": "hindmata",
            "display_name": "Mumbai: Dadar-Hindmata-Parel (Primary Area)",
            "format": "areaA",
            "model_compatible": is_compat,
            "dt_seconds": 600.0,
            "status": "ready" if (has_nodes and has_scenarios) else "nodes_only",
            "reason": "" if is_compat else f"Model trained for N={expected_n}, Hindmata has N=89",
        })
    else:
        results.append({
            "name": "hindmata",
            "display_name": "Mumbai: Dadar-Hindmata-Parel (Primary Area)",
            "format": "areaA",
            "model_compatible": False,
            "dt_seconds": 600.0,
            "status": "disabled",
            "reason": "Area files not yet generated on this branch",
        })

    # 2. Kurla area (Scalability catchment)
    kurla_path = d / "kurla"
    if kurla_path.exists():
        has_nodes = (kurla_path / "nodes.csv").exists()
        results.append({
            "name": "kurla",
            "display_name": "Mumbai: Kurla-Mithi River (Scalability Area)",
            "format": "areaA",
            "model_compatible": False,
            "dt_seconds": 600.0,
            "status": "ready" if has_nodes else "incomplete",
            "reason": "Scalability evaluation area (N=91). ST-GNN specialized for Hindmata (N=89).",
        })
    else:
        results.append({
            "name": "kurla",
            "display_name": "Mumbai: Kurla-Mithi River (Scalability Area)",
            "format": "unrecognised",
            "model_compatible": False,
            "dt_seconds": 0.0,
            "status": "disabled",
            "reason": "Second area dataset not yet provided",
        })

    # 3. Sample benchmark area
    sample_path = d / "sample"
    if sample_path.exists():
        has_nodes = (sample_path / "nodes.csv").exists()
        has_depth = (sample_path / "depth.npy").exists()
        status = "contract" if (has_nodes and has_depth) else "incomplete"
        is_compat = (expected_n == 200)
        results.append({
            "name": "sample",
            "display_name": "Sample Synthetic Area (Mumbai Ward Benchmark)",
            "format": "contract",
            "model_compatible": is_compat,
            "dt_seconds": 900.0,
            "status": "ready" if status == "contract" else "incomplete",
            "reason": "" if is_compat else f"Model trained for N={expected_n}; sample has N=200",
        })
        
    return results


def load_area(name: str, base_dir: Optional[Path] = None) -> Tuple[Optional[Dataset], Optional[str]]:
    """Loads a named area dataset.
    Returns (Dataset, None) on success or (None, error_str) on failure.
    """
    d = base_dir or DATA_DIR
    target_dir = d / name
    
    if not target_dir.exists():
        return None, f"Directory '{target_dir}' does not exist."
        
    meta = get_model_constraints()
    expected_n = meta.get("num_nodes", 89)
    
    if name == "hindmata":
        try:
            nodes = pd.read_csv(target_dir / "nodes.csv")
            edges = pd.read_csv(target_dir / "edges.csv")
            N = len(nodes)
            E = len(edges)
            
            known_spots = pd.read_csv(target_dir / "known_spots.csv") if (target_dir / "known_spots.csv").exists() else None
            pumps = pd.read_csv(target_dir / "pumps.csv") if (target_dir / "pumps.csv").exists() else None

            # Check if canonical pitch scenarios exist in results/scenarios/
            scen_dir = RESULTS_DIR / "scenarios"
            has_pitch_scenarios = (
                (scen_dir / "light_rain.npy").exists() and
                (scen_dir / "heavy_rain.npy").exists() and
                (scen_dir / "extreme_rain.npy").exists()
            )
            
            from src.sim.simulator import simulate, DT
            
            if has_pitch_scenarios:
                light_rain = np.load(scen_dir / "light_rain.npy")
                light_river = np.load(scen_dir / "light_river.npy")
                heavy_rain = np.load(scen_dir / "heavy_rain.npy")
                heavy_river = np.load(scen_dir / "heavy_river.npy")
                extreme_rain = np.load(scen_dir / "extreme_rain.npy")
                extreme_river = np.load(scen_dir / "extreme_river.npy")
                
                T = light_rain.shape[0]
                rain = np.stack([light_rain, heavy_rain, extreme_rain], axis=0) # [3, T, N]
                river = np.stack([light_river, heavy_river, extreme_river], axis=0) # [3, T]
                
                # Simulate baseline depth and fill for each scenario
                no_pump = np.zeros((T, N), dtype=np.float32)
                all_open = np.ones((T, E), dtype=np.float32)
                
                sim_light = simulate(nodes, edges, light_rain, light_river, no_pump, all_open)
                sim_heavy = simulate(nodes, edges, heavy_rain, heavy_river, no_pump, all_open)
                sim_extreme = simulate(nodes, edges, extreme_rain, extreme_river, no_pump, all_open)
                
                depth = np.stack([sim_light["depth"], sim_heavy["depth"], sim_extreme["depth"]], axis=0)
                fill = np.stack([sim_light["fill"], sim_heavy["fill"], sim_extreme["fill"]], axis=0)
                pump_on = np.zeros((3, T, N), dtype=np.float32)
                gate_open = np.ones((3, T, E), dtype=np.float32)
                
                scenarios = {"Light rain": 0, "Heavy storm": 1, "Extreme cloudburst": 2}
                S = 3
            elif (target_dir / "dataset" / "X.npy").exists():
                X = np.load(target_dir / "dataset" / "X.npy") # [S, 96, N, 4]
                S, T, _, _ = X.shape
                rain = X[..., 0]
                river = X[..., 1].mean(axis=-1)
                depth = X[..., 2]
                pump_on = X[..., 3]
                fill = np.zeros_like(depth)
                gate_open = np.ones((S, T, E), dtype=np.float32)
                scenarios = {"Light rain": 0, "Heavy storm": 1, "Extreme cloudburst": 2}
            else:
                T = 72
                S = 1
                rain = np.zeros((1, T, N), dtype=np.float32)
                river = np.zeros((1, T), dtype=np.float32)
                depth = np.zeros((1, T, N), dtype=np.float32)
                fill = np.zeros((1, T, N), dtype=np.float32)
                pump_on = np.zeros((1, T, N), dtype=np.float32)
                gate_open = np.ones((1, T, E), dtype=np.float32)
                scenarios = {"Light rain": 0, "Heavy storm": 0, "Extreme cloudburst": 0}

            is_compat = (N == expected_n)
            incompat_reason = "" if is_compat else f"Node count {N} != model {expected_n}"
            
            ds = Dataset(
                name="hindmata",
                nodes=nodes,
                edges=edges,
                rain=rain,
                river=river,
                pump_on=pump_on,
                gate_open=gate_open,
                fill=fill,
                depth=depth,
                dt_seconds=DT,
                T=T, N=N, E=E, S=S,
                is_model_compatible=is_compat,
                incompatible_reason=incompat_reason,
                known_spots=known_spots,
                pumps=pumps,
                scenarios=scenarios,
                notes=["Primary real study area: Mumbai Dadar-Hindmata-Parel (N=89, E=140, 3 pumps). ST-GNN forecast & AI dispatch active."],
            )
            return ds, None
        except Exception as e:
            return None, f"Error loading hindmata area: {str(e)}"

    elif name == "kurla":
        try:
            nodes = pd.read_csv(target_dir / "nodes.csv")
            edges = pd.read_csv(target_dir / "edges.csv")
            N = len(nodes)
            E = len(edges)
            known_spots = pd.read_csv(target_dir / "known_spots.csv") if (target_dir / "known_spots.csv").exists() else None
            pumps = pd.read_csv(target_dir / "pumps.csv") if (target_dir / "pumps.csv").exists() else None
            
            if (target_dir / "dataset" / "X.npy").exists():
                X = np.load(target_dir / "dataset" / "X.npy") # [S, 96, N, 4]
                S, T, _, _ = X.shape
                rain = X[..., 0]
                river = X[..., 1].mean(axis=-1)
                depth = X[..., 2]
                pump_on = X[..., 3]
                fill = np.zeros_like(depth)
                gate_open = np.ones((S, T, E), dtype=np.float32)
                
                scenarios = {"Light rain": 61, "Heavy storm": 73, "Extreme cloudburst": 90}
                if (target_dir / "dataset" / "scenarios.json").exists():
                    try:
                        with open(target_dir / "dataset" / "scenarios.json") as sf:
                            s_raw = json.load(sf)
                            scenarios = {
                                "Light rain": s_raw.get("light_rain", 61),
                                "Heavy storm": s_raw.get("heavy_storm", 73),
                                "Extreme cloudburst": s_raw.get("extreme_cloudburst", 90),
                            }
                    except Exception:
                        pass
            else:
                T = 72
                S = 1
                rain = np.zeros((1, T, N), dtype=np.float32)
                river = np.zeros((1, T), dtype=np.float32)
                depth = np.zeros((1, T, N), dtype=np.float32)
                fill = np.zeros((1, T, N), dtype=np.float32)
                pump_on = np.zeros((1, T, N), dtype=np.float32)
                gate_open = np.ones((1, T, E), dtype=np.float32)
                scenarios = {"Light rain": 0, "Heavy storm": 0, "Extreme cloudburst": 0}
            
            ds = Dataset(
                name="kurla",
                nodes=nodes,
                edges=edges,
                rain=rain,
                river=river,
                pump_on=pump_on,
                gate_open=gate_open,
                depth=depth,
                fill=fill,
                dt_seconds=600.0,
                T=T, N=N, E=E, S=S,
                is_model_compatible=False,
                incompatible_reason="Scalability validation area (N=91). ST-GNN model is trained on Hindmata (N=89).",
                known_spots=known_spots,
                pumps=pumps,
                scenarios=scenarios,
                notes=["Secondary Mumbai catchment: Kurla / Mithi River (N=91, E=126, 3 pumps). Demonstrates cross-ward scalability."],
            )
            return ds, None
        except Exception as e:
            return None, f"Error loading kurla area: {str(e)}"

    elif name == "sample":
        try:
            nodes = pd.read_csv(target_dir / "nodes.csv")
            edges = pd.read_csv(target_dir / "edges.csv")
            rain = np.load(target_dir / "rain.npy")
            river = np.load(target_dir / "river.npy")
            pump_on = np.load(target_dir / "pump_on.npy") if (target_dir / "pump_on.npy").exists() else None
            gate_open = np.load(target_dir / "gate_open.npy") if (target_dir / "gate_open.npy").exists() else None
            fill = np.load(target_dir / "fill.npy") if (target_dir / "fill.npy").exists() else None
            depth = np.load(target_dir / "depth.npy")
            
            S, T, N = depth.shape
            E = len(edges)
            
            # Scenarios by peak rain
            peak_per_storm = rain.max(axis=(1, 2))
            order = np.argsort(peak_per_storm)
            scenarios = {
                "Light rain": int(order[int(S * 0.1)]),
                "Heavy storm": int(order[int(S * 0.6)]),
                "Extreme cloudburst": int(order[-1]),
            }
            
            is_compat = (N == expected_n)
            incompat_reason = "" if is_compat else f"Node count {N} != model {expected_n}"
            
            ds = Dataset(
                name="sample",
                nodes=nodes,
                edges=edges,
                rain=rain,
                river=river,
                pump_on=pump_on,
                gate_open=gate_open,
                fill=fill,
                depth=depth,
                dt_seconds=900.0,
                T=T, N=N, E=E, S=S,
                is_model_compatible=is_compat,
                incompatible_reason=incompat_reason,
                scenarios=scenarios,
                notes=["Synthetic benchmark format (dt=900s, T=72, N=200, S=60)."],
            )
            return ds, None
        except Exception as e:
            return None, f"Error loading sample dataset: {str(e)}"
            
    else:
        return None, f"Unrecognised area format for '{name}'."
