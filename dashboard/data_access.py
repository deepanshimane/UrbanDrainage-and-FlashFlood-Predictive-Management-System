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
    dt_seconds: float = 900.0
    T: int = 72
    N: int = 200
    E: int = 358
    S: int = 60
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
    return {"num_nodes": 200, "history_steps": 8, "horizon_steps": 12, "dt_seconds": 900.0}


def list_areas(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scans data/ directory and returns detected areas with format and compatibility."""
    d = base_dir or DATA_DIR
    results = []
    
    # 1. Sample area
    sample_path = d / "sample"
    if sample_path.exists():
        has_nodes = (sample_path / "nodes.csv").exists()
        has_depth = (sample_path / "depth.npy").exists()
        status = "contract" if (has_nodes and has_depth) else "incomplete"
        results.append({
            "name": "sample",
            "display_name": "Sample Synthetic Area (Mumbai Ward)",
            "format": "contract",
            "model_compatible": True,
            "dt_seconds": 900.0,
            "status": "ready" if status == "contract" else "incomplete",
            "reason": "" if status == "contract" else "Missing .npy arrays (run make_fake_data)",
        })
        
    # 2. Hindmata area
    hindmata_path = d / "hindmata"
    if hindmata_path.exists():
        has_nodes = (hindmata_path / "nodes.csv").exists()
        has_dataset = (hindmata_path / "dataset" / "X.npy").exists()
        status = "areaA" if has_nodes else "incomplete"
        results.append({
            "name": "hindmata",
            "display_name": "Mumbai: Dadar-Hindmata-Parel (Area A)",
            "format": "areaA",
            "model_compatible": False,
            "dt_seconds": 600.0,
            "status": "ready" if has_dataset else "nodes_only",
            "reason": "Trained ST-GNN requires N=200, dt=15m (Sample format). Hindmata uses N=89, dt=10m.",
        })
    else:
        results.append({
            "name": "hindmata",
            "display_name": "Mumbai: Dadar-Hindmata-Parel (Area A)",
            "format": "areaA",
            "model_compatible": False,
            "dt_seconds": 600.0,
            "status": "disabled",
            "reason": "Area files not yet generated on this branch",
        })
        
    # 3. Kurla area
    kurla_path = d / "kurla"
    if kurla_path.exists():
        results.append({
            "name": "kurla",
            "display_name": "Mumbai: Kurla-Mithi River",
            "format": "areaA",
            "model_compatible": False,
            "dt_seconds": 600.0,
            "status": "ready",
            "reason": "Model retrain needed for Kurla topography",
        })
    else:
        results.append({
            "name": "kurla",
            "display_name": "Mumbai: Kurla-Mithi River (Scalability)",
            "format": "unrecognised",
            "model_compatible": False,
            "dt_seconds": 0.0,
            "status": "disabled",
            "reason": "Second area dataset not yet provided by Person A",
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
    expected_n = meta.get("num_nodes", 200)
    
    if name == "sample":
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
                notes=["Master contract format (dt=900s, T=72, N=200, S=60)."],
            )
            return ds, None
        except Exception as e:
            return None, f"Error loading sample dataset: {str(e)}"
            
    elif name == "hindmata":
        # Load Hindmata format
        try:
            nodes = pd.read_csv(target_dir / "nodes.csv")
            edges = pd.read_csv(target_dir / "edges.csv")
            
            known_spots = pd.read_csv(target_dir / "known_spots.csv") if (target_dir / "known_spots.csv").exists() else None
            pumps = pd.read_csv(target_dir / "pumps.csv") if (target_dir / "pumps.csv").exists() else None
            
            dataset_dir = target_dir / "dataset"
            if (dataset_dir / "X.npy").exists():
                X = np.load(dataset_dir / "X.npy") # [S, 96, N, 4]
                S, T, N, F = X.shape
                rain = X[..., 0]   # rain_mm
                river = X[..., 1].mean(axis=-1) # tide_m
                depth = X[..., 2]  # depth_m
                pump_on = X[..., 3]
                
                scenarios = {"Light rain": 0, "Heavy storm": 1, "Extreme cloudburst": 2}
                if (dataset_dir / "scenarios.json").exists():
                    try:
                        with open(dataset_dir / "scenarios.json") as f:
                            scen_json = json.load(f)
                        scenarios = {
                            "Light rain": int(scen_json.get("light_rain", 0)),
                            "Heavy storm": int(scen_json.get("heavy_storm", 1)),
                            "Extreme cloudburst": int(scen_json.get("extreme_cloudburst", 2)),
                        }
                    except Exception:
                        pass
                
                ds = Dataset(
                    name="hindmata",
                    nodes=nodes,
                    edges=edges,
                    rain=rain,
                    river=river,
                    pump_on=pump_on,
                    gate_open=None,
                    fill=None,
                    depth=depth,
                    dt_seconds=600.0,
                    T=T, N=N, E=len(edges), S=S,
                    is_model_compatible=False,
                    incompatible_reason="Forecast model was trained on the sample format (dt=15m, N=200); retrain needed for Hindmata (dt=10m, N=89).",
                    known_spots=known_spots,
                    pumps=pumps,
                    scenarios=scenarios,
                    notes=["Person A dataset (dt=600s, T=96, N=89). ML forecast disabled."],
                )
                return ds, None
            else:
                # Nodes & edges only
                ds = Dataset(
                    name="hindmata",
                    nodes=nodes,
                    edges=edges,
                    rain=np.zeros((1, 96, len(nodes)), dtype=np.float32),
                    river=np.zeros((1, 96), dtype=np.float32),
                    depth=np.zeros((1, 96, len(nodes)), dtype=np.float32),
                    dt_seconds=600.0,
                    T=96, N=len(nodes), E=len(edges), S=1,
                    is_model_compatible=False,
                    incompatible_reason="Simulation arrays X.npy not found.",
                    known_spots=known_spots,
                    pumps=pumps,
                    notes=["Topological graph available."],
                )
                return ds, None
        except Exception as e:
            return None, f"Error loading hindmata area: {str(e)}"
            
    else:
        return None, f"Unrecognised area format for '{name}'."
