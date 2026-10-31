"""Pure helper functions for FloodSense dashboard and API.
Unit-testable, independent of Streamlit UI.
"""
from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
import pandas as pd


def step_to_clock(step: int, dt_seconds: float = 900.0) -> str:
    """Converts a simulation time step index to clock format 'T+HH:MM'."""
    if step < 0:
        step = 0
    total_seconds = int(step * dt_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"T+{hours:02d}:{minutes:02d}"


def horizon_steps(horizon_hours: float, dt_seconds: float = 900.0) -> int:
    """Calculates number of simulation steps for a given horizon in hours."""
    if dt_seconds <= 0:
        return 0
    return int(round((horizon_hours * 3600.0) / dt_seconds))


def depth_band(depth_cm: float) -> Tuple[str, List[int]]:
    """Maps depth in centimeters to severity category and RGBA color.
    Bands:
      <10 cm:  Safe (blue)
      10-30 cm: Minor (yellow)
      30-60 cm: Moderate (orange)
      60-100 cm: Severe (red)
      >100 cm: Extreme (dark red)
    """
    if depth_cm < 10.0:
        return "Safe", [56, 189, 248, 180]        # Sky blue
    elif depth_cm < 30.0:
        return "Minor", [250, 204, 21, 200]       # Yellow
    elif depth_cm < 60.0:
        return "Moderate", [249, 115, 22, 220]    # Orange
    elif depth_cm <= 100.0:
        return "Severe", [239, 68, 68, 230]       # Bright red
    else:
        return "Extreme", [159, 18, 57, 240]      # Dark red / Rose


def human_severity_desc(depth_cm: float) -> str:
    """Translates depth into everyday physical reference for citizens and municipal teams."""
    if depth_cm < 10.0:
        return "🟢 Safe (Pavement dry, normal traffic)"
    elif depth_cm < 25.0:
        return "🟡 Ankle-Deep (Slow traffic, small pooling)"
    elif depth_cm < 50.0:
        return "🟠 Knee-Deep (Small cars stall, traffic diverted)"
    else:
        return "🔴 Waist-Deep (Road closed, shops inundated)"


# Known civic landmarks for Dadar-Hindmata-Parel and Kurla
MUMBAI_LANDMARKS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "hindmata": {
        "Hindmata Flyover Junction": (19.0091, 72.8419),
        "Pramod Mahajan Kala Park (Holding Tank)": (19.0105, 72.8415),
        "Dadar TT Circle": (19.0175, 72.8435),
        "Gandhi Market / King's Circle": (19.0352, 72.8592),
        "Britannia Outfall / Sewri": (18.9760, 72.8465),
        "Parel TT / Bhoiwada": (19.0035, 72.8410),
        "Chitra Cinema / Dr. Ambedkar Rd": (19.0140, 72.8425),
    },
    "kurla": {
        "Bail Bazar Junction": (19.0750, 72.8830),
        "Kranti Nagar (Mithi River Bank)": (19.0725, 72.8755),
        "CST Road Mithi River Bridge": (19.0680, 72.8680),
        "Kurla Station West Subway": (19.0655, 72.8790),
    }
}


def get_landmark_for_node(area_name: str, node_id: int, lat: float, lon: float) -> str:
    """Resolves an abstract node index to a recognizable real-world street location."""
    area_key = "hindmata" if "hindmata" in area_name.lower() or "dadar" in area_name.lower() else ("kurla" if "kurla" in area_name.lower() else None)
    
    # Specific known node overrides for Hindmata
    if area_key == "hindmata":
        if node_id in (50, 53):
            return "Hindmata Cinema Junction"
        elif node_id in (70, 71):
            return "Dadar TT Circle"
        elif node_id == 76:
            return "Gandhi Market / King's Circle"
        elif node_id == 16:
            return "Britannia Pumping Outfall"
        elif node_id == 6:
            return "Parel TT / Bhoiwada"
        elif node_id == 10:
            return "Pramod Mahajan Park Tank"
            
    if area_key == "kurla":
        if node_id in (80, 81):
            return "Bail Bazar Junction"
        elif node_id in (75, 76):
            return "Kranti Nagar (Mithi Bank)"
        elif node_id in (21, 22):
            return "CST Road Mithi Bridge"
        elif node_id in (46, 47):
            return "Kurla Station West Area"

    # Nearest landmark search
    if area_key and area_key in MUMBAI_LANDMARKS:
        best_name = None
        min_dist = float("inf")
        for name, (l_lat, l_lon) in MUMBAI_LANDMARKS[area_key].items():
            dist = haversine_distance_m(lat, lon, l_lat, l_lon)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        if best_name and min_dist <= 350.0:
            return f"{best_name} ({int(min_dist)}m)"
        elif best_name:
            return f"Near {best_name}"

    return f"Street Junction #{node_id}"


def build_recent(ds: Any, storm_idx: int, t_current: int, history_steps: int = 8) -> Optional[Dict[str, np.ndarray]]:
    """Extracts recent window of history [history_steps, N] ending at t_current.
    If t_current < history_steps, returns None.
    """
    if t_current < history_steps:
        return None
    
    t_start = t_current - history_steps
    try:
        rain = ds.rain[storm_idx, t_start:t_current]       # [8, N]
        fill = ds.fill[storm_idx, t_start:t_current] if ds.fill is not None else np.zeros_like(rain)
        river = ds.river[storm_idx, t_start:t_current]     # [8]
        if ds.pump_on is not None:
            pump_on = ds.pump_on[storm_idx, t_start:t_current]
        else:
            pump_on = np.zeros_like(rain)
        return {
            "rain": np.asarray(rain, dtype=np.float32),
            "fill": np.asarray(fill, dtype=np.float32),
            "river": np.asarray(river, dtype=np.float32),
            "pump_on": np.asarray(pump_on, dtype=np.float32),
        }
    except Exception:
        return None


def rank_storms_by_peak_rain(rain_array: np.ndarray) -> Dict[str, int]:
    """Ranks storms in dataset by peak spatial rainfall and selects Light, Heavy, Extreme."""
    if rain_array.ndim == 3:
        peak_per_storm = rain_array.max(axis=(1, 2))
    elif rain_array.ndim == 2:
        peak_per_storm = rain_array.max(axis=1)
    else:
        return {"Light rain": 0, "Heavy storm": 0, "Extreme cloudburst": 0}
    
    S = len(peak_per_storm)
    order = np.argsort(peak_per_storm)
    
    return {
        "Light rain": int(order[max(0, int(S * 0.1))]),
        "Heavy storm": int(order[min(S - 1, int(S * 0.6))]),
        "Extreme cloudburst": int(order[-1]),
    }


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in metres."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def nearest_node(lat: float, lon: float, nodes: pd.DataFrame) -> Tuple[int, float]:
    """Finds nearest node index and distance in metres."""
    if len(nodes) == 0:
        return -1, 0.0
    
    node_id_col = "node_id" if "node_id" in nodes.columns else ("id" if "id" in nodes.columns else nodes.columns[0])
    best_id = int(nodes.iloc[0][node_id_col])
    min_dist = float("inf")
    
    for idx, row in nodes.iterrows():
        dist = haversine_distance_m(lat, lon, float(row["lat"]), float(row["lon"]))
        if dist < min_dist:
            min_dist = dist
            best_id = int(row[node_id_col])
            
    return best_id, round(min_dist, 1)


def risk_table(depth_by_node_m: np.ndarray, nodes: pd.DataFrame, top_k: int = 10, pumps: Optional[pd.DataFrame] = None, area_name: str = "hindmata") -> pd.DataFrame:
    """Generates a ranked risk table for top-k inundated nodes with recognizable landmark names."""
    N = len(depth_by_node_m)
    node_id_col = "node_id" if "node_id" in nodes.columns else ("id" if "id" in nodes.columns else None)
    
    records = []
    for i in range(N):
        d_cm = round(float(depth_by_node_m[i]) * 100.0, 1)
        nid = int(nodes.iloc[i][node_id_col]) if node_id_col else i
        lat = float(nodes.iloc[i]["lat"])
        lon = float(nodes.iloc[i]["lon"])
        elev = float(nodes.iloc[i]["elevation_m"]) if "elevation_m" in nodes.columns else (float(nodes.iloc[i]["elevation"]) if "elevation" in nodes.columns else 0.0)
        has_pump = int(nodes.iloc[i]["has_pump"]) if "has_pump" in nodes.columns else 0
        band_name, _ = depth_band(d_cm)
        landmark = get_landmark_for_node(area_name, nid, lat, lon)
        severity_desc = human_severity_desc(d_cm)
        
        # Distance to nearest pump
        pump_dist_m = 0.0
        if pumps is not None and len(pumps) > 0 and "lat" in pumps.columns:
            pump_dist_m = min(haversine_distance_m(lat, lon, float(p["lat"]), float(p["lon"])) for _, p in pumps.iterrows())
        
        records.append({
            "node_id": nid,
            "landmark": landmark,
            "depth_cm": d_cm,
            "severity": severity_desc,
            "band": band_name,
            "elevation_m": round(elev, 2),
            "lat": lat,
            "lon": lon,
            "has_pump": bool(has_pump),
            "nearest_pump_m": round(pump_dist_m, 1),
        })
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="depth_cm", ascending=False).reset_index(drop=True)
    return df.head(top_k)
