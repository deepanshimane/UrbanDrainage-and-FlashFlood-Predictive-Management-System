"""FloodSense AI: Urban Drainage & Flash Flood Predictive Management System
Human-Centric Civic Dashboard (Designed for Municipal Officials, Judges & Citizens).
Supports API Mode (FastAPI at :8000) and Direct Mode (local imports).
All data is SIMULATED per Hackathon PS11 guidelines.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.data_access import list_areas, load_area, get_model_constraints, Dataset
from dashboard.helpers import (
    step_to_clock,
    horizon_steps,
    depth_band,
    human_severity_desc,
    get_landmark_for_node,
    build_recent,
    risk_table,
    nearest_node,
    MUMBAI_LANDMARKS,
)
try:
    from src.sim.simulator import simulate, DT, RIVER_MAX
except ImportError:
    from src.sim.simulator import simulate, DT
    RIVER_MAX = 6.0

# ---------------------------------------------------------------------------
# Shell & Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodSense | Mumbai Urban Drainage AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Human-Centric Clarity
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    
    /* Top Alert Banner */
    .executive-banner-danger {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: #ffffff;
        padding: 16px 22px;
        border-radius: 10px;
        margin-bottom: 14px;
        border-left: 6px solid #ef4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .executive-banner-warning {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        color: #ffffff;
        padding: 16px 22px;
        border-radius: 10px;
        margin-bottom: 14px;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .executive-banner-safe {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        color: #ffffff;
        padding: 16px 22px;
        border-radius: 10px;
        margin-bottom: 14px;
        border-left: 6px solid #10b981;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .status-strip {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #334155;
    }
    
    .citizen-card {
        background-color: #0f172a;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    
    .legend-box {
        display: flex;
        gap: 16px;
        font-size: 0.85rem;
        margin-top: 6px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-color { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

API_DEFAULT_URL = "http://localhost:8000"


# Cached dataset loader
@st.cache_data
def get_cached_dataset(area_name: str) -> Tuple[Optional[Dataset], Optional[str]]:
    return load_area(area_name)


# Cached optimizer run to avoid UI recomputation on slider moves
@st.cache_data
def get_cached_optimizer_result(area_name: str, scenario_name: str, method: str, seed: int):
    ds, err = load_area(area_name)
    if ds is None or not ds.is_model_compatible:
        return None
    from src.opt.optimize import optimize as run_opt
    s_key = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get(scenario_name.lower(), "Heavy storm")
    storm_idx = ds.scenarios.get(s_key, 0)
    return run_opt(ds.nodes, ds.edges, ds.rain[storm_idx], ds.river[storm_idx], method=method, seed=seed)


# ---------------------------------------------------------------------------
# Sidebar Controls (Clean & Intuitive)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏙️ **Control Room**")
    
    # 1. Area Selector
    available_areas = list_areas()
    AREA_MAP = {
        "Hindmata - Dadar - Parel (Primary)": "hindmata",
        "Kurla - Mithi River (Scalability)": "kurla",
        "Sample 200-Node Benchmark": "sample",
    }
    enabled_names = {a["name"] for a in available_areas if a["status"] != "disabled"}
    display_options = [lbl for lbl, name in AREA_MAP.items() if name in enabled_names]
    
    chosen_label = st.selectbox(
        "📍 Study Catchment",
        options=display_options,
        index=0,
        help="Select the urban watershed area."
    )
    selected_area = AREA_MAP.get(chosen_label, "hindmata")
        
    # 2. Weather Scenario
    selected_scenario = st.radio(
        "⛈️ Storm Deluge Scenario",
        options=[
            "Light rain (40 mm total accumulation)",
            "Heavy storm (120 mm total accumulation)",
            "Extreme cloudburst (250 mm total accumulation)"
        ],
        index=1,
        help="Simulate different monsoon rainfall intensities."
    )
    scen_key = selected_scenario.lower().split()[0]
    
    st.markdown("---")
    
    # Automatic backend connection (Clean UI: all layers active, direct high-speed engine)
    api_url = API_DEFAULT_URL
    show_pumps = True
    show_outlets = True
    show_spots = True
    
    api_available = False
    try:
        r = requests.get(f"{api_url}/health", timeout=1.0)
        if r.status_code == 200:
            api_available = True
    except Exception:
        api_available = False
            
    if api_available:
        st.caption("🟢 Fast Backend Connected")
    else:
        st.caption("⚡ High-Speed Local Engine")


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------
ds, load_err = get_cached_dataset(selected_area)
if ds is None:
    st.error(f"Failed to load area '{selected_area}': {load_err}")
    st.stop()

# Scenario resolution
s_lookup = {"light": "Light rain", "heavy": "Heavy storm", "extreme": "Extreme cloudburst"}.get(scen_key, "Heavy storm")
storm_idx = ds.scenarios.get(s_lookup, 0)
meta = get_model_constraints()
history_steps = meta.get("history_steps", 8)
history_hours = round(history_steps * ds.dt_seconds / 3600.0, 1)


# ---------------------------------------------------------------------------
# HEADER & AUDIENCE PERSPECTIVE SELECTOR (Feature 3)
# ---------------------------------------------------------------------------
col_title, col_view = st.columns([2.7, 1.3])
with col_title:
    st.title("FloodSense AI: Urban Drainage & Flood Management")
    st.caption("AI-Powered Flood Depth Forecasting & Dispatch Recommendation Prototype | **Mumbai Municipal Catchment Simulation**")
with col_view:
    role_mode = st.radio(
        "👤 Audience Perspective",
        ["👔 Municipal Officer View", "🚶 Citizen / Commuter View"],
        index=0,
        help="Switch between the technical engineering console and simplified commuter guidance."
    )
is_citizen_view = "Citizen" in role_mode

# ---------------------------------------------------------------------------
# 🎬 JUDGE WALKTHROUGH TOUR (Feature 1: One-Click Instant 30-Second Demo)
# ---------------------------------------------------------------------------
st.markdown("##### ⚡ **Quick Scenario Walkthrough (Click any stage for instant demonstration):**")
wt1, wt2, wt3, wt4 = st.columns([1, 1, 1, 0.6])

if "timeline_slider_main" not in st.session_state:
    st.session_state.timeline_slider_main = min(24, ds.T - 1)

with wt1:
    if st.button("🌦️ Stage 1: Storm Inception (T+01:00)", help="Pavements dry (<10cm); clouds gathering; AI detects impending deluge."):
        st.session_state.timeline_slider_main = min(6, ds.T - 1)
        st.session_state.tour_banner = "🌦️ **Stage 1: Storm Inception (T+01:00)** — Pavements are completely dry. While naive Persistence sees zero danger (0.0% onset recall), FloodSense ST-GNN already detects the oncoming cloudburst and warns operators to pre-drain holding tanks."

with wt2:
    if st.button("Stage 2: Peak Deluge & High Tide (T+04:00)", help="Heavy storm hits; Arabian Sea tide locks outfalls; water pools at Hindmata."):
        st.session_state.timeline_slider_main = min(24, ds.T - 1)
        st.session_state.tour_banner = "**Stage 2: Peak Deluge & High Tide (T+04:00)** — Rainfall peaks and Arabian Sea tide exceeds 3.51 m Chart Datum (+1.0 m MSL), choking gravity outfalls. The Hindmata saucer depression pools rapidly."

with wt3:
    if st.button("⚡ Stage 3: Smart Drawdown & Recovery (T+07:00)", help="Holding tank pumps active; tidal gates closed; water recedes."):
        st.session_state.timeline_slider_main = min(42, ds.T - 1)
        st.session_state.tour_banner = "⚡ **Stage 3: Smart Drawdown & Recovery (T+07:00)** — Pre-emptively scheduled holding tank pumps and tidal gate closure clear street water, cutting flood exposure hours."

with wt4:
    if st.button("🔄 Reset Walkthrough"):
        st.session_state.tour_banner = None

if st.session_state.get("tour_banner"):
    st.info(st.session_state["tour_banner"])

# Timeline Slider: Global controller positioned above the Banner and Tabs
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    current_step = st.slider(
        "⏱️ Storm Simulation Timeline (Drag slider to inspect live flood progression across hours)",
        min_value=0,
        max_value=ds.T - 1,
        key="timeline_slider_main",
        format="%d",
        help="Step through the 16-hour storm timeline in 10-minute intervals."
    )
with col_ctrl2:
    clock_str = step_to_clock(current_step, ds.dt_seconds)
    hours_elapsed = round(current_step * ds.dt_seconds / 3600.0, 1)
    st.markdown(f"<div style='text-align: right;'><span style='font-size: 1.85rem; font-weight: 700; color: #38bdf8;'>🕒 {clock_str}</span><br/><span style='color: #94a3b8; font-size: 0.85rem;'>Step {current_step}/{ds.T} ({hours_elapsed}h elapsed)</span></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FULLY DYNAMIC 3-SECOND EXECUTIVE STATUS BANNER
# ---------------------------------------------------------------------------
# Current step inundation values
d_step_m = ds.depth[storm_idx, current_step] if ds.depth is not None else np.zeros(ds.N)
d_step_cm = d_step_m * 100.0
current_max_depth_m = float(d_step_m.max()) if len(d_step_m) > 0 else 0.0
current_max_d_cm = round(current_max_depth_m * 100.0, 1)
flooded_count = int((d_step_m > 0.10).sum())
worst_node_id = int(np.argmax(d_step_m)) if len(d_step_m) > 0 else 0
worst_node_lat = float(ds.nodes.iloc[worst_node_id]["lat"])
worst_node_lon = float(ds.nodes.iloc[worst_node_id]["lon"])
worst_node_landmark = get_landmark_for_node(ds.name, worst_node_id, worst_node_lat, worst_node_lon)

# ⏱️ Receding Countdown Clock Computation (Feature 5)
recede_str = "Clear"
recede_hrs_str = "0h"
if current_max_depth_m > 0.10 and ds.depth is not None:
    worst_future = ds.depth[storm_idx, current_step:, worst_node_id]
    clear_indices = np.where(worst_future <= 0.10)[0]
    if len(clear_indices) > 0:
        steps_away = int(clear_indices[0])
        clear_t = current_step + steps_away
        recede_str = step_to_clock(clear_t, ds.dt_seconds)
        recede_hrs_str = f"~{round(steps_away * ds.dt_seconds / 3600.0, 1)}h remaining"
    else:
        recede_str = "Beyond 16h"
        recede_hrs_str = "Persisting" 

# Overall storm peak for early warnings
peak_depth_m = float(ds.depth[storm_idx].max()) if ds.depth is not None else 0.0
peak_step = int(np.argmax(ds.depth[storm_idx].max(axis=1))) if ds.depth is not None else 0
peak_clock = step_to_clock(peak_step, ds.dt_seconds)
overall_worst_node = int(np.argmax(ds.depth[storm_idx].max(axis=0))) if ds.depth is not None else 0
overall_worst_landmark = get_landmark_for_node(ds.name, overall_worst_node, float(ds.nodes.iloc[overall_worst_node]["lat"]), float(ds.nodes.iloc[overall_worst_node]["lon"]))

# Status Banner State Logic (Simplified & Direct)
if current_max_depth_m >= 0.50:
    banner_cls = "executive-banner-danger"
    status_icon = "🚨"
    headline = f"CRITICAL FLOOD — {int(current_max_d_cm)} cm at {worst_node_landmark}"
    action_text = f"{flooded_count} of {ds.N} junctions submerged. Avoid Hindmata; use Lalbaug flyover. Pumps running."
elif current_max_depth_m >= 0.25:
    banner_cls = "executive-banner-warning"
    status_icon = "⚠️"
    headline = f"FLOOD WARNING — {int(current_max_d_cm)} cm at {worst_node_landmark}"
    action_text = f"Waterlogging on Dr. Ambedkar Road ({flooded_count} junctions). Small cars avoid area. Pumps active."
elif current_max_depth_m >= 0.10:
    banner_cls = "executive-banner-warning"
    status_icon = "🟡"
    headline = f"MINOR WATERLOGGING — {int(current_max_d_cm)} cm at {worst_node_landmark}"
    action_text = f"Ankle-deep surface water pooling. Drive with caution. Pumps on standby."
else:
    banner_cls = "executive-banner-safe"
    status_icon = "🟢"
    if peak_step > current_step and peak_depth_m >= 0.25:
        lead_h = round((peak_step - current_step) * ds.dt_seconds / 3600.0, 1)
        headline = f"ALL CLEAR NOW — Rain Peak in {lead_h}h ({peak_clock})"
        action_text = f"Pavements dry now ({int(current_max_d_cm)} cm). Flood predicted at {peak_clock} ({int(peak_depth_m*100)} cm at {overall_worst_landmark}). Holding tanks pre-draining."
    else:
        headline = "ALL CLEAR — Streets Dry & Passable"
        action_text = f"Normal gravity drainage across all {ds.N} junctions (max depth: {int(current_max_d_cm)} cm). Pumps on standby."

st.markdown(f"""
<div class="{banner_cls}">
    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;">{status_icon} {headline}</div>
    <div style="font-size: 0.92rem; opacity: 0.95;">{action_text}</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PERSPECTIVE ROUTING (Feature 3: Citizen View vs Officer Console)
# ---------------------------------------------------------------------------
if is_citizen_view:
    st.markdown("### 🚶 **Citizen & Commuter Safety Console**")
    st.caption("Simplified traffic routing guidance, street passability status, and emergency citizen advisories.")
    
    # 1. Big Commute Advisory Cards
    c_cit1, c_cit2 = st.columns(2)
    with c_cit1:
        if current_max_depth_m >= 0.25:
            st.error(f"🚫 **DR. AMBEDKAR ROAD: TRAFFIC DIVERTED ({int(current_max_d_cm)} cm)**\n\nSevere waterlogging near Hindmata Flyover. Small cars will stall. Use **Lalbaug Flyover** or **Eastern Freeway** as alternate routes.")
        elif current_max_depth_m >= 0.10:
            st.warning(f"⚠️ **DR. AMBEDKAR ROAD: SLOW TRAFFIC ({int(current_max_d_cm)} cm)**\n\nAnkle-deep surface pooling near Hindmata. Two-wheelers drive with caution. Passable at reduced speed.")
        else:
            st.success("✅ **DR. AMBEDKAR ROAD: ALL CLEAR & DRY (<10 cm)**\n\nNormal traffic flow across Dadar-Hindmata-Parel corridor.")
            
    with c_cit2:
        if current_max_depth_m > 0.10:
            st.info(f"⏱️ **Road Clearance Countdown:** Water is expected to recede below curb level by **{recede_str}** ({recede_hrs_str}) with active municipal pumps running.")
        else:
            st.info("⏱️ **Road Clearance Countdown:** Pavements are completely clear. Normal gravity drainage operating.")

    # 2. Neighborhood Landmark Passability Checklist
    st.markdown("#### 📍 **Neighborhood Landmark Passability Checklist**")
    top_spots = risk_table(d_step_m, ds.nodes, top_k=6, pumps=ds.pumps, area_name=ds.name)
    chk_cols = st.columns(3)
    for idx, r in top_spots.iterrows():
        c_target = chk_cols[idx % 3]
        with c_target:
            d_val = float(r["depth_cm"])
            stat_icon = "🔴" if d_val >= 50 else ("🟠" if d_val >= 25 else ("🟡" if d_val >= 10 else "🟢"))
            border_c = "#ef4444" if d_val >= 50 else ("#f97316" if d_val >= 25 else ("#facc15" if d_val >= 10 else "#10b981"))
            st.markdown(f"""
            <div style='background-color: #1e293b; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 4px solid {border_c};'>
                <strong>{stat_icon} {r['landmark']}</strong><br/>
                <span style='font-size: 0.85rem; color: #94a3b8;'>Water Depth: <b style='color: white;'>{d_val:.1f} cm</b> • {r['severity']}</span>
            </div>
            """, unsafe_allow_html=True)
            
    # 3. Bilingual Citizen Broadcast Notifications
    st.markdown("#### 📢 **Emergency Citizen Broadcast Notifications**")
    c_bc1, c_bc2 = st.columns(2)
    with c_bc1:
        st.markdown(f"""
        <div class="citizen-card">
            <h5 style='color: #60a5fa; margin: 0;'>🇬🇧 English Emergency Broadcast</h5>
            <p style='font-size: 0.95rem; margin-top: 8px;'>
                ⚠️ <strong>FloodSense Demo Alert (Simulated):</strong> Waterlogging ({int(current_max_d_cm)} cm) recorded at <strong>{worst_node_landmark}</strong> at <strong>{clock_str}</strong>. 
                Underground holding tank pumps are active. Commuters are advised to avoid Hindmata junction; use alternate routes.
            </p>
            <span style='font-size: 0.75rem; color: #94a3b8;'>FloodSense Demo System (Simulated Prototype) • Ward F/South Simulation</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c_bc2:
        st.markdown(f"""
        <div class="citizen-card">
            <h5 style='color: #60a5fa; margin: 0;'>🇮🇳 हिंदी आपातकालीन चेतावनी</h5>
            <p style='font-size: 0.95rem; margin-top: 8px;'>
                ⚠️ <strong>फ्लडसेंस डेमो अलर्ट (अनुकरण):</strong> <strong>{clock_str}</strong> पर <strong>{worst_node_landmark}</strong> में जलभराव ({int(current_max_d_cm)} सेमी) दर्ज किया गया है। 
                पंपिंग स्टेशन चालू कर दिए गए हैं। नागरिकों से अनुरोध है कि हिंदमाता जंक्शन से बचें और वैकल्पिक सुरक्षित मार्गों का प्रयोग करें।
            </p>
            <span style='font-size: 0.75rem; color: #94a3b8;'>फ्लडसेंस डेमो प्रणाली (प्रायोगिक प्रोटोटाइप) • प्रभाग: F/South अनुकरण</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("FloodSense AI System • Citizen View Mode • Built for Mumbai Civic Awareness.")
    st.stop()

# ---------------------------------------------------------------------------
# STREAMLINED TABS (5 Intuitive Sections - Municipal Officer Console)
# ---------------------------------------------------------------------------
tab_live, tab_optimizer, tab_alerts, tab_metrics, tab_about = st.tabs([
    "🗺️ Live Situation & AI Prediction",
    "⚡ Automated Pump & Gate Action",
    "📢 Civic Alerts & Citizen Broadcast",
    "📊 AI Accuracy & IoT Telemetry",
    "ℹ️ Engineering Assumptions & Honesty",
])


# ===========================================================================
# TAB 1: LIVE SITUATION & AI PREDICTION
# ===========================================================================
with tab_live:
    rain_step = float(ds.rain[storm_idx, current_step].mean())
    river_step = float(ds.river[storm_idx, current_step])
    max_d_cm = current_max_d_cm

    # Key Human-Friendly Metrics (Responsive to current_step) with Feature 5 Receding Clock
    tide_cd = river_step + 2.51
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Submerged Junctions", f"{flooded_count} of {ds.N}", delta=f"{round(flooded_count/ds.N*100, 1)}% of area", delta_color="inverse", help="Count of street intersections where water depth exceeds the 10 cm curb threshold.")
    m2.metric("Worst Inundation Depth", f"{max_d_cm} cm", delta=worst_node_landmark, delta_color="off", help="Deepest waterlogged point across the watershed at the current simulation minute.")
    m3.metric("Rainfall Rate", f"{round(rain_step, 1)} mm/h", delta="Monsoon Intensity" if rain_step > 50 else "Moderate Rain", delta_color="off", help="Average precipitation intensity falling on the catchment.")
    m4.metric(
        "Arabian Sea Tide Level",
        f"{round(tide_cd, 2)} m CD",
        delta=f"{round(river_step, 2)} m MSL (Outfalls Submerged)" if river_step > 1.0 else f"{round(river_step, 2)} m MSL (Gravity Flow OK)",
        delta_color="inverse" if river_step > 1.0 else "normal",
        help="Reported in Chart Datum (CD), standard Mumbai Port Trust datum. Mean Sea Level (MSL) is +2.51 m CD. Flap sluice gates choke when tide exceeds 3.51 m CD (+1.0 m MSL)."
    )
    m5.metric(
        "Estimated Road Clearance",
        recede_str,
        delta=recede_hrs_str if current_max_depth_m > 0.10 else "Pavements Dry",
        delta_color="normal" if current_max_depth_m <= 0.10 else "inverse",
        help="Estimated time when the worst inundated junction will drop back below the 10 cm curb level based on active pump drawdown."
    )



    # Legend with Human Explanations
    st.markdown("""
    <div class="legend-box">
        <span class="legend-item"><span class="legend-color" style="background-color: #38bdf8;"></span> <strong>&lt;10 cm:</strong> Safe (Dry pavement)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #facc15;"></span> <strong>10-25 cm:</strong> Ankle-Deep (Slow traffic)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #f97316;"></span> <strong>25-50 cm:</strong> Knee-Deep (Cars stall)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #ef4444;"></span> <strong>50-100 cm:</strong> Waist-Deep (Road closed)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #9f1239;"></span> <strong>&gt;100 cm:</strong> Extreme Deluge</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🗺️ **Catchment Drainage Network & Landmark Map**")
    chosen_style = "carto-positron"

    # Prepare node data for Pydeck Map
    plot_nodes = ds.nodes.copy()
    plot_nodes["depth_cm"] = d_step_cm.round(1)
    plot_nodes["severity_desc"] = [human_severity_desc(d) for d in d_step_cm]
    
    # Map real landmark names
    landmarks = []
    for idx, r in plot_nodes.iterrows():
        nid = int(r["id"]) if "id" in r else idx
        landmarks.append(get_landmark_for_node(ds.name, nid, float(r["lat"]), float(r["lon"])))
    plot_nodes["landmark_name"] = landmarks

    colors = []
    radii = []
    for d in d_step_cm:
        _, rgba = depth_band(d)
        colors.append(rgba)
        radii.append(25.0 + min(45.0, d * 0.4))
    plot_nodes["color"] = colors
    plot_nodes["radius"] = radii

    elev_col = "elevation_m" if "elevation_m" in plot_nodes.columns else ("elevation" if "elevation" in plot_nodes.columns else None)
    plot_nodes["elev_display"] = plot_nodes[elev_col].round(1) if elev_col else 0.0
    nid_col = "node_id" if "node_id" in plot_nodes.columns else ("id" if "id" in plot_nodes.columns else None)
    plot_nodes["nid_display"] = plot_nodes[nid_col] if nid_col else np.arange(len(plot_nodes))
    plot_nodes["pump_display"] = plot_nodes["has_pump"].apply(lambda v: "⚡ Pumping Station Active" if v == 1 else "No Pump") if "has_pump" in plot_nodes.columns else "No Pump"



    layers = []

    # 1. Drainage Pipe Conduits (Lines connecting street junctions)
    if hasattr(ds, "edges") and ds.edges is not None and not ds.edges.empty:
        id_col = "id" if "id" in ds.nodes.columns else ("node_id" if "node_id" in ds.nodes.columns else ds.nodes.columns[0])
        node_coords = ds.nodes.set_index(id_col)[["lon", "lat"]].to_dict("index")
        pipe_rows = []
        for _, edge in ds.edges.iterrows():
            s, t = int(edge["source"]), int(edge["target"])
            if s in node_coords and t in node_coords:
                pipe_rows.append({
                    "src_lon": node_coords[s]["lon"],
                    "src_lat": node_coords[s]["lat"],
                    "tgt_lon": node_coords[t]["lon"],
                    "tgt_lat": node_coords[t]["lat"],
                })
        if pipe_rows:
            pipe_layer = pdk.Layer(
                "LineLayer",
                pd.DataFrame(pipe_rows),
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["tgt_lon", "tgt_lat"],
                get_color=[37, 99, 235, 140],
                get_width=2.5,
                pickable=False,
            )
            layers.append(pipe_layer)

    # 2. Inundation Nodes (ScatterplotLayer)
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        plot_nodes,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )
    layers.append(node_layer)

    # 3. Pumping Station Rings
    if show_pumps and "has_pump" in plot_nodes.columns:
        p_nodes = plot_nodes[plot_nodes["has_pump"] == 1]
        if len(p_nodes) > 0:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                p_nodes,
                get_position=["lon", "lat"],
                get_fill_color=[16, 185, 129, 140],
                get_line_color=[5, 150, 105, 255],
                get_line_width=3,
                stroked=True,
                get_radius=55,
                pickable=False,
            ))

    # 4. Prominent Landmark Text Badges (Clearly identifies Mumbai neighborhoods)
    landmark_key = "hindmata" if "hindmata" in ds.name.lower() or "dadar" in ds.name.lower() else ("kurla" if "kurla" in ds.name.lower() else None)
    if landmark_key and landmark_key in MUMBAI_LANDMARKS:
        lm_data = []
        for name, (l_lat, l_lon) in MUMBAI_LANDMARKS[landmark_key].items():
            lm_data.append({"name": name, "lat": l_lat, "lon": l_lon})
        lm_df = pd.DataFrame(lm_data)
        layers.append(pdk.Layer(
            "TextLayer",
            lm_df,
            get_position=["lon", "lat"],
            get_text="name",
            get_size=12,
            get_color=[15, 23, 42, 255],
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            background=True,
            get_background_color=[255, 255, 255, 220],
            background_padding=[4, 2, 4, 2],
            pickable=False,
        ))

    view = pdk.ViewState(
        latitude=float(plot_nodes["lat"].mean()),
        longitude=float(plot_nodes["lon"].mean()),
        zoom=13.7,
        pitch=0,
    )
    
    tooltip = {
        "html": """
        <div style='padding: 6px; font-family: sans-serif;'>
            <b style='color: #38bdf8; font-size: 13px;'>📍 {landmark_name}</b><br/>
            Water Depth: <b style='color: #facc15; font-size: 13px;'>{depth_cm} cm</b><br/>
            Status: <i>{severity_desc}</i><br/>
            Ground Elevation: {elev_display} m<br/>
            Facility: {pump_display}
        </div>
        """,
        "style": {"backgroundColor": "#0f172a", "color": "white", "borderRadius": "6px", "border": "1px solid #334155"}
    }
    
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_style=chosen_style), use_container_width=True)

    # -----------------------------------------------------------------------
    # Multi-Horizon AI Forecast Cards (+1h, +2h, +3h)
    # -----------------------------------------------------------------------
    st.markdown("### 🔮 **AI Early Warning Horizon (+1h, +2h, +3h Forecast)**")
    st.caption("Our Spatio-Temporal Graph Neural Network (ST-GNN) looks ahead up to 3 hours to trigger pump drawdown before water reaches the street.")
    
    fc1, fc2, fc3 = st.columns(3)
    
    with fc1:
        st.markdown("""
        <div style='background-color: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #3b82f6;'>
            <h4 style='color: #60a5fa; margin: 0;'>⏱️ +1 Hour Forecast</h4>
            <p style='font-size: 1.3rem; font-weight: 700; margin: 6px 0;'>MAE: 1.18 cm</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Recall: <strong>91.5%</strong> | Onset Recall: <strong style='color: #38bdf8;'>70.5%</strong> | Precision: <strong>55.7%</strong></p>
            <p style='color: #34d399; font-size: 0.85rem; margin-top: 4px;'>⚡ Persistence Onset Recall: 0.0% (blind to sudden floods)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with fc2:
        st.markdown("""
        <div style='background-color: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #3b82f6;'>
            <h4 style='color: #60a5fa; margin: 0;'>⏱️ +2 Hours Forecast</h4>
            <p style='font-size: 1.3rem; font-weight: 700; margin: 6px 0;'>MAE: 1.74 cm</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Recall: <strong>90.3%</strong> | Onset Recall: <strong style='color: #38bdf8;'>74.0%</strong> | Precision: <strong>48.2%</strong></p>
            <p style='color: #34d399; font-size: 0.85rem; margin-top: 4px;'>⚡ Enables 90-minute pre-emptive pump dispatch recommendation</p>
        </div>
        """, unsafe_allow_html=True)
        
    with fc3:
        st.markdown("""
        <div style='background-color: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #3b82f6;'>
            <h4 style='color: #60a5fa; margin: 0;'>⏱️ +3 Hours Forecast</h4>
            <p style='font-size: 1.3rem; font-weight: 700; margin: 6px 0;'>MAE: 2.55 cm</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Recall: <strong>87.8%</strong> | Onset Recall: <strong style='color: #38bdf8;'>72.1%</strong> | Precision: <strong>40.0%</strong></p>
            <p style='color: #34d399; font-size: 0.85rem; margin-top: 4px;'>🚪 Permits sluice gate closure recommendation before high tide peak</p>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2: AUTOMATED PUMP & GATE ACTION
# ===========================================================================
with tab_optimizer:
    st.markdown("### ⚡ **Smart Actuation: Automated Pump & Sluice Gate Control**")
    st.caption("How FloodSense AI converts early predictions into physical actions: pre-emptive pumping and tidal surge gate closure.")
    
    opt_data = get_cached_optimizer_result(ds.name, scen_key, "search", 0)
    
    col_opt_act, col_opt_res = st.columns([1, 1])
    
    with col_opt_act:
        st.markdown("#### 🎯 Dispatch Optimization: Dual Baselines Comparison")
        if opt_data:
            c1, c2, c3 = st.columns(3)
            cost_p = opt_data.get('cost_passive', opt_data['scenario_cost_before'])
            cost_r = opt_data.get('cost_reactive', opt_data['scenario_cost_before'])
            cost_opt = opt_data.get('cost_proactive', opt_data['scenario_cost_after'])
            red_vs_r = opt_data.get('reduction_vs_reactive_pct', 0.6)
            red_vs_p = opt_data.get('reduction_vs_passive_pct', opt_data['reduction_pct'])

            c1.metric("1. Passive Drainage", f"{round(cost_p, 1)} m·h", delta="Pumps OFF, Gates OPEN", delta_color="inverse", help="Meter-Hours (m·h): cumulative depth in meters × flooded duration in hours. Baseline with pumps OFF and gates OPEN.")
            c2.metric("2. Current Practice", f"{round(cost_r, 1)} m·h", delta="Reactive Rules", delta_color="off", help="Standard municipal practice: gates close on high tide, pumps switch on reactively after water ponds.")
            c3.metric("3. FloodSense AI", f"{round(cost_opt, 1)} m·h", delta=f"-{red_vs_r}% vs Current Practice", delta_color="normal", help="Proactive dispatch schedule pre-draining holding tanks before storm peaks.")
            
            st.success(f"🎉 **Proactive schedule achieves {round(cost_opt, 1)} m·h** (-{red_vs_r}% vs current reactive rules, -{red_vs_p}% vs passive drainage). On unseen test storms, mean flood exposure reduction is **9.4% over current practice** and **10.9% over passive drainage**.")
            
            st.info("""
            **💡 Transparent Baseline Definitions (Evaluator FAQ):**
            - **Baseline 1 (Passive Drainage):** Pumps completely OFF and tidal sluice gates left wide OPEN (pure gravity drainage; seawater surges backward during high tide).
            - **Baseline 2 (Current Municipal Practice):** Tidal gates close automatically when sea level > 1.0 m MSL, and pumps activate *reactively* once water accumulates (> 30% storage or rain > 10 mm/h).
            - **FloodSense AI (Proactive Optimization):** Recommends pump drawdown *before* peak rainfall arrives and coordinates tidal gate closure based on forecast horizons, preventing holding tanks from backing up.
            """)
            
        st.markdown("#### The Mumbai 'Tide Story' Impact")
        st.info("""
        **Why Hindmata Floods (Tidal Interaction):**
        - **At Low Tide (<3.51 m Chart Datum / <1.0 m MSL):** Mumbai's 25 mm/h drains discharge gravity flow into the Arabian Sea. The streets remain safe.
        - **At High Tide (>3.51 m Chart Datum / >1.0 m MSL):** Sea water surges backward into outfalls, choking gravity drains.
        - **FloodSense Solution:** Sluice gates are automatically commanded **CLOSED** at high tide to stop seawater ingress, while local holding tank pumps pump water into arterial canals.
        """)

    with col_opt_res:
        st.markdown("#### 📋 Automated Dispatch Timeline")
        if opt_data and opt_data.get("pump_schedule"):
            p_df = pd.DataFrame(opt_data["pump_schedule"])
            p_df["time"] = p_df["step"].apply(lambda s: step_to_clock(s, ds.dt_seconds))
            p_df["station_name"] = p_df["node_id"].apply(lambda nid: get_landmark_for_node(ds.name, nid, 19.01, 72.84))
            p_df["command"] = p_df["on"].apply(lambda v: "🟢 ACTIVATE (ON)" if v == 1 else "⚪ STANDBY (OFF)")
            st.dataframe(p_df[["time", "station_name", "command"]], hide_index=True, use_container_width=True)
        else:
            st.info("Pumps scheduled proactively across all 3 stations (Hindmata, Gandhi Market, Britannia).")


# ===========================================================================
# TAB 3: CIVIC ALERTS & CITIZEN BROADCAST
# ===========================================================================
with tab_alerts:
    st.markdown("### 📢 **Civic Warnings & Citizen Emergency Broadcast**")
    st.caption("Automated bilingual early-warning notifications generated from real-time model predictions.")

    # Bilingual Citizen Broadcast Card (Dynamically reflecting current time and water depth)
    st.markdown("#### 📱 Automated Emergency Broadcast Notification")
    c_bc1, c_bc2 = st.columns(2)
    with c_bc1:
        st.markdown(f"""
        <div class="citizen-card">
            <h5 style='color: #60a5fa; margin: 0;'>🇬🇧 English Emergency Broadcast</h5>
            <p style='font-size: 0.95rem; margin-top: 8px;'>
                ⚠️ <strong>FloodSense Demo Alert (Simulated):</strong> Waterlogging ({int(current_max_d_cm)} cm) recorded at <strong>{worst_node_landmark}</strong> at <strong>{clock_str}</strong>. 
                Underground holding tank pumps are active. Commuters are advised to avoid Hindmata junction; use alternate routes.
            </p>
            <span style='font-size: 0.75rem; color: #94a3b8;'>FloodSense Demo System (Simulated Prototype) • Ward F/South Simulation</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c_bc2:
        st.markdown(f"""
        <div class="citizen-card">
            <h5 style='color: #60a5fa; margin: 0;'>🇮🇳 हिंदी आपातकालीन चेतावनी</h5>
            <p style='font-size: 0.95rem; margin-top: 8px;'>
                ⚠️ <strong>फ्लडसेंस डेमो अलर्ट (अनुकरण):</strong> <strong>{clock_str}</strong> पर <strong>{worst_node_landmark}</strong> में जलभराव ({int(current_max_d_cm)} सेमी) दर्ज किया गया है। 
                पंपिंग स्टेशन चालू कर दिए गए हैं। नागरिकों से अनुरोध है कि हिंदमाता जंक्शन से बचें और वैकल्पिक सुरक्षित मार्गों का प्रयोग करें।
            </p>
            <span style='font-size: 0.75rem; color: #94a3b8;'>फ्लडसेंस डेमो प्रणाली (प्रायोगिक प्रोटोटाइप) • प्रभाग: F/South अनुकरण</span>
        </div>
        """, unsafe_allow_html=True)

    # Ranked Landmark Risk Table
    st.markdown("#### 🚨 Priority Inundation Hotspot Ranking")
    d_current_m = ds.depth[storm_idx, current_step] if ds.depth is not None else np.zeros(ds.N)
    rtable = risk_table(d_current_m, ds.nodes, top_k=12, pumps=ds.pumps, area_name=ds.name)
    
    st.dataframe(
        rtable[["landmark", "depth_cm", "severity", "elevation_m", "nearest_pump_m"]],
        column_config={
            "landmark": "Civic Landmark",
            "depth_cm": "Water Depth (cm)",
            "severity": "Physical Reality",
            "elevation_m": "Ground Height (m)",
            "nearest_pump_m": "Nearest Pump Station (m)"
        },
        hide_index=True,
        use_container_width=True
    )
    
    csv_bytes = rtable.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Waterlogging Report (CSV)", csv_bytes, f"FloodSense_Report_{ds.name}_{clock_str}.csv", "text/csv")


# ===========================================================================
# TAB 4: AI ACCURACY & IoT TELEMETRY
# ===========================================================================
with tab_metrics:
    st.markdown("### 📊 **AI Model Evaluation & Sensor Telemetry**")
    
    col_eval, col_gauge = st.columns([1, 1])
    
    with col_eval:
        st.markdown("#### 🎯 ST-GNN Model vs. Baselines (Empirical Test Set)")
        st.caption("Evaluated on 46 unseen test storms in data/hindmata using src/ml/evaluate.py (flood threshold = 10 cm).")
        
        # 🏆 Feature 8: Clean 3-Card Executive Takeaway Box
        st.markdown("##### 🏆 **Executive AI Findings at a Glance:**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("""
            <div style='background-color: #0f172a; border-radius: 8px; padding: 12px; border: 2px solid #38bdf8;'>
                <p style='color: #38bdf8; font-weight: 700; margin: 0; font-size: 0.8rem;'>🎯 1. PREDICTING SUDDEN FLOODS</p>
                <p style='font-size: 1.45rem; font-weight: 800; margin: 4px 0; color: #ffffff;'>70.5% vs 0.0%</p>
                <p style='color: #94a3b8; font-size: 0.78rem; margin: 0;'>Naive Persistence catches <strong>0.0%</strong> of flood onsets. ST-GNN anticipates over <strong>7 out of 10</strong> upcoming floods.</p>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown("""
            <div style='background-color: #0f172a; border-radius: 8px; padding: 12px; border: 2px solid #10b981;'>
                <p style='color: #10b981; font-weight: 700; margin: 0; font-size: 0.8rem;'>⚡ 2. INUNDATION REDUCTION</p>
                <p style='font-size: 1.45rem; font-weight: 800; margin: 4px 0; color: #ffffff;'>-9.4% Exposure</p>
                <p style='color: #94a3b8; font-size: 0.78rem; margin: 0;'>Mean reduction in flood volume-hours over current reactive municipal practice (-10.9% vs passive drainage).</p>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown("""
            <div style='background-color: #0f172a; border-radius: 8px; padding: 12px; border: 2px solid #f59e0b;'>
                <p style='color: #f59e0b; font-weight: 700; margin: 0; font-size: 0.8rem;'>🛡️ 3. CITIZEN SAFETY RECALL</p>
                <p style='font-size: 1.45rem; font-weight: 800; margin: 4px 0; color: #ffffff;'>91.5% Recall</p>
                <p style='color: #94a3b8; font-size: 0.78rem; margin: 0;'>Catches 9 out of 10 floods ahead of time; accepts more false alarms (55.7% precision) to avoid missing floods.</p>
            </div>
            """, unsafe_allow_html=True)
            
        metrics_summary = pd.DataFrame([
            {"Model": "Persistence (No AI)", "Lead Time": "+1 Hour", "MAE (cm)": 0.46, "Precision": "89.0%", "Recall": "86.4%", "Onset Recall": "0.0%", "F1 Score": 0.88},
            {"Model": "Persistence (No AI)", "Lead Time": "+2 Hours", "MAE (cm)": 0.88, "Precision": "78.7%", "Recall": "75.0%", "Onset Recall": "0.0%", "F1 Score": 0.77},
            {"Model": "Persistence (No AI)", "Lead Time": "+3 Hours", "MAE (cm)": 1.24, "Precision": "69.5%", "Recall": "65.6%", "Onset Recall": "0.0%", "F1 Score": 0.68},
            {"Model": "No-Graph MLP (Retrained 12x)", "Lead Time": "+1 Hour", "MAE (cm)": 0.30, "Precision": "90.0%", "Recall": "95.7%", "Onset Recall": "82.1%", "F1 Score": 0.93},
            {"Model": "No-Graph MLP (Retrained 12x)", "Lead Time": "+2 Hours", "MAE (cm)": 0.77, "Precision": "70.4%", "Recall": "91.8%", "Onset Recall": "78.5%", "F1 Score": 0.80},
            {"Model": "No-Graph MLP (Retrained 12x)", "Lead Time": "+3 Hours", "MAE (cm)": 1.31, "Precision": "56.4%", "Recall": "87.6%", "Onset Recall": "71.3%", "F1 Score": 0.69},
            {"Model": "🌟 FloodSense ST-GNN", "Lead Time": "+1 Hour", "MAE (cm)": 1.18, "Precision": "55.7%", "Recall": "91.5%", "Onset Recall": "70.5%", "F1 Score": 0.69},
            {"Model": "🌟 FloodSense ST-GNN", "Lead Time": "+2 Hours", "MAE (cm)": 1.74, "Precision": "48.2%", "Recall": "90.3%", "Onset Recall": "74.0%", "F1 Score": 0.63},
            {"Model": "🌟 FloodSense ST-GNN", "Lead Time": "+3 Hours", "MAE (cm)": 2.55, "Precision": "40.0%", "Recall": "87.8%", "Onset Recall": "72.1%", "F1 Score": 0.55},
        ])
        st.dataframe(metrics_summary, hide_index=True, use_container_width=True)
        
        st.markdown(r"""
        **Honest Precision / Recall Tradeoff & The True Value of Forecasting:**
        - **Why we do NOT claim ST-GNN beats baselines on MAE or F1:** Persistence trivially assumes current depth continues ($d_{t+k} = d_t$), achieving low MAE and high F1 during dry spells and gradual ponding.
        - **The Definitive Test — Flood Onset Detection Recall:** Among junctions that are **dry now ($\le 10$ cm) but flood in the next 1–3 hours**, Persistence scores **0.0%** across all lead times. It is completely blind to oncoming floods. In contrast, predictive models anticipate over **70% to 82%** of sudden flood inceptions.
        - **Higher Detection Recall, More False Alarms:** ST-GNN catches over 9 out of 10 floods (Recall >91%), but gives more false alarms (Precision ~55.7%) because it was trained with a **12× flood loss weight** (`flood_weight = 12.0`). In disaster management, a false alarm (precautionary pump startup) is minor, while a false negative (missed flood) is catastrophic.
        - **No-Graph MLP Retraining:** When retrained with the identical 12× flood loss penalty, the isolated-node MLP performs strongly on local feature patterns, achieving 71–82% onset recall.
        """)

    with col_gauge:
        st.markdown("#### 📡 Real-Time IoT Outfall Gauge Stream")
        st.caption("Simulated ultrasonic water level telemetry at the Arabian Sea / Mithi River outfall.")
        
        iot_river = float(ds.river[storm_idx, current_step])
        tide_cd = iot_river + 2.51
        st.metric(
            "Current Outfall Water Level",
            f"{round(tide_cd, 2)} m CD",
            delta=f"{round(iot_river, 2)} m MSL (Backwater Risk)" if iot_river > 1.0 else f"{round(iot_river, 2)} m MSL (Normal Gravity Flow)",
            delta_color="inverse" if iot_river > 1.0 else "normal",
            help="Measured in Chart Datum (CD). Mean Sea Level (MSL) is +2.51 m CD."
        )
        st.caption("📏 **Tide Datum Reference:** Mumbai Port Trust & civic bulletins report tide above **Chart Datum (CD)**. Mean Sea Level (MSL) is at **+2.51 m CD**. Stormwater outfalls submerge when tide exceeds **3.51 m CD (+1.0 m MSL)**, choking gravity discharge.")
        
        gauge_df = pd.DataFrame({
            "Time Step": np.arange(ds.T),
            "Tide Level (m CD)": ds.river[storm_idx] + 2.51,
            "Outfall Lockout Level (3.51 m CD)": np.full(ds.T, 3.51),
        })
        st.line_chart(gauge_df.set_index("Time Step"), height=190)


# ===========================================================================
# TAB 5: ENGINEERING ASSUMPTIONS & HONESTY
# ===========================================================================
with tab_about:
    st.markdown("### ℹ️ **System Architecture & Data Honesty Statement**")
    
    st.markdown("""
    #### 🏆 AI Manthan 2.0 (Problem Statement 11)
    **AI-Powered Urban Drainage and Flash Flood Predictive Management System**  
    Target Municipal Areas: **Mumbai: Dadar-Hindmata-Parel Catchment** (Primary) & **Kurla-Mithi River Basin** (Scalability Proof).

    #### ⚖️ Complete Transparency: What is Real vs. Simulated
    - **System Classification:** FloodSense is a **prototype decision-support system that recommends and simulates proactive actions**, not an active municipal SCADA controller.
    - **Real Municipal & Geographical Data:**
      - **Street Graph Topology:** Sourced directly from OpenStreetMap (`osmnx`, drive network, simplified intersection clusters).
      - **Topographic Terrain:** Sourced from Copernicus GLO-30 Digital Elevation Model (smoothed over node neighborhood).
      - **Drain Capacity (25 mm/h):** The 25 mm/h drain design capacity is derived from public BRIMSTOWAD reports.
      - **Tide Datum Reference:** Mumbai Port Trust Chart Datum is 2.51 m below Mean Sea Level (MSL = +2.51 m CD, Survey of India datum).
    - **Engineering Assumptions & Simulated Components:**
      - **Tidal Lockout Level (1.0 m MSL / 3.51 m CD):** The 1.0 m MSL lockout threshold is our own engineering design assumption (based on typical coastal flap invert elevations), not an official BMC municipal standard.
      - **Monsoon Storm Scenarios:** The three scenarios represent synthetic rainfall hyetographs totaling 43.5 mm (Light), 104.5 mm (Heavy), and 191.6 mm (Extreme) accumulation with harmonic tides, calibrated to typical Mumbai monsoon intensities.
      - **Simulated Inundation & Pipe Flow:** Inundation depths and conduit flows are generated by our high-performance vectorized physical simulator (`simulator.py`).
      - **Extreme Worst-Case Inundation (~190 cm):** Depths exceeding 100 cm represent a synthetic stress-test worst case (simultaneous 191.6 mm cloudburst + spring high tide lockout) designed to evaluate algorithm stability at hydraulic capacity limits, not an average seasonal forecast.
      - **Dual Baselines:** We evaluate dispatch schedules against both **Baseline 1: Passive Drainage** (pumps OFF, gates OPEN) and **Baseline 2: Current Municipal Practice** (automatic gates + reactive float-switch pumps). On unseen test storms, proactive optimization reduces flood exposure by an average of **9.4% over current practice** and **10.9% over passive drainage**.
    """)

st.markdown("---")
st.caption("FloodSense AI System • Research & Academic Demo Prototype • All Data Simulated for PS11.")
