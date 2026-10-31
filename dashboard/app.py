"""FloodSense AI: Urban Drainage & Flash Flood Predictive Management System
Person C Interactive Dashboard.
Supports API Mode (FastAPI at :8000) and Direct Mode (local imports).
All data is SIMULATED.
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
    build_recent,
    risk_table,
    nearest_node,
)
from src.sim.simulator import simulate, DT, RIVER_MAX

# ---------------------------------------------------------------------------
# 4.1 Shell & Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodSense | Dadar-Hindmata-Parel | SIMULATED DATA",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    .status-strip {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.82rem;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #334155;
    }
    .badge-fallback {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-model {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .legend-box {
        display: flex;
        gap: 12px;
        font-size: 0.8rem;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .legend-color { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 FloodSense: Urban Drainage & Flash Flood Management")
st.caption("AI-Powered Real-Time Spatio-Temporal Flood Prediction & Automated Sluice/Pump Dispatch | **ALL DATA IS SIMULATED**")

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
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Room")
    
    # 1. Area Selector
    available_areas = list_areas()
    area_options = [a["name"] for a in available_areas if a["status"] != "disabled"]
    disabled_areas = [a["name"] for a in available_areas if a["status"] == "disabled"]
    
    selected_area = st.selectbox(
        "Study Area",
        options=area_options,
        index=0,
        help="Select urban catchment area."
    )
    if disabled_areas:
        st.caption(f"🔒 Disabled areas: {', '.join(disabled_areas)} (files not yet provided)")
        
    # 2. Scenario Radio
    selected_scenario = st.radio(
        "Weather Scenario",
        options=["Light rain", "Heavy storm", "Extreme cloudburst"],
        index=1,
        help="Select rain intensity event."
    )
    scen_key = selected_scenario.lower().split()[0]
    
    # 3. Connection Mode Toggle
    mode = st.radio("Operating Mode", ["Auto (Try API -> Direct)", "Direct Import Only"], index=0)
    
    # Check API connectivity
    api_available = False
    api_url = API_DEFAULT_URL
    if "Auto" in mode:
        try:
            r = requests.get(f"{api_url}/health", timeout=1.5)
            if r.status_code == 200:
                api_available = True
        except Exception:
            api_available = False
            
    if not api_available and "Auto" in mode:
        st.warning("⚠️ FastAPI server (:8000) offline. Running in **Direct Import Mode**.")
    elif api_available:
        st.success("🟢 FastAPI Server Connected (:8000)")

    st.markdown("---")
    st.markdown("### 🗺️ Map Layers")
    show_pumps = st.checkbox("Show Pump Stations", value=True)
    show_outlets = st.checkbox("Show Outlets / Outfalls", value=True)
    show_spots = st.checkbox("Show Known Flood Hotspots", value=True)


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------
ds, load_err = get_cached_dataset(selected_area)
if ds is None:
    st.error(f"Failed to load area '{selected_area}': {load_err}")
    st.stop()

storm_idx = ds.scenarios.get(selected_scenario, 0)
meta = get_model_constraints()
history_steps = meta.get("history_steps", 8)
history_hours = round(history_steps * ds.dt_seconds / 3600.0, 1)

# Status Strip
model_status_badge = '<span class="badge-model">ST-GNN LOADED</span>' if ds.is_model_compatible else f'<span class="badge-fallback">MODEL INCOMPATIBLE ({ds.incompatible_reason})</span>'
st.markdown(f"""
<div class="status-strip">
    <span>📍 <strong>Area:</strong> {ds.name} (N={ds.N}, T={ds.T}, dt={int(ds.dt_seconds)}s)</span>
    <span>⚡ <strong>Status:</strong> {model_status_badge}</span>
    <span>⏱️ <strong>Lead Times:</strong> +1h, +2h, +3h</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------
tab_map, tab_forecast, tab_alerts, tab_optimizer, tab_iot, tab_metrics, tab_about = st.tabs([
    "🗺️ Interactive Flood Map",
    "🔮 Multi-Horizon AI Forecast",
    "🚨 Risk & Alerts",
    "⚡ Pump/Gate Optimizer",
    "📡 IoT Sensor Gauge",
    "📊 Model Performance",
    "ℹ️ About & Honesty",
])


# ---------------------------------------------------------------------------
# TAB 1: INTERACTIVE FLOOD MAP
# ---------------------------------------------------------------------------
with tab_map:
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        current_step = st.slider(
            "Timeline Navigation",
            min_value=0,
            max_value=ds.T - 1,
            value=min(24, ds.T - 1),
            format=f"%d",
            help="Step along the storm timeline."
        )
    with col_ctrl2:
        clock_str = step_to_clock(current_step, ds.dt_seconds)
        st.markdown(f"<h3 style='margin: 0; color: #38bdf8;'>{clock_str}</h3>", unsafe_allow_html=True)
        st.caption(f"Step {current_step} of {ds.T} ({round(current_step * ds.dt_seconds / 3600.0, 2)} h)")
        
    # Get current depths & rain
    d_step_m = ds.depth[storm_idx, current_step] if ds.depth is not None else np.zeros(ds.N)
    d_step_cm = d_step_m * 100.0
    rain_step = float(ds.rain[storm_idx, current_step].mean())
    river_step = float(ds.river[storm_idx, current_step])
    
    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    flooded_count = int((d_step_m > 0.10).sum())
    max_d_cm = round(float(d_step_cm.max()), 1)
    worst_node_id = int(np.argmax(d_step_m))
    
    m1.metric("Flooded Junctions (>10 cm)", f"{flooded_count} / {ds.N}", delta=f"{round(flooded_count/ds.N*100, 1)}% network", delta_color="inverse")
    m2.metric("Worst Inundation", f"{max_d_cm} cm", delta=f"Node #{worst_node_id}", delta_color="off")
    m3.metric("Rainfall Intensity", f"{round(rain_step, 1)} mm/h")
    m4.metric("River Level Gauge", f"{round(river_step, 2)} m", delta=f"Cap: {RIVER_MAX} m", delta_color="off")
    
    # Map Legend
    st.markdown("""
    <div class="legend-box">
        <span class="legend-item"><span class="legend-color" style="background-color: #38bdf8;"></span> &lt;10cm (Safe)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #facc15;"></span> 10-30cm (Minor)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #f97316;"></span> 30-60cm (Moderate)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #ef4444;"></span> 60-100cm (Severe)</span>
        <span class="legend-item"><span class="legend-color" style="background-color: #9f1239;"></span> &gt;100cm (Extreme)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Build Map Data
    plot_nodes = ds.nodes.copy()
    plot_nodes["depth_cm"] = d_step_cm.round(1)
    
    # Calculate colors and radii
    colors = []
    radii = []
    for d in d_step_cm:
        _, rgba = depth_band(d)
        colors.append(rgba)
        radii.append(20.0 + min(40.0, d * 0.4))
    plot_nodes["color"] = colors
    plot_nodes["radius"] = radii
    
    elev_col = "elevation_m" if "elevation_m" in plot_nodes.columns else ("elevation" if "elevation" in plot_nodes.columns else None)
    plot_nodes["elev_display"] = plot_nodes[elev_col].round(1) if elev_col else 0.0
    nid_col = "node_id" if "node_id" in plot_nodes.columns else ("id" if "id" in plot_nodes.columns else None)
    plot_nodes["nid_display"] = plot_nodes[nid_col] if nid_col else np.arange(len(plot_nodes))
    plot_nodes["has_pump_display"] = plot_nodes["has_pump"].apply(lambda v: "YES (Pumping)" if v == 1 else "No") if "has_pump" in plot_nodes.columns else "No"
    plot_nodes["is_outlet_display"] = plot_nodes["is_outlet"].apply(lambda v: "YES (Outfall)" if v == 1 else "No") if "is_outlet" in plot_nodes.columns else "No"
    
    layers = []
    
    # Base Node Scatter Layer
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
    
    # Pump Station Rings Layer
    if show_pumps and "has_pump" in plot_nodes.columns:
        pump_nodes = plot_nodes[plot_nodes["has_pump"] == 1]
        if len(pump_nodes) > 0:
            pump_layer = pdk.Layer(
                "ScatterplotLayer",
                pump_nodes,
                get_position=["lon", "lat"],
                get_fill_color=[16, 185, 129, 100],
                get_line_color=[16, 185, 129, 255],
                get_line_width=3,
                stroked=True,
                get_radius=45,
                pickable=False,
            )
            layers.append(pump_layer)
            
    # Outlet Diamond/Ring Layer
    if show_outlets and "is_outlet" in plot_nodes.columns:
        outlet_nodes = plot_nodes[plot_nodes["is_outlet"] == 1]
        if len(outlet_nodes) > 0:
            outlet_layer = pdk.Layer(
                "ScatterplotLayer",
                outlet_nodes,
                get_position=["lon", "lat"],
                get_fill_color=[168, 85, 247, 80],
                get_line_color=[168, 85, 247, 240],
                get_line_width=3,
                stroked=True,
                get_radius=40,
                pickable=False,
            )
            layers.append(outlet_layer)
            
    # Known Spots Markers
    if show_spots and ds.known_spots is not None and len(ds.known_spots) > 0:
        spot_layer = pdk.Layer(
            "ScatterplotLayer",
            ds.known_spots,
            get_position=["lon", "lat"],
            get_fill_color=[255, 255, 255, 220],
            get_line_color=[0, 0, 0, 255],
            get_line_width=2,
            stroked=True,
            get_radius=35,
            pickable=True,
        )
        layers.append(spot_layer)

    view = pdk.ViewState(
        latitude=float(plot_nodes["lat"].mean()),
        longitude=float(plot_nodes["lon"].mean()),
        zoom=13.6,
        pitch=0,
    )
    
    tooltip = {
        "html": """
        <b>Node #{nid_display}</b><br/>
        Depth: <b>{depth_cm} cm</b><br/>
        Elevation: {elev_display} m<br/>
        Pump: {has_pump_display}<br/>
        Outfall: {is_outlet_display}
        """,
        "style": {"backgroundColor": "#0f172a", "color": "white", "fontSize": "12px"}
    }
    
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_style="carto-darkmatter"), use_container_width=True)
    
    # Storm Flood Evolution Timeline Chart
    st.subheader("📈 Network Flood Evolution Over Time")
    if ds.depth is not None:
        flooded_curve = (ds.depth[storm_idx] > 0.10).sum(axis=1)
        rain_curve = ds.rain[storm_idx].mean(axis=-1)
        chart_df = pd.DataFrame({
            "Flooded Nodes (>10cm)": flooded_curve,
            "Average Rain (mm/h)": rain_curve,
        })
        st.line_chart(chart_df, height=200)


# ---------------------------------------------------------------------------
# TAB 2: MULTI-HORIZON AI FORECAST
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("🔮 Spatio-Temporal Graph Neural Network (ST-GNN) Multi-Horizon Forecast")
    
    if not ds.is_model_compatible:
        st.error(f"❌ Forecast model was trained on the sample format; retrain needed for this area.\n\n*{ds.incompatible_reason}*")
    else:
        st.info(f"Model requirement: Requires {history_hours} hours ({history_steps} steps) of input sensor history.")
        
        c_lead1, c_lead2 = st.columns([2, 2])
        with c_lead1:
            horizon_sel = st.radio("Select Prediction Horizon", ["Current Sensor State (Ground Truth)", "+1 Hour Forecast", "+2 Hours Forecast", "+3 Hours Forecast"], index=2, horizontal=True)
        with c_lead2:
            forecast_time_step = st.slider(
                "Current Simulation Step t_now (Min: 8 steps history)",
                min_value=history_steps,
                max_value=ds.T - 13,
                value=min(20, ds.T - 13),
                help="Prediction starts from t_now looking into future horizons."
            )

        # Build recent dict
        recent_dict = build_recent(ds, storm_idx, forecast_time_step, history_steps)
        
        pred_res = None
        source_badge = ""
        latency_str = ""
        
        if recent_dict is not None:
            if api_available and "Auto" in mode:
                try:
                    payload = {
                        "area": ds.name,
                        "recent_rain": recent_dict["rain"].tolist(),
                        "recent_fill": recent_dict["fill"].tolist(),
                        "recent_river": recent_dict["river"].tolist(),
                        "recent_pump_on": recent_dict["pump_on"].tolist(),
                    }
                    r = requests.post(f"{api_url}/predict", json=payload, timeout=5.0)
                    if r.status_code == 200:
                        pred_res = r.json()
                        source_badge = pred_res.get("source", "model")
                        latency_str = f"{pred_res.get('latency_ms', 0)} ms"
                except Exception:
                    pass
                    
            if pred_res is None:
                # Direct import fallback
                try:
                    from src.ml.predict import predict as ml_predict
                    t0 = time.perf_counter()
                    pred_res = ml_predict(ds.nodes, ds.edges, recent_dict)
                    latency_str = f"{round((time.perf_counter() - t0)*1000, 1)} ms"
                    source_badge = "model (direct)"
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if pred_res is not None:
            h_key = {"+1 Hour Forecast": "1h", "+2 Hours Forecast": "2h", "+3 Hours Forecast": "3h"}.get(horizon_sel, None)
            
            if h_key is None:
                # Truth at t_now
                display_depths_m = ds.depth[storm_idx, forecast_time_step]
                st.caption(f"Displaying Ground Truth at step {forecast_time_step} ({step_to_clock(forecast_time_step, ds.dt_seconds)})")
            else:
                # Model prediction for h_key
                display_depths_m = np.array([pred_res["nodes"][str(i)][h_key] for i in range(ds.N)])
                h_step_offset = horizon_steps(int(h_key[0]), ds.dt_seconds)
                truth_step = forecast_time_step + h_step_offset
                truth_depths_m = ds.depth[storm_idx, truth_step]
                mae_cm = round(float(np.abs(display_depths_m - truth_depths_m).mean()) * 100.0, 2)
                
                # Metric comparison
                fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
                fc_col1.metric("Predicted Flooded Nodes", int((display_depths_m > 0.10).sum()))
                fc_col2.metric("Actual Flooded Nodes (Truth)", int((truth_depths_m > 0.10).sum()))
                fc_col3.metric(f"MAE Error at +{h_key}", f"{mae_cm} cm", delta="Real validation", delta_color="normal")
                fc_col4.metric("Inference Engine", source_badge.upper(), delta=latency_str, delta_color="off")
                
            # Render Forecast Map
            f_nodes = ds.nodes.copy()
            f_nodes["depth_cm"] = (display_depths_m * 100.0).round(1)
            f_colors = [depth_band(d)[1] for d in f_nodes["depth_cm"]]
            f_radii = [20.0 + min(40.0, d * 0.4) for d in f_nodes["depth_cm"]]
            f_nodes["color"] = f_colors
            f_nodes["radius"] = f_radii
            f_nodes["nid_display"] = f_nodes["node_id"] if "node_id" in f_nodes.columns else f_nodes.index
            
            f_layer = pdk.Layer(
                "ScatterplotLayer",
                f_nodes,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
            )
            f_view = pdk.ViewState(latitude=float(f_nodes["lat"].mean()), longitude=float(f_nodes["lon"].mean()), zoom=13.6)
            f_tooltip = {"html": "<b>Node #{nid_display}</b><br/>Predicted Depth: <b>{depth_cm} cm</b>"}
            st.pydeck_chart(pdk.Deck(layers=[f_layer], initial_view_state=f_view, tooltip=f_tooltip, map_style="carto-darkmatter"), use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: RISK & ALERTS
# ---------------------------------------------------------------------------
with tab_alerts:
    st.subheader("🚨 Inundation Risk Ranking & Priority Hotspots")
    
    d_current_m = ds.depth[storm_idx, current_step] if ds.depth is not None else np.zeros(ds.N)
    rtable = risk_table(d_current_m, ds.nodes, top_k=15, pumps=ds.pumps)
    
    # Text Alerts for critical nodes
    severe_nodes = rtable[rtable["depth_cm"] >= 30.0]
    if len(severe_nodes) > 0:
        for _, row in severe_nodes.iterrows():
            st.error(f"⚠️ **Node {int(row['node_id'])}** inundated: **{row['depth_cm']} cm** ({row['band']}). Elevation: {row['elevation_m']} m.")
    else:
        st.success("✅ No severe street inundation (>30 cm) at the selected simulation timestamp.")
        
    st.dataframe(rtable, use_container_width=True)
    
    csv_data = rtable.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Risk Table CSV", csv_data, f"risk_table_{ds.name}_{scen_key}_step{current_step}.csv", "text/csv")


# ---------------------------------------------------------------------------
# TAB 4: PUMPS & GATES OPTIMIZER
# ---------------------------------------------------------------------------
with tab_optimizer:
    st.subheader("⚡ Automated Pump & Sluice Gate Dispatch Optimizer")
    st.caption("Stochastic local search & heuristic gate scheduling minimizing total flood exposure (m·h).")
    
    if not ds.is_model_compatible:
        st.warning("Optimizer requires standard network structure with pump rates and gate edges (available in Sample area).")
    else:
        opt_col1, opt_col2 = st.columns([2, 1])
        with opt_col1:
            st.write(f"Current Target: **{selected_scenario}** ({ds.name})")
        with opt_col2:
            run_opt_btn = st.button("🚀 Run Smart Dispatch Optimizer", type="primary", use_container_width=True)
            
        opt_data = None
        if run_opt_btn or f"opt_{selected_scenario}" in st.session_state:
            st.session_state[f"opt_{selected_scenario}"] = True
            
            with st.spinner("Executing stochastic local search optimizer..."):
                opt_data = get_cached_optimizer_result(ds.name, scen_key, "search", 0)
                
            if opt_data:
                o1, o2, o3 = st.columns(3)
                o1.metric("Baseline Flood Cost", f"{opt_data['scenario_cost_before']} m·h")
                o2.metric("Optimized Flood Cost", f"{opt_data['scenario_cost_after']} m·h", delta=f"-{opt_data['reduction_pct']}% reduction")
                o3.metric("Averted Inundation", f"{opt_data['reduction_pct']}%", delta="Optimization Pass", delta_color="normal")
                
                # Before / After Verification via simulate()
                st.markdown("#### 🔬 Simulator Verification")
                try:
                    # Run baseline
                    T, N = ds.rain[storm_idx].shape
                    E = len(ds.edges)
                    no_pump = np.zeros((T, N), dtype=np.float32)
                    all_open = np.ones((T, E), dtype=np.float32)
                    
                    sim_before = simulate(ds.nodes, ds.edges, ds.rain[storm_idx], ds.river[storm_idx], no_pump, all_open)
                    
                    # Run with opt schedules if files exist
                    p_path = REPO_ROOT / "results" / "opt_pump_on.npy"
                    g_path = REPO_ROOT / "results" / "opt_gate_open.npy"
                    if p_path.exists() and g_path.exists():
                        p_arr = np.load(p_path)
                        g_arr = np.load(g_path)
                        sim_after = simulate(ds.nodes, ds.edges, ds.rain[storm_idx], ds.river[storm_idx], p_arr, g_arr)
                        
                        curve_before = (sim_before["depth"] > 0.10).sum(axis=1)
                        curve_after = (sim_after["depth"] > 0.10).sum(axis=1)
                        
                        comp_df = pd.DataFrame({
                            "Baseline (No Control)": curve_before,
                            "Optimized (Smart Dispatch)": curve_after,
                        })
                        st.line_chart(comp_df, height=220)
                except Exception as e:
                    st.caption(f"Simulator comparison note: {e}")

                # Dispatch Schedules Table
                st.markdown("#### 📋 Actionable Actuation Timeline")
                t_p, t_g = st.columns(2)
                with t_p:
                    st.markdown("**⚡ Pump Dispatch:**")
                    if opt_data.get("pump_schedule"):
                        p_df = pd.DataFrame(opt_data["pump_schedule"])
                        p_df["time"] = p_df["step"].apply(lambda s: step_to_clock(s, ds.dt_seconds))
                        st.dataframe(p_df[["time", "node_id", "on"]], hide_index=True)
                    else:
                        st.info("Pumps run proactively across storm duration.")
                with t_g:
                    st.markdown("**🚪 Sluice Gate Throttling:**")
                    if opt_data.get("gate_schedule"):
                        g_df = pd.DataFrame(opt_data["gate_schedule"])
                        g_df["time"] = g_df["step"].apply(lambda s: step_to_clock(s, ds.dt_seconds))
                        st.dataframe(g_df[["time", "edge_id", "open"]], hide_index=True)
                    else:
                        st.info("All gates maintained open.")


# ---------------------------------------------------------------------------
# TAB 5: IoT SENSOR GAUGE
# ---------------------------------------------------------------------------
with tab_iot:
    st.subheader("📡 Simulated IoT Sensor Gauge & Tidal Interaction")
    
    iot_step = current_step
    iot_rain = float(ds.rain[storm_idx, iot_step].mean())
    iot_river = float(ds.river[storm_idx, iot_step])
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.markdown(f"**Current Gauge Elevation:** `{round(iot_river, 2)} m` (Max Discharge Limit: `{RIVER_MAX} m`)")
        gauge_df = pd.DataFrame({
            "Step": np.arange(ds.T),
            "River Level (m)": ds.river[storm_idx],
            "Critical Limit (m)": np.full(ds.T, RIVER_MAX),
        })
        st.line_chart(gauge_df.set_index("Step"), height=240)
    with col_g2:
        st.markdown("**IoT Stream Status:**")
        st.json({
            "sensor_id": "river_gauge_outfall_01",
            "timestamp": step_to_clock(iot_step, ds.dt_seconds),
            "river_level_m": round(iot_river, 2),
            "rain_intensity_mmh": round(iot_rain, 1),
            "status": "BACKWATER WARNING" if iot_river > 3.5 else "NORMAL FLOW",
        })


# ---------------------------------------------------------------------------
# TAB 6: MODEL METRICS
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("📊 Empirical Model Evaluation & Baseline Comparison")
    st.caption("Real metrics measured by Person B on test storms (Source: results/final_metrics.json).")
    
    metrics_path = REPO_ROOT / "results" / "final_metrics.json"
    if metrics_path.exists():
        try:
            import json
            with open(metrics_path) as f:
                metrics_data = json.load(f)
            m_df = pd.DataFrame(metrics_data)
            st.dataframe(m_df[["model", "lead_time", "mae_cm", "rmse_cm", "f1", "precision", "recall"]], hide_index=True, use_container_width=True)
            
            # Chart F1 comparison
            st.markdown("#### F1 Score Comparison Across Horizons")
            f1_chart = m_df.pivot(index="lead_time", columns="model", values="f1")
            st.bar_chart(f1_chart)
        except Exception as e:
            st.error(f"Error reading metrics: {e}")
    else:
        st.info("No metrics file found yet in results/final_metrics.json.")


# ---------------------------------------------------------------------------
# TAB 7: ABOUT & HONESTY
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ Project Architecture & Data Honesty Statement")
    
    st.markdown("""
    ### 🎯 Hackathon Problem Statement 11
    **AI-Powered Urban Drainage and Flash Flood Predictive Management System**  
    Target Study Catchment: **Mumbai: Dadar-Hindmata-Parel** & scalability check on **Kurla-Mithi River**.

    ### ⚖️ Ground Truth & Simulation Honesty
    - **Real Inputs:** Street network topology extracted from OpenStreetMap, Digital Elevation Model (DEM) topography, and historical monsoon rainfall statistics.
    - **Simulated Components:** Subsurface drainage conduit capacities, sluice gate actuators, IoT river gauges, and flooded street ground-truth labels are simulated using hydraulic approximations because municipal sensor data is not publicly accessible.
    - **Model Architecture:** Spatio-Temporal Graph Neural Network (ST-GNN) passing features across drainage pipe connections, evaluated against Persistence and No-Graph baselines.
    """)
    
st.markdown("---")
st.caption("FloodSense System | Person C Integration Build | All data simulated for evaluation purposes.")
