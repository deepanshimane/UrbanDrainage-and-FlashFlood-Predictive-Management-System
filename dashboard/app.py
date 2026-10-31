from pathlib import Path
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path("data/sample")
st.set_page_config(layout="wide")
st.title("FloodSense: Dadar-Hindmata-Parel flood map (SIMULATED DATA)")


@st.cache_data
def load():
    nodes = pd.read_csv(DATA / "nodes.csv")
    depth = np.load(DATA / "depth.npy")   # [storm, time, node], metres
    rain = np.load(DATA / "rain.npy")     # [storm, time, node]
    return nodes, depth, rain


nodes, depth, rain = load()
S, T, N = depth.shape

# Scenarios = storms ranked by peak rain
order = np.argsort(rain.max(axis=(1, 2)))
scenarios = {
    "Light rain": order[int(S * 0.1)],
    "Heavy storm": order[int(S * 0.6)],
    "Extreme cloudburst": order[-1],
}
name = st.sidebar.selectbox("Scenario", list(scenarios))
s = scenarios[name]
t = st.sidebar.slider("Time step", 0, T - 1, T // 3)

d_cm = depth[s, t] * 100
frac = np.clip(d_cm / 50, 0, 1)   # 0 cm = blue, 50+ cm = red
nodes["depth_cm"] = d_cm.round(1)
nodes["color"] = np.c_[255 * frac, np.full(N, 80), 255 * (1 - frac), np.full(N, 180)].astype(int).tolist()

layer = pdk.Layer(
    "ScatterplotLayer", nodes,
    get_position=["lon", "lat"], get_fill_color="color",
    get_radius=18, pickable=True,
)
view = pdk.ViewState(latitude=nodes.lat.mean(), longitude=nodes.lon.mean(), zoom=14)
st.pydeck_chart(pdk.Deck(
    layers=[layer], initial_view_state=view,
    tooltip={"text": "Node {node_id}\nDepth: {depth_cm} cm\nElevation: {elevation_m} m"},
))

c1, c2, c3 = st.columns(3)
c1.metric("Nodes flooded (>10 cm)", int((d_cm > 10).sum()))
c2.metric("Max depth (cm)", round(float(d_cm.max()), 1))
c3.metric("Avg rain now (mm)", round(float(rain[s, t].mean()), 1))

st.caption("Flooded nodes over time")
st.line_chart((depth[s] > 0.10).sum(axis=1))
