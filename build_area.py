"""Build nodes.csv / edges.csv / pumps.csv for one area.
Usage: python build_area.py "Hindmata, Mumbai, India" data/hindmata dem/mumbai_dem.tif
Needs: pip install osmnx networkx geopandas rasterio scipy pyproj pandas numpy matplotlib
"""
import sys, os, time
import numpy as np, pandas as pd, networkx as nx, osmnx as ox, rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer

PLACE = sys.argv[1] if len(sys.argv) > 1 else "Hindmata, Mumbai, India"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/hindmata"
DEM = sys.argv[3] if len(sys.argv) > 3 else "dem/mumbai_dem.tif"

DIST = 1000          # radius in metres (about 3 sq km). Keep nodes <= 600.
TOL = 35             # merge intersections closer than this (m). Raise TOL for fewer nodes.
os.makedirs(OUT, exist_ok=True)

# 1. street graph -> simplified graph
if "," in PLACE and all(part.strip().replace(".", "", 1).replace("-", "", 1).isdigit() for part in PLACE.split(",")):
    center = tuple(float(x.strip()) for x in PLACE.split(","))
    print(f"Using coordinates for center: {center}", flush=True)
else:
    center = ox.geocode(PLACE)
    if "hindmata" in PLACE.lower() and (center[0] < 19.0 or center[0] > 19.03):
        print(f"Notice: geocode for '{PLACE}' returned {center}, using verified Hindmata junction (19.009138, 72.841930)", flush=True)
        center = (19.009138, 72.841930)
    else:
        print(f"Geocoded '{PLACE}' to center: {center}", flush=True)

print(f"Fetching street graph within {DIST}m radius...", flush=True)
ox.settings.requests_timeout = 60
ox.settings.request_headers = {"User-Agent": "FloodSenseMumbai/1.0 (urban-drainage-research)"}

endpoints = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.private.coffee/api"
]

G = None
for ep in endpoints:
    ox.settings.overpass_url = ep
    print(f"Connecting to Overpass endpoint {ep}...", flush=True)
    try:
        G = ox.graph_from_point(center, dist=DIST, network_type="drive")
        print(f"Successfully downloaded graph from {ep} ({len(G)} raw nodes)", flush=True)
        break
    except Exception as e:
        print(f"Endpoint {ep} error: {e}. Retrying with next endpoint...", flush=True)
        time.sleep(2)

if G is None:
    raise RuntimeError("Failed to fetch graph from all Overpass endpoints.")

print(f"Original graph has {len(G)} nodes and {G.number_of_edges()} edges. Consolidating intersections (TOL={TOL}m)...", flush=True)
G = ox.consolidate_intersections(ox.project_graph(G), tolerance=TOL, rebuild_graph=True, dead_ends=False)
G = ox.project_graph(G, to_crs="EPSG:4326")
U = nx.Graph(G.to_undirected())
U = U.subgraph(max(nx.connected_components(U), key=len)).copy()
U = nx.convert_node_labels_to_integers(U)               # ids 0..N-1
N = U.number_of_nodes()
lon = np.array([U.nodes[i]["x"] for i in range(N)], float)
lat = np.array([U.nodes[i]["y"] for i in range(N)], float)
print(f"Consolidated largest connected component: {N} nodes, {U.number_of_edges()} edges.", flush=True)

# 2. elevation from the DEM (+ one smoothing pass because 30 m DEMs are noisy in cities)
print(f"Sampling elevation from {DEM}...", flush=True)
with rasterio.open(DEM) as src:
    z = np.array([v[0] for v in src.sample(list(zip(lon, lat)))], float)
z[(z < -50) | (z > 500)] = np.nan
z = np.where(np.isnan(z), np.nanmedian(z), z)
nbr_mean = np.array([np.mean([z[j] for j in U.neighbors(i)]) if U.degree(i) else z[i] for i in range(N)])
z_s = 0.5 * z + 0.5 * nbr_mean
print(f"elevation min/max/range (m): {z_s.min():.2f} / {z_s.max():.2f} / {(z_s.max() - z_s.min()):.2f}", flush=True)

# 3. area each node drains (nearest-node on a 25 m grid inside the circle)
print("Computing catchment area per node...", flush=True)
tr = Transformer.from_crs(4326, 32643, always_xy=True)   # UTM 43N covers Mumbai
X, Y = tr.transform(lon, lat)
cx, cy = tr.transform(center[1], center[0])
g = np.arange(-DIST, DIST, 25.0) + 12.5
gx, gy = np.meshgrid(cx + g, cy + g)
inside = (gx - cx) ** 2 + (gy - cy) ** 2 <= DIST ** 2
_, near = cKDTree(np.c_[X, Y]).query(np.c_[gx[inside], gy[inside]])
area = np.bincount(near, minlength=N) * 625.0
area = np.maximum(area, 625.0)

# 4. pumps: create <OUT>/pumps_input.csv with columns name,lat,lon,cap_m3s; each node is served by the nearest pump
pumps_input_file = os.path.join(OUT, "pumps_input.csv")
if not os.path.exists(pumps_input_file):
    raise FileNotFoundError(f"Missing {pumps_input_file}. Create it first.")

pin = pd.read_csv(pumps_input_file)
px, py = tr.transform(pin.lon.values, pin.lat.values)
_, pnode = cKDTree(np.c_[X, Y]).query(np.c_[px, py])
_, zone = cKDTree(np.c_[X[pnode], Y[pnode]]).query(np.c_[X, Y])
pumps = pd.DataFrame(dict(pump_id=range(len(pin)), node_id=pnode, name=pin.name, cap_m3s=pin.cap_m3s))
has_pump = np.isin(np.arange(N), pnode).astype(int)

# 5. save
nodes_df = pd.DataFrame(dict(
    id=range(N), lat=lat, lon=lon, elevation=z_s.round(3), area_m2=area,
    runoff_coeff=0.85, pump_zone=zone, has_pump=has_pump
))
nodes_df.to_csv(f"{OUT}/nodes.csv", index=False)

CAP = {"motorway": 3.0, "trunk": 3.0, "primary": 2.0, "secondary": 1.5, "tertiary": 1.0}
rows = []
for u, v, d in U.edges(data=True):
    hw = d.get("highway", "residential")
    hw = hw[0] if isinstance(hw, list) else hw
    length = d.get("length", 50)
    length = length[0] if isinstance(length, list) else length
    rows.append(dict(source=u, target=v, capacity_m3s=CAP.get(hw, 0.5), length_m=float(length)))

edges_df = pd.DataFrame(rows)
edges_df.to_csv(f"{OUT}/edges.csv", index=False)
pumps.to_csv(f"{OUT}/pumps.csv", index=False)

print(f"Successfully generated files in {OUT}:", flush=True)
print(f" - {OUT}/nodes.csv ({len(nodes_df)} nodes)", flush=True)
print(f" - {OUT}/edges.csv ({len(edges_df)} edges)", flush=True)
print(f" - {OUT}/pumps.csv ({len(pumps)} pumps linked to nodes {pnode.tolist()})", flush=True)
